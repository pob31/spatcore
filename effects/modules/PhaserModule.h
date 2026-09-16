#pragma once

#include "../EffectModule.h"
#include "../../dsp/FastDecibels.h"
#include "../../dsp/FrDiffusionModel.h"
#include "../../dsp/LFOWaveforms.h"
#include "../../dsp/LfoPhasor.h"
#include "../../dsp/OnePoleSmoother.h"
#include <cmath>
#include <cstdint>

namespace spatcore::effects
{

/**
    A swept notch filter: N first-order allpasses in series, summed with the dry
    signal, with the stage frequencies swept by an LFO.

    Designed from the effects plan rather than traced from a prototype. The user
    judged their own Max phaser unsatisfactory, so nothing here comes from it.

    Where the notches come from is worth stating, because it is what every
    decision below protects. An allpass passes every frequency at unity gain and
    only turns its phase: one stage turns the signal by 90 degrees at its own
    frequency, and by more below it and less above. Summed with the dry signal,
    the frequencies the chain has turned by 180 degrees cancel, and sweeping the
    stage frequencies walks those cancellations through the spectrum. So the wet
    leg on its own has no notches at all: mix 100 is a phase rotation and
    nothing else, and mix 50 is where the notches are deepest. That is why 50 is
    the default, and why a "more effect equals more wet" reading of the mix
    control is wrong here.

    Non-obvious decisions, and the failure each one avoids:

    - The coefficients are recomputed every kCoefficientUpdateInterval samples
      and interpolated linearly in between. One tan() per stage per sample is
      the entire cost of this module, and twelve of them per sample would make
      the phaser dearer than the reverb. Updating without the interpolation
      trades that for a different fault: an allpass coefficient that steps
      inside a feedback loop ticks on every update, three thousand times a
      second.

    - The LFO and the three geometry smoothers (centre, spread, depth) are only
      read at those updates, so they run on a sample-rate reference of
      sr / kCoefficientUpdateInterval. One step per update then covers the same
      10 ms glide every other module uses; advancing them per sample and reading
      one value in sixteen would instead make every geometry move sixteen times
      slower than the parameter tau promises.

    - Stage frequencies are clamped to 20 Hz .. 0.45 sr. That clamp is not
      cosmetic: it is what keeps t = tan(pi f / sr) finite and strictly
      positive, so a = (t - 1) / (t + 1) stays inside (-1, 1) and every stage
      stays a stable allpass. Without it a centre frequency the sweep drives
      past Nyquist flips the sign of t, puts the pole outside the unit circle
      and turns the stage into a growing exponential.

    - Feedback is the self-oscillation case, and an allpass chain sharpens it:
      the chain's magnitude is exactly one, so the loop gain IS the feedback
      gain. At the plan's 95 % ceiling the steady-state resonance reaches
      1 / (1 - 0.95) = 20 times, which is loud but bounded and stable. What is
      not guaranteed is passivity while the coefficients MOVE: a modulated
      allpass can hand energy back to the loop, and with only 5 % of margin a
      fast deep sweep could pump it. The recirculating sample is therefore hard
      limited at kLoopCeiling, which sits above the largest gain the algorithm
      can legitimately reach, so it never shapes a musical signal and exists
      only to stop a pumped loop from running away.

    - That ceiling is why the module has to test its own loop for finiteness
      once per sample. A recursive module cannot lean on the slot's NaN guard
      the way a memoryless one can: a NaN from upstream lands in the allpass
      history, and a ceiling that clamps the OUTPUT would hand the slot a
      perfectly finite sample while the history stayed poisoned - a permanent
      full-scale DC latch that the one guard designed to catch it could no
      longer see. The test is what keeps the state and the output honest about
      each other.

    - Mix 0 stops the module rather than multiplying by zero, so it is bit
      transparent down to the negative zeros and denormals the tests feed it.
      The LFO freezes with it, exactly as it does when the slot bypasses the
      module, so a phaser turned down costs nothing per sample.

    - stages is a VARIANT parameter: changing it changes how many filters are in
      the chain, which cannot be interpolated, so the module keeps running the
      old count until the slot has faded to silence and committed the new one.

    The sweep starts at the BOTTOM of its range, because the shared Sine shape
    is -cos and therefore -1 at phase 0.

    Zero latency. No allocation after prepare().
*/
class PhaserModule : public IEffectModule
{
public:
    /** The plan's largest validated stage count. */
    static constexpr int kMaxStages = 12;

    /** Samples between coefficient recomputations. Sixteen is 3 kHz at 48 kHz,
        three hundred updates per cycle of the fastest LFO the plan allows, so
        the sweep is sampled far above the rate at which it could step audibly,
        and the interpolation covers what is left. */
    static constexpr int kCoefficientUpdateInterval = 16;

    /** Hard bound on the recirculating sample, about +30 dBFS. Above the 20x
        peak the maximum feedback can legitimately produce from a full-scale
        input, so no musical signal reaches it - a signal already far past full
        scale does (the house awkward block, whose samples are +-1e7, pins it at
        twelve stages), and that is the point: it bounds the loop rather than
        shaping the sound. */
    static constexpr float kLoopCeiling = 32.0f;

    ModuleId type() const noexcept override { return ModuleId::Phaser; }

    void prepare (const ChainConfig& config) override
    {
        sampleRate = config.sampleRate > 0.0 ? config.sampleRate : 48000.0;
        piOverSampleRate = static_cast<float> (3.14159265358979323846 / sampleRate);
        nyquistLimit = static_cast<float> (sampleRate) * 0.45f;

        // The control rate: what the LFO and the geometry smoothers see.
        const double controlRate = sampleRate / static_cast<double> (kCoefficientUpdateInterval);

        lfo.setNoiseKey (spatcore::dsp::FrDiffusion::makeKey (static_cast<int> (config.noiseKey),
                                                             static_cast<int> (ModuleId::Phaser)));
        lfo.prepare (controlRate);

        centre.setTimeConstant (controlRate, kParamTauSeconds);
        centre.setSnapEpsilon (0.5f);               // hertz, not a unit gain
        spread.setTimeConstant (controlRate, kParamTauSeconds);
        depth.setTimeConstant (controlRate, kParamTauSeconds);

        feedbackGain.setTimeConstant (sampleRate, kParamTauSeconds);
        wetMix.setTimeConstant (sampleRate, kParamTauSeconds);

        for (int k = 0; k < kMaxStages; ++k)
        {
            coefA[k] = 0.0f;
            coefStep[k] = 0.0f;
        }

        snapOnNextApply = true;
        reset();
    }

    void reset() noexcept override
    {
        clearAudioState();
        lfo.reset();

        centre.snap (centre.getTarget());
        spread.snap (spread.getTarget());
        depth.snap (depth.getTarget());
        feedbackGain.snap (feedbackGain.getTarget());
        wetMix.snap (wetMix.getTarget());

        // Recompute on the first sample of the next block and take the new
        // coefficients outright: there is no state left for them to jump.
        coefCountdown = 0;
        coefSnap = true;
        idle = false;
    }

    ParamApplyInfo applyParams (const EffectChannelParams& params, int) noexcept override
    {
        const PhaserParams& ph = params.phaser;

        lfo.setRateHz (clamp (ph.rateHz, 0.02f, 10.0f));    // phase stays continuous
        shapeId = clampInt (static_cast<int> (ph.shape),
                            spatcore::dsp::LFOWaveforms::Sine,
                            spatcore::dsp::LFOWaveforms::Random);

        centre.setTarget (clamp (ph.centreHz, 100.0f, 5000.0f));
        spread.setTarget (clamp (ph.spreadOct, 0.0f, 3.0f));
        depth.setTarget (clamp (ph.depthOct, 0.0f, 4.0f));
        feedbackGain.setTarget (clamp (ph.feedback, -95.0f, 95.0f) * 0.01f);
        wetMix.setTarget (clamp (ph.mix, 0.0f, 100.0f) * 0.01f);

        pendingStages = validateStages (ph.stages);

        if (snapOnNextApply)
        {
            centre.snap (centre.getTarget());
            spread.snap (spread.getTarget());
            depth.snap (depth.getTarget());
            feedbackGain.snap (feedbackGain.getTarget());
            wetMix.snap (wetMix.getTarget());

            runningStages = pendingStages;
            coefCountdown = 0;
            coefSnap = true;
            snapOnNextApply = false;
        }

        return { ph.bypass != 0, pendingStages != runningStages };
    }

    void commitPendingVariant() noexcept override
    {
        runningStages = pendingStages;

        // The stages that were not in the chain still carry whatever they held
        // when they last were, which would otherwise arrive as a click the
        // moment the count grows.
        clearAudioState();
        coefCountdown = 0;
        coefSnap = true;
    }

    void process (float* inout, int numSamples) noexcept override
    {
        // Mix 0 is the identity and must be BIT transparent: no multiply by
        // one, no crossfade of the signal against itself. Stopping outright
        // also means a phaser turned down costs nothing per sample.
        if (wetMix.isSettled() && wetMix.getCurrent() == 0.0f)
        {
            if (! idle)
            {
                clearAudioState();
                coefCountdown = 0;
                coefSnap = true;                    // fresh coefficients on the way back
                idle = true;
            }

            return;
        }

        idle = false;
        const int stages = runningStages;

        for (int i = 0; i < numSamples; ++i)
        {
            if (coefCountdown <= 0)
                updateCoefficients (stages);

            --coefCountdown;

            const float fb = feedbackGain.next();
            const float w = wetMix.next();
            const float in = inout[i];

            float v = in + fb * feedbackState;

            for (int k = 0; k < stages; ++k)
            {
                coefA[k] += coefStep[k];

                // y = a*x + x1 - a*y1, one multiply and two adds per stage.
                const float y = coefA[k] * (v - apY[k]) + apX[k];
                apX[k] = v;
                apY[k] = y;
                v = y;
            }

            // The safety net, and it has to be two tests rather than one.
            //
            // A non-finite sample arriving from upstream has already reached
            // apX/apY by the time we get here - they are loop STATE, and a NaN
            // in them poisons every sample that follows for ever. Clamping the
            // value alone would make that permanent AND invisible: the module
            // would put out a finite -kLoopCeiling from then on, and the slot's
            // own guard (EffectModule.h, looking for a non-finite sample at the
            // end of the block) would never fire to reset us. So the poisoned
            // state is thrown away and the sample is dropped; the module is
            // clean again on the very next one. Dropped means written as zero
            // whatever the mix says, including on the dry-only path below: the
            // buffer holds the non-finite sample itself there, and passing it
            // on would be the one outcome worse than losing it. The write to
            // feedbackState is skipped because clearAudioState() has just set
            // it, which is the value that belongs there.
            if (! std::isfinite (v))
            {
                clearAudioState();
                inout[i] = 0.0f;
                continue;
            }

            // The ceiling proper: the bound on a loop that a fast deep sweep
            // has pumped. Still written in the negated house form, though the
            // test above has already taken every NaN out of its way.
            if (v > kLoopCeiling)
                v = kLoopCeiling;
            else if (! (v > -kLoopCeiling))
                v = -kLoopCeiling;

            feedbackState = v;

            if (w >= 1.0f)
                inout[i] = v;
            else if (w > 0.0f)
                inout[i] = in + w * (v - in);
            // w <= 0: dry, already in the buffer
        }

        flushTinyState (stages);
    }

    int getLatencySamples() const noexcept override { return 0; }

private:
    static float clamp (float v, float lo, float hi) noexcept
    {
        // Negated comparisons so a NaN parameter lands on the low bound rather
        // than propagating into the audio.
        if (! (v > lo)) return lo;
        if (v > hi)     return hi;
        return v;
    }

    static int clampInt (int v, int lo, int hi) noexcept
    {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    }

    /** The plan validates 4, 6, 8 and 12. Anything between two of them takes
        the LOWER one, so a control that ever sends a raw number degrades to
        fewer filters rather than to more than was asked for. */
    static int validateStages (std::uint8_t requested) noexcept
    {
        const int v = static_cast<int> (requested);
        if (v >= 12) return 12;
        if (v >= 8)  return 8;
        if (v >= 6)  return 6;
        return 4;
    }

    void clearAudioState() noexcept
    {
        for (int k = 0; k < kMaxStages; ++k)
        {
            apX[k] = 0.0f;
            apY[k] = 0.0f;
        }

        feedbackState = 0.0f;
    }

    /** New target coefficients, and the per-sample step that reaches them by
        the next update. */
    void updateCoefficients (int stages) noexcept
    {
        const float lfoValue = lfo.nextValue (shapeId);         // -1 .. +1
        const float centreNow = centre.next();
        const float depthNow = depth.next();
        const float spreadNow = spread.next();

        // exp2 rather than std::pow: libm-free, reproducible, and exact for a
        // whole number of octaves, so depth 0 leaves the centre frequency
        // untouched to the bit rather than nearly so.
        const float sweep = spatcore::dsp::FastDecibels::exp2 (depthNow * lfoValue);

        // 2k/(N-1) - 1 spans -1 .. +1 across the chain. N = 1 would divide by
        // zero; the validated counts start at 4, but a single stage must still
        // land on the centre frequency rather than on a NaN.
        const float denom = stages > 1 ? static_cast<float> (stages - 1) : 1.0f;

        for (int k = 0; k < stages; ++k)
        {
            const float offset = 2.0f * static_cast<float> (k) / denom - 1.0f;
            float f = centreNow * sweep * spatcore::dsp::FastDecibels::exp2 (spreadNow * offset);

            if (! (f > 20.0f))     f = 20.0f;       // negated: a NaN lands here
            if (f > nyquistLimit)  f = nyquistLimit;

            const float t = std::tan (piOverSampleRate * f);
            const float a = (t - 1.0f) / (t + 1.0f);

            if (coefSnap)
            {
                coefA[k] = a;
                coefStep[k] = 0.0f;
                continue;
            }

            const float difference = a - coefA[k];

            // A step this small is worth nothing over sixteen samples and can
            // itself be denormal, which costs more than the move it makes.
            if (std::fabs (difference) < 1.0e-9f)
            {
                coefA[k] = a;
                coefStep[k] = 0.0f;
            }
            else
            {
                coefStep[k] = difference * (1.0f / static_cast<float> (kCoefficientUpdateInterval));
            }
        }

        coefSnap = false;
        coefCountdown = kCoefficientUpdateInterval;
    }

    /** A recursion decaying towards zero in float would otherwise pass through
        the denormal range on its way, where a silent channel costs more per
        sample than a loud one. Twenty-five comparisons once per block put the
        floor at -400 dB instead, eighteen decades above the first denormal,
        which no listener and no meter can tell from silence. It does not make
        the tail short: at 95 % feedback the decay to that floor still takes
        seconds. It makes the tail cheap, and it ends at a true zero.

        Negated, like every other guard here: `fabs (x) < tiny` is FALSE for a
        NaN, so the plain form would leave poisoned state in place - which is
        exactly the failure the per-sample test in process() is there to stop,
        and this is its second line. */
    void flushTinyState (int stages) noexcept
    {
        constexpr float tiny = 1.0e-20f;

        for (int k = 0; k < stages; ++k)
        {
            if (! (std::fabs (apX[k]) > tiny)) apX[k] = 0.0f;
            if (! (std::fabs (apY[k]) > tiny)) apY[k] = 0.0f;
        }

        if (! (std::fabs (feedbackState) > tiny))
            feedbackState = 0.0f;
    }

    double sampleRate = 48000.0;
    float piOverSampleRate = 0.0f;
    float nyquistLimit = 21600.0f;

    spatcore::dsp::LfoPhasor lfo;
    spatcore::dsp::OnePoleSmoother centre, spread, depth, feedbackGain, wetMix;

    float apX[kMaxStages] = {};                 // stage input at n-1
    float apY[kMaxStages] = {};                 // stage output at n-1
    float coefA[kMaxStages] = {};
    float coefStep[kMaxStages] = {};
    float feedbackState = 0.0f;

    int coefCountdown = 0;
    int shapeId = spatcore::dsp::LFOWaveforms::Sine;
    int runningStages = 6, pendingStages = 6;
    bool coefSnap = true;
    bool snapOnNextApply = true;
    bool idle = false;
};

} // namespace spatcore::effects
