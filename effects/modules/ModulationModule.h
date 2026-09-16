#pragma once

#include "../EffectModule.h"
#include "../../dsp/DcBlocker.h"
#include "../../dsp/FractionalDelayLine.h"
#include "../../dsp/FrDiffusionModel.h"
#include "../../dsp/LFOWaveforms.h"
#include "../../dsp/LfoPhasor.h"
#include "../../dsp/OnePoleSmoother.h"
#include "../../dsp/OutputEQBiquadFilter.h"
#include <atomic>
#include <cmath>
#include <cstdint>

namespace spatcore::effects
{

/**
    Chorus and flanger: one modulated, feedback-fed delay line read by up to
    three voices, from the user's fx_flanger gen~ prototype.

    The modulation MULTIPLIES the centre delay: t = centre * (1 + depth * lfo),
    with depth a fraction of the centre. The prototype builds it as an addition
    of a term proportional to the base time, which is the same arithmetic, and
    it matters: an absolute deviation in milliseconds would make the sweep depth
    of a 2 ms flange and a 20 ms chorus mean completely different things, and
    the presets are written against the multiplicative law.

    The LFO runs per SAMPLE, not per block. In the prototype every box in the
    phasor -> sin -> multiply chain is signal rate, so the delay time is
    recomputed and the line re-read every sample; evaluating it once per block
    stair-steps the sweep into a zipper at exactly the rates a chorus uses.

    The low cut sits only in the FORWARD path, as in the prototype: the dry leg
    is the raw input and the feedback re-enters downstream of the filter. Moving
    the filter inside the loop would darken every repeat instead of once, which
    is a different effect from the one the prototype makes.

    Feedback is where the prototype is genuinely unsafe - a bare multiply with
    no clamp, no DC blocker and no damping, so unity gain sustains for ever and
    anything above it blows up, while the loop accumulates DC because the low
    cut is outside it. Here the gain is clamped to +-0.95 and the loop runs
    through a DcBlocker and a gentle high cut, so a long flange thickens rather
    than turning into a rumbling wall. The feedback is taken from the FIRST
    voice only: feeding the summed voices back would make the loop gain scale
    with the voice count, so a setting that is stable at one voice would howl at
    three.

    The wet sum is scaled by 1/sqrt(voices), NOT by the voice count. The
    prototype's two-line chorus sums uncompensated, which is about +6 dB when
    the second line comes in, and a voice count that changes the level is a
    trap, not a feature - but the voices are DECORRELATED, which is the whole
    point of the 120 degree offsets, so they sum in POWER and not in amplitude.
    Measured on broadband material at the plan's default depth, three taps are
    +4.77 dB over one, i.e. exactly sqrt(3): 1/sqrt(voices) puts that back to
    0.00 dB, where 1/voices leaves the channel 4.77 dB DOWN for the crime of
    adding a voice. The one setting this does not hold is depth exactly zero,
    where the taps coincide and three of them genuinely are +4.8 dB. No
    constant serves both cases - the level of a multi-voice chorus depends on
    how far apart the taps are, which is depth's job - and depth zero is the
    setting that asks three voices to be one.

    Two parameters are VARIANTS - voices and through-zero. Both change topology
    rather than a value: a third comb appearing instantly clicks, and
    through-zero starts delaying the dry path by the whole centre delay, which
    is a step of some tens of milliseconds and a change of reported latency. The
    slot fades out, resets and commits them at silence. MODE IS NOT A VARIANT
    and in fact changes nothing here: chorus and flanger differ only by the
    delay range and the feedback amount the GUI offers, and a second, narrower
    clamp inside the module would silently disagree with the value the app shows
    to a user who arrived by OSC.

    Mix 0 is bit-transparent and free - WHILE THROUGH-ZERO IS OFF: the block is
    not touched at all, not even multiplied by one, so negative zeros and
    denormals survive it. With through-zero ON there is nothing transparent to
    fall back to, because the dry leg is itself a delay: taking the early-out
    would swap x[n - centre] for x[n] the instant the mix settled, a step of up
    to twice the signal amplitude and a channel that jumps up to 30 ms forward
    in time in the middle of a block. Through-zero therefore keeps running at
    mix 0 and pays for the delay line it asked for - which is also what makes
    getLatencySamples() honest there, instead of reporting a delay on a block
    it had just passed through untouched. The LFO phase still advances while
    idle, because several channels sharing a rate are kept apart by
    effectModPhase and a channel whose phase froze while its mix was down would
    come back in the wrong place.

    Shapes beyond the sine are this plan's addition. The discontinuous ones
    (square, sawtooth) step the delay time at the edge and therefore click; that
    is inherent to asking for them, not a defect of the implementation.

    One deviation to know about before A/B-ing against the patch: the shape
    parameter is a LFOWaveforms id, as the plan's parameter table asks, and
    LFOWaveforms::Sine is -cos - it sits at -1 at phase 0, i.e. at the SHORTEST
    delay - while fx_flanger.gendsp's phasor -> sin starts at 0, at the centre
    delay. The two are a quarter cycle apart, so the prototype's sound is this
    module at phaseDeg 90, where -cos is zero and rising (fx_chorus.gendsp's
    cycle, a cosine, is phaseDeg 180 - half a cycle, not three quarters, since
    -cos and +cos are opposites). Matching the patch instead would have put this
    one LFO a quarter cycle away from every other LFO in the product, which is
    the worse surprise; the phase parameter closes the gap either way.

    Latency is zero unless through-zero is on, when it is the centre delay -
    at every mix setting, because the dry leg is delayed at every mix setting.
*/
class ModulationModule : public IEffectModule
{
public:
    /** Three voices fill the cycle exactly at 120 degrees apart. */
    static constexpr int kMaxVoices = 3;

    /** The published effectModDelay ceiling, in milliseconds. At 100 % depth
        the modulated read reaches twice it, which is what the line is sized
        for - reading past the end would fold the sweep back on itself. */
    static constexpr double kMaxCentreMs = 30.0;

    ModuleId type() const noexcept override { return ModuleId::Mod; }

    void prepare (const ChainConfig& config) override
    {
        sampleRate = config.sampleRate > 0.0 ? config.sampleRate : 48000.0;

        maxCentre = static_cast<float> (kMaxCentreMs * 0.001 * sampleRate);
        const int centreSamplesMax = static_cast<int> (std::ceil (maxCentre)) + 1;

        wetLine.prepare (2 * centreSamplesMax);
        dryLine.prepare (centreSamplesMax);

        loCut.prepare (sampleRate);
        fbDc.prepare (sampleRate, 5.0f);
        fbHiCut.prepare (sampleRate);

        // Fixed gentle damping of the loop, kept below Nyquist at every device
        // rate so the design cannot degenerate at 32 kHz.
        float damping = 12000.0f;
        const float ceiling = 0.45f * static_cast<float> (sampleRate);
        if (damping > ceiling) damping = ceiling;
        if (damping < 1000.0f) damping = 1000.0f;
        fbHiCut.setParameters (6 /* HighCut */, damping, 0.0f, 0.7071068f, 0.7f);

        lfo.prepare (sampleRate);

        centre.setTimeConstant (sampleRate, kParamTauSeconds);
        centre.setSnapEpsilon (1.0e-3f);                // samples, not a unit gain
        depth.setTimeConstant (sampleRate, kParamTauSeconds);
        feedbackGain.setTimeConstant (sampleRate, kParamTauSeconds);
        wetMix.setTimeConstant (sampleRate, kParamTauSeconds);
        phaseOffset.setTimeConstant (sampleRate, kParamTauSeconds);
        loCutHz.setTimeConstant (sampleRate, kParamTauSeconds);
        loCutHz.setSnapEpsilon (0.5f);                  // hertz

        noiseKey = spatcore::dsp::FrDiffusion::makeKey (static_cast<int> (config.noiseKey),
                                                        static_cast<int> (ModuleId::Mod));

        snapOnNextApply = true;
        reset();
    }

    void reset() noexcept override
    {
        clearTail();
        lfo.reset();
        cycleCount = 0.0;
        lastPhase = 0.0f;
        idleCleared = false;

        centre.snap (centre.getTarget());
        depth.snap (depth.getTarget());
        feedbackGain.snap (feedbackGain.getTarget());
        wetMix.snap (wetMix.getTarget());
        phaseOffset.snap (phaseOffset.getTarget());
        loCutHz.snap (loCutHz.getTarget());
    }

    ParamApplyInfo applyParams (const EffectChannelParams& params, int) noexcept override
    {
        const ModulationParams& m = params.mod;

        float centreTarget = clamp (m.delayMs, 0.1f, static_cast<float> (kMaxCentreMs))
                                 * static_cast<float> (sampleRate * 0.001);
        centreTarget = clamp (centreTarget, 1.0f, maxCentre);
        centre.setTarget (centreTarget);

        depth.setTarget (clamp (m.depth, 0.0f, 100.0f) * 0.01f);
        feedbackGain.setTarget (clampSigned (m.feedback, -95.0f, 95.0f) * 0.01f);
        wetMix.setTarget (clamp (m.mix, 0.0f, 100.0f) * 0.01f);
        phaseOffset.setTarget (clamp (m.phaseDeg, 0.0f, 360.0f) * (1.0f / 360.0f));
        loCutHz.setTarget (clamp (m.loCutHz, 20.0f, 2000.0f));

        lfo.setRateHz (clamp (m.rateHz, 0.05f, 10.0f));   // the phase stays continuous

        shapeId = m.shape > 8 ? 8 : static_cast<int> (m.shape);

        pendingVoices = m.voices < 1 ? 1
                                     : (m.voices > kMaxVoices ? kMaxVoices
                                                              : static_cast<int> (m.voices));
        pendingThroughZero = m.throughZero != 0 ? 1 : 0;

        if (snapOnNextApply)
        {
            centre.snap (centre.getTarget());
            depth.snap (depth.getTarget());
            feedbackGain.snap (feedbackGain.getTarget());
            wetMix.snap (wetMix.getTarget());
            phaseOffset.snap (phaseOffset.getTarget());
            loCutHz.snap (loCutHz.getTarget());
            runningVoices = pendingVoices;
            runningThroughZero = pendingThroughZero;
            snapOnNextApply = false;
        }

        // Reported from the RUNNING topology, so a through-zero edit still
        // waiting for silence does not announce a latency it is not producing.
        updateReportedLatency();

        return { m.bypass != 0,
                 pendingVoices != runningVoices || pendingThroughZero != runningThroughZero };
    }

    void commitPendingVariant() noexcept override
    {
        runningVoices = pendingVoices;
        runningThroughZero = pendingThroughZero;
        dryLine.reset();
        updateReportedLatency();
    }

    void process (float* inout, int numSamples) noexcept override
    {
        if (inout == nullptr || numSamples <= 0)
            return;

        if (! wetLine.isPrepared())
            return;                                 // process() before prepare()

        // Fully dry AND the dry leg is the input itself: leave the buffer
        // exactly as it is. Clearing the state once on the way in means raising
        // the mix again fades in from silence rather than replaying a moment
        // from before.
        //
        // Through-zero disqualifies the early-out: there the dry leg is a
        // delay, so "leave the buffer alone" is not the quiet continuation of
        // the running path, it is a jump from x[n - centre] to x[n]. Running on
        // at mix 0 costs a block of work on a channel nobody is listening to,
        // which is the price of the topology, not a missed optimisation.
        if (wetMix.isSettled() && wetMix.getCurrent() == 0.0f && runningThroughZero == 0)
        {
            runIdle (numSamples);
            return;
        }

        idleCleared = false;
        updateLoCut();

        const int voices = runningVoices;
        const float voiceGain = 1.0f / std::sqrt (static_cast<float> (voices));
        const float maxRead = static_cast<float> (wetLine.getMaxDelaySamples());
        const bool throughZero = runningThroughZero != 0;

        float loudest = 0.0f;

        for (int i = 0; i < numSamples; ++i)
        {
            const float x = inout[i];

            const float c = centre.next();
            const float d = depth.next();
            const float g = feedbackGain.next();
            const float w = wetMix.next();
            const float off = phaseOffset.next();
            loCutHz.next();                             // glides in step, read once per block

            const double cycles = advancePhase();

            const float in = loCut.processSample (x) + fbState;
            wetLine.write (in);

            float wet = 0.0f;
            float firstTap = 0.0f;

            for (int v = 0; v < voices; ++v)
            {
                const double p = cycles + static_cast<double> (off)
                                        + static_cast<double> (v) * (1.0 / 3.0);

                float t = c * (1.0f + d * lfoValue (p));

                // A minimum of one sample: the loop reads a line it has already
                // written this sample, so a shorter read would close a
                // delay-free feedback path. Negated so a NaN lands on the floor.
                if (! (t > 1.0f)) t = 1.0f;
                if (t > maxRead)  t = maxRead;

                const float tap = wetLine.readLinear (t);
                if (v == 0)
                    firstTap = tap;

                wet += tap;
            }

            wet *= voiceGain;

            fbState = fbHiCut.processSample (fbDc.processSample (firstTap)) * g;

            float dry = x;

            if (throughZero)
            {
                dryLine.write (x);
                dry = dryLine.readLinear (c);
            }

            if (w >= 1.0f)
                inout[i] = wet;
            else if (w > 0.0f)
                inout[i] = dry + w * (wet - dry);
            else
                inout[i] = dry;

            const float ax = std::fabs (x);
            const float ai = std::fabs (in);
            const float at = std::fabs (firstTap);
            if (ax > loudest) loudest = ax;
            if (ai > loudest) loudest = ai;
            if (at > loudest) loudest = at;
        }

        // The recursive parts park in denormal territory for ever once the
        // input stops, where x86 runs an order of magnitude slower - a silent
        // channel would cost more than a loud one. The delay lines need no such
        // help: they refill with true zeros on their own.
        //
        // All THREE of the signals that touch this state get a vote. The INPUT,
        // because the low cut holds a large x[n-1] while its output sits at zero
        // on a DC input, and clearing that would step. The TAP, because that is
        // the tail. And what is actually WRITTEN into the line, because the
        // other two miss the case that matters most: when a DC input stops, the
        // low cut emits a step response that decays for some 330 ms at 48 kHz
        // while the input is an exact zero and the tap - still reading the
        // settled DC region a whole delay back - is an exact zero too. Flushing
        // there truncates a hot filter to zero in one sample, and the click
        // lands at stepSample + delay + blockSize, so it gets LOUDER as the
        // buffer gets shorter: measured -8.5 dBFS at a 64-sample block.
        if (! (loudest > 1.0e-15f))
        {
            loCut.reset();
            fbDc.reset();
            fbHiCut.reset();
            fbState = 0.0f;
        }
    }

    int getLatencySamples() const noexcept override
    {
        return latencySamples.load (std::memory_order_relaxed);
    }

private:
    static float clamp (float v, float lo, float hi) noexcept
    {
        // Negated comparisons so a NaN parameter lands on the low bound rather
        // than propagating into the audio. Every parameter that takes this one
        // has its SAFE end at the bottom: no depth, the shortest delay, the
        // lowest cut, the slowest rate, a dry mix.
        if (! (v > lo)) return lo;
        if (v > hi)     return hi;
        return v;
    }

    /** The same clamp for a BIPOLAR parameter, where the low bound is not the
        safe end but the far one: -95 % feedback is maximum INVERTED
        regeneration, the loudest and least recoverable thing this module can be
        asked for, and it is where the negated comparison above sends a NaN. A
        NaN here lands on zero - the neutral value - instead. */
    static float clampSigned (float v, float lo, float hi) noexcept
    {
        if (v != v) return 0.0f;                    // NaN -> neutral, not -95 %
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    }

    /** Everything the audio has left behind. Deliberately NOT the LFO phase:
        reset() puts that back to zero as the contract asks, but the idle path
        must not, or a channel whose mix was down would rejoin its linked
        partners at the wrong point of a shared sweep. */
    void clearTail() noexcept
    {
        wetLine.reset();
        dryLine.reset();
        loCut.reset();
        fbDc.reset();
        fbHiCut.reset();
        fbState = 0.0f;
        lastLoCutHz = -1.0f;
    }

    void runIdle (int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
            advancePhase();

        // Wake up settled rather than gliding in from wherever the glide had
        // reached when the mix arrived at zero.
        centre.snap (centre.getTarget());
        depth.snap (depth.getTarget());
        feedbackGain.snap (feedbackGain.getTarget());
        phaseOffset.snap (phaseOffset.getTarget());
        loCutHz.snap (loCutHz.getTarget());

        if (! idleCleared)
        {
            clearTail();
            idleCleared = true;
        }
    }

    /** The phase as an absolute count of cycles since the last reset. The
        offset voices need it unwrapped: wrapping first and adding the offset
        afterwards loses which cycle a voice is in, which the random shape needs
        in order to pick its endpoints. */
    double advancePhase() noexcept
    {
        const float ph = lfo.nextPhase();

        if (ph < lastPhase)
            cycleCount += 1.0;

        lastPhase = ph;
        return cycleCount + static_cast<double> (ph);
    }

    float lfoValue (double cyclePosition) const noexcept
    {
        const double whole = std::floor (cyclePosition);
        const float frac = static_cast<float> (cyclePosition - whole);

        if (shapeId == spatcore::dsp::LFOWaveforms::Random)
        {
            // LfoPhasor keeps ONE pair of random endpoints, tied to its own
            // wrap, so a phase-shifted voice asking it for a value would read
            // the first voice's segment at the wrong place. The endpoints are a
            // pure function of the cycle index, so every voice can derive its
            // own and the stream stays reproducible.
            const auto index = static_cast<std::uint32_t> (static_cast<std::int64_t> (whole));
            const float a = spatcore::dsp::FrDiffusion::hashNoiseBipolar (index, noiseKey);
            const float b = spatcore::dsp::FrDiffusion::hashNoiseBipolar (index + 1u, noiseKey);
            return a + (b - a) * frac;
        }

        return spatcore::dsp::LfoPhasor::shapeValue (shapeId, frac);
    }

    void updateLoCut() noexcept
    {
        const float hz = loCutHz.getCurrent();

        if (lastLoCutHz > 0.0f && std::fabs (hz - lastLoCutHz) < 0.5f)
            return;                                     // not worth recomputing

        lastLoCutHz = hz;

        // Q = 0.6 is the prototype's alpha = sin(w0)/1.2, whose codebox comment
        // reads "1.2 = 2*Q". The cookbook high pass here is the same design, so
        // the wet leg is filtered exactly as the patch filters it.
        loCut.setParameters (1 /* LowCut */, hz, 0.0f, 0.6f, 0.7f);
    }

    void updateReportedLatency() noexcept
    {
        const int reported = runningThroughZero != 0
                               ? static_cast<int> (std::lround (centre.getTarget()))
                               : 0;
        latencySamples.store (reported, std::memory_order_relaxed);
    }

    double sampleRate = 48000.0;
    float maxCentre = 1440.0f;

    spatcore::dsp::FractionalDelayLine wetLine, dryLine;
    spatcore::dsp::OutputEQBiquadFilter loCut, fbHiCut;
    spatcore::dsp::DcBlocker fbDc;
    spatcore::dsp::LfoPhasor lfo;

    spatcore::dsp::OnePoleSmoother centre, depth, feedbackGain, wetMix, phaseOffset, loCutHz;

    double cycleCount = 0.0;
    float lastPhase = 0.0f;
    float fbState = 0.0f;
    float lastLoCutHz = -1.0f;

    int shapeId = spatcore::dsp::LFOWaveforms::Sine;
    int runningVoices = 1, pendingVoices = 1;
    int runningThroughZero = 0, pendingThroughZero = 0;

    std::uint32_t noiseKey = 1;
    bool snapOnNextApply = true;
    bool idleCleared = false;

    std::atomic<int> latencySamples { 0 };
};

} // namespace spatcore::effects
