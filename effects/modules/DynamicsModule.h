#pragma once

#include "../EffectModule.h"
#include "../../dsp/EnvelopeFollower.h"
#include "../../dsp/FastDecibels.h"
#include "../../dsp/FractionalDelayLine.h"
#include "../../dsp/OnePoleSmoother.h"
#include "../../dsp/OutputEQBiquadFilter.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>

namespace spatcore::effects
{

/**
    A compressor stage followed by an expander stage, from the user's gen~
    prototype.

    The topology is the prototype's and is unusual on purpose: BOTH detectors
    tap the module INPUT, not the output of the stage in front of them. Only the
    audio runs in series. Compressing therefore does not move the expander's
    threshold, so the expander keeps opening and closing on the dynamics the
    player actually produced rather than on a signal whose dynamics have just
    been flattened, which is what lets the pair work as "tame the peaks, then
    close the gaps" instead of as two effects arguing about the same envelope.

    Three things the prototype does are deliberately not reproduced.

    Its expander slope is (1 - 1/R), copied from its own compressor, which
    realises a ratio of 2 - 1/R: 1.5:1 at R = 2 and never steeper than 2:1
    however far the control is pushed, so the gate at the top of the range
    cannot be reached at all. A downward expander needs (R - 1), which is what
    runs here, with the range floor and the hold timer that make it a gate.

    Its compressor attack and release reach the wrong inlets of its slide. The
    smoothed quantity is the LINEAR gain, which FALLS when compression engages,
    so in the prototype Crelease times the onset and Cattack the recovery -
    backwards, and only on the compressor, since the expander's identical wiring
    is the conventional one for a gate. Here attack always times the gain moving
    towards more reduction and release the return to unity, on both stages.

    Its sidechain high cut defaults to 20 Hz, typed at the top level after a
    sub-patch inlet that was named for a shelf, which leaves a 20 Hz low pass
    across the detector and annihilates it. The default here is 20 kHz, bounded
    below Nyquist - see applyParams, where raising that default turned out to
    need a guard the prototype never needed at 20 Hz.

    Three smaller departures, for the record.

    The prototype recomputes both sidechain biquads from cos and sin EVERY
    SAMPLE, because all fourteen of its controls are signal inlets; here the
    coefficients are cached and recomputed on a parameter change. That is also
    why the four sidechain frequencies do not ramp although the plan's table
    marks them "Ramp yes": setParameters recomputes in place over hot history.
    The filters feed the detector and never the audio, so a coefficient jump is
    a detector transient, not a click - which is the trade this takes.

    The prototype's atodb floors at -999 dB; FastDecibels::gainToDb floors at
    -200. Immaterial to both laws, since the compressor branches to 0 dB below
    its threshold and the expander's result is clamped by the range.

    The gain computers BRANCH rather than transcribe the prototype's
    `1 + gate*(dbtoa(g) - 1)`. That idiom evaluates dbtoa(g) even where the gate
    is about to discard it: on a silent detector, whose atodb floors at -999 dB,
    (T - L)(1 - 1/R) at Cratio 100 asks for dbtoa(+989 dB), which overflows
    float32 to inf, and the gate then multiplies that inf by zero and produces
    NaN - a hazard the verified spec calls out. Branching removes it.

    lookaheadMs and compDetectorDelayMs are opposites and both exist.
    Lookahead delays the AUDIO so the reduction is already in place when the
    transient arrives, and it costs reported latency. The detector delay delays
    the DETECTOR so the front of a transient passes at unity and the reduction
    lands behind it, and it costs no latency at all. A slow attack is not a
    substitute for the second: a slow attack starts reducing immediately and
    merely ramps.

    The gain state starts at UNITY. The prototype's slide registers start at 0
    while their target is 1, so it fades in from silence over the attack time
    every time the DSP starts; here the first block comes out at full level.

    Peak detection is the prototype's abs; the plan's RMS option is a one-pole
    on x squared over a fixed window. Switching between them changes what the
    detector state MEANS, and a lookahead change moves the reported latency, so
    both wait for the slot to reach silence rather than stepping mid-signal.

    prepare() allocates the two delay lines; nothing else allocates.
*/
class DynamicsModule : public IEffectModule
{
public:
    ModuleId type() const noexcept override { return ModuleId::Dyn; }

    void prepare (const ChainConfig& config) override
    {
        sampleRate = config.sampleRate > 0.0 ? config.sampleRate : 48000.0;
        const float sr = static_cast<float> (sampleRate);

        maxLookaheadSamples = static_cast<int> (kMaxLookaheadMs * 0.001f * sr) + 1;
        maxDetectorDelaySamples = static_cast<int> (kMaxDetectorDelayMs * 0.001f * sr) + 1;

        audioLine.prepare (maxLookaheadSamples);
        detectorLine.prepare (maxDetectorDelaySamples);

        prepareStage (compressor);
        prepareStage (expander);

        compKneeDb.setTimeConstant (sampleRate, kParamTauSeconds);
        compKneeDb.setSnapEpsilon (1.0e-3f);                // decibels
        expRangeDb.setTimeConstant (sampleRate, kParamTauSeconds);
        expRangeDb.setSnapEpsilon (1.0e-3f);
        compDelaySamples.setTimeConstant (sampleRate, kParamTauSeconds);
        compDelaySamples.setSnapEpsilon (1.0e-2f);          // samples
        // The class default, stated rather than inherited: unlike the three
        // above, this smoother carries a LINEAR gain, spanning 0.063 (-24 dB)
        // to about 484 (+24 dB of makeup on top of the steepest auto makeup).
        // 1e-4 is a 0.014 dB arrival at the quiet end and vanishing at the loud
        // one, so there is nothing to scale it to; what matters is that it is
        // absolute, because the early-out in process() needs a unity target to
        // arrive on exactly 1.0f.
        makeupGain.setTimeConstant (sampleRate, kParamTauSeconds);
        makeupGain.setSnapEpsilon (1.0e-4f);

        runningDetector = pendingDetector = 0;
        pendingLookaheadSamples = 0;
        runningLookaheadSamples.store (0, std::memory_order_relaxed);
        applyDetectorMode();

        snapOnNextApply = true;
        reset();
    }

    void reset() noexcept override
    {
        clearStage (compressor);
        clearStage (expander);

        detectorLine.reset();
        audioLine.reset();
        expHoldCounter = 0;

        snapSmoothers();
        meterDb.store (0.0f, std::memory_order_relaxed);
    }

    ParamApplyInfo applyParams (const EffectChannelParams& params, int instance) noexcept override
    {
        const DynamicsParams& d = params.dyn[instance & 1];
        const float sr = static_cast<float> (sampleRate);

        // OutputEQBiquadFilter clamps its frequency to 20..20000 Hz and has no
        // Nyquist guard, so a 20 kHz low pass - the SHIPPED DEFAULT of both
        // sidechains - designs w0 > pi at any rate under 40 kHz: alpha =
        // sin(w0)/1.2 turns negative, a2 = (1 - alpha)/(1 + alpha) leaves the
        // unit circle and the detector chain diverges. Measured: |pole| 1.967
        // at 32 kHz, 1.643 at 22.05 kHz, exactly 1.000 at 40 kHz, safe from
        // 44.1 kHz up. The damage is silent rather than loud, which is worse:
        // the audio never passes through these filters, so the slot's NaN trap
        // never fires; instead gainToDb(NaN) floors at kMinDb, both gain
        // computers read -200 dBFS and the meter reports a reduction with no
        // relation to the signal. The app offers whatever the device reports,
        // so 32 kHz and 22.05 kHz are selectable. 0.45 * sr keeps the design
        // well inside the circle at every rate. The low cut needs no bound: its
        // 2000 Hz ceiling is under Nyquist at any rate the biquad accepts.
        const float scHiMax = std::min (20000.0f, 0.45f * sr);
        const float scHiMin = std::min (1000.0f, scHiMax);

        const bool wantComp = d.compOn != 0;
        const bool wantExp  = d.expOn != 0;

        // A stage that has been off carries the filter memory and the pending
        // detector history of whatever was playing when it was switched off.
        // Starting it again on that would duck on a sound that stopped minutes
        // ago, so its state is cleared on the edge. The delay line fill costs a
        // microsecond and happens once per switch-on, never per block.
        if (wantComp && ! compOn)
        {
            clearStage (compressor);
            detectorLine.reset();
        }

        if (wantExp && ! expOn)
        {
            clearStage (expander);
            expHoldCounter = 0;
        }

        compOn = wantComp;
        expOn = wantExp;

        const float compThreshold = clamp (d.compThresholdDb, -60.0f, 0.0f);
        const float compRatio = clamp (d.compRatio, 1.0f, 100.0f);

        compressor.thresholdDb.setTarget (compThreshold);
        compressor.slope.setTarget (1.0f / compRatio - 1.0f);          // never positive
        compKneeDb.setTarget (clamp (d.compKneeDb, 0.0f, 24.0f));
        compressor.gain.fallCoef = coefficientFor (clamp (d.compAttackMs, 0.05f, 200.0f));
        compressor.gain.riseCoef = coefficientFor (clamp (d.compReleaseMs, 5.0f, 2000.0f));
        compDelaySamples.setTarget (delaySamplesFor (clamp (d.compDetectorDelayMs, 0.0f, kMaxDetectorDelayMs),
                                                     detectorLine.getMaxDelaySamples()));
        compressor.loCut.setParameters (1, clamp (d.compScLoCutHz, 20.0f, 2000.0f), 0.0f, kSidechainQ, 0.7f);
        compressor.hiCut.setParameters (6, clamp (d.compScHiCutHz, scHiMin, scHiMax), 0.0f, kSidechainQ, 0.7f);

        expander.thresholdDb.setTarget (clamp (d.expThresholdDb, -90.0f, 0.0f));
        expander.slope.setTarget (clamp (d.expRatio, 1.0f, 100.0f) - 1.0f);   // never negative
        expRangeDb.setTarget (clamp (d.expRangeDb, -80.0f, 0.0f));
        expander.gain.riseCoef = coefficientFor (clamp (d.expAttackMs, 0.05f, 200.0f));
        expander.gain.fallCoef = coefficientFor (clamp (d.expReleaseMs, 5.0f, 2000.0f));
        expHoldSamples = static_cast<int> (clamp (d.expHoldMs, 0.0f, 500.0f) * 0.001f * sr + 0.5f);
        expander.loCut.setParameters (1, clamp (d.expScLoCutHz, 20.0f, 2000.0f), 0.0f, kSidechainQ, 0.7f);
        expander.hiCut.setParameters (6, clamp (d.expScHiCutHz, scHiMin, scHiMax), 0.0f, kSidechainQ, 0.7f);

        float makeup = clamp (d.makeupDb, -24.0f, 24.0f);

        // Half of what the gain computer takes off a signal sitting exactly at
        // the threshold. Compensating in full makes every ratio sound louder
        // than the last, and the operator then hears "better" rather than
        // hearing what the compressor is doing.
        if (d.autoMakeup != 0 && wantComp)
            makeup += -compThreshold * (1.0f - 1.0f / compRatio) * 0.5f;

        makeupGain.setTarget (spatcore::dsp::FastDecibels::dbToGain (makeup));

        // A stage that is off never advances its smoothers, so they are held at
        // their targets: switching it back on starts settled instead of gliding
        // from whatever was current when it went quiet.
        if (! wantComp) snapCompressorSmoothers();
        if (! wantExp)  snapExpanderSmoothers();

        pendingDetector = (d.detector != 0) ? 1 : 0;
        pendingLookaheadSamples = static_cast<int> (clamp (d.lookaheadMs, 0.0f, kMaxLookaheadMs) * 0.001f * sr + 0.5f);
        if (pendingLookaheadSamples > maxLookaheadSamples)
            pendingLookaheadSamples = maxLookaheadSamples;

        if (snapOnNextApply)
        {
            runningDetector = pendingDetector;
            runningLookaheadSamples.store (pendingLookaheadSamples, std::memory_order_relaxed);
            applyDetectorMode();
            snapSmoothers();
            snapOnNextApply = false;
        }

        return { d.bypass != 0,
                 pendingDetector != runningDetector
                     || pendingLookaheadSamples != runningLookaheadSamples.load (std::memory_order_relaxed) };
    }

    void commitPendingVariant() noexcept override
    {
        runningDetector = pendingDetector;
        runningLookaheadSamples.store (pendingLookaheadSamples, std::memory_order_relaxed);

        applyDetectorMode();
        audioLine.reset();
        detectorLine.reset();
        compressor.detector.reset();
        expander.detector.reset();
    }

    void process (float* inout, int numSamples) noexcept override
    {
        const bool compActive = compOn;
        const bool expActive = expOn;
        const int lookahead = runningLookaheadSamples.load (std::memory_order_relaxed);

        // Nothing switched on and nothing to delay: leave the buffer alone.
        // Bit-transparency at this setting is guarded twice - the loop below
        // also writes through rather than multiplying when the total gain is
        // exactly one - but only this early-out makes a silent channel free,
        // because it skips both sidechains and both detectors as well.
        if (! compActive && ! expActive && lookahead == 0
             && makeupGain.isSettled() && makeupGain.getCurrent() == 1.0f)
        {
            meterDb.store (0.0f, std::memory_order_relaxed);
            return;
        }

        float minGain = 1.0f;
        compressor.scPeak = 0.0f;
        expander.scPeak = 0.0f;

        for (int i = 0; i < numSamples; ++i)
        {
            const float x = inout[i];
            float gain = 1.0f;

            if (compActive)
                gain *= compressorGain (x);

            if (expActive)
                gain *= expanderGain (x);

            float y = x;

            if (lookahead > 0)
            {
                audioLine.write (x);
                y = audioLine.readInteger (lookahead);      // whole samples, so exact
            }

            // A total of exactly one is written through rather than multiplied
            // through. dbToGain(0) is exactly 1, so a stage switched ON at
            // ratio 1 - or simply sitting under its threshold - lands here on
            // every sample, and a multiply by one is NOT the identity on the
            // audio thread: under juce::ScopedNoDenormals a denormal input is
            // flushed to zero by the multiply and survives the plain store.
            // This is what makes "on but doing nothing" transparent to the bit
            // and not merely to the ear. The smoother is stepped either way.
            const float g = gain * makeupGain.next();
            inout[i] = (g == 1.0f) ? y : y * g;

            if (gain < minGain)
                minGain = gain;
        }

        if (numSamples > 0)
            flushSilentDetectors (compActive, expActive);

        meterDb.store (minGain < 1.0f ? spatcore::dsp::FastDecibels::gainToDb (minGain) : 0.0f,
                       std::memory_order_relaxed);
    }

    int getLatencySamples() const noexcept override
    {
        return runningLookaheadSamples.load (std::memory_order_relaxed);
    }

    /** The deepest gain reduction the last block applied, makeup excluded. */
    float getMeterDb() const noexcept override
    {
        return meterDb.load (std::memory_order_relaxed);
    }

private:
    static constexpr float kMaxLookaheadMs = 5.0f;
    static constexpr float kMaxDetectorDelayMs = 50.0f;
    static constexpr float kSidechainQ = 0.6f;          // the prototype's alpha = sin(w0)/1.2
    static constexpr float kRmsWindowMs = 10.0f;
    static constexpr float kSilenceFloor = 1.0e-25f;    // about -500 dBFS

    /** One-pole on the LINEAR gain with separate rise and fall coefficients,
        standing in for the prototype's slide.

        slide is y += (x - y)/n with n in samples; the house coefficient
        1 - exp(-1/n) agrees with 1/n to first order and is about 10 % gentler
        at the fastest attack the plan allows (0.05 ms is n = 2.4 samples at
        48 kHz), which is the only setting where the two are told apart.

        The exact arrival is not decoration. A plain one-pole stalls an ULP
        short of its target, so a compressor that is not compressing would
        multiply every sample by 0.99999994 for ever: not transparent, and a
        failure that reads as a wrong law rather than as rounding. */
    struct GainRamp
    {
        float y = 1.0f;
        float riseCoef = 1.0f, fallCoef = 1.0f;

        void snap (float v) noexcept { y = v; }

        float next (float target) noexcept
        {
            if (y == target)
                return y;

            const float stepped = y + (target > y ? riseCoef : fallCoef) * (target - y);
            y = (stepped == y || std::fabs (target - stepped) <= 1.0e-6f) ? target : stepped;
            return y;
        }
    };

    /** What the two stages have in common: a sidechain, a detector, the gain
        computer's two inputs and the smoothed gain it produces. */
    struct Stage
    {
        spatcore::dsp::OutputEQBiquadFilter loCut, hiCut;
        spatcore::dsp::EnvelopeFollower detector;
        spatcore::dsp::OnePoleSmoother thresholdDb, slope;
        GainRamp gain;
        float scPeak = 0.0f;
    };

    static float clamp (float v, float lo, float hi) noexcept
    {
        // Negated comparisons so a NaN parameter lands on the low bound rather
        // than propagating into the audio.
        if (! (v > lo)) return lo;
        if (v > hi)     return hi;
        return v;
    }

    float coefficientFor (float timeMs) const noexcept
    {
        const float n = static_cast<float> (sampleRate) * timeMs * 0.001f;
        return (n > 0.0f) ? 1.0f - std::exp (-1.0f / n) : 1.0f;
    }

    float delaySamplesFor (float ms, int maxSamples) const noexcept
    {
        const float s = ms * 0.001f * static_cast<float> (sampleRate);
        return s > static_cast<float> (maxSamples) ? static_cast<float> (maxSamples) : s;
    }

    void prepareStage (Stage& s)
    {
        s.loCut.prepare (sampleRate);
        s.hiCut.prepare (sampleRate);
        s.detector.prepare (sampleRate);
        s.thresholdDb.setTimeConstant (sampleRate, kParamTauSeconds);
        s.thresholdDb.setSnapEpsilon (1.0e-3f);
        s.slope.setTimeConstant (sampleRate, kParamTauSeconds);
        s.slope.setSnapEpsilon (1.0e-5f);
    }

    static void clearStage (Stage& s) noexcept
    {
        s.loCut.reset();
        s.hiCut.reset();
        s.detector.reset();
        s.gain.snap (1.0f);                 // unity, so the first block is not attenuated
        s.scPeak = 0.0f;
    }

    void applyDetectorMode() noexcept
    {
        using Mode = spatcore::dsp::EnvelopeFollower::Mode;

        const bool rms = runningDetector != 0;
        rmsDetector = rms;

        // Peak is the prototype's bare abs: times of 0 make the follower
        // instant, so its output IS the rectified sample.
        compressor.detector.setMode (rms ? Mode::Rms : Mode::Peak);
        compressor.detector.setTimes (rms ? kRmsWindowMs : 0.0f, rms ? kRmsWindowMs : 0.0f);
        expander.detector.setMode (rms ? Mode::Rms : Mode::Peak);
        expander.detector.setTimes (rms ? kRmsWindowMs : 0.0f, rms ? kRmsWindowMs : 0.0f);
    }

    void snapCompressorSmoothers() noexcept
    {
        compressor.thresholdDb.snap (compressor.thresholdDb.getTarget());
        compressor.slope.snap (compressor.slope.getTarget());
        compKneeDb.snap (compKneeDb.getTarget());
        compDelaySamples.snap (compDelaySamples.getTarget());
    }

    void snapExpanderSmoothers() noexcept
    {
        expander.thresholdDb.snap (expander.thresholdDb.getTarget());
        expander.slope.snap (expander.slope.getTarget());
        expRangeDb.snap (expRangeDb.getTarget());
    }

    void snapSmoothers() noexcept
    {
        snapCompressorSmoothers();
        snapExpanderSmoothers();
        makeupGain.snap (makeupGain.getTarget());
    }

    /** The detector's level in dB. The RMS follower carries x squared, so its
        decibels are half of a linear gain's. */
    float levelDb (float detectorValue) const noexcept
    {
        const float db = spatcore::dsp::FastDecibels::gainToDb (detectorValue);
        return rmsDetector ? 0.5f * db : db;
    }

    float compressorGain (float x) noexcept
    {
        float sc = compressor.loCut.processSample (x);
        sc = compressor.hiCut.processSample (sc);

        const float mag = std::fabs (sc);
        if (mag > compressor.scPeak)
            compressor.scPeak = mag;

        // Rectify BEFORE the delay and convert AFTER it, as the prototype does:
        // the line carries magnitude, and delaying decibels would filter a
        // different envelope.
        detectorLine.write (compressor.detector.processSample (sc));
        const float level = levelDb (detectorLine.readLinear (compDelaySamples.next()));

        const float over = level - compressor.thresholdDb.next();
        const float slope = compressor.slope.next();
        const float knee = compKneeDb.next();
        const float half = 0.5f * knee;

        // out_dB = T + (L - T)/R above the threshold, which is a gain of
        // (1/R - 1)*over. The knee interpolates that in with a quadratic and
        // collapses to the hard comparison at width 0, because the middle
        // branch cannot be reached once half is 0.
        float gainDb;

        if (over >= half)
            gainDb = slope * over;
        else if (over <= -half)
            gainDb = 0.0f;
        else
        {
            const float t = over + half;
            gainDb = slope * t * t / (2.0f * knee);
        }

        return compressor.gain.next (spatcore::dsp::FastDecibels::dbToGain (gainDb));
    }

    float expanderGain (float x) noexcept
    {
        float sc = expander.loCut.processSample (x);
        sc = expander.hiCut.processSample (sc);

        const float mag = std::fabs (sc);
        if (mag > expander.scPeak)
            expander.scPeak = mag;

        const float level = levelDb (expander.detector.processSample (sc));
        const float over = level - expander.thresholdDb.next();
        const float slope = expander.slope.next();
        const float range = expRangeDb.next();

        // A downward expander: (R - 1) dB of attenuation per dB below the
        // threshold, floored by the range. Hold keeps the gate open for a while
        // after the level last crossed back up, which is what stops a peak
        // detector chattering at twice the signal frequency between the very
        // peaks that opened it.
        float gainDb;

        if (over >= 0.0f)
        {
            expHoldCounter = expHoldSamples;
            gainDb = 0.0f;
        }
        else if (expHoldCounter > 0)
        {
            --expHoldCounter;
            gainDb = 0.0f;
        }
        else
        {
            gainDb = slope * over;
            if (gainDb < range)
                gainDb = range;
        }

        return expander.gain.next (spatcore::dsp::FastDecibels::dbToGain (gainDb));
    }

    /** The sidechain filters are the only state here that can spend a long time
        in denormals, and they feed nothing but the detector. Once a whole block
        has stayed below -500 dBFS, clearing them costs nothing audible and
        keeps a silent channel as cheap as a loud one. The detector delay line
        is left alone on purpose: it drains to true zero by itself once the
        filters are clear, and clearing it would throw away a transient that is
        still waiting to be grabbed. */
    void flushSilentDetectors (bool compActive, bool expActive) noexcept
    {
        // With both stages off the filters are not being advanced at all, so
        // there is nothing to flush - without this the lookahead-only
        // configuration pays six pointless resets on every block.
        if (! compActive && ! expActive)
            return;

        if ((compActive && compressor.scPeak >= kSilenceFloor)
             || (expActive && expander.scPeak >= kSilenceFloor))
            return;

        compressor.loCut.reset();
        compressor.hiCut.reset();
        compressor.detector.reset();
        expander.loCut.reset();
        expander.hiCut.reset();
        expander.detector.reset();
    }

    double sampleRate = 48000.0;

    Stage compressor, expander;
    spatcore::dsp::OnePoleSmoother compKneeDb, compDelaySamples, expRangeDb, makeupGain;
    spatcore::dsp::FractionalDelayLine audioLine, detectorLine;

    int maxLookaheadSamples = 0, maxDetectorDelaySamples = 0;
    int expHoldSamples = 0, expHoldCounter = 0;
    int pendingLookaheadSamples = 0;

    /** Written on the audio thread (applyParams, commitPendingVariant) and read
        by the latency ledger through EffectChain::getLatencySamples on whatever
        thread asks, so a plain int here would be a data race even though every
        read would in practice see one of the two values. Relaxed is enough:
        nothing is published alongside it. DistortionModule's oversampling
        latency is an atomic for exactly this reason. */
    std::atomic<int> runningLookaheadSamples { 0 };

    bool compOn = false, expOn = false;
    bool rmsDetector = false;
    bool snapOnNextApply = true;
    std::uint8_t runningDetector = 0, pendingDetector = 0;

    std::atomic<float> meterDb { 0.0f };
};

} // namespace spatcore::effects
