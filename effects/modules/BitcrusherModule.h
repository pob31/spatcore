#pragma once

#include "../EffectModule.h"
#include "../../dsp/FastDecibels.h"
#include "../../dsp/FrDiffusionModel.h"
#include "../../dsp/OnePoleSmoother.h"
#include "../../dsp/OutputEQBiquadFilter.h"
#include <cmath>
#include <cstdint>

namespace spatcore::effects
{

/**
    Lo-fi degradation: word-length reduction and sample-rate reduction.

    The quantiser is the user's gen~ prototype - round(x * 2^bits) / 2^bits,
    with optional dither added before the rounding. Fractional bit counts are
    allowed, which gives a continuous control rather than 24 steps. 2^bits comes
    from FastDecibels::exp2, which is exact for whole bit counts, so 8 bits is
    exactly a step of 1/256 and not almost.

    The downsampler is this plan's addition: a fractional-rate sample-and-hold.
    The trigger is tested BEFORE the phase accumulates, and the phase starts at
    1, so the first hold lands on sample 0 and every run is the full length. The
    other order shortens the first run by a sample - inaudible, but it makes the
    hold pattern depend on when processing started, which is the kind of thing
    that turns a reproducible render into an almost-reproducible one.

    `filter` chooses between the classic aliasing hold and an anti-aliased one
    (a third-order low pass at 0.45 of the hold rate, as a biquad plus a one
    pole). It is a VARIANT parameter: switching it swaps filter topology, so it
    waits for the slot to reach silence rather than clicking.

    Dither noise is keyed, not random. Two runs of the same session produce the
    same noise, and its index advances every sample whether or not dither is
    switched on, so turning it on mid-show does not depend on when.

    Zero latency. No allocation after prepare().
*/
class BitcrusherModule : public IEffectModule
{
public:
    ModuleId type() const noexcept override { return ModuleId::Crush; }

    void prepare (const ChainConfig& config) override
    {
        sampleRate = config.sampleRate > 0.0 ? config.sampleRate : 48000.0;

        bits.setTimeConstant (sampleRate, kParamTauSeconds);
        bits.setSnapEpsilon (1.0e-3f);              // bits, not a unit gain
        rate.setTimeConstant (sampleRate, kParamTauSeconds);
        rate.setSnapEpsilon (0.5f);                 // hertz
        wet.setTimeConstant (sampleRate, kParamTauSeconds);
        ditherGain.setTimeConstant (sampleRate, kParamTauSeconds);

        antiAlias.prepare (sampleRate);
        noiseKey = spatcore::dsp::FrDiffusion::makeKey (static_cast<int> (config.noiseKey),
                                                        static_cast<int> (ModuleId::Crush));

        snapOnNextApply = true;
        reset();
    }

    /** Gives this instance its own dither stream. Chains hand out one key per
        slot so two crushers in one session never correlate. */
    void setNoiseKey (std::uint32_t key) noexcept { noiseKey = key; }

    void reset() noexcept override
    {
        hold = 0.0f;
        phase = 1.0;                                // first sample takes a fresh hold
        noiseIndex = 0;
        onePoleState = 0.0f;
        antiAlias.reset();
        lastCutoff = -1.0f;

        bits.snap (bits.getTarget());
        rate.snap (rate.getTarget());
        wet.snap (wet.getTarget());
        ditherGain.snap (ditherGain.getTarget());
    }

    ParamApplyInfo applyParams (const EffectChannelParams& params, int) noexcept override
    {
        const BitcrusherParams& c = params.crush;

        bits.setTarget (clamp (c.bits, 1.0f, 24.0f));
        rate.setTarget (clamp (c.rateHz, 100.0f, static_cast<float> (sampleRate)));
        wet.setTarget (clamp (c.mix, 0.0f, 100.0f) * 0.01f);

        // -96 dB is the documented "off", and off must mean silent, not
        // inaudible: a gain of exactly 0 skips the noise entirely.
        ditherGain.setTarget (c.ditherDb <= -95.9f
                                  ? 0.0f
                                  : spatcore::dsp::FastDecibels::dbToGain (clamp (c.ditherDb, -96.0f, 0.0f)));

        pendingFilter = (c.filter != 0) ? 1 : 0;

        if (snapOnNextApply)
        {
            bits.snap (bits.getTarget());
            rate.snap (rate.getTarget());
            wet.snap (wet.getTarget());
            ditherGain.snap (ditherGain.getTarget());
            runningFilter = pendingFilter;
            snapOnNextApply = false;
        }

        return { c.bypass != 0, pendingFilter != runningFilter };
    }

    void commitPendingVariant() noexcept override
    {
        runningFilter = pendingFilter;
        antiAlias.reset();
        onePoleState = 0.0f;
        lastCutoff = -1.0f;
    }

    void process (float* inout, int numSamples) noexcept override
    {
        if (runningFilter != 0)
            updateAntiAliasFilter();

        for (int i = 0; i < numSamples; ++i)
        {
            const float step = spatcore::dsp::FastDecibels::exp2 (bits.next());
            const float hz = rate.next();
            const float w = wet.next();
            const float dg = ditherGain.next();

            const float in = inout[i];
            float x = in;

            if (dg > 0.0f)
                x += spatcore::dsp::FrDiffusion::hashNoiseBipolar (noiseIndex, noiseKey) * dg;

            ++noiseIndex;        // always, so the stream cannot depend on when dither came on

            if (runningFilter != 0)
            {
                x = antiAlias.processSample (x);
                onePoleState += onePoleCoef * (x - onePoleState);
                x = onePoleState;
            }

            // Trigger first, then accumulate: sample 0 takes a hold and every
            // run is the full length.
            if (phase >= 1.0)
            {
                phase -= 1.0;
                hold = std::round (x * step) / step;
            }

            phase += static_cast<double> (hz) / sampleRate;

            if (w >= 1.0f)
                inout[i] = hold;
            else if (w > 0.0f)
                inout[i] = in + w * (hold - in);
            // w <= 0: dry, already in the buffer
        }
    }

    int getLatencySamples() const noexcept override { return 0; }

private:
    static float clamp (float v, float lo, float hi) noexcept
    {
        if (! (v > lo)) return lo;
        if (v > hi)     return hi;
        return v;
    }

    void updateAntiAliasFilter() noexcept
    {
        const float nyquist = static_cast<float> (sampleRate) * 0.5f;
        float cutoff = 0.45f * rate.getCurrent();
        if (cutoff > nyquist * 0.9f) cutoff = nyquist * 0.9f;
        if (cutoff < 20.0f)          cutoff = 20.0f;

        if (lastCutoff > 0.0f && std::fabs (cutoff - lastCutoff) < 1.0f)
            return;                                  // not worth recomputing

        lastCutoff = cutoff;

        // A third-order Butterworth factorises into a complex pair and a real
        // pole: the biquad is the pair, the one-pole is the rest.
        antiAlias.setParameters (6 /* HighCut */, cutoff, 0.0f, 1.0f, 0.7f);
        onePoleCoef = 1.0f - std::exp (-6.2831853071795864f * cutoff / static_cast<float> (sampleRate));
    }

    double sampleRate = 48000.0;

    spatcore::dsp::OnePoleSmoother bits, rate, wet, ditherGain;
    spatcore::dsp::OutputEQBiquadFilter antiAlias;

    float hold = 0.0f;
    double phase = 1.0;
    float onePoleState = 0.0f;
    float onePoleCoef = 1.0f;
    float lastCutoff = -1.0f;

    std::uint32_t noiseIndex = 0;
    std::uint32_t noiseKey = 1;
    std::uint8_t runningFilter = 0, pendingFilter = 0;
    bool snapOnNextApply = true;
};

} // namespace spatcore::effects
