#pragma once

#include "../EffectModule.h"
#include "../../dsp/LfoPhasor.h"
#include "../../dsp/LFOWaveforms.h"
#include "../../dsp/FastDecibels.h"
#include "../../dsp/OnePoleSmoother.h"
#include <atomic>

namespace spatcore::effects
{

/**
    Amplitude modulation, from the user's gen~ prototype.

    Two things about it are deliberate and easy to get wrong.

    The depth is in DECIBELS, so the modulation is exponential in amplitude
    rather than linear. That is what the prototype does and it is why it sounds
    smooth: a linear tremolo spends most of its cycle near full level and then
    drops abruptly, while a dB-linear one moves at a constant musical rate.

    The two waveform legs are PHASE-ALIGNED. The prototype builds them as
    (cos(2*pi*phi) - 1)/2 and -triangle(phi) - both 0 at phase 0 and -1 at phase
    0.5 - because Max's `cycle` is a cosine, not a sine. Writing the first leg
    with a sine (as an earlier draft of the plan did) rotates it a quarter cycle
    against the other, and the blend control then cancels the two shapes against
    each other instead of morphing between them. Here both legs come from
    LFOWaveforms, whose Sine and Triangle are already aligned (-1 at phase 0,
    +1 at phase 0.5), so the blend is a plain crossfade.

    The prototype's wet leg is in*w*(g-1), which subtracts rather than scales;
    that is a bug in the prototype and is not reproduced.

    Two settings are bit-transparent by construction, and the tests hold them to
    it: mix 0, and depth 0 with mix 100 - the latter because FastDecibels
    returns exactly 1 for 0 dB.

    Zero latency, no variant parameters, no allocation after prepare().
*/
class TremoloModule : public IEffectModule
{
public:
    ModuleId type() const noexcept override { return ModuleId::Trem; }

    void prepare (const ChainConfig& config) override
    {
        sampleRate = config.sampleRate > 0.0 ? config.sampleRate : 48000.0;

        lfo.prepare (sampleRate);
        depthDb.setTimeConstant (sampleRate, kParamTauSeconds);
        blend.setTimeConstant (sampleRate, kParamTauSeconds);
        wet.setTimeConstant (sampleRate, kParamTauSeconds);

        snapOnNextApply = true;
        reset();
    }

    void reset() noexcept override
    {
        lfo.reset();
        depthDb.snap (depthDb.getTarget());
        blend.snap (blend.getTarget());
        wet.snap (wet.getTarget());
        meterDb.store (0.0f, std::memory_order_relaxed);
    }

    ParamApplyInfo applyParams (const EffectChannelParams& params, int) noexcept override
    {
        const TremoloParams& t = params.trem;

        lfo.setRateHz (clamp (t.rateHz, 0.05f, 20.0f));    // phase stays continuous
        depthDb.setTarget (clamp (t.depthDb, 0.0f, 60.0f));
        blend.setTarget (clamp (t.shape, 0.0f, 1.0f));
        wet.setTarget (clamp (t.mix, 0.0f, 100.0f) * 0.01f);

        if (snapOnNextApply)
        {
            depthDb.snap (depthDb.getTarget());
            blend.snap (blend.getTarget());
            wet.snap (wet.getTarget());
            snapOnNextApply = false;
        }

        return { t.bypass != 0, false };
    }

    void process (float* inout, int numSamples) noexcept override
    {
        float lastDb = 0.0f;

        for (int i = 0; i < numSamples; ++i)
        {
            const float phase = lfo.nextPhase();
            const float s = blend.next();
            const float d = depthDb.next();
            const float w = wet.next();

            const float sine = spatcore::dsp::LfoPhasor::shapeValue (spatcore::dsp::LFOWaveforms::Sine, phase);
            const float tri  = spatcore::dsp::LfoPhasor::shapeValue (spatcore::dsp::LFOWaveforms::Triangle, phase);

            // Both legs run -1 .. +1 in step, so this is a crossfade, and m
            // lands in [-1, 0]: 0 at phase 0 (unity), -1 at phase 0.5.
            const float m = -0.5f * (1.0f + (sine + s * (tri - sine)));

            lastDb = m * d;
            inout[i] *= (1.0f - w) + w * spatcore::dsp::FastDecibels::dbToGain (lastDb);
        }

        meterDb.store (lastDb, std::memory_order_relaxed);
    }

    int getLatencySamples() const noexcept override { return 0; }

    float getMeterDb() const noexcept override
    {
        return meterDb.load (std::memory_order_relaxed);
    }

private:
    static float clamp (float v, float lo, float hi) noexcept
    {
        // Negated comparisons so a NaN parameter lands on the low bound rather
        // than propagating into the audio.
        if (! (v > lo)) return lo;
        if (v > hi)     return hi;
        return v;
    }

    double sampleRate = 48000.0;
    spatcore::dsp::LfoPhasor lfo;
    spatcore::dsp::OnePoleSmoother depthDb, blend, wet;
    bool snapOnNextApply = true;
    std::atomic<float> meterDb { 0.0f };
};

} // namespace spatcore::effects
