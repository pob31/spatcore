#pragma once

#include "ReverbDelayLine.h"
#include "ReverbLfo.h"
#include "ReverbTailModel.h"
#include "../../EffectPresets.h"
#include "../../../dsp/DcBlocker.h"
#include "../../../dsp/FastDecibels.h"
#include "../../../dsp/FrDiffusionModel.h"
#include "../../../dsp/OnePoleSmoother.h"
#include <cstdint>

namespace spatcore::effects
{

/**
    Model 4: a modulated hall - a sixteen-line feedback delay network whose
    lines are read at slowly moving positions.

    Sixteen lines, prime lengths spaced evenly on a log scale from 997 to 3407
    samples at 48 kHz (21 to 71 ms), scaled by the rate and Size and jittered up
    to 6.25 % per channel. Their outputs go through a three-band decay each,
    are mixed by an orthonormal Hadamard matrix, tapped (with signs drawn per
    channel) for the output, and written back with the input added. Four
    allpass diffusers smear the input first, at 0.75 times Diffusion.

    The modulation is what makes it a hall rather than the FDN again. A fixed
    network has fixed modes, and a long decay lets you hear them: a metallic
    ring on a sustained note. Moving every read point by up to a millisecond
    detunes the modes continuously, so no one of them lasts long enough to
    ring. Four LFOs at ModRate times 1, 1.13, 0.87 and 1.27 give a sine and a
    cosine each; line i takes signal i mod 8, the second eight lines inverted,
    so no two lines move together. The reads are Catmull-Rom, which loses about
    a decibel per pass at a quarter of the rate where a linear read would lose
    three - see ReverbDelayLine.

    The decay law is the FDN's: each line's three-band gains are
    exp2(-9.9658 L / (sr RT60 mult)) for its own length L, so every path through
    the mix loses 60 dB per RT60 in each band, and RT60, the multipliers and
    both crossovers mean what they mean on the other models. Unlike the FDN
    there is no fixed 8 kHz low pass inside: Tone owns brightness.

    Size is build-time (the lengths), everything else runtime; every line is
    allocated in prepare() at the largest size, so a rebuild is new lengths and
    a cleared state. libm-free throughout: a render hashes the same everywhere.
*/
class ModulatedHallModel final : public IEffectReverbModel
{
public:
    static constexpr int kLines = 16;

    /** ModDepth 100 %: the read point moves this far either way. */
    static constexpr float kMaxExcursionMs = 1.0f;

    /** How far a line may move per channel, either way. */
    static constexpr float kLineJitter = 0.0625f;

    /** Output make-up. Sixteen signed taps of an energy-preserving mix come
        out 9.5 dB louder than the FDN (+4.65 dB against the dry on noise at
        RT60 1.5 s, fully wet); a third puts the hall where the FDN sits
        (testModulatedHallModel pins it). */
    static constexpr float kOutputGain = 1.0f / 3.0f;

    void prepare (const ChainConfig& config, const ReverbParams& initial) override
    {
        sampleRate = config.sampleRate > 0.0 ? config.sampleRate : 48000.0;
        rateScale = sampleRate / 48000.0;

        const std::uint32_t key = spatcore::dsp::FrDiffusion::makeKey (static_cast<int> (config.noiseKey & 0x7FFFFFFFu),
                                                                        static_cast<int> (ModuleId::Reverb) + 128);
        auto hash = [key] (int n) { return spatcore::dsp::FrDiffusion::hashNoiseBipolar (static_cast<std::uint32_t> (n), key); };

        maxExcursion = static_cast<int> (kMaxExcursionMs * 0.001 * sampleRate) + 2;

        for (int i = 0; i < kLines; ++i)
        {
            jitter[i] = 1.0f + kLineJitter * hash (i);
            lines[i].prepare (lengthOf (kLineBase[i] * jitter[i], kReverbMaxSize) + maxExcursion + 4);
            inSign[i] = hash (48 + i) < 0.0f ? -1.0f : 1.0f;
            outSign[i] = hash (64 + i) < 0.0f ? -1.0f : 1.0f;
        }

        for (int d = 0; d < 4; ++d)
        {
            diffLen[d] = lengthOf (kDiffuserBase[d] * (1.0f + 0.1f * hash (32 + d)), 1.0f);
            diffusers[d].prepare (diffLen[d] + 1);
        }

        for (int k = 0; k < 4; ++k)
        {
            lfo[k].prepare (sampleRate);
            lfo[k].setStartPhase (0.5 + 0.5 * static_cast<double> (hash (80 + k)));
        }

        depth.setTimeConstant (sampleRate, 0.02f);
        depth.setSnapEpsilon (1.0e-4f);                     // samples
        dc.prepare (sampleRate, 5.0f);

        builtSize = -1.0f;                                  // nothing built yet
        setParams (initial);
        buildAt (pendingSize);
    }

    void reset() noexcept override
    {
        for (auto& line : lines)
            line.reset();
        for (auto& d : diffusers)
            d.reset();
        for (auto& d : decay)
            d.lowState = d.highState = 0.0f;
        for (auto& l : lfo)
            l.reset();

        depth.snap (depth.getTarget());
        dc.reset();
        fresh = true;
    }

    bool setParams (const ReverbParams& params) noexcept override
    {
        const float rt60      = clampParam (params.rt60, 0.2f, 8.0f);
        const float lowMult   = clampParam (params.rt60LowMult, 0.1f, 9.0f);
        const float highMult  = clampParam (params.rt60HighMult, 0.1f, 9.0f);
        const float xLow      = clampParam (params.crossoverLow, 50.0f, 500.0f);
        const float xHigh     = clampParam (params.crossoverHigh, 1000.0f, 10000.0f);

        if (rt60 != decayRt60 || lowMult != decayLow || highMult != decayHigh
            || xLow != crossLow || xHigh != crossHigh)
        {
            decayRt60 = rt60;
            decayLow = lowMult;
            decayHigh = highMult;
            crossLow = xLow;
            crossHigh = xHigh;
            updateDecay();
        }

        inDiff = 0.75f * clampParam (params.diffusion, 0.0f, 1.0f);

        const float rate = clampParam (params.modRateHz, 0.05f, 5.0f);
        for (int k = 0; k < 4; ++k)
            lfo[k].setRateHz (rate * kLfoRatio[k]);

        depth.setTarget (clampParam (params.modDepth, 0.0f, 100.0f) * 0.01f
                         * kMaxExcursionMs * 0.001f * static_cast<float> (sampleRate));

        // An instance that has not run since it was cleared starts at its
        // values rather than gliding in from the last ones it was given.
        if (fresh)
            depth.snap (depth.getTarget());

        pendingSize = quantiseReverbSize (clampParam (params.size, kReverbMinSize, kReverbMaxSize));
        return pendingSize != builtSize;
    }

    void commitPendingVariant() noexcept override
    {
        if (pendingSize != builtSize)
            buildAt (pendingSize);
    }

    void process (float* inout, int numSamples) noexcept override
    {
        fresh = false;

        for (int i = 0; i < numSamples; ++i)
            inout[i] = processSample (inout[i]);

        dc.processBlock (inout, numSamples);
    }

    bool isBuildDifferent (const ReverbParams& params) const noexcept override
    {
        return quantiseReverbSize (clampParam (params.size, kReverbMinSize, kReverbMaxSize)) != builtSize;
    }

    float getBuiltSize() const noexcept override { return builtSize; }

    /** Line i's length at the built size, in samples - for tests. */
    int getLineLength (int i) const noexcept { return (i >= 0 && i < kLines) ? len[i] : 0; }

private:
    /** Primes, log-spaced from 997 to 3407: 21 to 71 ms at 48 kHz. */
    static constexpr float kLineBase[kLines] = { 997.0f, 1087.0f, 1171.0f, 1277.0f, 1381.0f, 1499.0f, 1627.0f, 1777.0f,
                                                 1913.0f, 2083.0f, 2267.0f, 2459.0f, 2663.0f, 2887.0f, 3137.0f, 3407.0f };
    static constexpr float kDiffuserBase[4] = { 211.0f, 163.0f, 467.0f, 353.0f };
    static constexpr float kLfoRatio[4] = { 1.0f, 1.13f, 0.87f, 1.27f };

    /** Each line feeds in a quarter of the input: sixteen quarters squared is
        the input's energy once. */
    static constexpr float kInputGain = 0.25f;

    struct DecayPoint
    {
        float lowState = 0.0f, highState = 0.0f;
        float gLow = 1.0f, gMid = 1.0f, gHigh = 1.0f;

        float process (float x, float cLow, float cHigh) noexcept
        {
            lowState += cLow * (x - lowState);
            highState += cHigh * (x - highState);
            return lowState * gLow + (highState - lowState) * gMid + (x - highState) * gHigh;
        }
    };

    static float clampParam (float v, float lo, float hi) noexcept
    {
        if (! (v > lo)) return lo;
        if (v > hi)     return hi;
        return v;
    }

    int lengthOf (float samplesAt48k, float size) const noexcept
    {
        const int n = static_cast<int> (static_cast<double> (samplesAt48k) * rateScale * size + 0.5);
        return n > 2 ? n : 2;
    }

    /** 1 - exp(-2 pi f / sr), libm-free. */
    float onePole (float hz) const noexcept
    {
        const float sr = static_cast<float> (sampleRate);
        const float f = hz < sr * 0.49f ? hz : sr * 0.49f;
        return 1.0f - spatcore::dsp::FastDecibels::exp2 (-6.2831853071795864f * f / sr * 1.4426950408889634f);
    }

    void buildAt (float size) noexcept
    {
        for (int i = 0; i < kLines; ++i)
            len[i] = lengthOf (kLineBase[i] * jitter[i], size);

        builtSize = size;
        updateDecay();
        reset();
    }

    /** Each line's band gains, for its own length. */
    void updateDecay() noexcept
    {
        cLow = onePole (crossLow);
        cHigh = onePole (crossHigh);

        for (int i = 0; i < kLines; ++i)
        {
            const float perPass = -9.9657842846620870f * static_cast<float> (len[i])
                                  / (static_cast<float> (sampleRate) * decayRt60);
            decay[i].gLow  = spatcore::dsp::FastDecibels::exp2 (perPass / decayLow);
            decay[i].gMid  = spatcore::dsp::FastDecibels::exp2 (perPass);
            decay[i].gHigh = spatcore::dsp::FastDecibels::exp2 (perPass / decayHigh);
        }
    }

    static float allpass (ReverbDelayLine& line, int length, float g, float x) noexcept
    {
        const float delayed = line.readInteger (length);
        const float v = x - g * delayed;
        line.write (v);
        return delayed + g * v;
    }

    /** Orthonormal Hadamard, in place: four butterfly stages and a quarter. */
    static void hadamard (float* v) noexcept
    {
        for (int h = 1; h < kLines; h <<= 1)
        {
            for (int i = 0; i < kLines; i += h << 1)
            {
                for (int j = i; j < i + h; ++j)
                {
                    const float a = v[j];
                    const float b = v[j + h];
                    v[j] = a + b;
                    v[j + h] = a - b;
                }
            }
        }

        for (int i = 0; i < kLines; ++i)
            v[i] *= 0.25f;
    }

    float processSample (float x) noexcept
    {
        for (int d = 0; d < 4; ++d)
            x = allpass (diffusers[d], diffLen[d], inDiff, x);

        const float e = depth.next();
        float mod[8];
        for (int k = 0; k < 4; ++k)
            lfo[k].nextSinCos (mod[2 * k], mod[2 * k + 1]);

        float y[kLines];
        for (int i = 0; i < kLines; ++i)
        {
            const float offset = mod[i & 7] * (i < 8 ? e : -e);
            y[i] = decay[i].process (lines[i].readHermite (static_cast<float> (len[i]) + offset), cLow, cHigh);
        }

        hadamard (y);

        float out = 0.0f;
        for (int i = 0; i < kLines; ++i)
        {
            out += outSign[i] * y[i];
            lines[i].write (y[i] + inSign[i] * kInputGain * x);
        }

        return kOutputGain * out;
    }

    ReverbDelayLine lines[kLines];
    ReverbDelayLine diffusers[4];
    DecayPoint decay[kLines];
    ReverbLfo lfo[4];
    spatcore::dsp::OnePoleSmoother depth;
    spatcore::dsp::DcBlocker dc;

    double sampleRate = 48000.0;
    double rateScale = 1.0;
    int maxExcursion = 0;
    float jitter[kLines] = {};
    float inSign[kLines] = {};
    float outSign[kLines] = {};
    int len[kLines] = {};
    int diffLen[4] = {};

    float inDiff = 0.375f;
    float decayRt60 = 1.5f, decayLow = 1.3f, decayHigh = 0.4f;
    float crossLow = 200.0f, crossHigh = 4000.0f;
    float cLow = 0.0f, cHigh = 0.0f;
    float builtSize = 1.0f;
    float pendingSize = 1.0f;
    bool fresh = true;
};

} // namespace spatcore::effects
