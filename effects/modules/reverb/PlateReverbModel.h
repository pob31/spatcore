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
    Model 1: a plate - Dattorro's figure-of-eight tank (J. Dattorro, "Effect
    Design, Part 1: Reverberator and Other Filters", JAES 45(9), 1997).

    The input runs through four allpass diffusers (142, 107, 379, 277 samples
    at the paper's 29761 Hz), then into two tank halves that feed each other:
    each half is a modulated allpass (672 / 908), a delay (4453 / 4217), an
    allpass (1800 / 2656) and a second delay (3720 / 3163). The output is the
    paper's fourteen signed taps into those lines, its two sides summed to
    mono, then a DC blocker. Every length is rescaled from 29761 Hz to the
    device rate, and the tank's by Size.

    What differs from the paper, and why:

    - Every DECAY point is the FDN's three-band filter rather than the
      paper's damping one-pole and decay multiply. Each band's gain is
      exp2(-9.9658 L / (sr RT60 mult)) for the L samples of tank between two
      decay points, so a signal loses exactly 60 dB per RT60 in each band
      around the loop - RT60, Low and High Mult and both crossovers mean here
      what they mean on the FDN, and a preset can move between models.
    - Diffusion d drives all four diffusion coefficients: the input pairs at
      0.79 d and 0.66 d, the tank's modulated allpasses at 0.50 + 0.21 d, the
      second tank allpasses fixed at 0.50. At d = 0.95 that is the paper.
    - ModDepth 50 % is the paper's excursion (16 samples at 29761 Hz, 0.54 ms);
      ModRate is the LFO. The two halves take sine and cosine, and the
      excursion glides so a depth change does not jump the read position.
    - Per effects channel, each tank line is up to 3 % longer or shorter, the
      LFO starts at its own phase, and each of the fourteen output taps takes
      its own sign, all from the noise key: thirty-two plates built to the
      paper's numbers would be one plate thirty-two times, and would comb
      instead of spreading. The lengths alone are not enough - 3 % moves a
      tap by a fraction of a millisecond, which decorrelates nothing below a
      few kilohertz - so the signs do the spreading at every frequency, as
      the hall's per-channel output signs do (testPlateReverbModel).

    Size is build-time (the lengths), everything else runtime. Every line is
    allocated in prepare() at the largest size, so a rebuild is a matter of
    new lengths and a cleared state - no allocation, audio thread safe.

    All arithmetic is libm-free, so a render hashes the same on every platform.
*/
class PlateReverbModel final : public IEffectReverbModel
{
public:
    static constexpr double kPaperRate = 29761.0;

    /** How far a tank line may move per channel, either way. */
    static constexpr float kLineJitter = 0.03f;

    /** Output make-up: the paper's 0.6 per side, halved for the mono sum, then
        a measured calibration - the paper's gain puts a fully wet plate at
        +1.6 dB against the dry on noise at RT60 1.5 s, 6.5 dB above the FDN,
        and 0.47 brings it level with it (testPlateReverbModel). */
    static constexpr float kOutputGain = 0.6f * 0.5f * 0.47f;

    void prepare (const ChainConfig& config, const ReverbParams& initial) override
    {
        sampleRate = config.sampleRate > 0.0 ? config.sampleRate : 48000.0;
        rateScale = sampleRate / kPaperRate;

        const std::uint32_t key = spatcore::dsp::FrDiffusion::makeKey (static_cast<int> (config.noiseKey & 0x7FFFFFFFu),
                                                                        static_cast<int> (ModuleId::Reverb) + 96);

        for (int i = 0; i < 4; ++i)
        {
            const float j = 1.0f + 0.05f * spatcore::dsp::FrDiffusion::hashNoiseBipolar (static_cast<std::uint32_t> (i), key);
            inLen[i] = lengthOf (kInputAp[i] * j, 1.0f);
            inAp[i].prepare (inLen[i] + 1);
        }

        for (int i = 0; i < kTankLines; ++i)
        {
            jitter[i] = 1.0f + kLineJitter * spatcore::dsp::FrDiffusion::hashNoiseBipolar (static_cast<std::uint32_t> (16 + i), key);
            tank[i].prepare (lengthOf (kTankBase[i] * jitter[i], kReverbMaxSize) + maxExcursion() + 4);
        }

        // The paper's signs, each flipped or not per channel. A flip changes how
        // the taps add, not what they carry: the decay, the bands and the
        // level on average are the paper's.
        for (int t = 0; t < 14; ++t)
            tapSign[t] = spatcore::dsp::FrDiffusion::hashNoiseBipolar (static_cast<std::uint32_t> (64 + t), key) < 0.0f
                             ? -kTaps[t].sign : kTaps[t].sign;

        lfo.prepare (sampleRate);
        lfo.setStartPhase (0.5 + 0.5 * static_cast<double> (spatcore::dsp::FrDiffusion::hashNoiseBipolar (40u, key)));

        excursion.setTimeConstant (sampleRate, 0.02f);
        excursion.setSnapEpsilon (1.0e-4f);                 // samples
        dc.prepare (sampleRate, 5.0f);

        builtSize = -1.0f;                                  // nothing built yet
        setParams (initial);
        buildAt (pendingSize);
    }

    void reset() noexcept override
    {
        for (auto& line : inAp)
            line.reset();
        for (auto& line : tank)
            line.reset();
        for (auto& d : decay)
            d.lowState = d.highState = 0.0f;

        crossL = crossR = 0.0f;
        lfo.reset();
        excursion.snap (excursion.getTarget());
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
        const float diffusion = clampParam (params.diffusion, 0.0f, 1.0f);

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

        inDiff1 = 0.79f * diffusion;
        inDiff2 = 0.66f * diffusion;
        decDiff1 = 0.50f + 0.21f * diffusion;

        lfo.setRateHz (clampParam (params.modRateHz, 0.05f, 5.0f));
        excursion.setTarget (clampParam (params.modDepth, 0.0f, 100.0f) * (16.0f / 50.0f) * static_cast<float> (rateScale));

        // An instance that has not run since it was cleared starts at its
        // values rather than gliding in from the last ones it was given.
        if (fresh)
            excursion.snap (excursion.getTarget());

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

private:
    // Tank lines, in the order the signal meets them: left modulated allpass,
    // left delay, left allpass, left second delay, then the right half.
    enum Line { LAp1, LDel1, LAp2, LDel2, RAp1, RDel1, RAp2, RDel2, kTankLines };

    static constexpr float kInputAp[4] = { 142.0f, 107.0f, 379.0f, 277.0f };
    static constexpr float kTankBase[kTankLines] = { 672.0f, 4453.0f, 1800.0f, 3720.0f,
                                                     908.0f, 4217.0f, 2656.0f, 3163.0f };

    struct Tap { int line; float at; float sign; };

    /** The paper's output taps, both sides (they are summed to mono). */
    static constexpr Tap kTaps[14] =
    {
        { RDel1,  266.0f,  1.0f }, { RDel1, 2974.0f,  1.0f }, { RAp2, 1913.0f, -1.0f }, { RDel2, 1996.0f,  1.0f },
        { LDel1, 1990.0f, -1.0f }, { LAp2,   187.0f, -1.0f }, { LDel2, 1066.0f, -1.0f },
        { LDel1,  353.0f,  1.0f }, { LDel1, 3627.0f,  1.0f }, { LAp2, 1228.0f, -1.0f }, { LDel2, 2673.0f,  1.0f },
        { RDel1, 2111.0f, -1.0f }, { RAp2,   335.0f, -1.0f }, { RDel2,  121.0f, -1.0f },
    };

    /** The FDN's three-band decay: one-pole splits at the two crossovers, a
        gain per band. */
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

    int lengthOf (float paperSamples, float size) const noexcept
    {
        const int n = static_cast<int> (static_cast<double> (paperSamples) * rateScale * size + 0.5);
        return n > 1 ? n : 1;
    }

    int maxExcursion() const noexcept
    {
        return static_cast<int> (32.0 * rateScale) + 2;       // ModDepth 100 %
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
        for (int i = 0; i < kTankLines; ++i)
            len[i] = lengthOf (kTankBase[i] * jitter[i], size);

        for (int t = 0; t < 14; ++t)
        {
            int at = lengthOf (kTaps[t].at * jitter[kTaps[t].line], size);
            if (at > len[kTaps[t].line] - 1)
                at = len[kTaps[t].line] - 1;
            tapAt[t] = at;
        }

        builtSize = size;
        updateDecay();
        reset();
    }

    /** Each decay point's band gains, for the stretch of tank since the one
        before it: the modulated allpass and first delay, or the second
        allpass and second delay. */
    void updateDecay() noexcept
    {
        cLow = onePole (crossLow);
        cHigh = onePole (crossHigh);

        const int segment[4] = { len[LAp1] + len[LDel1], len[LAp2] + len[LDel2],
                                 len[RAp1] + len[RDel1], len[RAp2] + len[RDel2] };

        for (int p = 0; p < 4; ++p)
        {
            const float perSecond = -9.9657842846620870f * static_cast<float> (segment[p])
                                    / (static_cast<float> (sampleRate) * decayRt60);
            decay[p].gLow  = spatcore::dsp::FastDecibels::exp2 (perSecond / decayLow);
            decay[p].gMid  = spatcore::dsp::FastDecibels::exp2 (perSecond);
            decay[p].gHigh = spatcore::dsp::FastDecibels::exp2 (perSecond / decayHigh);
        }
    }

    static float allpass (ReverbDelayLine& line, int length, float g, float x) noexcept
    {
        const float delayed = line.readInteger (length);
        const float v = x - g * delayed;
        line.write (v);
        return delayed + g * v;
    }

    static float modulatedAllpass (ReverbDelayLine& line, float length, float g, float x) noexcept
    {
        const float delayed = line.readHermite (length);
        const float v = x - g * delayed;
        line.write (v);
        return delayed + g * v;
    }

    static float delay (ReverbDelayLine& line, int length, float x) noexcept
    {
        const float out = line.readInteger (length);
        line.write (x);
        return out;
    }

    float processSample (float x) noexcept
    {
        x = allpass (inAp[0], inLen[0], inDiff1, x);
        x = allpass (inAp[1], inLen[1], inDiff1, x);
        x = allpass (inAp[2], inLen[2], inDiff2, x);
        x = allpass (inAp[3], inLen[3], inDiff2, x);

        float s = 0.0f, c = 0.0f;
        lfo.nextSinCos (s, c);
        const float e = excursion.next();

        // Each half takes the input and the other half's last output. The
        // tank's first allpasses run with a negative coefficient, as in the
        // paper.
        float l = modulatedAllpass (tank[LAp1], static_cast<float> (len[LAp1]) + e * s, -decDiff1, x + crossR);
        l = decay[0].process (delay (tank[LDel1], len[LDel1], l), cLow, cHigh);
        l = allpass (tank[LAp2], len[LAp2], 0.5f, l);
        l = decay[1].process (delay (tank[LDel2], len[LDel2], l), cLow, cHigh);

        float r = modulatedAllpass (tank[RAp1], static_cast<float> (len[RAp1]) + e * c, -decDiff1, x + crossL);
        r = decay[2].process (delay (tank[RDel1], len[RDel1], r), cLow, cHigh);
        r = allpass (tank[RAp2], len[RAp2], 0.5f, r);
        r = decay[3].process (delay (tank[RDel2], len[RDel2], r), cLow, cHigh);

        crossL = l;
        crossR = r;

        // Taps read what the lines hold after this sample's writes: a tap at
        // t is the line's input t samples ago.
        float acc = 0.0f;
        for (int t = 0; t < 14; ++t)
            acc += tapSign[t] * tank[kTaps[t].line].readInteger (tapAt[t] + 1);

        return kOutputGain * acc;
    }

    ReverbDelayLine inAp[4];
    ReverbDelayLine tank[kTankLines];
    DecayPoint decay[4];
    ReverbLfo lfo;
    spatcore::dsp::OnePoleSmoother excursion;
    spatcore::dsp::DcBlocker dc;

    double sampleRate = 48000.0;
    double rateScale = 48000.0 / kPaperRate;
    float jitter[kTankLines] = {};
    int inLen[4] = {};
    int len[kTankLines] = {};
    int tapAt[14] = {};
    float tapSign[14] = {};

    float inDiff1 = 0.0f, inDiff2 = 0.0f, decDiff1 = 0.5f;
    float decayRt60 = 1.5f, decayLow = 1.3f, decayHigh = 0.4f;
    float crossLow = 200.0f, crossHigh = 4000.0f;
    float cLow = 0.0f, cHigh = 0.0f;
    float crossL = 0.0f, crossR = 0.0f;
    float builtSize = 1.0f;
    float pendingSize = 1.0f;
    bool fresh = true;
};

} // namespace spatcore::effects
