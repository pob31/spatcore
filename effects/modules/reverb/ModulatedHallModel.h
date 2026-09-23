#pragma once

#include "ReverbDelayLine.h"
#include "ReverbLfo.h"
#include "ReverbTailModel.h"
#include "ShimmerTap.h"
#include "../../EffectPresets.h"
#include "../../../dsp/DcBlocker.h"
#include "../../../dsp/FastDecibels.h"
#include "../../../dsp/FrDiffusionModel.h"
#include "../../../dsp/OnePoleSmoother.h"
#include <cmath>
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

    Model 5, Shimmer, is this hall with its four longest lines shimmering.
    Each reads itself as usual and through a pitch-shifting tap (ShimmerTap),
    mixed at EQUAL POWER - sqrt(1 - a) of the one, sqrt(a) of the other,
    trimmed, for a = ShimmerAmount - so round the loop the tail climbs by the
    interval every pass and fades as it climbs. The two-voice intervals split
    the four lines two and two.

    Equal power, not a plain (1 - a, a) crossfade, and for the sustain. The
    two reads are at different pitches, so they are uncorrelated and their
    POWERS add: a plain crossfade keeps (1 - a)^2 + a^2 of a line's power per
    pass - half of it at a = 0.5 - which drained a 5 s shimmer in 1.7 s. At
    equal power the line keeps (1 - a) + a trim^2. It is also why the loop
    stays put: the shifter never gives out more power than it reads (a convex
    pair of heads over a stationary signal), the two reads only correlate at
    DC, which the high pass below removes, so no line gains power on a pass
    and the decay gains are all under one. What else keeps it tame:
    - the trims - -2.0 dB an octave up, -4.5 two octaves up, -2.2 an octave
      down, -0.5 otherwise - make the higher voices fade faster than the
      tail, as a shimmer should;
    - two one-poles on those lines' writes, at min(12 kHz, sr / 4 ratio),
      keep a raised voice from folding past Nyquist, and a one-pole high pass
      at 80 Hz on the shifted read stops a lowered one piling up rumble;
    - past |8| a soft clip, as a last resort only.
    The shimmer (on or off) and the interval are build-time, so switching
    them spills over rather than jumping a sweeping head; the amount is
    runtime. Off, the lines run the exact model-4 arithmetic: a hall is the
    same hall whatever the shimmer settings say.

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

    /** The shimmer lines: the four longest. */
    static constexpr int kShimmerLines = 4;
    static constexpr int kFirstShimmerLine = kLines - kShimmerLines;

    /** A shimmer sweep's window: at most this, and at most 0.8 of the line. */
    static constexpr float kMaxWindowMs = 50.0f;

    /** The shifter's two triangle-faded heads carry 2/3 of the line's power
        on average (their weights' squares average 2/3 over a sweep); this
        makes the shifted read power-neutral. On the rare frequency whose two
        heads land in phase it lifts a single pass by 1.76 dB - which is what
        the trims answer for: an octave chain (the one that stays in phase
        pass after pass) still loses 0.2 dB a pass at -2 dB, and every other
        interval drifts out of phase on the next. */
    static constexpr float kShifterMakeup = 1.2247448713915890f;       // sqrt (3/2)

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

        const int halfWindow = static_cast<int> (0.5 * kMaxWindowMs * 0.001 * sampleRate) + 2;

        for (int i = 0; i < kLines; ++i)
        {
            jitter[i] = 1.0f + kLineJitter * hash (i);
            lines[i].prepare (lengthOf (kLineBase[i] * jitter[i], kReverbMaxSize) + maxExcursion + 4
                              + (i >= kFirstShimmerLine ? halfWindow : 0));
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
        amount.setTimeConstant (sampleRate, 0.02f);
        dc.prepare (sampleRate, 5.0f);
        hpCoef = onePole (80.0f);

        for (int s = 0; s < kShimmerLines; ++s)
            taps[s].setStartPhase (0.25 * s);

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
        for (int s = 0; s < kShimmerLines; ++s)
        {
            taps[s].reset();
            lp1[s] = lp2[s] = hpState[s] = 0.0f;
        }

        depth.snap (depth.getTarget());
        amount.snap (amount.getTarget());
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

        amount.setTarget (clampParam (params.shimmerAmount, 0.0f, 100.0f) * 0.01f);

        // An instance that has not run since it was cleared starts at its
        // values rather than gliding in from the last ones it was given.
        if (fresh)
        {
            depth.snap (depth.getTarget());
            amount.snap (amount.getTarget());
        }

        pendingSize = quantiseReverbSize (clampParam (params.size, kReverbMinSize, kReverbMaxSize));
        pendingShimmer = shimmerFor (params);
        return isPending();
    }

    void commitPendingVariant() noexcept override
    {
        if (isPending())
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
        return quantiseReverbSize (clampParam (params.size, kReverbMinSize, kReverbMaxSize)) != builtSize
            || shimmerFor (params) != builtShimmer;
    }

    float getBuiltSize() const noexcept override { return builtSize; }

    /** Line i's length at the built size, in samples - for tests. */
    int getLineLength (int i) const noexcept { return (i >= 0 && i < kLines) ? len[i] : 0; }

    /** The shimmer the lines are built with: -1 off, else the ShimmerInterval. */
    int getBuiltShimmer() const noexcept { return builtShimmer; }

    /** The pitch ratio of shimmer line s (0..3) for an interval. */
    static double shimmerRatio (int interval, int s) noexcept
    {
        constexpr double fifth = 1.4983070768766815;       // 2^(7/12)
        constexpr double fourth = 1.3348398541700344;      // 2^(5/12)
        constexpr double twelfth = 2.9966141537533632;     // 2^(19/12)
        const bool firstPair = s < 2;

        switch (interval)
        {
            case static_cast<int> (ShimmerInterval::FifthUp):         return fifth;
            case static_cast<int> (ShimmerInterval::FifthAndOctave):  return firstPair ? fifth : 2.0;
            case static_cast<int> (ShimmerInterval::Twelfth):         return twelfth;
            case static_cast<int> (ShimmerInterval::TwoOctaves):      return 4.0;
            case static_cast<int> (ShimmerInterval::FourthUp):        return fourth;
            case static_cast<int> (ShimmerInterval::OctaveDown):      return 0.5;
            case static_cast<int> (ShimmerInterval::OctaveDownAndUp): return firstPair ? 0.5 : 2.0;
            default:                                                  return 2.0;      // an octave up
        }
    }

    /** Past |8|, bend towards 16 - a last resort that a sane loop never meets.
        The identity below it, continuous and monotonic through it. */
    static float softClip (float x) noexcept
    {
        const float m = x < 0.0f ? -x : x;
        if (m <= 8.0f)
            return x;

        const float t = m - 8.0f;
        const float y = 8.0f + 8.0f * t / (8.0f + t);
        return x < 0.0f ? -y : y;
    }

    /** The shifted read's trim for a ratio, in dB. */
    static float shimmerTrimDb (double ratio) noexcept
    {
        if (ratio == 2.0) return -2.0f;
        if (ratio == 4.0) return -4.5f;
        if (ratio == 0.5) return -2.2f;
        return -0.5f;
    }

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

    /** -1 for no shimmer (every model but 5), else the interval. */
    static int shimmerFor (const ReverbParams& params) noexcept
    {
        if (resolveReverbModel (params.model) != static_cast<int> (ReverbModel::Shimmer))
            return -1;

        return params.shimmerPitch < static_cast<std::uint8_t> (ShimmerInterval::Count) ? params.shimmerPitch : 0;
    }

    bool isPending() const noexcept
    {
        return pendingSize != builtSize || pendingShimmer != builtShimmer;
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
        builtShimmer = pendingShimmer;

        if (builtShimmer >= 0)
        {
            const float maxWindow = static_cast<float> (kMaxWindowMs * 0.001 * sampleRate);
            for (int s = 0; s < kShimmerLines; ++s)
            {
                const double ratio = shimmerRatio (builtShimmer, s);
                const float lineWindow = 0.8f * static_cast<float> (len[kFirstShimmerLine + s]);
                taps[s].configure (ratio, lineWindow < maxWindow ? lineWindow : maxWindow);
                trim[s] = kShifterMakeup * spatcore::dsp::FastDecibels::dbToGain (shimmerTrimDb (ratio));

                const float guard = static_cast<float> (0.25 * sampleRate / ratio);
                lpCoef[s] = onePole (guard < 12000.0f ? guard : 12000.0f);
            }
        }

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

        if (builtShimmer < 0)
        {
            for (int i = 0; i < kLines; ++i)
            {
                const float offset = mod[i & 7] * (i < 8 ? e : -e);
                y[i] = decay[i].process (lines[i].readHermite (static_cast<float> (len[i]) + offset), cLow, cHigh);
            }
        }
        else
        {
            const float a = amount.next();
            const float keep = std::sqrt (1.0f - a);        // correctly rounded everywhere: no libm drift
            const float give = std::sqrt (a);

            for (int i = 0; i < kLines; ++i)
            {
                const float offset = mod[i & 7] * (i < 8 ? e : -e);
                float v = lines[i].readHermite (static_cast<float> (len[i]) + offset);

                if (i >= kFirstShimmerLine)
                {
                    const int s = i - kFirstShimmerLine;
                    float shifted = taps[s].read (lines[i], static_cast<float> (len[i]));
                    hpState[s] += hpCoef * (shifted - hpState[s]);
                    shifted -= hpState[s];
                    v = softClip (keep * v + give * trim[s] * shifted);
                }

                y[i] = decay[i].process (v, cLow, cHigh);
            }
        }

        hadamard (y);

        float out = 0.0f;
        for (int i = 0; i < kLines; ++i)
        {
            out += outSign[i] * y[i];
            float w = y[i] + inSign[i] * kInputGain * x;

            if (builtShimmer >= 0 && i >= kFirstShimmerLine)
            {
                const int s = i - kFirstShimmerLine;
                lp1[s] += lpCoef[s] * (w - lp1[s]);
                lp2[s] += lpCoef[s] * (lp1[s] - lp2[s]);
                w = lp2[s];
            }

            lines[i].write (w);
        }

        return kOutputGain * out;
    }

    ReverbDelayLine lines[kLines];
    ReverbDelayLine diffusers[4];
    DecayPoint decay[kLines];
    ReverbLfo lfo[4];
    ShimmerTap taps[kShimmerLines];
    spatcore::dsp::OnePoleSmoother depth, amount;
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
    int builtShimmer = -1;
    int pendingShimmer = -1;

    float trim[kShimmerLines] = {};
    float lpCoef[kShimmerLines] = {};
    float lp1[kShimmerLines] = {}, lp2[kShimmerLines] = {}, hpState[kShimmerLines] = {};
    float hpCoef = 0.0f;
    bool fresh = true;
};

} // namespace spatcore::effects
