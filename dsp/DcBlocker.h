#pragma once

#include <cmath>

namespace spatcore::dsp
{

/**
    First-order DC blocker: y[n] = x[n] - x[n-1] + R*y[n-1].

    Asymmetric waveshaping (a distortion bias, a rectifier) leaves a DC offset
    that eats headroom and thumps when the module is bypassed; a feedback loop
    lets it accumulate. R is set from a cutoff of a few hertz, well below
    anything musical, so the audible effect is nothing and the DC is gone.

    processBlock flushes a decayed tail to exact zero at the end of the block.
    That is not cosmetic: the recursion approaches zero geometrically and parks
    in denormal territory, where x86 runs 10-100x slower - so a SILENT effect
    channel would cost more CPU than a loud one. The effects threads get FTZ
    later; this costs one comparison per block and works regardless.

    One instance, one audio thread. No allocation.
*/
class DcBlocker
{
public:
    DcBlocker() = default;

    void prepare (double sampleRate, float cutoffHz = 5.0f) noexcept
    {
        if (sampleRate > 0.0 && cutoffHz > 0.0f)
        {
            const float r = 1.0f - (6.2831853071795864f * cutoffHz / static_cast<float> (sampleRate));
            R = r > 0.0f ? (r < 0.9999999f ? r : 0.9999999f) : 0.0f;
        }
        else
        {
            R = 0.995f;
        }

        reset();
    }

    void reset() noexcept
    {
        x1 = 0.0f;
        y1 = 0.0f;
    }

    float getCoefficient() const noexcept { return R; }

    float processSample (float x) noexcept
    {
        const float y = x - x1 + R * y1;
        x1 = x;
        y1 = y;
        return y;
    }

    void processBlock (float* samples, int numSamples) noexcept
    {
        if (samples == nullptr || numSamples <= 0)
            return;

        for (int i = 0; i < numSamples; ++i)
            samples[i] = processSample (samples[i]);

        // Park a decayed tail at true zero rather than in denormal land.
        if (std::fabs (y1) < 1.0e-15f)
            y1 = 0.0f;
        if (std::fabs (x1) < 1.0e-15f)
            x1 = 0.0f;
    }

private:
    float R = 0.995f;
    float x1 = 0.0f;
    float y1 = 0.0f;
};

} // namespace spatcore::dsp
