#pragma once

#include <cmath>

namespace spatcore::dsp
{

/**
    Static waveshaping curves for the distortion module.

    The user's gen~ prototype blends two shapers with one continuous control:
    a hard clip at +-0.8 at one end, a tanh at the other, with a bias that makes
    the curve asymmetric and so adds the even harmonics people hear as "tube"
    rather than "fuzz". These are those curves, stateless, so a module or a
    response plot can call them from anywhere.

    blend() takes the tanh of the bias as an argument rather than computing it:
    it is constant for a whole block, and std::tanh is far too expensive to call
    twice per sample. Subtracting it is what keeps the curve through the origin,
    so a bias adds harmonics without adding DC.

    A note on reproducibility: softTanh and blend call std::tanh, which is libm
    and therefore varies by an ULP or two across platforms. Anything hashing a
    distorted render must compare within a tolerance, never bit-for-bit across
    machines. The clip path (shape = 0) has no such problem.
*/
namespace Waveshaper
{

/** Symmetric hard clip. The default limit is the prototype's. */
inline float hardClip (float x, float limit = 0.8f) noexcept
{
    if (x > limit)
        return limit;
    if (x < -limit)
        return -limit;
    return x;
}

/** Smooth saturation. */
inline float softTanh (float x) noexcept
{
    return std::tanh (x);
}

/** Crossfade between the two curves, with an asymmetry term.

    @param shape     0 = hard clip, 1 = tanh, anything between = a mix
    @param bias      shifts the operating point into the asymmetric part
    @param tanhBias  std::tanh(bias), precomputed once per block; it is
                     subtracted so the curve still passes through the origin
*/
inline float blend (float x, float shape, float bias, float tanhBias) noexcept
{
    const float clipped = hardClip (x);
    const float soft = std::tanh (x + bias) - tanhBias;
    return clipped + shape * (soft - clipped);
}

} // namespace Waveshaper

} // namespace spatcore::dsp
