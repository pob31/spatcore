#pragma once

#include <cstdint>
#include <cstring>

namespace spatcore::dsp
{

/**
    dB <-> linear without libm.

    An effects chain converts decibels per sample (tremolo depth, dynamics gain
    computers, crusher step sizes), and std::pow / std::log10 are both slow and,
    more importantly, not reproducible: their results differ by an ULP or two
    between platforms and libc versions, so an offline render hashed on one
    machine would not match another. These routines are an exponent/mantissa
    split plus a fixed-coefficient polynomial - the same sequence of +, - and *
    everywhere - so a render stays reproducible by construction.

    Accuracy over the range that matters (-120..+24 dB): better than 1e-6
    relative on gains, better than 1e-4 dB coming back. Some values are exact
    rather than approximate, and callers may rely on them: exp2(k) == 2^k for
    every integer k in range, dbToGain(0) == 1.0f exactly (so a depth of 0 is
    bit-transparent), and log2(2^k) == k.

    Out-of-range input saturates rather than misbehaving: a gain at or below
    2^-125 reads as kMinDb, and so does anything negative or NaN. Every function
    is pure, allocation-free and safe on any thread.

    Note for future edits: the polynomials are written to be evaluated exactly
    as they read. Do not "simplify" them into a form a compiler may contract
    differently (a fused multiply-add on one target and not another), or the
    reproducibility this header exists for is gone.
*/
namespace FastDecibels
{

/** Floor for gainToDb: what silence, negative input and NaN all return. */
inline constexpr float kMinDb = -200.0f;

/** Smallest normal float; below it the exponent trick stops working. */
inline constexpr float kSmallestNormal = 1.17549435e-38f;

/** 2^x. Flushes to 0 below 2^-125, saturates at 2^127; NaN reads as 0. */
inline float exp2 (float x) noexcept
{
    // Negated comparison so NaN takes this branch too.
    if (! (x > -125.0f))
        return 0.0f;
    if (x > 127.0f)
        x = 127.0f;

    // Split x into an integer part (free: it is the exponent field) and a
    // fraction in [-0.5, 0.5]. Rounding is half-away-from-zero, done in float
    // so there is no library call.
    const int i = static_cast<int> (x >= 0.0f ? x + 0.5f : x - 0.5f);
    const float f = x - static_cast<float> (i);

    // 2^f = e^(f*ln2); |y| <= 0.3466, so a Taylor series through y^7/7! leaves
    // a truncation error near 5e-9 - an order below float rounding.
    const float y = f * 0.6931471805599453f;
    float p = 1.0f + y * (1.0f / 7.0f);
    p = 1.0f + y * p * (1.0f / 6.0f);
    p = 1.0f + y * p * (1.0f / 5.0f);
    p = 1.0f + y * p * (1.0f / 4.0f);
    p = 1.0f + y * p * (1.0f / 3.0f);
    p = 1.0f + y * p * (1.0f / 2.0f);
    p = 1.0f + y * p;

    // Scale by 2^i through the exponent field. p is in [0.707, 1.415] and i is
    // in [-125, 127], so the result is always a normal float.
    std::uint32_t bits;
    std::memcpy (&bits, &p, sizeof (bits));
    bits += static_cast<std::uint32_t> (i) << 23;
    std::memcpy (&p, &bits, sizeof (p));
    return p;
}

/** log2(x). Zero, denormal, negative and NaN all read as -126. */
inline float log2 (float x) noexcept
{
    // Negated so NaN lands here as well.
    if (! (x >= kSmallestNormal))
        return -126.0f;

    std::uint32_t bits;
    std::memcpy (&bits, &x, sizeof (bits));

    int e = static_cast<int> ((bits >> 23) & 0xFFu) - 127;

    const std::uint32_t mantissaBits = (bits & 0x007FFFFFu) | (127u << 23);
    float m;
    std::memcpy (&m, &mantissaBits, sizeof (m));     // m in [1, 2)

    // Centre the mantissa on 1 so the series converges fast.
    if (m > 1.41421356f)
    {
        m *= 0.5f;
        ++e;
    }

    // ln(m) = 2*(t + t^3/3 + t^5/5 + ...) with t = (m-1)/(m+1), |t| <= 0.1716;
    // through t^9 the truncation error is around 1e-9 in log2 units.
    const float t = (m - 1.0f) / (m + 1.0f);
    const float t2 = t * t;
    float p = 1.0f / 9.0f;
    p = 1.0f / 7.0f + t2 * p;
    p = 1.0f / 5.0f + t2 * p;
    p = 1.0f / 3.0f + t2 * p;
    p = 1.0f + t2 * p;

    return static_cast<float> (e) + t * p * 2.8853900817779268f;   // 2 / ln 2
}

/** Decibels to a linear gain. dbToGain(0) is exactly 1. */
inline float dbToGain (float dB) noexcept
{
    return exp2 (dB * 0.16609640474436813f);          // log2(10) / 20
}

/** Linear gain to decibels, floored at kMinDb. gainToDb(1) is exactly 0. */
inline float gainToDb (float gain) noexcept
{
    if (! (gain >= kSmallestNormal))
        return kMinDb;

    const float dB = log2 (gain) * 6.020599913279624f;   // 20 * log10(2)
    return dB > kMinDb ? dB : kMinDb;
}

} // namespace FastDecibels

} // namespace spatcore::dsp
