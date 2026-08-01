#pragma once

#include <cmath>
#include <complex>

namespace spatcore::dsp {

//==============================================================================
/**
    Normalized biquad coefficients, as produced by the spatcore filter classes.

    "Normalized" means a0 has already been folded into the other five
    coefficients (a0 == 1), which is exactly the form the audio-thread
    difference equation uses:

        y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]

    active == false means the band is OFF: the coefficients are the identity
    pass-through (b0 = 1, everything else 0) and the response is flat 0 dB.
    A default-constructed BiquadCoefficients is exactly that OFF state.

    This is a plain aggregate with no threading requirements of its own —
    it is copied by value from the audio-side filters to the GUI-side curve
    drawing, so a torn read is impossible.
*/
struct BiquadCoefficients
{
    float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    bool active = false;
};

//==============================================================================
/**
    Magnitude of a normalized biquad at one frequency, in dB.

    Evaluates |H(e^jw)| on the unit circle:

        H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)

    This is *display* math, not audio math, so the evaluation is done in
    double precision throughout even though the coefficients themselves are
    the float values the audio path actually runs — precision beats speed
    here, and a smooth curve matters more than a cycle or two.

    Behaviour at the edges:
      - returns 0.0f (flat) when the band is not active, or when sampleRate
        is not a positive, finite number;
      - freqHz is clamped into the valid open range (0, Nyquist);
      - the result is floored at -120 dB, which also covers a zero-magnitude
        response (log of zero) and a degenerate/unstable denominator;
      - a non-finite intermediate result yields -120 dB rather than NaN, so
        callers can feed the value straight into a path/plot without further
        guarding.

    Thread-safe and allocation-free: a pure function of its arguments, safe
    to call from the message thread, a paint() callback, or anywhere else.
*/
inline float biquadMagnitudeDb (const BiquadCoefficients& c, float freqHz, double sampleRate) noexcept
{
    if (! c.active)
        return 0.0f;

    // The negated form also rejects NaN.
    if (! (sampleRate > 0.0) || ! std::isfinite (sampleRate))
        return 0.0f;

    constexpr double pi = 3.14159265358979323846;
    constexpr double minHz = 1.0e-6;

    const double nyquist = 0.5 * sampleRate;
    const double maxHz   = nyquist > minHz ? std::nextafter (nyquist, 0.0) : minHz;

    double f = static_cast<double> (freqHz);

    if (! (f > minHz))      // catches NaN, negatives and DC
        f = minHz;
    else if (f > maxHz)
        f = maxHz;

    const double w = 2.0 * pi * f / sampleRate;

    const std::complex<double> z1 = std::polar (1.0, -w);
    const std::complex<double> z2 = z1 * z1;

    const std::complex<double> num = static_cast<double> (c.b0)
                                   + static_cast<double> (c.b1) * z1
                                   + static_cast<double> (c.b2) * z2;
    const std::complex<double> den = 1.0
                                   + static_cast<double> (c.a1) * z1
                                   + static_cast<double> (c.a2) * z2;

    const double denMag = std::abs (den);

    if (! (denMag > 0.0) || ! std::isfinite (denMag))
        return -120.0f;

    const double mag = std::abs (num / den);

    if (! std::isfinite (mag) || mag <= 1.0e-6)
        return -120.0f;

    const double db = 20.0 * std::log10 (mag);

    if (! std::isfinite (db) || db < -120.0)
        return -120.0f;

    return static_cast<float> (db);
}

} // namespace spatcore::dsp

// Extraction-compat aliases — app code migrates to qualified names later.
using spatcore::dsp::BiquadCoefficients;
using spatcore::dsp::biquadMagnitudeDb;
