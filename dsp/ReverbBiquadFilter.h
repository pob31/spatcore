#pragma once

#include "BiquadResponse.h"

#include <algorithm>   // std::min / std::max in the parameter clamps
#include <cmath>

namespace spatcore::dsp {

//==============================================================================
/**
    Parametric biquad filter for the reverb pre/post EQ.

    Supports 6 filter shapes matching reverbPreEQshape / reverbPostEQshape:
      0 = OFF (pass-through)
      1 = LowCut (2nd-order high-pass)
      2 = LowShelf
      3 = Peak / Notch
      4 = HighShelf
      5 = HighCut (2nd-order low-pass)
      6 = BandPass (constant 0 dB peak gain) — implemented and accepted by the
          0..6 clamp, though the reverb GUI does not currently offer it

    Note this numbering deliberately differs from OutputEQBiquadFilter, whose
    GUI orders the shapes differently (HighShelf/HighCut sit at 5/6 there, and
    BandPass at 4). Anything mapping a GUI shape ID to a filter must use the
    numbering of the filter it is talking to.

    Uses Audio EQ Cookbook formulas (Robert Bristow-Johnson).
    Designed for per-sample processing in the reverb engine thread.

    Threading: an instance belongs to that one engine thread —
    prepare/setParameters/processSample/processBlock are unsynchronised and
    must all be called from it. The static calculateCoefficients() is the
    exception: it holds no instance state, so a GUI response curve can call it
    from the message thread and get exactly what the audio path is running.
*/
class ReverbBiquadFilter
{
public:
    ReverbBiquadFilter() = default;

    //==========================================================================
    // Lifecycle
    //==========================================================================

    void prepare (double newSampleRate)
    {
        sampleRate = newSampleRate;
        reset();
        recalculate();
    }

    void reset()
    {
        x1 = x2 = y1 = y2 = 0.0;
    }

    //==========================================================================
    // Parameter setters — recalculates only when something changed
    //==========================================================================

    void setParameters (int newShape, float newFreq, float newGainDb,
                        float newQ, float newSlope)
    {
        newFreq = std::max (20.0f, std::min (newFreq, 20000.0f));
        newQ    = std::max (0.1f, std::min (newQ, 20.0f));
        newSlope = std::max (0.1f, std::min (newSlope, 20.0f));
        newGainDb = std::max (-24.0f, std::min (newGainDb, 24.0f));
        newShape = std::max (0, std::min (newShape, 6));

        if (shape != newShape || freq != newFreq || gainDb != newGainDb
            || q != newQ || slope != newSlope)
        {
            shape   = newShape;
            freq    = newFreq;
            gainDb  = newGainDb;
            q       = newQ;
            slope   = newSlope;
            recalculate();
        }
    }

    int getShape() const { return shape; }
    bool isActive() const { return shape != 0; }

    //==========================================================================
    // Coefficient design — shared with the GUI response curve
    //==========================================================================

    /** Designs one band and returns its normalized coefficients.

        Single source of truth for the reverb EQ response: the audio path
        routes through it, and a GUI curve should call it rather than
        re-deriving the cookbook formulas (a second copy inevitably drifts —
        notably on the shelves, where the separate slope "S" parameter is easy
        to drop).

        shape uses THIS class's numbering (0 OFF, 1 LowCut, 2 LowShelf,
        3 Peak, 4 HighShelf, 5 HighCut, 6 BandPass), not the Output EQ's.

        The same clamps setParameters() applies are applied here (freq
        20..20000, q 0.1..20, slope 0.1..20, gainDb -24..24, shape 0..6), so a
        caller passing raw parameter values agrees with the audio path. The
        clamps are idempotent, so routing already-clamped members back through
        is a no-op.

        Returns identity coefficients with active == false for shape 0 (OFF)
        or a non-positive sample rate.

        Pure and allocation-free — safe to call from any thread.
    */
    static BiquadCoefficients calculateCoefficients (int shape, float freqHz, float gainDb,
                                                     float q, float slope, double sampleRate) noexcept
    {
        freqHz = std::max (20.0f,  std::min (freqHz, 20000.0f));
        q      = std::max (0.1f,   std::min (q,      20.0f));
        slope  = std::max (0.1f,   std::min (slope,  20.0f));
        gainDb = std::max (-24.0f, std::min (gainDb, 24.0f));
        shape  = std::max (0, std::min (shape, 6));

        const double freq = freqHz;

        double b0 = 1.0, b1 = 0.0, b2 = 0.0;
        double a1 = 0.0, a2 = 0.0;

        if (sampleRate <= 0.0 || shape == 0)
        {
            b0 = 1.0; b1 = 0.0; b2 = 0.0;
            a1 = 0.0; a2 = 0.0;
            return { b0, b1, b2, a1, a2, false };
        }

        constexpr double pi = 3.14159265358979323846;
        double w0 = 2.0 * pi * freq / sampleRate;
        double cosw0 = std::cos (w0);
        double sinw0 = std::sin (w0);

        double a0_inv = 1.0;

        switch (shape)
        {
            case 1: // LowCut (high-pass)
            {
                double alpha = sinw0 / (2.0 * q);
                a0_inv = 1.0 / (1.0 + alpha);
                b0 = ((1.0 + cosw0) / 2.0) * a0_inv;
                b1 = -(1.0 + cosw0) * a0_inv;
                b2 = ((1.0 + cosw0) / 2.0) * a0_inv;
                a1 = (-2.0 * cosw0) * a0_inv;
                a2 = (1.0 - alpha) * a0_inv;
                break;
            }

            case 2: // LowShelf
            {
                double A = std::pow (10.0, gainDb / 40.0);
                double alpha = (sinw0 / 2.0) * std::sqrt (
                    (A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0);
                double sqrtA2alpha = 2.0 * std::sqrt (A) * alpha;

                a0_inv = 1.0 / ((A + 1.0) + (A - 1.0) * cosw0 + sqrtA2alpha);
                b0 = A * ((A + 1.0) - (A - 1.0) * cosw0 + sqrtA2alpha) * a0_inv;
                b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cosw0) * a0_inv;
                b2 = A * ((A + 1.0) - (A - 1.0) * cosw0 - sqrtA2alpha) * a0_inv;
                a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cosw0) * a0_inv;
                a2 = ((A + 1.0) + (A - 1.0) * cosw0 - sqrtA2alpha) * a0_inv;
                break;
            }

            case 3: // Peak / Notch
            {
                double A = std::pow (10.0, gainDb / 40.0);
                double alpha = sinw0 / (2.0 * q);

                a0_inv = 1.0 / (1.0 + alpha / A);
                b0 = (1.0 + alpha * A) * a0_inv;
                b1 = (-2.0 * cosw0) * a0_inv;
                b2 = (1.0 - alpha * A) * a0_inv;
                a1 = (-2.0 * cosw0) * a0_inv;
                a2 = (1.0 - alpha / A) * a0_inv;
                break;
            }

            case 4: // HighShelf
            {
                double A = std::pow (10.0, gainDb / 40.0);
                double alpha = (sinw0 / 2.0) * std::sqrt (
                    (A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0);
                double sqrtA2alpha = 2.0 * std::sqrt (A) * alpha;

                a0_inv = 1.0 / ((A + 1.0) - (A - 1.0) * cosw0 + sqrtA2alpha);
                b0 = A * ((A + 1.0) + (A - 1.0) * cosw0 + sqrtA2alpha) * a0_inv;
                b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cosw0) * a0_inv;
                b2 = A * ((A + 1.0) + (A - 1.0) * cosw0 - sqrtA2alpha) * a0_inv;
                a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cosw0) * a0_inv;
                a2 = ((A + 1.0) - (A - 1.0) * cosw0 - sqrtA2alpha) * a0_inv;
                break;
            }

            case 5: // HighCut (low-pass)
            {
                double alpha = sinw0 / (2.0 * q);
                a0_inv = 1.0 / (1.0 + alpha);
                b0 = ((1.0 - cosw0) / 2.0) * a0_inv;
                b1 = (1.0 - cosw0) * a0_inv;
                b2 = ((1.0 - cosw0) / 2.0) * a0_inv;
                a1 = (-2.0 * cosw0) * a0_inv;
                a2 = (1.0 - alpha) * a0_inv;
                break;
            }

            case 6: // BandPass
            {
                double alpha = sinw0 / (2.0 * q);
                a0_inv = 1.0 / (1.0 + alpha);
                b0 = alpha * a0_inv;
                b1 = 0.0;
                b2 = -alpha * a0_inv;
                a1 = (-2.0 * cosw0) * a0_inv;
                a2 = (1.0 - alpha) * a0_inv;
                break;
            }

            default:
                b0 = 1.0; b1 = 0.0; b2 = 0.0;
                a1 = 0.0; a2 = 0.0;
                break;
        }

        return { b0, b1, b2, a1, a2, true };
    }

    //==========================================================================
    // Processing
    //==========================================================================

    float processSample (float input)
    {
        if (shape == 0)
            return input;

        // Double state and coefficients (see BiquadCoefficients); float in and out.
        const double in = input;
        const double output = b0 * in + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = in;
        y2 = y1;
        y1 = output;
        return static_cast<float> (output);
    }

    void processBlock (float* samples, int numSamples)
    {
        if (shape == 0)
            return;

        for (int i = 0; i < numSamples; ++i)
            samples[i] = processSample (samples[i]);
    }

private:
    //==========================================================================
    void recalculate()
    {
        const auto c = calculateCoefficients (shape, freq, gainDb, q, slope, sampleRate);
        b0 = c.b0; b1 = c.b1; b2 = c.b2; a1 = c.a1; a2 = c.a2;
    }

    //==========================================================================
    // State
    //==========================================================================

    int shape = 0;          // 0=OFF, 1=LowCut, 2=LowShelf, 3=Peak, 4=HighShelf, 5=HighCut
    float freq = 1000.0f;
    float gainDb = 0.0f;
    float q = 0.7f;
    float slope = 0.7f;
    double sampleRate = 48000.0;

    // Biquad coefficients (normalized, a0 = 1)
    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a1 = 0.0, a2 = 0.0;

    // Delay elements
    double x1 = 0.0, x2 = 0.0;
    double y1 = 0.0, y2 = 0.0;
};

} // namespace spatcore::dsp

// Extraction-compat aliases — app code migrates to qualified names later.
using spatcore::dsp::ReverbBiquadFilter;
