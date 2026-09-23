#pragma once

namespace spatcore::effects
{

/**
    Sine LFO for the reverb models' delay modulation: a double-precision phase
    and a libm-free sine, so a render hashes the same on every platform.

    dsp::LfoPhasor already accumulates in double, but its waveforms are a
    shape table for the tremolo / phaser family and its sine goes through
    std::sin. A tank modulator needs only a sine and a cosine, needs them per
    sample on up to eight phasors, and must not wander between libm builds -
    the render baselines are hashes.

    The sine is folded to a quarter wave and evaluated with the Taylor series
    to the ninth power, whose remainder on [0, pi/2] is below 4e-6: far under
    anything a delay modulation of a millisecond can express, and exact to the
    same bits wherever the arithmetic is IEEE single precision.

    The phase accumulates in DOUBLE for the reason LfoPhasor gives: at 0.05 Hz
    and 192 kHz the increment is 2.6e-10 of a cycle, which a float accumulator
    would round away.

    No allocation, no locks; one instance belongs to one audio thread.
*/
class ReverbLfo
{
public:
    /** sin(2 pi p) for any finite p, libm-free. */
    static float sin2pi (double phase01) noexcept
    {
        double p = phase01 - static_cast<double> (static_cast<long long> (phase01));
        if (p < 0.0)
            p += 1.0;

        // y in [0, 2): sin(pi y). Past 1 the wave is the first half negated,
        // and each half is symmetric about its middle.
        float y = static_cast<float> (2.0 * p);
        float sign = 1.0f;
        if (y >= 1.0f)
        {
            y -= 1.0f;
            sign = -1.0f;
        }
        if (y > 0.5f)
            y = 1.0f - y;

        const float w = 3.14159265358979323846f * y;    // [0, pi/2]
        const float w2 = w * w;
        const float s = w * (1.0f + w2 * (-1.0f / 6.0f
                              + w2 * (1.0f / 120.0f
                              + w2 * (-1.0f / 5040.0f
                              + w2 * (1.0f / 362880.0f)))));
        return sign * s;
    }

    void prepare (double newSampleRate) noexcept
    {
        sampleRate = newSampleRate > 0.0 ? newSampleRate : 48000.0;
        setRateHz (rateHz);
        reset();
    }

    /** Clamped to [0, 20] Hz: a tank modulator above that is a vibrato, and a
        negative rate would run the phase backwards through the wrap. */
    void setRateHz (float hz) noexcept
    {
        rateHz = hz;
        double r = (hz > 0.0f) ? static_cast<double> (hz) : 0.0;
        if (r > 20.0)
            r = 20.0;
        increment = r / sampleRate;
    }

    /** The phase reset() returns to, in cycles (a per-instance offset). */
    void setStartPhase (double phase01) noexcept
    {
        double p = phase01 - static_cast<double> (static_cast<long long> (phase01));
        startPhase = p < 0.0 ? p + 1.0 : p;
    }

    void reset() noexcept              { phase = startPhase; }

    double getPhase() const noexcept   { return phase; }

    /** sin at the current phase, then advance one sample. */
    float nextSin() noexcept
    {
        const float s = sin2pi (phase);
        advance();
        return s;
    }

    /** sin and cos at the current phase, then advance one sample. */
    void nextSinCos (float& s, float& c) noexcept
    {
        s = sin2pi (phase);
        c = sin2pi (phase + 0.25);
        advance();
    }

private:
    void advance() noexcept
    {
        phase += increment;
        if (phase >= 1.0)
            phase -= 1.0;
    }

    double sampleRate = 48000.0;
    double phase = 0.0;
    double startPhase = 0.0;
    double increment = 0.0;
    float rateHz = 0.0f;
};

} // namespace spatcore::effects
