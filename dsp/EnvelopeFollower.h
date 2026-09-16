#pragma once

#include "FastDecibels.h"
#include <cmath>
#include <cstdint>

namespace spatcore::dsp
{

/**
    Peak or mean-square envelope with separate attack and release.

    LiveSourceLevelDetector already follows a level, but with an instant attack
    baked in and a fixed pair of release times, because that is what feedback
    taming needs. A compressor needs the attack to be a user parameter (it is
    most of the character), and a gate needs a different release from the
    compressor above it, so this generalises rather than copies: attack 0 ms
    reproduces the instant-attack detector exactly, and with a = 0 the release
    branch is exactly the multiplicative decay that detector runs.

    The coefficient law is the shared one, coef = 1 - exp(-1 / (sr * t)), so a
    time is a time constant, not a completion time.

    Peak mode follows |x|; Rms mode follows x*x, so getValue() is the mean
    square and getRms() its root. getDb() accounts for the difference. One
    instance, one audio thread. No allocation.
*/
class EnvelopeFollower
{
public:
    enum class Mode : std::uint8_t { Peak = 0, Rms = 1 };

    EnvelopeFollower() = default;

    void prepare (double newSampleRate) noexcept
    {
        sampleRate = newSampleRate > 0.0 ? newSampleRate : 48000.0;
        setTimes (attackMs, releaseMs);
        reset();
    }

    void reset() noexcept { env = 0.0f; }

    void setMode (Mode m) noexcept { mode = m; }
    Mode getMode() const noexcept  { return mode; }

    /** 0 ms means instant: the envelope takes the new value in one sample. */
    void setTimes (float newAttackMs, float newReleaseMs) noexcept
    {
        attackMs = newAttackMs;
        releaseMs = newReleaseMs;
        attackCoef = coefficientFor (newAttackMs);
        releaseCoef = coefficientFor (newReleaseMs);
    }

    float getAttackCoefficient() const noexcept  { return attackCoef; }
    float getReleaseCoefficient() const noexcept { return releaseCoef; }

    float processSample (float x) noexcept
    {
        const float a = (mode == Mode::Peak) ? std::fabs (x) : x * x;
        env += (a > env ? attackCoef : releaseCoef) * (a - env);
        return env;
    }

    /** Runs the whole block and returns the envelope at the end of it. */
    float processBlock (const float* samples, int numSamples) noexcept
    {
        if (samples == nullptr || numSamples <= 0)
            return env;

        for (int i = 0; i < numSamples; ++i)
            processSample (samples[i]);

        return env;
    }

    /** Peak mode: the envelope. Rms mode: the MEAN SQUARE. */
    float getValue() const noexcept { return env; }

    /** The envelope as an amplitude in both modes. */
    float getRms() const noexcept
    {
        return (mode == Mode::Rms) ? std::sqrt (env) : env;
    }

    float getDb() const noexcept
    {
        return (mode == Mode::Rms) ? 0.5f * FastDecibels::gainToDb (env)
                                   : FastDecibels::gainToDb (env);
    }

private:
    float coefficientFor (float timeMs) const noexcept
    {
        if (! (timeMs > 0.0f) || sampleRate <= 0.0)
            return 1.0f;                                  // instant

        const float samples = static_cast<float> (sampleRate) * timeMs * 0.001f;
        return samples > 0.0f ? 1.0f - std::exp (-1.0f / samples) : 1.0f;
    }

    double sampleRate = 48000.0;
    Mode mode = Mode::Peak;
    float attackMs = 0.0f;
    float releaseMs = 100.0f;
    float attackCoef = 1.0f;
    float releaseCoef = 1.0f;
    float env = 0.0f;
};

} // namespace spatcore::dsp
