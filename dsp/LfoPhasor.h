#pragma once

#include "LFOWaveforms.h"
#include "FrDiffusionModel.h"
#include <cstdint>

namespace spatcore::dsp
{

/**
    Audio-rate phase accumulator driving the shared LFO waveforms.

    LFOWaveforms is a pure function of a 0..1 ramp; this is the ramp, plus the
    two random endpoints shape 8 interpolates between. The position modulators
    (LFOProcessor, ClusterLFOProcessor) run at 50 Hz and keep their own ramp;
    an effects module needs one per sample, and needs it to be deterministic,
    which is what this adds.

    Two details that matter:

    - The phase accumulates in DOUBLE. At 0.05 Hz and 192 kHz the increment is
      2.6e-9, small enough that a float accumulator would lose low bits every
      sample and drift audibly over a long show.

    - Random targets come from FrDiffusion::hashNoiseBipolar, keyed per
      instance, never from a shared RNG. Two phasors given the same key produce
      the same stream whatever order they run in, so a render is reproducible
      and a worker-count change cannot alter the audio.

    One instance belongs to one audio thread. No allocation, no locks.
*/
class LfoPhasor
{
public:
    LfoPhasor() = default;

    /** Sets the rate reference and resets. */
    void prepare (double newSampleRate) noexcept
    {
        sampleRate = newSampleRate > 0.0 ? newSampleRate : 48000.0;
        setRateHz (rateHz);
        reset();
    }

    /** Back to phase 0 with a fresh pair of random endpoints. */
    void reset() noexcept
    {
        phase = 0.0;
        randomIndex = 1;
        lastRandom = FrDiffusion::hashNoiseBipolar (0, noiseKey);
        targetRandom = FrDiffusion::hashNoiseBipolar (1, noiseKey);
    }

    /** Rate in Hz, clamped to [0, sr] so the increment can never exceed one
        cycle per sample (which would alias the wrap logic). */
    void setRateHz (float hz) noexcept
    {
        rateHz = hz;
        const double clamped = hz > 0.0f ? static_cast<double> (hz) : 0.0;
        increment = (clamped < sampleRate ? clamped : sampleRate) / sampleRate;
    }

    /** Stream id for shape 8 (Random). Takes effect at the next reset(). */
    void setNoiseKey (std::uint32_t key) noexcept          { noiseKey = key; }

    float getRateHz() const noexcept                       { return rateHz; }

    /** Current phase in [0, 1). */
    float getPhase() const noexcept
    {
        const float f = static_cast<float> (phase);
        return f >= 1.0f ? 0.0f : f;      // rounding up on conversion must not escape the range
    }

    /** The current phase, then advance one sample. */
    float nextPhase() noexcept
    {
        const float current = getPhase();
        advance();
        return current;
    }

    /** The waveform value at the current phase, then advance one sample. */
    float nextValue (int shape) noexcept
    {
        const float v = LFOWaveforms::applyWaveform (shape, getPhase(), lastRandom, targetRandom);
        advance();
        return v;
    }

    /** A non-random shape sampled at an arbitrary phase, without touching any
        state - for a module that blends two shapes at one phase. */
    static float shapeValue (int shape, float phase01) noexcept
    {
        return LFOWaveforms::applyWaveform (shape, phase01, 0.0f, 0.0f);
    }

private:
    void advance() noexcept
    {
        phase += increment;

        if (phase >= 1.0)
        {
            phase -= 1.0;
            lastRandom = targetRandom;
            targetRandom = FrDiffusion::hashNoiseBipolar (++randomIndex, noiseKey);
        }
    }

    double sampleRate = 48000.0;
    double phase = 0.0;
    double increment = 0.0;
    float rateHz = 0.0f;

    float lastRandom = 0.0f;
    float targetRandom = 0.0f;
    std::uint32_t randomIndex = 1;
    std::uint32_t noiseKey = 1;
};

} // namespace spatcore::dsp
