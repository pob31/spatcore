#pragma once

#include <cmath>
#include <cstdint>
#include <algorithm>

/*
    FrDiffusionModel — the shared Floor-Reflection diffusion model.

    Faithful port of the Max/MSP prototype's [smooth] subpatcher (per input,
    per output channel), with d = Diffusion percent 0..100:

        mc.rand~ (300 - 2.5*d Hz)          band-limited noise: uniform values in
                                           [-1,1) with LINEAR interpolation
        mc.*~ (self)                       squared -> unipolar [0,1]
        mc.*~ line~($1 100)                x amplitude A = d * 3*sr/1e6 samples
                                           (= 0.003*d ms, max 0.3 ms), amplitude
                                           ramped over 100 ms on d changes
        mc.rampsmooth~ R R                 R = 20 + d*sr/10000 samples; retargeted
                                           every sample == one-pole y += (x-y)/R
        mc.+~                              ADDED to the FR delay (samples, >= 0:
                                           the reflection only ever gets LONGER)

    Every (input x output) routing uses an INDEPENDENT noise stream (hash-keyed
    segment endpoints), matching mc.rand~ @chans per-channel independence: each
    source scatters off its own patch of floor toward each speaker.

    Used identically by the CPU InputBuffer processor (per-sample), the CPU
    OutputBuffer processor (per-sample), and the GPU backends' host-side staging
    (per-launch advanceSpan; the kernels' prev->curr per-sample interpolation
    approximates the intra-block trajectory). One header so the algorithms can
    never drift apart in character.
*/
namespace spatcore::dsp {

namespace FrDiffusion
{
    // ==== Max-patch provenance constants (tunable) ====
    constexpr float kRateBaseHz       = 300.0f;  // rand~ freq at d=0:  f = 300 - 2.5*d
    constexpr float kRatePerPct       = 2.5f;    //   -> 50 Hz at d=100
    constexpr float kAmpMsPerPct      = 0.003f;  // amplitude: 0.003 ms per percent (0.3 ms max)
    constexpr float kAmpSlewPerSample = 0.003f;  // line~ "$1 100": full-scale amp ramp in 100 ms,
                                                 // (100*kAmpMsPerPct*sr/1000)/(0.1*sr) samples/sample
    constexpr float kSlewBaseSamples  = 20.0f;   // rampsmooth~ ramp: R = 20 + d*sr/10000
    constexpr float kSlewPerPctPerSr  = 1.0e-4f;
    constexpr float kEngageEps        = 1.0e-4f; // -80 dB: below this an FR tap counts as silent

    /** Bipolar white noise in [-1, 1) for step index n on the stream identified
        by key. Squirrel-style integer hash: consecutive n gives white-noise
        quality; distinct keys give fully independent streams (unlike xorshift,
        where every seed is just a point on one shared orbit). */
    inline float hashNoiseBipolar (uint32_t n, uint32_t key) noexcept
    {
        n *= 0xB5297A4Du;
        n += key;
        n ^= n >> 8;
        n += 0x68E31DA4u;
        n ^= n << 8;
        n *= 0x1B56C4E9u;
        n ^= n >> 8;
        return static_cast<float> (static_cast<int32_t> (n)) * (1.0f / 2147483648.0f);
    }

    /** Stream key for an (input, output) routing pair. */
    inline uint32_t makeKey (int inputIndex, int outputIndex) noexcept
    {
        return static_cast<uint32_t> (inputIndex) * 0x9E3779B9u
             + static_cast<uint32_t> (outputIndex + 1) * 0x85EBCA6Bu + 1u;
    }

    /** Per-stream state. Deterministic: segment endpoints are pure functions of
        (segIndex, key), so a stream's trajectory depends only on how many
        samples it has advanced since resetState() — no shared RNG, no
        processing-order dependence. */
    struct State
    {
        float    segA  = 0.0f;   // current rand~ segment start value, [-1,1)
        float    segB  = 0.0f;   // current rand~ segment end value,   [-1,1)
        float    phase = 0.0f;   // samples into the current segment
        uint32_t segIndex = 0u;  // hash index of segB
        float    amp    = 0.0f;  // line~-ramped amplitude (audio samples)
        float    smooth = 0.0f;  // rampsmooth~ one-pole state (audio samples, >= 0)
    };

    inline void resetState (State& s, uint32_t key) noexcept
    {
        s.segA = hashNoiseBipolar (0u, key);
        s.segB = hashNoiseBipolar (1u, key);
        s.segIndex = 1u;
        s.phase = 0.0f;
        s.amp = 0.0f;
        s.smooth = 0.0f;
    }

    struct Coeffs
    {
        float segLen     = 1.0f;  // rand~ segment length: sr / (300 - 2.5*d), samples
        float invSegLen  = 1.0f;
        float ampTarget  = 0.0f;  // d * kAmpMsPerPct ms in samples; 0 = diffusion off
        float invR       = 0.0f;  // rampsmooth~ one-pole: 1 / (20 + d*sr/10000)
        float ampSamples = 0.0f;  // == ampTarget (excursion bound, kept for consumer gates)
    };

    /** Derive the per-sample coefficients from the Diffusion fraction d (0..1)
        and the audio sample rate. Valid at any update rate: processSample()
        advances one sample, advanceSpan() advances k samples with identical
        composed behavior. */
    inline Coeffs computeCoeffs (float d, float sampleRate) noexcept
    {
        Coeffs c;
        if (sampleRate <= 0.0f)
            return c;

        const float dPct = 100.0f * std::min (1.0f, std::max (0.0f, d));

        c.segLen     = sampleRate / (kRateBaseHz - kRatePerPct * dPct);
        c.invSegLen  = 1.0f / c.segLen;
        c.ampTarget  = dPct * kAmpMsPerPct * (sampleRate / 1000.0f);
        c.invR       = 1.0f / (kSlewBaseSamples + dPct * sampleRate * kSlewPerPctPerSr);
        c.ampSamples = c.ampTarget;
        return c;
    }

    /** Advance the rand~ phase by k samples (crossing as many segment
        boundaries as needed) and return the linearly interpolated noise value
        at the new position. */
    inline float advancePhase (State& s, uint32_t key, const Coeffs& c, float k) noexcept
    {
        s.phase += k;
        while (s.phase >= c.segLen)
        {
            s.phase -= c.segLen;
            s.segA = s.segB;
            s.segB = hashNoiseBipolar (++s.segIndex, key);
        }
        return s.segA + (s.segB - s.segA) * (s.phase * c.invSegLen);
    }

    /** One per-sample step: advances the stream by one sample and returns the
        jitter in audio samples (unipolar, >= 0 — ADD to the FR delay). */
    inline float processSample (State& s, uint32_t key, const Coeffs& c) noexcept
    {
        const float v = advancePhase (s, key, c, 1.0f);
        const float da = c.ampTarget - s.amp;
        s.amp += std::min (kAmpSlewPerSample, std::max (-kAmpSlewPerSample, da));
        const float target = v * v * s.amp;     // square the INTERPOLATED value (Max *~ self)
        s.smooth += (target - s.smooth) * c.invR;
        return s.smooth;
    }

    /** Advance the stream by k samples in one step and return the jitter at
        the end of the span. Composes with processSample(): k one-sample steps
        against a frozen endpoint equal one advanceSpan(k) step (the target is
        sampled once per span, which is the GPU launches' approximation; the
        kernels' prev->curr lerp fills in the intra-span trajectory). Also used
        to keep silent/inactive streams phase-consistent at O(1) cost. */
    inline float advanceSpan (State& s, uint32_t key, const Coeffs& c, int k) noexcept
    {
        const float kf = static_cast<float> (k);
        const float v = advancePhase (s, key, c, kf);
        const float da = c.ampTarget - s.amp;
        const float maxDa = kAmpSlewPerSample * kf;
        s.amp += std::min (maxDa, std::max (-maxDa, da));
        const float target = v * v * s.amp;
        const float alphaK = 1.0f - std::pow (1.0f - c.invR, kf);
        s.smooth += (target - s.smooth) * alphaK;
        return s.smooth;
    }
} // namespace FrDiffusion

} // namespace spatcore::dsp

// Extraction-compat alias — app code migrates to qualified names later.
namespace FrDiffusion = spatcore::dsp::FrDiffusion;
