#pragma once

#include "BinauralTypes.h"
#include "../dsp/DelayTargetSmoother.h"
#include "../dsp/WFSHighShelfFilter.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <vector>
#include <cstdint>
#include <cmath>

namespace spatcore::binaural
{

/**
    Parametric (structural) HRTF renderer — spherical-head model after
    Brown & Duda (1998), per source and ear:

      1. ITD — Woodworth spherical-head delay from the incidence angle Θ
         between the source direction and the ear axis:
             T(Θ) = −(a/c)·cos Θ            for Θ <  π/2   (unshadowed side)
             T(Θ) =  (a/c)·(Θ − π/2)        for Θ >= π/2   (creeping wave)
         applied on top of the propagation delay r/c through a shared
         fractional delay line, smoothed by DelayTargetSmoother (C1 + teleport
         envelope for source jumps; head rotation never trips the teleport).

      2. Head shadow — one-pole/one-zero  H(s) = (2ω0 + αs)/(2ω0 + s),
         ω0 = c/a, α(Θ) = 1.05 + 0.95·cos(1.2·Θ). Unity at DC, α at Nyquist
         (+6 dB bright side, −20 dB deep shadow). Bilinear-transformed
         coefficients are recomputed per block and interpolated per sample —
         first-order, endpoint interpolation is stable — so fast head rotation
         never zippers.

      3. Elevation cue — RBJ peaking notch per ear whose center sweeps
         ~6 kHz (el = −40°) → ~10 kHz (el = +60°), depth −8 dB fading out
         above +20°. Coefficients updated at block rate.

      4. Distance — gain 1/max(r, 1 m) (same reference as the legacy mode, so
         switching modes keeps loudness comparable) and the familiar air
         absorption shelf (WFSHighShelfFilter, dB = r × −0.3).

    No near-field correction by design: sources sit several meters out.

    Threading: prepare()/reset() with the worker stopped; processSource() on
    the render worker only. No allocation or locks in processSource().
*/
class StructuralHrtfRenderer
{
public:
    void prepare (double newSampleRate, int newMaxBlockSize, int maxSources)
    {
        juce::ignoreUnused (newMaxBlockSize);
        sampleRate = newSampleRate;

        // 1-second max delay, matching the legacy mode's delay budget.
        delayLineLength = (int) (sampleRate * 1.0);

        const int smootherWindow = juce::jmax (2, (int) (sampleRate * kSmootherWindowSec));

        sources.clear();
        sources.resize ((size_t) juce::jmax (0, maxSources));
        for (auto& src : sources)
        {
            src.delayLine.assign ((size_t) delayLineLength, 0.0f);
            src.writePos = 0;
            for (auto* ear : { &src.left, &src.right })
            {
                ear->delaySmoother.prepare (smootherWindow);
                ear->airShelf.prepare (sampleRate);
            }
        }
    }

    void reset()
    {
        for (auto& src : sources)
        {
            std::fill (src.delayLine.begin(), src.delayLine.end(), 0.0f);
            src.writePos = 0;
            for (auto* ear : { &src.left, &src.right })
            {
                ear->delaySmoother.reset();
                ear->airShelf.reset();
                ear->shadow = {};
                ear->notch = {};
                ear->prevGain = 0.0f;
                ear->gainInit = false;
            }
        }
    }

    /**
        Render one source into the stereo accumulators.

        dir          source direction/distance in the head frame
        headRadius   meters (personalization)
        extraDelayMs global binaural delay offset (binauralDelay parameter)
        gainScale    linear pre-gain (solo/reverb-balance trims), 1 = unity
        blockStart   monotonic sample counter (engine-maintained)
    */
    void processSource (int sourceIdx,
                        const SourceDirection& dir,
                        float headRadius,
                        float extraDelayMs,
                        float gainScale,
                        const float* input,
                        float* outL, float* outR,
                        int numSamples,
                        std::int64_t blockStart)
    {
        if (sourceIdx < 0 || sourceIdx >= (int) sources.size())
            return;

        auto& src = sources[(size_t) sourceIdx];

        // ---- Block-rate parameter cook -----------------------------------
        // Unit direction in the head frame (az/el are vertical-polar).
        const float cosEl = std::cos (dir.elRad);
        const float ux = std::sin (dir.azRad) * cosEl;   // toward the right ear

        const float a = juce::jmax (0.02f, headRadius);
        const float invC = 1.0f / kSpeedOfSound;
        const float fs = (float) sampleRate;

        // Distance gain — legacy 1 m reference law — and air absorption.
        const float dist = juce::jmax (dir.distance, 0.05f);
        const float distGain = (dist > 1.0f ? 1.0f / dist : 1.0f) * gainScale;
        const float airDb = dist * kAirShelfDbPerMeter;

        // Propagation + per-ear Woodworth ITD, in samples.
        const float baseDelaySec = dist * invC + extraDelayMs * 0.001f;

        const float thetaR = std::acos (juce::jlimit (-1.0f, 1.0f, ux));
        const float thetaL = std::acos (juce::jlimit (-1.0f, 1.0f, -ux));

        auto woodworth = [a, invC] (float theta)
        {
            return theta < juce::MathConstants<float>::halfPi
                 ? -(a * invC) * std::cos (theta)
                 :  (a * invC) * (theta - juce::MathConstants<float>::halfPi);
        };

        const float delaySamplesL = juce::jmax (0.0f, (baseDelaySec + woodworth (thetaL)) * fs);
        const float delaySamplesR = juce::jmax (0.0f, (baseDelaySec + woodworth (thetaR)) * fs);

        src.left.delaySmoother.observe (delaySamplesL, blockStart);
        src.right.delaySmoother.observe (delaySamplesR, blockStart);

        // Shadow filter targets (per-sample interpolated in the loop).
        computeShadowCoeffs (thetaL, a, src.left);
        computeShadowCoeffs (thetaR, a, src.right);

        // Elevation notch (block-rate; identical both ears).
        updateNotchCoeffs (dir.elRad, src.left);
        updateNotchCoeffs (dir.elRad, src.right);

        src.left.airShelf.setGainDb (airDb);
        src.right.airShelf.setGainDb (airDb);

        // Output gain endpoints (linear ramp across the block, like the legacy path).
        for (auto* ear : { &src.left, &src.right })
        {
            if (! ear->gainInit) { ear->prevGain = distGain; ear->gainInit = true; }
        }

        // ---- Sample loop --------------------------------------------------
        float* delayData = src.delayLine.data();
        const float maxDelay = (float) (delayLineLength - 2);
        const float invNumSamples = 1.0f / (float) numSamples;

        const float gainStartL = src.left.prevGain,  gainEndL = distGain;
        const float gainStartR = src.right.prevGain, gainEndR = distGain;
        src.left.prevGain = distGain;
        src.right.prevGain = distGain;

        int writePos = src.writePos;
        for (int i = 0; i < numSamples; ++i)
        {
            delayData[writePos] = input[i];

            const float t = (float) i * invNumSamples;
            const std::int64_t now = blockStart + i;

            outL[i] += processEar (src.left, delayData, writePos, now, t,
                                   gainStartL + (gainEndL - gainStartL) * t, maxDelay);
            outR[i] += processEar (src.right, delayData, writePos, now, t,
                                   gainStartR + (gainEndR - gainStartR) * t, maxDelay);

            if (++writePos >= delayLineLength)
                writePos = 0;
        }
        src.writePos = writePos;
    }

private:
    static constexpr float kSpeedOfSound      = 343.0f;
    static constexpr float kAirShelfDbPerMeter = -0.3f;
    static constexpr double kSmootherWindowSec = 0.005;   // 5 ms — head motion stays fluid

    /** First-order head-shadow filter state + per-block coefficient ramp. */
    struct ShadowState
    {
        float b0 = 1.0f, b1 = 0.0f, a1 = 0.0f;             // current-block targets
        float pb0 = 1.0f, pb1 = 0.0f, pa1 = 0.0f;          // previous-block values
        float x1 = 0.0f, y1 = 0.0f;
        bool init = false;
    };

    /** RBJ peaking "notch" (negative gain) for the elevation cue. */
    struct NotchState
    {
        float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f, a1 = 0.0f, a2 = 0.0f;
        float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f;
    };

    struct EarState
    {
        spatcore::dsp::DelayTargetSmoother delaySmoother;
        ShadowState shadow;
        NotchState notch;
        spatcore::dsp::WFSHighShelfFilter airShelf;
        float prevGain = 0.0f;
        bool gainInit = false;
    };

    struct SourceState
    {
        std::vector<float> delayLine;   // shared line, two read taps
        int writePos = 0;
        EarState left, right;
    };

    void computeShadowCoeffs (float theta, float headRadius, EarState& ear)
    {
        // α(Θ): +6 dB HF at Θ=0 (ear toward source) down to ~0.1 at Θ=150°.
        const float alpha = 1.05f + 0.95f * std::cos (1.2f * theta);
        const float w0 = kSpeedOfSound / headRadius;       // rad/s
        const float fs2 = 2.0f * (float) sampleRate;       // bilinear 2/T

        const float invDen = 1.0f / (w0 + fs2);
        const float b0 = (w0 + alpha * fs2) * invDen;
        const float b1 = (w0 - alpha * fs2) * invDen;
        const float a1 = (w0 - fs2) * invDen;

        auto& s = ear.shadow;
        if (! s.init)
        {
            s.pb0 = b0; s.pb1 = b1; s.pa1 = a1;
            s.init = true;
        }
        else
        {
            s.pb0 = s.b0; s.pb1 = s.b1; s.pa1 = s.a1;
        }
        s.b0 = b0; s.b1 = b1; s.a1 = a1;
    }

    void updateNotchCoeffs (float elRad, EarState& ear)
    {
        const float elDeg = juce::jlimit (-40.0f, 60.0f, juce::radiansToDegrees (elRad));

        // Center sweeps 6 → 10 kHz over el −40° → +60°; depth −8 dB below
        // +20°, fading to 0 at +60° (cue vanishes overhead).
        const float fc = 6000.0f + (elDeg + 40.0f) * (4000.0f / 100.0f);
        float depthDb = -8.0f;
        if (elDeg > 20.0f)
            depthDb *= juce::jmax (0.0f, (60.0f - elDeg) / 40.0f);

        const float A = std::pow (10.0f, depthDb / 40.0f);
        const float w = juce::MathConstants<float>::twoPi * fc / (float) sampleRate;
        const float sn = std::sin (w), cs = std::cos (w);
        const float alpha = sn / (2.0f * kNotchQ);

        const float a0inv = 1.0f / (1.0f + alpha / A);
        auto& n = ear.notch;
        n.b0 = (1.0f + alpha * A) * a0inv;
        n.b1 = (-2.0f * cs) * a0inv;
        n.b2 = (1.0f - alpha * A) * a0inv;
        n.a1 = n.b1;                       // same -2cos(w)/a0 term
        n.a2 = (1.0f - alpha / A) * a0inv;
    }

    static constexpr float kNotchQ = 2.0f;

    /** One ear, one sample: smoothed delay tap → shadow → notch → air shelf → gain. */
    float processEar (EarState& ear, const float* delayData, int writePos,
                      std::int64_t now, float t, float gain, float maxDelay)
    {
        const auto d = ear.delaySmoother.smoothedAt (now);
        float delaySamples = juce::jlimit (0.0f, maxDelay, d.delay);

        float exactReadPos = (float) writePos - delaySamples;
        if (exactReadPos < 0.0f)
            exactReadPos += (float) delayLineLength;

        int readPos1 = (int) exactReadPos;
        if (readPos1 >= delayLineLength) readPos1 -= delayLineLength;
        int readPos2 = readPos1 + 1;
        if (readPos2 >= delayLineLength) readPos2 = 0;
        const float fraction = exactReadPos - std::floor (exactReadPos);

        float sample = delayData[readPos1] + fraction * (delayData[readPos2] - delayData[readPos1]);
        sample *= d.gain;   // teleport mute envelope (source jumps)

        // Head shadow, coefficients lerped across the block.
        {
            auto& s = ear.shadow;
            const float b0 = s.pb0 + (s.b0 - s.pb0) * t;
            const float b1 = s.pb1 + (s.b1 - s.pb1) * t;
            const float a1 = s.pa1 + (s.a1 - s.pa1) * t;
            const float y = b0 * sample + b1 * s.x1 - a1 * s.y1;
            s.x1 = sample;
            s.y1 = y;
            sample = y;
        }

        // Elevation notch (block-rate coefficients).
        {
            auto& n = ear.notch;
            const float y = n.b0 * sample + n.b1 * n.x1 + n.b2 * n.x2
                          - n.a1 * n.y1 - n.a2 * n.y2;
            n.x2 = n.x1; n.x1 = sample;
            n.y2 = n.y1; n.y1 = y;
            sample = y;
        }

        sample = ear.airShelf.processSample (sample);
        return sample * gain;
    }

    double sampleRate = 48000.0;
    int delayLineLength = 48000;
    std::vector<SourceState> sources;
};

} // namespace spatcore::binaural
