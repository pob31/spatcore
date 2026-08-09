#pragma once

#include <juce_core/juce_core.h>
#include <cmath>

namespace spatcore::dsp
{

/**
    Single-axis 1-Euro filter (Casiez, Roussel & Vogel, CHI 2012).

    Adaptive low-pass: the cutoff opens with the signal's speed, so slow
    movement is smoothed hard (jitter dies) while fast movement passes almost
    untouched (no lag when it would be felt). Unit-agnostic — the caller picks
    minCutoff/beta to match its units (metres for tracked positions, radians
    for head attitude; beta must scale with the unit).

    Extracted verbatim from TrackingPositionFilter, which had it as a private
    nested struct, so head tracking can share the same implementation. Not
    thread-safe on its own: one instance belongs to one producer thread (or is
    guarded by its owner, as TrackingPositionFilter does per input).
*/
struct OneEuroFilter
{
    float prevFiltered = 0.0f;
    float prevDerivative = 0.0f;
    double prevTimestamp = 0.0;
    bool initialized = false;

    void reset()
    {
        prevFiltered = 0.0f;
        prevDerivative = 0.0f;
        prevTimestamp = 0.0;
        initialized = false;
    }

    float filter (float rawValue, double timestamp,
                  float minCutoff, float beta, float derivativeCutoff)
    {
        if (! initialized)
        {
            prevFiltered = rawValue;
            prevDerivative = 0.0f;
            prevTimestamp = timestamp;
            initialized = true;
            return rawValue;
        }

        double dt = timestamp - prevTimestamp;
        if (dt < 0.001) dt = 0.001;
        if (dt > 1.0) dt = 1.0;
        prevTimestamp = timestamp;

        // Estimate derivative (speed)
        float rawDerivative = (rawValue - prevFiltered) / static_cast<float> (dt);

        // Smooth the derivative with fixed cutoff
        float dAlpha = computeAlpha (derivativeCutoff, dt);
        float filteredDerivative = lowPass (rawDerivative, prevDerivative, dAlpha);
        prevDerivative = filteredDerivative;

        // Adaptive cutoff based on speed
        float speed = std::abs (filteredDerivative);
        float cutoff = minCutoff + beta * speed;

        // Filter the value
        float alpha = computeAlpha (cutoff, dt);
        float filteredValue = lowPass (rawValue, prevFiltered, alpha);
        prevFiltered = filteredValue;

        return filteredValue;
    }

private:
    static float computeAlpha (float cutoff, double dt)
    {
        float tau = 1.0f / (2.0f * juce::MathConstants<float>::pi * cutoff);
        return 1.0f / (1.0f + tau / static_cast<float> (dt));
    }

    static float lowPass (float raw, float prev, float alpha)
    {
        return prev + alpha * (raw - prev);
    }
};

} // namespace spatcore::dsp
