#pragma once

#include "BinauralTypes.h"
#include "../dsp/OneEuroFilter.h"
#include <juce_core/juce_core.h>
#include <cmath>

namespace spatcore::binaural
{

/** Wrap an angle to (-pi, pi].

    fmod, not a subtract-until loop: the unwrapped yaw accumulates without
    bound (a listener turning the same way for a whole show), which would make
    the loop count grow with it on the tracker's callback thread. O(1)
    regardless of magnitude. */
inline float wrapPi (float a) noexcept
{
    constexpr float pi = juce::MathConstants<float>::pi;
    constexpr float twoPi = juce::MathConstants<float>::twoPi;

    a = std::fmod (a + pi, twoPi);
    if (a < 0.0f)
        a += twoPi;
    return a - pi;
}

/** 1-Euro tuning for a head-attitude source, in radians.

    beta multiplies rad/s, so it must be chosen against the source's real
    motion range, and minCutoffHz sets the at-rest smoothing — which is also
    the at-rest group delay, roughly 1/(2*pi*minCutoffHz). That trade differs
    sharply between sources: a webcam's landmark jitter needs heavy smoothing
    and can afford the delay (its own pipeline already costs tens of ms), while
    a fused IMU is quiet enough to barely need any and exists precisely to keep
    the latency budget small.

    minCutoffHz <= 0 bypasses filtering entirely (raw pass-through, with the
    yaw unwrap and the finite guards still applied). */
struct HeadAttitudeTuning
{
    float minCutoffHz   = 1.5f;
    float beta          = 3.0f;
    float derivCutoffHz = 1.0f;
};

/**
    The publish-side half of a head-orientation source: yaw unwrap, 1-Euro
    smoothing and the non-finite guards that keep poisoned state from becoming
    permanent.

    Deliberately does NOT include the zero calibration. That step is
    geometry-specific and the two sources need opposite sides of the product:

      - a WEBCAM measures head pose directly, so what a "set zero" cancels is
        the CAMERA's placement relative to the stage -- a rotation on the WORLD
        side, applied by pre-multiplying (R_zeroInv * R_raw);
      - a HEAD-MOUNTED IMU reports body->world, so what a "set zero" cancels is
        how the sensor sits on the headband -- a rotation on the BODY side,
        applied by post-multiplying (q * q_ref^-1).

    They are not two implementations of one idea, they are the two different
    sides of the product, each correct for its geometry. Folding either into
    this shared helper would guarantee the wrong one gets used eventually, so
    process() takes ALREADY-CALIBRATED angles and each source owns its tare.

    Threading: one instance belongs to one producing thread, like the
    OneEuroFilter it wraps.
*/
class HeadAttitudePipeline
{
public:
    HeadAttitudePipeline() = default;
    explicit HeadAttitudePipeline (const HeadAttitudeTuning& t) : tuning (t) {}

    void setTuning (const HeadAttitudeTuning& t) noexcept { tuning = t; }
    const HeadAttitudeTuning& getTuning() const noexcept  { return tuning; }

    /** Producer thread. Returns the orientation to publish.

        Non-finite input is refused before any state is latched: the filters
        carry prevFiltered forward forever, so a single NaN reaching them would
        make tracking never come back. A non-finite RESULT from finite input
        (a pathological timestamp can do it) drops the poisoned history too. */
    HeadOrientation process (float yaw, float pitch, float roll, double nowSeconds) noexcept
    {
        if (! (std::isfinite (yaw) && std::isfinite (pitch) && std::isfinite (roll)))
            return {};

        // Unwrap yaw against the previous sample so the filter never sees the
        // +/-pi seam as a full-circle jump.
        if (hasPrevYaw)
        {
            constexpr float twoPi = juce::MathConstants<float>::twoPi;
            float delta = yaw - std::fmod (prevUnwrappedYaw, twoPi);
            if (delta >  juce::MathConstants<float>::pi) delta -= twoPi;
            if (delta < -juce::MathConstants<float>::pi) delta += twoPi;
            yaw = prevUnwrappedYaw + delta;
        }
        prevUnwrappedYaw = yaw;
        hasPrevYaw = true;

        HeadOrientation out;

        if (tuning.minCutoffHz > 0.0f)
        {
            out.yawRad   = wrapPi (filterYaw.filter (yaw, nowSeconds, tuning.minCutoffHz,
                                                     tuning.beta, tuning.derivCutoffHz));
            out.pitchRad = filterPitch.filter (pitch, nowSeconds, tuning.minCutoffHz,
                                               tuning.beta, tuning.derivCutoffHz);
            out.rollRad  = filterRoll.filter  (roll,  nowSeconds, tuning.minCutoffHz,
                                               tuning.beta, tuning.derivCutoffHz);
        }
        else
        {
            out.yawRad   = wrapPi (yaw);
            out.pitchRad = pitch;
            out.rollRad  = roll;
        }

        out.valid = true;

        if (! isFiniteAttitude (out))
        {
            reset();
            return {};
        }

        return out;
    }

    /** Drop the smoothing history and the yaw-unwrap anchor. Call when the
        calibration reference changes or the stream restarts, so the filters
        do not slew from a now-meaningless previous frame. */
    void reset() noexcept
    {
        filterYaw.reset();
        filterPitch.reset();
        filterRoll.reset();
        prevUnwrappedYaw = 0.0f;
        hasPrevYaw = false;
    }

private:
    HeadAttitudeTuning tuning;

    spatcore::dsp::OneEuroFilter filterYaw, filterPitch, filterRoll;
    float prevUnwrappedYaw = 0.0f;
    bool hasPrevYaw = false;
};

} // namespace spatcore::binaural
