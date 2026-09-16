#pragma once

#include <cmath>

namespace spatcore::dsp
{

/**
    Per-sample one-pole glide towards a target, with an exact arrival.

    The coefficient law is the one the reverb pre/post processors already use,
    coef = 1 - exp(-1 / (sr * tau)), so tau is the time constant: the smoother
    covers 63 % of the distance in tau and ~99 % in 5 tau. It is NOT a
    completion time - a fade quoted as "5 ms" in the effects plan settles after
    roughly 9 tau.

    Why the snap: a one-pole in float STOPS before it arrives. Once
    coef * (target - current) falls below half an ULP of current the addition
    rounds back to current and the value freezes there forever - a 12 kHz ->
    96 kHz glide at tau = 10 ms parks a couple of hertz short, and, worse, a
    bypass fade would never reach exactly 0 or 1, so a "settled" slot would keep
    applying a crossfade that is only nearly transparent. next() therefore takes
    the target exactly both when a step no longer changes the float and when the
    remaining distance is within snapEpsilon, which lets callers treat
    isSettled() as bit-exact arrival.

    Not thread-safe and not meant to be: one instance belongs to one audio
    thread. No allocation anywhere.
*/
class OnePoleSmoother
{
public:
    OnePoleSmoother() = default;

    /** tau <= 0 (or a nonsensical sample rate) means "no smoothing": next()
        jumps straight to the target. */
    void setTimeConstant (double sampleRate, float tauSeconds) noexcept
    {
        if (sampleRate > 0.0 && tauSeconds > 0.0f)
            coef = 1.0f - std::exp (-1.0f / (static_cast<float> (sampleRate) * tauSeconds));
        else
            coef = 1.0f;
    }

    /** Distance at which next() stops gliding and takes the target exactly.
        Scale it to the parameter: 1e-4 is -80 dB on a unit gain, but a smoother
        carrying hertz or bits wants its own. */
    void setSnapEpsilon (float absoluteEpsilon) noexcept   { snapEpsilon = absoluteEpsilon; }

    /** Jump to v with no glide (use on prepare/reset, and for the first value
        after a parameter set is applied). */
    void snap (float v) noexcept                           { current = target = v; }

    void  setTarget (float v) noexcept                     { target = v; }
    float getTarget() const noexcept                       { return target; }
    float getCurrent() const noexcept                      { return current; }
    float getCoefficient() const noexcept                  { return coef; }

    /** True once current is bit-exactly the target. */
    bool isSettled() const noexcept                        { return current == target; }

    /** One step. Cheap enough to call per sample. */
    float next() noexcept
    {
        if (current == target)
            return current;

        const float stepped = current + coef * (target - current);

        // Arrived, stalled below an ULP, or close enough: take the target exactly.
        if (stepped == current || std::fabs (target - stepped) <= snapEpsilon)
            current = target;
        else
            current = stepped;

        return current;
    }

private:
    float coef = 1.0f;
    float current = 0.0f;
    float target = 0.0f;
    float snapEpsilon = 1.0e-4f;
};

} // namespace spatcore::dsp
