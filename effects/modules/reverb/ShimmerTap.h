#pragma once

#include "ReverbDelayLine.h"

namespace spatcore::effects
{

/**
    A pitch-shifting read of a reverb line - the shimmer in the Shimmer model.

    Two read heads sweep a sawtooth of delay around the line's own length, half
    a period apart. A head whose delay shrinks by (r - 1) samples per sample
    plays the line r times faster, i.e. r times higher; one whose delay grows
    by (1 - r) plays it r times lower. Each sweeps a window W and jumps back,
    and the heads are crossfaded by triangles - each weighs nothing at the
    moment it jumps, and the two always sum to one.

    Centred on the line length rather than starting from it, so the loop's
    AVERAGE length is unchanged: the recirculation time, and with it the decay
    law, hold whatever the interval.

    The shifted signal can never be louder than the line: it is a convex mix of
    two reads of the same line (|w1 a + w2 b| <= max(|a|, |b|)), which is what
    lets a shimmer line's feedback stay contractive at any amount.

    Reads are Catmull-Rom and the phase is double precision (a slow sawtooth
    in float would quantise its own slope). No allocation, no libm.
*/
class ShimmerTap
{
public:
    /** ratio: the pitch ratio, 2 an octave up, 0.5 an octave down. window:
        the samples one sweep covers. */
    void configure (double ratio, float windowSamples) noexcept
    {
        up = ratio >= 1.0;
        window = windowSamples > 0.0f ? windowSamples : 0.0f;

        const double slope = up ? ratio - 1.0 : 1.0 - ratio;       // delay change per sample
        increment = window > 0.0f ? slope / static_cast<double> (window) : 0.0;
    }

    /** Where reset() puts the sweep, in cycles - staggering several taps keeps
        their crossfades from landing together. */
    void setStartPhase (double phase01) noexcept
    {
        startPhase = phase01 - static_cast<double> (static_cast<long long> (phase01));
        if (startPhase < 0.0)
            startPhase += 1.0;
    }

    void reset() noexcept { phase = startPhase; }

    /** The shifted signal from `line`, around `centre` samples of delay, then
        one sample on. Call before the line's write for this sample, as every
        read in the tank does. */
    float read (const ReverbDelayLine& line, float centre) noexcept
    {
        double second = phase + 0.5;
        if (second >= 1.0)
            second -= 1.0;

        const float p1 = static_cast<float> (phase);
        const float p2 = static_cast<float> (second);

        const float d1 = up ? centre + window * (0.5f - p1) : centre + window * (p1 - 0.5f);
        const float d2 = up ? centre + window * (0.5f - p2) : centre + window * (p2 - 0.5f);

        // Triangles half a period apart: head one peaks mid-sweep and is silent
        // at its jump (phase 0), where head two is mid-sweep.
        const float t = 2.0f * p1 - 1.0f;
        const float w1 = 1.0f - (t < 0.0f ? -t : t);
        const float w2 = 1.0f - w1;

        phase += increment;
        if (phase >= 1.0)
            phase -= 1.0;

        return w1 * line.readHermite (d1) + w2 * line.readHermite (d2);
    }

    double getPhase() const noexcept { return phase; }

private:
    double phase = 0.0;
    double startPhase = 0.0;
    double increment = 0.0;
    float window = 0.0f;
    bool up = true;
};

} // namespace spatcore::effects
