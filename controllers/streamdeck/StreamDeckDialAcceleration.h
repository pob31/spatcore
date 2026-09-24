#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace spatcore::controllers {

/**
 * StreamDeckDialAcceleration — how many steps one click of a dial is worth.
 *
 * The Stream Deck+ reports a turning dial every 50 ms, with the signed number
 * of clicks it turned in that window (measured 2026-09-24 on the updated
 * firmware: 1-3 clicks on a slow or medium turn, up to 16 on a flick, never a
 * reversal inside a flick). The clicks of one report ARE the speed of the
 * turn, so no clock is needed.
 *
 * Up to kSlowClicks in a report, a click is one step, as a turn always was.
 * More clicks multiply the step, up to the dial's ceiling from kFastClicks on;
 * the slower tail of a flick is exact again, which is what lands the value.
 * The ceiling comes from the dial's range, so that two full-speed flicks sweep
 * it and a small range never speeds up at all.
 *
 * Pure and stateless, so it is testable without a device. The manager applies
 * it to unpressed turns only: press + turn stays the exact fine step.
 */
struct StreamDeckDialAcceleration
{
    /** Clicks in one report up to which each click is one step (40 clicks/s). */
    static constexpr int kSlowClicks = 2;

    /** Clicks in one report from which each click is worth the whole ceiling. */
    static constexpr int kFastClicks = 10;

    /** Full-speed clicks that sweep a dial's range: about two flicks. */
    static constexpr double kSweepClicks = 30.0;

    /** The most steps one click may ever be worth. */
    static constexpr int kMaxMultiplier = 50;

    /** The most steps one click of a dial may be worth.
        @param maxAcceleration  the binding's own cap: 0 = from the range,
                                1 = never, N = at most N
        @param exponential      the binding steps in normalised space, which
                                DialBinding::applyStep does only when
                                0 < minValue < maxValue as well */
    static int ceilingFor (int maxAcceleration, double minValue, double maxValue,
                           double step, bool exponential) noexcept
    {
        if (maxAcceleration > 0)
            return std::min (maxAcceleration, kMaxMultiplier);

        if (! (step > 0.0) || ! std::isfinite (step))
            return 1;

        const bool normalised = exponential && minValue > 0.0 && maxValue > minValue;
        const double sweep = normalised ? 1.0 / step : (maxValue - minValue) / step;

        if (! (sweep > 0.0) || ! std::isfinite (sweep))
            return 1;

        return static_cast<int> (std::clamp (std::floor (sweep / kSweepClicks),
                                             1.0, static_cast<double> (kMaxMultiplier)));
    }

    /** How many steps each click of one report is worth.
        @param clicks   the report's clicks, either sign
        @param ceiling  the dial's ceilingFor() */
    static int multiplier (int clicks, int ceiling) noexcept
    {
        const long long n = std::llabs (static_cast<long long> (clicks));

        if (ceiling <= 1 || n <= kSlowClicks)
            return 1;

        if (n >= kFastClicks)
            return ceiling;

        const double s = static_cast<double> (n - kSlowClicks) / static_cast<double> (kFastClicks - kSlowClicks);
        return 1 + static_cast<int> (std::lround ((ceiling - 1) * s * s));
    }
};

} // namespace spatcore::controllers
