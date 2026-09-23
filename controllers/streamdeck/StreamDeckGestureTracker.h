#pragma once

#include <cstdint>

namespace spatcore::controllers {

/**
 * StreamDeckGestureTracker — when hardware input becomes a new edit GESTURE.
 *
 * The GUI's rule is one undo step per gesture: a slider drag from mouse-down
 * to mouse-up. A Stream Deck dial has no touch sensing, so a gesture is a RUN
 * of turns of one control with no pause longer than the idle time between
 * them - turn, turn, turn is one step; a pause, or another control, starts the
 * next. A press (a button, a dial press that acts, a combo choice confirmed)
 * is always a gesture of its own and ends any run, and so does navigation (a
 * page, section or channel change): the next turn starts afresh.
 *
 * Pure and clock-free - the caller passes the time - so it is testable without
 * a device, and the manager stays the only thing that knows about hardware.
 */
class StreamDeckGestureTracker
{
public:
    /** The pause that ends a run. The effects link funnel uses the same 800 ms
        to decide when a bypassed edit is a new gesture. */
    static constexpr std::uint32_t kDefaultIdleMs = 800;

    explicit StreamDeckGestureTracker (std::uint32_t idleMs = kDefaultIdleMs) noexcept
        : idle (idleMs) {}

    void setIdleMs (std::uint32_t idleMs) noexcept { idle = idleMs; }
    std::uint32_t getIdleMs() const noexcept      { return idle; }

    /** A turn of control `key` at time `nowMs` (a millisecond counter that may
        wrap: the gap is taken modulo 2^32). True when it starts a new gesture. */
    bool turn (std::int64_t key, std::uint32_t nowMs) noexcept
    {
        const bool fresh = ! running || key != lastKey || static_cast<std::uint32_t> (nowMs - lastMs) > idle;
        running = true;
        lastKey = key;
        lastMs = nowMs;
        return fresh;
    }

    /** A press is a gesture of its own; it also ends any run of turns. */
    void press() noexcept { running = false; }

    /** Navigation: the next turn starts a new gesture whatever it touches. */
    void breakRun() noexcept { running = false; }

private:
    std::uint32_t idle;
    bool running = false;
    std::int64_t lastKey = 0;
    std::uint32_t lastMs = 0;
};

} // namespace spatcore::controllers
