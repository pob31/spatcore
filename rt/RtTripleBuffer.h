#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <type_traits>

namespace spatcore::rt
{

/**
    Wait-free message-thread -> realtime-thread parameter hand-off.

    The same job as RtSnapshot<T>, without the SpinLock. RtSnapshot holds a lock
    across the POD copy on BOTH sides, so a message thread descheduled inside
    publish() blocks the audio thread until it is scheduled again - a priority
    inversion that only shows up under load, and exactly when it hurts. Here each
    side owns a private slot and the hand-off is a single atomic exchange, so
    neither side can ever wait for the other.

    Three slots: one the writer is filling, one the reader is holding, one in the
    middle ("latest"). The three indices are always a permutation of {0, 1, 2},
    and each side swaps its private index with the shared one atomically, so the
    writer can never be filling a slot the reader is reading.

    The contract:
      1. T is a trivially copyable POD - enforced below. No Strings, no vectors,
         no pointers into message-thread-owned storage.
      2. publish() is called from ONE thread only, acquire() from ONE thread only
         (the usual pairing is message thread -> audio thread). Two publishers or
         two readers need a different primitive.
      3. Values are pre-cooked at publish time (dB -> linear, string -> index) so
         the realtime side does no conversion that could fault or allocate.
      4. LATEST WINS. A reader that acquires once per block and a writer that
         publishes several times in between will see only the newest value; the
         intermediate ones are dropped, not queued.
      5. The reference acquire() returns stays valid until that thread's next
         acquire() call, and no longer.

    Before the first publish() the reader sees a value-initialised T, so make T's
    defaults safe rather than relying on publish-before-enable ordering (the same
    rule as RtSnapshot, for the same reason).
*/
template <typename T>
class RtTripleBuffer
{
    static_assert (std::is_trivially_copyable_v<T>,
                   "RtTripleBuffer<T>: T must be trivially copyable - POD-only hand-off");
    static_assert (std::is_default_constructible_v<T>,
                   "RtTripleBuffer<T>: T must be default constructible - the reader sees T{} until the first publish");

public:
    RtTripleBuffer() = default;

    /** Writer thread: hand over a freshly built snapshot. Wait-free: one POD
        copy into this thread's private slot, then one exchange. */
    void publish (const T& next) noexcept
    {
        slots[writeIndex] = next;

        // Release: the copy above must be visible to whoever takes this slot.
        // Acquire: our next write into the slot we get back must not be
        // reordered before the reader's last reads of it.
        const std::uint8_t previous = latest.exchange (static_cast<std::uint8_t> (writeIndex | kDirtyBit),
                                                       std::memory_order_acq_rel);
        writeIndex = static_cast<std::uint8_t> (previous & kIndexMask);
    }

    /** Reader thread: the newest published value, or the one already held when
        nothing new arrived. Wait-free; never blocks, never allocates. */
    const T& acquire() noexcept
    {
        if ((latest.load (std::memory_order_acquire) & kDirtyBit) == 0)
            return slots[readIndex];        // nothing new - keep what we hold

        const std::uint8_t previous = latest.exchange (readIndex, std::memory_order_acq_rel);
        readIndex = static_cast<std::uint8_t> (previous & kIndexMask);
        return slots[readIndex];
    }

    /** Reader thread: is there a value the reader has not taken yet? Advisory -
        a publish can land the instant after this returns false. */
    bool hasPending() const noexcept
    {
        return (latest.load (std::memory_order_acquire) & kDirtyBit) != 0;
    }

private:
    static constexpr std::uint8_t kIndexMask = 0x03;
    static constexpr std::uint8_t kDirtyBit  = 0x04;

    std::array<T, 3> slots {};
    std::atomic<std::uint8_t> latest { 2 };     // clean; indices 0/1/2 all distinct

    std::uint8_t writeIndex = 0;                // writer thread only
    std::uint8_t readIndex  = 1;                // reader thread only
};

} // namespace spatcore::rt
