#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <cstdint>
#include <atomic>

namespace spatcore::rt {

//==============================================================================
/**
    Single-producer, multi-consumer ring buffer for sharing input audio data
    across multiple output processor threads.

    The producer (audio callback) writes via write().
    Each consumer tracks its own read position via readWithPosition().
    No contention between consumers since each has an independent cursor.
*/
class SharedInputRingBuffer
{
public:
    SharedInputRingBuffer() = default;

    void setSize(int numSamples)
    {
        bufferSize = numSamples;
        buffer.setSize(1, bufferSize);
        buffer.clear();
        writePos.store(0, std::memory_order_relaxed);
        totalWritten.store(0, std::memory_order_relaxed);
    }

    /** Write samples (called by single producer — audio callback thread). */
    int write(const float* data, int numSamples)
    {
        int wp = writePos.load(std::memory_order_relaxed);
        int toWrite = juce::jmin(numSamples, bufferSize - 1); // leave 1 empty slot
        auto* dst = buffer.getWritePointer(0);

        // Use memcpy with wrap-around handling (1-2 copies instead of per-sample loop)
        int firstChunk = juce::jmin(toWrite, bufferSize - wp);
        std::memcpy(dst + wp, data, static_cast<size_t>(firstChunk) * sizeof(float));

        int secondChunk = toWrite - firstChunk;
        if (secondChunk > 0)
            std::memcpy(dst, data + firstChunk, static_cast<size_t>(secondChunk) * sizeof(float));

        wp = (wp + toWrite) % bufferSize;

        // Published before the position, so a consumer that reads the counter
        // first can never conclude it is further behind than it really is.
        totalWritten.fetch_add(static_cast<std::uint64_t>(toWrite), std::memory_order_relaxed);
        writePos.store(wp, std::memory_order_release);
        return toWrite;
    }

    /** Read samples using a per-consumer read position (thread-safe, no shared mutation). */
    int readWithPosition(int& readPosition, float* data, int numSamples) const
    {
        int wp = writePos.load(std::memory_order_acquire);
        int available = getAvailableAt(wp, readPosition);
        int toRead = juce::jmin(numSamples, available);
        auto* src = buffer.getReadPointer(0);

        // Use memcpy with wrap-around handling (1-2 copies instead of per-sample loop)
        int firstChunk = juce::jmin(toRead, bufferSize - readPosition);
        std::memcpy(data, src + readPosition, static_cast<size_t>(firstChunk) * sizeof(float));

        int secondChunk = toRead - firstChunk;
        if (secondChunk > 0)
            std::memcpy(data + firstChunk, src, static_cast<size_t>(secondChunk) * sizeof(float));

        readPosition = (readPosition + toRead) % bufferSize;
        return toRead;
    }

    /** Get available samples for a given read position. */
    int getAvailableAt(int readPosition) const
    {
        int wp = writePos.load(std::memory_order_acquire);
        return getAvailableAt(wp, readPosition);
    }

    void reset()
    {
        writePos.store(0, std::memory_order_relaxed);
        totalWritten.store(0, std::memory_order_relaxed);
        buffer.clear();
    }

    int getBufferSize() const { return bufferSize; }

    /** Total samples ever written, monotonic since the last setSize/reset.

        The position alone cannot tell a consumer it has been LAPPED: the write
        head wraps, so a producer that has run a whole buffer ahead looks
        exactly like one that has not moved, and getAvailableAt() happily
        returns a small number computed from stale data. The consumer then reads
        samples the producer has already overwritten and hears a glitch with no
        counter anywhere to explain it.

        Against this counter a consumer that remembers how much it has taken can
        test `totalWritten - consumed > getBufferSize() - block` and know it has
        been overrun, in time to resync and count the event. Monotonic and
        additive: it never wraps in any practical session (at 192 kHz it lasts
        about three million years), and it is only ever incremented, so a torn
        read cannot invent a smaller value than the truth.

        Relaxed is enough. The consumer's synchronisation with the audio comes
        from the acquire load of the write position in readWithPosition; this
        value is only compared against the consumer's own running total. */
    std::uint64_t getTotalWritten() const noexcept
    {
        return totalWritten.load(std::memory_order_relaxed);
    }

private:
    juce::AudioBuffer<float> buffer;
    int bufferSize = 0;
    std::atomic<int> writePos{0};
    std::atomic<std::uint64_t> totalWritten{0};

    int getAvailableAt(int wp, int rp) const
    {
        if (wp >= rp)
            return wp - rp;
        else
            return bufferSize - rp + wp;
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SharedInputRingBuffer)
};

} // namespace spatcore::rt

// Extraction-compat aliases — app code migrates to qualified names later.
using spatcore::rt::SharedInputRingBuffer;
