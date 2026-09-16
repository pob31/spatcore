#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace spatcore::dsp
{

/**
    Mono fractional delay line: power-of-two ring, 2-tap linear interpolation.

    The interpolation is deliberately the SAME arithmetic the direct WFS path
    runs (wfs/InputBufferProcessor.h: wrap the read position into range, floor
    it, weight the two neighbours by the fraction). A delay of 141.6 samples
    therefore reads 0.6*x[n-142] + 0.4*x[n-141], matching dsp/AcousticTap.h, so
    a module and a send never disagree about where a sample is. Higher-order
    interpolation would sound better and would break that agreement; if it is
    ever wanted, it belongs everywhere at once.

    The ring is a power of two so the wrap is a mask rather than a modulo. For
    the non-negative indices used here the two are identical, which a test pins
    against a hand-rolled modulo reference.

    Convention: write() then readLinear(d) gives a delay of d measured from the
    sample just written, so readLinear(0) IS that sample. A feedback loop that
    must read before it writes asks for d - 1 instead. Getting this backwards
    costs one sample of delay, which is inaudible and wrong.

    prepare() allocates; nothing else does. One instance, one audio thread.
*/
class FractionalDelayLine
{
public:
    FractionalDelayLine() = default;

    /** Allocates a ring big enough for maxDelaySamples plus the interpolation
        partner. Not realtime-safe; call it from prepare(). */
    void prepare (int maxDelaySamples)
    {
        const int needed = (maxDelaySamples > 1 ? maxDelaySamples : 1) + 2;

        int length = 2;
        while (length < needed)
            length *= 2;

        buffer.assign (static_cast<size_t> (length), 0.0f);
        mask = length - 1;
        writePos = 0;
    }

    void reset() noexcept
    {
        std::fill (buffer.begin(), buffer.end(), 0.0f);
        writePos = 0;
    }

    int getLength() const noexcept            { return static_cast<int> (buffer.size()); }
    int getMaxDelaySamples() const noexcept   { return getLength() - 2; }
    bool isPrepared() const noexcept          { return buffer.size() >= 2; }

    void write (float x) noexcept
    {
        buffer[static_cast<size_t> (writePos)] = x;
        writePos = (writePos + 1) & mask;
    }

    /** Linear interpolation between the two samples straddling the delay.
        A negative, NaN or over-long delay is clamped rather than read out of
        bounds. */
    float readLinear (float delaySamples) const noexcept
    {
        const int length = getLength();

        // Negated comparison so NaN clamps to 0 instead of indexing wildly.
        float d = (delaySamples > 0.0f) ? delaySamples : 0.0f;
        const float maxDelay = static_cast<float> (length - 2);
        if (d > maxDelay)
            d = maxDelay;

        float pos = static_cast<float> (writePos - 1) - d;
        while (pos < 0.0f)
            pos += static_cast<float> (length);

        const int p = static_cast<int> (pos);
        const float frac = pos - static_cast<float> (p);

        const float s1 = buffer[static_cast<size_t> (p & mask)];
        const float s2 = buffer[static_cast<size_t> ((p + 1) & mask)];
        return s1 + frac * (s2 - s1);
    }

    /** Whole-sample read - the interpolation-free path. */
    float readInteger (int delaySamples) const noexcept
    {
        const int length = getLength();
        int d = delaySamples > 0 ? delaySamples : 0;
        if (d > length - 2)
            d = length - 2;

        int p = writePos - 1 - d;
        while (p < 0)
            p += length;

        return buffer[static_cast<size_t> (p & mask)];
    }

private:
    std::vector<float> buffer;
    int mask = 0;
    int writePos = 0;
};

} // namespace spatcore::dsp
