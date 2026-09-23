#pragma once

#include <algorithm>
#include <vector>

namespace spatcore::effects
{

/**
    Mono delay line for the reverb models' tanks, lines and early-reflection
    ring: an explicit-length ring with whole-sample and 4-point Hermite reads.

    Why not dsp::FractionalDelayLine. That class interpolates linearly on
    purpose - the direct WFS path does, and a module and a send must agree on
    where a sample is. A reverb tank is neither: it is a feedback loop, and in
    a loop the interpolator's high-frequency loss is paid on every pass. At a
    fraction of 0.5 a linear read loses 0.69 dB at fs/8 and 3.0 dB at fs/4 per
    pass; a Catmull-Rom read loses 0.07 dB and 1.07 dB. A sixteen-line hall
    recirculates about 25 times a second, so linear reads would quietly eat
    some 9 dB a second at 6 kHz - a decay nobody asked for, varying with the
    modulation depth. Catmull-Rom is also STATELESS and PASSIVE (|H| <= 1 at
    every fraction and frequency; a test sweeps it), which is what a feedback
    loop needs; allpass interpolation would carry state, ring at Nyquist for
    fractions near zero and misbehave under modulation.

    Why not a power-of-two ring. Every reverb instance exists twice (the
    spillover pool), and a line of 4200 samples would round up to 8192. The
    wrap is one compare instead of a mask, which costs nothing measurable next
    to the four loads of a Hermite read.

    Convention: a read is "the sample written d writes ago". readInteger(1) is
    the most recent write, so a loop that reads before it writes and asks for
    d gets a loop of exactly d samples; a line that writes first and wants a
    delay of t reads t + 1.

    prepare() allocates; nothing else does. One instance, one audio thread.
*/
class ReverbDelayLine
{
public:
    ReverbDelayLine() = default;

    /** Allocates a ring that can serve readInteger() up to maxDelaySamples and
        readHermite() up to maxDelaySamples as well (the kernel's two extra
        neighbours are included). Not realtime-safe. */
    void prepare (int maxDelaySamples)
    {
        const int maxDelay = maxDelaySamples > 2 ? maxDelaySamples : 2;
        size = maxDelay + 3;
        buffer.assign (static_cast<size_t> (size), 0.0f);
        writePos = 0;
    }

    void reset() noexcept
    {
        std::fill (buffer.begin(), buffer.end(), 0.0f);
        writePos = 0;
    }

    bool isPrepared() const noexcept          { return size > 0; }

    /** The longest delay either read can serve. */
    int getMaxDelaySamples() const noexcept   { return size - 3; }

    void write (float x) noexcept
    {
        buffer[static_cast<size_t> (writePos)] = x;
        if (++writePos >= size)
            writePos = 0;
    }

    /** The sample written `delay` writes ago, delay clamped to [1, max]. */
    float readInteger (int delay) const noexcept
    {
        int d = delay > 1 ? delay : 1;
        if (d > size - 3)
            d = size - 3;

        int p = writePos - d;
        if (p < 0)
            p += size;

        return buffer[static_cast<size_t> (p)];
    }

    /** Catmull-Rom (4-point Hermite) read at a fractional delay, clamped to
        [2, max] so all four neighbours exist. A NaN delay lands on 2 rather
        than indexing wildly. At an integer delay it returns the stored value:
        every term but the constant is multiplied by a zero fraction (a stored
        negative zero can come back positive - the only bit it may change). */
    float readHermite (float delay) const noexcept
    {
        const float maxDelay = static_cast<float> (size - 3);
        float d = (delay > 2.0f) ? delay : 2.0f;
        if (d > maxDelay)
            d = maxDelay;

        const int i = static_cast<int> (d);
        const float f = d - static_cast<float> (i);

        // Along increasing delay: ym1 is one sample NEWER than y0, y1 and y2
        // are older. The kernel is symmetric, so the naming only has to be
        // consistent.
        int p0 = writePos - i;
        if (p0 < 0)
            p0 += size;

        int pm1 = p0 + 1;
        if (pm1 >= size)
            pm1 -= size;

        int p1 = p0 - 1;
        if (p1 < 0)
            p1 += size;

        int p2 = p1 - 1;
        if (p2 < 0)
            p2 += size;

        const float ym1 = buffer[static_cast<size_t> (pm1)];
        const float y0  = buffer[static_cast<size_t> (p0)];
        const float y1  = buffer[static_cast<size_t> (p1)];
        const float y2  = buffer[static_cast<size_t> (p2)];

        const float c1 = 0.5f * (y1 - ym1);
        const float c2 = ym1 - 2.5f * y0 + 2.0f * y1 - 0.5f * y2;
        const float c3 = 0.5f * (y2 - ym1) + 1.5f * (y0 - y1);

        return ((c3 * f + c2) * f + c1) * f + y0;
    }

private:
    std::vector<float> buffer;
    int size = 0;
    int writePos = 0;
};

} // namespace spatcore::effects
