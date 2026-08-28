#pragma once

#include "DelayTargetSmoother.h"
#include "WFSHighShelfFilter.h"
#include <cstdint>

namespace spatcore::dsp {

//==============================================================================
/**
    One (source -> destination) acoustic tap: smoothed fractional propagation
    delay, air-absorption high shelf, and level, accumulated into a destination
    block.

    This is the cell the direct WFS path has always run per (input, output)
    pair — see spatcore/wfs/InputBufferProcessor.h, whose inner loop this
    mirrors. It is factored out here because the reverb send
    (spatcore/reverb/ReverbFeedThread.h, per source/node) and the reverb return
    (spatcore/reverb/ReverbReturnProcessor.h, per node/speaker) need exactly the
    same thing, and three hand-copied versions of a fractional-delay read is
    three places for the interpolation to drift apart.

    JUCE-free on purpose: raw pointers only, so it is directly unit-testable.
*/
struct AcousticTapCell
{
    DelayTargetSmoother smoother;
    WFSHighShelfFilter  shelf;
    bool                shelfEngaged = false;

    /** Call once per sample-rate change, before any processing. */
    void prepare (double sampleRate, int smoothingWindowSamples)
    {
        smoother.prepare (smoothingWindowSamples);
        shelf.prepare (sampleRate);
        shelf.setGainDb (0.0f);
        shelfEngaged = false;
    }

    void reset()
    {
        smoother.reset();
        shelf.reset();
        shelfEngaged = false;
    }
};

//==============================================================================
/**
    Accumulate one tap into `dest`.

    @param cell             per-(source, destination) state
    @param delayLine        the source ring buffer
    @param delayLineLength  its length in samples
    @param blockStartPos    ring index this block's first sample was written at
    @param sampleCounter    monotonic sample index of this block's first sample
    @param numSamples       block length
    @param delayMs          geometry-derived propagation delay, milliseconds
    @param hfGainDb         air-absorption shelf gain, dB (<= 0; 0 = bypass)
    @param level            linear send/return gain
    @param sampleRate       device sample rate
    @param dest             accumulation target, numSamples long
    @param allowFastPath    tests pass false to force the general loop and use
                            it as the oracle for the steady-state path

    A cell whose `hfGainDb` is effectively zero skips the biquad entirely rather
    than running it at unity. That keeps the default case cheap AND keeps it
    sample-identical to the plain `dest[s] += src[s] * level` matrix this
    replaced, which is what the null test asserts.

    Two loops, one meaning: when the smoothed delay has settled to a constant
    (a static source between 50 Hz matrix updates - most blocks), the ring is
    walked in contiguous runs at a fixed fractional offset, which vectorises and
    collapses to a plain MAC at a whole-sample delay with no shelf. The general
    loop runs while the delay is moving or a teleport envelope is open.
    `DelayTargetSmoother::isSteadyFrom` is conservative, so the two paths agree
    sample for sample and there is no seam to hear at the crossover.
*/
inline void processAcousticTap (AcousticTapCell& cell,
                                const float* delayLine,
                                int delayLineLength,
                                int blockStartPos,
                                std::int64_t sampleCounter,
                                int numSamples,
                                float delayMs,
                                float hfGainDb,
                                float level,
                                double sampleRate,
                                float* dest,
                                bool allowFastPath = true)
{
    if (delayLine == nullptr || dest == nullptr || delayLineLength <= 1 || numSamples <= 0)
        return;

    // Delay target in samples, clamped exactly like the direct path
    float delaySamples = (delayMs / 1000.0f) * static_cast<float> (sampleRate);
    if (! (delaySamples > 0.0f))
        delaySamples = 0.0f;                                  // also catches NaN
    if (delaySamples >= static_cast<float> (delayLineLength))
        delaySamples = static_cast<float> (delayLineLength - 1);

    cell.smoother.observe (delaySamples, sampleCounter);

    const bool useShelf = (hfGainDb < -0.005f);
    if (useShelf)
    {
        if (! cell.shelfEngaged)
        {
            cell.shelf.reset();          // re-engaging: do not ring out stale state
            cell.shelfEngaged = true;
        }
        cell.shelf.setGainDb (hfGainDb);
    }
    else
    {
        cell.shelfEngaged = false;
    }

    // ---- Steady state: constant delay, unity gain, no teleport in flight ----
    //
    // A static source between 50 Hz matrix updates sits here for most blocks,
    // and it is worth a lot: the general loop below spends most of its time in
    // the per-sample smoothedAt() call and the two modulos, none of which
    // depend on the sample when the delay is not moving. Walking the ring in
    // contiguous runs instead lets the compiler vectorise, and at a whole-sample
    // delay with no shelf it collapses to a plain multiply-accumulate.
    //
    // isSteadyFrom() is conservative: a true answer means the general path
    // would have produced exactly this constant, so the two agree sample for
    // sample and there is no fast/slow seam to hear.
    float steadyDelay = 0.0f;
    if (allowFastPath && cell.smoother.isSteadyFrom (sampleCounter, steadyDelay))
    {
        // Read position of the block's first sample, in the SAME floor
        // convention as the general loop: pos = blockStart - delay, wrapped
        // into [0, len); p is its integer part and frac the weight on p+1.
        // A delay of 141.6 therefore reads 0.6*x[n-142] + 0.4*x[n-141] - the
        // sample 141.6 back, not the one 140.4 back.
        float pos0 = static_cast<float> (blockStartPos) - steadyDelay;
        while (pos0 < 0.0f)
            pos0 += static_cast<float> (delayLineLength);

        int p = static_cast<int> (pos0);
        const float frac = pos0 - static_cast<float> (p);
        if (p >= delayLineLength)          // float rounding can land exactly on len
            p -= delayLineLength;

        int s = 0;

        if (! useShelf && frac == 0.0f)
        {
            // Whole-sample delay, no air absorption: a straight MAC.
            while (s < numSamples)
            {
                const int run = (numSamples - s) < (delayLineLength - p)
                                    ? (numSamples - s) : (delayLineLength - p);
                for (int k = 0; k < run; ++k)
                    dest[s + k] += delayLine[p + k] * level;

                s += run;
                p += run;
                if (p >= delayLineLength)
                    p -= delayLineLength;
            }
            return;
        }

        while (s < numSamples)
        {
            // -1 so the k+1 interpolation partner stays inside this run.
            int run = delayLineLength - p - 1;
            if (run > numSamples - s)
                run = numSamples - s;

            if (run <= 0)
            {
                // The one sample whose partner is across the wrap.
                const float a = delayLine[p];
                const float b = delayLine[0];
                float v = a + frac * (b - a);
                if (useShelf)
                    v = cell.shelf.processSample (v);
                dest[s] += v * level;

                ++s;
                p = 0;
                continue;
            }

            if (useShelf)
            {
                for (int k = 0; k < run; ++k)
                {
                    const float a = delayLine[p + k];
                    const float b = delayLine[p + k + 1];
                    dest[s + k] += cell.shelf.processSample (a + frac * (b - a)) * level;
                }
            }
            else
            {
                for (int k = 0; k < run; ++k)
                {
                    const float a = delayLine[p + k];
                    const float b = delayLine[p + k + 1];
                    dest[s + k] += (a + frac * (b - a)) * level;
                }
            }

            s += run;
            p += run;
            if (p >= delayLineLength)
                p -= delayLineLength;
        }

        return;
    }

    // ---- General path: the delay is moving, or a teleport envelope is open ----
    for (int s = 0; s < numSamples; ++s)
    {
        const auto smoothed = cell.smoother.smoothedAt (sampleCounter + s);

        float exactReadPos = static_cast<float> (blockStartPos) + static_cast<float> (s) - smoothed.delay;
        while (exactReadPos < 0.0f)
            exactReadPos += static_cast<float> (delayLineLength);

        const int readPos1 = static_cast<int> (exactReadPos) % delayLineLength;
        const int readPos2 = (readPos1 + 1) % delayLineLength;
        const float fraction = exactReadPos - static_cast<float> (static_cast<int> (exactReadPos));

        const float s1 = delayLine[readPos1];
        const float s2 = delayLine[readPos2];
        float v = s1 + fraction * (s2 - s1);

        if (useShelf)
            v = cell.shelf.processSample (v);

        dest[s] += v * level * smoothed.gain;
    }
}

} // namespace spatcore::dsp

// Extraction-compat aliases — app code migrates to qualified names later.
using spatcore::dsp::AcousticTapCell;
using spatcore::dsp::processAcousticTap;
