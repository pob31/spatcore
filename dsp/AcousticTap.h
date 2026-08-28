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

    A cell whose `hfGainDb` is effectively zero skips the biquad entirely rather
    than running it at unity. That keeps the default case cheap AND keeps it
    sample-identical to the plain `dest[s] += src[s] * level` matrix this
    replaced, which is what the null test asserts.
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
                                float* dest)
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
