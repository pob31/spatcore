#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include "../dsp/AcousticTap.h"
#include <cstdint>
#include <cstring>
#include <vector>

namespace spatcore::reverb {

//==============================================================================
/**
    Input -> reverb-node send matrix: one delay line per source, one acoustic
    tap cell per (source, node).

    Pure computation - no thread, no ring-buffer plumbing - so it is directly
    unit-testable. ReverbFeedThread owns one and drives it once per batch. The
    mirror image of ReverbReturnProcessor, which does node -> speaker.

    Per batch:

        writeInputs (inputBlocks, numSamples)            // every source, once
        for each node (one node per worker is safe):
            computeNodeFeed (dest, n, node, levels, delaysMs, hfDb, stride)
        advance (numSamples)                             // once, after every node

    Thread-safety of the sweep: computeNodeFeed for node A and node B touch
    disjoint tap cells (column A vs column B), write disjoint destinations, and
    only READ the delay lines, which writeInputs filled before the sweep. The
    read pointer accessor is const and side-effect free, unlike getWritePointer.
*/
class ReverbSendMatrix
{
public:
    ReverbSendMatrix() = default;

    //==========================================================================
    void prepare (double newSampleRate, int numSources, int numNodes)
    {
        sampleRate = newSampleRate;
        preparedSources = juce::jmax (0, numSources);
        preparedNodes = juce::jmax (0, numNodes);

        // One second per source, the direct path's rule. Positions are +/-50 m
        // per axis, so a source->node path tops out near 505 ms before the
        // +/-100 ms reverbDelayLatency and the latency terms are added.
        delayBufferLength = juce::jmax (2, (int) (newSampleRate * 1.0));
        delayBuffer.setSize (preparedSources, delayBufferLength);
        delayBuffer.clear();
        writePosition = 0;
        sampleCounter = 0;

        const size_t cells = (size_t) preparedSources * (size_t) preparedNodes;
        const int windowSamples = juce::jmax (2, (int) (newSampleRate * 0.010));
        tapCells.assign (cells, AcousticTapCell{});
        for (auto& c : tapCells)
            c.prepare (newSampleRate, windowSamples);
    }

    void reset()
    {
        delayBuffer.clear();
        writePosition = 0;
        sampleCounter = 0;
        for (auto& c : tapCells)
            c.reset();
    }

    bool isPrepared() const noexcept
    {
        return preparedSources > 0 && preparedNodes > 0 && delayBufferLength > 2;
    }

    int getPreparedSources() const noexcept { return preparedSources; }
    int getPreparedNodes() const noexcept { return preparedNodes; }

    //==========================================================================
    /** Copy this batch of every source into its delay line at the write head.
        The head stays at the batch START until advance(), so the sweep that
        follows reads this batch at zero delay and earlier batches further back. */
    void writeInputs (const juce::AudioBuffer<float>& inputBlocks, int numSamples)
    {
        if (delayBufferLength <= 0 || numSamples <= 0)
            return;

        const int first = juce::jmin (numSamples, delayBufferLength - writePosition);
        const int second = numSamples - first;
        const int n = juce::jmin (preparedSources, inputBlocks.getNumChannels());

        for (int ch = 0; ch < n; ++ch)
        {
            const float* src = inputBlocks.getReadPointer (ch);
            float* dst = delayBuffer.getWritePointer (ch);

            std::memcpy (dst + writePosition, src, (size_t) first * sizeof (float));
            if (second > 0)
                std::memcpy (dst, src + first, (size_t) second * sizeof (float));
        }
    }

    /** Sum every active source into one node's feed. OVERWRITES dest.

        @param levels   linear send gain,  [source * stride + node]
        @param delaysMs send delay in ms,  same indexing (may be nullptr)
        @param hfDb     send HF shelf dB,  same indexing (may be nullptr)
        @param stride   the matrix row stride - the ENGINE's max node count,
                        which is not the live node count this was prepared with
    */
    void computeNodeFeed (float* dest, int numSamples, int nodeIdx,
                          const float* levels, const float* delaysMs, const float* hfDb,
                          int stride)
    {
        if (dest == nullptr || numSamples <= 0)
            return;

        juce::FloatVectorOperations::clear (dest, numSamples);

        if (levels == nullptr || nodeIdx < 0 || nodeIdx >= preparedNodes || ! isPrepared())
            return;

        for (int src = 0; src < preparedSources; ++src)
        {
            const int matrixIdx = src * stride + nodeIdx;
            const float level = levels[matrixIdx];

            if (level <= 0.0001f)
                continue;

            const size_t cell = (size_t) src * (size_t) preparedNodes + (size_t) nodeIdx;
            const float delayMs = (delaysMs != nullptr) ? delaysMs[matrixIdx] : 0.0f;
            const float shelfDb = (hfDb != nullptr) ? hfDb[matrixIdx] : 0.0f;

            processAcousticTap (tapCells[cell],
                                delayBuffer.getReadPointer (src), delayBufferLength,
                                writePosition, sampleCounter, numSamples,
                                delayMs, shelfDb, level, sampleRate,
                                dest);
        }
    }

    /** Move the write head past this batch. Call once, after every node. */
    void advance (int numSamples)
    {
        if (delayBufferLength > 0)
            writePosition = (writePosition + numSamples) % delayBufferLength;
        sampleCounter += numSamples;
    }

private:
    double sampleRate = 48000.0;
    int preparedSources = 0;
    int preparedNodes = 0;

    juce::AudioBuffer<float> delayBuffer;   // one line per source
    int delayBufferLength = 0;
    int writePosition = 0;
    std::int64_t sampleCounter = 0;

    std::vector<AcousticTapCell> tapCells;   // [source * preparedNodes + node]

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ReverbSendMatrix)
};

} // namespace spatcore::reverb

// Extraction-compat alias — app code migrates to qualified names later.
using spatcore::reverb::ReverbSendMatrix;
