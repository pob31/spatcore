#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include "AcousticTap.h"
#include <cstdint>
#include <cstring>
#include <vector>

namespace spatcore::dsp {

//==============================================================================
/**
    Source -> node acoustic send matrix: one delay line per source, one acoustic
    tap cell per (source, node). Today's owner is the reverb feed
    (reverb/ReverbFeedThread.h, nodes = reverb nodes); the effects engine reuses
    it unchanged with nodes = effects channels (effects-channels plan, section
    2.2). reverb/ReverbSendMatrix.h keeps the old name as an alias.

    Pure computation - no thread, no ring-buffer plumbing - so it is directly
    unit-testable. The owner drives it once per batch. The mirror image of
    reverb/ReverbReturnProcessor, which does node -> speaker.

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
class AcousticSendMatrix
{
public:
    AcousticSendMatrix() = default;

    //==========================================================================
    /** @param historySeconds  how far back a tap may read. The default is the
                                direct path's rule and what every caller wanted
                                before the effects engine: positions are +/-50 m
                                per axis, so a source-to-node path tops out near
                                505 ms before the latency trims are added. A
                                caller that knows its geometry is tighter can
                                ask for less and save the memory, which matters
                                once there is a second matrix: one line per
                                source per matrix is 52 MB at 136 sources and
                                96 kHz.

                                It must exceed the longest delay you will ask
                                for BY AT LEAST ONE BLOCK. writeInputs fills the
                                current block before the taps read, so a delay
                                within a block of the line length reads samples
                                from the block just written rather than from the
                                past. A tap asking for more than the line holds
                                is clamped to one sample short of it, which is
                                inside that degenerate zone - the clamp keeps it
                                in bounds, it does not keep it meaningful. */
    void prepare (double newSampleRate, int numSources, int numNodes,
                  double historySeconds = 1.0)
    {
        sampleRate = newSampleRate;
        preparedSources = juce::jmax (0, numSources);
        preparedNodes = juce::jmax (0, numNodes);

        // See the historySeconds note above for why one second is the default.
        const double clampedHistory = historySeconds > 0.0 ? historySeconds : 1.0;
        delayBufferLength = juce::jmax (2, (int) (newSampleRate * clampedHistory));
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

    /** Sum a range of sources into one node's feed.

        @param levels   linear send gain,  [source * stride + node]
        @param delaysMs send delay in ms,  same indexing (may be nullptr)
        @param hfDb     send HF shelf dB,  same indexing (may be nullptr)
        @param stride   the matrix row stride - the ENGINE's max node count,
                        which is not the live node count this was prepared with
        @param srcBegin first source row to sum
        @param srcEnd   one past the last, or -1 for "all the rest"
        @param clearDest true overwrites dest, false ACCUMULATES onto it

        The default arguments are the whole range and overwrite, which is what
        the reverb feed has always done.

        The range exists for the effects engine's loop guard. An effects channel
        is fed by inputs AND by other effects channels, and a runaway can only
        come from the second group, so the guard has to be able to attenuate
        that part alone. Rendering the two groups into one buffer makes that
        impossible; rendering them as two passes over disjoint source rows makes
        it a multiply. The passes touch disjoint tap cells, so each cell's
        smoother still advances exactly once per batch and the result is
        bit-identical to one full-range pass. */
    void computeNodeFeed (float* dest, int numSamples, int nodeIdx,
                          const float* levels, const float* delaysMs, const float* hfDb,
                          int stride,
                          int srcBegin = 0, int srcEnd = -1, bool clearDest = true)
    {
        if (dest == nullptr || numSamples <= 0)
            return;

        if (clearDest)
            juce::FloatVectorOperations::clear (dest, numSamples);

        if (levels == nullptr || nodeIdx < 0 || nodeIdx >= preparedNodes || ! isPrepared())
            return;

        const int begin = juce::jmax (0, srcBegin);
        const int end = (srcEnd < 0) ? preparedSources : juce::jmin (srcEnd, preparedSources);

        for (int src = begin; src < end; ++src)
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

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AcousticSendMatrix)
};

} // namespace spatcore::dsp
