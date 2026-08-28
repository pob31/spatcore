#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include "../rt/AudioParallelFor.h"
#include "../dsp/AcousticTap.h"
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace spatcore::reverb {

//==============================================================================
/**
    Reverb return -> speaker distribution.

    A node return is an ambient source sitting at its return position, so the
    path from it to each speaker is a real acoustic path: propagation delay
    (with parallax, and minimal-latency aligned), distance attenuation, and
    air absorption from the speaker HF damping. The calculation engine has
    always produced all three matrices; this is what consumes them.

    Usage per block, from the audio callback:

        for each node:  pushNodeReturn (node, wetData, numSamples)
        once:           mixToOutputs (buffer, start, n, nodes, outs, levels, delays, hf, stride)

    pushNodeReturn takes the wet signal exactly where the old scalar matrix
    took it: post node trim, post SR-ratio upsample, device rate.

    The mix parallelises across OUTPUT channels, not nodes. Each worker owns a
    slice of speakers and sweeps every node for them, so no two workers ever
    touch the same output sample and no per-node scratch buffer is needed.
*/
class ReverbReturnProcessor
{
public:
    ReverbReturnProcessor() = default;
    ~ReverbReturnProcessor() { mixPool.shutdown(); }

    //==========================================================================
    void prepare (double newSampleRate, int maxBlockSize, int numNodes, int numOutputs,
                  AudioWorkgroupCoordinator* coordinator = nullptr)
    {
        sampleRate = newSampleRate;
        preparedNodes = juce::jmax (0, numNodes);
        preparedOutputs = juce::jmax (0, numOutputs);

        // One second per node, the same rule the direct path uses. The return
        // delay is min-delay aligned and parallax-relative, so it is far
        // smaller than that in practice; the headroom is what keeps a
        // pathological geometry from wrapping the read head.
        delayBufferLength = juce::jmax (2, (int) (newSampleRate * 1.0));
        returnDelayBuffer.setSize (preparedNodes, delayBufferLength);
        returnDelayBuffer.clear();
        writePosition = 0;
        sampleCounter = 0;

        const size_t cells = (size_t) preparedNodes * (size_t) preparedOutputs;
        const int windowSamples = juce::jmax (2, (int) (newSampleRate * 0.010));
        tapCells.assign (cells, AcousticTapCell{});
        for (auto& c : tapCells) c.prepare (newSampleRate, windowSamples);

        {
            mixPool.shutdown();
            int workers = 0;
            if (cells >= 256)
            {
                int hwThreads = static_cast<int> (std::thread::hardware_concurrency());
                workers = juce::jmin (hwThreads / 2 - 1, preparedOutputs - 1);
                workers = juce::jlimit (0, 3, workers);
            }
            double blockMs = newSampleRate > 0.0 ? (maxBlockSize / newSampleRate) * 1000.0 : 0.0;
            mixPool.prepare (workers, blockMs, blockMs, coordinator);
        }
    }

    void reset()
    {
        returnDelayBuffer.clear();
        writePosition = 0;
        sampleCounter = 0;
        for (auto& c : tapCells) c.reset();
    }

    bool isPrepared() const noexcept { return preparedNodes > 0 && preparedOutputs > 0 && delayBufferLength > 2; }
    int getPreparedNodes() const noexcept { return preparedNodes; }
    int getPreparedOutputs() const noexcept { return preparedOutputs; }

    //==========================================================================
    /** Publish one node wet block into its delay line. The write head stays at
        the block START until mixToOutputs() has taken every tap. */
    void pushNodeReturn (int nodeIndex, const float* data, int numSamples)
    {
        if (nodeIndex < 0 || nodeIndex >= preparedNodes || data == nullptr || delayBufferLength <= 0)
            return;

        const int first = juce::jmin (numSamples, delayBufferLength - writePosition);
        const int second = numSamples - first;

        float* dst = returnDelayBuffer.getWritePointer (nodeIndex);
        std::memcpy (dst + writePosition, data, (size_t) first * sizeof (float));
        if (second > 0)
            std::memcpy (dst, data + first, (size_t) second * sizeof (float));
    }

    /** Write silence into every delay line and advance the write head. Used for
        the post-mute path, so lifting the mute cannot tap stale wet audio at a
        non-zero return delay. */
    void skipBlock (int numSamples)
    {
        if (! isPrepared() || numSamples <= 0)
            return;

        const int first = juce::jmin (numSamples, delayBufferLength - writePosition);
        const int second = numSamples - first;

        for (int n = 0; n < preparedNodes; ++n)
        {
            float* dst = returnDelayBuffer.getWritePointer (n);
            juce::FloatVectorOperations::clear (dst + writePosition, first);
            if (second > 0)
                juce::FloatVectorOperations::clear (dst, second);
        }

        advance (numSamples);
    }

    /** Accumulate every pushed node into the speaker buffer, then advance the
        write head. Call exactly once per block, after all pushNodeReturn calls.

        @param levels   linear return gain,  [node * stride + output]
        @param delaysMs return delay in ms,  same indexing (may be nullptr)
        @param hfDb     return HF shelf dB,  same indexing (may be nullptr)
    */
    void mixToOutputs (juce::AudioBuffer<float>& outputBuffer, int startSample, int numSamples,
                       int numActiveNodes, int numActiveOutputs,
                       const float* levels, const float* delaysMs, const float* hfDb, int stride)
    {
        if (! isPrepared() || levels == nullptr || numSamples <= 0)
            return;

        const int nodes = juce::jmin (numActiveNodes, preparedNodes);
        const int outs = juce::jmin (juce::jmin (numActiveOutputs, preparedOutputs),
                                     outputBuffer.getNumChannels());

        if (nodes <= 0 || outs <= 0)
        {
            advance (numSamples);
            return;
        }

        mixPool.parallelFor (outs, [&] (int outIdx)
        {
            mixOneOutput (outputBuffer, startSample, numSamples, outIdx, nodes,
                          levels, delaysMs, hfDb, stride);
        });

        advance (numSamples);
    }

private:
    void advance (int numSamples)
    {
        if (delayBufferLength > 0)
            writePosition = (writePosition + numSamples) % delayBufferLength;
        sampleCounter += numSamples;
    }

    void mixOneOutput (juce::AudioBuffer<float>& outputBuffer, int startSample, int numSamples,
                       int outIdx, int nodes,
                       const float* levels, const float* delaysMs, const float* hfDb, int stride)
    {
        float* outputData = outputBuffer.getWritePointer (outIdx, startSample);

        for (int nodeIdx = 0; nodeIdx < nodes; ++nodeIdx)
        {
            const int matrixIdx = nodeIdx * stride + outIdx;
            const float returnLevel = levels[matrixIdx];

            if (returnLevel <= 0.0001f)
                continue;

            const size_t cell = (size_t) nodeIdx * (size_t) preparedOutputs + (size_t) outIdx;

            const float delayMs = (delaysMs != nullptr) ? delaysMs[matrixIdx] : 0.0f;
            const float shelfDb = (hfDb != nullptr) ? hfDb[matrixIdx] : 0.0f;

            processAcousticTap (tapCells[cell],
                                returnDelayBuffer.getReadPointer (nodeIdx), delayBufferLength,
                                writePosition, sampleCounter, numSamples,
                                delayMs, shelfDb, returnLevel, sampleRate,
                                outputData);
        }
    }

    //==========================================================================
    double sampleRate = 48000.0;
    int preparedNodes = 0;
    int preparedOutputs = 0;

    juce::AudioBuffer<float> returnDelayBuffer;   // one delay line per node
    int delayBufferLength = 0;
    int writePosition = 0;
    std::int64_t sampleCounter = 0;

    std::vector<AcousticTapCell> tapCells;   // [node * preparedOutputs + output]

    AudioParallelFor mixPool;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ReverbReturnProcessor)
};

} // namespace spatcore::reverb

// Extraction-compat alias — app code migrates to qualified names later.
using spatcore::reverb::ReverbReturnProcessor;
