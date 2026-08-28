#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include "../rt/SharedInputRingBuffer.h"
#include "../rt/AudioParallelFor.h"
#include "../dsp/AcousticTap.h"
#include "ReverbEngine.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace spatcore::reverb {

/**
    Dedicated thread for computing reverb feed sums from input audio.
    Reads from shared input ring buffers, computes the input->reverb send for
    every (source, node) pair, handles downsampling, and pushes to ReverbEngine
    node inputs.

    Each send is a full acoustic path, not a bare gain: the source is read from
    a per-source delay line at the propagation delay the calculation engine
    derived from the geometry, filtered by an air-absorption high shelf, then
    scaled by the send level. That mirrors the direct WFS path
    (spatcore/wfs/InputBufferProcessor.h), which has always done all three.

    Runs one block behind the audio callback (2.67ms at 256/96kHz — imperceptible for reverb).
*/
class ReverbFeedThread : public juce::Thread
{
public:
    ReverbFeedThread() : juce::Thread("ReverbFeed") {}
    ~ReverbFeedThread() override { stopThread(2000); feedPool.shutdown(); }

    /** Optional: realtime workgroup to (re)join from the worker thread (macOS).
        Set this BEFORE prepare() — the send-path worker pool joins the same
        workgroup and is created there. */
    void setWorkgroupCoordinator (AudioWorkgroupCoordinator* c) { workgroupCoordinator = c; }

    /** Wire up the send path.

        @param reverbLevelsPtr   linear send gain,   [input * calcReverbStride + node]
        @param reverbDelaysPtr   send delay in ms,   same indexing (may be nullptr)
        @param reverbHFPtr       send HF shelf dB,   same indexing (may be nullptr)
    */
    void prepare(const std::vector<std::unique_ptr<SharedInputRingBuffer>>& sharedInputs,
                 ReverbEngine* engine,
                 const float* reverbLevelsPtr,
                 const float* reverbDelaysPtr,
                 const float* reverbHFPtr,
                 int calcReverbStride,
                 int numInputCh,
                 int numReverbNodes,
                 int blockSize,
                 int srRatio,
                 double newSampleRate)
    {
        inputBuffers.clear();
        for (auto& buf : sharedInputs)
            inputBuffers.push_back(buf.get());

        reverbEngine = engine;
        reverbLevels = reverbLevelsPtr;
        reverbDelays = reverbDelaysPtr;
        reverbHF = reverbHFPtr;
        reverbStride = calcReverbStride;
        numInputs = numInputCh;
        numRevs = numReverbNodes;
        preparedNumRevs = numReverbNodes;
        processingBlockSize = blockSize;
        reverbSRRatio = srRatio;
        sampleRate = newSampleRate;

        readPositions.assign(inputBuffers.size(), 0);

        // Temp buffer: one row per input channel for batch reading
        inputBlocks.setSize(numInputCh, blockSize);

        feedBuffer.setSize(numReverbNodes, blockSize);

        int dsBlockSize = (srRatio > 1) ? (blockSize / srRatio) : blockSize;
        downsampleBuffer.setSize(numReverbNodes, dsBlockSize);

        // Per-source delay lines. One second matches the direct path rule and
        // covers the worst geometry the parameter ranges allow (positions are
        // +/-50 m on each axis, so the source->node path tops out near 505 ms,
        // before the +/-100 ms reverbDelayLatency and the latency terms).
        delayBufferLength = juce::jmax(2, (int)(newSampleRate * 1.0));
        feedDelayBuffer.setSize(numInputCh, delayBufferLength);
        feedDelayBuffer.clear();
        writePosition = 0;
        sampleCounter = 0;

        // One (source, node) tap cell each: delay smoother + air-absorption
        // shelf, ~10 ms box window like the direct path per-output smoothers.
        const size_t cells = (size_t)juce::jmax(0, numInputCh) * (size_t)juce::jmax(0, numReverbNodes);
        const int windowSamples = juce::jmax(2, (int)(newSampleRate * 0.010));
        tapCells.assign(cells, AcousticTapCell{});
        for (auto& c : tapCells) c.prepare(newSampleRate, windowSamples);

        feedRowPtrs.assign((size_t)juce::jmax(0, numReverbNodes), nullptr);   // never grows in run()

        // Node-parallel send computation. Each worker owns one node, so it
        // writes one feedBuffer row and touches only its own cell column; the
        // delay lines are written once before the sweep and read-only during it.
        // Small configurations stay sequential — the fork/join costs more than
        // the work saved.
        {
            feedPool.shutdown();
            int workers = 0;
            if (cells >= 256)
            {
                int hwThreads = static_cast<int>(std::thread::hardware_concurrency());
                workers = juce::jmin(hwThreads / 2 - 1, numReverbNodes - 1);
                workers = juce::jlimit(0, 3, workers);
            }
            double blockMs = newSampleRate > 0.0 ? (blockSize / newSampleRate) * 1000.0 : 0.0;
            feedPool.prepare(workers, blockMs, blockMs, workgroupCoordinator);
        }
    }

    void notifyInputAvailable()
    {
        dataReady.store(true, std::memory_order_release);
        notify();
    }

    void setMuted(bool muted)
    {
        isMuted.store(muted, std::memory_order_relaxed);
    }

    void updateReverbMatrices(const float* newLevelsPtr, const float* newDelaysPtr,
                              const float* newHFPtr, int newStride, int newNumReverbs)
    {
        // Publish the tuple under a brief SpinLock so the worker thread can
        // never observe a torn (pointers, stride, count) update. Matches the
        // pattern used for pendingPreParams / pendingPostParams in
        // ReverbEngine. The worker snapshots once per batch in run(); the
        // hot inner loop never touches this lock.
        juce::SpinLock::ScopedLockType lock (matrixLock);
        reverbLevels = newLevelsPtr;
        reverbDelays = newDelaysPtr;
        reverbHF = newHFPtr;
        reverbStride = newStride;
        numRevs = newNumReverbs;
    }

    //==========================================================================
    // Duty telemetry — always-on (GPU host-path optimization M0). One
    // steady_clock pair per processed batch (~750/s at 96k/128) is noise;
    // relaxed atomics, read by the 20 Hz metering sampler on the message
    // thread. Non-destructive (no reset-on-read).
    //==========================================================================

    /** Microseconds spent computing the last feed batch (read inputs + gated
        mix + downsample + push). 0 until the first batch. */
    float getLastBatchUs() const noexcept
    {
        return lastBatchUs.load (std::memory_order_relaxed);
    }

    /** Batches processed since prepare (wraps; UI uses it as a liveness tick). */
    uint32_t getBatchCount() const noexcept
    {
        return batchCounter.load (std::memory_order_relaxed);
    }

private:
    void run() override
    {
        // Audio workgroup membership: token lives on (and is destroyed on) this thread.
        juce::WorkgroupToken wgToken;
        uint32_t wgSeenGeneration = 0;

        while (!threadShouldExit())
        {
            if (workgroupCoordinator != nullptr)
                workgroupCoordinator->joinIfChanged(wgToken, wgSeenGeneration);

            if (!dataReady.load(std::memory_order_acquire))
            {
                wait(1);
                continue;
            }
            dataReady.store(false, std::memory_order_relaxed);

            // Snapshot the (levels, delays, HF, stride, numRevs) tuple once per
            // batch under the matrix lock. The rest of this batch uses only the
            // locals, so the inner per-sample / per-input / per-node loop runs
            // with zero synchronisation. A matrix update via
            // updateReverbMatrices() becomes visible on the *next* batch — one
            // block of stale values is inaudible for reverb send routing (matrix
            // updates happen at user-interaction cadence, not per block), and
            // the delay smoothers absorb any step.
            const float* levelsSnap;
            const float* delaysSnap;
            const float* hfSnap;
            int strideSnap;
            int numRevsSnap;
            {
                juce::SpinLock::ScopedLockType lock (matrixLock);
                levelsSnap  = reverbLevels;
                delaysSnap  = reverbDelays;
                hfSnap      = reverbHF;
                strideSnap  = reverbStride;
                numRevsSnap = juce::jmin (numRevs, preparedNumRevs);
            }

            if (reverbEngine == nullptr || numRevsSnap <= 0 || numInputs <= 0)
                continue;

            int numSamples = processingBlockSize;

            // Check all channels have enough data
            int minAvail = std::numeric_limits<int>::max();
            for (int ch = 0; ch < numInputs && ch < (int)inputBuffers.size(); ++ch)
                minAvail = juce::jmin(minAvail, inputBuffers[ch]->getAvailableAt(readPositions[ch]));

            if (minAvail < numSamples)
                continue;

            // Duty telemetry: time the real batch work (input read + gated mix
            // + downsample + push). Early-outs above (no data / no engine) are
            // idle, not work, and stay untimed.
            const auto batchStart = std::chrono::steady_clock::now();

            // Read all input channels into temp buffers (one read per channel)
            for (int ch = 0; ch < numInputs && ch < (int)inputBuffers.size(); ++ch)
                inputBuffers[ch]->readWithPosition(readPositions[ch], inputBlocks.getWritePointer(ch), numSamples);

            // Publish this block into the per-source delay lines. Done even when
            // muted, so unmuting does not replay stale history.
            writeInputsToDelayLines(numSamples);

            int pushSamples = numSamples / reverbSRRatio;

            if (isMuted.load(std::memory_order_relaxed))
            {
                // Push silence — reverb tail decays naturally
                for (int revIdx = 0; revIdx < numRevsSnap; ++revIdx)
                {
                    downsampleBuffer.clear(revIdx, 0, pushSamples);
                    reverbEngine->pushNodeInput(revIdx, downsampleBuffer.getReadPointer(revIdx), pushSamples);
                }
            }
            else
            {
                // Compute the sends. One node per work item: delay-tap, shelf
                // and level for every source feeding it.
                // Resolve the feed-row write pointers on THIS thread:
                // AudioBuffer::getWritePointer clears the buffer's isClear flag
                // as a side effect, so calling it from N workers at once is a
                // data race. Each worker then owns one row pointer.
                feedRowPtrs.resize((size_t)numRevsSnap);
                for (int r = 0; r < numRevsSnap; ++r)
                    feedRowPtrs[(size_t)r] = feedBuffer.getWritePointer(r);

                feedPool.parallelFor(numRevsSnap, [&](int revIdx)
                {
                    computeNodeFeed(feedRowPtrs[(size_t)revIdx], numSamples,
                                    revIdx, levelsSnap, delaysSnap, hfSnap, strideSnap);
                });

                // Downsample and push (serial: pushNodeInput notifies the engine)
                for (int revIdx = 0; revIdx < numRevsSnap; ++revIdx)
                {
                    float* feedData = feedBuffer.getWritePointer(revIdx);

                    if (reverbSRRatio > 1)
                    {
                        float* dsData = downsampleBuffer.getWritePointer(revIdx);
                        float invRatio = 1.0f / static_cast<float>(reverbSRRatio);
                        for (int i = 0; i < pushSamples; ++i)
                        {
                            float sum = 0.0f;
                            for (int j = 0; j < reverbSRRatio; ++j)
                                sum += feedData[i * reverbSRRatio + j];
                            dsData[i] = sum * invRatio;
                        }
                        reverbEngine->pushNodeInput(revIdx, dsData, pushSamples);
                    }
                    else
                    {
                        reverbEngine->pushNodeInput(revIdx, feedData, numSamples);
                    }
                }
            }

            writePosition = (writePosition + numSamples) % delayBufferLength;
            sampleCounter += numSamples;

            lastBatchUs.store(std::chrono::duration<float, std::micro>(
                                  std::chrono::steady_clock::now() - batchStart).count(),
                              std::memory_order_relaxed);
            batchCounter.fetch_add(1, std::memory_order_relaxed);
        }
    }

    /** Copy this block into each source delay line at the current write head.
        writePosition stays at the block START until the sends have been taken. */
    void writeInputsToDelayLines(int numSamples)
    {
        if (delayBufferLength <= 0)
            return;

        const int first = juce::jmin(numSamples, delayBufferLength - writePosition);
        const int second = numSamples - first;

        for (int ch = 0; ch < numInputs && ch < feedDelayBuffer.getNumChannels(); ++ch)
        {
            const float* src = inputBlocks.getReadPointer(ch);
            float* dst = feedDelayBuffer.getWritePointer(ch);

            std::memcpy(dst + writePosition, src, (size_t)first * sizeof(float));
            if (second > 0)
                std::memcpy(dst, src + first, (size_t)second * sizeof(float));
        }
    }

    /** One node send bus: sum every active source through its own delay tap
        and air-absorption shelf. */
    void computeNodeFeed(float* feedData, int numSamples, int revIdx,
                         const float* levelsSnap, const float* delaysSnap,
                         const float* hfSnap, int strideSnap)
    {
        if (feedData == nullptr)
            return;

        juce::FloatVectorOperations::clear(feedData, numSamples);

        const int maxCh = juce::jmin(numInputs, feedDelayBuffer.getNumChannels());

        for (int inIdx = 0; inIdx < maxCh; ++inIdx)
        {
            const int matrixIdx = inIdx * strideSnap + revIdx;
            const float feedLevel = levelsSnap[matrixIdx];

            if (feedLevel <= 0.0001f)
                continue;

            const size_t cell = (size_t)inIdx * (size_t)preparedNumRevs + (size_t)revIdx;

            const float delayMs = (delaysSnap != nullptr) ? delaysSnap[matrixIdx] : 0.0f;
            const float hfDb = (hfSnap != nullptr) ? hfSnap[matrixIdx] : 0.0f;

            processAcousticTap (tapCells[cell],
                                feedDelayBuffer.getReadPointer(inIdx), delayBufferLength,
                                writePosition, sampleCounter, numSamples,
                                delayMs, hfDb, feedLevel, sampleRate,
                                feedData);
        }
    }

    std::vector<SharedInputRingBuffer*> inputBuffers;
    std::vector<int> readPositions;
    ReverbEngine* reverbEngine = nullptr;
    AudioWorkgroupCoordinator* workgroupCoordinator = nullptr;

    // (reverbLevels, reverbDelays, reverbHF, reverbStride, numRevs) form a
    // tuple published by updateReverbMatrices() from the message thread and
    // consumed by run() on the worker thread. matrixLock serialises publication
    // so the worker never observes a torn tuple. See updateReverbMatrices() and
    // run() for the read-snapshot pattern.
    const float* reverbLevels = nullptr;
    const float* reverbDelays = nullptr;
    const float* reverbHF = nullptr;
    int reverbStride = 0;
    int numRevs = 0;
    juce::SpinLock matrixLock;

    int numInputs = 0;
    int preparedNumRevs = 0;      // cell-array stride; numRevs is clamped to it
    int processingBlockSize = 256;
    int reverbSRRatio = 1;
    double sampleRate = 48000.0;
    std::atomic<bool> dataReady{false};
    std::atomic<bool> isMuted{false};

    // Always-on duty telemetry (see accessors above)
    std::atomic<float> lastBatchUs{0.0f};
    std::atomic<uint32_t> batchCounter{0};

    juce::AudioBuffer<float> inputBlocks;      // numInputs channels, one block each
    juce::AudioBuffer<float> feedBuffer;       // numReverbs channels, feed sums
    juce::AudioBuffer<float> downsampleBuffer;

    // Send path state: one delay line per source, one (source, node) cell each
    // for the delay smoother and the air-absorption shelf.
    juce::AudioBuffer<float> feedDelayBuffer;
    int delayBufferLength = 0;
    int writePosition = 0;
    std::int64_t sampleCounter = 0;
    std::vector<AcousticTapCell> tapCells;   // [source * preparedNumRevs + node]
    std::vector<float*> feedRowPtrs;         // resolved per batch, feed thread only

    AudioParallelFor feedPool;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ReverbFeedThread)
};

} // namespace spatcore::reverb

// Extraction-compat alias — app code migrates to qualified names later.
using spatcore::reverb::ReverbFeedThread;
