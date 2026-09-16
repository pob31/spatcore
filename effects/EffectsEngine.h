#pragma once

#include <juce_audio_basics/juce_audio_basics.h>

#include "EffectParams.h"
#include "EffectsEngineCore.h"
#include "../rt/AudioWorkgroupCoordinator.h"
#include "../rt/SharedInputRingBuffer.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace spatcore::effects
{

//==============================================================================
/**
    The effects engine's driver thread: a wake flag and a drain loop around
    EffectsEngineCore, shaped exactly like reverb/ReverbFeedThread.

    Everything that could be tested is in the core; this class is the part that
    cannot be, so it is kept to the smallest thing that still works. It owns one
    atomic flag and one loop, and forwards everything else.

    THE WAKE. A release store plus notify() from the audio callback, and on this
    side an acquire load with a 1 ms wait when there is nothing to do. A lost
    wakeup therefore costs one millisecond rather than a stall, and waking with
    nothing ready costs one atomic load. The flag is cleared BEFORE the work, so
    a notify that lands during a batch is not swallowed.

    notify() is WaitableEvent::signal(), which is a SetEvent syscall on Windows
    and a futex wake elsewhere - non-blocking, and exactly what the reverb feed
    already does from the same callback, but it is the one thing besides
    pullReturn that the audio thread does here.

    WHY IT DRAINS RATHER THAN DOING ONE BATCH PER WAKE. The reverb feed
    processes at most one batch per wake and lets any surplus sit in the ring,
    which is harmless for a send that has four blocks of headroom and one
    consumer. Thirty-two chains behind one fork-join sweep can lose a block to a
    scheduling hole and would then never make it up: the backlog would grow
    until the producer lapped the cursors. drainAvailable() takes every complete
    block that is waiting and jumps forward rather than working through a
    backlog, and reports how many batches each wake actually needed so that a
    driver which is chronically late is visible instead of merely audible.

    PRIORITY IS THE OWNER'S JOB, as it is for the reverb feed. The owner calls
    startRealtimeThread (RealtimeOptions{}.withApproximateAudioProcessingTime
    (blockSize, sampleRate)) after prepare(). The worker pool raises itself from
    inside AudioParallelFor, using the block period the core passes down.

    LIFETIME. The core holds raw pointers into the consumer's ring buffers and
    the consumer's feed matrices, and none of it is refcounted. release() joins
    this thread before the core drops the ready flag and frees, and the owner
    must call it (or destroy this object) before the rings it was given. A ready
    flag alone is not enough: a flag does not evict a thread already inside
    readWithPosition.
*/
class EffectsEngine : public juce::Thread
{
public:
    using Config = EffectsEngineCore::Config;

    EffectsEngine() : juce::Thread ("EffectsDriver") {}

    ~EffectsEngine() override
    {
        joinDriver();
        core.release();
    }

    /** Optional realtime workgroup to (re)join from this thread (macOS).
        Set this BEFORE prepare() - the sweep's worker pool joins the same
        workgroup and is created there. */
    void setWorkgroupCoordinator (spatcore::rt::AudioWorkgroupCoordinator* c) noexcept
    {
        coordinator = c;
    }

    /** Allocates. Joins this thread first, because prepare() rebuilds every
        buffer and restarts the worker pool, and does NOT start it again: the
        owner chooses the priority, exactly as it does for the reverb feed. */
    bool prepare (const Config& config,
                  const std::vector<std::unique_ptr<spatcore::rt::SharedInputRingBuffer>>& sources)
    {
        joinDriver();
        dataReady.store (false, std::memory_order_relaxed);
        return core.prepare (config, sources, coordinator);
    }

    /** Joins this thread, then drops the ready flag and frees. The audio thread
        may call pullReturn throughout and gets silence. */
    void release()
    {
        joinDriver();
        dataReady.store (false, std::memory_order_relaxed);
        core.release();
    }

    /** AUDIO THREAD, once per callback after the render sources have been
        written into their rings. */
    void notifyInputAvailable() noexcept
    {
        dataReady.store (true, std::memory_order_release);
        notify();
    }

    /** AUDIO THREAD. Never blocks, never allocates, silence when not ready. */
    bool pullReturn (int fx, float* dst, int numSamples) noexcept
    {
        return core.pullReturn (fx, dst, numSamples);
    }

    void setMuted (bool muted) noexcept { core.setMuted (muted); }
    void requestClear (int fx = -1) noexcept { core.requestClear (fx); }

    void publishChannelParams (int fx, const EffectChannelParams& p) noexcept
    {
        core.publishChannelParams (fx, p);
    }

    void setFeedMatrices (const float* delaysMs, const float* levels, const float* hfDb,
                          int stride, int numSources, int numEffects) noexcept
    {
        core.setFeedMatrices (delaysMs, levels, hfDb, stride, numSources, numEffects);
    }

    bool isReady() const noexcept { return core.isReady(); }

    /** Microseconds spent on the last batch, and a liveness tick. The rest of
        the telemetry, including everything per channel, is on the core. */
    float getLastBatchUs() const noexcept        { return core.getLastBatchUs(); }
    std::uint32_t getBatchCount() const noexcept { return core.getBatchCount(); }

    /** The engine proper. A consumer that owns its own realtime thread drives
        this directly and never instantiates the wrapper at all. */
    EffectsEngineCore& getCore() noexcept             { return core; }
    const EffectsEngineCore& getCore() const noexcept { return core; }

private:
    /** stopThread's result is the only thing that says the driver really went
        away, and all three callers need it to be true: core.release() frees
        buffers that a thread still inside readWithPosition would be reading.
        Two seconds against a batch of a few milliseconds means a timeout is a
        deadlock rather than a slow machine, so it is worth an assertion instead
        of a discarded bool. */
    void joinDriver()
    {
        const bool stopped = stopThread (2000);
        jassert (stopped);
        juce::ignoreUnused (stopped);
    }

    void run() override
    {
        // The workgroup token must live on, and be destroyed on, this thread,
        // so it is a local of run() rather than a member.
        juce::WorkgroupToken wgToken;
        std::uint32_t wgSeenGeneration = 0;

        while (! threadShouldExit())
        {
            if (coordinator != nullptr)
                coordinator->joinIfChanged (wgToken, wgSeenGeneration);

            if (! dataReady.load (std::memory_order_acquire))
            {
                wait (1);
                continue;
            }

            dataReady.store (false, std::memory_order_relaxed);
            core.drainAvailable();
        }
    }

    EffectsEngineCore core;
    spatcore::rt::AudioWorkgroupCoordinator* coordinator = nullptr;
    std::atomic<bool> dataReady { false };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (EffectsEngine)
};

} // namespace spatcore::effects
