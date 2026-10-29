#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "RealtimeThreadUtil.h"
#include "AudioWorkgroupCoordinator.h"

//==============================================================================
/**
    Lightweight fork-join thread pool for parallelising per-node DSP.

    Usage:
        AudioParallelFor pool;
        pool.prepare (3);                       // 3 workers + calling thread = 4 cores
        pool.parallelFor (numNodes, [&](int n) {
            processNode (n);
        });
        pool.shutdown();                        // joins all workers

    Workers sleep on a condition variable between dispatches.
    Work distribution uses an atomic counter (implicit work-stealing).
    The calling thread participates, then waits for all items to complete.
    Falls back to sequential if count <= 1 or no workers are prepared.
*/
class AudioParallelFor
{
public:
    AudioParallelFor() = default;

    ~AudioParallelFor()
    {
        shutdown();
    }

    //==========================================================================
    /** Create persistent worker threads. Safe to call multiple times (shuts down first).

        @param numWorkers     Number of persistent worker threads to spawn.
        @param periodMs       Audio block duration in ms. When > 0, each worker upgrades itself
                              to a realtime time-constraint policy on macOS (P-core placement).
        @param computationMs  Expected processing time per block in ms (defaults to periodMs).
        @param coordinator    Optional audio workgroup coordinator. When non-null, each worker
                              joins the CoreAudio device's realtime workgroup (macOS) so the
                              kernel schedules them coherently with the rest of the DSP workers.
    */
    void prepare (int numWorkers, double periodMs = 0.0, double computationMs = 0.0,
                  AudioWorkgroupCoordinator* coordinator = nullptr)
    {
        shutdown();

        if (numWorkers <= 0)
            return;

        realtimePeriodMs      = periodMs;
        realtimeComputationMs = computationMs;
        workgroupCoordinator  = coordinator;

        running.store (true, std::memory_order_release);
        numActiveWorkers = numWorkers;

        workers.reserve (static_cast<size_t> (numWorkers));
        for (int i = 0; i < numWorkers; ++i)
            workers.emplace_back ([this] { workerLoop(); });
    }

    /** Join all workers. Safe to call multiple times. */
    void shutdown()
    {
        if (workers.empty())
            return;

        running.store (false, std::memory_order_release);

        {
            std::lock_guard<std::mutex> lock (dispatchMutex);
            dispatch.store (true, std::memory_order_release);
        }
        dispatchCV.notify_all();

        for (auto& w : workers)
        {
            if (w.joinable())
                w.join();
        }

        workers.clear();
        numActiveWorkers = 0;
    }

    //==========================================================================
    /**
        Distribute func(0), func(1), ... func(count-1) across workers + calling thread.
        Blocks until all items are complete.
    */
    template <typename Func>
    void parallelFor (int count, Func&& func)
    {
        if (count <= 0)
            return;

        // Sequential fallback: no workers or only 1 item
        if (workers.empty() || count == 1)
        {
            for (int i = 0; i < count; ++i)
                func (i);
            return;
        }

        // Set up the work. The claim word is published LAST: a worker can still
        // be looping in executeItems() from the previous dispatch, and it claims
        // an item the moment the word is reset, so the function and the zeroed
        // doneCount must already be visible to it. doneCount used to be reset
        // after the index, and a worker that finished an item in between had its
        // increment wiped - the dispatch then waited forever (the reverb thread
        // hung holding its algorithm lock, and the UI froze behind it).
        jassert (count <= 0xFFFF);
        currentFunc = [&func](int i) { func (i); };
        doneCount.store (0, std::memory_order_relaxed);
        nextClaim.store (static_cast<uint64_t> (count) << 32, std::memory_order_release);

        // Wake workers
        {
            std::lock_guard<std::mutex> lock (dispatchMutex);
            dispatch.store (true, std::memory_order_release);
        }
        dispatchCV.notify_all();

        // Calling thread participates
        executeItems();

        // Wait for all items to complete (CV-based, no spin)
        {
            std::unique_lock<std::mutex> lock (doneMutex);
            doneCV.wait (lock, [this, count] {
                return doneCount.load (std::memory_order_acquire) >= count;
            });
        }

        // Reset dispatch flag so workers go back to sleep
        {
            std::lock_guard<std::mutex> lock (dispatchMutex);
            dispatch.store (false, std::memory_order_release);
        }
        dispatchCV.notify_all();

        currentFunc = nullptr;
    }

    /** Get number of active workers (not counting the calling thread). */
    int getNumWorkers() const { return numActiveWorkers; }

private:
    //==========================================================================
    void workerLoop()
    {
        // Place this worker on a performance core (macOS); no-op elsewhere.
        if (realtimePeriodMs > 0.0)
            setCurrentThreadRealtimeAudio (realtimePeriodMs, realtimeComputationMs);

        // Workgroup membership: token lives on (and is destroyed on) this thread.
        juce::WorkgroupToken wgToken;
        uint32_t wgSeenGeneration = 0;

        while (running.load (std::memory_order_acquire))
        {
            // (Re)join the audio workgroup if it changed (no-op off macOS / when unset).
            if (workgroupCoordinator != nullptr)
                workgroupCoordinator->joinIfChanged (wgToken, wgSeenGeneration);

            // Wait for work dispatch
            {
                std::unique_lock<std::mutex> lock (dispatchMutex);
                dispatchCV.wait (lock, [this] {
                    return dispatch.load (std::memory_order_acquire);
                });
            }

            if (! running.load (std::memory_order_acquire))
                break;

            executeItems();

            // Wait until dispatch is cleared (main thread has collected results)
            {
                std::unique_lock<std::mutex> lock (dispatchMutex);
                dispatchCV.wait (lock, [this] {
                    return ! dispatch.load (std::memory_order_acquire)
                        || ! running.load (std::memory_order_acquire);
                });
            }
        }
    }

    void executeItems()
    {
        while (true)
        {
            // Index and count come from one atomic read, so a worker left over
            // from the previous dispatch can never pair a new index with an old
            // count (or the reverse); a claim it wins belongs to the dispatch
            // that published the word, and the acquire makes that dispatch's
            // function and doneCount reset visible before the item runs.
            const uint64_t claim = nextClaim.fetch_add (1, std::memory_order_acquire);
            const int idx   = static_cast<int> (claim & 0xFFFFFFFFu);
            const int total = static_cast<int> (claim >> 32);
            if (idx >= total)
                break;

            if (currentFunc)
                currentFunc (idx);

            if (doneCount.fetch_add (1, std::memory_order_release) + 1 >= total)
            {
                std::lock_guard<std::mutex> lock (doneMutex);
                doneCV.notify_one();
            }
        }
    }

    //==========================================================================
    // Worker threads
    std::vector<std::thread> workers;
    int numActiveWorkers = 0;
    std::atomic<bool> running { false };

    // Realtime scheduling hints (macOS P-core placement); 0 = disabled.
    double realtimePeriodMs = 0.0;
    double realtimeComputationMs = 0.0;

    // Optional audio workgroup membership (macOS); null = disabled.
    AudioWorkgroupCoordinator* workgroupCoordinator = nullptr;

    // Dispatch signalling (main → workers)
    std::mutex dispatchMutex;
    std::condition_variable dispatchCV;
    std::atomic<bool> dispatch { false };

    // Completion signalling (workers → main)
    std::mutex doneMutex;
    std::condition_variable doneCV;

    // Work items. nextClaim packs the dispatch's item count (high 32 bits) with
    // the next unclaimed index (low 32 bits).
    std::function<void (int)> currentFunc;
    std::atomic<uint64_t> nextClaim { 0 };
    std::atomic<int> doneCount { 0 };
};

// Extraction-compat: the class itself stays at global scope for now because
// app headers forward-declare `class AudioParallelFor;` (Source/DSP/
// ReverbAlgorithm.h), and a global using-alias of a namespaced class would
// conflict with that forward declaration. The spatcore-qualified name below
// is the migration target; the class body moves inside the namespace once
// the app's forward declarations are qualified.
namespace spatcore::rt { using ::AudioParallelFor; }
