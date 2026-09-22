#pragma once

#include <juce_audio_basics/juce_audio_basics.h>

#include "EffectChain.h"
#include "EffectParams.h"
#include "EffectsTypes.h"
#include "LoopGuard.h"
#include "../dsp/AcousticSendMatrix.h"
#include "../rt/AudioParallelFor.h"
#include "../rt/AudioWorkgroupCoordinator.h"
#include "../rt/LockFreeRingBuffer.h"
#include "../rt/RtTripleBuffer.h"
#include "../rt/SharedInputRingBuffer.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

namespace spatcore::effects
{

//==============================================================================
/**
    The effects engine without its thread: everything that turns N independent
    effect chains into one realtime batch.

    WHY THE THREAD IS NOT IN HERE. A driver whose drain loop is a private
    run() cannot be stepped, so every test of it would be a test of the
    scheduler: write some blocks, sleep, hope, assert. Splitting the thread off
    makes processBatch() a function the test calls, which is the only way the
    latency ledger can be asserted exactly rather than eventually. The offline
    render harness gets the same benefit, and a consumer that already owns a
    realtime thread (the plan's section 4.8 contract) can drive this directly
    instead of inheriting ours. effects/EffectsEngine.h is the juce::Thread
    wrapper and holds nothing but the wake flag.

    THE BATCH, in the order it must happen:

        1. lap check, availability gate and backlog skip on every source cursor
        2. honour any pending Clear (a batch boundary, never mid batch)
        3. read one block of every source ring into the source block buffer
        4. feed.writeInputs - even when muted, see below
        5. snapshot the feed matrix triplet under the SpinLock
        6. acquire every channel's parameter snapshot ON THIS THREAD
        7. resolve every scratch-row write pointer ON THIS THREAD
        8. parallelFor over channels: two feed passes, guard, chain, return ring
        9. publish the chain latencies, advance the feed once, telemetry

    Steps 6 and 7 are on the driver rather than in the worker item on purpose,
    and both would be silent races otherwise. RtTripleBuffer::acquire() mutates
    a non-atomic read index and documents a single-reader contract, so calling
    it from whichever worker happened to take the item means the reader thread
    for a channel changes from batch to batch. And, in the template's own words
    (reverb/ReverbFeedThread.h):

        // Resolve the feed-row write pointers on THIS thread:
        // AudioBuffer::getWritePointer clears the buffer's isClear flag
        // as a side effect, so calling it from N workers at once is a
        // data race. Each worker then owns one row pointer.

    THE LEDGER. A chain's output leaves through returnRing[fx]; the audio
    callback pops it at the top of block n+1 and writes it into the render
    source row that belongs to that channel, from where it re-enters this
    matrix on the NEXT batch. So one hop costs one device block and A -> B costs
    two. That extra block is not an oversight to be optimised away: it is the
    only thing that keeps the parallel sweep sound. AcousticSendMatrix's whole
    thread-safety argument is that the workers only READ delay lines that
    writeInputs filled before the sweep, and a worker that fed its output back
    into those lines mid sweep would make every other worker's result depend on
    the order the pool happened to run them in.

    The return cushion is the only thing that moves those numbers, and it moves
    them exactly: a cushion of one is one block of latency and no slack, a
    cushion of two is two blocks and one late batch costs nothing. So the hop
    figures above are the cushion-one figures, and the automatic cushion is two
    at or below a 128-sample block - at 64 samples one hop is n+2 and A -> B is
    n+3. The design document disagreed with itself here, its ledger charging
    nothing for a cushion of one while its discard rule allowed a block more
    than the ledger on every path. The ledger is what a user is promised, so the
    ring is primed with exactly `cushion` blocks and anything above that is
    discarded.

    DETERMINISM. Worker count 0 and worker count N must produce bit-identical
    output, because an offline render and a live render have to be the same
    render. Each item writes only state indexed by its own channel, there is no
    reduction across items, and the two feed passes always run in the same
    order, so nothing in the arithmetic can observe how the work was split.

    MUTE silences the feed row and lets everything else run. The template's
    muted branch skips its sweep entirely, which is right for the reverb send
    (the tail it feeds lives downstream and keeps decaying on silence) and wrong
    here: the tails live inside the chains being skipped, so skipping would
    freeze every delay and reverb, starve the return rings, and resume a frozen
    tail on unmute. The source history is written whether muted or not, so
    unmuting cannot replay stale audio.

    THE AUDIO CALLBACK only ever calls pullReturn(). It never blocks: a ready
    flag plus a try-lock, and silence rather than a wait whenever either says
    no. Everything that reallocates holds that lock, so the worst a device
    thread can suffer while the engine is being rebuilt is a silent block.

    NOT REALTIME: prepare() and release(). Both allocate, both rewrite every
    member, and neither may run while the driver is inside a batch - the owner
    stops the driver first, exactly as the app does for the reverb feed.
*/
class EffectsEngineCore
{
public:
    //==========================================================================
    struct Config
    {
        double sampleRate = 48000.0;
        int blockSize = 256;

        /** Every render source the consumer writes into a ring, effect returns
            included, in the consumer's own row order. */
        int numSources = 0;

        /** Live effects channels. Clamped to kMaxEffectChannels. */
        int numEffects = 0;

        /** Row stride of the feed triplet: the CONSUMER's maximum effects
            count, which is not the live count. Getting it too LARGE reads the
            wrong cell of the matrix rather than failing. Getting it too small
            would read off the end of the array, so the batch clamps the routed
            channel count to the published stride and leaves the channels above
            it unrouted - running, but fed from nothing. */
        int matrixStride = 0;

        /** Source row of effect return 0. The returns must be contiguous and
            last, which is what lets one source range describe the fx -> fx part
            of the feed. -1 means there are none (no channel can feed another). */
        int firstEffectSourceRow = -1;

        /** -1 auto: none below a 128-sample block, else jlimit(0, 4, hw/2 - 2).
            The ceiling is deliberately low. On Windows every one of these
            registers as MMCSS "Pro Audio", and the session already runs a
            gather thread per render source plus the reverb engine's and the
            reverb send's pools. */
        int workerThreads = -1;

        /** Resident blocks in a return ring, which IS the return latency: the
            ring is primed with this many blocks of silence at prepare, one pops
            per callback and one arrives per batch, and a pull that finds more
            discards the surplus oldest so the latency can never creep.

            One means no slack at all, and a single late batch then costs TWO
            blocks rather than one: the pull that finds the ring empty writes
            silence and counts an underrun, and the pull after the catch-up
            finds two blocks, discards the older and counts a discard. Two means
            a late batch costs nothing. Zero is not a setting - it is promoted
            to one, because a ring nothing primes drains to empty on every pull.

            -1 auto: 2 at or below a 128-sample block, 1 above. */
        int returnCushionBlocks = -1;

        /** More than this many blocks pending on a source ring means the driver
            has fallen behind; it jumps to the newest block rather than working
            through the backlog, so latency never creeps here either. */
        int maxSourceBacklogBlocks = 2;

        /** Return ring depth, in blocks. Raised if it cannot hold the cushion. */
        int returnRingBlocks = 8;

        /** Feed delay-line length. One line per source, so this is the engine's
            dominant allocation: at 136 sources and 96 kHz a second of history
            is 52 MB. A consumer whose geometry is tighter should say so. */
        double maxFeedDelaySeconds = 1.0;

        double maxEffectDelaySeconds = 5.0;
        int reverbMaxDelaySamples = 16384;
        std::uint32_t noiseKeyBase = 1;

        /** effects/LoopGuard.h's own defaults, and its own units: the two ramp
            figures are COMPLETION times there, not time constants. Disabled,
            the guard is neither observed nor applied, so an effect -> effect
            feed is summed exactly as it was rendered. */
        bool loopGuardEnabled = true;
        float loopGuardCeilingDb = LoopGuard::kDefaultCeilingDb;
        double loopGuardTripSeconds = LoopGuard::kDefaultTripSeconds;
        double loopGuardReleaseSeconds = LoopGuard::kDefaultReleaseSeconds;
        double loopGuardRampDownSeconds = LoopGuard::kDefaultRampDownSeconds;
        double loopGuardRampUpSeconds = LoopGuard::kDefaultRampUpSeconds;
        double loopGuardHysteresisDb = LoopGuard::kDefaultHysteresisDb;

        /** Injectable so a test can drive real chains with instrumented
            stand-ins, the same reason EffectChain::prepare takes one. */
        ModuleFactory moduleFactory = &createModule;
    };

    EffectsEngineCore() = default;
    ~EffectsEngineCore() { release(); }

    //==========================================================================
    /** Allocates everything and publishes the ready flag. NOT realtime, and not
        callable while a batch is running: it rebuilds every buffer, re-prepares
        the chains and restarts the worker pool. The owner stops its driver
        thread first.

        The ring vector is borrowed, not owned - raw pointers are cached out of
        it exactly as the reverb feed does, so the consumer must destroy the
        engine (or call release()) before the rings it points at.

        @returns false when the configuration cannot be honoured at all. */
    bool prepare (const Config& newConfig,
                  const std::vector<std::unique_ptr<spatcore::rt::SharedInputRingBuffer>>& sources,
                  spatcore::rt::AudioWorkgroupCoordinator* newCoordinator = nullptr)
    {
        // Held across the whole rebuild. The audio thread only ever TRY-locks,
        // so it is never delayed by this - it silence-fills for as long as the
        // rebuild takes, which is the correct output for an engine that does
        // not exist yet.
        const juce::SpinLock::ScopedLockType lock (procLock);

        ready.store (false, std::memory_order_release);
        releaseLocked();

        config = newConfig;
        coordinator = newCoordinator;

        blockSize = juce::jmax (1, config.blockSize);
        sampleRate = config.sampleRate > 0.0 ? config.sampleRate : 48000.0;
        numEffectsLive = juce::jlimit (0, kMaxEffectChannels, config.numEffects);
        numSourcesLive = juce::jmax (0, juce::jmin (config.numSources, static_cast<int> (sources.size())));

        if (numEffectsLive <= 0 || numSourcesLive <= 0)
        {
            numEffectsLive = 0;
            numSourcesLive = 0;
            return false;
        }

        firstEffectRow = (config.firstEffectSourceRow >= 0)
                             ? juce::jmin (config.firstEffectSourceRow, numSourcesLive)
                             : numSourcesLive;

        returnCushion = config.returnCushionBlocks >= 0
                            ? juce::jmax (1, config.returnCushionBlocks)
                            : (blockSize <= 128 ? 2 : 1);

        backlogBlocks = juce::jmax (1, config.maxSourceBacklogBlocks);
        loopGuardActive = config.loopGuardEnabled;
        loopGuardWanted.store (config.loopGuardEnabled, std::memory_order_relaxed);
        returnRingBlocks = juce::jmax (config.returnRingBlocks, returnCushion + 3);

        // Sources, cursors and the lap counters. Seeding `consumed` at the
        // ring's CURRENT total rather than at zero matters: the producer has
        // usually been writing for a while by the time an engine is wired up,
        // and a counter seeded at zero would report a lap on the first batch
        // and reset every chain for no reason.
        sourceRings.clear();
        sourceRings.reserve (sources.size());
        for (auto& ring : sources)
            sourceRings.push_back (ring.get());

        // A ring that cannot hold two blocks can never work, and both ways it
        // fails are silent. At or below one block the availability gate never
        // opens - SharedInputRingBuffer::write reserves a slot, so a ring of
        // exactly blockSize never offers a full one - and the engine processes
        // nothing for ever, with batches, wraps and skips all reading zero.
        // Below two blocks the lap threshold (capacity - blockSize) drops under
        // the one block that is legitimately in flight between a callback and
        // its batch, so every batch declares a lap and resets every chain.
        // Refusing turns both into a visible false from prepare().
        for (int src = 0; src < numSourcesLive; ++src)
        {
            auto* ring = sourceRings[static_cast<size_t> (src)];

            if (ring != nullptr && ring->getBufferSize() < blockSize * 2)
            {
                releaseLocked();
                return false;
            }
        }

        cursors.assign (static_cast<size_t> (numSourcesLive), 0);
        consumed.assign (static_cast<size_t> (numSourcesLive), 0);

        for (int src = 0; src < numSourcesLive; ++src)
        {
            auto* ring = sourceRings[static_cast<size_t> (src)];
            if (ring == nullptr)
                continue;

            // getAvailableAt(0) is the write position: the distance from cursor
            // zero to the head. Starting there means the engine begins with an
            // empty view rather than with whatever history the ring happens to
            // hold.
            cursors[static_cast<size_t> (src)] = ring->getAvailableAt (0);
            consumed[static_cast<size_t> (src)] = ring->getTotalWritten();
        }

        sourceBlocks.setSize (numSourcesLive, blockSize);
        sourceBlocks.clear();

        feedBuffer.setSize (numEffectsLive, blockSize);
        feedBuffer.clear();

        // The fx -> fx part of every channel's feed gets its own row so the
        // loop guard can attenuate it alone, and so the input part of the sum
        // is bit-identical whether the guard is tripped or not.
        fxBusBuffer.setSize (numEffectsLive, blockSize);
        fxBusBuffer.clear();

        feedRowPtrs.assign (static_cast<size_t> (numEffectsLive), nullptr);   // never grows in a batch
        fxBusRowPtrs.assign (static_cast<size_t> (numEffectsLive), nullptr);  // never grows in a batch
        paramSnapshots.assign (static_cast<size_t> (numEffectsLive), nullptr);

        discardScratch.assign (static_cast<size_t> (blockSize), 0.0f);

        // Delay lines and tap cells at device rate, sources x effects.
        //
        // THE FLOOR IS NOT OPTIONAL. Config::maxFeedDelaySeconds is a geometry
        // hint from a consumer that knows nothing about the device block size,
        // and AcousticSendMatrix::writeInputs is not merely useless below one
        // block, it is unsafe: its wrap-around second memcpy copies
        // (numSamples - lineLength) floats to the START of a lineLength row, so
        // a block longer than the line writes off the end of it and, for the
        // last source, off the allocation. Five milliseconds of history is 240
        // samples, which is an ordinary-looking number that corrupts the heap
        // at a 512-sample block. The matrix's own header asks for one block
        // more than the longest delay; two blocks is that, with the block the
        // taps read from already accounted for.
        const double minHistorySeconds = (static_cast<double> (blockSize) * 2.0 + 1.0) / sampleRate;

        feed.prepare (sampleRate, numSourcesLive, numEffectsLive,
                      juce::jmax (config.maxFeedDelaySeconds, minHistorySeconds));

        ChainConfig chainConfig;
        chainConfig.sampleRate = sampleRate;
        chainConfig.maxBlock = blockSize;
        chainConfig.reverbMaxDelaySamples = config.reverbMaxDelaySamples;
        chainConfig.maxEffectDelaySeconds = config.maxEffectDelaySeconds;

        // EffectChain and RtTripleBuffer both hold a std::atomic, so neither is
        // movable and neither can live in a resizable vector of values. This is
        // the sketch in the design document that does not compile.
        chains.clear();
        params.clear();
        chains.reserve (static_cast<size_t> (numEffectsLive));
        params.reserve (static_cast<size_t> (numEffectsLive));

        guards.assign (static_cast<size_t> (numEffectsLive), LoopGuard{});
        returnRings.clear();
        returnRings.reserve (static_cast<size_t> (numEffectsLive));

        std::vector<float> silence (static_cast<size_t> (blockSize), 0.0f);

        for (int fx = 0; fx < numEffectsLive; ++fx)
        {
            chainConfig.noiseKey = config.noiseKeyBase + static_cast<std::uint32_t> (fx);

            auto chain = std::make_unique<EffectChain>();
            chain->prepare (chainConfig, config.moduleFactory);
            chains.push_back (std::move (chain));

            params.push_back (std::make_unique<spatcore::rt::RtTripleBuffer<EffectChannelParams>>());

            auto ring = std::make_unique<spatcore::rt::LockFreeRingBuffer>();
            ring->setSize (returnRingBlocks * blockSize);

            // Prime the cushion, which is what makes it one. The ring then
            // holds `cushion` blocks at every pull: one pops, one is produced.
            // Without the priming the ring would drain to empty on every pull
            // no matter what the cushion said, so a late batch would cost a
            // whole block of audio instead of eating into slack that is already
            // there - and the cushion would be a ceiling and nothing else.
            for (int b = 0; b < returnCushion; ++b)
                ring->write (silence.data(), blockSize);

            returnRings.push_back (std::move (ring));

            guards[static_cast<size_t> (fx)].prepare (sampleRate,
                                                      config.loopGuardEnabled,
                                                      config.loopGuardCeilingDb,
                                                      config.loopGuardTripSeconds,
                                                      config.loopGuardReleaseSeconds,
                                                      config.loopGuardRampDownSeconds,
                                                      config.loopGuardRampUpSeconds,
                                                      config.loopGuardHysteresisDb);
        }

        // Every slot, not just the live ones: a channel count that shrank would
        // otherwise leave the GUI reading a meter that nothing updates any more.
        for (auto& telemetry : channels)
            telemetry.clearAll();

        // One item per channel, so more workers than channels is waste. The
        // pool must be prepared here and shut down after the driver is joined,
        // never in between: AudioParallelFor::prepare shuts the pool down
        // first, which races a sweep in flight.
        int workers = config.workerThreads;

        if (workers < 0)
        {
            workers = (blockSize < 128)
                          ? 0
                          : juce::jlimit (0, 4, static_cast<int> (std::thread::hardware_concurrency()) / 2 - 2);
        }

        workers = juce::jlimit (0, juce::jmax (0, numEffectsLive - 1), workers);

        const double blockMs = (sampleRate > 0.0) ? (blockSize / sampleRate) * 1000.0 : 0.0;
        pool.prepare (workers, blockMs, blockMs, coordinator);

        clearMask.store (0, std::memory_order_relaxed);
        clearAllPending.store (false, std::memory_order_relaxed);

        // A rebuild invalidates the caller's matrices as often as not, so the
        // engine starts silent and waits for the next control tick rather than
        // rendering one batch through pointers that may already be dangling.
        {
            const juce::SpinLock::ScopedLockType matrixReset (matrixLock);
            matrixDelays = nullptr;
            matrixLevels = nullptr;
            matrixHf = nullptr;
            matrixStride = config.matrixStride;
            matrixSources = numSourcesLive;
            matrixEffects = numEffectsLive;
        }

        ready.store (true, std::memory_order_release);
        return true;
    }

    /** Drops the ready flag, joins the worker pool and frees everything. The
        owner joins its driver thread FIRST - a flag does not evict a thread
        that is already inside a batch. */
    void release()
    {
        const juce::SpinLock::ScopedLockType lock (procLock);
        ready.store (false, std::memory_order_release);
        releaseLocked();
    }

    bool isReady() const noexcept { return ready.load (std::memory_order_acquire); }

    //==========================================================================
    /** Process every batch whose block is already complete on every source.

        The reverb feed deliberately does one batch per wake and lets the
        surplus sit in the ring; with 32 chains and a fork-join sweep a single
        long batch would then never be made up, so this drains instead and
        reports how many batches a wake actually needed. A wake that finds more
        than the backlog allowance jumps forward rather than working through it.

        @returns batches processed. */
    int drainAvailable() noexcept
    {
        int batches = 0;

        while (batches < kMaxBatchesPerWake && processBatch())
            ++batches;

        batchesPerWake.store (static_cast<std::uint32_t> (batches), std::memory_order_relaxed);
        return batches;
    }

    /** One batch, or false when no source has a full block ready. Everything
        realtime in this class is reachable from here.

        noexcept is a statement about this code, not about the pool: the fork
        and join inside AudioParallelFor take two std::mutexes, and a lock that
        threw std::system_error would meet this specifier and terminate rather
        than unwind. That is the right failure for a driver thread whose
        scheduler has stopped working, and it is worth knowing that it is the
        failure mode rather than a caught error. */
    bool processBatch() noexcept
    {
        // The driver's serial path needs its own FTZ/DAZ arming: MXCSR is
        // per-thread state, so the one inside the worker items does not cover
        // the feed reads, the ring writes or the telemetry done out here.
        const juce::ScopedNoDenormals noDenormals;

        // `ready` alone, and deliberately: it is raised last and lowered first,
        // both under procLock, so it is never true for an engine that is not
        // fully built. A second plain bool beside it would be a formal data
        // race between this thread and prepare/release for no added safety.
        if (! ready.load (std::memory_order_acquire))
            return false;

        const int n = blockSize;

        // ---- Lap detection, before anything is read ------------------------
        // The per-cursor "available" is modular arithmetic, so it cannot
        // express a lag longer than the ring: a consumer a whole buffer behind
        // looks exactly like one that is up to date and reads audio the
        // producer is concurrently overwriting. The additive counter is the
        // only thing that can see it.
        bool lapped = false;

        for (int src = 0; src < numSourcesLive; ++src)
        {
            auto* ring = sourceRings[static_cast<size_t> (src)];
            if (ring == nullptr)
                continue;

            const std::uint64_t written = ring->getTotalWritten();
            const std::uint64_t taken = consumed[static_cast<size_t> (src)];
            const int capacity = ring->getBufferSize();

            if (capacity <= n)
                continue;

            // Unsigned on purpose: a ring reset behind our back leaves `taken`
            // ahead of `written`, the subtraction wraps to something enormous,
            // and the resync below puts us back in step. A silent resync is the
            // right answer to a ring that restarted under us.
            if (written - taken > static_cast<std::uint64_t> (capacity - n))
            {
                resyncSourceToNewest (src);
                lapped = true;
            }
        }

        if (lapped)
        {
            ringWraps.fetch_add (1, std::memory_order_relaxed);
            resetAllChains();       // the input stream jumped; tails no longer belong to it
        }

        // ---- Gate: all or nothing across every source -----------------------
        int minAvail = std::numeric_limits<int>::max();

        for (int src = 0; src < numSourcesLive; ++src)
        {
            auto* ring = sourceRings[static_cast<size_t> (src)];
            minAvail = juce::jmin (minAvail, ring != nullptr ? ring->getAvailableAt (cursors[static_cast<size_t> (src)]) : 0);
        }

        if (minAvail < n)
            return false;

        // ---- Backlog: jump forward rather than work through it --------------
        if (minAvail > backlogBlocks * n)
        {
            for (int src = 0; src < numSourcesLive; ++src)
                resyncSourceToNewest (src);

            sourceSkips.fetch_add (1, std::memory_order_relaxed);
            resetAllChains();
        }

        const auto batchStart = std::chrono::steady_clock::now();

        honourClearRequests();
        honourLoopGuardSwitch();

        // ---- Read one block of every source ---------------------------------
        for (int src = 0; src < numSourcesLive; ++src)
        {
            float* dest = sourceBlocks.getWritePointer (src);
            auto* ring = sourceRings[static_cast<size_t> (src)];

            const int got = (ring != nullptr) ? ring->readWithPosition (cursors[static_cast<size_t> (src)], dest, n) : 0;

            if (got < n)
                juce::FloatVectorOperations::clear (dest + got, n - got);   // never render stale tail

            consumed[static_cast<size_t> (src)] += static_cast<std::uint64_t> (got);
        }

        // Published into the per-source delay lines whether muted or not, so
        // unmuting cannot replay stale history.
        feed.writeInputs (sourceBlocks, n);

        // ---- Snapshot the matrix triplet once, under the lock ---------------
        // The rest of the batch runs off these locals, so the per-sample, per
        // source, per channel loops need no synchronisation at all. A matrix
        // published mid batch lands on the NEXT one; one block of stale send
        // levels is inaudible at 50 Hz control cadence and the tap smoothers
        // absorb the step.
        const float* levelsSnap = nullptr;
        const float* delaysSnap = nullptr;
        const float* hfSnap = nullptr;
        int strideSnap = 0;
        int sourcesSnap = 0;
        int routedEffects = 0;

        {
            const juce::SpinLock::ScopedLockType lock (matrixLock);
            levelsSnap = matrixLevels;
            delaysSnap = matrixDelays;
            hfSnap = matrixHf;
            strideSnap = matrixStride;

            // EVERY count is clamped, the stride included. The template clamps
            // only its node count, which is safe there because its source count
            // cannot change; here the source count moves whenever the
            // effects-channel count does.
            //
            // The stride clamp is a bounds check rather than tidiness.
            // computeNodeFeed indexes levels[src * stride + node] into an array
            // the caller sized numSources * stride, so a node index at or above
            // the stride reads the NEXT source's row and, on the last source,
            // past the end of the allocation - a garbage send gain, which is an
            // unbounded-loudness bug rather than a crash. setFeedMatrices
            // validates nothing, so this is the only place it can be caught.
            sourcesSnap = juce::jmin (matrixSources, feed.getPreparedSources(), sourceBlocks.getNumChannels());
            routedEffects = juce::jmin (juce::jmin (matrixEffects, strideSnap),
                                        juce::jmin (feed.getPreparedNodes(), numEffectsLive));
        }

        // THE ROUTED COUNT IS NOT THE LIVE COUNT, and conflating them is how a
        // channel goes silent for good. `routedEffects` bounds the matrix
        // READS; every loop below runs over numEffectsLive, because a channel
        // the published matrix does not reach still has to run: its chain holds
        // the tails, and its return ring is what the audio callback pops on
        // every single callback. Publish two effects while four are live and
        // the other two would freeze mid-tail and underrun the callback for
        // ever, with no telemetry anywhere saying why. An unrouted channel is
        // simply fed from no sources at all.
        if (routedEffects <= 0)
        {
            // Nothing published yet, or a stride too narrow to index with: the
            // same treatment, applied to every channel. A count of zero says
            // nothing about how long the levels array is, and a stride of zero
            // would fold every source onto one cell of it.
            routedEffects = 0;
            sourcesSnap = 0;
            levelsSnap = nullptr;
            delaysSnap = nullptr;
            hfSnap = nullptr;
        }

        const int firstFx = juce::jmin (firstEffectRow, sourcesSnap);

        // ---- Parameters: acquired HERE, never inside an item ----------------
        // RtTripleBuffer mutates a non-atomic read index and documents one
        // reader thread; from inside the sweep the reader for a given channel
        // would be a different thread every batch. The references stay valid
        // until this thread's next acquire, which is the next batch.
        for (int fx = 0; fx < numEffectsLive; ++fx)
            paramSnapshots[static_cast<size_t> (fx)] = &params[static_cast<size_t> (fx)]->acquire();

        // ---- Write pointers: resolved HERE, never inside an item ------------
        // Quoting reverb/ReverbFeedThread.h, which is the reason this line
        // exists at all:
        //
        //     // Resolve the feed-row write pointers on THIS thread:
        //     // AudioBuffer::getWritePointer clears the buffer's isClear flag
        //     // as a side effect, so calling it from N workers at once is a
        //     // data race. Each worker then owns one row pointer.
        for (int fx = 0; fx < numEffectsLive; ++fx)
        {
            feedRowPtrs[static_cast<size_t> (fx)] = feedBuffer.getWritePointer (fx);
            fxBusRowPtrs[static_cast<size_t> (fx)] = fxBusBuffer.getWritePointer (fx);
        }

        const bool muted = isMuted.load (std::memory_order_relaxed);

        // ---- The sweep -------------------------------------------------------
        pool.parallelFor (numEffectsLive,
                          [this, n, firstFx, sourcesSnap, routedEffects, strideSnap,
                           levelsSnap, delaysSnap, hfSnap, muted] (int fx)
        {
            // An unrouted channel sums no sources, which keeps every matrix
            // read inside the array the consumer published and still leaves the
            // chain, the guard, the meters and the return ring exactly as busy
            // as a routed one.
            const int mySources = (fx < routedEffects) ? sourcesSnap : 0;

            renderChannel (fx, n, juce::jmin (firstFx, mySources), mySources,
                           strideSnap, levelsSnap, delaysSnap, hfSnap, muted);
        });

        // ---- After the join --------------------------------------------------
        // The chain latency getter reads plain bools that applyParams wrote
        // inside the sweep, so it is a data race from any other thread and can
        // tear across a bypass edit. Computed once here, where the pool's join
        // has already made the sweep's writes visible, and published as an
        // atomic: nobody else ever calls the chain getter.
        for (int fx = 0; fx < numEffectsLive; ++fx)
            channels[static_cast<size_t> (fx)].latencySamples.store (chains[static_cast<size_t> (fx)]->getLatencySamples(),
                                                                     std::memory_order_relaxed);

        feed.advance (n);

        lastBatchUs.store (std::chrono::duration<float, std::micro> (std::chrono::steady_clock::now() - batchStart).count(),
                           std::memory_order_relaxed);
        batchCounter.fetch_add (1, std::memory_order_relaxed);
        return true;
    }

    //==========================================================================
    /** AUDIO THREAD. One channel's return block, or silence.

        Never blocks, never allocates, never waits: a ready flag and a try-lock,
        and silence the moment either says no. The try-lock is what makes a
        rebuild safe - the message thread holds the same lock across the whole
        reallocation, and a device callback that lands in the middle of one
        silence-fills instead of dereferencing a ring that is being destroyed.

        @returns true when real audio was written. */
    bool pullReturn (int fx, float* dst, int numSamples) noexcept
    {
        if (dst == nullptr || numSamples <= 0)
            return false;

        if (! ready.load (std::memory_order_acquire))
        {
            juce::FloatVectorOperations::clear (dst, numSamples);
            return false;
        }

        const juce::SpinLock::ScopedTryLockType lock (procLock);

        if (! lock.isLocked())
        {
            juce::FloatVectorOperations::clear (dst, numSamples);
            lockFailures.fetch_add (1, std::memory_order_relaxed);
            return false;
        }

        // Re-read under the lock. The flag is lowered while the lock is held,
        // so this is the point at which a teardown becomes visible; the fast
        // path above only avoids taking the lock at all in the common case.
        if (! ready.load (std::memory_order_relaxed) || fx < 0 || fx >= numEffectsLive)
        {
            juce::FloatVectorOperations::clear (dst, numSamples);
            return false;
        }

        auto& ring = *returnRings[static_cast<size_t> (fx)];
        auto& telemetry = channels[static_cast<size_t> (fx)];

        // The ceiling IS the latency. The design document's ledger says a
        // cushion of one costs no extra block, while its discard rule allows
        // cushion + 1 resident, which is one block more than the ledger
        // promises on every path. The ledger wins: it is the user-visible claim
        // and the cheaper answer, so the surplus goes when the ring holds more
        // than the cushion, not more than the cushion plus one. Never below
        // what this pull needs, or a caller asking for more than a block would
        // starve itself.
        const int ceiling = juce::jmax (returnCushion * blockSize, numSamples);
        int avail = ring.getAvailableData();

        if (avail > ceiling)
        {
            discardFromRing (ring, avail - ceiling);
            avail = ceiling;
            telemetry.discards.fetch_add (1, std::memory_order_relaxed);
        }

        if (avail < numSamples)
        {
            juce::FloatVectorOperations::clear (dst, numSamples);
            telemetry.underruns.fetch_add (1, std::memory_order_relaxed);
            return false;
        }

        ring.read (dst, numSamples);
        return true;
    }

    //==========================================================================
    /** Control thread. One channel's cooked parameters; the chain re-applies
        them only when `revision` moves, so a publisher that forgets to bump it
        will see its edit ignored. Same thread as prepare/release. */
    void publishChannelParams (int fx, const EffectChannelParams& p) noexcept
    {
        if (fx >= 0 && fx < static_cast<int> (params.size()))
            params[static_cast<size_t> (fx)]->publish (p);
    }

    /** Control thread. Publishes the (delays, levels, hf, stride, counts) tuple
        for the next batch. The held region is six scalar copies; the reader
        snapshots it once per batch and never touches the lock again. */
    void setFeedMatrices (const float* delaysMs, const float* levels, const float* hfDb,
                          int stride, int numSources, int numEffects) noexcept
    {
        const juce::SpinLock::ScopedLockType lock (matrixLock);
        matrixDelays = delaysMs;
        matrixLevels = levels;
        matrixHf = hfDb;
        matrixStride = stride;
        matrixSources = numSources;
        matrixEffects = numEffects;
    }

    /** Any thread. Silences the feed; the chains keep running. */
    void setMuted (bool muted) noexcept { isMuted.store (muted, std::memory_order_relaxed); }
    bool getMuted() const noexcept      { return isMuted.load (std::memory_order_relaxed); }

    /** Any thread. The operator's global loop-guard switch, honoured at the
        next batch boundary rather than by a rebuild. */
    void setLoopGuardEnabled (bool shouldBeEnabled) noexcept
    {
        loopGuardWanted.store (shouldBeEnabled, std::memory_order_relaxed);
    }

    bool isLoopGuardEnabled() const noexcept { return loopGuardWanted.load (std::memory_order_relaxed); }

    /** Any thread. Emergency Clear, honoured at the next batch boundary.

        WHAT IT COSTS, honestly. Per channel it is cheap: reset the chain, drop
        its tails, release its loop guard. For ALL channels it also clears the
        feed's delay lines, and that is one memset over one line per source -
        52 MB at 136 sources, a second of history and 96 kHz, several
        milliseconds of pure memory traffic on a realtime thread. It will drop
        audio. That is defensible for a button whose entire purpose is to stop
        an unpleasant noise now, but "silences everything instantly" is not what
        it does, and a clear is counted so that a duty spike next to one is
        attributable rather than mysterious.

        What is already in a return ring still plays out, so the silence arrives
        within returnCushionBlocks blocks rather than immediately. Draining the
        rings would need the audio thread's cursor, and a second clear path
        through the callback buys a millisecond at the cost of an audio-thread
        code path that runs at the worst possible moment. */
    void requestClear (int fx = -1) noexcept
    {
        if (fx < 0)
        {
            clearAllPending.store (true, std::memory_order_relaxed);
            clearMask.fetch_or (0xFFFFFFFFu, std::memory_order_relaxed);
            return;
        }

        if (fx < kMaxEffectChannels)
            clearMask.fetch_or (1u << static_cast<std::uint32_t> (fx), std::memory_order_relaxed);
    }

    //==========================================================================
    // Telemetry. Relaxed atomics, never reset on read, sampled by the GUI at
    // 20-50 Hz. Per-channel values live one cache line apart so that four
    // workers writing four channels' meters every block do not fight over one.

    float getLastBatchUs() const noexcept          { return lastBatchUs.load (std::memory_order_relaxed); }
    std::uint32_t getBatchCount() const noexcept   { return batchCounter.load (std::memory_order_relaxed); }
    std::uint32_t getBatchesPerWake() const noexcept { return batchesPerWake.load (std::memory_order_relaxed); }
    std::uint32_t getSourceSkips() const noexcept  { return sourceSkips.load (std::memory_order_relaxed); }
    std::uint32_t getRingWraps() const noexcept    { return ringWraps.load (std::memory_order_relaxed); }
    std::uint32_t getClearCount() const noexcept   { return clearCount.load (std::memory_order_relaxed); }
    float getLastClearUs() const noexcept          { return lastClearUs.load (std::memory_order_relaxed); }
    std::uint32_t getLockFailures() const noexcept { return lockFailures.load (std::memory_order_relaxed); }

    std::uint32_t getUnderruns (int fx) const noexcept       { return channelValue (fx, &ChannelTelemetry::underruns); }
    std::uint32_t getReturnDiscards (int fx) const noexcept  { return channelValue (fx, &ChannelTelemetry::discards); }
    std::uint32_t getReturnOverflows (int fx) const noexcept { return channelValue (fx, &ChannelTelemetry::overflows); }
    std::uint32_t getNanTrips (int fx) const noexcept        { return channelValue (fx, &ChannelTelemetry::nanTrips); }
    std::uint32_t getLoopGuardTrips (int fx) const noexcept  { return channelValue (fx, &ChannelTelemetry::loopGuardTrips); }

    bool isLoopGuardTripped (int fx) const noexcept
    {
        return inRange (fx) && channels[static_cast<size_t> (fx)].loopGuardTripped.load (std::memory_order_relaxed);
    }

    int getChainLatencySamples (int fx) const noexcept
    {
        return inRange (fx) ? channels[static_cast<size_t> (fx)].latencySamples.load (std::memory_order_relaxed) : 0;
    }

    /** One module slot's meter, in dB: the Dynamics module reports its gain
        reduction, the others their output peak (see IEffectModule::getMeterDb).
        Every module already keeps this in a relaxed atomic it writes per
        block, so the GUI reads it straight from the module - no telemetry copy,
        no reset on read. The chains are allocated in prepare() and stable until
        release(), and a slot index out of range or a chain not built reads 0. */
    float getSlotMeterDb (int fx, int slot) const noexcept
    {
        if (slot < 0 || slot >= kNumModuleSlots)
            return 0.0f;

        if (fx < 0 || fx >= static_cast<int> (chains.size()))
            return 0.0f;

        const auto& chain = chains[static_cast<size_t> (fx)];
        return chain != nullptr ? chain->getSlot (slot).getMeterDb() : 0.0f;
    }

    float getFeedPeak (int fx) const noexcept      { return channelFloat (fx, &ChannelTelemetry::feedPeak); }
    float getFeedMeanSq (int fx) const noexcept    { return channelFloat (fx, &ChannelTelemetry::feedMeanSq); }
    float getReturnPeak (int fx) const noexcept    { return channelFloat (fx, &ChannelTelemetry::returnPeak); }
    float getReturnMeanSq (int fx) const noexcept  { return channelFloat (fx, &ChannelTelemetry::returnMeanSq); }

    int getBlockSize() const noexcept              { return blockSize; }
    int getNumEffects() const noexcept             { return numEffectsLive; }
    int getNumSources() const noexcept             { return numSourcesLive; }
    int getReturnCushionBlocks() const noexcept    { return returnCushion; }
    int getNumWorkers() const noexcept             { return pool.getNumWorkers(); }

private:
    //==========================================================================
    /** One channel's telemetry, alone on a cache line.

        Each worker item writes only its own channel's entries, so a whole line
        is dirtied by one thread per block. Packed into one array of plain
        atomics instead, sixteen channels would share a line and four workers
        would ping-pong two lines between them every block for nothing. */
    // MSVC warns that alignas padded the struct. That IS the point: one
    // telemetry line per channel on its own cache line, so 32 workers
    // storing counters do not ping the same line back and forth between
    // cores. Silenced here rather than left to rattle in every consumer's
    // build, since nothing in the tree treats warnings as errors and a
    // standing warning is one nobody reads.
    #if defined (_MSC_VER)
     #pragma warning (push)
     #pragma warning (disable: 4324)
    #endif
    struct alignas (64) ChannelTelemetry
    {
        std::atomic<float> feedPeak { 0.0f };
        std::atomic<float> feedMeanSq { 0.0f };
        std::atomic<float> returnPeak { 0.0f };
        std::atomic<float> returnMeanSq { 0.0f };
        std::atomic<std::uint32_t> underruns { 0 };
        std::atomic<std::uint32_t> discards { 0 };
        std::atomic<std::uint32_t> overflows { 0 };
        std::atomic<std::uint32_t> nanTrips { 0 };
        std::atomic<std::uint32_t> loopGuardTrips { 0 };
        std::atomic<int> latencySamples { 0 };
        std::atomic<bool> loopGuardTripped { false };

        void clearAll() noexcept
        {
            feedPeak.store (0.0f, std::memory_order_relaxed);
            feedMeanSq.store (0.0f, std::memory_order_relaxed);
            returnPeak.store (0.0f, std::memory_order_relaxed);
            returnMeanSq.store (0.0f, std::memory_order_relaxed);
            underruns.store (0, std::memory_order_relaxed);
            discards.store (0, std::memory_order_relaxed);
            overflows.store (0, std::memory_order_relaxed);
            nanTrips.store (0, std::memory_order_relaxed);
            loopGuardTrips.store (0, std::memory_order_relaxed);
            latencySamples.store (0, std::memory_order_relaxed);
            loopGuardTripped.store (false, std::memory_order_relaxed);
        }
    };
    #if defined (_MSC_VER)
     #pragma warning (pop)
    #endif

    //==========================================================================
    /** One work item: one channel, start to finish.

        Everything it touches is indexed by `fx` - its two scratch rows, its
        guard, its chain, its ring, its telemetry line - so the pool may run the
        items in any order on any thread and the arithmetic cannot tell. That is
        the whole determinism argument. */
    void renderChannel (int fx, int n, int firstFx, int sourcesSnap, int strideSnap,
                        const float* levelsSnap, const float* delaysSnap, const float* hfSnap,
                        bool muted) noexcept
    {
        // FTZ/DAZ is per-thread MXCSR state and this body runs on whichever
        // worker took the item, so the driver's own guard does not reach it.
        // Chains are full of IIR tails and feedback delays, which is exactly
        // where denormals cost whole percent of a core.
        const juce::ScopedNoDenormals noDenormals;

        float* row = feedRowPtrs[static_cast<size_t> (fx)];
        float* bus = fxBusRowPtrs[static_cast<size_t> (fx)];
        auto& guard = guards[static_cast<size_t> (fx)];
        auto& telemetry = channels[static_cast<size_t> (fx)];

        // Pass one: the input rows, ascending source index, overwriting.
        feed.computeNodeFeed (row, n, fx, levelsSnap, delaysSnap, hfSnap, strideSnap,
                              0, firstFx, true);

        // Pass two: the effect-return rows, into their own buffer. Two passes
        // over disjoint source ranges touch disjoint tap cells, so every
        // smoother still advances exactly once per batch.
        feed.computeNodeFeed (bus, n, fx, levelsSnap, delaysSnap, hfSnap, strideSnap,
                              firstFx, sourcesSnap, true);

        if (loopGuardActive)
        {
            // The peak is taken BEFORE the gain, which is the guard's own
            // contract: measured after its multiply it would describe the guard
            // rather than the loop. LoopGuard::peakOf keeps a non-finite sample
            // as the peak, where a plain running max would drop it.
            const float busPeak = LoopGuard::peakOf (bus, n);

            // The return peak is the previous block's, written by this same
            // item last batch and made visible by the pool's join. It can only
            // ever delay a release, so one batch of lag on it changes nothing.
            guard.observeBlock (busPeak, telemetry.returnPeak.load (std::memory_order_relaxed), n);

            // applyGain does not touch the row at all while the guard is armed
            // and settled, so an untripped channel adds the bus exactly as it
            // was rendered. The input part of the sum is therefore bit-identical
            // whether the guard is tripped or not, which is the property the
            // separate scratch row exists to keep.
            guard.applyGain (bus, n);

            telemetry.loopGuardTripped.store (guard.isTripped(), std::memory_order_relaxed);
            telemetry.loopGuardTrips.store (guard.getTripCount(), std::memory_order_relaxed);
        }

        juce::FloatVectorOperations::add (row, bus, n);

        // Muted silences what the chain is fed, and nothing else: every chain
        // still runs, every tail still decays, every return ring still gets its
        // block, so unmuting resumes a tail that has moved on rather than one
        // that was frozen, and the callback counts no underruns meanwhile.
        if (muted)
            juce::FloatVectorOperations::clear (row, n);

        float feedPeak = 0.0f;
        double feedSumSq = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const float v = row[i];
            const float a = std::fabs (v);
            if (a > feedPeak)
                feedPeak = a;
            feedSumSq += static_cast<double> (v) * static_cast<double> (v);
        }

        telemetry.feedPeak.store (feedPeak, std::memory_order_relaxed);
        telemetry.feedMeanSq.store (static_cast<float> (feedSumSq / static_cast<double> (n)), std::memory_order_relaxed);

        chains[static_cast<size_t> (fx)]->process (row, n, *paramSnapshots[static_cast<size_t> (fx)]);

        // THE WHOLE BLOCK IS CHECKED, not just its last sample.
        //
        // ModuleSlot and EffectChain both test isfinite(buf[n-1]), which is the
        // right trade inside the chain: the structures that actually go bad are
        // recursive, and once they do they stay bad, so the last sample tells
        // you. A memoryless module can still pass a single mid-block NaN
        // straight through untripped, and this is the last place before the
        // speakers. The scan is free here because the meters already walk every
        // sample: a non-finite value anywhere poisons the sum of squares, so
        // one isfinite() on the accumulator detects it exactly. The accumulator
        // is a double so that no finite float block can overflow it into a
        // false trip - the largest possible sum of squares of finite floats is
        // about 1e80, and a double holds 1e308.
        float returnPeak = 0.0f;
        double returnSumSq = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const float v = row[i];
            const float a = std::fabs (v);
            if (a > returnPeak)
                returnPeak = a;
            returnSumSq += static_cast<double> (v) * static_cast<double> (v);
        }

        if (! std::isfinite (returnSumSq))
        {
            juce::FloatVectorOperations::clear (row, n);
            chains[static_cast<size_t> (fx)]->reset();
            telemetry.nanTrips.fetch_add (1, std::memory_order_relaxed);
            returnPeak = 0.0f;
            returnSumSq = 0.0;
        }

        telemetry.returnPeak.store (returnPeak, std::memory_order_relaxed);
        telemetry.returnMeanSq.store (static_cast<float> (returnSumSq / static_cast<double> (n)), std::memory_order_relaxed);

        // The ring's producer is whichever worker took this item, so it changes
        // from batch to batch. That is still SPSC because the pool's fork and
        // join go through a mutex and a condition variable, which supplies the
        // happens-before edge between one batch's writer and the next one's. A
        // lock-free join would silently break this.
        if (returnRings[static_cast<size_t> (fx)]->write (row, n) < n)
            telemetry.overflows.fetch_add (1, std::memory_order_relaxed);
    }

    //==========================================================================
    /** The operator's global loop-guard switch, taken at a batch boundary
        because the guards belong to the driver. Turning it off re-arms every
        guard, so a channel that was being held down gets its feed back in the
        same batch: someone who switches the guard off mid-trip is asking for
        their loop now, and the batch that does it is also the batch in which
        nothing else about the channel changes. */
    void honourLoopGuardSwitch() noexcept
    {
        const bool wanted = loopGuardWanted.load (std::memory_order_relaxed);

        if (wanted == loopGuardActive)
            return;

        loopGuardActive = wanted;

        for (int fx = 0; fx < numEffectsLive; ++fx)
        {
            guards[static_cast<size_t> (fx)].setEnabled (wanted);
            channels[static_cast<size_t> (fx)].loopGuardTripped.store (false, std::memory_order_relaxed);
        }
    }

    void honourClearRequests() noexcept
    {
        const std::uint32_t mask = clearMask.exchange (0, std::memory_order_relaxed);
        const bool all = clearAllPending.exchange (false, std::memory_order_relaxed);

        if (mask == 0 && ! all)
            return;

        const auto clearStart = std::chrono::steady_clock::now();

        for (int fx = 0; fx < numEffectsLive; ++fx)
        {
            if (all || (mask & (1u << static_cast<std::uint32_t> (fx))) != 0)
            {
                chains[static_cast<size_t> (fx)]->reset();
                guards[static_cast<size_t> (fx)].reset();
                channels[static_cast<size_t> (fx)].loopGuardTripped.store (false, std::memory_order_relaxed);
            }
        }

        // Only "all" touches the shared history, and deliberately so: a
        // per-channel clear drops that channel's tails, it does not rewrite
        // what every other channel and the reverb send are reading.
        if (all)
            feed.reset();

        clearCount.fetch_add (1, std::memory_order_relaxed);
        lastClearUs.store (std::chrono::duration<float, std::micro> (std::chrono::steady_clock::now() - clearStart).count(),
                           std::memory_order_relaxed);
    }

    /** Leave this source exactly one block behind the write head.

        Works for both a lap and a backlog because `available` is the modular
        distance from the cursor to the head either way, so moving the cursor
        forward by (available - block) lands one block short of the head with no
        assumption about where the head actually is. */
    void resyncSourceToNewest (int src) noexcept
    {
        auto* ring = sourceRings[static_cast<size_t> (src)];
        if (ring == nullptr)
            return;

        const int capacity = ring->getBufferSize();
        if (capacity <= 0)
            return;

        // Read the counter first. A write that lands between these two reads
        // makes `consumed` look one block further behind than it is, which
        // costs nothing; reading them the other way round would make it look
        // closer and could hide the next lap.
        const std::uint64_t written = ring->getTotalWritten();
        const int avail = ring->getAvailableAt (cursors[static_cast<size_t> (src)]);
        const int surplus = avail - blockSize;

        if (surplus > 0)
            cursors[static_cast<size_t> (src)] = (cursors[static_cast<size_t> (src)] + surplus) % capacity;

        const std::uint64_t lag = static_cast<std::uint64_t> (juce::jmax (0, juce::jmin (avail, blockSize)));
        consumed[static_cast<size_t> (src)] = (written >= lag) ? (written - lag) : 0;
    }

    void resetAllChains() noexcept
    {
        for (auto& chain : chains)
            chain->reset();
    }

    /** AUDIO THREAD. Drop the oldest samples of a return ring by reading them
        into a scratch, which is the only way to move a cursor the reader owns.
        Bounded by the ring depth; no allocation, the scratch is sized once. */
    void discardFromRing (spatcore::rt::LockFreeRingBuffer& ring, int numSamples) noexcept
    {
        int remaining = numSamples;
        const int chunk = static_cast<int> (discardScratch.size());

        while (remaining > 0 && chunk > 0)
        {
            const int got = ring.read (discardScratch.data(), juce::jmin (remaining, chunk));
            if (got <= 0)
                break;

            remaining -= got;
        }
    }

    /** Everything release() and prepare() both need, with procLock already held. */
    void releaseLocked()
    {
        // The pool must be shut down while nothing is dispatching: prepare()
        // itself shuts down first and would otherwise mutate the worker vector
        // under a sweep in flight.
        pool.shutdown();

        chains.clear();
        params.clear();
        returnRings.clear();
        guards.clear();
        sourceRings.clear();
        cursors.clear();
        consumed.clear();
        paramSnapshots.clear();
        feedRowPtrs.clear();
        fxBusRowPtrs.clear();
        discardScratch.clear();

        sourceBlocks.setSize (0, 0);
        feedBuffer.setSize (0, 0);
        fxBusBuffer.setSize (0, 0);

        // The feed's delay lines are the engine's dominant allocation - 52 MB
        // at 136 sources, a second of history and 96 kHz - and clearing the
        // vectors above does not touch them. AcousticSendMatrix has no
        // release(), so re-preparing it at zero sources is how the memory is
        // handed back; it also leaves isPrepared() false, which makes a stray
        // computeNodeFeed clear its destination and return rather than read a
        // line nobody owns.
        feed.prepare (sampleRate, 0, 0, 0.001);

        numEffectsLive = 0;
        numSourcesLive = 0;
    }

    bool inRange (int fx) const noexcept { return fx >= 0 && fx < kMaxEffectChannels; }

    std::uint32_t channelValue (int fx, std::atomic<std::uint32_t> ChannelTelemetry::* member) const noexcept
    {
        return inRange (fx) ? (channels[static_cast<size_t> (fx)].*member).load (std::memory_order_relaxed) : 0u;
    }

    float channelFloat (int fx, std::atomic<float> ChannelTelemetry::* member) const noexcept
    {
        return inRange (fx) ? (channels[static_cast<size_t> (fx)].*member).load (std::memory_order_relaxed) : 0.0f;
    }

    //==========================================================================
    static constexpr int kMaxBatchesPerWake = 32;

    Config config;
    spatcore::rt::AudioWorkgroupCoordinator* coordinator = nullptr;

    double sampleRate = 48000.0;
    int blockSize = 256;
    int numSourcesLive = 0;
    int numEffectsLive = 0;
    int firstEffectRow = 0;
    int returnCushion = 1;
    int backlogBlocks = 2;
    int returnRingBlocks = 8;
    bool loopGuardActive = true;      // driver thread only; mirrors loopGuardWanted

    spatcore::dsp::AcousticSendMatrix feed;

    // Non-movable, every one of them: EffectChain and RtTripleBuffer hold a
    // std::atomic and LockFreeRingBuffer is non-copyable, so none of these can
    // live in a resizable vector of values.
    std::vector<std::unique_ptr<EffectChain>> chains;
    std::vector<std::unique_ptr<spatcore::rt::RtTripleBuffer<EffectChannelParams>>> params;
    std::vector<std::unique_ptr<spatcore::rt::LockFreeRingBuffer>> returnRings;

    // Legal as values precisely because LoopGuard holds no atomic.
    std::vector<LoopGuard> guards;

    std::vector<spatcore::rt::SharedInputRingBuffer*> sourceRings;   // borrowed, never owned
    std::vector<int> cursors;
    std::vector<std::uint64_t> consumed;

    juce::AudioBuffer<float> sourceBlocks;   // one row per source, one block
    juce::AudioBuffer<float> feedBuffer;     // one row per channel: the feed, then the return
    juce::AudioBuffer<float> fxBusBuffer;    // one row per channel: the fx -> fx part alone

    std::vector<float*> feedRowPtrs;
    std::vector<float*> fxBusRowPtrs;
    std::vector<const EffectChannelParams*> paramSnapshots;
    std::vector<float> discardScratch;       // audio thread only, under procLock

    AudioParallelFor pool;

    // (delays, levels, hf, stride, sources, effects) is one tuple published by
    // the control thread and snapshotted once per batch by the driver, so the
    // driver can never see a torn set of pointers and counts.
    const float* matrixDelays = nullptr;
    const float* matrixLevels = nullptr;
    const float* matrixHf = nullptr;
    int matrixStride = 0;
    int matrixSources = 0;
    int matrixEffects = 0;
    juce::SpinLock matrixLock;

    // Lifetime, not data: taken for the whole of prepare/release, try-locked by
    // the audio thread, never taken by the driver on the batch path.
    juce::SpinLock procLock;
    std::atomic<bool> ready { false };

    std::atomic<bool> isMuted { false };
    std::atomic<bool> loopGuardWanted { true };
    std::atomic<std::uint32_t> clearMask { 0 };
    std::atomic<bool> clearAllPending { false };

    std::atomic<float> lastBatchUs { 0.0f };
    std::atomic<float> lastClearUs { 0.0f };
    std::atomic<std::uint32_t> batchCounter { 0 };
    std::atomic<std::uint32_t> batchesPerWake { 0 };
    std::atomic<std::uint32_t> sourceSkips { 0 };
    std::atomic<std::uint32_t> ringWraps { 0 };
    std::atomic<std::uint32_t> clearCount { 0 };
    std::atomic<std::uint32_t> lockFailures { 0 };

    std::array<ChannelTelemetry, kMaxEffectChannels> channels;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (EffectsEngineCore)
};

} // namespace spatcore::effects
