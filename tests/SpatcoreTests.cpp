/*
    SpatcoreTests.cpp - minimal, dependency-free unit tests for spatcore
    (no gtest; plain asserts + exit code). Built as the `spatcore-tests`
    console app when SPATCORE_STANDALONE_TESTS=ON; run it -> exit 0 = pass.

    Coverage (Phase 6 prep - a seed, not a suite):
      1. rt/LockFreeRingBuffer   write/read roundtrip, wraparound, capacity clamp
      2. dsp/DelayTargetSmoother determinism: same observation sequence on two
                                 fresh instances -> bit-identical output streams
                                 (includes a teleport jump)
      3. rt/RtSnapshot           publish/acquire roundtrip; instantiating it is
                                 the compile-proof of the POD static_assert
                                 (a non-trivially-copyable T must NOT compile -
                                 can't be expressed in a passing build)
      4. control/osc parser+serializer   OSCSerializer::serializeMessage ->
                                 OSCParser::parseMessage roundtrip + byte-stable
                                 re-serialization
*/

// OSCParser.h / OSCSerializer.h use juce::OSC* types but (verbatim-moved,
// hygiene pass pending) only include juce_core themselves; every includer in
// the app provides juce_osc first, and so do we.
#include <juce_osc/juce_osc.h>

#include "spatcore/rt/LockFreeRingBuffer.h"
#include "spatcore/rt/RtSnapshot.h"
#include "spatcore/rt/RtThreadPriority.h"
#include "spatcore/gpu/GpuHostWorkPool.h"
#include "spatcore/dsp/DelayTargetSmoother.h"
#include "spatcore/control/osc/OSCSerializer.h"
#include "spatcore/control/osc/OSCParser.h"
#include "spatcore/reverb/ReverbSDNAlgorithm.h"
#include "spatcore/reverb/ReverbFDNAlgorithm.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;

#define CHECK(expr)                                                          \
    do                                                                       \
    {                                                                        \
        if (!(expr))                                                         \
        {                                                                    \
            std::fprintf (stderr, "FAIL %s:%d: %s\n",                        \
                          __FILE__, __LINE__, #expr);                        \
            ++failures;                                                      \
        }                                                                    \
    } while (false)

static bool bitEqualFloat (float a, float b) noexcept
{
    return std::memcmp (&a, &b, sizeof (float)) == 0;
}

//==============================================================================
static void testLockFreeRingBuffer()
{
    spatcore::rt::LockFreeRingBuffer rb;
    rb.setSize (8);   // one slot stays empty -> usable capacity 7

    float in[16], out[16];
    for (int i = 0; i < 16; ++i)
        in[i] = static_cast<float> (i + 1);

    CHECK (rb.getAvailableData() == 0);

    // Simple roundtrip
    CHECK (rb.write (in, 5) == 5);
    CHECK (rb.getAvailableData() == 5);
    CHECK (rb.read (out, 5) == 5);
    CHECK (std::memcmp (out, in, 5 * sizeof (float)) == 0);
    CHECK (rb.getAvailableData() == 0);

    // Wraparound: write pointer sits at 5 of 8; 6 samples span the seam
    CHECK (rb.write (in + 5, 6) == 6);
    CHECK (rb.getAvailableData() == 6);
    CHECK (rb.read (out, 6) == 6);
    CHECK (std::memcmp (out, in + 5, 6 * sizeof (float)) == 0);

    // Overfill clamps to capacity (7), data intact
    CHECK (rb.write (in, 16) == 7);
    CHECK (rb.read (out, 16) == 7);
    CHECK (std::memcmp (out, in, 7 * sizeof (float)) == 0);

    // reset() empties
    rb.write (in, 3);
    rb.reset();
    CHECK (rb.getAvailableData() == 0);
}

//==============================================================================
static std::vector<float> runSmootherSequence()
{
    spatcore::dsp::DelayTargetSmoother s;
    s.prepare (64);   // W = 64 samples -> teleport threshold 192

    // ~50 Hz-style observations at block boundaries; 500 - 120 = 380 > 192
    // exercises the teleport (mute-move-unmute) path.
    static const float targets[] = { 100.0f, 110.0f, 108.0f, 120.0f,
                                     500.0f, 505.0f, 490.0f, 495.0f,
                                     495.0f, 480.0f };
    const int blockSize = 48;

    std::vector<float> stream;
    stream.reserve (2 * blockSize * (sizeof (targets) / sizeof (targets[0])));

    std::int64_t t = 0;
    for (float target : targets)
    {
        s.observe (target, t);
        for (int i = 0; i < blockSize; ++i)
        {
            const auto o = s.smoothedAt (t + i);
            stream.push_back (o.delay);
            stream.push_back (o.gain);
        }
        t += blockSize;
    }
    return stream;
}

static void testDelayTargetSmootherDeterminism()
{
    const auto a = runSmootherSequence();
    const auto b = runSmootherSequence();

    CHECK (! a.empty());
    CHECK (a.size() == b.size());
    CHECK (std::memcmp (a.data(), b.data(), a.size() * sizeof (float)) == 0);

    // Basic sanity on the stream: finite delays, gains within [0, 1]
    for (size_t i = 0; i + 1 < a.size(); i += 2)
    {
        CHECK (std::isfinite (a[i]));
        CHECK (a[i + 1] >= 0.0f && a[i + 1] <= 1.0f);
    }
}

//==============================================================================
static void testRtSnapshot()
{
    struct Pod
    {
        float gain;
        int index;
        double position;
    };

    // Instantiation is the compile-proof of RtSnapshot's internal
    // static_assert(std::is_trivially_copyable_v<T>).
    spatcore::rt::RtSnapshot<Pod> snap;

    // Default-constructed snapshot is value-initialized (T value {})
    const Pod def = snap.acquire();
    CHECK (bitEqualFloat (def.gain, 0.0f));
    CHECK (def.index == 0);
    CHECK (def.position == 0.0);

    snap.publish ({ 1.5f, 42, -2.25 });
    const Pod got = snap.acquire();
    CHECK (bitEqualFloat (got.gain, 1.5f));
    CHECK (got.index == 42);
    CHECK (got.position == -2.25);

    // Second publish overwrites
    snap.publish ({ 0.25f, -7, 12.5 });
    const Pod got2 = snap.acquire();
    CHECK (bitEqualFloat (got2.gain, 0.25f));
    CHECK (got2.index == -7);
    CHECK (got2.position == 12.5);
}

//==============================================================================
static void testOscRoundtrip()
{
    juce::OSCMessage msg (juce::OSCAddressPattern ("/spatcore/test"));
    msg.addInt32 (42);
    msg.addFloat32 (3.5f);
    msg.addString ("hello");

    const juce::MemoryBlock bytes = spatcore::control::osc::OSCSerializer::serializeMessage (msg);
    CHECK (bytes.getSize() > 0);
    CHECK (bytes.getSize() % 4 == 0);   // OSC packets are 4-byte aligned

    int pos = 0;
    const juce::OSCMessage parsed = spatcore::control::osc::OSCParser::parseMessage (
        static_cast<const char*> (bytes.getData()),
        static_cast<int> (bytes.getSize()), pos);

    CHECK (pos == static_cast<int> (bytes.getSize()));
    CHECK (parsed.getAddressPattern().toString() == "/spatcore/test");
    CHECK (parsed.size() == 3);
    CHECK (parsed[0].isInt32() && parsed[0].getInt32() == 42);
    CHECK (parsed[1].isFloat32() && bitEqualFloat (parsed[1].getFloat32(), 3.5f));
    CHECK (parsed[2].isString() && parsed[2].getString() == "hello");

    // Encode(decode(x)) is byte-identical
    const juce::MemoryBlock bytes2 = spatcore::control::osc::OSCSerializer::serializeMessage (parsed);
    CHECK (bytes2 == bytes);
}

//==============================================================================
// RtThreadPriority smoke: elevating the calling thread and querying the core
// count must not crash and must return sane values. The elevation is a
// scheduling hint only (never affects computed audio), so this is a "does it
// run" check, not a value check — the return value is allowed to be false on a
// machine/policy that declines the request (e.g. no RLIMIT_RTPRIO on Linux, or
// avrt.dll absent on Windows -> HIGHEST fallback returns false).
static void testRtThreadPriority()
{
    // periodMs = one 128-sample block at 96 kHz; computationMs a slice of it.
    const bool elevated = spatcore::rt::setCurrentThreadAudioPriority (1.3333, 0.5);
    (void) elevated;   // platform/policy-dependent; must not crash regardless

    // A second call on the same thread must be idempotent (Windows: reuses the
    // per-thread MMCSS task handle rather than re-registering).
    spatcore::rt::setCurrentThreadAudioPriority (1.3333, 0.5);

    const int cores = spatcore::rt::physicalCoreCount();
    CHECK (cores >= 1);
    CHECK (cores <= 4096);   // sanity upper bound
}

//==============================================================================
// GpuHostWorkPool worker-count invariance: the SAME per-item workload run
// through parallelFor with 0 workers (sequential kill switch) and 3 workers
// must be BIT-identical. This is the executable form of the M3 determinism
// contract — each item writes only its own row and its FP sequence is a pure
// function of the item index, so the dynamic work-stealing schedule cannot
// affect the result. (The real backends' GPU 15/15 cross-check under
// WFS_GPU_HOST_WORKERS=0 vs =3 is the on-hardware version of this test.)
static std::vector<float> runPoolWorkload (int numWorkers)
{
    spatcore::gpu::GpuHostWorkPool pool;
    pool.prepare (numWorkers, 1.3333, 0.5);

    const int count  = 257;   // deliberately not a multiple of the worker count
    const int rowLen = 31;
    std::vector<float> out ((size_t) count * (size_t) rowLen, 0.0f);

    pool.parallelFor (count, [&] (int i)
    {
        // Per-item state partitioning: item i touches ONLY its own row, and the
        // sample sequence is a pure function of i — no cross-item accumulation.
        float* row = out.data() + (size_t) i * (size_t) rowLen;
        float acc = (float) i * 0.5f;
        for (int s = 0; s < rowLen; ++s)
        {
            acc = acc * 0.9999f + std::sin ((float) (i + 1) * 0.017f * (float) (s + 1));
            row[s] = acc;
        }
    });

    pool.shutdown();
    return out;
}

static void testGpuHostWorkPoolDeterminism()
{
    const auto seq = runPoolWorkload (0);   // sequential (kill switch)
    const auto par = runPoolWorkload (3);   // 3 workers + caller = 4 lanes

    CHECK (! seq.empty());
    CHECK (seq.size() == par.size());
    CHECK (std::memcmp (seq.data(), par.data(), seq.size() * sizeof (float)) == 0);

    for (float v : seq)
        CHECK (std::isfinite (v));
}

// Cross-generation barrier stress test. The pump calls parallelFor once per
// audio block, back-to-back, forever. A worker that finishes the LAST item of
// generation N must not bleed into generation N+1's item state (the M3 audit's
// confirmed critical race: without a worker-quiescence barrier a straggler
// re-reads nextItem/currentFunc that the next call is overwriting -> torn read /
// use-after-free of the previous call's captured frame). Here: many tight
// back-to-back generations with a small item count (workers finish fast, so the
// finish->redispatch window is narrow and hit often), each generation carrying a
// UNIQUE base captured by a fresh lambda and validating its OWN result. On the
// pre-fix code this reliably mismatches or crashes on a multicore box; with the
// generation barrier it must pass. (Probabilistic by nature — a stress guard,
// not a proof; the proof is the audit + the ordering argument in the header.)
static void testGpuHostWorkPoolCrossGenBarrier()
{
    // Oversubscribe: many more workers than cores forces the OS to preempt a
    // worker constantly, so the finish->redispatch window (a worker preempted
    // right after the last item, before it re-checks the item counter) is hit
    // often. With a tiny item count most workers find no work and race straight
    // to the completion barrier — exactly the interleaving the bug needs.
    const unsigned hw = std::thread::hardware_concurrency();
    const int workers = (int) (hw > 0 ? hw * 2u : 8u);   // oversubscribed
    const int count   = 3;               // tiny => tight finish/redispatch window

    // The pool is RE-PREPARED many times (backends re-prepare on any SR/block/
    // channel change: release()->pool.shutdown() then pool.prepare()). The first
    // parallelFor after each fresh prepare is the window for the phantom-
    // generation defect (fresh workers seed myGen=0 while a stale dispatchGen>0
    // would make them serve a bogus generation). So: many prepares, each followed
    // immediately by tight back-to-back generations, all validated per round.
    spatcore::gpu::GpuHostWorkPool pool;
    std::vector<int> out ((size_t) count, 0);

    const int prepares         = 500;    // 500 fresh-prepare (phantom) windows
    const int roundsPerPrepare = 200;    // tight back-to-back gens after each

    int g = 0;
    for (int p = 0; p < prepares; ++p)
    {
        pool.prepare (workers, 1.3333, 0.5);   // re-prepare must reset dispatchGen

        for (int r = 0; r < roundsPerPrepare; ++r)
        {
            const int base = (++g) * 7 + 1;    // unique per generation
            std::fill (out.begin(), out.end(), -1);

            // Fresh lambda each round; its captured frame is destroyed when this
            // parallelFor returns, so a bled straggler invoking a STALE currentFunc
            // would read a destroyed capture (UAF), write a wrong base, or hit a
            // nullptr currentFunc (std::bad_function_call).
            pool.parallelFor (count, [&out, base] (int i) { out[(size_t) i] = base + i; });

            for (int i = 0; i < count; ++i)
                CHECK (out[(size_t) i] == base + i);
        }
    }

    pool.shutdown();
}

//==============================================================================
// SDN output level vs node count.
//
// Renders the algorithm's impulse response (node-feed impulse in, per-node wet
// out) and integrates total output energy across all nodes. Historically the
// SDN staging (1/N injection x (1+18/N)*0.25 output gain) made the reverb
// drop several dB as the mesh grew — a 19-node venue mesh sat ~5-10 dB under
// the documented 9-13 node sweet spot and far under FDN at the same wet level.
// The output gain law k*N/sqrt(N-1) flattens the sparse-feed response vs N;
// this test measures the curve and pins it.
namespace sdnlevel {

constexpr double kSr = 48000.0;
constexpr int    kBlock = 512;

static void ringGeometry (std::vector<spatcore::reverb::NodePosition>& nodes, int n)
{
    nodes.resize (static_cast<size_t> (n));
    for (int i = 0; i < n; ++i)
    {
        const float a = juce::MathConstants<float>::twoPi * (float) i / (float) n;
        nodes[(size_t) i] = { 4.0f * std::sin (a), 4.0f * std::cos (a), 3.0f };
    }
}

// Total impulse-response energy summed over all node outputs, in dB.
static double measureEnergyDb (spatcore::reverb::ReverbAlgorithm& algo,
                               int numNodes, bool feedAllNodes, double seconds)
{
    const int totalSamples = (int) (kSr * seconds);
    juce::AudioBuffer<float> in (numNodes, kBlock), out (numNodes, kBlock);
    double energy = 0.0;
    bool impulseSent = false;

    for (int done = 0; done < totalSamples; done += kBlock)
    {
        in.clear();
        out.clear();
        if (! impulseSent)
        {
            const int last = feedAllNodes ? numNodes : 1;
            for (int n = 0; n < last; ++n)
                in.setSample (n, 0, 1.0f);
            impulseSent = true;
        }

        algo.processBlock (in, out, kBlock);

        for (int n = 0; n < numNodes; ++n)
        {
            const float* p = out.getReadPointer (n);
            for (int s = 0; s < kBlock; ++s)
                energy += (double) p[s] * (double) p[s];
        }
    }

    return 10.0 * std::log10 (juce::jmax (energy, 1e-30));
}

static double sdnEnergyDb (int numNodes, bool feedAllNodes, double seconds = 2.0)
{
    spatcore::reverb::SDNAlgorithm sdn;
    sdn.prepare (kSr, kBlock, numNodes);

    std::vector<spatcore::reverb::NodePosition> nodes;
    ringGeometry (nodes, numNodes);
    sdn.updateGeometry (nodes);

    spatcore::reverb::AlgorithmParameters params;
    params.rt60 = 2.0f;
    sdn.setParameters (params);

    return measureEnergyDb (sdn, numNodes, feedAllNodes, seconds);
}

static double fdnEnergyDb (int numNodes, double seconds = 2.0)
{
    spatcore::reverb::FDNAlgorithm fdn;
    fdn.prepare (kSr, kBlock, numNodes);

    spatcore::reverb::AlgorithmParameters params;
    params.rt60 = 2.0f;
    fdn.setParameters (params);

    // FDN nodes are independent; a single-node feed exercises one FDN.
    return measureEnergyDb (fdn, numNodes, false, seconds);
}

} // namespace sdnlevel

static void testSdnLevelVsNodeCount()
{
    using namespace sdnlevel;

    const int counts[] = { 9, 11, 13, 19, 25, 32 };
    constexpr int numCounts = (int) (sizeof (counts) / sizeof (counts[0]));

    const double fdnRefDb  = fdnEnergyDb (11);
    const double anchorDb  = sdnEnergyDb (11, false);
    std::printf ("SDN level vs node count (impulse-response energy, rt60=2s):\n");
    std::printf ("  FDN reference (any N, independent nodes): %+7.2f dB\n", fdnRefDb);

    // Absolute anchor: the N=11 sweet-spot level is the calibration point of
    // the k=0.1895 constant (chosen to preserve the pre-renormalization
    // hand-tuned level). Moving it means deliberately recalibrating k.
    CHECK (std::abs (anchorDb - (-6.0)) < 1.0);

    // SDN should sit in the same loudness ballpark as FDN at the sweet spot.
    CHECK (std::abs (anchorDb - fdnRefDb) < 3.0);

    for (int i = 0; i < numCounts; ++i)
    {
        const int n = counts[i];
        const double sparseDb = sdnEnergyDb (n, false);
        const double denseDb  = sdnEnergyDb (n, true);
        std::printf ("  N=%2d  sparse %+7.2f dB   dense %+7.2f dB\n",
                     n, sparseDb, denseDb);

        // Level must be ~flat vs node count (this was the venue bug: the old
        // (1+18/N)*0.25 law drooped ~6 dB from N=9 to N=19).
        CHECK (std::abs (sparseDb - anchorDb) < 1.5);

        // Dense feed (all nodes hit with a unity impulse at once, input
        // energy = N = 10*log10(N) dB) must not run away.
        CHECK (denseDb < 10.0 * std::log10 ((double) n) + 3.0);
    }
}

//==============================================================================
int main()
{
    try
    {
        testLockFreeRingBuffer();
        testDelayTargetSmootherDeterminism();
        testRtSnapshot();
        testOscRoundtrip();
        testRtThreadPriority();
        testGpuHostWorkPoolDeterminism();
        testGpuHostWorkPoolCrossGenBarrier();
        testSdnLevelVsNodeCount();
    }
    catch (const std::exception& e)
    {
        std::fprintf (stderr, "FAIL: unexpected exception: %s\n", e.what());
        ++failures;
    }

    if (failures == 0)
    {
        std::printf ("spatcore-tests: all tests passed\n");
        return 0;
    }

    std::fprintf (stderr, "spatcore-tests: %d check(s) FAILED\n", failures);
    return 1;
}
