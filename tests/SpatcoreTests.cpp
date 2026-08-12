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
      5. dsp/ shared parametric EQ   MultiChannelEQBank neutrality (bit-exact)
                                 and enable semantics; bank == a hand-rolled
                                 std::array of biquads (bit-exact); the static
                                 calculateCoefficients() is provably the
                                 coefficient set the audio path runs;
                                 biquadMagnitudeDb sanity + a golden coefficient
                                 table for both filter classes; OutputEQProcessor
                                 neutral at defaults
      6. io/HardwareIndexMap     hardware -> compact callback index translation:
                                 contiguous (identity), sparse (holes shift
                                 every index), empty, and policy clamping
      7. io/DeviceHost           the enable-all channel-mask policy, including
                                 clearing useDefault*Channels - without which
                                 JUCE silently replaces the mask
      8. io/TestSignalGenerator  tone pitch survives prepare()/sample-rate
                                 change; the 500 ms protective ramp; seeded
                                 pink noise is reproducible
*/

// OSCParser.h / OSCSerializer.h use juce::OSC* types but (verbatim-moved,
// hygiene pass pending) only include juce_core themselves; every includer in
// the app provides juce_osc first, and so do we.
#include <juce_osc/juce_osc.h>

// Both biquad headers clamp their parameters with std::min/std::max but include
// only <cmath> themselves (in the app they get <algorithm> transitively from
// JUCE). MultiChannelEQBank.h works around that for its own filter include;
// ReverbBiquadFilter.h is reached directly from here, so make it visible first.
#include <algorithm>

#include "spatcore/rt/LockFreeRingBuffer.h"
#include "spatcore/rt/RtSnapshot.h"
#include "spatcore/rt/RtThreadPriority.h"
#include "spatcore/gpu/GpuHostWorkPool.h"
#include "spatcore/dsp/DelayTargetSmoother.h"
#include "spatcore/dsp/BiquadResponse.h"
#include "spatcore/dsp/OutputEQBiquadFilter.h"
#include "spatcore/dsp/ReverbBiquadFilter.h"
#include "spatcore/dsp/MultiChannelEQBank.h"
#include "spatcore/dsp/OutputEQProcessor.h"
#include "spatcore/control/osc/OSCSerializer.h"
#include "spatcore/control/osc/OSCParser.h"
#include "spatcore/reverb/ReverbSDNAlgorithm.h"
#include "spatcore/reverb/ReverbFDNAlgorithm.h"
#include "spatcore/io/HardwareIndexMap.h"
#include "spatcore/io/DeviceHost.h"
#include "spatcore/io/TestSignalGenerator.h"
#include "spatcore/binaural/HeadFrame.h"
#include "spatcore/binaural/BinauralEngine.h"
#include "spatcore/binaural/HeadOrientationSource.h"
#include "spatcore/binaural/StructuralHrtfRenderer.h"
#include "spatcore/dsp/OneEuroFilter.h"
#ifdef SPATCORE_TEST_SOFA_FIXTURE
#include "spatcore/binaural/SofaLoader.h"
#include "spatcore/binaural/SofaHrtfRenderer.h"
#endif

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
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
// Shared parametric EQ: BiquadResponse / OutputEQBiquadFilter /
// ReverbBiquadFilter / MultiChannelEQBank / OutputEQProcessor.
//
// The bank is the piece WFS-DIY's OutputEQProcessor and XOA's
// SpeakerCompProcessor both migrate onto, and the static calculateCoefficients()
// is what the GUI response curves draw from. Two properties have to hold or the
// extraction is a regression:
//   - a neutral EQ is BIT-transparent (the golden-render gate compares renders
//     byte-for-byte, so "close enough" is a failure);
//   - display math and audio math are the same math (a second copy of the
//     cookbook formulas drifts, historically on the shelves' S parameter).
namespace eqtests {

constexpr int    kNumBands = 6;
constexpr double kSampleRate = 48000.0;

//==============================================================================
/** One band's worth of parameters, in the plain-int shape mapping the filters
    document (no app enums cross the spatcore boundary). */
struct BandSetting
{
    int   shape;
    float freq, gainDb, q, slope;
};

//==============================================================================
/** Direct-form-I reference biquad: a transcription of the filter classes'
    processSample(). Everything that pins "the coefficients the GUI is shown are
    the coefficients the audio path runs" compares against this, so the
    difference-equation expression below must stay character-identical to the one
    in OutputEQBiquadFilter.h / ReverbBiquadFilter.h — including the operand
    order, which is what makes the comparison bit-exact rather than approximate. */
struct RefBiquad
{
    void setCoefficients (const spatcore::dsp::BiquadCoefficients& c) noexcept
    {
        b0 = c.b0; b1 = c.b1; b2 = c.b2; a1 = c.a1; a2 = c.a2;
    }

    float processSample (float input) noexcept
    {
        float output = b0 * input + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = input;
        y2 = y1;
        y1 = output;
        return output;
    }

    float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f;
};

//==============================================================================
/** A non-trivial signal seeded with the values a "harmless" identity multiply
    would silently damage, so a bit-exact comparison against it actually bites:

      - a NEGATIVE ZERO. Running it through the identity biquad yields
        1*(-0) + 0*0 + 0*0 - 0*0 - 0*0 == +0.0f, a different bit pattern. This is
        the value that catches a bank which "processes" a shape-0 band instead of
        skipping it.
      - two DENORMALS, which an FTZ/DAZ-armed multiply flushes to zero. They are
        constexpr so the compiler materialises the bit patterns directly —
        computing them at run time could flush them before they ever reach the
        buffer.
      - two LARGE values, to catch a scaling slip that rounds off in the noise
        floor of an epsilon compare.
*/
static std::vector<float> makeAwkwardSignal (int numSamples)
{
    constexpr float tinyDenormal  = std::numeric_limits<float>::denorm_min();
    constexpr float smallDenormal = std::numeric_limits<float>::min() * 0.5f;

    std::vector<float> v (static_cast<size_t> (std::max (0, numSamples)), 0.0f);

    for (size_t i = 0; i < v.size(); ++i)
        v[i] = 0.37f * std::sin (0.11f * static_cast<float> (i))
             + 0.13f * std::sin (1.90f * static_cast<float> (i) + 0.7f);

    if (v.size() >= 8)
    {
        v[0] = std::copysign (0.0f, -1.0f);   // negative zero
        v[1] = tinyDenormal;
        v[2] = smallDenormal;
        v[3] = 1.0e7f;
        v[4] = -1.0e7f;
        v[5] = 0.0f;
    }

    return v;
}

static bool bitEqualBlock (const std::vector<float>& a, const std::vector<float>& b) noexcept
{
    return a.size() == b.size()
        && std::memcmp (a.data(), b.data(), a.size() * sizeof (float)) == 0;
}

//==============================================================================
/** Six bands that all do something, used to drive the bank and its hand-rolled
    twin through the same parameter sequence. Shapes are the OutputEQBiquadFilter
    mapping: 1 LowCut, 2 LowShelf, 3 Peak, 4 BandPass, 5 HighShelf, 6 HighCut,
    7 AllPass. */
static const BandSetting kTickA[kNumBands] =
{
    { 1,   80.0f,  0.0f, 0.707f, 0.7f },
    { 2,  250.0f,  6.0f, 0.7f,   0.7f },
    { 3, 1000.0f, -9.0f, 3.0f,   0.7f },
    { 5, 5000.0f,  4.0f, 0.7f,   0.5f },
    { 6, 9000.0f,  0.0f, 0.707f, 0.7f },
    { 7, 1500.0f,  0.0f, 0.5f,   0.7f },
};

static const BandSetting kTickB[kNumBands] =
{
    { 1,  120.0f,  0.0f, 1.2f,   0.7f },
    { 2,  250.0f,  6.0f, 0.7f,   0.7f },   // unchanged -> exercises the no-change short circuit
    { 3,  900.0f,  6.0f, 0.8f,   0.7f },
    { 4, 3000.0f,  0.0f, 1.5f,   0.7f },
    { 0, 5000.0f,  4.0f, 0.7f,   0.5f },   // a band switched OFF mid-sequence
    { 7, 1500.0f,  0.0f, 0.5f,   0.7f },
};

//==============================================================================
/** Runs one shape through both paths and requires bit-identical output:
      - a prepare()/setParameters()-configured filter instance, and
      - RefBiquad fed the coefficients the STATIC calculateCoefficients() returns
        for the very same arguments.
    Templated because OutputEQBiquadFilter and ReverbBiquadFilter expose the same
    surface with different shape numbering. */
template <typename FilterType>
static void checkCoefficientsDriveAudio (const char* label, const BandSetting& s, double sampleRate)
{
    FilterType f;
    f.prepare (sampleRate);
    f.setParameters (s.shape, s.freq, s.gainDb, s.q, s.slope);
    CHECK (f.getShape() == s.shape);
    CHECK (f.isActive() == (s.shape != 0));

    const auto coeffs = FilterType::calculateCoefficients (s.shape, s.freq, s.gainDb,
                                                           s.q, s.slope, sampleRate);
    CHECK (coeffs.active == (s.shape != 0));

    constexpr int n = 128;
    std::vector<float> audio (static_cast<size_t> (n), 0.0f);
    std::vector<float> want  (static_cast<size_t> (n), 0.0f);
    audio[0] = 1.0f;   // unit impulse -> the filter's impulse response

    // Reference first: processBlock() rewrites `audio` in place.
    RefBiquad ref;
    ref.setCoefficients (coeffs);
    for (size_t i = 0; i < want.size(); ++i)
        want[i] = ref.processSample (audio[i]);

    f.processBlock (audio.data(), n);

    const bool identical = bitEqualBlock (audio, want);
    CHECK (identical);

    if (! identical)
        std::fprintf (stderr,
                      "  %s shape %d (%.1f Hz, %.1f dB, Q %.3f, S %.3f, %.0f Hz SR): "
                      "impulse response differs from the static coefficients\n",
                      label, s.shape, s.freq, s.gainDb, s.q, s.slope, sampleRate);

    for (float v : audio)
        CHECK (std::isfinite (v));
}

//==============================================================================
/** A pinned (parameters -> coefficients) row. Expected values were computed by
    hand from the cookbook formulas in the headers, in float precision, and are
    compared with a tolerance loose enough for last-bit libm differences across
    compilers but far tighter than any real formula change. */
struct GoldenCase
{
    int    shape;
    float  freq, gainDb, q, slope;
    double sampleRate;
    float  b0, b1, b2, a1, a2;
};

constexpr float kCoeffTolerance = 1.0e-5f;

static void checkGolden (const char* label, const GoldenCase& g,
                         const spatcore::dsp::BiquadCoefficients& c)
{
    const bool ok = std::abs (c.b0 - g.b0) <= kCoeffTolerance
                 && std::abs (c.b1 - g.b1) <= kCoeffTolerance
                 && std::abs (c.b2 - g.b2) <= kCoeffTolerance
                 && std::abs (c.a1 - g.a1) <= kCoeffTolerance
                 && std::abs (c.a2 - g.a2) <= kCoeffTolerance;

    CHECK (ok);

    if (! ok)
        std::fprintf (stderr,
                      "  %s shape %d (%.1f Hz, %.1f dB, Q %.3f, S %.3f, %.0f Hz SR)\n"
                      "    want b %.9g %.9g %.9g   a %.9g %.9g\n"
                      "    got  b %.9g %.9g %.9g   a %.9g %.9g\n",
                      label, g.shape, g.freq, g.gainDb, g.q, g.slope, g.sampleRate,
                      g.b0, g.b1, g.b2, g.a1, g.a2,
                      c.b0, c.b1, c.b2, c.a1, c.a2);
}

} // namespace eqtests

//==============================================================================
// 1. Bank neutrality. A prepared bank whose bands are all shape 0, and a bank
// whose channel is disabled, must both leave the buffer BIT-identical — not
// "inaudibly close". The WFS-DIY golden-render gate compares rendered files
// byte-for-byte, so a spurious identity multiply (y = 1.0f*x + 0*... ) is a
// regression even though it is mathematically a no-op: it turns a negative zero
// into a positive zero, and flushes denormals under FTZ/DAZ.
static void testMultiChannelEQBankNeutrality()
{
    using namespace eqtests;

    const auto input = makeAwkwardSignal (256);
    const int  n     = static_cast<int> (input.size());

    // The awkward values must actually be awkward, or this test proves nothing.
    CHECK (std::signbit (input[0]) && input[0] == 0.0f);
    CHECK (input[1] > 0.0f && input[1] < std::numeric_limits<float>::min());
    CHECK (input[2] > 0.0f && input[2] < std::numeric_limits<float>::min());

    // (a) Channel ENABLED, every band pushed at shape 0 (OFF).
    {
        spatcore::dsp::MultiChannelEQBank<kNumBands> bank;
        bank.prepare (kSampleRate, 4);
        CHECK (bank.getNumChannels() == 4);

        bank.setChannelEnabled (1, true);
        CHECK (bank.isChannelEnabled (1));

        for (int b = 0; b < kNumBands; ++b)
            bank.pushBandParameters (1, b, 0, 1000.0f, 6.0f, 1.0f, 0.7f);

        auto buffer = input;
        bank.processChannel (1, buffer.data(), n);
        CHECK (bitEqualBlock (buffer, input));
    }

    // (b) Channel DISABLED (the state prepare() leaves every channel in).
    {
        spatcore::dsp::MultiChannelEQBank<kNumBands> bank;
        bank.prepare (kSampleRate, 4);
        CHECK (! bank.isChannelEnabled (0));

        auto buffer = input;
        bank.processChannel (0, buffer.data(), n);
        CHECK (bitEqualBlock (buffer, input));
    }

    // Out-of-range indices are silent no-ops, not faults: a stale channel count
    // arriving from the message thread must never reach the audio thread.
    {
        spatcore::dsp::MultiChannelEQBank<kNumBands> bank;
        bank.prepare (kSampleRate, 2);

        CHECK (! bank.isChannelEnabled (-1));
        CHECK (! bank.isChannelEnabled (99));
        bank.setChannelEnabled (99, true);
        CHECK (! bank.isChannelEnabled (99));
        bank.pushBandParameters (99, 0, 3, 1000.0f, 6.0f, 1.0f, 0.7f);
        bank.pushBandParameters (0, 99, 3, 1000.0f, 6.0f, 1.0f, 0.7f);
        bank.pushBandParameters (0, -1, 3, 1000.0f, 6.0f, 1.0f, 0.7f);

        auto buffer = input;
        bank.processChannel (99, buffer.data(), n);
        bank.processChannel (0, nullptr, n);
        bank.processChannel (0, buffer.data(), 0);
        CHECK (bitEqualBlock (buffer, input));
    }
}

//==============================================================================
// 2. Enable semantics — the "disabled channel forces shape 0" contract and the
// push ORDERING it implies. Both apps push the enable flag and then the bands,
// unconditionally, from a timer tick; pushing them the other way round latches
// the previous enable state into the coefficients for one tick. This test pins
// that: parameters pushed while disabled leave the channel a pass-through even
// after it is later enabled, and only a re-push makes it active.
static void testMultiChannelEQBankEnableSemantics()
{
    using namespace eqtests;

    const auto input = makeAwkwardSignal (256);
    const int  n     = static_cast<int> (input.size());

    spatcore::dsp::MultiChannelEQBank<kNumBands> bank;
    bank.prepare (kSampleRate, 2);
    CHECK (! bank.isChannelEnabled (0));

    // Push real, non-zero shapes while the channel is DISABLED. Every band must
    // be forced to shape 0.
    for (int b = 0; b < kNumBands; ++b)
        bank.pushBandParameters (0, b, kTickA[b].shape, kTickA[b].freq,
                                 kTickA[b].gainDb, kTickA[b].q, kTickA[b].slope);

    {
        auto buffer = input;
        bank.processChannel (0, buffer.data(), n);
        CHECK (bitEqualBlock (buffer, input));   // disabled -> processChannel no-ops
    }

    // Enabling WITHOUT re-pushing must still be a pass-through: the forced
    // shape 0 is latched in the coefficients, not re-derived at process time.
    // (This is exactly the one-tick artefact the ordering contract avoids.)
    bank.setChannelEnabled (0, true);
    CHECK (bank.isChannelEnabled (0));

    {
        auto buffer = input;
        bank.processChannel (0, buffer.data(), n);
        CHECK (bitEqualBlock (buffer, input));
    }

    // Re-pushing the SAME parameters now that the channel is enabled must make
    // it active.
    for (int b = 0; b < kNumBands; ++b)
        bank.pushBandParameters (0, b, kTickA[b].shape, kTickA[b].freq,
                                 kTickA[b].gainDb, kTickA[b].q, kTickA[b].slope);

    {
        auto buffer = input;
        bank.processChannel (0, buffer.data(), n);
        CHECK (! bitEqualBlock (buffer, input));

        for (float v : buffer)
            CHECK (std::isfinite (v));
    }

    // Channel 1 was never touched: per-channel state really is independent.
    {
        auto buffer = input;
        bank.processChannel (1, buffer.data(), n);
        CHECK (bitEqualBlock (buffer, input));
    }

    // Disabling again silences the chain at the process() gate, and a push made
    // while disabled once more forces the bands back to OFF.
    bank.setChannelEnabled (0, false);

    for (int b = 0; b < kNumBands; ++b)
        bank.pushBandParameters (0, b, kTickA[b].shape, kTickA[b].freq,
                                 kTickA[b].gainDb, kTickA[b].q, kTickA[b].slope);

    bank.setChannelEnabled (0, true);

    {
        auto buffer = input;
        bank.processChannel (0, buffer.data(), n);
        CHECK (bitEqualBlock (buffer, input));
    }
}

//==============================================================================
// 3. Bank equivalence. One bank channel and a hand-rolled
// std::array<OutputEQBiquadFilter, 6> — the shape both WFS-DIY's
// OutputEQProcessor and XOA's SpeakerCompProcessor had before the extraction —
// fed the same parameter sequence and the same audio must produce BIT-identical
// output. This is the test that says the migrations onto the shared bank are
// renders-unchanged.
static void testMultiChannelEQBankEquivalence()
{
    using namespace eqtests;

    spatcore::dsp::MultiChannelEQBank<kNumBands> bank;
    bank.prepare (kSampleRate, 3);
    bank.setChannelEnabled (2, true);

    std::array<spatcore::dsp::OutputEQBiquadFilter, kNumBands> hand;
    for (auto& f : hand)
        f.prepare (kSampleRate);

    auto pushBoth = [&] (const BandSetting* tick)
    {
        for (int b = 0; b < kNumBands; ++b)
        {
            bank.pushBandParameters (2, b, tick[b].shape, tick[b].freq,
                                     tick[b].gainDb, tick[b].q, tick[b].slope);
            hand[static_cast<size_t> (b)].setParameters (tick[b].shape, tick[b].freq,
                                                         tick[b].gainDb, tick[b].q,
                                                         tick[b].slope);
        }
    };

    auto processHand = [&] (float* samples, int numSamples)
    {
        for (int b = 0; b < kNumBands; ++b)
            hand[static_cast<size_t> (b)].processBlock (samples, numSamples);
    };

    const auto noise = makeAwkwardSignal (192);
    const int  n     = static_cast<int> (noise.size());

    // Parameter sequence: tick A, tick A again (no-change short circuit), then
    // tick B — including a band switched OFF mid-stream. Filter state carries
    // across the blocks on both sides, so a divergence anywhere accumulates.
    const BandSetting* const ticks[] = { kTickA, kTickA, kTickB, kTickB };

    bool sawActiveOutput = false;

    for (const BandSetting* tick : ticks)
    {
        pushBoth (tick);

        for (int block = 0; block < 3; ++block)
        {
            auto viaBank = noise;
            auto viaHand = noise;

            bank.processChannel (2, viaBank.data(), n);
            processHand (viaHand.data(), n);

            CHECK (bitEqualBlock (viaBank, viaHand));

            if (! bitEqualBlock (viaBank, noise))
                sawActiveOutput = true;

            for (float v : viaBank)
                CHECK (std::isfinite (v));
        }
    }

    // Guard against the whole comparison being trivially satisfied by two
    // pass-throughs.
    CHECK (sawActiveOutput);
}

//==============================================================================
// 4. The static calculateCoefficients() IS the audio path. The filters do not
// expose their coefficients, so this is verified behaviourally: an impulse
// through a setParameters()-configured filter, and the same impulse through a
// direct-form-I difference equation driven by the static call's return value,
// must match bit-for-bit. Every shape of BOTH classes (their shape numbering
// differs on purpose) at representative parameters, across three sample rates.
// This is what stops the GUI response curve and the audio path drifting apart.
static void testBiquadCoefficientsMatchAudioPath()
{
    using namespace eqtests;

    // OutputEQBiquadFilter: 0 OFF, 1 LowCut, 2 LowShelf, 3 Peak, 4 BandPass,
    // 5 HighShelf, 6 HighCut, 7 AllPass.
    static const BandSetting outputCases[] =
    {
        { 0, 1000.0f,  0.0f, 0.707f, 0.7f },
        { 1,  120.0f,  0.0f, 0.707f, 0.7f },
        { 2,  250.0f,  6.0f, 0.7f,   0.7f },
        { 2,  250.0f, -6.0f, 0.7f,   0.4f },
        { 3, 1000.0f,  9.0f, 3.0f,   0.7f },
        { 3, 1000.0f, -9.0f, 0.5f,   0.7f },
        { 4, 2000.0f,  0.0f, 1.5f,   0.7f },
        { 5, 5000.0f,  6.0f, 0.7f,   0.5f },
        { 5, 5000.0f, -3.0f, 0.7f,   1.0f },
        { 6, 8000.0f,  0.0f, 0.707f, 0.7f },
        { 7, 1000.0f,  0.0f, 0.5f,   0.7f },
    };

    // ReverbBiquadFilter: 0 OFF, 1 LowCut, 2 LowShelf, 3 Peak, 4 HighShelf,
    // 5 HighCut, 6 BandPass — deliberately NOT the Output EQ order.
    static const BandSetting reverbCases[] =
    {
        { 0, 1000.0f,  0.0f, 0.707f, 0.7f },
        { 1,   80.0f,  0.0f, 0.707f, 0.7f },
        { 2,  200.0f,  4.0f, 0.7f,   0.8f },
        { 2,  200.0f, -4.0f, 0.7f,   0.3f },
        { 3,  800.0f,  9.0f, 3.0f,   0.7f },
        { 3,  800.0f, -9.0f, 0.5f,   0.7f },
        { 4, 4000.0f,  5.0f, 0.7f,   1.0f },
        { 4, 4000.0f, -5.0f, 0.7f,   0.6f },
        { 5, 6000.0f,  0.0f, 0.707f, 0.7f },
        { 6, 1500.0f,  0.0f, 1.5f,   0.7f },
    };

    static const double rates[] = { 44100.0, 48000.0, 96000.0 };

    for (double sr : rates)
    {
        for (const auto& c : outputCases)
            checkCoefficientsDriveAudio<spatcore::dsp::OutputEQBiquadFilter> ("OutputEQ", c, sr);

        for (const auto& c : reverbCases)
            checkCoefficientsDriveAudio<spatcore::dsp::ReverbBiquadFilter> ("ReverbEQ", c, sr);
    }

    // Shape 0 and a degenerate sample rate both design to the inactive identity.
    for (const auto& c : { spatcore::dsp::OutputEQBiquadFilter::calculateCoefficients (0, 1000.0f, 6.0f, 1.0f, 0.7f, kSampleRate),
                           spatcore::dsp::OutputEQBiquadFilter::calculateCoefficients (3, 1000.0f, 6.0f, 1.0f, 0.7f, 0.0),
                           spatcore::dsp::ReverbBiquadFilter::calculateCoefficients   (0, 1000.0f, 6.0f, 1.0f, 0.7f, kSampleRate),
                           spatcore::dsp::ReverbBiquadFilter::calculateCoefficients   (3, 1000.0f, 6.0f, 1.0f, 0.7f, -1.0) })
    {
        CHECK (! c.active);
        CHECK (bitEqualFloat (c.b0, 1.0f));
        CHECK (bitEqualFloat (c.b1, 0.0f));
        CHECK (bitEqualFloat (c.b2, 0.0f));
        CHECK (bitEqualFloat (c.a1, 0.0f));
        CHECK (bitEqualFloat (c.a2, 0.0f));
    }

    // Out-of-range shapes clamp exactly as setParameters() clamps them, so a
    // caller feeding raw parameter values agrees with the audio path.
    {
        const auto hi   = spatcore::dsp::OutputEQBiquadFilter::calculateCoefficients (99, 1000.0f, 0.0f, 0.5f, 0.7f, kSampleRate);
        const auto ap   = spatcore::dsp::OutputEQBiquadFilter::calculateCoefficients (7,  1000.0f, 0.0f, 0.5f, 0.7f, kSampleRate);
        CHECK (bitEqualFloat (hi.b0, ap.b0) && bitEqualFloat (hi.a2, ap.a2));

        const auto lo   = spatcore::dsp::OutputEQBiquadFilter::calculateCoefficients (-5, 1000.0f, 0.0f, 0.5f, 0.7f, kSampleRate);
        CHECK (! lo.active);

        // Frequency / Q / gain clamps are idempotent.
        const auto wild    = spatcore::dsp::ReverbBiquadFilter::calculateCoefficients (3, 1.0e9f, 400.0f, 1.0e6f, 1.0e6f, kSampleRate);
        const auto clamped = spatcore::dsp::ReverbBiquadFilter::calculateCoefficients (3, 20000.0f, 24.0f, 20.0f, 20.0f, kSampleRate);
        CHECK (bitEqualFloat (wild.b0, clamped.b0));
        CHECK (bitEqualFloat (wild.b1, clamped.b1));
        CHECK (bitEqualFloat (wild.b2, clamped.b2));
        CHECK (bitEqualFloat (wild.a1, clamped.a1));
        CHECK (bitEqualFloat (wild.a2, clamped.a2));
    }
}

//==============================================================================
// 5. biquadMagnitudeDb sanity. Loose tolerances on purpose — this is a net that
// catches a curve drawn upside-down, off by an octave, or in the wrong units,
// not a golden lock (testBiquadGoldenCoefficients is the lock).
static void testBiquadMagnitudeResponse()
{
    using spatcore::dsp::BiquadCoefficients;
    using spatcore::dsp::biquadMagnitudeDb;
    using spatcore::dsp::OutputEQBiquadFilter;
    using spatcore::dsp::ReverbBiquadFilter;

    constexpr double sr = eqtests::kSampleRate;

    // A default-constructed (== OFF) coefficient set reads exactly flat.
    CHECK (bitEqualFloat (biquadMagnitudeDb (BiquadCoefficients {}, 1000.0f, sr), 0.0f));

    const auto off = OutputEQBiquadFilter::calculateCoefficients (0, 1000.0f, 12.0f, 1.0f, 0.7f, sr);
    CHECK (bitEqualFloat (biquadMagnitudeDb (off, 20.0f,    sr), 0.0f));
    CHECK (bitEqualFloat (biquadMagnitudeDb (off, 1000.0f,  sr), 0.0f));
    CHECK (bitEqualFloat (biquadMagnitudeDb (off, 20000.0f, sr), 0.0f));

    // Peak: |H(f0)| == A^2 == the requested gain, flat far from the centre.
    const auto peak = OutputEQBiquadFilter::calculateCoefficients (3, 1000.0f, 6.0f, 1.0f, 0.7f, sr);
    CHECK (peak.active);
    CHECK (std::abs (biquadMagnitudeDb (peak, 1000.0f, sr) - 6.0f) < 0.25f);
    CHECK (std::abs (biquadMagnitudeDb (peak, 50.0f,   sr))        < 0.5f);
    CHECK (std::abs (biquadMagnitudeDb (peak, 20000.0f, sr))       < 0.5f);

    // ...and a cut is its mirror image.
    const auto dip = OutputEQBiquadFilter::calculateCoefficients (3, 1000.0f, -6.0f, 1.0f, 0.7f, sr);
    CHECK (std::abs (biquadMagnitudeDb (dip, 1000.0f, sr) + 6.0f) < 0.25f);

    // Low cut: deep rejection a decade below the corner (2nd order -> ~-40 dB),
    // transparent a decade above it, -3 dB at the corner for Q = 1/sqrt(2).
    const auto lowCut = OutputEQBiquadFilter::calculateCoefficients (1, 1000.0f, 0.0f, 0.70710678f, 0.7f, sr);
    CHECK (biquadMagnitudeDb (lowCut, 100.0f, sr) < -25.0f);
    CHECK (std::abs (biquadMagnitudeDb (lowCut, 10000.0f, sr))          < 0.5f);
    CHECK (std::abs (biquadMagnitudeDb (lowCut, 1000.0f,  sr) + 3.01f)  < 0.3f);

    // High cut is the mirror (and lives at a different shape ID in each class).
    const auto highCut    = OutputEQBiquadFilter::calculateCoefficients (6, 1000.0f, 0.0f, 0.70710678f, 0.7f, sr);
    const auto revHighCut = ReverbBiquadFilter::calculateCoefficients   (5, 1000.0f, 0.0f, 0.70710678f, 0.7f, sr);
    CHECK (biquadMagnitudeDb (highCut,    10000.0f, sr) < -25.0f);
    CHECK (biquadMagnitudeDb (revHighCut, 10000.0f, sr) < -25.0f);
    CHECK (std::abs (biquadMagnitudeDb (highCut,    100.0f, sr)) < 0.5f);
    CHECK (std::abs (biquadMagnitudeDb (revHighCut, 100.0f, sr)) < 0.5f);

    // Shelves approach their gain in the passband and unity in the stopband.
    const auto lowShelf = OutputEQBiquadFilter::calculateCoefficients (2, 500.0f, 6.0f, 0.7f, 0.7f, sr);
    CHECK (std::abs (biquadMagnitudeDb (lowShelf, 10.0f,   sr) - 6.0f) < 0.5f);
    CHECK (std::abs (biquadMagnitudeDb (lowShelf, 10000.0f, sr))       < 0.5f);

    const auto highShelf = OutputEQBiquadFilter::calculateCoefficients (5, 2000.0f, 6.0f, 0.7f, 0.7f, sr);
    CHECK (std::abs (biquadMagnitudeDb (highShelf, 20000.0f, sr) - 6.0f) < 0.5f);
    CHECK (std::abs (biquadMagnitudeDb (highShelf, 50.0f,    sr))        < 0.5f);

    // Same two shelves through the reverb class, whose HighShelf is shape 4.
    const auto revLowShelf  = ReverbBiquadFilter::calculateCoefficients (2, 500.0f,  -6.0f, 0.7f, 0.7f, sr);
    const auto revHighShelf = ReverbBiquadFilter::calculateCoefficients (4, 2000.0f, -6.0f, 0.7f, 0.7f, sr);
    CHECK (std::abs (biquadMagnitudeDb (revLowShelf,  10.0f,    sr) + 6.0f) < 0.5f);
    CHECK (std::abs (biquadMagnitudeDb (revHighShelf, 20000.0f, sr) + 6.0f) < 0.5f);

    // Band pass: 0 dB at the centre, rolling off both ways.
    const auto bandPass = OutputEQBiquadFilter::calculateCoefficients (4, 1000.0f, 0.0f, 1.0f, 0.7f, sr);
    CHECK (std::abs (biquadMagnitudeDb (bandPass, 1000.0f, sr)) < 0.2f);
    CHECK (biquadMagnitudeDb (bandPass, 100.0f,   sr) < -12.0f);
    CHECK (biquadMagnitudeDb (bandPass, 10000.0f, sr) < -12.0f);

    // All pass: unity magnitude everywhere (phase only).
    const auto allPass = OutputEQBiquadFilter::calculateCoefficients (7, 1000.0f, 0.0f, 0.5f, 0.7f, sr);
    for (float f : { 20.0f, 500.0f, 1000.0f, 5000.0f, 20000.0f })
        CHECK (std::abs (biquadMagnitudeDb (allPass, f, sr)) < 0.05f);

    // Edge guards: a bad sample rate reads flat, and out-of-band frequencies are
    // clamped into (0, Nyquist) rather than returning NaN.
    CHECK (bitEqualFloat (biquadMagnitudeDb (peak, 1000.0f,  0.0), 0.0f));
    CHECK (bitEqualFloat (biquadMagnitudeDb (peak, 1000.0f, -1.0), 0.0f));
    for (float f : { -100.0f, 0.0f, 1.0e9f })
        CHECK (std::isfinite (biquadMagnitudeDb (peak, f, sr)));
}

//==============================================================================
// 6. Golden coefficient table — a regression lock for both filter classes.
// Expected values were derived by evaluating the cookbook formulas in the
// headers in float precision. The tolerance is ~80x the float epsilon at these
// magnitudes: tight enough that any change to a formula, a clamp or the pi
// constant shows up, loose enough to survive a different libm's last bit.
static void testBiquadGoldenCoefficients()
{
    using namespace eqtests;

    static const GoldenCase outputGolden[] =
    {
        { 1,  100.0f,  0.0f, 0.70710678f, 0.7f, 48000.0,
          0.990786731f, -1.98157346f, 0.990786731f, -1.98148847f, 0.98165822f },
        { 2,  250.0f,  6.0f, 0.7f,        0.7f, 48000.0,
          1.00964761f, -1.95301175f, 0.9448421f, -1.95338047f, 0.954121232f },
        { 3, 1000.0f, -6.0f, 2.0f,        0.7f, 48000.0,
          0.978021026f, -1.8955189f, 0.933854163f, -1.8955189f, 0.911875308f },
        { 4, 2000.0f,  0.0f, 1.0f,        0.7f, 96000.0,
          0.0612647682f, 0.0f, -0.0612647682f, -1.86140835f, 0.877470434f },
        { 5, 5000.0f, -3.0f, 0.7f,        0.5f, 48000.0,
          0.772829175f, -0.709621847f, 0.162000552f, -1.04882085f, 0.274028808f },
        { 6, 8000.0f,  0.0f, 0.70710678f, 0.7f, 44100.0,
          0.177245006f, 0.354490012f, 0.177245006f, -0.508717537f, 0.217697605f },
        { 7, 1000.0f,  0.0f, 0.5f,        0.7f, 48000.0,
          0.769087732f, -1.75395298f, 0.99999994f, -1.75395298f, 0.769087732f },
    };

    static const GoldenCase reverbGolden[] =
    {
        { 1,   80.0f,  0.0f, 0.70710678f, 0.7f, 48000.0,
          0.992622495f, -1.98524499f, 0.992622495f, -1.98519063f, 0.985299468f },
        { 2,  200.0f,  4.0f, 0.7f,        0.8f, 48000.0,
          1.0047797f, -1.96299291f, 0.959060431f, -1.96314931f, 0.963683903f },
        { 3,  800.0f,  9.0f, 3.0f,        0.7f, 48000.0,
          1.01867604f, -1.96861494f, 0.960782528f, -1.96861494f, 0.97945857f },
        { 4, 4000.0f, -5.0f, 0.7f,        0.6f, 48000.0,
          0.63234663f, -0.686702669f, 0.199771792f, -1.27773619f, 0.423152119f },
        { 5, 6000.0f,  0.0f, 0.70710678f, 0.7f, 44100.0,
          0.112055212f, 0.224110425f, 0.112055212f, -0.855989456f, 0.304210305f },
        { 6, 1500.0f,  0.0f, 1.5f,        0.7f, 96000.0,
          0.031638667f, 0.0f, -0.031638667f, -1.92739677f, 0.936722636f },
    };

    for (const auto& g : outputGolden)
        checkGolden ("OutputEQ",
                     g,
                     spatcore::dsp::OutputEQBiquadFilter::calculateCoefficients (
                         g.shape, g.freq, g.gainDb, g.q, g.slope, g.sampleRate));

    for (const auto& g : reverbGolden)
        checkGolden ("ReverbEQ",
                     g,
                     spatcore::dsp::ReverbBiquadFilter::calculateCoefficients (
                         g.shape, g.freq, g.gainDb, g.q, g.slope, g.sampleRate));
}

//==============================================================================
// 7. OutputEQProcessor neutrality at defaults. This is the class sitting in the
// WFS-DIY output path, and the golden-render gate compares renders byte-for-byte
// — so with every channel disabled (the default, and the state prepare() leaves
// behind) processBlock() must leave the buffer BIT-identical, on a partial slice
// as well as a whole block.
static void testOutputEQProcessorNeutrality()
{
    using namespace eqtests;

    constexpr int numChannels = 4;
    constexpr int blockSize   = 256;

    const auto reference = makeAwkwardSignal (blockSize);
    CHECK (static_cast<int> (reference.size()) == blockSize);

    auto fillBuffer = [&] (juce::AudioBuffer<float>& buf)
    {
        for (int c = 0; c < buf.getNumChannels(); ++c)
        {
            float* d = buf.getWritePointer (c);
            for (int i = 0; i < blockSize; ++i)
            {
                // A per-channel rotation of the same awkward signal, so every
                // channel carries the negative zero / denormals somewhere.
                const size_t src = static_cast<size_t> ((i + 37 * c) % blockSize);
                d[i] = reference[src];
            }
        }
    };

    auto buffersBitEqual = [] (const juce::AudioBuffer<float>& a,
                               const juce::AudioBuffer<float>& b)
    {
        if (a.getNumChannels() != b.getNumChannels() || a.getNumSamples() != b.getNumSamples())
            return false;

        for (int c = 0; c < a.getNumChannels(); ++c)
            if (std::memcmp (a.getReadPointer (c), b.getReadPointer (c),
                             static_cast<size_t> (a.getNumSamples()) * sizeof (float)) != 0)
                return false;

        return true;
    };

    juce::AudioBuffer<float> buffer (numChannels, blockSize), expected (numChannels, blockSize);
    fillBuffer (buffer);
    fillBuffer (expected);
    CHECK (buffersBitEqual (buffer, expected));

    spatcore::dsp::OutputEQProcessor eq;
    eq.prepare (kSampleRate, blockSize, numChannels);

    // (a) Straight after prepare(), before any setParameters() call.
    eq.processBlock (buffer, 0, blockSize);
    CHECK (buffersBitEqual (buffer, expected));

    // (b) After pushing default Params: every channel disabled, every band at
    //     the default shape 0.
    spatcore::dsp::OutputEQProcessor::Params params;
    params.channels.resize (static_cast<size_t> (numChannels));
    eq.setParameters (params);

    eq.processBlock (buffer, 0, blockSize);
    CHECK (buffersBitEqual (buffer, expected));

    // (c) Bands loaded with real, active shapes while the channels stay
    //     disabled — the enabled-flag-first ordering must keep them OFF.
    for (auto& cp : params.channels)
    {
        cp.enabled = false;
        for (int b = 0; b < spatcore::dsp::OutputEQProcessor::NUM_EQ_BANDS; ++b)
        {
            auto& bp = cp.bands[static_cast<size_t> (b)];
            bp.shape = kTickA[b].shape;
            bp.freq  = kTickA[b].freq;
            bp.gain  = kTickA[b].gainDb;
            bp.q     = kTickA[b].q;
            bp.slope = kTickA[b].slope;
        }
    }

    eq.setParameters (params);
    eq.processBlock (buffer, 0, blockSize);
    CHECK (buffersBitEqual (buffer, expected));

    // (d) A partial slice (the WFS output path processes sub-block ranges).
    eq.processBlock (buffer, 64, 128);
    CHECK (buffersBitEqual (buffer, expected));

    // (e) reset() on a neutral processor changes nothing.
    eq.reset();
    eq.processBlock (buffer, 0, blockSize);
    CHECK (buffersBitEqual (buffer, expected));

    // Sanity that the harness would have noticed a change: enabling the
    // channels and re-pushing the SAME bands must make the EQ audible.
    for (auto& cp : params.channels)
        cp.enabled = true;

    eq.setParameters (params);
    eq.processBlock (buffer, 0, blockSize);
    CHECK (! buffersBitEqual (buffer, expected));

    for (int c = 0; c < numChannels; ++c)
    {
        const float* d = buffer.getReadPointer (c);
        for (int i = 0; i < blockSize; ++i)
            CHECK (std::isfinite (d[i]));
    }
}

//==============================================================================
// io/ - device layer.
//
// DeviceIoCallback itself is not driven here: exercising it needs a live
// juce::AudioIODevice, which cannot be faked headlessly without reimplementing
// a backend. Everything about it that can be wrong independently of a driver
// lives in HardwareIndexMap (the hardware -> compact index translation) and is
// covered below; the buffer assembly is verified against real hardware by the
// app's patch-window acceptance checks.
//==============================================================================
static juce::BigInteger maskFromBits (std::initializer_list<int> bits)
{
    juce::BigInteger mask;

    for (int b : bits)
        mask.setBit (b);

    return mask;
}

static void testHardwareIndexMapContiguous()
{
    juce::BigInteger ins, outs;
    ins.setRange (0, 8, true);
    outs.setRange (0, 64, true);

    const auto map = spatcore::io::HardwareIndexMap::fromMasks (ins, outs, 512);

    // One past the highest active bit of EITHER mask.
    CHECK (map.numChannels == 64);
    CHECK ((int) map.inputIndexForHw.size() == 64);
    CHECK ((int) map.outputIndexForHw.size() == 64);

    // Contiguous from 0: hardware index == compact index, which is exactly the
    // assumption the old AudioSourcePlayer path silently relied on.
    CHECK (map.isIdentityMapping());

    for (int hw = 0; hw < 8; ++hw)
        CHECK (map.inputIndexForHw[(size_t) hw] == hw);

    for (int hw = 8; hw < 64; ++hw)
        CHECK (map.inputIndexForHw[(size_t) hw] == -1);

    for (int hw = 0; hw < 64; ++hw)
        CHECK (map.outputIndexForHw[(size_t) hw] == hw);
}

static void testHardwareIndexMapSparse()
{
    // Holes: hardware index and compact index diverge. Reading the callback
    // arrays by hardware number here would silently address the wrong channel.
    const auto ins  = maskFromBits ({ 1, 3 });
    const auto outs = maskFromBits ({ 0, 2, 5 });

    const auto map = spatcore::io::HardwareIndexMap::fromMasks (ins, outs, 512);

    CHECK (map.numChannels == 6);
    CHECK (! map.isIdentityMapping());

    const int expectedIn[6]  = { -1, 0, -1, 1, -1, -1 };
    const int expectedOut[6] = {  0, -1, 1, -1, -1,  2 };

    for (int hw = 0; hw < 6; ++hw)
    {
        CHECK (map.inputIndexForHw[(size_t) hw]  == expectedIn[hw]);
        CHECK (map.outputIndexForHw[(size_t) hw] == expectedOut[hw]);
    }
}

static void testHardwareIndexMapEmptyAndClamp()
{
    const juce::BigInteger empty;

    const auto none = spatcore::io::HardwareIndexMap::fromMasks (empty, empty, 512);
    CHECK (none.numChannels == 0);
    CHECK (none.inputIndexForHw.empty());
    CHECK (none.outputIndexForHw.empty());
    CHECK (none.isIdentityMapping());        // vacuously

    // A device wider than the policy limit is addressed up to the limit only;
    // nothing is a fixed-size array, so this is the ONLY cap in the path.
    juce::BigInteger wide;
    wide.setRange (0, 600, true);

    const auto clamped = spatcore::io::HardwareIndexMap::fromMasks (wide, wide, 512);
    CHECK (clamped.numChannels == 512);
    CHECK (clamped.outputIndexForHw[511] == 511);

    // 512 is a policy number, not a structural one: ask for more, get more.
    const auto unclamped = spatcore::io::HardwareIndexMap::fromMasks (wide, wide, 1024);
    CHECK (unclamped.numChannels == 600);
}

static void testDeviceHostEnableAllPolicy()
{
    juce::AudioDeviceManager::AudioDeviceSetup setup;

    // The default state is the bug: while useDefault*Channels is true,
    // AudioDeviceManager::setAudioDeviceSetup discards the caller's mask and
    // substitutes range(0, numChansNeeded).
    CHECK (setup.useDefaultInputChannels);
    CHECK (setup.useDefaultOutputChannels);

    spatcore::io::DeviceHost::applyEnableAllPolicy (setup, 32, 128, 512);

    CHECK (! setup.useDefaultInputChannels);
    CHECK (! setup.useDefaultOutputChannels);
    CHECK (setup.inputChannels.countNumberOfSetBits() == 32);
    CHECK (setup.outputChannels.countNumberOfSetBits() == 128);
    CHECK (setup.inputChannels.getHighestBit() == 31);
    CHECK (setup.outputChannels.getHighestBit() == 127);

    // Re-applying with smaller counts must CLEAR the old bits, not union them
    // (switching to a smaller interface).
    spatcore::io::DeviceHost::applyEnableAllPolicy (setup, 2, 4, 512);
    CHECK (setup.inputChannels.countNumberOfSetBits() == 2);
    CHECK (setup.outputChannels.countNumberOfSetBits() == 4);

    // Clamped to the policy limit.
    spatcore::io::DeviceHost::applyEnableAllPolicy (setup, 600, 600, 512);
    CHECK (setup.inputChannels.countNumberOfSetBits() == 512);
    CHECK (setup.outputChannels.countNumberOfSetBits() == 512);

    // An output-only device: no inputs is a valid mask, not a disabled flag.
    spatcore::io::DeviceHost::applyEnableAllPolicy (setup, 0, 8, 512);
    CHECK (setup.inputChannels.isZero());
    CHECK (setup.outputChannels.countNumberOfSetBits() == 8);
    CHECK (! setup.useDefaultInputChannels);
}

//==============================================================================
// Counts sign changes, which measures the tone's pitch independently of its
// amplitude (the protective ramp means early samples are tiny).
static int countZeroCrossings (const float* data, int numSamples)
{
    int crossings = 0;

    for (int i = 1; i < numSamples; ++i)
        if ((data[i - 1] < 0.0f && data[i] >= 0.0f) || (data[i - 1] >= 0.0f && data[i] < 0.0f))
            ++crossings;

    return crossings;
}

static void renderTone (spatcore::io::TestSignalGenerator& gen,
                        juce::AudioBuffer<float>& buffer,
                        int blockSize)
{
    for (int start = 0; start + blockSize <= buffer.getNumSamples(); start += blockSize)
        gen.renderNextBlock (buffer, start, blockSize);
}

static void testTestSignalGeneratorToneFollowsSampleRate()
{
    using Gen = spatcore::io::TestSignalGenerator;

    // No setFrequency() call anywhere: prepare() alone must leave the tone
    // playable. This is the regression the app shipped with - the phase
    // increment defaulted to zero and was only ever written by setFrequency(),
    // so a Tone selected on any path that skipped it was pure silence.
    Gen gen;
    gen.prepare (48000.0, 480);
    gen.setSignalType (Gen::SignalType::Tone);
    gen.setOutputChannel (0);

    // 480 divides 48000, so every sample of the buffer is rendered - a partial
    // trailing block would read as a pitch error.
    const int blockSize = 480;
    juce::AudioBuffer<float> buffer (2, 48000);
    buffer.clear();
    renderTone (gen, buffer, blockSize);

    CHECK (buffer.getMagnitude (0, 0, 48000) > 0.0f);
    CHECK (buffer.getMagnitude (1, 0, 48000) == 0.0f);   // other channels untouched

    // 1000 Hz default over the second half second (past the ramp): 2 crossings
    // per cycle.
    const int crossings48k = countZeroCrossings (buffer.getReadPointer (0) + 24000, 24000);
    CHECK (crossings48k >= 995 && crossings48k <= 1005);

    // Same generator re-prepared at double the rate: the pitch must hold, so
    // the crossings over the same SAMPLE count must halve. Without the
    // recompute in prepare() the tone would come back an octave up.
    gen.prepare (96000.0, blockSize);
    gen.setSignalType (Gen::SignalType::Off);
    gen.setSignalType (Gen::SignalType::Tone);
    gen.setOutputChannel (-1);
    gen.setOutputChannel (0);

    buffer.clear();
    renderTone (gen, buffer, blockSize);

    const int crossings96k = countZeroCrossings (buffer.getReadPointer (0) + 24000, 24000);
    CHECK (crossings96k >= 495 && crossings96k <= 505);
}

static void testTestSignalGeneratorProtectiveRamp()
{
    using Gen = spatcore::io::TestSignalGenerator;

    // HEARING PROTECTION: the continuous signals must ramp up over 500 ms from
    // silence, every time they start. A rig under test can be pointed at
    // someone's head at full system gain.
    Gen gen;
    gen.prepare (48000.0, 512);
    gen.setLevel (0.0f);                       // unity, so the envelope is readable
    gen.setSignalType (Gen::SignalType::Tone);
    gen.setOutputChannel (0);

    const int blockSize = 480;                 // 10 ms
    const int numBlocks = 100;                 // 1 s: 500 ms of ramp + 500 ms at level
    juce::AudioBuffer<float> buffer (1, blockSize * numBlocks);
    buffer.clear();
    renderTone (gen, buffer, blockSize);

    CHECK (buffer.getReadPointer (0)[0] == 0.0f);   // starts from silence

    float previousPeak = -1.0f;
    for (int b = 0; b < 50; ++b)                    // across the ramp
    {
        const float peak = buffer.getMagnitude (0, b * blockSize, blockSize);
        CHECK (peak >= previousPeak);               // never steps back up
        previousPeak = peak;
    }

    // First 10 ms is a small fraction of level; the ramp is complete by 500 ms.
    CHECK (buffer.getMagnitude (0, 0, blockSize) < 0.05f);
    CHECK (buffer.getMagnitude (0, 50 * blockSize, blockSize) > 0.98f);

    // Restarting on another channel restarts the ramp - not a step to full.
    gen.setOutputChannel (-1);
    gen.setOutputChannel (0);
    buffer.clear();
    gen.renderNextBlock (buffer, 0, blockSize);
    CHECK (buffer.getMagnitude (0, 0, blockSize) < 0.05f);

    // EVERY type ramps — the transient envelopes and SpeakerId's declick ride
    // on top of the 500 ms protective ramp rather than replacing it. A Dirac
    // at unity into a rig at full gain is exactly the step this guards
    // against: the operator must get time to stop it before it hurts.
    gen.setSignalType (Gen::SignalType::DiracPulse);
    juce::AudioBuffer<float> longBuf (1, 96000);   // 2 s: pulses at ~0 s and ~1 s
    longBuf.clear();
    renderTone (gen, longBuf, blockSize);
    CHECK (longBuf.getMagnitude (0, 0, blockSize) < 0.05f);      // first pulse ramped down
    CHECK (longBuf.getMagnitude (0, 48000, 48000) > 1.9f);       // post-ramp pulse at full 2.0

    gen.setSignalType (Gen::SignalType::Sweep);
    buffer.clear();
    gen.renderNextBlock (buffer, 0, blockSize);
    CHECK (buffer.getMagnitude (0, 0, blockSize) < 0.05f);       // sweep start ramped

    gen.setSignalType (Gen::SignalType::SpeakerId);
    juce::AudioBuffer<float> idBuf (2, blockSize);
    idBuf.clear();
    gen.renderNextBlock (idBuf, 0, blockSize);
    CHECK (idBuf.getMagnitude (0, 0, blockSize) < 0.05f);        // first burst ramped
}

static void testTestSignalGeneratorDeterministicSeed()
{
    using Gen = spatcore::io::TestSignalGenerator;

    const int blockSize = 256;
    const int numBlocks = 8;

    juce::AudioBuffer<float> a (1, blockSize * numBlocks), b (1, blockSize * numBlocks);
    a.clear();
    b.clear();

    for (auto* pair : { &a, &b })
    {
        Gen gen;
        gen.setDeterministicSeed (42);
        gen.prepare (48000.0, blockSize);
        gen.setSignalType (Gen::SignalType::PinkNoise);
        gen.setOutputChannel (0);
        renderTone (gen, *pair, blockSize);
    }

    CHECK (a.getMagnitude (0, 0, a.getNumSamples()) > 0.0f);

    for (int i = 0; i < a.getNumSamples(); ++i)
        if (! bitEqualFloat (a.getReadPointer (0)[i], b.getReadPointer (0)[i]))
        {
            CHECK (false);
            break;
        }
}

static void testTestSignalGeneratorSpeakerIdSequencing()
{
    using Gen = spatcore::io::TestSignalGenerator;

    // SpeakerId steps a declicked pink burst across each buffer channel in
    // turn (0.75 s on / 0.25 s gap): at any instant exactly one channel
    // carries energy, matching getCurrentSpeakerIndex(); the gap is silent on
    // every channel. Ported from the XOA fork this mode originated in.
    constexpr double sr = 48000.0;
    constexpr int numOut = 4;

    Gen gen;
    gen.setDeterministicSeed (42);
    gen.prepare (sr, 8192);
    gen.setLevel (0.0f);
    gen.setSignalType (Gen::SignalType::SpeakerId);

    CHECK (gen.isActive());                     // active with no target channel

    auto advance = [&] (double seconds)
    {
        juce::AudioBuffer<float> scratch (numOut, static_cast<int> (seconds * sr));
        scratch.clear();
        gen.renderNextBlock (scratch, 0, scratch.getNumSamples());
    };

    // Probe a short window and assert exactly `expected` carries energy.
    auto probeExclusive = [&] (int expected)
    {
        juce::AudioBuffer<float> probe (numOut, 2400);   // 50 ms, inside one burst
        probe.clear();
        gen.renderNextBlock (probe, 0, 2400);

        CHECK (gen.getCurrentSpeakerIndex() == expected);
        for (int c = 0; c < numOut; ++c)
        {
            const bool hot = probe.getMagnitude (c, 0, 2400) > 0.0f;
            CHECK (hot == (c == expected));
        }
    };

    advance (0.30);  probeExclusive (0);   // ~0.30 s: channel 0 burst
    advance (0.95);  probeExclusive (1);   // ~1.30 s: channel 1 burst
    advance (0.95);  probeExclusive (2);   // ~2.30 s: channel 2 burst

    // Into channel 2's gap (withinSlot [0.75, 1.0)): silent everywhere, index
    // cleared.
    advance (0.45);                        // ~2.80 s
    juce::AudioBuffer<float> gap (numOut, 1200);
    gap.clear();
    gen.renderNextBlock (gap, 0, 1200);
    CHECK (gen.getCurrentSpeakerIndex() == -1);
    for (int c = 0; c < numOut; ++c)
        CHECK (gap.getMagnitude (c, 0, 1200) == 0.0f);
}

//==============================================================================
//==============================================================================
// binaural/HeadFrame — rotation conventions are load-bearing for every HRTF
// mode: pin them against hand-computed cases.
static void testBinauralHeadFrame()
{
    using namespace spatcore::binaural;
    namespace hf = spatcore::binaural::headframe;
    constexpr float pi = 3.14159265358979f;
    const float tol = 1e-4f;

    // Baseline α = 0 → identity: head faces +y (the legacy listener at (0,−d)
    // facing the origin).
    ListenerPose pose;   // origin, identity R
    {
        auto front = hf::directionInHeadFrame (pose, 0.0f, 5.0f, 0.0f);
        CHECK (std::fabs (front.azRad) < tol);
        CHECK (std::fabs (front.elRad) < tol);
        CHECK (std::fabs (front.distance - 5.0f) < tol);

        auto right = hf::directionInHeadFrame (pose, 3.0f, 0.0f, 0.0f);
        CHECK (std::fabs (right.azRad - pi / 2.0f) < tol);   // +az = listener's right

        auto above = hf::directionInHeadFrame (pose, 0.0f, 0.0f, 2.0f);
        CHECK (std::fabs (above.elRad - pi / 2.0f) < tol);
    }

    // Positive yaw turns the head right: a source on +x lands dead ahead.
    {
        float offset[9];
        hf::yawPitchRollToMatrix (pi / 2.0f, 0.0f, 0.0f, offset);
        hf::composeWithBaseline (0.0f, offset, pose.R);
        auto d = hf::directionInHeadFrame (pose, 4.0f, 0.0f, 0.0f);
        CHECK (std::fabs (d.azRad) < tol);
        CHECK (std::fabs (d.elRad) < tol);
    }

    // Positive pitch looks up: a source straight ahead drops below the gaze.
    {
        float offset[9];
        hf::yawPitchRollToMatrix (0.0f, pi / 6.0f, 0.0f, offset);   // +30°
        hf::composeWithBaseline (0.0f, offset, pose.R);
        auto d = hf::directionInHeadFrame (pose, 0.0f, 5.0f, 0.0f);
        CHECK (std::fabs (d.elRad + pi / 6.0f) < tol);              // el = −30°
        CHECK (std::fabs (d.azRad) < tol);
    }

    // Baseline α = 90°: listener placed at (d, 0), facing the origin (−x).
    {
        float offset[9];
        hf::yawPitchRollToMatrix (0.0f, 0.0f, 0.0f, offset);
        pose.x = 5.0f; pose.y = 0.0f; pose.z = 0.0f;
        hf::composeWithBaseline (pi / 2.0f, offset, pose.R);
        auto d = hf::directionInHeadFrame (pose, 0.0f, 0.0f, 0.0f);
        CHECK (std::fabs (d.azRad) < tol);                          // origin dead ahead
        CHECK (std::fabs (d.distance - 5.0f) < tol);
    }

    // Composed rotation stays orthonormal (RᵀR = I).
    {
        float offset[9], R[9];
        hf::yawPitchRollToMatrix (0.7f, -0.4f, 0.3f, offset);
        hf::composeWithBaseline (1.1f, offset, R);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                float dot = 0.0f;
                for (int k = 0; k < 3; ++k)
                    dot += R[k * 3 + i] * R[k * 3 + j];
                CHECK (std::fabs (dot - (i == j ? 1.0f : 0.0f)) < 1e-4f);
            }
    }
}

//==============================================================================
// Non-finite head attitude must never reach the audio. Regression for the
// class of bug XOA hit first (its 1560330): the tracker ABI hands over RAW
// angles, a degenerate face box makes the estimator emit NaN, and NaN in a
// rotation matrix becomes NaN in the delay lines / filter state / convolution
// history — where it stays long after the frame that caused it.
static void testNonFiniteAttitudeIsRefused()
{
    using namespace spatcore::binaural;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();

    // isFiniteAttitude is the shared predicate both guards are built on.
    {
        HeadOrientation o;
        o.yawRad = 0.1f; o.pitchRad = -0.2f; o.rollRad = 0.05f; o.valid = true;
        CHECK (isFiniteAttitude (o));
        o.pitchRad = nan;  CHECK (! isFiniteAttitude (o));
        o.pitchRad = -0.2f; o.rollRad = inf;
        CHECK (! isFiniteAttitude (o));
    }

    // The publish choke point every source goes through refuses to put a
    // non-finite attitude on the render thread, reporting "nothing
    // trustworthy" so the consumer's existing fallback takes over.
    {
        struct TestSource : SnapshotHeadOrientationSource
        {
            juce::String getSourceId() const override    { return "test"; }
            juce::String getDisplayName() const override { return "Test"; }
            bool isConnected() const override            { return true; }
            using SnapshotHeadOrientationSource::publishOrientation;
        };

        TestSource src;
        HeadOrientation good;
        good.yawRad = 0.3f; good.valid = true;
        src.publishOrientation (good);
        CHECK (src.getOrientation().valid);
        CHECK (std::fabs (src.getOrientation().yawRad - 0.3f) < 1e-6f);

        HeadOrientation bad;
        bad.yawRad = nan; bad.pitchRad = 0.0f; bad.rollRad = 0.0f; bad.valid = true;
        src.publishOrientation (bad);
        CHECK (! src.getOrientation().valid);
        CHECK (isFiniteAttitude (src.getOrientation()));   // and not merely flagged
    }

    // isFinitePose catches the same poison one hop later, as a matrix.
    {
        ListenerPose p;                       // default = identity, origin
        CHECK (isFinitePose (p));
        p.R[4] = nan;  CHECK (! isFinitePose (p));
        p.R[4] = 1.0f; p.z = inf;
        CHECK (! isFinitePose (p));
    }
}

// ...and if one slips through anyway, the render core must not emit NaN.
static void testEngineSurvivesNonFinitePose()
{
    using namespace spatcore::binaural;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const double fs = 48000.0;
    const int block = 256;
    const int numBlocks = 6;

    auto allFinite = [] (const std::vector<float>& v)
    {
        for (float s : v)
            if (! std::isfinite (s))
                return false;
        return true;
    };

    BinauralEngine engine;
    engine.prepare (fs, block, 2);
    engine.setMode (RenderMode::Structural);

    float positions[2][3] = { { 1.0f, 3.0f, 1.6f }, { -2.0f, 4.0f, 1.6f } };
    std::vector<float> tone ((size_t) block, 0.0f);
    for (int i = 0; i < block; ++i)
        tone[(size_t) i] = 0.25f * std::sin (2.0f * 3.14159265f * 440.0f * (float) i / (float) fs);
    const float* inputs[2] = { tone.data(), tone.data() };

    std::vector<float> L ((size_t) block), R ((size_t) block);

    // A pose whose rotation is entirely NaN — what a NaN attitude produces.
    ListenerPose poisoned;
    for (int i = 0; i < 9; ++i)
        poisoned.R[i] = nan;
    poisoned.y = -5.0f;

    for (int b = 0; b < numBlocks; ++b)
    {
        std::fill (L.begin(), L.end(), 0.0f);
        std::fill (R.begin(), R.end(), 0.0f);
        engine.processBlock (poisoned, positions, inputs, nullptr, 0.0f,
                             L.data(), R.data(), 2, block);
        CHECK (allFinite (L));
        CHECK (allFinite (R));
    }

    // A single bad source position must not take the other source with it.
    {
        ListenerPose pose;
        pose.y = -5.0f;
        float mixed[2][3] = { { nan, nan, nan }, { -2.0f, 4.0f, 1.6f } };
        for (int b = 0; b < numBlocks; ++b)
        {
            std::fill (L.begin(), L.end(), 0.0f);
            std::fill (R.begin(), R.end(), 0.0f);
            engine.processBlock (pose, mixed, inputs, nullptr, 0.0f,
                                 L.data(), R.data(), 2, block);
            CHECK (allFinite (L));
            CHECK (allFinite (R));
        }
        float peak = 0.0f;
        for (float s : L) peak = std::max (peak, std::fabs (s));
        CHECK (peak > 0.0f);        // the healthy source still renders
    }

    // Recovery: once the pose is good again the output must be audible, i.e.
    // no filter/delay state was permanently poisoned by the bad blocks.
    {
        ListenerPose pose;
        pose.y = -5.0f;
        float peak = 0.0f;
        for (int b = 0; b < numBlocks; ++b)
        {
            std::fill (L.begin(), L.end(), 0.0f);
            std::fill (R.begin(), R.end(), 0.0f);
            engine.processBlock (pose, positions, inputs, nullptr, 0.0f,
                                 L.data(), R.data(), 2, block);
            CHECK (allFinite (L));
            CHECK (allFinite (R));
            for (float s : L) peak = std::max (peak, std::fabs (s));
        }
        CHECK (peak > 0.0f);
    }
}

//==============================================================================
// binaural/StructuralHrtfRenderer — ITD sign/magnitude per direction, and
// DC transparency of the filter chain (shadow is unity at DC by construction).
static void testStructuralHrtfItdAndDc()
{
    using namespace spatcore::binaural;
    constexpr float pi = 3.14159265358979f;
    const double fs = 48000.0;
    const int block = 512;
    const float headRadius = 0.0875f;

    auto renderImpulse = [&] (float azRad, std::vector<float>& L, std::vector<float>& R)
    {
        StructuralHrtfRenderer r;
        r.prepare (fs, block, 1);

        SourceDirection dir;
        dir.azRad = azRad;
        dir.elRad = 0.0f;
        dir.distance = 2.0f;

        const int numBlocks = 2;
        L.assign ((size_t) (numBlocks * block), 0.0f);
        R.assign ((size_t) (numBlocks * block), 0.0f);
        std::vector<float> input ((size_t) block, 0.0f);

        std::int64_t pos = 0;
        for (int b = 0; b < numBlocks; ++b)
        {
            std::fill (input.begin(), input.end(), 0.0f);
            if (b == 0)
                input[0] = 1.0f;
            r.processSource (0, dir, headRadius, 0.0f, 1.0f, input.data(),
                             L.data() + pos, R.data() + pos, block, pos);
            pos += block;
        }
    };

    auto peakIndex = [] (const std::vector<float>& v)
    {
        size_t best = 0;
        for (size_t i = 1; i < v.size(); ++i)
            if (std::fabs (v[i]) > std::fabs (v[best]))
                best = i;
        return (int) best;
    };

    // Source hard right: right ear leads by (a/c)·(1 + π/2) seconds.
    {
        std::vector<float> L, R;
        renderImpulse (pi / 2.0f, L, R);
        const int itdSamples = peakIndex (L) - peakIndex (R);
        const int expected = (int) std::lround ((headRadius / 343.0f) * (1.0f + pi / 2.0f) * fs);
        CHECK (itdSamples > 0);                            // right leads
        CHECK (std::abs (itdSamples - expected) <= 3);
        // Shadowed (left) ear is noticeably quieter at the peak.
        CHECK (std::fabs (R[(size_t) peakIndex (R)]) > std::fabs (L[(size_t) peakIndex (L)]));
    }

    // Source dead ahead: no ITD, symmetric level.
    {
        std::vector<float> L, R;
        renderImpulse (0.0f, L, R);
        CHECK (std::abs (peakIndex (L) - peakIndex (R)) <= 1);
        const float pl = std::fabs (L[(size_t) peakIndex (L)]);
        const float pr = std::fabs (R[(size_t) peakIndex (R)]);
        CHECK (std::fabs (pl - pr) < 0.05f * (pl + pr));
    }

    // DC transparency: constant input, front source at 1 m (unity distance
    // gain) settles to ~1 on both ears — the whole chain is unity at DC.
    {
        StructuralHrtfRenderer r;
        r.prepare (fs, block, 1);
        SourceDirection dir;                // front, 1 m
        dir.distance = 1.0f;

        std::vector<float> L ((size_t) block), R ((size_t) block), input ((size_t) block, 1.0f);
        std::int64_t pos = 0;
        float lastL = 0.0f, lastR = 0.0f;
        for (int b = 0; b < 6; ++b)         // 64 ms — past the 1 m delay + filters
        {
            std::fill (L.begin(), L.end(), 0.0f);
            std::fill (R.begin(), R.end(), 0.0f);
            r.processSource (0, dir, headRadius, 0.0f, 1.0f, input.data(),
                             L.data(), R.data(), block, pos);
            pos += block;
            lastL = L.back();
            lastR = R.back();
        }
        CHECK (std::fabs (lastL - 1.0f) < 0.05f);
        CHECK (std::fabs (lastR - 1.0f) < 0.05f);
    }
}

//==============================================================================
// binaural/StructuralHrtfRenderer — continuity under head rotation: sweeping
// the azimuth across blocks must never produce sample-to-sample jumps beyond
// what a slow Doppler on a low-frequency sine can explain (no zipper).
static void testStructuralHrtfRotationContinuity()
{
    using namespace spatcore::binaural;
    constexpr float pi = 3.14159265358979f;
    const double fs = 48000.0;
    const int block = 512;
    const int numBlocks = 40;   // ~0.43 s

    StructuralHrtfRenderer r;
    r.prepare (fs, block, 1);

    SourceDirection dir;
    dir.distance = 3.0f;

    std::vector<float> L ((size_t) block), R ((size_t) block), input ((size_t) block);
    std::int64_t pos = 0;
    double phase = 0.0;
    const double phaseInc = 2.0 * pi * 200.0 / fs;

    float prevL = 0.0f, prevR = 0.0f, maxDelta = 0.0f;
    for (int b = 0; b < numBlocks; ++b)
    {
        // 0 → 90° sweep across the run — a fast but plausible head turn.
        dir.azRad = (pi / 2.0f) * (float) b / (float) numBlocks;

        for (int i = 0; i < block; ++i)
        {
            input[(size_t) i] = (float) std::sin (phase);
            phase += phaseInc;
        }
        std::fill (L.begin(), L.end(), 0.0f);
        std::fill (R.begin(), R.end(), 0.0f);
        r.processSource (0, dir, 0.0875f, 0.0f, 1.0f, input.data(),
                         L.data(), R.data(), block, pos);
        pos += block;

        for (int i = 0; i < block; ++i)
        {
            if (b > 1 || i > 0)   // skip the initial fill-in transient
            {
                maxDelta = std::max (maxDelta, std::fabs (L[(size_t) i] - prevL));
                maxDelta = std::max (maxDelta, std::fabs (R[(size_t) i] - prevR));
            }
            prevL = L[(size_t) i];
            prevR = R[(size_t) i];
        }
    }

    // A clean 200 Hz sine at this level moves ≲0.01/sample; rotation-induced
    // modulation stays the same order. A zipper/click would blow well past 0.1.
    CHECK (maxDelta < 0.1f);
}

//==============================================================================
// binaural/HeadFrame — matrixToYawPitchRoll is the inverse used by head
// trackers to recover angles after zero-calibration composition.
static void testHeadFrameMatrixToYawPitchRoll()
{
    namespace hf = spatcore::binaural::headframe;
    constexpr float pi = 3.14159265358979f;
    const float tol = 1e-4f;

    // Roundtrip over a grid, away from gimbal lock.
    for (float yaw = -3.0f; yaw <= 3.0f; yaw += 0.75f)
        for (float pitch = -1.4f; pitch <= 1.4f; pitch += 0.35f)
            for (float roll = -1.4f; roll <= 1.4f; roll += 0.35f)
            {
                float R[9];
                hf::yawPitchRollToMatrix (yaw, pitch, roll, R);

                float y2, p2, r2;
                hf::matrixToYawPitchRoll (R, y2, p2, r2);

                // Compare through the matrix: distinct angle triples can name
                // the same rotation, the rotation itself must match.
                float R2[9];
                hf::yawPitchRollToMatrix (y2, p2, r2, R2);
                for (int i = 0; i < 9; ++i)
                    CHECK (std::fabs (R[i] - R2[i]) < 1e-3f);
            }

    // Gimbal lock: looking straight up stays finite and reproduces the rotation.
    {
        float R[9];
        hf::yawPitchRollToMatrix (0.6f, pi / 2.0f, 0.0f, R);
        float y, p, r;
        hf::matrixToYawPitchRoll (R, y, p, r);
        CHECK (std::isfinite (y) && std::isfinite (p) && std::isfinite (r));
        CHECK (r == 0.0f);                                  // roll folded into yaw
        CHECK (std::fabs (p - pi / 2.0f) < 1e-3f);
    }

    // transpose() inverts an orthonormal rotation: Rᵀ·R = I.
    {
        float R[9], Rt[9], I[9];
        hf::yawPitchRollToMatrix (0.7f, -0.4f, 0.3f, R);
        hf::transpose (R, Rt);
        hf::multiply (Rt, R, I);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                CHECK (std::fabs (I[i * 3 + j] - (i == j ? 1.0f : 0.0f)) < tol);
    }
}

//==============================================================================
// The zero-calibration property head trackers rely on: capturing R_zero and
// pre-multiplying by its inverse maps that attitude to identity, and any
// later attitude to its offset FROM the calibration pose.
static void testHeadTrackerZeroComposition()
{
    namespace hf = spatcore::binaural::headframe;
    const float tol = 1e-3f;

    float rZero[9], rZeroInv[9];
    hf::yawPitchRollToMatrix (0.9f, -0.3f, 0.15f, rZero);   // user looking off-axis
    hf::transpose (rZero, rZeroInv);

    // Calibration pose itself → zero angles.
    {
        float corrected[9];
        hf::multiply (rZeroInv, rZero, corrected);
        float y, p, r;
        hf::matrixToYawPitchRoll (corrected, y, p, r);
        CHECK (std::fabs (y) < tol);
        CHECK (std::fabs (p) < tol);
        CHECK (std::fabs (r) < tol);
    }

    // A later attitude → the rotation from calibration to now, and NOT the
    // per-angle difference (the trap this composition exists to avoid).
    {
        float rNow[9], corrected[9];
        hf::yawPitchRollToMatrix (1.3f, 0.2f, -0.1f, rNow);
        hf::multiply (rZeroInv, rNow, corrected);

        float y, p, r;
        hf::matrixToYawPitchRoll (corrected, y, p, r);

        // Recomposing must reproduce the corrected rotation exactly.
        float check[9];
        hf::yawPitchRollToMatrix (y, p, r, check);
        for (int i = 0; i < 9; ++i)
            CHECK (std::fabs (corrected[i] - check[i]) < tol);

        // Naive subtraction would give yaw 0.4/pitch 0.5/roll −0.25; with a
        // non-zero calibration pitch/roll the true composition differs.
        const bool differsFromNaive = std::fabs (y - 0.4f) > 1e-2f
                                   || std::fabs (p - 0.5f) > 1e-2f
                                   || std::fabs (r + 0.25f) > 1e-2f;
        CHECK (differsFromNaive);
    }
}

//==============================================================================
// dsp/OneEuroFilter — promoted out of TrackingPositionFilter for head tracking.
static void testOneEuroFilter()
{
    spatcore::dsp::OneEuroFilter f;

    // First sample passes through untouched, and seeds the state.
    CHECK (f.filter (0.5f, 0.0, 1.5f, 3.0f, 1.0f) == 0.5f);

    // A constant signal stays put (no drift, no overshoot).
    double t = 0.0;
    for (int i = 0; i < 50; ++i)
    {
        t += 1.0 / 60.0;
        f.filter (0.5f, t, 1.5f, 3.0f, 1.0f);
    }
    CHECK (std::fabs (f.prevFiltered - 0.5f) < 1e-4f);

    // Steady jitter around a mean is attenuated (that is the whole point).
    spatcore::dsp::OneEuroFilter jf;
    double tj = 0.0;
    float maxDeviation = 0.0f;
    jf.filter (0.0f, tj, 1.5f, 3.0f, 1.0f);
    for (int i = 0; i < 200; ++i)
    {
        tj += 1.0 / 60.0;
        const float noise = (i % 2 == 0 ? 0.02f : -0.02f);   // ±0.02 alternating
        const float out = jf.filter (noise, tj, 1.5f, 3.0f, 1.0f);
        if (i > 20)
            maxDeviation = std::max (maxDeviation, std::fabs (out));
    }
    CHECK (maxDeviation < 0.01f);        // less than half the input excursion

    // A fast ramp is tracked closely (adaptive cutoff opens with speed).
    spatcore::dsp::OneEuroFilter rf;
    double tr = 0.0;
    float value = 0.0f, out = 0.0f;
    rf.filter (0.0f, tr, 1.5f, 3.0f, 1.0f);
    for (int i = 0; i < 60; ++i)         // 1 s at 60 Hz, 2 rad/s
    {
        tr += 1.0 / 60.0;
        value += 2.0f / 60.0f;
        out = rf.filter (value, tr, 1.5f, 3.0f, 1.0f);
    }
    CHECK (std::fabs (out - value) < 0.15f);   // lag well under 0.1 s of travel

    f.reset();
    CHECK (! f.initialized);
}

//==============================================================================
#ifdef SPATCORE_TEST_SOFA_FIXTURE
// binaural/SofaLoader + SofaHrtfRenderer — real-file coverage against the
// bundled SADIE II KU100 set: grid bake sanity, ITD extraction, FFT cook,
// and an end-to-end impulse render through the partitioned convolver.
static void testSofaLoaderAndRenderer()
{
    using namespace spatcore::binaural;
    constexpr float pi = 3.14159265358979f;
    const double fs = 48000.0;
    const int block = 512;

    const juce::File fixture (SPATCORE_TEST_SOFA_FIXTURE);
    const auto load = sofa::loadSofaFile (fixture, fs);
    if (load.database == nullptr)
    {
        std::fprintf (stderr, "FAIL: SOFA fixture load: %s\n", load.status.toRawUTF8());
        ++failures;
        return;
    }
    const auto& db = *load.database;
    CHECK (db.hrirLength >= 64 && db.hrirLength <= 1024);
    CHECK (db.sampleRate == fs);

    // ITD extraction at ear level, el = 0 (grid el index 9):
    //   az 90° (source right) → LEFT ear is far: relL in ~[0.4, 1.0] ms, relR = 0.
    //   az 270° mirrors. az 0° is symmetric within ~0.15 ms.
    const int elMid = 9;
    {
        const float relL = db.relDelaySec[(size_t) db.delayIndex (18, elMid, 0)];   // az 90°
        const float relR = db.relDelaySec[(size_t) db.delayIndex (18, elMid, 1)];
        CHECK (relR == 0.0f);
        CHECK (relL > 0.0004f && relL < 0.0010f);

        const float relL2 = db.relDelaySec[(size_t) db.delayIndex (54, elMid, 0)];  // az 270°
        const float relR2 = db.relDelaySec[(size_t) db.delayIndex (54, elMid, 1)];
        CHECK (relL2 == 0.0f);
        CHECK (relR2 > 0.0004f && relR2 < 0.0010f);

        const float fl = db.relDelaySec[(size_t) db.delayIndex (0, elMid, 0)];      // az 0°
        const float fr = db.relDelaySec[(size_t) db.delayIndex (0, elMid, 1)];
        CHECK (std::fabs (fl - fr) < 0.00015f);
    }

    // HRIRs are aligned: every grid point's max |tap| lands in the first
    // quarter of the IR (the onset strip worked).
    {
        int lateOnsets = 0;
        for (int az = 0; az < HrirDatabase::kNumAz; az += 6)
            for (int el = 0; el < HrirDatabase::kNumEl; el += 3)
                for (int ear = 0; ear < 2; ++ear)
                {
                    const float* h = db.hrirs.data() + db.hrirIndex (az, el, ear);
                    int peak = 0;
                    for (int i = 1; i < db.hrirLength; ++i)
                        if (std::fabs (h[i]) > std::fabs (h[peak]))
                            peak = i;
                    if (peak > db.hrirLength / 4)
                        ++lateOnsets;
                }
        CHECK (lateOnsets == 0);
    }

    // FFT cook shape.
    const auto cooked = cookHrirSet (load.database, block);
    CHECK (cooked != nullptr);
    CHECK (cooked->blockSize == block);
    CHECK (cooked->fftSize == 2 * block);
    CHECK (cooked->numPartitions == (db.hrirLength + block - 1) / block);

    // End-to-end: impulse from hard right through the renderer — right ear
    // leads by the measured ITD and carries more energy.
    {
        SofaHrtfRenderer r;
        r.prepare (fs, block, 1);
        r.publishSet (cooked);
        r.processBlockBegin();
        CHECK (r.hasActiveSet());

        SourceDirection dir;
        dir.azRad = pi / 2.0f;
        dir.distance = 2.0f;

        const int numBlocks = 3;
        std::vector<float> L ((size_t) (numBlocks * block), 0.0f), R (L), input ((size_t) block, 0.0f);
        std::int64_t pos = 0;
        for (int b = 0; b < numBlocks; ++b)
        {
            std::fill (input.begin(), input.end(), 0.0f);
            if (b == 0)
                input[0] = 1.0f;
            r.processSource (0, dir, 0.0f, 1.0f, input.data(),
                             L.data() + pos, R.data() + pos, block, pos);
            pos += block;
        }

        float energyL = 0.0f, energyR = 0.0f;
        for (size_t i = 0; i < L.size(); ++i) { energyL += L[i] * L[i]; energyR += R[i] * R[i]; }
        CHECK (energyR > 0.0f);
        CHECK (energyR > energyL);                       // right ear louder

        // Rendered ITD via interaural cross-correlation (same estimator the
        // loader used), lag of L relative to R, positive = left later.
        int bestLag = 0;
        double bestVal = -1.0;
        for (int lag = -100; lag <= 100; ++lag)
        {
            double acc = 0.0;
            for (int i = 0; i < (int) L.size(); ++i)
            {
                const int j = i - lag;
                if (j >= 0 && j < (int) R.size())
                    acc += (double) L[(size_t) i] * (double) R[(size_t) j];
            }
            if (acc > bestVal) { bestVal = acc; bestLag = lag; }
        }

        const float relL = db.relDelaySec[(size_t) db.delayIndex (18, elMid, 0)];
        CHECK (bestLag > 0);                             // right leads
        CHECK (std::abs (bestLag - (int) std::lround (relL * fs)) <= 3);
    }
}
#endif // SPATCORE_TEST_SOFA_FIXTURE

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
        testMultiChannelEQBankNeutrality();
        testMultiChannelEQBankEnableSemantics();
        testMultiChannelEQBankEquivalence();
        testBiquadCoefficientsMatchAudioPath();
        testBiquadMagnitudeResponse();
        testBiquadGoldenCoefficients();
        testOutputEQProcessorNeutrality();
        testHardwareIndexMapContiguous();
        testHardwareIndexMapSparse();
        testHardwareIndexMapEmptyAndClamp();
        testDeviceHostEnableAllPolicy();
        testTestSignalGeneratorToneFollowsSampleRate();
        testTestSignalGeneratorProtectiveRamp();
        testTestSignalGeneratorDeterministicSeed();
        testTestSignalGeneratorSpeakerIdSequencing();
        testBinauralHeadFrame();
        testHeadFrameMatrixToYawPitchRoll();
        testHeadTrackerZeroComposition();
        testOneEuroFilter();
        testNonFiniteAttitudeIsRefused();
        testEngineSurvivesNonFinitePose();
        testStructuralHrtfItdAndDc();
        testStructuralHrtfRotationContinuity();
#ifdef SPATCORE_TEST_SOFA_FIXTURE
        testSofaLoaderAndRenderer();
#endif
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
