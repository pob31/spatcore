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
      9. dsp/AcousticTap         the (delay + air absorption + level) cell the
                                 direct WFS path, the reverb send and the reverb
                                 return all run: bit-exact degeneracy to a plain
                                 gain matrix at zero delay / zero damping,
                                 delay placement from real geometry, and the
                                 shelf leaving DC alone while cutting HF
     10. reverb/ReverbReturnProcessor  node -> speaker distribution: the matrix
                                 stride (engine max-channel) is honoured
                                 independently of the cell stride (live output
                                 count), per-output delay really is per-output,
                                 the mix ACCUMULATES onto the direct sound
                                 already in the buffer, and skipBlock keeps the
                                 write head moving while post-muted
     11. reverb/ReverbSendMatrix  source -> node send, the mirror image: same
                                 stride trap, per-(source, node) delay, and a
                                 node with no active source is silence rather
                                 than stale data
     12. dsp/AcousticSendMatrix  the promoted send matrix: spatcore::reverb::
                                 ReverbSendMatrix is the SAME type (std::is_same),
                                 so the three tests of item 11 are its tests and
                                 run unmodified; a dsp-qualified instance behaves
                                 identically
     13. rt/RtTripleBuffer      wait-free parameter hand-off: latest wins, the
                                 reader keeps its slot until something new is
                                 published, and a value read while a writer
                                 thread hammers publish() is never torn
     14. dsp/ effects primitives OnePoleSmoother (coefficient law, and the snap
                                 that stops a float one-pole freezing short of
                                 its target), FastDecibels (libm-free dB <-> gain:
                                 accuracy, saturation and the values that must be
                                 EXACT), LfoPhasor (phase wrap, shape values,
                                 keyed-noise determinism), FractionalDelayLine
                                 (the direct path's interpolation, to the bit),
                                 DcBlocker, EnvelopeFollower, Waveshaper curves
     15. effects/ contract       the chain-order parser (strict about exactly the
                                 11 tokens), the parameter PODs and their
                                 defaults, and ModuleSlot: a settled slot does no
                                 arithmetic in either direction, a module that
                                 fades to silence is reset exactly once, a
                                 variant change waits for silence and cancels
                                 cleanly, and a non-finite sample is caught
     16. effects/modules        Tremolo (the prototype's dB-linear law, with the
                                 two waveform legs phase-ALIGNED), Bitcrusher
                                 (exact quantiser steps, full-length hold runs,
                                 keyed dither) and the EQ (bit-identical to the
                                 output EQ bank it wraps); plus every module
                                 bit-transparent when bypassed and at identity
     20. effects/LoopGuard      a runaway on the effect-to-effect feed is
                                 caught in a fixed TIME rather than a fixed
                                 number of blocks, ramped away and released only
                                 once the feed itself is quiet
     21. effects/EffectsEngine  the driver: a block written at n returns at n+1
                                 and an effect-to-effect route at n+2, the same
                                 audio with 0 workers and with N, a lapped ring
                                 detected and resynced, and a pull that yields
                                 silence rather than blocking before prepare or
                                 after release
     19. the effects ENGINE's foundations  the shared input ring's wrap
                                 counter, and effect returns appearing in the
                                 render-source map as their own kind of source
     18. effects/modules, part 2 distortion, dynamics, chorus/flanger, phaser,
                                 reverb and multitap delay: each one's
                                 characteristic law pinned numerically, identity
                                 settings transparent, tails cleared by reset,
                                 and no parameter in range able to produce a
                                 non-finite sample or a runaway
     17. effects/EffectChain     eleven slots in a user-chosen order: a reorder
                                 out and back returns to the reference bit for
                                 bit (module state survives it), latency is the
                                 sum over live slots, and chain bypass and mute
                                 land on exactly dry and exactly silence
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
#include "spatcore/rt/RtTripleBuffer.h"
#include "spatcore/rt/RtThreadPriority.h"
#include "spatcore/gpu/GpuHostWorkPool.h"
#include "spatcore/dsp/DelayTargetSmoother.h"
#include "spatcore/dsp/OnePoleSmoother.h"
#include "spatcore/dsp/FastDecibels.h"
#include "spatcore/dsp/LfoPhasor.h"
#include "spatcore/dsp/FractionalDelayLine.h"
#include "spatcore/dsp/DcBlocker.h"
#include "spatcore/dsp/EnvelopeFollower.h"
#include "spatcore/dsp/Waveshaper.h"
#include "spatcore/effects/EffectsTypes.h"
#include "spatcore/effects/EffectParams.h"
#include "spatcore/effects/EffectModule.h"
#include "spatcore/effects/modules/TremoloModule.h"
#include "spatcore/effects/modules/BitcrusherModule.h"
#include "spatcore/effects/modules/EffectEQModule.h"
#include "spatcore/effects/modules/DistortionModule.h"
#include "spatcore/effects/modules/DynamicsModule.h"
#include "spatcore/effects/modules/ModulationModule.h"
#include "spatcore/effects/modules/PhaserModule.h"
#include "spatcore/effects/modules/reverb/ReverbDelayLine.h"
#include "spatcore/effects/modules/reverb/EarlyReflections.h"
#include "spatcore/effects/modules/reverb/ReverbLfo.h"
#include "spatcore/effects/modules/EffectReverbModule.h"
#include "spatcore/effects/modules/MultitapDelayModule.h"
#include "spatcore/effects/EffectChain.h"
#include "spatcore/effects/LoopGuard.h"
#include "spatcore/effects/EffectsEngineCore.h"
#include "spatcore/effects/EffectsEngine.h"
#include "spatcore/dsp/AcousticTap.h"
#include "spatcore/reverb/ReverbReturnProcessor.h"
#include "spatcore/reverb/ReverbSendMatrix.h"
#include "spatcore/dsp/AcousticSendMatrix.h"
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
#include "spatcore/binaural/HeadAttitudePipeline.h"
#include "spatcore/binaural/BinauralEngine.h"
#include "spatcore/binaural/HeadOrientationSource.h"
#include "spatcore/binaural/StructuralHrtfRenderer.h"
#include "spatcore/dsp/OneEuroFilter.h"
#include "spatcore/wfs/RenderSourceMap.h"
#include "spatcore/rt/SharedInputRingBuffer.h"
#include "spatcore/dsp/StereoDecomposer.h"
#ifdef SPATCORE_TEST_SOFA_FIXTURE
#include "spatcore/binaural/SofaLoader.h"
#include "spatcore/binaural/SofaHrtfRenderer.h"
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <thread>
#include <type_traits>
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
// Heap-allocation probe. A realtime path's promise not to allocate can only be
// checked by counting: every operator new in the program comes through the
// replacement below, and a test that wants to know opens a Scope around the
// calls it is asking about. Counting is off everywhere else, so the other
// tests (and their threads) run exactly as they did.
namespace alloc_probe
{
    static std::atomic<bool> counting { false };
    static std::atomic<long> count { 0 };

    struct Scope
    {
        Scope()  { count.store (0); counting.store (true); }
        ~Scope() { counting.store (false); }

        long allocations() const { return count.load(); }
    };
}

void* operator new (std::size_t n)
{
    if (alloc_probe::counting.load (std::memory_order_relaxed))
        alloc_probe::count.fetch_add (1, std::memory_order_relaxed);

    if (void* p = std::malloc (n > 0 ? n : 1))
        return p;

    throw std::bad_alloc();
}

void* operator new[] (std::size_t n)                    { return operator new (n); }
void operator delete (void* p) noexcept                  { std::free (p); }
void operator delete[] (void* p) noexcept                { std::free (p); }
void operator delete (void* p, std::size_t) noexcept     { std::free (p); }
void operator delete[] (void* p, std::size_t) noexcept   { std::free (p); }

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
//==============================================================================
// rt/RtTripleBuffer - the wait-free twin of RtSnapshot. Two properties matter:
// a reader that polls once per block always sees the NEWEST published value
// (never a queue, never a stale one), and a value it reads is never a mixture
// of two publishes - which is the whole point of the three-slot dance.
//==============================================================================

namespace triplebuffer_test
{
    struct Payload
    {
        std::uint32_t seq = 0;
        std::uint32_t twice = 0;      // must always be 2*seq
        std::uint32_t inverted = 0;   // must always be ~seq
        float scaled = 0.0f;          // must always be seq * 0.5f

        static Payload make (std::uint32_t n) noexcept
        {
            return { n, n * 2u, ~n, (float) n * 0.5f };
        }

        bool isConsistent() const noexcept
        {
            return twice == seq * 2u && inverted == ~seq && scaled == (float) seq * 0.5f;
        }
    };
}

static void testRtTripleBuffer()
{
    using namespace spatcore::rt;
    using triplebuffer_test::Payload;

    // --- single threaded: the hand-off protocol ------------------------------
    {
        RtTripleBuffer<Payload> buf;

        // Before anything is published the reader sees a value-initialised T.
        CHECK (! buf.hasPending());
        CHECK (buf.acquire().seq == 0);
        CHECK (buf.acquire().scaled == 0.0f);

        buf.publish (Payload::make (7));
        CHECK (buf.hasPending());
        CHECK (buf.acquire().seq == 7);
        CHECK (! buf.hasPending());

        // Nothing new: the reader keeps the slot it holds.
        CHECK (buf.acquire().seq == 7);

        // Latest wins - the intermediate publish is dropped, not queued.
        buf.publish (Payload::make (8));
        buf.publish (Payload::make (9));
        CHECK (buf.acquire().seq == 9);
        CHECK (buf.acquire().seq == 9);

        // Many publishes without a reader in between must not corrupt anything.
        for (std::uint32_t i = 10; i < 200; ++i)
            buf.publish (Payload::make (i));
        const Payload& p = buf.acquire();
        CHECK (p.seq == 199);
        CHECK (p.isConsistent());
    }

    // --- two threads: never torn, never stale, never blocking ---------------
    {
        RtTripleBuffer<Payload> buf;
        constexpr std::uint32_t kLast = 199999;

        std::atomic<bool> writerDone { false };
        std::thread writer ([&buf, &writerDone, kLast]
        {
            for (std::uint32_t i = 1; i <= kLast; ++i)      // seq 0 is the pre-publish default
                buf.publish (Payload::make (i));
            writerDone.store (true, std::memory_order_release);
        });

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds (10);
        std::uint32_t highest = 0, previous = 0;
        long long iterations = 0, acquired = 0;
        bool torn = false, wentBackwards = false;

        for (;;)
        {
            const Payload& p = buf.acquire();
            ++iterations;

            // seq 0 is the value-initialised payload the reader sees until the
            // first publish lands - not a torn read.
            if (p.seq != 0 && ! p.isConsistent())
                torn = true;                       // a mixture of two publishes
            if (p.seq < previous)
                wentBackwards = true;              // an older value after a newer one
            if (p.seq > highest)
            {
                highest = p.seq;
                ++acquired;
            }
            previous = p.seq;

            if (highest == kLast)
                break;
            if (writerDone.load (std::memory_order_acquire) && iterations > 4)
                break;                             // writer finished: one more sweep is plenty
            if (iterations > 50000000 || std::chrono::steady_clock::now() > deadline)
                break;                             // never hang the suite
        }

        writer.join();

        // Drain whatever landed after the loop exited.
        for (int i = 0; i < 4; ++i)
        {
            const Payload& p = buf.acquire();
            if (p.seq != 0 && ! p.isConsistent())
                torn = true;
            if (p.seq > highest)
                highest = p.seq;
        }

        CHECK (! torn);
        CHECK (! wentBackwards);
        CHECK (highest == kLast);          // the last publish is always observable
        CHECK (acquired >= 1);
        CHECK (acquired <= iterations);    // the reader never spun waiting for a value
    }
}

//==============================================================================
// dsp/ - the shared primitives the effects modules are built from.
//==============================================================================

static void testOnePoleSmoother()
{
    using namespace spatcore::dsp;

    const double sr = 48000.0;

    // The coefficient is the law the reverb pre/post processors already use.
    {
        OnePoleSmoother s;
        s.setTimeConstant (sr, 0.010f);
        const float expected = 1.0f - std::exp (-1.0f / (48000.0f * 0.010f));
        CHECK (std::fabs (s.getCoefficient() - expected) < 1.0e-9f);
    }

    // tau is a TIME CONSTANT: 63 % of the way after tau, monotonic throughout,
    // and bit-exactly arrived well before 10 tau.
    {
        OnePoleSmoother s;
        s.setTimeConstant (sr, 0.010f);
        s.snap (0.0f);
        s.setTarget (1.0f);

        float previous = 0.0f;
        bool monotonic = true;
        for (int i = 0; i < 480; ++i)                 // one tau
        {
            const float v = s.next();
            if (v < previous)
                monotonic = false;
            previous = v;
        }
        CHECK (monotonic);
        CHECK (std::fabs (previous - 0.6321f) < 1.0e-3f);
        CHECK (! s.isSettled());

        for (int i = 0; i < 4800 - 480; ++i)
            s.next();
        CHECK (s.isSettled());
        CHECK (s.getCurrent() == 1.0f);               // bit-exact, not "close"
    }

    // tau = 0 means no smoothing at all.
    {
        OnePoleSmoother s;
        s.setTimeConstant (sr, 0.0f);
        s.snap (0.0f);
        s.setTarget (0.25f);
        CHECK (s.next() == 0.25f);
        CHECK (s.isSettled());
    }

    // snap() arrives without gliding.
    {
        OnePoleSmoother s;
        s.setTimeConstant (sr, 0.010f);
        s.setTarget (3.0f);
        s.snap (3.0f);
        CHECK (s.isSettled());
        CHECK (s.next() == 3.0f);
    }

    // The stall guard. Without it this glide parks a few hertz short of 96000
    // FOREVER: at that magnitude coef*(target-current) falls under an ULP long
    // before the two are equal.
    {
        OnePoleSmoother s;
        s.setTimeConstant (sr, 0.010f);
        s.setSnapEpsilon (1.0e-4f);
        s.snap (12000.0f);
        s.setTarget (96000.0f);

        for (int i = 0; i < 480 * 20; ++i)            // 20 tau is ample
            s.next();

        CHECK (s.isSettled());
        CHECK (s.getCurrent() == 96000.0f);
    }
}

static void testFastDecibels()
{
    using namespace spatcore::dsp;
    namespace fd = spatcore::dsp::FastDecibels;

    // --- the values that must be EXACT, not merely accurate -----------------
    CHECK (fd::dbToGain (0.0f) == 1.0f);
    CHECK (fd::dbToGain (-0.0f) == 1.0f);
    CHECK (fd::gainToDb (1.0f) == 0.0f);

    for (int k = -100; k <= 100; ++k)
    {
        CHECK (fd::exp2 ((float) k) == std::ldexp (1.0f, k));
        CHECK (fd::log2 (std::ldexp (1.0f, k)) == (float) k);
    }

    // --- accuracy across the range the modules actually use -----------------
    {
        double worstGainRel = 0.0, worstDb = 0.0, worstRoundTrip = 0.0;

        for (int step = -12000; step <= 2400; ++step)          // -120 .. +24 dB
        {
            const float dB = (float) step * 0.01f;
            const double reference = std::pow (10.0, (double) dB / 20.0);

            const float gain = fd::dbToGain (dB);
            const double rel = std::fabs ((double) gain - reference) / reference;
            if (rel > worstGainRel)
                worstGainRel = rel;

            const double back = (double) fd::gainToDb ((float) reference);
            if (std::fabs (back - (double) dB) > worstDb)
                worstDb = std::fabs (back - (double) dB);

            const double round = std::fabs ((double) fd::gainToDb (gain) - (double) dB);
            if (round > worstRoundTrip)
                worstRoundTrip = round;
        }

        CHECK (worstGainRel <= 1.0e-6);
        CHECK (worstDb <= 1.0e-4);
        CHECK (worstRoundTrip <= 1.0e-4);
    }

    // --- saturation instead of misbehaviour ---------------------------------
    const float nan = std::numeric_limits<float>::quiet_NaN();
    CHECK (fd::gainToDb (0.0f) == fd::kMinDb);
    CHECK (fd::gainToDb (-1.0f) == fd::kMinDb);
    CHECK (fd::gainToDb (nan) == fd::kMinDb);
    CHECK (fd::dbToGain (-900.0f) == 0.0f);
    CHECK (fd::exp2 (nan) == 0.0f);
    CHECK (fd::exp2 (200.0f) == fd::exp2 (127.0f));
    CHECK (fd::log2 (0.0f) == -126.0f);
    CHECK (fd::log2 (-2.0f) == -126.0f);
    CHECK (fd::log2 (nan) == -126.0f);

    // A gain of 0.5 is -6.0206 dB, and -6 dB is 0.50119 - the two numbers the
    // tremolo test leans on.
    CHECK (std::fabs (fd::gainToDb (0.5f) + 6.0205999f) < 1.0e-4f);
    CHECK (std::fabs (fd::dbToGain (-6.0f) - 0.5011872f) < 1.0e-6f);
    CHECK (std::fabs (fd::dbToGain (-12.0f) - 0.2511886f) < 1.0e-6f);
}

static void testLfoPhasor()
{
    using namespace spatcore::dsp;

    // --- the ramp ------------------------------------------------------------
    {
        LfoPhasor p;
        p.prepare (1000.0);
        p.setRateHz (125.0f);            // exactly 8 samples per cycle

        const float expected[8] = { 0.0f, 0.125f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f, 0.875f };
        for (int cycle = 0; cycle < 3; ++cycle)
            for (int i = 0; i < 8; ++i)
                CHECK (p.nextPhase() == expected[i]);

        // Rate 0 parks the phase; a rate above the sample rate is clamped
        // rather than aliasing the wrap.
        p.reset();
        p.setRateHz (0.0f);
        CHECK (p.nextPhase() == 0.0f);
        CHECK (p.nextPhase() == 0.0f);
        p.setRateHz (100000.0f);
        p.nextPhase();
        CHECK (p.getPhase() >= 0.0f && p.getPhase() < 1.0f);
    }

    // --- the shapes, at the phases the tremolo blend depends on -------------
    {
        CHECK (LfoPhasor::shapeValue (LFOWaveforms::Sine, 0.0f) == -1.0f);
        CHECK (std::fabs (LfoPhasor::shapeValue (LFOWaveforms::Sine, 0.5f) - 1.0f) < 1.0e-6f);
        CHECK (std::fabs (LfoPhasor::shapeValue (LFOWaveforms::Sine, 0.25f)) < 1.0e-6f);

        CHECK (LfoPhasor::shapeValue (LFOWaveforms::Triangle, 0.0f) == -1.0f);
        CHECK (LfoPhasor::shapeValue (LFOWaveforms::Triangle, 0.25f) == 0.0f);
        CHECK (LfoPhasor::shapeValue (LFOWaveforms::Triangle, 0.5f) == 1.0f);
        CHECK (LfoPhasor::shapeValue (LFOWaveforms::Triangle, 0.125f) == -0.5f);

        CHECK (LfoPhasor::shapeValue (LFOWaveforms::Sawtooth, 0.0f) == -1.0f);
        CHECK (LfoPhasor::shapeValue (LFOWaveforms::Square, 0.25f) == -1.0f);
        CHECK (LfoPhasor::shapeValue (LFOWaveforms::Square, 0.75f) == 1.0f);
        CHECK (LfoPhasor::shapeValue (LFOWaveforms::Off, 0.3f) == 0.0f);
    }

    // --- Random is deterministic per key, and piecewise linear --------------
    {
        LfoPhasor a, b, c;
        a.prepare (48000.0);  a.setNoiseKey (12345);  a.reset();  a.setRateHz (50.0f);
        b.prepare (48000.0);  b.setNoiseKey (12345);  b.reset();  b.setRateHz (50.0f);
        c.prepare (48000.0);  c.setNoiseKey (999);    c.reset();  c.setRateHz (50.0f);

        bool sameKeyIdentical = true, differentKeyDiffers = false, inRange = true;
        for (int i = 0; i < 1000; ++i)
        {
            const float va = a.nextValue (LFOWaveforms::Random);
            const float vb = b.nextValue (LFOWaveforms::Random);
            const float vc = c.nextValue (LFOWaveforms::Random);

            if (! bitEqualFloat (va, vb))
                sameKeyIdentical = false;
            if (! bitEqualFloat (va, vc))
                differentKeyDiffers = true;
            if (va < -1.0f || va > 1.0f)
                inRange = false;
        }
        CHECK (sameKeyIdentical);
        CHECK (differentKeyDiffers);
        CHECK (inRange);
    }

    // Inside one period the Random shape is a straight line: the second
    // difference is zero except where a wrap picks a new target.
    {
        LfoPhasor p;
        p.prepare (48000.0);
        p.setNoiseKey (7);
        p.reset();
        p.setRateHz (48.0f);                  // 1000 samples per period

        float v0 = p.nextValue (LFOWaveforms::Random);
        float v1 = p.nextValue (LFOWaveforms::Random);
        bool linear = true;
        for (int i = 2; i < 900; ++i)         // stay well inside the first period
        {
            const float v2 = p.nextValue (LFOWaveforms::Random);
            if (std::fabs ((v2 - v1) - (v1 - v0)) > 1.0e-6f)
                linear = false;
            v0 = v1;
            v1 = v2;
        }
        CHECK (linear);
    }
}

static void testFractionalDelayLine()
{
    using namespace spatcore::dsp;

    // Sizing: a power-of-two ring with room for the interpolation partner.
    {
        FractionalDelayLine d;
        d.prepare (64);
        CHECK (d.getLength() == 128);
        CHECK (d.getMaxDelaySamples() == 126);
    }

    // An impulse at a half-sample delay splits evenly across two samples -
    // the property that makes a moving source glide instead of stepping.
    {
        FractionalDelayLine d;
        d.prepare (64);

        float out[40] = {};
        for (int n = 0; n < 40; ++n)
        {
            d.write (n == 0 ? 1.0f : 0.0f);
            out[n] = d.readLinear (10.5f);
        }

        CHECK (out[10] == 0.5f);
        CHECK (out[11] == 0.5f);
        for (int n = 0; n < 40; ++n)
            if (n != 10 && n != 11)
                CHECK (out[n] == 0.0f);
    }

    // A whole-sample delay is exact, and delay 0 is the sample just written.
    {
        FractionalDelayLine d;
        d.prepare (64);

        float out[40] = {};
        for (int n = 0; n < 40; ++n)
        {
            d.write (n == 0 ? 1.0f : 0.0f);
            out[n] = d.readLinear (10.0f);
        }
        CHECK (out[10] == 1.0f);
        CHECK (out[9] == 0.0f);
        CHECK (out[11] == 0.0f);

        d.reset();
        d.write (0.75f);
        CHECK (d.readLinear (0.0f) == 0.75f);
        CHECK (d.readInteger (0) == 0.75f);
    }

    // Reading across the wrap is the same as reading anywhere else.
    {
        FractionalDelayLine d;
        d.prepare (64);
        for (int n = 0; n < 300; ++n)
            d.write (0.001f * (float) n);
        CHECK (std::fabs (d.readLinear (100.0f) - 0.001f * 199.0f) < 1.0e-6f);
    }

    // The mask is only legitimate if it agrees with a modulo reference on every
    // index, so check it against one over a long hash-driven trajectory.
    {
        FractionalDelayLine d;
        d.prepare (256);
        const int length = d.getLength();
        std::vector<float> reference ((size_t) length, 0.0f);
        int refWrite = 0;
        bool identical = true;

        for (int n = 0; n < 1000; ++n)
        {
            const float x = FrDiffusion::hashNoiseBipolar ((std::uint32_t) n, 4242u);
            d.write (x);
            reference[(size_t) refWrite] = x;
            refWrite = (refWrite + 1) % length;

            const float delay = 0.5f * (float) ((n * 37) % 250);

            // The same arithmetic as InputBufferProcessor's read, with modulo.
            float pos = (float) (refWrite - 1) - delay;
            while (pos < 0.0f)
                pos += (float) length;
            const int p1 = (int) pos % length;
            const int p2 = (p1 + 1) % length;
            const float frac = pos - (float) (int) pos;
            const float expected = reference[(size_t) p1]
                                 + frac * (reference[(size_t) p2] - reference[(size_t) p1]);

            if (! bitEqualFloat (d.readLinear (delay), expected))
                identical = false;
        }
        CHECK (identical);
    }

    // Out-of-range asks clamp instead of reading out of bounds.
    {
        FractionalDelayLine d;
        d.prepare (32);
        for (int n = 0; n < 100; ++n)
            d.write (0.5f);
        CHECK (std::fabs (d.readLinear (-5.0f) - 0.5f) < 1.0e-6f);
        CHECK (std::fabs (d.readLinear (1.0e9f) - 0.5f) < 1.0e-6f);
        CHECK (std::fabs (d.readLinear (std::numeric_limits<float>::quiet_NaN()) - 0.5f) < 1.0e-6f);
    }
}

static void testDcBlocker()
{
    using namespace spatcore::dsp;

    // DC goes away...
    {
        DcBlocker b;
        b.prepare (48000.0, 5.0f);
        float y = 0.0f;
        for (int i = 0; i < 24000; ++i)
            y = b.processSample (1.0f);
        CHECK (std::fabs (y) < 1.0e-4f);
    }

    // ...and audio does not. A 1 kHz tone is untouched to within a fraction of
    // a percent (a cutoff of 5 Hz is three decades below it).
    {
        DcBlocker b;
        b.prepare (48000.0, 5.0f);
        float peak = 0.0f;
        for (int i = 0; i < 48000; ++i)
        {
            const float x = std::sin (6.2831853f * 1000.0f * (float) i / 48000.0f);
            const float y = b.processSample (x);
            if (i > 4800 && std::fabs (y) > peak)
                peak = std::fabs (y);
        }
        CHECK (peak > 0.99f);
        CHECK (peak < 1.001f);
    }

    // reset() really clears the recursion, and a decayed tail parks at true
    // zero rather than in denormal territory.
    {
        DcBlocker b;
        b.prepare (48000.0);
        for (int i = 0; i < 1000; ++i)
            b.processSample (1.0f);
        b.reset();
        CHECK (b.processSample (0.0f) == 0.0f);

        for (int i = 0; i < 100; ++i)
            b.processSample (1.0f);
        std::vector<float> silence (256, 0.0f);
        for (int block = 0; block < 400; ++block)
        {
            std::fill (silence.begin(), silence.end(), 0.0f);   // processBlock is IN PLACE
            b.processBlock (silence.data(), 256);
        }
        CHECK (b.processSample (0.0f) == 0.0f);
    }
}

static void testEnvelopeFollower()
{
    using namespace spatcore::dsp;

    // Peak, instant attack: one sample to the top, then an exponential decay
    // with the release as its time constant.
    {
        EnvelopeFollower e;
        e.prepare (48000.0);
        e.setMode (EnvelopeFollower::Mode::Peak);
        e.setTimes (0.0f, 100.0f);

        CHECK (e.processSample (1.0f) == 1.0f);
        for (int i = 0; i < 4800; ++i)
            e.processSample (0.0f);
        CHECK (std::fabs (e.getValue() - 0.36788f) < 2.0e-3f);
    }

    // A real attack time is a time constant too.
    {
        EnvelopeFollower e;
        e.prepare (48000.0);
        e.setTimes (10.0f, 100.0f);
        for (int i = 0; i < 480; ++i)
            e.processSample (1.0f);
        CHECK (std::fabs (e.getValue() - 0.63212f) < 2.0e-3f);
    }

    // Rms mode follows the mean square; getRms and getDb undo that for you.
    {
        EnvelopeFollower e;
        e.prepare (48000.0);
        e.setMode (EnvelopeFollower::Mode::Rms);
        e.setTimes (1.0f, 1.0f);
        for (int i = 0; i < 48000; ++i)
            e.processSample (0.5f);

        CHECK (std::fabs (e.getValue() - 0.25f) < 1.0e-5f);
        CHECK (std::fabs (e.getRms() - 0.5f) < 1.0e-5f);
        CHECK (std::fabs (e.getDb() + 6.0206f) < 1.0e-2f);
    }

    // A block run agrees with the per-sample path, and reset clears it.
    {
        EnvelopeFollower a, b;
        a.prepare (48000.0);
        b.prepare (48000.0);
        a.setTimes (5.0f, 50.0f);
        b.setTimes (5.0f, 50.0f);

        std::vector<float> data (512);
        for (int i = 0; i < 512; ++i)
            data[(size_t) i] = FrDiffusion::hashNoiseBipolar ((std::uint32_t) i, 77u);

        for (int i = 0; i < 512; ++i)
            a.processSample (data[(size_t) i]);
        b.processBlock (data.data(), 512);
        CHECK (bitEqualFloat (a.getValue(), b.getValue()));

        a.reset();
        CHECK (a.getValue() == 0.0f);
    }
}

static void testWaveshaperCurves()
{
    using namespace spatcore::dsp;
    namespace ws = spatcore::dsp::Waveshaper;

    CHECK (ws::hardClip (0.5f) == 0.5f);
    CHECK (ws::hardClip (2.0f) == 0.8f);
    CHECK (ws::hardClip (-2.0f) == -0.8f);
    CHECK (ws::hardClip (2.0f, 1.0f) == 1.0f);

    // Whatever the blend, silence in is silence out - so a bias adds harmonics
    // without adding DC.
    for (float shape : { 0.0f, 0.5f, 1.0f })
    {
        CHECK (ws::blend (0.0f, shape, 0.0f, 0.0f) == 0.0f);
        CHECK (std::fabs (ws::blend (0.0f, shape, 0.3f, std::tanh (0.3f))) < 1.0e-7f);
    }

    // Monotonic across the range (a shaper that folded back would buzz).
    for (float shape : { 0.0f, 0.5f, 1.0f })
    {
        float previous = ws::blend (-3.0f, shape, 0.0f, 0.0f);
        bool monotonic = true;
        for (int i = 1; i <= 600; ++i)
        {
            const float x = -3.0f + 0.01f * (float) i;
            const float v = ws::blend (x, shape, 0.0f, 0.0f);
            if (v < previous - 1.0e-7f)
                monotonic = false;
            previous = v;
        }
        CHECK (monotonic);
    }

    // No bias: odd symmetry (odd harmonics only). With bias: asymmetry, which
    // is where the even harmonics come from.
    {
        const float up = ws::blend (0.5f, 1.0f, 0.0f, 0.0f);
        const float down = ws::blend (-0.5f, 1.0f, 0.0f, 0.0f);
        CHECK (std::fabs (up + down) < 1.0e-6f);

        const float tb = std::tanh (0.3f);
        const float upB = ws::blend (0.5f, 1.0f, 0.3f, tb);
        const float downB = ws::blend (-0.5f, 1.0f, 0.3f, tb);
        CHECK (std::fabs (upB + downB) > 1.0e-3f);
    }
}

//==============================================================================
// effects/ - the contract every module and chain is built on.
//==============================================================================

static void testChainOrderParse()
{
    using namespace spatcore::effects;

    ChainOrder order {};

    // The canonical order, and one that reverses it, with whitespace the app's
    // text field will let through.
    CHECK (parseChainOrder ("dist,eq1,eq2,dyn1,dyn2,mod,phaser,trem,reverb,delay,crush", order));
    CHECK (order == kDefaultOrder);

    CHECK (parseChainOrder (" crush , delay ,\treverb,trem,phaser,mod,dyn2,dyn1,eq2,eq1,dist ", order));
    for (int i = 0; i < kNumModuleSlots; ++i)
        CHECK (order[(size_t) i] == (std::uint8_t) (kNumModuleSlots - 1 - i));

    CHECK (parseChainOrder ("DIST,Eq1,eQ2,DYN1,dyn2,MOD,Phaser,TREM,reverb,DELAY,Crush", order));
    CHECK (order == kDefaultOrder);

    // Everything else is refused, and - the part that matters live - a refused
    // string leaves the caller's order untouched, so a typo cannot silently
    // reorder a running chain.
    const ChainOrder sentinel { 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9 };
    const char* bad[] =
    {
        "dist,dist,eq2,dyn1,dyn2,mod,phaser,trem,reverb,delay,crush",   // duplicate
        "dist,eq1,eq2,dyn1,dyn2,mod,phaser,trem,reverb,delay",          // one short
        "dist,eq1,eq2,dyn1,dyn2,mod,phaser,trem,reverb,delay,crush,eq1",// one too many
        "dist,eq3,eq2,dyn1,dyn2,mod,phaser,trem,reverb,delay,crush",    // unknown
        "dist,eq1,eq2,dyn1,dyn2,mod,phaser,trem,reverb,delay,crush,",   // trailing comma
        "dist,,eq2,dyn1,dyn2,mod,phaser,trem,reverb,delay,crush",       // empty token
        "dist eq1,eq2,dyn1,dyn2,mod,phaser,trem,reverb,delay,crush",    // missing comma
        "",
        "   "
    };

    for (const char* csv : bad)
    {
        ChainOrder out = sentinel;
        CHECK (! parseChainOrder (csv, out));
        CHECK (out == sentinel);
    }

    ChainOrder nullOut = sentinel;
    CHECK (! parseChainOrder (nullptr, nullOut));
    CHECK (nullOut == sentinel);

    // The cooked form has the same rule.
    CHECK (isValidChainOrder (kDefaultOrder));
    ChainOrder duplicated = kDefaultOrder;
    duplicated[3] = duplicated[4];
    CHECK (! isValidChainOrder (duplicated));
    ChainOrder outOfRange = kDefaultOrder;
    outOfRange[0] = kNumModuleSlots;
    CHECK (! isValidChainOrder (outOfRange));

    // Every slot token is distinct and non-empty - the parser's uniqueness
    // assumption, and the wire's.
    for (int i = 0; i < kNumModuleSlots; ++i)
    {
        CHECK (kSlots[i].token != nullptr && kSlots[i].token[0] != '\0');
        for (int j = i + 1; j < kNumModuleSlots; ++j)
            CHECK (std::strcmp (kSlots[i].token, kSlots[j].token) != 0);
    }
}

static void testEffectParamsPod()
{
    using namespace spatcore::effects;

    EffectChannelParams p;

    // Every module starts bypassed: a new effects channel is transparent until
    // somebody asks for something.
    CHECK (p.dist.bypass == 1);
    CHECK (p.eq[0].bypass == 1 && p.eq[1].bypass == 1);
    CHECK (p.dyn[0].bypass == 1 && p.dyn[1].bypass == 1);
    CHECK (p.mod.bypass == 1);
    CHECK (p.phaser.bypass == 1);
    CHECK (p.trem.bypass == 1);
    CHECK (p.reverb.bypass == 1);
    CHECK (p.delay.bypass == 1);
    CHECK (p.crush.bypass == 1);

    CHECK (p.mute == 0);
    CHECK (p.chainBypass == 0);
    CHECK (p.revision == 0);
    CHECK (p.inputTrimLin == 1.0f);
    CHECK (p.order == kDefaultOrder);

    // Spot-check the tables that are easy to mistype.
    CHECK (p.eq[1].shape[0] == 1 && p.eq[1].shape[4] == 5 && p.eq[1].shape[5] == 6);
    CHECK (p.eq[0].freqHz[0] == 80.0f && p.eq[0].freqHz[5] == 12000.0f);
    CHECK (p.eq[0].q[3] == 0.7f);
    CHECK (p.crush.ditherDb == -96.0f);          // dither OFF by default
    CHECK (p.crush.bits == 8.0f && p.crush.rateHz == 12000.0f);
    CHECK (p.trem.depthDb == 12.0f && p.trem.rateHz == 4.0f);
    CHECK (p.delay.tapTimeMs[0] == 375.0f && p.delay.tapTimeMs[7] == 3000.0f);
    CHECK (p.delay.tapLevelDb[1] == -2.0f && p.delay.tapLevelDb[7] == -14.0f);
    CHECK (p.dyn[0].compRatio == 4.0f && p.dyn[0].lookaheadMs == 1.0f);
    CHECK (p.dyn[0].compDetectorDelayMs == 0.0f);
    CHECK (p.dist.shape == 0.5f && p.dist.outputDb == -6.0f);

    // It crosses a thread boundary by value, so keep an eye on the size.
    CHECK (sizeof (EffectChannelParams) < 2048);
}

//==============================================================================
// A stand-in module for the slot tests: a gain with counters, taking its bypass
// and its "variant" from the crusher's real fields so nothing test-only leaks
// into the parameter structs.
//==============================================================================

namespace slot_test
{
    using namespace spatcore::effects;

    class GainModule : public IEffectModule
    {
    public:
        ModuleId type() const noexcept override { return ModuleId::Crush; }

        void prepare (const ChainConfig& cfg) override
        {
            preparedRate = cfg.sampleRate;
            preparedBlock = cfg.maxBlock;
        }

        void reset() noexcept override { ++resets; }

        ParamApplyInfo applyParams (const EffectChannelParams& p, int) noexcept override
        {
            pendingFilter = p.crush.filter;
            return { p.crush.bypass != 0, pendingFilter != runningFilter };
        }

        void commitPendingVariant() noexcept override
        {
            runningFilter = pendingFilter;
            ++commits;
        }

        void process (float* inout, int n) noexcept override
        {
            ++processCalls;

            for (int i = 0; i < n; ++i)
                inout[i] *= gain;

            if (emitNonFinite != 0)
                inout[n - 1] = (emitNonFinite == 1)
                                 ? std::numeric_limits<float>::quiet_NaN()
                                 : std::numeric_limits<float>::infinity();
        }

        int getLatencySamples() const noexcept override { return latency; }

        float gain = 0.5f;
        int latency = 0;
        int emitNonFinite = 0;          // 0 none, 1 NaN, 2 Inf
        int resets = 0, commits = 0, processCalls = 0;
        std::uint8_t runningFilter = 0, pendingFilter = 0;
        double preparedRate = 0.0;
        int preparedBlock = 0;
    };

    struct Rig
    {
        ModuleSlot slot;
        GainModule* module = nullptr;
        EffectChannelParams params;

        explicit Rig (int maxBlock = 64)
        {
            ChainConfig cfg;
            cfg.sampleRate = 48000.0;
            cfg.maxBlock = maxBlock;

            auto owned = std::make_unique<GainModule>();
            module = owned.get();
            slot.prepare (cfg, std::move (owned));
        }

        void apply() noexcept { slot.applyParams (params, 0); }

        /** Runs blocks of DC 1.0 and returns the last output sample. */
        float run (int blocks, int blockSize = 64) noexcept
        {
            float last = 0.0f;
            std::vector<float> buf ((size_t) blockSize);

            for (int b = 0; b < blocks; ++b)
            {
                std::fill (buf.begin(), buf.end(), 1.0f);
                slot.process (buf.data(), blockSize);
                last = buf[(size_t) blockSize - 1];
            }

            return last;
        }
    };
}

static void testModuleSlotBypassFade()
{
    using namespace spatcore::effects;
    slot_test::Rig rig;

    CHECK (rig.module->preparedRate == 48000.0);
    CHECK (rig.module->preparedBlock == 64);

    // A slot starts bypassed and SETTLED, so the buffer is not touched at all -
    // not multiplied by one, not crossfaded against itself.
    CHECK (rig.slot.isBypassedSettled());
    CHECK (rig.run (4) == 1.0f);
    CHECK (rig.module->processCalls == 0);

    // Switch the module on: the fade is monotonic and hits 63 % after one tau
    // (5 ms = 240 samples at 48 kHz).
    rig.params.crush.bypass = 0;
    rig.apply();

    // Exactly one tau of audio, handed over in a single call so the slot's
    // internal chunking at maxBlock is exercised at the same time.
    float previous = 1.0f;
    bool monotonic = true;
    std::vector<float> buf (240, 1.0f);
    rig.slot.process (buf.data(), 240);
    for (int i = 0; i < 240; ++i)
    {
        if (buf[(size_t) i] > previous)              // gain 0.5: output FALLS as g rises
            monotonic = false;
        previous = buf[(size_t) i];
    }
    CHECK (monotonic);
    CHECK (std::fabs (rig.slot.getFadeGain() - 0.6321f) < 0.01f);
    CHECK (! rig.slot.isActiveSettled());

    // Settled, the module's output goes through untouched: exactly 0.5, not
    // 0.5 plus a crossfade rounding.
    rig.run (40);
    CHECK (rig.slot.isActiveSettled());
    CHECK (rig.run (2) == 0.5f);

    const int resetsBeforeBypass = rig.module->resets;

    // And back. Fading out resets the module exactly once, at silence.
    rig.params.crush.bypass = 1;
    rig.apply();
    rig.run (40);
    CHECK (rig.slot.isBypassedSettled());
    CHECK (rig.run (2) == 1.0f);
    CHECK (rig.slot.silentResets.load() == 1);
    CHECK (rig.module->resets == resetsBeforeBypass + 1);

    // Bypassed and settled again, the module is skipped entirely.
    const int callsBefore = rig.module->processCalls;
    rig.run (4);
    CHECK (rig.module->processCalls == callsBefore);

    // A bypass revoked mid-fade just turns the fade around - no reset.
    rig.params.crush.bypass = 0;
    rig.apply();
    rig.run (2);
    const int resetsMidFade = rig.module->resets;
    rig.params.crush.bypass = 1;
    rig.apply();
    rig.run (1);
    rig.params.crush.bypass = 0;
    rig.apply();
    rig.run (40);
    CHECK (rig.slot.isActiveSettled());
    CHECK (rig.module->resets == resetsMidFade);
}

static void testModuleSlotVariantSwitch()
{
    using namespace spatcore::effects;
    slot_test::Rig rig;

    // Get it running and settled.
    rig.params.crush.bypass = 0;
    rig.apply();
    rig.run (40);
    CHECK (rig.slot.isActiveSettled());
    CHECK (rig.module->commits == 0);

    // A change that cannot be interpolated: the module keeps running the old
    // value while the slot fades out, and the swap happens in silence.
    rig.params.crush.filter = 1;
    rig.apply();
    CHECK (rig.module->runningFilter == 0);       // still the old one
    CHECK (! rig.slot.isBypassedSettled());

    rig.run (40);
    CHECK (rig.module->commits == 1);
    CHECK (rig.module->runningFilter == 1);

    // ...and it comes straight back up, because bypass was never asked for.
    rig.run (40);
    CHECK (rig.slot.isActiveSettled());
    CHECK (rig.run (2) == 0.5f);

    // A change taken back before the fade completes cancels: no commit, no
    // reset, no audible dip to the bottom.
    const int commitsBefore = rig.module->commits;
    const int resetsBefore = rig.module->resets;
    rig.params.crush.filter = 0;
    rig.apply();
    rig.run (2);
    CHECK (! rig.slot.isBypassedSettled());
    rig.params.crush.filter = 1;                  // back to what is running
    rig.apply();
    rig.run (40);
    CHECK (rig.slot.isActiveSettled());
    CHECK (rig.module->commits == commitsBefore);
    CHECK (rig.module->resets == resetsBefore);

    // A variant change while the slot is already bypassed and silent is taken
    // immediately - there is nothing to fade.
    rig.params.crush.bypass = 1;
    rig.apply();
    rig.run (40);
    CHECK (rig.slot.isBypassedSettled());
    rig.params.crush.filter = 0;
    rig.apply();
    rig.run (2);
    CHECK (rig.module->runningFilter == 0);
}

static void testNaNGuard()
{
    using namespace spatcore::effects;
    slot_test::Rig rig;

    rig.params.crush.bypass = 0;
    rig.apply();
    rig.run (40);
    CHECK (rig.slot.isActiveSettled());

    // A module that goes bad is silenced and reset, not passed on: a NaN
    // reaching the return ring would spread through the whole render.
    const int resetsBefore = rig.module->resets;
    rig.module->emitNonFinite = 1;

    std::vector<float> buf (64, 1.0f);
    rig.slot.process (buf.data(), 64);
    for (int i = 0; i < 64; ++i)
        CHECK (buf[(size_t) i] == 0.0f);
    CHECK (rig.slot.nanTrips.load() == 1);
    CHECK (rig.module->resets == resetsBefore + 1);

    // Recovered: the next clean block passes normally and nothing re-trips.
    rig.module->emitNonFinite = 0;
    CHECK (rig.run (1) == 0.5f);
    CHECK (rig.slot.nanTrips.load() == 1);

    // Infinity counts too.
    rig.module->emitNonFinite = 2;
    std::fill (buf.begin(), buf.end(), 1.0f);
    rig.slot.process (buf.data(), 64);
    for (int i = 0; i < 64; ++i)
        CHECK (buf[(size_t) i] == 0.0f);
    CHECK (rig.slot.nanTrips.load() == 2);
}

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

    // Yaw is SEAT-RELATIVE, not a world heading. This is the property the app's
    // two controls are built on: the placement angle picks a seat and points the
    // head at the origin, and yaw turns the head from there — so yaw 0 faces the
    // origin at EVERY seat, and a given yaw produces the same azimuth for the
    // origin whatever the seat. Get this backwards and "turn my head 90° right"
    // would mean different things at different seats, and a head tracker's Set
    // Zero would stop meaning "looking at the stage".
    {
        float offset[9];
        hf::yawPitchRollToMatrix (pi / 2.0f, 0.0f, 0.0f, offset);    // +90° = turn right

        for (float alpha : { 0.0f, pi / 2.0f, -pi / 4.0f, 2.5f })
        {
            // Seat on the circle of radius 5 at bearing alpha, per the app's
            // placement law (d·sin α, −d·cos α).
            pose.x = 5.0f * std::sin (alpha);
            pose.y = -5.0f * std::cos (alpha);
            pose.z = 0.0f;
            hf::composeWithBaseline (alpha, offset, pose.R);

            // Having turned 90° to the right, the origin is now 90° to the LEFT.
            auto d = hf::directionInHeadFrame (pose, 0.0f, 0.0f, 0.0f);
            CHECK (std::fabs (d.azRad + pi / 2.0f) < tol);
            CHECK (std::fabs (d.distance - 5.0f) < tol);
        }
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
// binaural/HeadFrame — trackerQuatToYawPitchRoll converts a hardware tracker's
// body->world quaternion (X forward / Y left / Z up, per the headtracker
// PROTOCOL.md) into the head-frame angles the binaural renderer consumes.
//
// The three directional cases below ARE the sign contract: get one of them
// backwards and the binaural image rotates the wrong way, which is trivially
// audible but not something any other test would catch.
static void testTrackerQuatToHeadAngles()
{
    namespace hf = spatcore::binaural::headframe;
    constexpr float pi = 3.14159265358979f;
    const float tol = 1e-3f;

    // Quaternion for a rotation of `angle` about a body axis, Hamilton w,x,y,z.
    auto axisAngle = [] (float ax, float ay, float az, float angle,
                         float& w, float& x, float& y, float& z)
    {
        const float s = std::sin (angle * 0.5f);
        w = std::cos (angle * 0.5f);
        x = ax * s; y = ay * s; z = az * s;
    };

    float w, x, y, z, yaw, pitch, roll;

    // Identity → level and facing front.
    hf::trackerQuatToYawPitchRoll (1.0f, 0.0f, 0.0f, 0.0f, yaw, pitch, roll);
    CHECK (std::fabs (yaw) < tol && std::fabs (pitch) < tol && std::fabs (roll) < tol);

    for (const float a : { 15.0f, 45.0f, 90.0f, -30.0f })
    {
        const float rad = a * pi / 180.0f;

        // Turn RIGHT: in a forward/left/up body frame that is a NEGATIVE
        // rotation about Z (the nose swings from +X toward -Y).
        axisAngle (0.0f, 0.0f, 1.0f, -rad, w, x, y, z);
        hf::trackerQuatToYawPitchRoll (w, x, y, z, yaw, pitch, roll);
        CHECK (std::fabs (yaw - rad) < tol);        // +yaw = turn right
        CHECK (std::fabs (pitch) < tol);
        CHECK (std::fabs (roll) < tol);

        // Look UP: negative rotation about body Y (Ry(+t) tips the nose down).
        axisAngle (0.0f, 1.0f, 0.0f, -rad, w, x, y, z);
        hf::trackerQuatToYawPitchRoll (w, x, y, z, yaw, pitch, roll);
        CHECK (std::fabs (pitch - rad) < tol);      // +pitch = look up
        CHECK (std::fabs (yaw) < tol);
        CHECK (std::fabs (roll) < tol);

        // RIGHT EAR DOWN: positive rotation about body X lifts the left ear.
        axisAngle (1.0f, 0.0f, 0.0f, rad, w, x, y, z);
        hf::trackerQuatToYawPitchRoll (w, x, y, z, yaw, pitch, roll);
        CHECK (std::fabs (roll - rad) < tol);       // +roll = right ear down
        CHECK (std::fabs (yaw) < tol);
        CHECK (std::fabs (pitch) < tol);
    }

    // Round-trip through the rotation, not through the angle triple: distinct
    // triples can name the same rotation. This also pins the change of basis
    // as a whole rather than three independent signs.
    {
        // A tumbling sequence of arbitrary unit quaternions (deterministic).
        uint32_t s = 0x9E3779B9u;
        auto nextf = [&s] { s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                            return (float) (s & 0xFFFFFFu) / 8388608.0f - 1.0f; };

        for (int i = 0; i < 400; ++i)
        {
            float q[4] = { nextf(), nextf(), nextf(), nextf() };
            const float n = std::sqrt (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
            if (n < 0.1f)
                continue;
            for (auto& c : q) c /= n;

            hf::trackerQuatToYawPitchRoll (q[0], q[1], q[2], q[3], yaw, pitch, roll);
            CHECK (std::isfinite (yaw) && std::isfinite (pitch) && std::isfinite (roll));

            // Rebuild R_offset from the angles and from the quaternion via the
            // documented permutation; the two must agree.
            float fromAngles[9];
            hf::yawPitchRollToMatrix (yaw, pitch, roll, fromAngles);

            const float w2 = q[0], x2 = q[1], y2 = q[2], z2 = q[3];
            const float r0 = 1.0f - 2.0f * (y2*y2 + z2*z2);
            const float r1 = 2.0f * (x2*y2 - w2*z2);
            const float r2 = 2.0f * (x2*z2 + w2*y2);
            const float r3 = 2.0f * (x2*y2 + w2*z2);
            const float r4 = 1.0f - 2.0f * (x2*x2 + z2*z2);
            const float r5 = 2.0f * (y2*z2 - w2*x2);
            const float r6 = 2.0f * (x2*z2 - w2*y2);
            const float r7 = 2.0f * (y2*z2 + w2*x2);
            const float r8 = 1.0f - 2.0f * (x2*x2 + y2*y2);
            const float expected[9] = { r4, -r3, -r5, -r1, r0, r2, -r7, r6, r8 };

            for (int k = 0; k < 9; ++k)
                CHECK (std::fabs (fromAngles[k] - expected[k]) < 2e-3f);
        }
    }

    // Degenerate and poisoned wire values must not reach the renderer as NaN:
    // a tracker frame can pass its CRC and still carry a NaN or an infinity.
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const float inf = std::numeric_limits<float>::infinity();
        const float bad[][4] = { { 0, 0, 0, 0 }, { nan, 0, 0, 0 }, { 1, nan, 0, 0 },
                                 { inf, 0, 0, 0 }, { 1, 0, inf, 0 },
                                 { nan, nan, nan, nan } };
        for (const auto& q : bad)
        {
            hf::trackerQuatToYawPitchRoll (q[0], q[1], q[2], q[3], yaw, pitch, roll);
            CHECK (std::isfinite (yaw) && std::isfinite (pitch) && std::isfinite (roll));
        }
    }

    // Gimbal pose (nose straight up) stays finite; roll folds into yaw exactly
    // as matrixToYawPitchRoll documents.
    {
        axisAngle (0.0f, 1.0f, 0.0f, -pi / 2.0f, w, x, y, z);
        hf::trackerQuatToYawPitchRoll (w, x, y, z, yaw, pitch, roll);
        CHECK (std::isfinite (yaw) && std::isfinite (pitch) && std::isfinite (roll));
        CHECK (std::fabs (pitch - pi / 2.0f) < 1e-3f);
        CHECK (roll == 0.0f);
    }
}

//==============================================================================
// binaural/HeadAttitudePipeline — the publish-side half every head-orientation
// source shares: yaw unwrap, 1-Euro smoothing, and the guards that stop a
// single poisoned sample from latching forever.
static void testHeadAttitudePipeline()
{
    using spatcore::binaural::HeadAttitudePipeline;
    using spatcore::binaural::HeadAttitudeTuning;
    constexpr float pi = 3.14159265358979f;

    // Bypass tuning: unwrap and guards only, so the arithmetic is checkable.
    const HeadAttitudeTuning bypass { 0.0f, 0.0f, 1.0f };

    // Crossing the +/-pi seam must not be seen as a full-circle jump. Walk yaw
    // past pi in small steps; every output step must stay small.
    {
        HeadAttitudePipeline p (bypass);
        float prev = 0.0f;
        double t = 0.0;
        for (int i = 0; i <= 40; ++i)
        {
            const float trueYaw = spatcore::binaural::wrapPi (3.0f + (float) i * 0.05f);
            const auto o = p.process (trueYaw, 0.0f, 0.0f, t);
            t += 0.02;
            CHECK (o.valid);
            if (i > 0)
            {
                const float step = std::fabs (spatcore::binaural::wrapPi (o.yawRad - prev));
                CHECK (step < 0.2f);        // ~0.05 rad expected, never ~2*pi
            }
            prev = o.yawRad;
        }
    }

    // Output yaw is always wrapped, however far the unwrapped value has run.
    {
        HeadAttitudePipeline p (bypass);
        double t = 0.0;
        for (int i = 0; i < 500; ++i)
        {
            const auto o = p.process (spatcore::binaural::wrapPi ((float) i * 0.3f),
                                      0.2f, -0.1f, t);
            t += 0.01;
            CHECK (o.yawRad > -pi - 1e-4f && o.yawRad <= pi + 1e-4f);
        }
    }

    // Non-finite input is refused WITHOUT poisoning state: the very next good
    // sample must track normally. This is the property that keeps a single bad
    // frame from killing tracking for the rest of the session.
    {
        HeadAttitudePipeline p (bypass);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const float inf = std::numeric_limits<float>::infinity();

        CHECK (p.process (0.4f, 0.1f, 0.0f, 0.0).valid);
        CHECK (! p.process (nan, 0.1f, 0.0f, 0.01).valid);
        CHECK (! p.process (0.4f, inf, 0.0f, 0.02).valid);

        const auto good = p.process (0.5f, 0.1f, 0.0f, 0.03);
        CHECK (good.valid);
        CHECK (std::fabs (good.yawRad - 0.5f) < 1e-4f);
        CHECK (std::fabs (good.pitchRad - 0.1f) < 1e-4f);
    }

    // Filtering enabled: a constant input settles ON that input (no bias), and
    // the first sample is passed through so tracking starts instantly.
    {
        HeadAttitudePipeline p (HeadAttitudeTuning { 1.5f, 3.0f, 1.0f });
        const auto first = p.process (0.7f, -0.2f, 0.05f, 0.0);
        CHECK (first.valid);
        CHECK (std::fabs (first.yawRad - 0.7f) < 1e-4f);

        double t = 0.0;
        spatcore::binaural::HeadOrientation o {};
        for (int i = 0; i < 400; ++i)
        {
            t += 1.0 / 60.0;
            o = p.process (0.7f, -0.2f, 0.05f, t);
        }
        CHECK (std::fabs (o.yawRad - 0.7f) < 1e-3f);
        CHECK (std::fabs (o.pitchRad + 0.2f) < 1e-3f);
        CHECK (std::fabs (o.rollRad - 0.05f) < 1e-3f);
    }

    // reset() drops the unwrap anchor and the filter history: after it, the
    // next sample is again passed through untouched (the behaviour a zero
    // calibration relies on so it doesn't slew from the pre-tare pose).
    {
        HeadAttitudePipeline p (HeadAttitudeTuning { 1.5f, 3.0f, 1.0f });
        double t = 0.0;
        for (int i = 0; i < 50; ++i)
            p.process (0.0f, 0.0f, 0.0f, t += 1.0 / 60.0);

        p.reset();
        const auto o = p.process (1.1f, 0.3f, -0.2f, t + 1.0 / 60.0);
        CHECK (std::fabs (o.yawRad - 1.1f) < 1e-4f);
        CHECK (std::fabs (o.pitchRad - 0.3f) < 1e-4f);
        CHECK (std::fabs (o.rollRad + 0.2f) < 1e-4f);
    }

    // A pathological timestamp sequence must not produce non-finite output.
    {
        HeadAttitudePipeline p (HeadAttitudeTuning { 1.5f, 3.0f, 1.0f });
        const double times[] = { 0.0, 1e9, -1e9, 0.0, 1e-9 };
        for (const double t : times)
        {
            const auto o = p.process (0.3f, 0.1f, 0.0f, t);
            CHECK (std::isfinite (o.yawRad));
            CHECK (std::isfinite (o.pitchRad));
            CHECK (std::isfinite (o.rollRad));
        }
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


//==============================================================================
// RenderSourceMap — the slot allocator behind the stereo channel type.
// Slot stability is the load-bearing property: OSC ids, matrix rows and GPU
// buffers all key off these slots, so nothing but the channel-type vector may
// move them.
static void testRenderSourceMapBuild()
{
    using Map = spatcore::wfs::RenderSourceMap;

    // Identity: all-mono, channel i <-> source i, nothing derived.
    {
        Map m;
        CHECK (Map::buildIdentity (8, m));
        CHECK (m.count == 8);
        CHECK (m.numInputChannels == 8);
        for (int i = 0; i < 8; ++i)
        {
            CHECK (m.desc[(size_t) i].owningInputChannel == i);
            CHECK (m.desc[(size_t) i].sliceIndex == 0);
            CHECK (! m.desc[(size_t) i].isStereoSlice);
            CHECK (m.firstDerivedSlot[(size_t) i] == -1);
        }
    }

    // Mixed: channels 1 and 3 stereo out of 6.
    {
        uint8_t types[6] = { 0, 1, 0, 1, 0, 0 };
        Map m;
        CHECK (Map::build (types, 6, m));
        CHECK (m.count == 6 + 2 * Map::kDerivedPerStereo);

        // Primary slots keep the channel <-> row identity for every channel.
        for (int i = 0; i < 6; ++i)
        {
            CHECK (m.desc[(size_t) i].owningInputChannel == i);
            CHECK (m.desc[(size_t) i].sliceIndex == 0);
        }
        CHECK (m.desc[1].isStereoSlice);
        CHECK (! m.desc[2].isStereoSlice);

        // Derived slots: contiguous, ordinal-ordered, slices 1..5.
        CHECK (m.firstDerivedSlot[1] == 6);
        CHECK (m.firstDerivedSlot[3] == 6 + Map::kDerivedPerStereo);
        for (int s = 1; s <= Map::kDerivedPerStereo; ++s)
        {
            const auto& d1 = m.desc[(size_t) (6 + s - 1)];
            CHECK (d1.owningInputChannel == 1);
            CHECK (d1.sliceIndex == s);
            CHECK (d1.isStereoSlice);

            const auto& d3 = m.desc[(size_t) (6 + Map::kDerivedPerStereo + s - 1)];
            CHECK (d3.owningInputChannel == 3);
            CHECK (d3.sliceIndex == s);
        }

        // No two sources share (owner, slice).
        for (int a = 0; a < m.count; ++a)
            for (int b = a + 1; b < m.count; ++b)
                CHECK (! (m.desc[(size_t) a].owningInputChannel == m.desc[(size_t) b].owningInputChannel
                       && m.desc[(size_t) a].sliceIndex == m.desc[(size_t) b].sliceIndex));
    }

    // Slot stability: the map is a pure function of the type vector alone.
    // Two builds from equal vectors must agree slot for slot, and a build
    // differing only in a LATER channel's type must not move earlier slots.
    {
        uint8_t typesA[6] = { 0, 1, 0, 0, 0, 0 };
        uint8_t typesB[6] = { 0, 1, 0, 0, 0, 1 };   // channel 5 became stereo
        Map a, b;
        CHECK (Map::build (typesA, 6, a));
        CHECK (Map::build (typesB, 6, b));
        CHECK (a.firstDerivedSlot[1] == b.firstDerivedSlot[1]);
        for (int s = 0; s < a.count; ++s)
        {
            CHECK (a.desc[(size_t) s].owningInputChannel == b.desc[(size_t) s].owningInputChannel);
            CHECK (a.desc[(size_t) s].sliceIndex == b.desc[(size_t) s].sliceIndex);
        }
    }

    // Budget enforcement: 9 stereo channels is one past the cap, and must fail
    // cleanly (empty map, not an overflow).
    {
        uint8_t types[Map::kMaxInputChannels] = {};
        for (int i = 0; i < Map::kMaxStereoChannels + 1; ++i)
            types[i] = Map::Stereo;
        Map m;
        CHECK (! Map::build (types, Map::kMaxInputChannels, m));
        CHECK (m.count == 0);

        // Exactly at the cap is fine, and fills the whole budget.
        for (int i = 0; i < Map::kMaxInputChannels; ++i)
            types[i] = (i < Map::kMaxStereoChannels) ? Map::Stereo : Map::Mono;
        CHECK (Map::build (types, Map::kMaxInputChannels, m));
        CHECK (m.count == Map::kMaxRenderSources);
    }

    // Invalid input: over-count and null both refuse.
    {
        Map m;
        uint8_t t = 0;
        CHECK (! Map::build (nullptr, 1, m));
        CHECK (! Map::build (&t, Map::kMaxInputChannels + 1, m));
    }
}


//==============================================================================
// StereoDecomposer — the Phase 0 pass-through backend, tested through the BASE
// CLASS wherever possible so the Phase 1 STFT backend inherits the contract
// tests (in particular the reconstruction invariant, doc §8) for free.

// Deterministic squirrel-hash test signal, bipolar, distinct per (channel, n).
static float stereoTestSample (int channel, int n)
{
    uint32_t h = (uint32_t) (n * 2654435761u) ^ (uint32_t) (channel * 0x9E3779B9u);
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return ((float) (h & 0xFFFFFF) / (float) 0x7FFFFF) - 1.0f;
}

// Base-class contract: sum of slices reconstructs L+R within -120 dBFS, and
// inactive slots are cleared. Phase 1 backends must pass unchanged (with their
// latency accounted for — latency 0 is asserted separately per backend).
static void checkStereoReconstruction (spatcore::dsp::StereoDecomposer& d,
                                       int numSamples, int seedOffset)
{
    constexpr int kMax = spatcore::dsp::StereoDecomposer::kMaxSlices;
    std::vector<float> left ((size_t) numSamples), right ((size_t) numSamples);
    std::vector<std::vector<float>> slices ((size_t) kMax,
                                            std::vector<float> ((size_t) numSamples, -12345.0f));
    float* slicePtrs[kMax];
    for (int k = 0; k < kMax; ++k)
        slicePtrs[k] = slices[(size_t) k].data();

    for (int n = 0; n < numSamples; ++n)
    {
        left[(size_t) n]  = stereoTestSample (0, n + seedOffset);
        right[(size_t) n] = stereoTestSample (1, n + seedOffset);
    }

    d.process (left.data(), right.data(), slicePtrs, numSamples);

    constexpr float tol = 1.0e-6f;   // -120 dBFS
    for (int n = 0; n < numSamples; ++n)
    {
        float sum = 0.0f;
        for (int k = 0; k < kMax; ++k)
            sum += slices[(size_t) k][(size_t) n];
        CHECK (std::abs (sum - (left[(size_t) n] + right[(size_t) n])) <= tol);
    }

    // Inactive slots: exactly zero, never the sentinel they were prefilled with.
    for (int k = d.getNumActiveSlices(); k < kMax; ++k)
        for (int n = 0; n < numSamples; ++n)
            CHECK (slices[(size_t) k][(size_t) n] == 0.0f);
}

//==============================================================================
// dsp/AcousticTap - the (smoothed fractional delay + air-absorption shelf +
// level) cell shared by the direct WFS path, the reverb send (ReverbFeedThread)
// and the reverb return (ReverbReturnProcessor).
//==============================================================================

namespace acoustictap_test
{
    constexpr int kLineLength = 4096;

    // Deterministic, non-trivial source material.
    inline float src (int n) noexcept
    {
        return 0.37f * std::sin (0.031f * (float) n)
             + 0.19f * std::sin (0.211f * (float) n + 0.7f);
    }

    // Write one block into a ring at blockStart, wrapping.
    inline void writeBlock (std::vector<float>& line, int blockStart,
                            const std::vector<float>& block)
    {
        const int n = (int) block.size();
        for (int i = 0; i < n; ++i)
            line[(size_t) ((blockStart + i) % kLineLength)] = block[(size_t) i];
    }
}

static void testAcousticTapNullIdentity()
{
    using namespace spatcore::dsp;
    using namespace acoustictap_test;

    // Zero delay and zero damping must degenerate EXACTLY to the scalar matrix
    // the reverb feed and return used before this cell existed:
    //     dest[s] += src[s] * level
    // This is the regression guard for "the rewrite changed nothing at
    // defaults" - reverbHFdamping defaults to 0.0 dB/m and a co-located source
    // has zero propagation delay.
    const int numSamples = 256;
    const double sr = 48000.0;
    const float level = 0.6431f;

    AcousticTapCell cell;
    cell.prepare (sr, (int) (sr * 0.010));

    std::vector<float> line ((size_t) kLineLength, 0.0f);
    std::int64_t counter = 0;
    int blockStart = 0;

    for (int block = 0; block < 8; ++block)
    {
        std::vector<float> in ((size_t) numSamples);
        for (int i = 0; i < numSamples; ++i)
            in[(size_t) i] = src (block * numSamples + i);

        writeBlock (line, blockStart, in);

        std::vector<float> got ((size_t) numSamples, 0.0f);

        processAcousticTap (cell, line.data(), kLineLength, blockStart, counter,
                            numSamples, 0.0f, 0.0f, level, sr, got.data());

        for (int i = 0; i < numSamples; ++i)
            CHECK (bitEqualFloat (got[(size_t) i], in[(size_t) i] * level));

        blockStart = (blockStart + numSamples) % kLineLength;
        counter += numSamples;
    }
}

static void testAcousticTapDelayPlacement()
{
    using namespace spatcore::dsp;
    using namespace acoustictap_test;

    // An impulse must come back out delayed by distance / speedOfSound. Uses a
    // 30 m path, the geometry the send matrix would produce for a source 30 m
    // from a reverb node: 30 / 343 * 1000 = 87.463 ms.
    const int numSamples = 512;
    const double sr = 48000.0;
    const float distanceM = 30.0f;
    const float speedOfSound = 343.0f;
    const float delayMs = (distanceM / speedOfSound) * 1000.0f;
    const int expectedSamples = (int) ((delayMs / 1000.0f) * (float) sr);   // 4198

    AcousticTapCell cell;
    cell.prepare (sr, (int) (sr * 0.010));

    // Long enough line for an 87 ms delay at 48 kHz.
    const int lineLen = 1 << 15;
    std::vector<float> line ((size_t) lineLen, 0.0f);

    std::int64_t counter = 0;
    int blockStart = 0;
    int impulseAt = -1;
    int foundAt = -1;
    float peak = 0.0f;

    // The smoother bootstraps on its first observation, so let it settle on
    // silence for a few blocks before the impulse goes in.
    const int settleBlocks = 4;
    const int totalBlocks = 40;

    for (int block = 0; block < totalBlocks; ++block)
    {
        std::vector<float> in ((size_t) numSamples, 0.0f);
        if (block == settleBlocks)
        {
            in[0] = 1.0f;
            impulseAt = block * numSamples;
        }

        for (int i = 0; i < numSamples; ++i)
            line[(size_t) ((blockStart + i) % lineLen)] = in[(size_t) i];

        std::vector<float> got ((size_t) numSamples, 0.0f);
        processAcousticTap (cell, line.data(), lineLen, blockStart, counter,
                            numSamples, delayMs, 0.0f, 1.0f, sr, got.data());

        for (int i = 0; i < numSamples; ++i)
        {
            if (std::fabs (got[(size_t) i]) > peak)
            {
                peak = std::fabs (got[(size_t) i]);
                foundAt = (int) (counter + i);
            }
        }

        blockStart = (blockStart + numSamples) % lineLen;
        counter += numSamples;
    }

    CHECK (impulseAt >= 0);
    CHECK (foundAt >= 0);
    CHECK (peak > 0.4f);   // linear interpolation splits it across two taps at most

    // Within one sample of the geometric delay (the fractional part lands
    // between two taps, and the box smoother has settled by then).
    const int measured = foundAt - impulseAt;
    CHECK (std::abs (measured - expectedSamples) <= 1);

    // And it must NOT be at zero - the whole point is that distance costs time.
    // 30 m of air is 87.5 ms, so anything under ~4000 samples means the delay
    // was dropped somewhere.
    CHECK (measured > 4000);
}

static void testAcousticTapAirAbsorptionShelf()
{
    using namespace spatcore::dsp;
    using namespace acoustictap_test;

    // The shelf is fixed at 800 Hz / Q 0.3, so a 100 Hz tone should pass
    // essentially untouched while an 8 kHz tone takes close to the full cut.
    const double sr = 48000.0;
    const float shelfDb = -12.0f;
    const int numSamples = 4096;

    // RMS, not sample peak: at 8 kHz / 48 kHz a sine is only 6 samples per
    // period, so its sample peak sits well below 1.0 for reasons that have
    // nothing to do with the filter under test. A sine RMS is 0.7071 at every
    // frequency, which makes the ratio a clean gain measurement.
    auto measure = [&] (float freqHz, float hfDb) -> double
    {
        AcousticTapCell cell;
        cell.prepare (sr, (int) (sr * 0.010));

        const int lineLen = 1 << 14;
        std::vector<float> line ((size_t) lineLen, 0.0f);

        std::int64_t counter = 0;
        int blockStart = 0;
        double sumSq = 0.0;
        int counted = 0;

        const int blocks = 8;
        for (int block = 0; block < blocks; ++block)
        {
            std::vector<float> got ((size_t) numSamples, 0.0f);

            for (int i = 0; i < numSamples; ++i)
            {
                const double t = (double) (counter + i) / sr;
                line[(size_t) ((blockStart + i) % lineLen)] =
                    (float) std::sin (2.0 * 3.14159265358979 * (double) freqHz * t);
            }

            processAcousticTap (cell, line.data(), lineLen, blockStart, counter,
                                numSamples, 0.0f, hfDb, 1.0f, sr, got.data());

            // Ignore the early blocks: filter start-up transient.
            if (block >= blocks - 2)
            {
                for (int i = 0; i < numSamples; ++i)
                {
                    const double v = (double) got[(size_t) i];
                    sumSq += v * v;
                    ++counted;
                }
            }

            blockStart = (blockStart + numSamples) % lineLen;
            counter += numSamples;
        }

        return counted > 0 ? std::sqrt (sumSq / (double) counted) : 0.0;
    };

    const double sineRms = 0.70710678;
    const double lowRef  = measure (100.0f, 0.0f);
    const double lowCut  = measure (100.0f, shelfDb);
    const double highRef = measure (8000.0f, 0.0f);
    const double highCut = measure (8000.0f, shelfDb);

    // Bypass really is unity at both ends.
    CHECK (std::fabs (lowRef - sineRms) < 0.01);
    CHECK (std::fabs (highRef - sineRms) < 0.01);

    // Below the corner: within ~1 dB of untouched.
    CHECK (lowCut > lowRef * 0.89);

    // Well above the corner: close to the full -12 dB shelf.
    const double target = std::pow (10.0, (double) shelfDb / 20.0);   // 0.2512
    CHECK (highCut < highRef * target * 1.25);
    CHECK (highCut > highRef * target * 0.80);

    // And a zero-damping cell must be bit-exact unity, not "nearly" unity.
    {
        AcousticTapCell cell;
        cell.prepare (sr, (int) (sr * 0.010));
        std::vector<float> line ((size_t) kLineLength, 0.0f);
        std::vector<float> in ((size_t) 64), got ((size_t) 64, 0.0f);
        for (int i = 0; i < 64; ++i)
            in[(size_t) i] = src (i);
        writeBlock (line, 0, in);
        processAcousticTap (cell, line.data(), kLineLength, 0, 0, 64,
                            0.0f, 0.0f, 1.0f, sr, got.data());
        for (int i = 0; i < 64; ++i)
            CHECK (bitEqualFloat (got[(size_t) i], in[(size_t) i]));
    }
}

static void testAcousticTapAccumulatesAndClamps()
{
    using namespace spatcore::dsp;
    using namespace acoustictap_test;

    const double sr = 48000.0;
    const int numSamples = 128;

    // Accumulation, not assignment: the reverb return sums every node into the
    // same speaker buffer, so a tap must add to what is already there.
    {
        AcousticTapCell a, b;
        a.prepare (sr, 480);
        b.prepare (sr, 480);

        std::vector<float> lineA ((size_t) kLineLength, 0.0f);
        std::vector<float> lineB ((size_t) kLineLength, 0.0f);
        std::vector<float> inA ((size_t) numSamples), inB ((size_t) numSamples);
        for (int i = 0; i < numSamples; ++i)
        {
            inA[(size_t) i] = src (i);
            inB[(size_t) i] = src (i + 1000);
        }
        writeBlock (lineA, 0, inA);
        writeBlock (lineB, 0, inB);

        std::vector<float> got ((size_t) numSamples, 0.0f);
        processAcousticTap (a, lineA.data(), kLineLength, 0, 0, numSamples,
                            0.0f, 0.0f, 0.5f, sr, got.data());
        processAcousticTap (b, lineB.data(), kLineLength, 0, 0, numSamples,
                            0.0f, 0.0f, 0.25f, sr, got.data());

        for (int i = 0; i < numSamples; ++i)
            CHECK (bitEqualFloat (got[(size_t) i],
                                  inA[(size_t) i] * 0.5f + inB[(size_t) i] * 0.25f));
    }

    // A delay longer than the line, a negative delay and a NaN must all clamp
    // rather than read out of bounds. (The line is 4096 samples = 85 ms at
    // 48 kHz; 10 000 ms is far past it.)
    const float badDelays[3] = { 10000.0f, -50.0f,
                                 std::numeric_limits<float>::quiet_NaN() };
    for (float badDelayMs : badDelays)
    {
        AcousticTapCell cell;
        cell.prepare (sr, 480);
        std::vector<float> line ((size_t) kLineLength, 0.25f);
        std::vector<float> got ((size_t) numSamples, 0.0f);

        processAcousticTap (cell, line.data(), kLineLength, 0, 0, numSamples,
                            badDelayMs, 0.0f, 1.0f, sr, got.data());

        for (int i = 0; i < numSamples; ++i)
            CHECK (std::isfinite (got[(size_t) i]));
    }

    // Degenerate arguments are no-ops, not crashes.
    {
        AcousticTapCell cell;
        cell.prepare (sr, 480);
        std::vector<float> line ((size_t) kLineLength, 1.0f);
        std::vector<float> got ((size_t) numSamples, 7.0f);

        processAcousticTap (cell, nullptr, kLineLength, 0, 0, numSamples,
                            0.0f, 0.0f, 1.0f, sr, got.data());
        processAcousticTap (cell, line.data(), kLineLength, 0, 0, numSamples,
                            0.0f, 0.0f, 1.0f, sr, nullptr);
        processAcousticTap (cell, line.data(), 1, 0, 0, numSamples,
                            0.0f, 0.0f, 1.0f, sr, got.data());
        processAcousticTap (cell, line.data(), kLineLength, 0, 0, 0,
                            0.0f, 0.0f, 1.0f, sr, got.data());

        for (int i = 0; i < numSamples; ++i)
            CHECK (got[(size_t) i] == 7.0f);
    }
}

//==============================================================================
// reverb/ReverbReturnProcessor - node -> speaker distribution.
//
// The trap this pins down: the calculation engine indexes its return matrices
// with the ENGINE stride (max reverb/output channels), while the processor
// indexes its own tap cells with the LIVE output count. Those two are usually
// equal on a dev box and never equal on a real rig, so a test that uses one
// number for both proves nothing.
//==============================================================================

static void testAcousticTapFastPathMatchesGeneral()
{
    using namespace spatcore::dsp;
    using namespace acoustictap_test;

    // processAcousticTap has two loops: a run-walking one for a settled delay
    // and a per-sample general one while the delay is moving. The general loop
    // is the oracle: for identical inputs and identical cell state, a cell
    // that is allowed the fast path must produce what a cell that is not
    // would have - at every sample, across every crossover in both directions.
    //
    // The trajectory holds at FRACTIONAL delays on purpose (2.95 ms = 141.6
    // samples), because an off-by-one in the run-walker's read position is
    // invisible at a whole-sample delay and was exactly the bug this test was
    // written to catch.
    const double sr = 48000.0;
    const int numSamples = 128;
    const int lineLen = 1 << 14;

    for (float hfDb : { 0.0f, -6.0f })
    {
        AcousticTapCell fast, ref;
        fast.prepare (sr, (int) (sr * 0.010));
        ref.prepare  (sr, (int) (sr * 0.010));

        std::vector<float> line ((size_t) lineLen, 0.0f);
        std::int64_t counter = 0;
        int blockStart = 0;

        double maxDiff = 0.0;
        int steadyBlocks = 0, movingBlocks = 0;
        float peak = 0.0f;

        const int totalBlocks = 240;
        for (int block = 0; block < totalBlocks; ++block)
        {
            for (int i = 0; i < numSamples; ++i)
            {
                const double t = (double) (counter + i) / sr;
                line[(size_t) ((blockStart + i) % lineLen)] =
                    (float) (0.5 * std::sin (2.0 * 3.14159265358979 * 220.0 * t)
                           + 0.2 * std::sin (2.0 * 3.14159265358979 * 3100.0 * t));
            }

            // Move for 12 blocks, hold for 28, alternating between two
            // fractional endpoints. Long holds so the smoother actually
            // settles (it takes W + W/2 samples after the last change).
            const int cycle = block % 80;
            float delayMs;
            if (cycle < 12)       delayMs = 2.95f + 0.0125f * (float) cycle;    // 2.95 -> 3.0875
            else if (cycle < 40)  delayMs = 2.95f + 0.0125f * 11.0f;
            else if (cycle < 52)  delayMs = 3.0875f - 0.0125f * (float) (cycle - 40);
            else                  delayMs = 3.0875f - 0.0125f * 11.0f;         // 2.95

            std::vector<float> a ((size_t) numSamples, 0.0f), b ((size_t) numSamples, 0.0f);
            processAcousticTap (fast, line.data(), lineLen, blockStart, counter,
                                numSamples, delayMs, hfDb, 0.8f, sr, a.data(), true);
            processAcousticTap (ref,  line.data(), lineLen, blockStart, counter,
                                numSamples, delayMs, hfDb, 0.8f, sr, b.data(), false);

            float steady = 0.0f;
            if (fast.smoother.isSteadyFrom (counter, steady)) ++steadyBlocks; else ++movingBlocks;

            if (block >= 8)
            {
                for (int i = 0; i < numSamples; ++i)
                {
                    maxDiff = std::fmax (maxDiff, std::fabs ((double) a[(size_t) i] - (double) b[(size_t) i]));
                    peak = std::fmax (peak, std::fabs (a[(size_t) i]));
                }
            }

            blockStart = (blockStart + numSamples) % lineLen;
            counter += numSamples;
        }

        // Both paths must actually have run, or the comparison is vacuous.
        CHECK (steadyBlocks > 40);
        CHECK (movingBlocks > 20);
        CHECK (peak > 0.3f);

        // Not bit-equal by construction: the general loop recomputes
        // (blockStart + s - delay) per sample and its fractional part rounds
        // differently as s crosses a power of two, while the run-walker fixes
        // it once. That is one ulp of read position, ~1e-5 of signal. An
        // off-by-one sample is ~1e-2 here and was caught at 0.017.
        CHECK (maxDiff < 1.0e-4);
    }
}

static void testReverbReturnProcessorMatrixStride()
{
    using namespace spatcore::reverb;

    const double sr = 48000.0;
    const int numSamples = 256;
    const int nodes = 2;
    const int outs = 3;
    const int stride = 16;          // engine max-output stride, deliberately > outs

    ReverbReturnProcessor proc;
    proc.prepare (sr, numSamples, nodes, outs);
    CHECK (proc.isPrepared());
    CHECK (proc.getPreparedNodes() == nodes);
    CHECK (proc.getPreparedOutputs() == outs);

    // Node 0 -> output 0 only; node 1 -> output 2 only. Output 1 gets nothing.
    std::vector<float> levels ((size_t) (nodes * stride), 0.0f);
    std::vector<float> delays ((size_t) (nodes * stride), 0.0f);
    std::vector<float> hf     ((size_t) (nodes * stride), 0.0f);
    levels[(size_t) (0 * stride + 0)] = 0.5f;
    levels[(size_t) (1 * stride + 2)] = 0.25f;

    juce::AudioBuffer<float> out (outs, numSamples);
    out.clear();

    std::vector<float> wet0 ((size_t) numSamples), wet1 ((size_t) numSamples);
    for (int i = 0; i < numSamples; ++i)
    {
        wet0[(size_t) i] = 0.31f * std::sin (0.05f * (float) i);
        wet1[(size_t) i] = 0.17f * std::sin (0.11f * (float) i + 1.1f);
    }

    proc.pushNodeReturn (0, wet0.data(), numSamples);
    proc.pushNodeReturn (1, wet1.data(), numSamples);
    proc.mixToOutputs (out, 0, numSamples, nodes, outs,
                       levels.data(), delays.data(), hf.data(), stride);

    for (int i = 0; i < numSamples; ++i)
    {
        CHECK (bitEqualFloat (out.getSample (0, i), wet0[(size_t) i] * 0.5f));
        CHECK (out.getSample (1, i) == 0.0f);
        CHECK (bitEqualFloat (out.getSample (2, i), wet1[(size_t) i] * 0.25f));
    }
}

static void testReverbReturnProcessorAccumulatesOntoDirect()
{
    using namespace spatcore::reverb;

    // The return mixes into a speaker buffer the WFS renderer has ALREADY
    // written. Overwriting instead of accumulating would delete the dry sound,
    // which is the loudest possible way to get this wrong.
    const double sr = 48000.0;
    const int numSamples = 128;
    const int nodes = 2;
    const int outs = 2;
    const int stride = outs;

    ReverbReturnProcessor proc;
    proc.prepare (sr, numSamples, nodes, outs);

    std::vector<float> levels ((size_t) (nodes * stride), 0.0f);
    levels[(size_t) (0 * stride + 0)] = 1.0f;
    levels[(size_t) (1 * stride + 0)] = 1.0f;   // both nodes into output 0

    juce::AudioBuffer<float> out (outs, numSamples);
    std::vector<float> direct ((size_t) numSamples);
    for (int i = 0; i < numSamples; ++i)
    {
        direct[(size_t) i] = 0.4f * std::sin (0.02f * (float) i);
        out.setSample (0, i, direct[(size_t) i]);
        out.setSample (1, i, 0.0f);
    }

    std::vector<float> wet ((size_t) numSamples, 0.125f);
    proc.pushNodeReturn (0, wet.data(), numSamples);
    proc.pushNodeReturn (1, wet.data(), numSamples);
    proc.mixToOutputs (out, 0, numSamples, nodes, outs,
                       levels.data(), nullptr, nullptr, stride);

    // Same association order the two accumulating taps use - (a + b) + c is not
    // bit-identical to a + (b + c) in float, and the point here is accumulation
    // semantics, not a claim about summation order.
    for (int i = 0; i < numSamples; ++i)
        CHECK (bitEqualFloat (out.getSample (0, i),
                              (direct[(size_t) i] + 0.125f) + 0.125f));
}

static void testReverbReturnProcessorPerOutputDelay()
{
    using namespace spatcore::reverb;

    // Two speakers at different distances from one node: the far one must get
    // the wet signal LATER, and by the amount the matrix asked for.
    const double sr = 48000.0;
    const int numSamples = 256;
    const int nodes = 1;
    const int outs = 2;
    const int stride = outs;
    const float farDelayMs = 20.0f;
    const int farDelaySamples = (int) ((farDelayMs / 1000.0f) * (float) sr);   // 960

    ReverbReturnProcessor proc;
    proc.prepare (sr, numSamples, nodes, outs);

    std::vector<float> levels ((size_t) (nodes * stride), 1.0f);
    std::vector<float> delays ((size_t) (nodes * stride), 0.0f);
    delays[(size_t) (0 * stride + 1)] = farDelayMs;    // output 1 is the far one

    int nearAt = -1, farAt = -1;
    float nearPeak = 0.0f, farPeak = 0.0f;
    int impulseAt = -1;

    const int settleBlocks = 4;
    const int totalBlocks = 24;

    for (int block = 0; block < totalBlocks; ++block)
    {
        std::vector<float> wet ((size_t) numSamples, 0.0f);
        if (block == settleBlocks)
        {
            wet[0] = 1.0f;
            impulseAt = block * numSamples;
        }

        juce::AudioBuffer<float> out (outs, numSamples);
        out.clear();

        proc.pushNodeReturn (0, wet.data(), numSamples);
        proc.mixToOutputs (out, 0, numSamples, nodes, outs,
                           levels.data(), delays.data(), nullptr, stride);

        for (int i = 0; i < numSamples; ++i)
        {
            const float a = std::fabs (out.getSample (0, i));
            if (a > nearPeak) { nearPeak = a; nearAt = block * numSamples + i; }

            const float b = std::fabs (out.getSample (1, i));
            if (b > farPeak) { farPeak = b; farAt = block * numSamples + i; }
        }
    }

    CHECK (impulseAt >= 0);
    CHECK (nearPeak > 0.9f);
    CHECK (farPeak > 0.4f);
    CHECK (nearAt == impulseAt);                                  // zero delay: same sample
    CHECK (std::abs ((farAt - impulseAt) - farDelaySamples) <= 1);
    CHECK (farAt > nearAt);
}

static void testReverbReturnProcessorSkipBlockAdvances()
{
    using namespace spatcore::reverb;

    // While post-muted the callback calls skipBlock instead of push+mix. If it
    // did not advance the write head, the next real block would land on top of
    // the last one and the delay line would stop being a timeline.
    const double sr = 48000.0;
    const int numSamples = 64;
    const int nodes = 1;
    const int outs = 1;
    const int stride = 1;
    const float delayMs = (float) (numSamples * 1000.0 / sr);   // exactly one block

    ReverbReturnProcessor proc;
    proc.prepare (sr, numSamples, nodes, outs);

    std::vector<float> levels ((size_t) 1, 1.0f);
    std::vector<float> delays ((size_t) 1, delayMs);

    // Block 0: real audio. Block 1: muted (skipBlock). Block 2: silence pushed.
    // With a one-block delay, block 1's OUTPUT would be block 0's audio - but
    // block 1 is muted so nothing is mixed. Block 2 taps block 1, which
    // skipBlock filled with silence. If skipBlock had not advanced the head,
    // block 2 would tap block 0 and the audio would reappear after the mute.
    std::vector<float> tone ((size_t) numSamples);
    for (int i = 0; i < numSamples; ++i)
        tone[(size_t) i] = 0.5f + 0.1f * (float) i;

    juce::AudioBuffer<float> out (outs, numSamples);

    out.clear();
    proc.pushNodeReturn (0, tone.data(), numSamples);
    proc.mixToOutputs (out, 0, numSamples, nodes, outs,
                       levels.data(), delays.data(), nullptr, stride);

    proc.skipBlock (numSamples);                       // muted block

    out.clear();
    std::vector<float> silence ((size_t) numSamples, 0.0f);
    proc.pushNodeReturn (0, silence.data(), numSamples);
    proc.mixToOutputs (out, 0, numSamples, nodes, outs,
                       levels.data(), delays.data(), nullptr, stride);

    // Tapping one block back reaches the skipped (silent) block, not the tone.
    for (int i = 0; i < numSamples; ++i)
        CHECK (std::fabs (out.getSample (0, i)) < 1.0e-6f);
}

//==============================================================================
// reverb/ReverbSendMatrix - source -> node send. The computation ReverbFeedThread
// runs per batch, minus the thread, so the stride and delay semantics can be
// pinned the same way the return side's are.
//==============================================================================

static void testReverbSendMatrixStride()
{
    using namespace spatcore::reverb;

    const double sr = 48000.0;
    const int numSamples = 256;
    const int sources = 3;
    const int nodes = 2;
    const int stride = 16;          // engine max-reverb stride, deliberately > nodes

    ReverbSendMatrix m;
    m.prepare (sr, sources, nodes);
    CHECK (m.isPrepared());
    CHECK (m.getPreparedSources() == sources);
    CHECK (m.getPreparedNodes() == nodes);

    // Source 0 -> node 0 at 0.5; source 2 -> node 1 at 0.25; source 1 feeds nothing.
    std::vector<float> levels ((size_t) (sources * stride), 0.0f);
    levels[(size_t) (0 * stride + 0)] = 0.5f;
    levels[(size_t) (2 * stride + 1)] = 0.25f;

    juce::AudioBuffer<float> in (sources, numSamples);
    for (int i = 0; i < numSamples; ++i)
    {
        in.setSample (0, i, 0.31f * std::sin (0.05f * (float) i));
        in.setSample (1, i, 0.90f);                                   // must NOT leak
        in.setSample (2, i, 0.17f * std::sin (0.11f * (float) i + 1.1f));
    }

    std::vector<float> feed0 ((size_t) numSamples, -1.0f), feed1 ((size_t) numSamples, -1.0f);

    m.writeInputs (in, numSamples);
    m.computeNodeFeed (feed0.data(), numSamples, 0, levels.data(), nullptr, nullptr, stride);
    m.computeNodeFeed (feed1.data(), numSamples, 1, levels.data(), nullptr, nullptr, stride);
    m.advance (numSamples);

    for (int i = 0; i < numSamples; ++i)
    {
        CHECK (bitEqualFloat (feed0[(size_t) i], in.getSample (0, i) * 0.5f));
        CHECK (bitEqualFloat (feed1[(size_t) i], in.getSample (2, i) * 0.25f));
    }
}

static void testReverbSendMatrixPerNodeDelay()
{
    using namespace spatcore::reverb;

    // One source, two nodes at different distances: the far node must receive
    // the same impulse LATER, by the amount the matrix asked for.
    const double sr = 48000.0;
    const int numSamples = 256;
    const int sources = 1;
    const int nodes = 2;
    const int stride = 32;
    const float farDelayMs = 25.0f;
    const int farDelaySamples = (int) ((farDelayMs / 1000.0f) * (float) sr);   // 1200

    ReverbSendMatrix m;
    m.prepare (sr, sources, nodes);

    std::vector<float> levels ((size_t) (sources * stride), 0.0f);
    std::vector<float> delays ((size_t) (sources * stride), 0.0f);
    levels[(size_t) (0 * stride + 0)] = 1.0f;
    levels[(size_t) (0 * stride + 1)] = 1.0f;
    delays[(size_t) (0 * stride + 1)] = farDelayMs;

    int nearAt = -1, farAt = -1, impulseAt = -1;
    float nearPeak = 0.0f, farPeak = 0.0f;

    const int settleBlocks = 4;
    const int totalBlocks = 24;

    for (int block = 0; block < totalBlocks; ++block)
    {
        juce::AudioBuffer<float> in (sources, numSamples);
        in.clear();
        if (block == settleBlocks)
        {
            in.setSample (0, 0, 1.0f);
            impulseAt = block * numSamples;
        }

        std::vector<float> feed0 ((size_t) numSamples, 0.0f), feed1 ((size_t) numSamples, 0.0f);

        m.writeInputs (in, numSamples);
        m.computeNodeFeed (feed0.data(), numSamples, 0, levels.data(), delays.data(), nullptr, stride);
        m.computeNodeFeed (feed1.data(), numSamples, 1, levels.data(), delays.data(), nullptr, stride);
        m.advance (numSamples);

        for (int i = 0; i < numSamples; ++i)
        {
            const float a = std::fabs (feed0[(size_t) i]);
            if (a > nearPeak) { nearPeak = a; nearAt = block * numSamples + i; }
            const float b = std::fabs (feed1[(size_t) i]);
            if (b > farPeak) { farPeak = b; farAt = block * numSamples + i; }
        }
    }

    CHECK (impulseAt >= 0);
    CHECK (nearPeak > 0.9f);
    CHECK (farPeak > 0.4f);
    CHECK (nearAt == impulseAt);
    CHECK (std::abs ((farAt - impulseAt) - farDelaySamples) <= 1);
    CHECK (farAt > nearAt);
}

static void testReverbSendMatrixSilentNodeIsSilent()
{
    using namespace spatcore::reverb;

    // computeNodeFeed OVERWRITES its destination. A node that no source feeds
    // must come out as silence, not as whatever the buffer held before - the
    // feed thread reuses the same row buffer batch after batch.
    const double sr = 48000.0;
    const int numSamples = 64;

    ReverbSendMatrix m;
    m.prepare (sr, 2, 2);

    std::vector<float> levels ((size_t) (2 * 2), 0.0f);
    levels[(size_t) (0 * 2 + 0)] = 1.0f;                 // only node 0 is fed

    juce::AudioBuffer<float> in (2, numSamples);
    for (int i = 0; i < numSamples; ++i)
    {
        in.setSample (0, i, 0.5f);
        in.setSample (1, i, 0.5f);
    }

    std::vector<float> stale ((size_t) numSamples, 123.0f);

    m.writeInputs (in, numSamples);
    m.computeNodeFeed (stale.data(), numSamples, 1, levels.data(), nullptr, nullptr, 2);
    m.advance (numSamples);

    for (int i = 0; i < numSamples; ++i)
        CHECK (stale[(size_t) i] == 0.0f);

    // Out-of-range node and a null matrix are silence too, never a crash.
    std::fill (stale.begin(), stale.end(), 123.0f);
    m.computeNodeFeed (stale.data(), numSamples, 7, levels.data(), nullptr, nullptr, 2);
    for (int i = 0; i < numSamples; ++i)
        CHECK (stale[(size_t) i] == 0.0f);

    std::fill (stale.begin(), stale.end(), 123.0f);
    m.computeNodeFeed (stale.data(), numSamples, 0, nullptr, nullptr, nullptr, 2);
    for (int i = 0; i < numSamples; ++i)
        CHECK (stale[(size_t) i] == 0.0f);
}

//==============================================================================
// dsp/AcousticSendMatrix - the send matrix promoted out of reverb/ so the
// effects channels can reuse it. reverb/ReverbSendMatrix.h is an alias header:
// the type is the same, so the three tests above ARE its tests; this one pins
// the identity and drives it under its new name once.
//==============================================================================

static void testAcousticSendMatrixAlias()
{
    static_assert (std::is_same_v<spatcore::reverb::ReverbSendMatrix,
                                  spatcore::dsp::AcousticSendMatrix>,
                   "reverb/ReverbSendMatrix.h must alias the promoted type, never wrap it");

    const double sr = 48000.0;
    const int numSamples = 64;
    const int stride = 4;

    spatcore::dsp::AcousticSendMatrix m;
    m.prepare (sr, 2, 1);
    CHECK (m.isPrepared());

    std::vector<float> levels ((size_t) (2 * stride), 0.0f);
    levels[(size_t) (0 * stride + 0)] = 0.5f;        // source 0 -> node 0; source 1 silent

    juce::AudioBuffer<float> in (2, numSamples);
    for (int i = 0; i < numSamples; ++i)
    {
        in.setSample (0, i, 0.37f * std::sin (0.07f * (float) i));
        in.setSample (1, i, 0.9f);
    }

    std::vector<float> feed ((size_t) numSamples, -1.0f);
    m.writeInputs (in, numSamples);
    m.computeNodeFeed (feed.data(), numSamples, 0, levels.data(), nullptr, nullptr, stride);
    m.advance (numSamples);

    for (int i = 0; i < numSamples; ++i)
        CHECK (bitEqualFloat (feed[(size_t) i], in.getSample (0, i) * 0.5f));
}

//==============================================================================
// effects/modules - the three shipped in this release.
//==============================================================================

namespace module_test
{
    using namespace spatcore::effects;

    inline ChainConfig config (double sr = 48000.0, int maxBlock = 512, std::uint32_t key = 1)
    {
        ChainConfig cfg;
        cfg.sampleRate = sr;
        cfg.maxBlock = maxBlock;
        cfg.noiseKey = key;
        return cfg;
    }

    /** Runs a module directly (no slot, no fade) over one buffer. */
    inline void render (IEffectModule& m, std::vector<float>& buf)
    {
        m.process (buf.data(), (int) buf.size());
    }

    inline std::vector<float> dc (int n, float value)
    {
        return std::vector<float> ((size_t) n, value);
    }

    inline std::vector<float> awkwardBlock (int n, int seed)
    {
        std::vector<float> v ((size_t) n);
        for (int i = 0; i < n; ++i)
            v[(size_t) i] = 0.7f * FrDiffusion::hashNoiseBipolar ((std::uint32_t) (i + seed * 1000), 31u);
        return v;
    }
}

static void testTremoloLaw()
{
    using namespace spatcore::effects;
    namespace fd = spatcore::dsp::FastDecibels;

    const int n = 12001;

    // Sine leg. 4 Hz at 48 kHz: phase 0 at sample 0, a quarter at 3000, a half
    // at 6000, back to the start at 12000.
    {
        TremoloModule m;
        m.prepare (module_test::config());
        EffectChannelParams p;
        p.trem.bypass = 0;
        p.trem.rateHz = 4.0f;
        p.trem.depthDb = 12.0f;
        p.trem.shape = 0.0f;
        p.trem.mix = 100.0f;
        CHECK (! m.applyParams (p, 0).bypass);

        std::vector<float> buf = module_test::dc (n, 0.5f);
        module_test::render (m, buf);

        CHECK (buf[0] == 0.5f);                                        // unity at phase 0, exactly
        CHECK (std::fabs (buf[6000] / 0.5f - fd::dbToGain (-12.0f)) < 1.0e-4f);
        CHECK (std::fabs (buf[3000] / 0.5f - fd::dbToGain (-6.0f)) < 1.0e-4f);
        CHECK (std::fabs (buf[12000] - buf[0]) < 1.0e-4f);             // one full cycle
    }

    // Triangle leg, at the phases that tell the two shapes apart: a triangle is
    // half way down at an eighth of a cycle, a sine is not. If the legs were a
    // quarter cycle out of step (the bug this pins), the -6 dB point below would
    // not land at 3000 at all.
    {
        TremoloModule m;
        m.prepare (module_test::config());
        EffectChannelParams p;
        p.trem.bypass = 0;
        p.trem.rateHz = 4.0f;
        p.trem.depthDb = 12.0f;
        p.trem.shape = 1.0f;
        p.trem.mix = 100.0f;
        m.applyParams (p, 0);

        std::vector<float> buf = module_test::dc (n, 0.5f);
        module_test::render (m, buf);

        CHECK (buf[0] == 0.5f);
        CHECK (std::fabs (buf[3000] / 0.5f - fd::dbToGain (-6.0f)) < 1.0e-4f);
        CHECK (std::fabs (buf[6000] / 0.5f - fd::dbToGain (-12.0f)) < 1.0e-4f);
        CHECK (std::fabs (buf[1500] / 0.5f - fd::dbToGain (-3.0f)) < 1.0e-4f);
    }

    // Mix scales the modulation against the dry signal.
    {
        TremoloModule m;
        m.prepare (module_test::config());
        EffectChannelParams p;
        p.trem.bypass = 0;
        p.trem.rateHz = 4.0f;
        p.trem.depthDb = 12.0f;
        p.trem.shape = 0.0f;
        p.trem.mix = 50.0f;
        m.applyParams (p, 0);

        std::vector<float> buf = module_test::dc (n, 0.5f);
        module_test::render (m, buf);

        const float expected = 0.5f * (0.5f + 0.5f * fd::dbToGain (-12.0f));
        CHECK (std::fabs (buf[6000] - expected) < 1.0e-4f);
    }
}

static void testBitcrusherQuantiser()
{
    using namespace spatcore::effects;

    // 8 bits is a step of exactly 1/256 - not almost, because 2^bits comes from
    // an exact power of two.
    {
        BitcrusherModule m;
        m.prepare (module_test::config());
        EffectChannelParams p;
        p.crush.bypass = 0;
        p.crush.bits = 8.0f;
        p.crush.rateHz = 48000.0f;            // no decimation
        p.crush.mix = 100.0f;
        p.crush.ditherDb = -96.0f;            // dither off
        m.applyParams (p, 0);

        std::vector<float> buf (1001);
        for (int i = 0; i < 1001; ++i)
            buf[(size_t) i] = (float) i / 1000.0f - 0.5f;
        const std::vector<float> input = buf;

        module_test::render (m, buf);

        bool exact = true, integral = true;
        for (int i = 0; i < 1001; ++i)
        {
            const float expected = std::round (input[(size_t) i] * 256.0f) / 256.0f;
            if (! bitEqualFloat (buf[(size_t) i], expected))
                exact = false;
            const float scaled = buf[(size_t) i] * 256.0f;
            if (scaled != std::round (scaled))
                integral = false;
        }
        CHECK (exact);
        CHECK (integral);
    }

    // One bit leaves three levels.
    {
        BitcrusherModule m;
        m.prepare (module_test::config());
        EffectChannelParams p;
        p.crush.bypass = 0;
        p.crush.bits = 1.0f;
        p.crush.rateHz = 48000.0f;
        p.crush.mix = 100.0f;
        m.applyParams (p, 0);

        std::vector<float> buf (1001);
        for (int i = 0; i < 1001; ++i)
            buf[(size_t) i] = (float) i / 1000.0f - 0.5f;
        module_test::render (m, buf);

        bool onlyThree = true;
        for (int i = 0; i < 1001; ++i)
        {
            const float v = buf[(size_t) i];
            if (v != -0.5f && v != 0.0f && v != 0.5f)
                onlyThree = false;
        }
        CHECK (onlyThree);
    }

    // Dither is keyed noise: reproducible for a given key, different for
    // another, and actually doing something.
    {
        EffectChannelParams p;
        p.crush.bypass = 0;
        p.crush.bits = 8.0f;
        p.crush.rateHz = 48000.0f;
        p.crush.mix = 100.0f;
        p.crush.ditherDb = 0.0f;

        std::vector<float> a = module_test::awkwardBlock (512, 1);
        std::vector<float> b = a, c = a, plain = a;

        BitcrusherModule ma, mb, mc, mplain;
        ma.prepare (module_test::config (48000.0, 512, 5));
        mb.prepare (module_test::config (48000.0, 512, 5));
        mc.prepare (module_test::config (48000.0, 512, 6));
        mplain.prepare (module_test::config (48000.0, 512, 5));

        ma.applyParams (p, 0);
        mb.applyParams (p, 0);
        mc.applyParams (p, 0);

        EffectChannelParams noDither = p;
        noDither.crush.ditherDb = -96.0f;
        mplain.applyParams (noDither, 0);

        module_test::render (ma, a);
        module_test::render (mb, b);
        module_test::render (mc, c);
        module_test::render (mplain, plain);

        CHECK (eqtests::bitEqualBlock (a, b));               // same key, same stream

        int differsFromOtherKey = 0, differsFromUndithered = 0;
        for (int i = 0; i < 512; ++i)
        {
            if (! bitEqualFloat (a[(size_t) i], c[(size_t) i]))
                ++differsFromOtherKey;
            if (! bitEqualFloat (a[(size_t) i], plain[(size_t) i]))
                ++differsFromUndithered;
        }
        CHECK (differsFromOtherKey > 100);
        CHECK (differsFromUndithered > 100);
    }
}

static void testBitcrusherHoldRate()
{
    using namespace spatcore::effects;

    // 12 kHz holds at 96 kHz: runs of exactly 8 samples, the first starting at
    // sample 0.
    {
        BitcrusherModule m;
        m.prepare (module_test::config (96000.0));
        EffectChannelParams p;
        p.crush.bypass = 0;
        p.crush.bits = 24.0f;
        p.crush.rateHz = 12000.0f;
        p.crush.mix = 100.0f;
        m.applyParams (p, 0);

        std::vector<float> buf (4096);
        for (int i = 0; i < 4096; ++i)
            buf[(size_t) i] = std::sin (6.2831853f * 1000.0f * (float) i / 96000.0f);
        const std::vector<float> input = buf;

        module_test::render (m, buf);

        const float step = 16777216.0f;             // 2^24
        bool runsHold = true, runsDiffer = true, firstSampleHeld = true;

        for (int k = 0; k + 8 <= 4096; k += 8)
        {
            for (int j = 1; j < 8; ++j)
                if (! bitEqualFloat (buf[(size_t) (k + j)], buf[(size_t) k]))
                    runsHold = false;

            const float expected = std::round (input[(size_t) k] * step) / step;
            if (! bitEqualFloat (buf[(size_t) k], expected))
                firstSampleHeld = false;

            if (k + 8 < 4096 && bitEqualFloat (buf[(size_t) (k + 8)], buf[(size_t) k]))
                runsDiffer = false;                 // a 1 kHz tone always moves
        }

        CHECK (runsHold);
        CHECK (firstSampleHeld);
        CHECK (runsDiffer);
    }

    // At the device rate there is no decimation at all: every sample is its own.
    {
        BitcrusherModule m;
        m.prepare (module_test::config (96000.0));
        EffectChannelParams p;
        p.crush.bypass = 0;
        p.crush.bits = 24.0f;
        p.crush.rateHz = 96000.0f;
        p.crush.mix = 100.0f;
        m.applyParams (p, 0);

        std::vector<float> buf (512);
        for (int i = 0; i < 512; ++i)
            buf[(size_t) i] = std::sin (6.2831853f * 1000.0f * (float) i / 96000.0f);
        const std::vector<float> input = buf;

        module_test::render (m, buf);

        const float step = 16777216.0f;
        bool perSample = true;
        for (int i = 0; i < 512; ++i)
            if (! bitEqualFloat (buf[(size_t) i], std::round (input[(size_t) i] * step) / step))
                perSample = false;
        CHECK (perSample);
    }
}

static void testEffectEQModuleMatchesBank()
{
    using namespace spatcore::effects;
    using spatcore::dsp::MultiChannelEQBank;

    EffectChannelParams p;
    p.eq[0].bypass = 0;

    EffectEQModule m;
    m.prepare (module_test::config());
    m.applyParams (p, 0);

    MultiChannelEQBank<6> bank;
    bank.prepare (48000.0, 1);
    bank.setChannelEnabled (0, true);
    for (int b = 0; b < 6; ++b)
        bank.pushBandParameters (0, b, p.eq[0].shape[b], p.eq[0].freqHz[b],
                                 p.eq[0].gainDb[b], p.eq[0].q[b], p.eq[0].slope[b]);

    // Bit-identical, block after block - including a first block full of
    // negative zeros and denormals.
    for (int block = 0; block < 10; ++block)
    {
        std::vector<float> a = (block == 0) ? eqtests::makeAwkwardSignal (256)
                                            : module_test::awkwardBlock (256, block);
        std::vector<float> b = a;

        m.process (a.data(), 256);
        bank.processChannel (0, b.data(), 256);
        CHECK (eqtests::bitEqualBlock (a, b));
    }

    // A parameter change lands on both the same way.
    p.eq[0].gainDb[2] = 6.0f;
    p.eq[0].shape[0] = 3;
    p.eq[0].freqHz[2] = 900.0f;
    m.applyParams (p, 0);
    bank.setChannelEnabled (0, true);
    for (int b = 0; b < 6; ++b)
        bank.pushBandParameters (0, b, p.eq[0].shape[b], p.eq[0].freqHz[b],
                                 p.eq[0].gainDb[b], p.eq[0].q[b], p.eq[0].slope[b]);

    for (int block = 0; block < 4; ++block)
    {
        std::vector<float> a = module_test::awkwardBlock (256, 20 + block);
        std::vector<float> b = a;
        m.process (a.data(), 256);
        bank.processChannel (0, b.data(), 256);
        CHECK (eqtests::bitEqualBlock (a, b));
    }

    // Every band off is the identity, to the bit.
    {
        EffectChannelParams flat;
        flat.eq[1].bypass = 0;
        for (int b = 0; b < 6; ++b)
            flat.eq[1].shape[b] = 0;

        EffectEQModule off;
        off.prepare (module_test::config());
        off.applyParams (flat, 1);                  // the SECOND instance

        std::vector<float> a = eqtests::makeAwkwardSignal (256);
        const std::vector<float> reference = a;
        off.process (a.data(), 256);
        CHECK (eqtests::bitEqualBlock (a, reference));
    }
}

static void testResetOnFullBypass()
{
    using namespace spatcore::effects;

    // A resonant peak rings for a long time. Bypassing the slot must clear that
    // tail, so switching back on cannot replay a moment from before.
    ChainConfig cfg = module_test::config (48000.0, 256);

    ModuleSlot slot;
    slot.prepare (cfg, std::make_unique<EffectEQModule>());

    EffectChannelParams p;
    p.eq[0].bypass = 0;
    p.eq[0].shape[2] = 3;                           // peak
    p.eq[0].freqHz[2] = 1000.0f;
    p.eq[0].gainDb[2] = 24.0f;
    p.eq[0].q[2] = 20.0f;
    slot.applyParams (p, 0);

    std::vector<float> buf (256, 0.0f);
    buf[0] = 1.0f;                                  // one impulse, then silence
    slot.process (buf.data(), 256);

    for (int b = 0; b < 40; ++b)                    // let the fade settle in
    {
        std::fill (buf.begin(), buf.end(), 0.0f);
        slot.process (buf.data(), 256);
    }
    CHECK (slot.isActiveSettled());

    // Control: the tail is still ringing at this point.
    {
        std::fill (buf.begin(), buf.end(), 0.0f);
        slot.process (buf.data(), 256);
        float peak = 0.0f;
        for (int i = 0; i < 256; ++i)
            peak = std::fabs (buf[(size_t) i]) > peak ? std::fabs (buf[(size_t) i]) : peak;
        CHECK (peak > 1.0e-6f);
    }

    p.eq[0].bypass = 1;
    slot.applyParams (p, 0);
    for (int b = 0; b < 40; ++b)
    {
        std::fill (buf.begin(), buf.end(), 0.0f);
        slot.process (buf.data(), 256);
    }
    CHECK (slot.isBypassedSettled());
    CHECK (slot.silentResets.load() == 1);

    p.eq[0].bypass = 0;
    slot.applyParams (p, 0);

    bool silent = true;
    for (int b = 0; b < 20; ++b)
    {
        std::fill (buf.begin(), buf.end(), 0.0f);
        slot.process (buf.data(), 256);
        for (int i = 0; i < 256; ++i)
            if (buf[(size_t) i] != 0.0f)
                silent = false;
    }
    CHECK (silent);
}

static void testEffectModulesNeutralAtDefaults()
{
    using namespace spatcore::effects;

    ChainConfig cfg = module_test::config (48000.0, 256);
    const EffectChannelParams defaults;              // every module bypassed

    std::unique_ptr<IEffectModule> modules[3];
    modules[0] = std::make_unique<TremoloModule>();
    modules[1] = std::make_unique<BitcrusherModule>();
    modules[2] = std::make_unique<EffectEQModule>();

    for (auto& module : modules)
    {
        ModuleSlot slot;
        slot.prepare (cfg, std::move (module));
        slot.applyParams (defaults, 0);
        CHECK (slot.isBypassedSettled());

        for (int block = 0; block < 8; ++block)
        {
            std::vector<float> buf = (block == 0) ? eqtests::makeAwkwardSignal (256)
                                                  : module_test::awkwardBlock (256, block);
            const std::vector<float> reference = buf;
            slot.process (buf.data(), 256);
            CHECK (eqtests::bitEqualBlock (buf, reference));
        }

        CHECK (slot.nanTrips.load() == 0);
        CHECK (slot.getLatencySamples() == 0);
    }
}

static void testEffectModulesIdentityWhenActive()
{
    using namespace spatcore::effects;

    // Settings at which an ACTIVE module must still be transparent. These are
    // the ones an operator reaches by turning one control to its end, so a
    // rounding error here is audible as a change that should not be there.
    ChainConfig cfg = module_test::config (48000.0, 256);

    // Tremolo: mix 0, and depth 0 at full mix - both bit-exact, the second
    // because 0 dB converts to exactly 1.
    for (int variant = 0; variant < 2; ++variant)
    {
        TremoloModule m;
        m.prepare (cfg);
        EffectChannelParams p;
        p.trem.bypass = 0;
        p.trem.depthDb = (variant == 0) ? 12.0f : 0.0f;
        p.trem.mix = (variant == 0) ? 0.0f : 100.0f;
        m.applyParams (p, 0);

        std::vector<float> buf = eqtests::makeAwkwardSignal (256);
        const std::vector<float> reference = buf;
        m.process (buf.data(), 256);
        CHECK (eqtests::bitEqualBlock (buf, reference));
    }

    // Crusher: mix 0 is exact; 24 bits at the device rate is a quantiser step
    // smaller than the samples themselves, so it is transparent to well inside
    // a float's precision.
    {
        BitcrusherModule m;
        m.prepare (cfg);
        EffectChannelParams p;
        p.crush.bypass = 0;
        p.crush.mix = 0.0f;
        m.applyParams (p, 0);

        std::vector<float> buf = eqtests::makeAwkwardSignal (256);
        const std::vector<float> reference = buf;
        m.process (buf.data(), 256);
        CHECK (eqtests::bitEqualBlock (buf, reference));
    }
    {
        BitcrusherModule m;
        m.prepare (cfg);
        EffectChannelParams p;
        p.crush.bypass = 0;
        p.crush.bits = 24.0f;
        p.crush.rateHz = 48000.0f;
        p.crush.mix = 100.0f;
        m.applyParams (p, 0);

        std::vector<float> buf = module_test::awkwardBlock (256, 3);
        const std::vector<float> reference = buf;
        m.process (buf.data(), 256);

        float worst = 0.0f;
        for (int i = 0; i < 256; ++i)
            worst = std::fabs (buf[(size_t) i] - reference[(size_t) i]) > worst
                      ? std::fabs (buf[(size_t) i] - reference[(size_t) i]) : worst;
        CHECK (worst <= 1.0e-6f);
    }
}

//==============================================================================
// effects/EffectChain
//==============================================================================

namespace chain_test
{
    using namespace spatcore::effects;

    inline int slotIndexFor (ModuleId type, int instance) noexcept
    {
        for (int i = 0; i < kNumModuleSlots; ++i)
            if (kSlots[i].type == type && (int) kSlots[i].instance == instance)
                return i;
        return 0;
    }

    inline bool bypassFor (const EffectChannelParams& p, ModuleId type, int instance) noexcept
    {
        switch (type)
        {
            case ModuleId::Dist:   return p.dist.bypass != 0;
            case ModuleId::EQ:     return p.eq[instance & 1].bypass != 0;
            case ModuleId::Dyn:    return p.dyn[instance & 1].bypass != 0;
            case ModuleId::Mod:    return p.mod.bypass != 0;
            case ModuleId::Phaser: return p.phaser.bypass != 0;
            case ModuleId::Trem:   return p.trem.bypass != 0;
            case ModuleId::Reverb: return p.reverb.bypass != 0;
            case ModuleId::Delay:  return p.delay.bypass != 0;
            case ModuleId::Crush:  return p.crush.bypass != 0;
            case ModuleId::Count:
            default:               return true;
        }
    }

    // The factory is a plain function pointer, so the per-slot settings the
    // tests want live here rather than in a capture.
    inline float slotGain[kNumModuleSlots] =
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    inline int slotLatency[kNumModuleSlots] =
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    inline int slotResets[kNumModuleSlots] =
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    inline void resetSlotSettings() noexcept
    {
        for (int i = 0; i < kNumModuleSlots; ++i)
        {
            slotGain[i] = 1.0f;
            slotLatency[i] = 0;
            slotResets[i] = 0;
        }
    }

    class TypedModule : public IEffectModule
    {
    public:
        TypedModule (ModuleId t, int inst) : assigned (t), slot (slotIndexFor (t, inst)) {}

        ModuleId type() const noexcept override { return assigned; }
        void prepare (const ChainConfig&) override {}
        void reset() noexcept override { ++slotResets[slot]; }

        ParamApplyInfo applyParams (const EffectChannelParams& p, int instance) noexcept override
        {
            return { bypassFor (p, assigned, instance), false };
        }

        void process (float* inout, int n) noexcept override
        {
            const float g = slotGain[slot];
            for (int i = 0; i < n; ++i)
                inout[i] *= g;
        }

        int getLatencySamples() const noexcept override { return slotLatency[slot]; }

    private:
        ModuleId assigned;
        int slot;
    };

    inline std::unique_ptr<IEffectModule> typedFactory (ModuleId type, int instance, const ChainConfig&)
    {
        return std::make_unique<TypedModule> (type, instance);
    }

    /** Bumps the revision so the chain actually re-reads the parameters. */
    inline void publish (EffectChannelParams& p) noexcept { ++p.revision; }

    inline ChainConfig config (int maxBlock = 64)
    {
        ChainConfig cfg;
        cfg.sampleRate = 48000.0;
        cfg.maxBlock = maxBlock;
        cfg.noiseKey = 7;
        return cfg;
    }
}

static void testChainReorderDeterminism()
{
    using namespace spatcore::effects;

    // Two chains of REAL modules, one kept on the default order throughout and
    // one taken to a different order and back. Tremolo before crusher is not
    // the same sound as crusher before tremolo, so the middle section must
    // differ - and once the order is restored and the envelope has settled, the
    // two must agree bit for bit again. That is the property that says a
    // reorder moves audio and not module state.
    const int blockSize = 64;

    EffectChannelParams p;
    p.trem.bypass = 0;
    p.trem.rateHz = 3.0f;
    p.trem.depthDb = 9.0f;
    p.trem.mix = 100.0f;
    p.crush.bypass = 0;
    p.crush.bits = 4.0f;
    p.crush.rateHz = 4800.0f;
    p.crush.mix = 100.0f;

    ChainOrder swapped = kDefaultOrder;
    const int tremSlot = chain_test::slotIndexFor (ModuleId::Trem, 0);
    const int crushSlot = chain_test::slotIndexFor (ModuleId::Crush, 0);
    std::swap (swapped[(size_t) tremSlot], swapped[(size_t) crushSlot]);
    CHECK (isValidChainOrder (swapped));

    EffectChain reference, tested;
    reference.prepare (chain_test::config (blockSize));
    tested.prepare (chain_test::config (blockSize));

    EffectChannelParams refParams = p, testParams = p;

    std::vector<std::vector<float>> referenceOut, testedOut;
    referenceOut.reserve (400);
    testedOut.reserve (400);

    for (int block = 0; block < 400; ++block)
    {
        if (block == 100)
        {
            testParams.order = swapped;
            chain_test::publish (testParams);
        }
        else if (block == 200)
        {
            testParams.order = kDefaultOrder;
            chain_test::publish (testParams);
        }

        std::vector<float> a ((size_t) blockSize), b ((size_t) blockSize);
        for (int i = 0; i < blockSize; ++i)
        {
            const float x = 0.6f * std::sin (0.031f * (float) (block * blockSize + i))
                          + 0.2f * FrDiffusion::hashNoiseBipolar ((std::uint32_t) (block * blockSize + i), 91u);
            a[(size_t) i] = x;
            b[(size_t) i] = x;
        }

        reference.process (a.data(), blockSize, refParams);
        tested.process (b.data(), blockSize, testParams);

        referenceOut.push_back (a);
        testedOut.push_back (b);
    }

    // The other order really is a different sound.
    int differingBlocks = 0;
    for (int block = 150; block < 200; ++block)
        if (! eqtests::bitEqualBlock (referenceOut[(size_t) block], testedOut[(size_t) block]))
            ++differingBlocks;
    CHECK (differingBlocks > 40);

    // ...and coming back is exact.
    bool converged = true;
    for (int block = 300; block < 400; ++block)
        if (! eqtests::bitEqualBlock (referenceOut[(size_t) block], testedOut[(size_t) block]))
            converged = false;
    CHECK (converged);

    CHECK (! tested.isReorderPending());
    CHECK (tested.getCurrentOrder() == kDefaultOrder);
    CHECK (tested.nanTrips.load() == 0);

    // An order that is not a permutation is ignored, not obeyed.
    EffectChannelParams broken = testParams;
    broken.order[3] = broken.order[4];
    chain_test::publish (broken);
    std::vector<float> buf ((size_t) blockSize, 0.25f);
    tested.process (buf.data(), blockSize, broken);
    CHECK (tested.getCurrentOrder() == kDefaultOrder);
    CHECK (! tested.isReorderPending());
}

static void testChainLatencySum()
{
    using namespace spatcore::effects;
    chain_test::resetSlotSettings();

    for (int i = 0; i < kNumModuleSlots; ++i)
        chain_test::slotLatency[i] = 10 * (i + 1);

    EffectChain chain;
    chain.prepare (chain_test::config(), &chain_test::typedFactory);

    EffectChannelParams p;
    std::vector<float> buf (64, 0.0f);

    // Everything bypassed: a chain that is doing nothing reports no latency,
    // whatever the modules would cost if they were switched on.
    chain.process (buf.data(), 64, p);
    CHECK (chain.getLatencySamples() == 0);

    // Only live slots count, and they add up.
    p.dist.bypass = 0;                                    // slot 0 -> 10
    p.eq[1].bypass = 0;                                   // slot 2 -> 30
    chain_test::publish (p);
    chain.process (buf.data(), 64, p);
    CHECK (chain.getLatencySamples() == 40);

    // A bypassed chain reports nothing at all.
    p.chainBypass = 1;
    chain_test::publish (p);
    chain.process (buf.data(), 64, p);
    CHECK (chain.getLatencySamples() == 0);

    chain_test::resetSlotSettings();
}

static void testChainBypassAndMute()
{
    using namespace spatcore::effects;
    chain_test::resetSlotSettings();

    // Two live slots at half gain each: a chain output of 0.25 for a DC of 1.
    chain_test::slotGain[chain_test::slotIndexFor (ModuleId::Dist, 0)] = 0.5f;
    chain_test::slotGain[chain_test::slotIndexFor (ModuleId::Trem, 0)] = 0.5f;

    EffectChain chain;
    chain.prepare (chain_test::config(), &chain_test::typedFactory);

    EffectChannelParams p;
    p.dist.bypass = 0;
    p.trem.bypass = 0;

    auto run = [&chain, &p] (int blocks) -> float
    {
        std::vector<float> buf (64);
        float last = 0.0f;
        for (int b = 0; b < blocks; ++b)
        {
            std::fill (buf.begin(), buf.end(), 1.0f);
            chain.process (buf.data(), 64, p);
            last = buf[63];
        }
        return last;
    };

    run (60);                                             // let the slot fades settle
    CHECK (run (1) == 0.25f);

    // Chain bypass reaches EXACTLY the dry signal, monotonically, and resets the
    // slots once it is there.
    p.chainBypass = 1;
    chain_test::publish (p);

    float previous = 0.25f;
    bool monotonic = true;
    std::vector<float> buf (64);
    for (int b = 0; b < 40; ++b)
    {
        std::fill (buf.begin(), buf.end(), 1.0f);
        chain.process (buf.data(), 64, p);
        for (int i = 0; i < 64; ++i)
        {
            if (buf[(size_t) i] < previous - 1.0e-7f)
                monotonic = false;
            previous = buf[(size_t) i];
        }
    }
    CHECK (monotonic);
    CHECK (run (1) == 1.0f);

    const int distSlot = chain_test::slotIndexFor (ModuleId::Dist, 0);
    const int resetsWhileBypassed = chain_test::slotResets[distSlot];
    run (10);
    CHECK (chain_test::slotResets[distSlot] == resetsWhileBypassed);   // reset once, not every block

    // Back again.
    p.chainBypass = 0;
    chain_test::publish (p);
    run (60);
    CHECK (run (1) == 0.25f);

    // Mute reaches exactly silence, and unmuting comes straight back.
    p.mute = 1;
    chain_test::publish (p);
    run (60);
    CHECK (run (1) == 0.0f);

    p.mute = 0;
    chain_test::publish (p);
    run (60);
    CHECK (run (1) == 0.25f);

    // Mute and chain bypass compose: dry, then silenced.
    p.mute = 1;
    p.chainBypass = 1;
    chain_test::publish (p);
    run (60);
    CHECK (run (1) == 0.0f);

    chain_test::resetSlotSettings();
}

static void testChainNeutralAndGuarded()
{
    using namespace spatcore::effects;

    // A chain of real modules at defaults is bit-transparent - the state a
    // freshly created effects channel is in.
    EffectChain chain;
    chain.prepare (chain_test::config (256));

    const EffectChannelParams defaults;

    for (int block = 0; block < 8; ++block)
    {
        std::vector<float> buf = (block == 0) ? eqtests::makeAwkwardSignal (256)
                                              : module_test::awkwardBlock (256, 40 + block);
        const std::vector<float> reference = buf;
        chain.process (buf.data(), 256, defaults);
        CHECK (eqtests::bitEqualBlock (buf, reference));
    }

    CHECK (chain.getLatencySamples() == 0);
    CHECK (chain.nanTrips.load() == 0);

    // A non-finite sample arriving from outside is caught by the chain's own
    // guard even with every slot bypassed, so nothing reaches the return ring.
    std::vector<float> poisoned (256, 0.5f);
    poisoned[255] = std::numeric_limits<float>::quiet_NaN();
    chain.process (poisoned.data(), 256, defaults);
    for (int i = 0; i < 256; ++i)
        CHECK (poisoned[(size_t) i] == 0.0f);
    CHECK (chain.nanTrips.load() == 1);
}

//==============================================================================
// effects/modules - Distortion (FxDist)
//==============================================================================

namespace disttest
{
    using namespace spatcore::effects;

    /** Active, oversampling off, every stage at its identity: drive and output
        at 0 dB are EXACTLY 1, the four shelves at 0 dB are switched off, and
        the shaper sits on the hard clip. */
    inline EffectChannelParams flat()
    {
        EffectChannelParams p;
        p.dist.bypass = 0;
        p.dist.oversample = 1;                      // off: nothing here depends on JUCE
        p.dist.driveDb = 0.0f;
        p.dist.shape = 0.0f;
        p.dist.bias = 0.0f;
        p.dist.outputDb = 0.0f;
        p.dist.mix = 100.0f;
        p.dist.preLoShelfDb = p.dist.preHiShelfDb = 0.0f;
        p.dist.postLoShelfDb = p.dist.postHiShelfDb = 0.0f;
        return p;
    }

    /** First output sample for a DC input, from a module whose first (and so
        snapping) parameter set is p. Sample 0 is before the DC blocker or any
        filter has had a chance to move it. */
    inline float firstOut (const EffectChannelParams& p, float dcIn)
    {
        DistortionModule m;
        m.prepare (module_test::config (48000.0, 256));
        m.applyParams (p, 0);
        std::vector<float> buf = module_test::dc (64, dcIn);
        module_test::render (m, buf);
        return buf[0];
    }

    /** Small enough that a +6 dB shelf still cannot reach the clip ceiling. */
    inline std::vector<float> smallNoise (int n, int seed)
    {
        std::vector<float> v ((size_t) n);
        for (int i = 0; i < n; ++i)
            v[(size_t) i] = 0.05f * FrDiffusion::hashNoiseBipolar ((std::uint32_t) (i + seed * 977), 17u);
        return v;
    }
}

static void testDistortionBypassAndIdentity()
{
    using namespace spatcore::effects;

    // Bypassed in a slot at the defaults: the slot never runs the module, so
    // the buffer comes back untouched, negative zeros and denormals included.
    {
        ModuleSlot slot;
        slot.prepare (module_test::config (48000.0, 256), std::make_unique<DistortionModule>());
        EffectChannelParams p;                              // dist.bypass == 1
        slot.applyParams (p, 0);
        std::vector<float> buf = eqtests::makeAwkwardSignal (256);
        const std::vector<float> reference = buf;
        for (int b = 0; b < 4; ++b)
            slot.process (buf.data(), 256);
        CHECK (eqtests::bitEqualBlock (buf, reference));
    }

    // Mix 0 while ACTIVE is the identity AT THE LATENCY THE MODULE REPORTS.
    // With no oversampler there is no latency to honour, so it is the plain
    // identity and has to be BIT-exact: the module returns early rather than
    // crossfading the input against itself, and the negative zero and the two
    // denormals come back untouched. 40 dB of drive would be unmissable at any
    // other mix.
    {
        DistortionModule m;
        m.prepare (module_test::config (48000.0, 256));
        EffectChannelParams p = disttest::flat();
        p.dist.mix = 0.0f;
        p.dist.driveDb = 40.0f;
        p.dist.oversample = 1;
        CHECK (! m.applyParams (p, 0).bypass);
        CHECK (m.getLatencySamples() == 0);
        std::vector<float> buf = eqtests::makeAwkwardSignal (256);
        const std::vector<float> reference = buf;
        module_test::render (m, buf);
        CHECK (eqtests::bitEqualBlock (buf, reference));
    }

    // With the oversampler ON the module reports a latency, and what mix 0
    // owes is the input DELAYED BY IT. Handing back an UNDELAYED block here
    // would be a module that reports five samples of delay on a block it just
    // passed through, and a channel that jumps five samples through time the
    // moment the mix smoother lands on zero - testDistortionMixAlignment
    // measures that step. Still bit-exact, because the alignment line is a
    // plain ring: the awkward signal arrives intact, five samples late. The
    // first L samples are the module's own pre-roll.
    {
        DistortionModule m;
        m.prepare (module_test::config (48000.0, 256));
        EffectChannelParams p = disttest::flat();
        p.dist.mix = 0.0f;
        p.dist.driveDb = 40.0f;
        p.dist.oversample = 3;
        CHECK (! m.applyParams (p, 0).bypass);
        const int L = m.getLatencySamples();
        CHECK (L > 0);

        std::vector<float> buf = eqtests::makeAwkwardSignal (256);
        const std::vector<float> reference = buf;
        module_test::render (m, buf);

        bool delayedIdentity = true;
        for (int i = L; i < 256; ++i)
            if (! bitEqualFloat (buf[(size_t) i], reference[(size_t) (i - L)]))
                delayedIdentity = false;
        CHECK (delayedIdentity);
    }

    // Fully wet with every stage at its identity is bit-exact too: 0 dB is
    // exactly 1, a 0 dB shelf is switched OFF rather than run at unity, and
    // the hard clip leaves anything inside +-0.8 alone. The exceptions are the
    // two +-1e7 samples, which it must catch.
    {
        DistortionModule m;
        m.prepare (module_test::config (48000.0, 256));
        m.applyParams (disttest::flat(), 0);
        std::vector<float> buf = eqtests::makeAwkwardSignal (256);
        const std::vector<float> reference = buf;
        module_test::render (m, buf);

        bool transparent = true;
        for (int i = 0; i < 256; ++i)
            if (i != 3 && i != 4 && ! bitEqualFloat (buf[(size_t) i], reference[(size_t) i]))
                transparent = false;
        CHECK (transparent);
        CHECK (bitEqualFloat (buf[3], 0.8f));
        CHECK (bitEqualFloat (buf[4], -0.8f));
    }
}

static void testDistortionShaperLaw()
{
    using namespace spatcore::effects;
    namespace fd = spatcore::dsp::FastDecibels;
    EffectChannelParams p = disttest::flat();

    // Hard clip with 12 dB of drive: 0.5 * 10^(12/20) = 1.991, past the +-0.8
    // ceiling. That SAMPLE 0 already sits on it pins the first applyParams
    // after prepare snapping its smoothers rather than gliding up.
    p.dist.driveDb = 12.0f;
    CHECK (bitEqualFloat (disttest::firstOut (p, 0.5f), 0.8f));

    // Pure tanh: tanh(0.2) = 0.1973753. Shape 0.5 is the plain crossfade,
    // 0.5 + 0.5*(tanh(0.5) - 0.5) = 0.5 + 0.5*(0.4621172 - 0.5) = 0.4810586,
    // which also pins the SENSE of the control (0 clip, 1 tanh - the
    // prototype's percentage reads the other way). Tolerances because
    // std::tanh moves by an ULP or two across platforms.
    p.dist.driveDb = 0.0f;
    p.dist.shape = 1.0f;
    CHECK (std::fabs (disttest::firstOut (p, 0.2f) - 0.1973753f) < 1.0e-6f);
    p.dist.shape = 0.5f;
    CHECK (std::fabs (disttest::firstOut (p, 0.5f) - 0.4810586f) < 1.0e-6f);

    // Bias, the plan's addition. The curve still passes through the origin, so
    // tanh(0 + 0.3) - tanh(0.3) is exactly 0 and silence stays silent; DC in
    // gives tanh(0.5) - tanh(0.3) = 0.4621172 - 0.2913126 = 0.1708046 at
    // sample 0, before the DC blocker the bias engages removes the offset.
    p.dist.shape = 1.0f;
    p.dist.bias = 0.3f;
    CHECK (disttest::firstOut (p, 0.0f) == 0.0f);
    CHECK (std::fabs (disttest::firstOut (p, 0.2f) - 0.1708046f) < 1.0e-6f);

    // The output control is a GAIN: the prototype's inlet is named
    // "outputAttenuation" but a positive value boosts, and the plan keeps that
    // arithmetic. 0.1 never reaches the ceiling, so this is the gain alone,
    // computed the same way and therefore bit-exact.
    p = disttest::flat();
    p.dist.outputDb = 12.0f;
    CHECK (bitEqualFloat (disttest::firstOut (p, 0.1f), 0.1f * fd::dbToGain (12.0f)));
}

static void testDistortionShelves()
{
    using namespace spatcore::effects;
    using spatcore::dsp::OutputEQBiquadFilter;

    // With the clipper chosen and the signal well inside +-0.8 the shaper is
    // the identity and both gains are exactly 1, so the module IS its four
    // shelves: the shared RBJ pair, in the prototype's order, at slope 0.7
    // with the gain in the GAIN slot. A q/slope mix-up or a shelf in the wrong
    // place breaks the bit match.
    {
        DistortionModule m;
        m.prepare (module_test::config (48000.0, 256));
        EffectChannelParams p = disttest::flat();
        p.dist.preLoShelfHz  = 120.0f;  p.dist.preLoShelfDb  =  6.0f;
        p.dist.preHiShelfHz  = 6000.0f; p.dist.preHiShelfDb  = -4.0f;
        p.dist.postLoShelfHz = 200.0f;  p.dist.postLoShelfDb = -8.0f;
        p.dist.postHiShelfHz = 9000.0f; p.dist.postHiShelfDb =  3.0f;
        m.applyParams (p, 0);

        OutputEQBiquadFilter ref[4];
        for (int i = 0; i < 4; ++i)
            ref[i].prepare (48000.0);
        ref[0].setParameters (2, 120.0f,   6.0f, 0.7f, 0.7f);
        ref[1].setParameters (5, 6000.0f, -4.0f, 0.7f, 0.7f);
        ref[2].setParameters (2, 200.0f,  -8.0f, 0.7f, 0.7f);
        ref[3].setParameters (5, 9000.0f,  3.0f, 0.7f, 0.7f);

        bool matches = true;
        for (int block = 0; block < 4; ++block)
        {
            std::vector<float> a = disttest::smallNoise (256, 11 + block);
            std::vector<float> b = a;
            module_test::render (m, a);
            for (int i = 0; i < 4; ++i)
                ref[i].processBlock (b.data(), 256);
            if (! eqtests::bitEqualBlock (a, b))
                matches = false;
        }
        CHECK (matches);
    }

    // The CORRECTED RBJ gain law: a low shelf passes DC at A^2 =
    // 10^(gainDb/20), so 0.1 in gives 0.1, 0.398107 and 0.0251189. The
    // prototype's shelves can only ever BOOST (0 dB is +1.00 dB there, -12 dB
    // still +0.25 dB), so a faithful port fails the cut. 20000 samples is ~46
    // time constants of a 20 Hz corner; 2 % covers the float32 steady state.
    {
        const float gains[3]    = { 0.0f, 12.0f, -12.0f };
        const float expected[3] = { 0.0f, 0.398107f, 0.0251189f };  // [0] unused: 0 dB takes the bit-exact branch

        for (int k = 0; k < 3; ++k)
        {
            DistortionModule m;
            m.prepare (module_test::config (48000.0, 256));
            EffectChannelParams p = disttest::flat();
            p.dist.preLoShelfHz = 20.0f;
            p.dist.preLoShelfDb = gains[k];
            m.applyParams (p, 0);
            std::vector<float> buf = module_test::dc (20000, 0.1f);
            module_test::render (m, buf);

            if (k == 0)
                CHECK (bitEqualFloat (buf[19999], 0.1f));   // 0 dB is OFF, not "nearly unity"
            else
                CHECK (std::fabs (buf[19999] - expected[k]) < 0.02f * expected[k]);
        }
    }
}

static void testDistortionMixAlignment()
{
    using namespace spatcore::effects;

    // The dry leg of the mix is delayed to match the oversampler, and nothing
    // else in this file looks at it: the latency probe in
    // testDistortionOversampling runs at mix 100, so it measures the WET leg
    // alone and would pass unchanged with the alignment line deleted.

    // 1. NO COMB AT A PARTIAL MIX. A 0.2 sine never reaches the +-0.8 ceiling
    // and the clipper is the chosen curve, so the shaper is the identity and
    // the wet leg is a bare oversampler round trip. A 50 % mix must then be
    // exactly half the DELAYED dry plus half that wet leg. Summing an
    // undelayed dry against it instead is a comb - five samples of offset puts
    // a notch at 4.8 kHz - and even at 1 kHz it costs level and turns the
    // phase.
    {
        const int n = 4096;
        std::vector<float> in ((size_t) n);
        for (int i = 0; i < n; ++i)
            in[(size_t) i] = 0.2f * std::sin (6.283185307f * 1000.0f * (float) i / 48000.0f);

        EffectChannelParams p = disttest::flat();
        p.dist.oversample = 3;

        DistortionModule wetOnly;
        wetOnly.prepare (module_test::config (48000.0, 256));
        wetOnly.applyParams (p, 0);
        std::vector<float> wet = in;
        module_test::render (wetOnly, wet);

        p.dist.mix = 50.0f;
        DistortionModule halfway;
        halfway.prepare (module_test::config (48000.0, 256));
        halfway.applyParams (p, 0);
        const int L = halfway.getLatencySamples();
        std::vector<float> half = in;
        module_test::render (halfway, half);

        CHECK (L > 0);

        float worst = 0.0f;
        for (int i = L; i < n; ++i)
        {
            const float want = 0.5f * in[(size_t) (i - L)] + 0.5f * wet[(size_t) i];
            worst = std::fmax (worst, std::fabs (half[(size_t) i] - want));
        }
        CHECK (worst < 1.0e-6f);
    }

    // 2. MIX 0 AND MIX 1 % AGREE ABOUT WHERE IN TIME THE OUTPUT SITS, and the
    // module reports that delay honestly at both. An early-out that skips the
    // alignment line returns x[n] at mix 0 and x[n - 5] one per cent later,
    // while claiming five samples of latency at both.
    {
        const int n = 4096;
        std::vector<float> in ((size_t) n);
        for (int i = 0; i < n; ++i)
            in[(size_t) i] = 0.2f * std::sin (6.283185307f * 1000.0f * (float) i / 48000.0f)
                           + 0.1f * std::sin (6.283185307f * 3100.0f * (float) i / 48000.0f);

        const float mixes[2] = { 0.0f, 1.0f };
        int measured[2] = { -1, -1 };

        for (int k = 0; k < 2; ++k)
        {
            EffectChannelParams p = disttest::flat();
            p.dist.oversample = 3;
            p.dist.mix = mixes[k];

            DistortionModule m;
            m.prepare (module_test::config (48000.0, 256));
            m.applyParams (p, 0);
            std::vector<float> out = in;
            module_test::render (m, out);

            double bestErr = -1.0;
            for (int d = 0; d < 16; ++d)
            {
                double err = 0.0;
                for (int i = 1024; i < 3072; ++i)
                {
                    const double e = (double) out[(size_t) i] - (double) in[(size_t) (i - d)];
                    err += e * e;
                }
                if (bestErr < 0.0 || err < bestErr) { bestErr = err; measured[k] = d; }
            }

            CHECK (measured[k] == m.getLatencySamples());
        }

        CHECK (measured[0] == measured[1]);
    }

    // 3. CROSSING MIX 0 DOES NOT MOVE THE CHANNEL IN TIME. oversample 0 is
    // auto, which is 4x at 48 kHz, so this is the DEFAULT configuration: an
    // ordinary fader ride to zero in 64-sample blocks. Before the dry leg was
    // delivered at mix 0 this measured an excess step of 0.143 on a 0.30-peak
    // sine - 48 % of the amplitude, at exactly the block where the mix smoother
    // landed on zero.
    {
        const int n = 24000, block = 64, dropAt = 4096;
        std::vector<float> in ((size_t) n);
        for (int i = 0; i < n; ++i)
            in[(size_t) i] = 0.30f * std::sin (6.283185307f * 1000.0f * (float) i / 48000.0f);
        std::vector<float> out = in;

        float natural = 0.0f;
        for (int i = 1; i < n; ++i)
            natural = std::fmax (natural, std::fabs (in[(size_t) i] - in[(size_t) (i - 1)]));

        EffectChannelParams p = disttest::flat();
        p.dist.oversample = 0;                              // auto
        DistortionModule m;
        m.prepare (module_test::config (48000.0, block));
        m.applyParams (p, 0);
        CHECK (m.getLatencySamples() > 0);                  // auto really did pick a factor

        for (int off = 0; off + block <= n; off += block)
        {
            if (off == dropAt)
            {
                p.dist.mix = 0.0f;
                m.applyParams (p, 0);
            }
            m.process (out.data() + off, block);
        }

        float excess = 0.0f;
        for (int i = dropAt; i < n; ++i)
            excess = std::fmax (excess, std::fabs (out[(size_t) i] - out[(size_t) (i - 1)]) - natural);
        CHECK (excess < 0.01f);
    }
}

static void testDistortionDcBlocker()
{
    using namespace spatcore::effects;

    // 1. IT REMOVES DC. tanh(x + bias) - tanh(bias) passes through the origin,
    // so silence stays silent, but it is not odd-symmetric: the mean of a sine
    // through it is NOT zero. The same curve applied by hand is the reference
    // for how much there was to remove, so this cannot pass by the blocker
    // doing nothing to a signal that had no offset.
    {
        const int n = 48000;
        std::vector<float> in ((size_t) n);
        for (int i = 0; i < n; ++i)
            in[(size_t) i] = 0.6f * std::sin (6.283185307f * 200.0f * (float) i / 48000.0f);

        EffectChannelParams p = disttest::flat();
        p.dist.shape = 1.0f;
        p.dist.bias = 0.4f;

        DistortionModule m;
        m.prepare (module_test::config (48000.0, 256));
        m.applyParams (p, 0);
        std::vector<float> out = in;
        module_test::render (m, out);

        const float tb = std::tanh (0.4f);
        double shaped = 0.0, blocked = 0.0;

        for (int i = n / 2; i < n; ++i)                     // past the blocker's own settling
        {
            shaped  += (double) spatcore::dsp::Waveshaper::blend (in[(size_t) i], 1.0f, 0.4f, tb);
            blocked += (double) out[(size_t) i];
        }

        shaped  /= (double) (n / 2);
        blocked /= (double) (n / 2);

        CHECK (std::fabs (shaped) > 0.01);                  // there really was DC to remove
        CHECK (std::fabs (blocked) < 0.05 * std::fabs (shaped));
    }

    // 2. IT IS NOT TAKEN BACK OUT OF LIVE AUDIO. A 5 Hz high pass is not
    // transparent to low-frequency material - it turns a 50 Hz tone by 5.7
    // degrees - so switching it out in one sample is a step, not a no-op. Both
    // routes into the gate are checked: the shape control leaving the tanh leg,
    // and the bias itself going to zero. Before this was fixed the worst second
    // difference after the move was 1300x and 2300x the baseline; it is now
    // within a factor of a few, which is the block-endpoint ramp of shape and
    // bias, not a switch. The amplitude stays inside the +-0.8 ceiling so the
    // clipper contributes no corner of its own to the measurement.
    {
        for (int route = 0; route < 2; ++route)
        {
            const int n = 24000, block = 64, moveAt = 6400;
            std::vector<float> out ((size_t) n);
            for (int i = 0; i < n; ++i)
                out[(size_t) i] = 0.25f * std::sin (6.283185307f * 50.0f * (float) i / 48000.0f);

            EffectChannelParams p = disttest::flat();
            p.dist.shape = 1.0f;
            p.dist.bias = 0.3f;

            DistortionModule m;
            m.prepare (module_test::config (48000.0, block));
            m.applyParams (p, 0);

            for (int off = 0; off + block <= n; off += block)
            {
                if (off == moveAt)
                {
                    if (route == 0) p.dist.shape = 0.0f;    // the tanh leg leaves
                    else            p.dist.bias = 0.0f;     // the asymmetry leaves
                    m.applyParams (p, 0);
                }
                m.process (out.data() + off, block);
            }

            float before = 0.0f, after = 0.0f;
            for (int i = 1; i + 1 < moveAt; ++i)
                before = std::fmax (before, std::fabs (out[(size_t) (i + 1)] - 2.0f * out[(size_t) i] + out[(size_t) (i - 1)]));
            for (int i = moveAt; i + 1 < n; ++i)
                after = std::fmax (after, std::fabs (out[(size_t) (i + 1)] - 2.0f * out[(size_t) i] + out[(size_t) (i - 1)]));

            CHECK (before > 0.0f);
            CHECK (after < 10.0f * before);
        }
    }
}

static void testDistortionOversampling()
{
    using namespace spatcore::effects;

    // The enum is 0 auto, 1 off, 2 = 2x, 3 = 4x. Off reports NO latency, and
    // the factor is a VARIANT: it stays pending, the module running the old
    // value, until the slot reaches silence and commits.
    DistortionModule m;
    m.prepare (module_test::config (48000.0, 512));
    EffectChannelParams p = disttest::flat();               // oversample = 1, off
    CHECK (! m.applyParams (p, 0).variantPending);          // off IS the running factor, nothing to commit
    CHECK (m.getLatencySamples() == 0);                     // (the snap itself is pinned in testDistortionShaperLaw)

    p.dist.oversample = 3;
    CHECK (m.applyParams (p, 0).variantPending);
    CHECK (m.getLatencySamples() == 0);                     // still running the old one

    p.dist.oversample = 1;                                  // taken back before the fade ends
    CHECK (! m.applyParams (p, 0).variantPending);          // cancels itself, no reset

    p.dist.oversample = 3;
    CHECK (m.applyParams (p, 0).variantPending);
    m.reset();
    m.commitPendingVariant();
    const int l4 = m.getLatencySamples();
    CHECK (l4 > 0);
    CHECK (! m.applyParams (p, 0).variantPending);          // now it IS the running value

    // Auto: 4x through 48 kHz, 2x at 96 kHz (one stage, so no more latency
    // than 4x), off from 176.4 kHz up.
    {
        EffectChannelParams a = disttest::flat();
        a.dist.oversample = 0;
        DistortionModule low, mid, high;
        low.prepare (module_test::config (48000.0, 256));
        mid.prepare (module_test::config (96000.0, 256));
        high.prepare (module_test::config (192000.0, 256));
        low.applyParams (a, 0);
        mid.applyParams (a, 0);
        high.applyParams (a, 0);
        CHECK (low.getLatencySamples() == l4);
        CHECK (mid.getLatencySamples() > 0 && mid.getLatencySamples() <= l4);
        CHECK (high.getLatencySamples() == 0);
    }

    // The reported latency is the delay the module really applies. A 1 kHz
    // sine at 0.05 stays in the linear part of both curves (tanh's third
    // harmonic is near -74 dB), so this is close to a bare oversampler round
    // trip. One sample of slack: half-band IIR group delay is only near flat.
    {
        p.dist.shape = 1.0f;
        m.applyParams (p, 0);
        m.reset();                                          // snap the shape glide away

        std::vector<float> in ((size_t) 4096);
        for (int i = 0; i < 4096; ++i)
            in[(size_t) i] = 0.05f * std::sin (6.283185307f * 1000.0f * (float) i / 48000.0f);
        std::vector<float> out = in;
        module_test::render (m, out);

        int best = 0;
        double bestErr = -1.0;
        for (int d = 0; d < 16; ++d)
        {
            double err = 0.0;
            for (int i = 1024; i < 3072; ++i)
            {
                const double e = (double) out[(size_t) i] - (double) in[(size_t) (i - d)];
                err += e * e;
            }
            if (bestErr < 0.0 || err < bestErr) { bestErr = err; best = d; }
        }
        CHECK (best - l4 <= 1 && l4 - best <= 1);
    }
}

static void testDistortionResetAndRange()
{
    using namespace spatcore::effects;

    // reset() must leave nothing behind: four shelves, a DC blocker, an
    // oversampler and the dry leg's alignment line all carry state. Exact zero
    // is the right assertion - a cleared linear filter fed zeros rounds
    // nothing, and blend(0) is exactly 0 whenever the bias and its tanh agree,
    // which settled smoothers give. The mix is PARTIAL on purpose: at 100 %
    // wet the alignment line is not in the output at all, and it is the one
    // piece of state the mix-0 path deliberately does not drop, so nothing
    // else here would notice reset() forgetting it - the first five samples
    // would come back as the last five of the loud block, at 40 %.
    {
        DistortionModule m;
        m.prepare (module_test::config (48000.0, 256));
        EffectChannelParams p = disttest::flat();
        p.dist.mix = 60.0f;
        p.dist.oversample = 3;
        p.dist.driveDb = 24.0f;
        p.dist.shape = 1.0f;
        p.dist.bias = 0.4f;
        p.dist.preLoShelfHz  = 40.0f;   p.dist.preLoShelfDb  =  18.0f;
        p.dist.postHiShelfHz = 3000.0f; p.dist.postHiShelfDb = -18.0f;
        m.applyParams (p, 0);

        std::vector<float> loud = module_test::awkwardBlock (256, 3);
        module_test::render (m, loud);
        m.reset();

        std::vector<float> silence = module_test::dc (512, 0.0f);
        module_test::render (m, silence);

        bool silent = true;
        for (int i = 0; i < 512; ++i)
            if (silence[(size_t) i] != 0.0f)
                silent = false;
        CHECK (silent);
    }

    // Every control at a stop, plus NaN, must stay finite and bounded. The
    // bound is a BLOW-UP DETECTOR, not a level check: by hand the worst steady
    // state is about 63 (no drive pushes the shaper past 1, +24 dB of shelf is
    // 15.85x and +12 dB of output 3.98x), and 2000 is thirty times that, so it
    // catches an oscillation or a NaN leak and would happily pass a 20 dB
    // gain-staging error. The NaN frequency lands on a LIVE shelf on purpose,
    // because OutputEQBiquadFilter's own std::min/std::max pair passes a NaN
    // through.
    {
        const float drives[3] = { 0.0f, 40.0f, std::numeric_limits<float>::quiet_NaN() };
        const float shapes[3] = { 0.0f, 0.5f, 1.0f };
        const float biases[3] = { -0.5f, 0.0f, 0.5f };
        const std::uint8_t factors[3] = { 1, 2, 3 };

        DistortionModule m;
        m.prepare (module_test::config (48000.0, 256));
        bool finite = true, bounded = true;

        for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 3; ++c)
        {
            EffectChannelParams p = disttest::flat();
            p.dist.driveDb = drives[a];
            p.dist.shape = shapes[b];
            p.dist.bias = biases[c];
            p.dist.oversample = factors[(a + b + c) % 3];
            p.dist.outputDb = 12.0f;
            p.dist.mix = (c == 1) ? 0.0f : 55.0f;
            p.dist.preLoShelfDb = 24.0f;
            p.dist.postHiShelfDb = 24.0f;
            p.dist.preHiShelfDb = -12.0f;
            p.dist.preHiShelfHz = std::numeric_limits<float>::quiet_NaN();

            m.applyParams (p, 0);
            m.reset();
            m.commitPendingVariant();
            m.applyParams (p, 0);

            std::vector<float> buf = module_test::awkwardBlock (256, 7 + a * 9 + b * 3 + c);
            module_test::render (m, buf);

            for (int i = 0; i < 256; ++i)
            {
                const float v = buf[(size_t) i];
                if (! std::isfinite (v))     finite = false;
                if (std::fabs (v) > 2000.0f) bounded = false;
            }
        }
        CHECK (finite);
        CHECK (bounded);
    }

    // Two instances, same settings, same audio - and no dependence on where
    // the block boundaries fall. The module carries per-block ramps and
    // retunes its shelves once a block, so a SETTLED parameter set must render
    // the same whatever size the host hands over. Both factors are checked,
    // because the oversampler is the one stage that could carry a block size
    // into its own state.
    {
        EffectChannelParams base = disttest::flat();
        base.dist.driveDb = 18.0f;
        base.dist.shape = 0.6f;
        base.dist.bias = 0.2f;
        base.dist.outputDb = -6.0f;
        base.dist.mix = 70.0f;
        base.dist.preLoShelfDb = 5.0f;
        base.dist.postHiShelfDb = -5.0f;

        const std::uint8_t factors[2] = { 1, 3 };           // off, and 4x

        for (int k = 0; k < 2; ++k)
        {
            EffectChannelParams p = base;
            p.dist.oversample = factors[k];

            DistortionModule whole, chunked;
            whole.prepare (module_test::config (48000.0, 512));
            chunked.prepare (module_test::config (48000.0, 512));
            whole.applyParams (p, 0);
            chunked.applyParams (p, 0);

            std::vector<float> va = disttest::smallNoise (1024, 4);
            std::vector<float> vb = va;
            whole.process (va.data(), 1024);
            for (int off = 0; off < 1024; off += 128)
                chunked.process (vb.data() + off, 128);
            CHECK (eqtests::bitEqualBlock (va, vb));
        }
    }

    // And the limit of that, stated rather than avoided: while shape or bias is
    // MOVING the render does depend on the block size, because both are a
    // linear ramp between the endpoints of whatever block arrives and the
    // shelves are retuned once per block. One 1024-sample call against 8 x 128
    // over a full shape glide differs by 2.8e-2 on a 0.4 signal - about -23 dB,
    // inaudible as a difference between two renders but a long way from zero,
    // and the reason a hash of a render taken DURING a parameter move is not a
    // portable reference.
    {
        EffectChannelParams p = disttest::flat();
        p.dist.driveDb = 12.0f;
        p.dist.shape = 1.0f;
        p.dist.mix = 100.0f;

        DistortionModule whole, chunked;
        whole.prepare (module_test::config (48000.0, 1024));
        chunked.prepare (module_test::config (48000.0, 1024));
        whole.applyParams (p, 0);                           // snaps to shape 1
        chunked.applyParams (p, 0);

        p.dist.shape = 0.0f;                                // now GLIDE it away
        whole.applyParams (p, 0);
        chunked.applyParams (p, 0);

        std::vector<float> va ((size_t) 1024);
        for (int i = 0; i < 1024; ++i)
            va[(size_t) i] = 0.4f * std::sin (0.07f * (float) i);
        std::vector<float> vb = va;

        whole.process (va.data(), 1024);
        for (int off = 0; off < 1024; off += 128)
            chunked.process (vb.data() + off, 128);

        float worst = 0.0f;
        for (int i = 0; i < 1024; ++i)
            worst = std::fmax (worst, std::fabs (va[(size_t) i] - vb[(size_t) i]));
        CHECK (worst > 0.0f);                               // it really is block-size dependent
        CHECK (worst < 0.05f);                              // and only by the ramp's own resolution
    }
}

//==============================================================================
// effects/modules - Dynamics (FxDyn, x2)
//==============================================================================

//==============================================================================
// effects/modules/DynamicsModule
//==============================================================================

namespace dyntests
{
    using namespace spatcore::effects;

    /** Instance 0 switched on with both stages neutral, so each test switches
        on only the thing it means to measure. */
    inline EffectChannelParams base()
    {
        EffectChannelParams p;
        DynamicsParams& d = p.dyn[0];
        d.bypass = 0;   d.compOn = 0;      d.expOn = 0;
        d.makeupDb = 0.0f;  d.lookaheadMs = 0.0f;  d.compDetectorDelayMs = 0.0f;
        return p;
    }

    /** The sample rate is a PARAMETER, not a literal. Every law below used to
        be measured at 48 kHz only, which is how a sidechain that diverged below
        40 kHz reached the shipped defaults unnoticed. */
    inline std::vector<float> sine (int n, float freqHz, float amp, float sr = 48000.0f)
    {
        std::vector<float> v ((size_t) n);
        for (int i = 0; i < n; ++i)
            v[(size_t) i] = amp * std::sin (6.283185307179586f * freqHz * (float) i / sr);
        return v;
    }

    inline float peak (const std::vector<float>& v, int from, int to)
    {
        float m = 0.0f;
        for (int i = from; i < to && i < (int) v.size(); ++i)
            m = std::max (m, std::fabs (v[(size_t) i]));
        return m;
    }

    /** The gain a settled stage is applying, averaged over a window so the
        detector's ripple does not decide the answer. Samples near a zero
        crossing are skipped: their ratio is all rounding.

        Every steady-state tolerance below is 0.008 on this number, and the
        ripple it has to absorb is set by DynamicsModule::kRmsWindowMs (10 ms).
        Retune that constant and these tests go loose or start failing, with
        nothing in the failure text to say why - so: this comment. */
    inline float steadyGain (const std::vector<float>& out, const std::vector<float>& in,
                             int from, int to)
    {
        const float floorLevel = 0.2f * peak (in, from, to);
        float sum = 0.0f;
        int count = 0;

        for (int i = from; i < to; ++i)
            if (std::fabs (in[(size_t) i]) > floorLevel)
            {
                sum += out[(size_t) i] / in[(size_t) i];
                ++count;
            }

        return count > 0 ? sum / (float) count : 0.0f;
    }

    /** The first sample the module did not pass through untouched. -1 if it
        passed the whole buffer. */
    inline int firstAltered (const std::vector<float>& out, const std::vector<float>& in)
    {
        for (int i = 0; i < (int) out.size() && i < (int) in.size(); ++i)
            if (! bitEqualFloat (out[(size_t) i], in[(size_t) i]))
                return i;

        return -1;
    }
}

static void testDynamicsBypassAndIdentity()
{
    using namespace spatcore::effects;
    ChainConfig cfg = module_test::config (48000.0, 256);

    // Bypassed in a slot at the shipped defaults: the buffer is not touched.
    {
        ModuleSlot slot;
        slot.prepare (cfg, std::make_unique<DynamicsModule>());
        slot.applyParams (EffectChannelParams(), 0);
        CHECK (slot.isBypassedSettled());
        CHECK (slot.getLatencySamples() == 0);

        for (int block = 0; block < 4; ++block)
        {
            std::vector<float> buf = (block == 0) ? eqtests::makeAwkwardSignal (256)
                                                  : module_test::awkwardBlock (256, block);
            const std::vector<float> reference = buf;
            slot.process (buf.data(), 256);
            CHECK (eqtests::bitEqualBlock (buf, reference));
        }

        CHECK (slot.nanTrips.load() == 0);
    }

    // Active and transparent, to the bit in both cases. Variant 0 is both
    // stages off, which returns early; variant 1 runs both at 1:1, where the
    // slope is exactly 0, so the gain computer returns exactly 0 dB and
    // FastDecibels::dbToGain (0) is exactly 1.
    //
    // Variant 1 is the one that needs the module to WRITE the sample through
    // rather than multiply it by one: makeAwkwardSignal plants two denormals,
    // and the audio thread runs inside juce::ScopedNoDenormals, where
    // denormal * 1.0f is 0.0f. This suite does not arm FTZ/DAZ, so it cannot
    // observe that on its own - it is pinned here by inspection of the fact
    // that process() stores y unmultiplied when the total gain is exactly 1.
    for (int variant = 0; variant < 2; ++variant)
    {
        DynamicsModule m;
        m.prepare (cfg);

        EffectChannelParams p = dyntests::base();
        p.dyn[0].compOn = (std::uint8_t) variant;
        p.dyn[0].expOn = (std::uint8_t) variant;
        p.dyn[0].compRatio = 1.0f;
        p.dyn[0].expRatio = 1.0f;
        m.applyParams (p, 0);

        std::vector<float> buf = eqtests::makeAwkwardSignal (256);
        const std::vector<float> reference = buf;
        module_test::render (m, buf);
        CHECK (eqtests::bitEqualBlock (buf, reference));
        CHECK (m.getLatencySamples() == 0);
    }

    // The same, with the two denormals swapped for ordinary values, so that the
    // assertion above is not the only thing standing between a multiply-by-one
    // and a green suite: this one would pass either way, and its job is to say
    // that nothing ELSE in the 1:1 path moves a bit.
    {
        DynamicsModule m;
        m.prepare (cfg);

        EffectChannelParams p = dyntests::base();
        p.dyn[0].compOn = p.dyn[0].expOn = 1;
        p.dyn[0].compRatio = p.dyn[0].expRatio = 1.0f;
        m.applyParams (p, 0);

        std::vector<float> buf = module_test::awkwardBlock (256, 3);
        const std::vector<float> reference = buf;
        module_test::render (m, buf);
        CHECK (eqtests::bitEqualBlock (buf, reference));
    }
}

static void testDynamicsCompressorLaw()
{
    using namespace spatcore::effects;

    // A 1 kHz sine at 0.5 read as RMS settles at 20*log10 (0.5/sqrt 2) =
    // -9.031 dBFS. Threshold -20, ratio 4:1, so the gain is
    // (1/4 - 1)*(-9.031 + 20) = -8.227 dB = 0.3878 linear. The sidechain (a
    // 20 Hz low cut and a 20 kHz high cut) moves the level by under 0.01 dB at
    // 1 kHz, so 2 % is generous rather than loose.
    DynamicsModule m;
    m.prepare (module_test::config());

    EffectChannelParams p = dyntests::base();
    p.dyn[0].compOn = 1;
    p.dyn[0].detector = 1;                      // RMS: a steady level to predict from
    p.dyn[0].compThresholdDb = -20.0f;  p.dyn[0].compRatio = 4.0f;
    p.dyn[0].compAttackMs = 5.0f;       p.dyn[0].compReleaseMs = 50.0f;
    m.applyParams (p, 0);

    const std::vector<float> in = dyntests::sine (48000, 1000.0f, 0.5f);
    std::vector<float> out = in;
    module_test::render (m, out);

    CHECK (std::fabs (dyntests::steadyGain (out, in, 40000, 48000) - 0.38785f) <= 0.008f);
    CHECK (std::fabs (m.getMeterDb() + 8.227f) <= 0.3f);

    // The same settings 23 dB below the threshold: nothing engages, and
    // "nothing" is bit-exact rather than nearly so.
    DynamicsModule quiet;
    quiet.prepare (module_test::config());
    quiet.applyParams (p, 0);

    std::vector<float> below = dyntests::sine (4800, 1000.0f, 0.01f);
    const std::vector<float> reference = below;
    module_test::render (quiet, below);
    CHECK (eqtests::bitEqualBlock (below, reference));
    CHECK (quiet.getMeterDb() == 0.0f);
}

static void testDynamicsPeakDetectorLaw()
{
    using namespace spatcore::effects;

    // PEAK is the shipped default and the prototype's only mode, and it needs
    // its own derivation: with attack and release both 0 ms the follower is the
    // bare rectifier, so the level sweeps from the signal peak down to kMinDb
    // and back twice per cycle and there is no closed form for the average.
    //
    // A release long enough to be negligible removes the sweep from the answer.
    // At 2000 ms the ramp recovers about 1/96000 of its distance per sample, so
    // over one 1 kHz period it gives back ~0.05 % while the 0.05 ms attack
    // takes the whole gap at each peak: the gain RATCHETS down to the static
    // law evaluated at the PEAK level and stays there.
    //
    //   peak of a 0.5 sine = -6.0206 dBFS, threshold -20 -> over = 13.9794
    //   slope = 1/4 - 1 = -0.75  ->  -10.4845 dB  ->  0.299066 linear
    //
    // Measured 0.3012, i.e. the residual recovery, so the band is one-sided in
    // the direction the derivation predicts.
    DynamicsModule m;
    m.prepare (module_test::config());

    EffectChannelParams p = dyntests::base();
    p.dyn[0].compOn = 1;
    p.dyn[0].detector = 0;                      // PEAK
    p.dyn[0].compThresholdDb = -20.0f;  p.dyn[0].compRatio = 4.0f;
    p.dyn[0].compAttackMs = 0.05f;      p.dyn[0].compReleaseMs = 2000.0f;
    m.applyParams (p, 0);

    const std::vector<float> in = dyntests::sine (48000, 1000.0f, 0.5f);
    std::vector<float> out = in;
    module_test::render (m, out);

    const float g = dyntests::steadyGain (out, in, 40000, 48000);
    CHECK (g >= 0.299066f - 0.002f);
    CHECK (g <= 0.299066f + 0.006f);
    CHECK (std::fabs (m.getMeterDb() + 10.4845f) <= 0.3f);

    // And the two modes really are different laws, not a flag with no effect:
    // the same numbers in RMS settle at 0.3878, because the RMS of the sine is
    // 3 dB under its peak and the compressor works on 3 dB less overshoot.
    DynamicsModule rms;
    rms.prepare (module_test::config());
    p.dyn[0].detector = 1;
    rms.applyParams (p, 0);

    std::vector<float> rmsOut = in;
    module_test::render (rms, rmsOut);
    const float gr = dyntests::steadyGain (rmsOut, in, 40000, 48000);
    CHECK (std::fabs (gr - 0.38785f) <= 0.008f);
    CHECK (gr - g > 0.05f);
}

static void testDynamicsExpanderLawAndRange()
{
    using namespace spatcore::effects;

    // A 1 kHz sine at 0.05 reads as -29.031 dBFS RMS, 9.031 dB under a
    // threshold of -20. A 2:1 downward expander takes (R - 1)*over = -9.031 dB
    // off, a gain of 0.3536. The prototype's (1 - 1/R) slope would give
    // -4.52 dB = 0.594, which this separates by a factor of 1.7.
    DynamicsModule m;
    m.prepare (module_test::config());

    EffectChannelParams p = dyntests::base();
    p.dyn[0].expOn = 1;                 p.dyn[0].detector = 1;
    p.dyn[0].expThresholdDb = -20.0f;   p.dyn[0].expRatio = 2.0f;
    p.dyn[0].expRangeDb = -60.0f;       p.dyn[0].expHoldMs = 0.0f;
    p.dyn[0].expAttackMs = 5.0f;        p.dyn[0].expReleaseMs = 50.0f;
    m.applyParams (p, 0);

    const std::vector<float> in = dyntests::sine (48000, 1000.0f, 0.05f);
    std::vector<float> out = in;
    module_test::render (m, out);
    CHECK (std::fabs (dyntests::steadyGain (out, in, 40000, 48000) - 0.35355f) <= 0.008f);

    // The meter is the deepest reduction of the block, so on a settled tone it
    // is the law's own answer read back - the expander's side of the derived
    // meter check the compressor law already makes.
    CHECK (std::fabs (m.getMeterDb() + 9.031f) <= 0.3f);

    // At 100:1 the law asks for 99*(-9.031) = -894 dB; the range floor stops it
    // at -20 dB, a gain of exactly 0.1.
    p.dyn[0].expRatio = 100.0f;
    p.dyn[0].expRangeDb = -20.0f;
    m.applyParams (p, 0);

    std::vector<float> floored = in;
    module_test::render (m, floored);
    CHECK (std::fabs (dyntests::steadyGain (floored, in, 40000, 48000) - 0.1f) <= 0.002f);
}

static void testDynamicsKneeAndAutoMakeup()
{
    using namespace spatcore::effects;

    // The soft knee is the plan's main addition to the prototype and the whole
    // of it is one line - slope * t * t / (2 * knee) with t = over + knee/2 - so
    // a one-character slip there would survive every other test in this file.
    // Nine points across three widths and three overs, against the law written
    // out by hand rather than recomputed from the same expression.
    //
    //   W = 0:   hard, so 0 dB below the threshold and slope*over above it
    //   W = 12:  the quadratic runs over -6..+6, meeting both lines exactly
    //   W = 24:  the quadratic runs over -12..+12, so all three points are in it
    struct KneePoint { float widthDb, overDb, expectDb; };

    static const KneePoint points[9] =
    {
        {  0.0f, -6.0f,  0.0000f }, {  0.0f, 0.0f,  0.0000f }, {  0.0f, 6.0f, -4.5000f },
        { 12.0f, -6.0f,  0.0000f }, { 12.0f, 0.0f, -1.1250f }, { 12.0f, 6.0f, -4.5000f },
        { 24.0f, -6.0f, -0.5625f }, { 24.0f, 0.0f, -2.2500f }, { 24.0f, 6.0f, -5.0625f }
    };

    for (const KneePoint& pt : points)
    {
        // RMS of a sine is its amplitude over sqrt 2, so this amplitude puts the
        // detector exactly `overDb` above the -20 dB threshold.
        const float amp = std::pow (10.0f, (-20.0f + pt.overDb) / 20.0f) * 1.41421356f;

        DynamicsModule m;
        m.prepare (module_test::config());

        EffectChannelParams p = dyntests::base();
        p.dyn[0].compOn = 1;                p.dyn[0].detector = 1;
        p.dyn[0].compThresholdDb = -20.0f;  p.dyn[0].compRatio = 4.0f;
        p.dyn[0].compKneeDb = pt.widthDb;
        p.dyn[0].compAttackMs = 5.0f;       p.dyn[0].compReleaseMs = 50.0f;
        m.applyParams (p, 0);

        const std::vector<float> in = dyntests::sine (48000, 1000.0f, amp);
        std::vector<float> out = in;
        module_test::render (m, out);

        const float gainDb = 20.0f * std::log10 (dyntests::steadyGain (out, in, 40000, 48000));
        CHECK (std::fabs (gainDb - pt.expectDb) <= 0.06f);
    }

    // Auto makeup is HALF of what the gain computer takes off a signal sitting
    // exactly at the threshold: -T*(1 - 1/R)/2 = 20*0.75/2 = +7.5 dB at T = -20,
    // R = 4. The plan carries an open question on that /2, so what is pinned
    // here is that the code implements what the plan currently says.
    //
    // Measured as a RATIO of the same render with auto makeup off, so the
    // compressor's own -8.227 dB cancels and only the makeup is under test.
    float withAuto = 0.0f, without = 0.0f;

    for (int variant = 0; variant < 2; ++variant)
    {
        DynamicsModule m;
        m.prepare (module_test::config());

        EffectChannelParams p = dyntests::base();
        p.dyn[0].compOn = 1;                p.dyn[0].detector = 1;
        p.dyn[0].autoMakeup = (std::uint8_t) variant;
        p.dyn[0].compThresholdDb = -20.0f;  p.dyn[0].compRatio = 4.0f;
        p.dyn[0].compAttackMs = 5.0f;       p.dyn[0].compReleaseMs = 50.0f;
        m.applyParams (p, 0);

        const std::vector<float> in = dyntests::sine (48000, 1000.0f, 0.5f);
        std::vector<float> out = in;
        module_test::render (m, out);

        (variant == 0 ? without : withAuto) = dyntests::steadyGain (out, in, 40000, 48000);

        // The meter is gain reduction with makeup EXCLUDED, so it reads the
        // same -8.227 dB whether the makeup is there or not.
        CHECK (std::fabs (m.getMeterDb() + 8.227f) <= 0.3f);
    }

    CHECK (std::fabs (without - 0.38785f) <= 0.008f);
    CHECK (std::fabs (withAuto - 0.91955f) <= 0.020f);
    CHECK (std::fabs (withAuto / without - 2.37137f) <= 0.020f);      // +7.5 dB
}

static void testDynamicsLookaheadAndDetectorDelay()
{
    using namespace spatcore::effects;
    ChainConfig cfg = module_test::config();

    // Lookahead delays the AUDIO by exactly what it reports as latency.
    {
        DynamicsModule m;
        m.prepare (cfg);
        EffectChannelParams p = dyntests::base();
        p.dyn[0].lookaheadMs = 1.0f;                 // 48 samples at 48 kHz
        m.applyParams (p, 0);
        CHECK (m.getLatencySamples() == 48);

        std::vector<float> buf (128, 0.0f);
        buf[0] = 1.0f;
        module_test::render (m, buf);

        CHECK (buf[48] == 1.0f);                     // whole samples, so no interpolation
        float elsewhere = 0.0f;
        for (int i = 0; i < 128; ++i)
            if (i != 48)
                elsewhere += std::fabs (buf[(size_t) i]);
        CHECK (elsewhere == 0.0f);
    }

    // The detector delay delays the DETECTOR and costs no latency. With 5 ms of
    // it the first 240 samples of a burst come out at exactly unity, because
    // the detector is still reading the silence in front of the burst, and the
    // reduction lands behind them. Without it the burst is crushed from the
    // first sample.
    //
    // WHERE it lands is the assertion that matters: 5 ms at 48 kHz is 240
    // samples, so the first altered sample is 240 and not 239 or 2400. A
    // "nothing is touched for a while" check alone passes on a detector delay
    // ten times too long, and on one that never engages at all.
    EffectChannelParams p = dyntests::base();
    p.dyn[0].compOn = 1;                p.dyn[0].compThresholdDb = -60.0f;
    p.dyn[0].compRatio = 100.0f;
    p.dyn[0].compAttackMs = 0.05f;      p.dyn[0].compReleaseMs = 5.0f;

    const std::vector<float> in = dyntests::sine (1024, 1000.0f, 0.5f);

    DynamicsModule late;
    late.prepare (cfg);
    p.dyn[0].compDetectorDelayMs = 5.0f;
    late.applyParams (p, 0);
    std::vector<float> delayed = in;
    module_test::render (late, delayed);
    CHECK (late.getLatencySamples() == 0);

    bool untouched = true;
    for (int i = 0; i < 200; ++i)
        untouched = untouched && bitEqualFloat (delayed[(size_t) i], in[(size_t) i]);
    CHECK (untouched);
    CHECK (dyntests::firstAltered (delayed, in) == 241);
    CHECK (dyntests::peak (delayed, 400, 700) < 0.25f * dyntests::peak (in, 400, 700));

    DynamicsModule prompt;
    prompt.prepare (cfg);
    p.dyn[0].compDetectorDelayMs = 0.0f;
    prompt.applyParams (p, 0);
    std::vector<float> immediate = in;
    module_test::render (prompt, immediate);
    CHECK (dyntests::firstAltered (immediate, in) == 1);     // sample 0 is the sine's own zero
    CHECK (dyntests::peak (immediate, 100, 200) < 0.25f * dyntests::peak (in, 100, 200));
}

static void testDynamicsHoldAndReset()
{
    using namespace spatcore::effects;

    // 100 ms of a 1 kHz sine at 0.5, well above the gate's threshold, then
    // 300 ms at 0.001, well below it. Peak detection sees the level cross back
    // under the threshold at every zero crossing, so hold is what keeps the
    // gate open between the very peaks that opened it, and for 50 ms after the
    // signal stops.
    //
    // Three windows, not one: the hold has to EXPIRE as well as hold. 50 ms at
    // 48 kHz is 2400 samples from the drop at 4800, so the gate is wide open
    // through 7199 and must then close. A counter that reloaded instead of
    // decrementing passes both of the first two windows.
    std::vector<float> in = dyntests::sine (19200, 1000.0f, 1.0f);
    for (int i = 0; i < 19200; ++i)
        in[(size_t) i] *= (i < 4800) ? 0.5f : 0.001f;

    float inHold[2] = { 0.0f, 0.0f };
    float atExpiry[2] = { 0.0f, 0.0f };
    float afterExpiry[2] = { 0.0f, 0.0f };

    for (int variant = 0; variant < 2; ++variant)
    {
        DynamicsModule m;
        m.prepare (module_test::config());

        EffectChannelParams p = dyntests::base();
        p.dyn[0].expOn = 1;                 p.dyn[0].expThresholdDb = -40.0f;
        p.dyn[0].expRatio = 4.0f;           p.dyn[0].expRangeDb = -60.0f;
        p.dyn[0].expAttackMs = 5.0f;        p.dyn[0].expReleaseMs = 5.0f;
        p.dyn[0].expHoldMs = (variant == 0) ? 50.0f : 0.0f;
        m.applyParams (p, 0);

        std::vector<float> out = in;
        module_test::render (m, out);
        inHold[variant]      = dyntests::peak (out, 5600, 6000)   / dyntests::peak (in, 5600, 6000);
        atExpiry[variant]    = dyntests::peak (out, 7000, 7150)   / dyntests::peak (in, 7000, 7150);
        afterExpiry[variant] = dyntests::peak (out, 12000, 12400) / dyntests::peak (in, 12000, 12400);
    }

    CHECK (inHold[0] > 0.9f);            // 50 ms of hold: still wide open
    CHECK (atExpiry[0] > 0.9f);          // and still open at 7000, just inside it
    CHECK (afterExpiry[0] < 0.01f);      // expired and shut: the -60 dB range floor
    CHECK (inHold[1] < 0.2f);            // no hold: shut as soon as the level fell
    CHECK (afterExpiry[1] < 0.01f);

    // reset() clears the tail. A loud burst leaves the lookahead line loaded
    // and both gain states away from unity; silence fed afterwards must come
    // out as true zeros rather than replaying any of it.
    DynamicsModule m;
    m.prepare (module_test::config());

    EffectChannelParams p = dyntests::base();
    p.dyn[0].compOn = 1;  p.dyn[0].expOn = 1;  p.dyn[0].lookaheadMs = 2.0f;
    m.applyParams (p, 0);

    std::vector<float> burst = dyntests::sine (2048, 1000.0f, 0.9f);
    module_test::render (m, burst);
    m.reset();

    std::vector<float> silence (512, 0.0f);
    module_test::render (m, silence);

    bool allZero = true;
    for (float v : silence)
        allZero = allZero && (v == 0.0f);
    CHECK (allZero);

    // Stronger than a list of members: a module dirtied through every piece of
    // state it owns - RMS detector history, a 25 ms detector delay line, a 4 ms
    // lookahead line, a knee, a hold counter and an auto makeup smoother - must
    // render bit-identically to one that was only ever prepared. Anything reset
    // forgets to clear shows up here without having to be named.
    {
        EffectChannelParams dirty = dyntests::base();
        dirty.dyn[0].compOn = 1;                dirty.dyn[0].expOn = 1;
        dirty.dyn[0].detector = 1;              dirty.dyn[0].autoMakeup = 1;
        dirty.dyn[0].compDetectorDelayMs = 25.0f;
        dirty.dyn[0].lookaheadMs = 4.0f;        dirty.dyn[0].compKneeDb = 9.0f;
        dirty.dyn[0].expHoldMs = 250.0f;        dirty.dyn[0].makeupDb = 6.0f;

        DynamicsModule used, fresh;
        used.prepare (module_test::config());
        fresh.prepare (module_test::config());
        used.applyParams (dirty, 0);
        fresh.applyParams (dirty, 0);

        std::vector<float> loud = dyntests::sine (4000, 130.0f, 0.9f);
        module_test::render (used, loud);
        used.reset();

        std::vector<float> a = eqtests::makeAwkwardSignal (1024);
        std::vector<float> b = a;
        module_test::render (used, a);
        module_test::render (fresh, b);
        CHECK (eqtests::bitEqualBlock (a, b));
    }
}

static void testDynamicsVariantAndLatency()
{
    using namespace spatcore::effects;

    // The variant contract is this module's only conversation with ModuleSlot
    // beyond bypass, and all four of its clauses are separately breakable.
    DynamicsModule m;
    m.prepare (module_test::config());

    EffectChannelParams p = dyntests::base();
    p.dyn[0].lookaheadMs = 1.0f;

    // 1. The first apply after prepare SNAPS: no variant is pending and the
    //    latency is already the new one.
    CHECK (m.applyParams (p, 0).variantPending == false);
    CHECK (m.getLatencySamples() == 48);

    // 2. A later change is a pending STATE: the old lookahead keeps running and
    //    the reported latency does not move yet.
    p.dyn[0].lookaheadMs = 3.0f;
    CHECK (m.applyParams (p, 0).variantPending == true);
    CHECK (m.getLatencySamples() == 48);

    // 3. A state, not an edge: handed the same value again it still says yes.
    CHECK (m.applyParams (p, 0).variantPending == true);
    CHECK (m.getLatencySamples() == 48);

    // 4. Revoked before the slot got to silence: it cancels itself, with no
    //    reset and nothing for the slot to do.
    p.dyn[0].lookaheadMs = 1.0f;
    CHECK (m.applyParams (p, 0).variantPending == false);
    CHECK (m.getLatencySamples() == 48);

    // 5. The latency moves at the COMMIT, which is where the slot has faded out
    //    and called reset() first.
    p.dyn[0].lookaheadMs = 3.0f;
    CHECK (m.applyParams (p, 0).variantPending == true);
    m.reset();
    m.commitPendingVariant();
    CHECK (m.getLatencySamples() == 144);
    CHECK (m.applyParams (p, 0).variantPending == false);

    // 6. A detector switch is a variant too, and costs no latency.
    p.dyn[0].detector = 1;
    CHECK (m.applyParams (p, 0).variantPending == true);
    CHECK (m.getLatencySamples() == 144);
    m.reset();
    m.commitPendingVariant();
    CHECK (m.getLatencySamples() == 144);
    CHECK (m.applyParams (p, 0).variantPending == false);

    // And the slot agrees with the module about all of it.
    {
        ModuleSlot slot;
        slot.prepare (module_test::config(), std::make_unique<DynamicsModule>());
        EffectChannelParams q = dyntests::base();
        q.dyn[0].lookaheadMs = 2.0f;
        slot.applyParams (q, 0);
        CHECK (slot.getLatencySamples() == 96);

        q.dyn[0].bypass = 1;
        slot.applyParams (q, 0);
        CHECK (slot.getLatencySamples() == 0);      // a bypassed slot reports none
    }
}

static void testDynamicsSampleRates()
{
    using namespace spatcore::effects;

    // Everything else here runs at 48 kHz, which is how a sidechain high cut
    // that leaves the unit circle below 40 kHz reached the shipped defaults:
    // OutputEQBiquadFilter clamps frequency to 20..20000 Hz with no Nyquist
    // guard, so the default 20 kHz low pass designs w0 > pi at 32 kHz and
    // 22.05 kHz and the detector chain diverges to NaN. The audio never passes
    // through those filters, so the slot's NaN trap does not fire - what
    // happens instead is that both gain computers read kMinDb and the meter
    // reports a reduction with no relation to the signal (-14.2 dB at
    // 22.05 kHz, -9.7 at 32 kHz, against a true -8.98).
    //
    // So: the SHIPPED defaults, the same tone, at every rate a device may
    // offer. The answer must be the same one everywhere.
    const double rates[] = { 22050.0, 32000.0, 40000.0, 44100.0, 48000.0, 88200.0, 96000.0 };

    for (double sr : rates)
    {
        DynamicsModule m;
        m.prepare (module_test::config (sr));

        EffectChannelParams p;                  // SHIPPED defaults, not base()
        p.dyn[0].bypass = 0;
        p.dyn[0].compOn = 1;
        p.dyn[0].expOn = 1;                     // both sidechains, both high cuts
        p.dyn[0].lookaheadMs = 0.0f;            // so out[i]/in[i] IS the gain
        m.applyParams (p, 0);

        const int n = (int) (sr / 2.0);
        const std::vector<float> in = dyntests::sine (n, 1000.0f, 0.5f, (float) sr);
        std::vector<float> out = in;
        module_test::render (m, out);

        bool finite = true;
        for (float v : out)
            finite = finite && std::isfinite (v);

        CHECK (finite);
        CHECK (std::fabs (dyntests::steadyGain (out, in, n - 4000, n) - 0.35617f) <= 0.008f);
        CHECK (std::fabs (m.getMeterDb() + 8.97f) <= 0.3f);
    }

    // The compressor law itself, at the two rates the plan's latency table
    // cares about besides 48 kHz. The detector window is a time, not a sample
    // count, so the settled answer is the same constant with the same
    // tolerance; only the ripple around it changes.
    for (double sr : { 44100.0, 96000.0 })
    {
        DynamicsModule m;
        m.prepare (module_test::config (sr));

        EffectChannelParams p = dyntests::base();
        p.dyn[0].compOn = 1;                p.dyn[0].detector = 1;
        p.dyn[0].compThresholdDb = -20.0f;  p.dyn[0].compRatio = 4.0f;
        p.dyn[0].compAttackMs = 5.0f;       p.dyn[0].compReleaseMs = 50.0f;
        m.applyParams (p, 0);

        const int n = (int) sr;
        const std::vector<float> in = dyntests::sine (n, 1000.0f, 0.5f, (float) sr);
        std::vector<float> out = in;
        module_test::render (m, out);

        CHECK (std::fabs (dyntests::steadyGain (out, in, n - 8000, n) - 0.38785f) <= 0.008f);
        CHECK (std::fabs (m.getMeterDb() + 8.227f) <= 0.3f);
    }

    // Lookahead and detector delay are times too, so their sample counts scale.
    {
        DynamicsModule m;
        m.prepare (module_test::config (96000.0));
        EffectChannelParams p = dyntests::base();
        p.dyn[0].lookaheadMs = 1.0f;
        m.applyParams (p, 0);
        CHECK (m.getLatencySamples() == 96);

        std::vector<float> buf (256, 0.0f);
        buf[0] = 1.0f;
        module_test::render (m, buf);
        CHECK (buf[96] == 1.0f);
    }
}

static void testDynamicsExtremesAndDeterminism()
{
    using namespace spatcore::effects;
    ChainConfig cfg = module_test::config();

    // Every control at an end stop, both detectors, auto makeup on and off, and
    // a pass with NaNs in four of them: the negated clamps must land those on
    // their low bound instead of letting a NaN into the audio. The compressor's
    // sidechain is crossed over on purpose, so its detector sees almost
    // nothing while the expander's sees everything.
    //
    // The ceiling is arithmetic, not a round number: makeAwkwardSignal peaks at
    // 1e7, both gains are at most 1, and the largest makeup reachable here is
    // 24 dB by hand plus -T*(1 - 1/R)/2 = 29.7 dB of auto makeup at T = -60,
    // R = 100, i.e. 53.7 dB = x484. So 4.84e9, and anything materially above it
    // is a sign error rather than a rounding one.
    for (int variant = 0; variant < 4; ++variant)
    {
        EffectChannelParams p = dyntests::base();
        DynamicsParams& d = p.dyn[0];
        d.compOn = d.expOn = 1;
        d.detector = (std::uint8_t) (variant & 1);
        d.autoMakeup = (std::uint8_t) ((variant >> 1) & 1);
        d.lookaheadMs = 5.0f;   d.makeupDb = 24.0f;
        d.compThresholdDb = -60.0f;  d.compRatio = 100.0f;  d.compKneeDb = 24.0f;
        d.compAttackMs = 0.05f;      d.compReleaseMs = 2000.0f;
        d.compDetectorDelayMs = 50.0f;
        d.compScLoCutHz = 2000.0f;   d.compScHiCutHz = 1000.0f;
        d.expThresholdDb = 0.0f;     d.expRatio = 100.0f;   d.expRangeDb = -80.0f;
        d.expAttackMs = 200.0f;      d.expReleaseMs = 5.0f; d.expHoldMs = 500.0f;

        if (variant == 3)
        {
            const float notANumber = std::numeric_limits<float>::quiet_NaN();
            d.compThresholdDb = d.compRatio = d.expRangeDb = d.lookaheadMs = notANumber;
        }

        DynamicsModule m;
        m.prepare (cfg);
        m.applyParams (p, 0);

        std::vector<float> buf = eqtests::makeAwkwardSignal (2048);
        module_test::render (m, buf);

        bool sane = true;
        for (float v : buf)
            sane = sane && std::isfinite (v) && std::fabs (v) < 5.0e9f;
        CHECK (sane);

        // Strictly negative, not merely non-positive: these settings DO reduce,
        // so a meter stuck at 0 - or at any constant - fails here. The derived
        // meter values live in the law tests above.
        CHECK (std::isfinite (m.getMeterDb()));
        CHECK (m.getMeterDb() < 0.0f);
        CHECK (m.getMeterDb() >= spatcore::dsp::FastDecibels::kMinDb);

        if (variant == 3)
            CHECK (m.getLatencySamples() == 0);      // a NaN lookahead clamps to none
    }

    // Determinism: the second chain slot with the same numbers is bit-identical
    // to the first, and the same module repeats itself exactly after a reset.
    EffectChannelParams p = dyntests::base();
    p.dyn[0].compOn = 1;                    p.dyn[0].expOn = 1;
    p.dyn[0].compThresholdDb = -30.0f;      p.dyn[0].compDetectorDelayMs = 3.0f;
    p.dyn[0].lookaheadMs = 1.0f;            p.dyn[0].makeupDb = 3.0f;
    p.dyn[1] = p.dyn[0];

    DynamicsModule first, second;
    first.prepare (cfg);
    second.prepare (cfg);
    first.applyParams (p, 0);
    second.applyParams (p, 1);

    std::vector<float> a = module_test::awkwardBlock (1024, 7);
    std::vector<float> b = a;
    module_test::render (first, a);
    module_test::render (second, b);
    CHECK (eqtests::bitEqualBlock (a, b));

    first.reset();
    std::vector<float> again = module_test::awkwardBlock (1024, 7);
    module_test::render (first, again);
    CHECK (eqtests::bitEqualBlock (a, again));

    // Block size must not be audible: one long call and many short ones give
    // the same samples, which is what lets the slot chunk at maxBlock.
    DynamicsModule whole, chunked;
    whole.prepare (cfg);
    chunked.prepare (cfg);
    whole.applyParams (p, 0);
    chunked.applyParams (p, 0);

    std::vector<float> longCall = module_test::awkwardBlock (4096, 11);
    std::vector<float> shortCalls = longCall;
    whole.process (longCall.data(), 4096);
    for (int offset = 0; offset < 4096; offset += 64)
        chunked.process (shortCalls.data() + offset, 64);
    CHECK (eqtests::bitEqualBlock (longCall, shortCalls));
}

//==============================================================================
// effects/modules - Chorus / Flanger (FxMod)
//==============================================================================

//==============================================================================
// effects/modules/ModulationModule - chorus and flanger.
//==============================================================================

namespace modtests
{
    using namespace spatcore::effects;

    /** One voice, no depth, no feedback and the slowest rate the range allows,
        so any test that wants a STATIC delay gets one. */
    inline EffectChannelParams basic()
    {
        EffectChannelParams p;
        p.mod.bypass = 0;   p.mod.mode = 0;       p.mod.voices = 1;
        p.mod.shape = 1;    p.mod.throughZero = 0;
        p.mod.rateHz = 0.05f;   p.mod.depth = 0.0f;     p.mod.delayMs = 15.0f;
        p.mod.feedback = 0.0f;  p.mod.phaseDeg = 0.0f;  p.mod.loCutHz = 20.0f;
        p.mod.mix = 50.0f;
        return p;
    }

    inline std::vector<float> impulse (int n)
    {
        std::vector<float> v ((size_t) n, 0.0f);
        v[0] = 1.0f;
        return v;
    }

    inline std::vector<float> sine (int n, float hz, double sr = 48000.0)
    {
        std::vector<float> v ((size_t) n);
        for (int i = 0; i < n; ++i)
            v[(size_t) i] = std::sin (6.2831853071795864f * hz * (float) i / (float) sr);
        return v;
    }

    inline int peakIndex (const std::vector<float>& v)
    {
        int best = 0;
        for (size_t i = 0; i < v.size(); ++i)
            if (std::fabs (v[i]) > std::fabs (v[(size_t) best])) best = (int) i;
        return best;
    }

    /** The SIGNED sample of largest magnitude in [from, to). */
    inline float signedPeak (const std::vector<float>& v, int from, int to)
    {
        float best = 0.0f;
        for (int i = from; i < to; ++i)
            if (std::fabs (v[(size_t) i]) > std::fabs (best)) best = v[(size_t) i];
        return best;
    }

    inline float rmsTail (const std::vector<float>& v, int from)
    {
        double acc = 0.0;
        for (size_t i = (size_t) from; i < v.size(); ++i) acc += (double) v[i] * (double) v[i];
        return (float) std::sqrt (acc / (double) ((int) v.size() - from));
    }

    /** The steepest sample-to-sample move in [from, end) - the measure a click
        shows up in and a sweep does not. */
    inline float worstStep (const std::vector<float>& v, int from)
    {
        float worst = 0.0f;
        for (size_t i = (size_t) (from < 1 ? 1 : from); i < v.size(); ++i)
            worst = std::max (worst, std::fabs (v[i] - v[i - 1]));
        return worst;
    }

    inline bool bounded (const std::vector<float>& v, float limit)
    {
        for (float s : v) if (! std::isfinite (s) || std::fabs (s) > limit) return false;
        return true;
    }

    inline float dbRatio (float a, float b)
    {
        return 20.0f * std::log10 (a / b);
    }
}

static void testModulationTransparency()
{
    using namespace spatcore::effects;
    // Bypassed at defaults, through a real slot: the buffer is not touched.
    {
        ModuleSlot slot;
        slot.prepare (module_test::config (48000.0, 256), std::make_unique<ModulationModule>());
        slot.applyParams (EffectChannelParams(), 0);
        CHECK (slot.isBypassedSettled());
        for (int block = 0; block < 6; ++block)
        {
            std::vector<float> buf = (block == 0) ? eqtests::makeAwkwardSignal (256)
                                                  : module_test::awkwardBlock (256, block);
            const std::vector<float> reference = buf;
            slot.process (buf.data(), 256);
            CHECK (eqtests::bitEqualBlock (buf, reference));
        }
    }
    // ACTIVE at mix 0 with everything else at an extreme. Bit-identical rather
    // than merely close, because the module leaves the block alone instead of
    // crossfading against a wet leg scaled to zero - a crossfade at g = 0 still
    // turns the negative zeros in the signal positive. Through-zero is OFF
    // here, and that is the whole condition: with it on the dry leg is itself a
    // delay and there is no transparent answer to give, which is what
    // testModulationThroughZeroAtMixZero pins. Latency must agree - a module
    // handing back the block it was given cannot be claiming 10 ms.
    {
        ModulationModule m;
        m.prepare (module_test::config (48000.0, 256));
        EffectChannelParams p = modtests::basic();
        p.mod.mix = 0.0f;       p.mod.depth = 100.0f;   p.mod.feedback = 95.0f;
        p.mod.voices = 3;       p.mod.rateHz = 10.0f;   p.mod.throughZero = 0;
        p.mod.delayMs = 10.0f;
        CHECK (! m.applyParams (p, 0).bypass);
        CHECK (m.getLatencySamples() == 0);
        for (int block = 0; block < 4; ++block)
        {
            std::vector<float> buf = (block == 0) ? eqtests::makeAwkwardSignal (256)
                                                  : module_test::awkwardBlock (256, 40 + block);
            const std::vector<float> reference = buf;
            module_test::render (m, buf);
            CHECK (eqtests::bitEqualBlock (buf, reference));
        }
    }
}

static void testModulationThroughZeroAtMixZero()
{
    using namespace spatcore::effects;
    // Through-zero delays the DRY leg, so at mix 0 the honest output is the
    // delayed dry and not the input. An early-out that handed the block back
    // untouched would swap x[n-480] for x[n] the instant the mix settled - a
    // step of up to twice the signal amplitude, and a channel that jumps 10 ms
    // forward in time. The dry leg is raw, so the impulse arrives at full
    // height: the low cut is in the wet path only.
    {
        ModulationModule m;
        m.prepare (module_test::config (48000.0, 256));
        EffectChannelParams p = modtests::basic();
        p.mod.delayMs = 10.0f;  p.mod.throughZero = 1;  p.mod.mix = 0.0f;
        m.applyParams (p, 0);
        CHECK (m.getLatencySamples() == 480);          // and the audio had better match
        std::vector<float> buf = modtests::impulse (2048);
        module_test::render (m, buf);
        CHECK (modtests::peakIndex (buf) == 480);
        CHECK (buf[480] == 1.0f);                      // exactly: an unfiltered whole-sample read
        CHECK (buf[0] == 0.0f);
    }
    // And riding the mix down to zero must not step. The yardstick is the
    // input's own steepest move: the discontinuity the early-out used to
    // produce here measured twenty-odd times that.
    {
        const int block = 512, blocks = 40;
        ModulationModule m;
        m.prepare (module_test::config (48000.0, block));
        EffectChannelParams p = modtests::basic();
        p.mod.delayMs = 10.0f;  p.mod.throughZero = 1;  p.mod.mix = 50.0f;
        m.applyParams (p, 0);

        const std::vector<float> in = modtests::sine (block * blocks, 220.0f);
        std::vector<float> out;
        EffectChannelParams zero = p;
        zero.mod.mix = 0.0f;

        for (int b = 0; b < blocks; ++b)
        {
            if (b == 10)
                m.applyParams (zero, 0);               // the fader arrives at 0

            std::vector<float> buf (in.begin() + b * block, in.begin() + (b + 1) * block);
            module_test::render (m, buf);
            out.insert (out.end(), buf.begin(), buf.end());
        }

        // From the edit onwards: the glide, the settle and the long tail after
        // it are all one continuous signal.
        CHECK (modtests::worstStep (out, 11 * block) < 2.0f * modtests::worstStep (in, 1));
    }
}

static void testModulationDelayLaw()
{
    using namespace spatcore::effects;
    // 10 ms at 48 kHz is exactly 480 samples. The law is t = centre*(1 + depth*lfo)
    // and the Sine shape is -cos, so it is -1 at phase 0 and +1 at half a cycle:
    // 480*(1 - 0.5) = 240 at 0 degrees and 480*(1 + 0.5) = 720 at 180. The rate is
    // the slowest the range allows, so the LFO moves less than a thousandth of a
    // cycle inside the buffer. This pins the multiplicative law AND the sign
    // convention: an ADDITIVE depth, or a cosine LFO, misses both numbers.
    const int expected[2] = { 240, 720 };
    const float phases[2] = { 0.0f, 180.0f };
    for (int k = 0; k < 2; ++k)
    {
        ModulationModule m;
        m.prepare (module_test::config());
        EffectChannelParams p = modtests::basic();
        p.mod.delayMs = 10.0f;  p.mod.depth = 50.0f;
        p.mod.phaseDeg = phases[k];     p.mod.mix = 100.0f;
        m.applyParams (p, 0);
        std::vector<float> buf = modtests::impulse (2048);
        module_test::render (m, buf);
        CHECK (modtests::peakIndex (buf) == expected[k]);
        CHECK (buf[(size_t) expected[k]] > 0.9f);      // the 20 Hz low cut costs ~0.2 %
    }
    // Phase 90 is where this module's LFO meets the prototype's. The shape id is
    // a LFOWaveforms one, as the plan's table asks, and LFOWaveforms::Sine is
    // -cos: the module starts at the SHORTEST delay where fx_flanger.gendsp's
    // phasor -> sin starts at the CENTRE. A quarter cycle in, -cos is zero and
    // the modulated time is the centre time - 2 ms, i.e. 96 samples at 48 kHz.
    // The delay is kept short here on purpose: at phase 90 the LFO is at its
    // steepest, and a 10 ms centre would have drifted three quarters of a sample
    // while the impulse was in flight.
    {
        ModulationModule m;
        m.prepare (module_test::config());
        EffectChannelParams p = modtests::basic();
        p.mod.delayMs = 2.0f;   p.mod.depth = 50.0f;
        p.mod.phaseDeg = 90.0f; p.mod.mix = 100.0f;
        m.applyParams (p, 0);
        std::vector<float> buf = modtests::impulse (1024);
        module_test::render (m, buf);
        CHECK (modtests::peakIndex (buf) == 96);
    }
    // MILLISECONDS, not samples. The prototype's delaytime inlet is in samples
    // (there is no mstosamps anywhere in fx_flanger.gendsp), so this conversion
    // is a deliberate deviation and the rest of the suite runs at 48 kHz only -
    // nothing else here would catch it regressing.
    {
        const double rates[3] = { 44100.0, 96000.0, 192000.0 };
        const int expectedAt[3] = { 441, 960, 1920 };
        for (int k = 0; k < 3; ++k)
        {
            ModulationModule m;
            m.prepare (module_test::config (rates[k]));
            EffectChannelParams p = modtests::basic();
            p.mod.delayMs = 10.0f;  p.mod.mix = 100.0f;
            m.applyParams (p, 0);
            std::vector<float> buf = modtests::impulse (4096);
            module_test::render (m, buf);
            CHECK (modtests::peakIndex (buf) == expectedAt[k]);
        }
    }
    // The comb. 0.25 ms at 48 kHz is exactly 12 samples, so at mix 50 the output
    // is 0.5*x[n] + 0.5*x[n-12]; 12 samples is half a period of 2 kHz, where the
    // legs cancel, and a whole period of 4 kHz, where they add. The null is about
    // -40 dB rather than zero because the 20 Hz low cut turns the wet leg by
    // roughly a degree at 2 kHz, so this is a tolerance, not bit-equality.
    for (int k = 0; k < 2; ++k)
    {
        ModulationModule m;
        m.prepare (module_test::config());
        EffectChannelParams p = modtests::basic();
        p.mod.delayMs = 0.25f;
        m.applyParams (p, 0);
        std::vector<float> buf = modtests::sine (4096, (k == 0) ? 2000.0f : 4000.0f);
        const float inRms = modtests::rmsTail (buf, 2048);
        module_test::render (m, buf);
        const float outRms = modtests::rmsTail (buf, 2048);
        CHECK ((k == 0) ? (outRms < 0.05f * inRms) : (outRms > 0.95f * inRms));
    }
}

static void testModulationLoCutPlacement()
{
    using namespace spatcore::effects;
    // The low cut sits in the FORWARD path only, as wetDry.gendsp wires it: the
    // dry leg is the raw input. A 100 Hz sine under a 2 kHz low cut therefore
    // comes back at full level at a mix of almost nothing - and collapses at
    // mix 100, where only the filtered leg is left. A port that filtered the dry
    // leg too would pass no test in this file except this one.
    {
        ModulationModule m;
        m.prepare (module_test::config());
        EffectChannelParams p = modtests::basic();
        p.mod.loCutHz = 2000.0f;    p.mod.mix = 0.001f;    // not zero: the module must RUN
        m.applyParams (p, 0);
        std::vector<float> buf = modtests::sine (8192, 100.0f);
        const float inRms = modtests::rmsTail (buf, 4096);
        module_test::render (m, buf);
        CHECK (std::fabs (modtests::rmsTail (buf, 4096) / inRms - 1.0f) < 0.01f);
    }
    {
        ModulationModule m;
        m.prepare (module_test::config());
        EffectChannelParams p = modtests::basic();
        p.mod.loCutHz = 2000.0f;    p.mod.mix = 100.0f;
        m.applyParams (p, 0);
        std::vector<float> buf = modtests::sine (8192, 100.0f);
        const float inRms = modtests::rmsTail (buf, 4096);
        module_test::render (m, buf);
        CHECK (modtests::rmsTail (buf, 4096) < 0.05f * inRms);
    }
}

static void testModulationVoiceLevelLaw()
{
    using namespace spatcore::effects;
    // The voice sum is scaled by 1/sqrt(voices), not by the voice count, and
    // this is the test that says why. Voices are decorrelated - that is what the
    // 120 degree offsets are for - so they sum in POWER: on broadband material
    // at the plan's default depth, three taps are sqrt(3) louder than one, and
    // 1/sqrt(3) is what puts the channel back where it was. Under 1/voices the
    // same measurement reads -3.0 dB and -4.8 dB, i.e. the module quietly drops
    // the channel by most of a fader for the crime of adding a voice.
    float rms[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int voices = 1; voices <= 3; ++voices)
    {
        ModulationModule m;
        m.prepare (module_test::config());
        EffectChannelParams p = modtests::basic();
        p.mod.voices = (std::uint8_t) voices;
        p.mod.depth = 50.0f;    p.mod.rateHz = 1.0f;    p.mod.mix = 100.0f;
        m.applyParams (p, 0);
        std::vector<float> buf = module_test::awkwardBlock (48000, 5);
        module_test::render (m, buf);
        rms[voices] = modtests::rmsTail (buf, 4800);    // past the fill of the line
    }
    CHECK (std::fabs (modtests::dbRatio (rms[2], rms[1])) < 1.0f);
    CHECK (std::fabs (modtests::dbRatio (rms[3], rms[1])) < 1.0f);

    // The corner the law does NOT hold, pinned deliberately so that nobody
    // "fixes" it by accident: at depth zero the voices read the same sample and
    // really are +3 / +4.8 dB. No constant serves both cases - how far apart the
    // taps sit is depth's business - and depth zero is the setting that asks
    // three voices to be one.
    float flat[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int voices = 1; voices <= 3; ++voices)
    {
        ModulationModule m;
        m.prepare (module_test::config());
        EffectChannelParams p = modtests::basic();
        p.mod.voices = (std::uint8_t) voices;
        p.mod.depth = 0.0f;     p.mod.mix = 100.0f;
        m.applyParams (p, 0);
        std::vector<float> buf = module_test::awkwardBlock (24000, 6);
        module_test::render (m, buf);
        flat[voices] = modtests::rmsTail (buf, 4800);
    }
    CHECK (std::fabs (modtests::dbRatio (flat[2], flat[1]) - 3.0103f) < 0.05f);
    CHECK (std::fabs (modtests::dbRatio (flat[3], flat[1]) - 4.7712f) < 0.05f);
}

static void testModulationFeedbackAndReset()
{
    using namespace spatcore::effects;
    // 1 ms at 48 kHz = 48 samples. The impulse leaves at 48; what it feeds back
    // re-enters the line on the next sample and comes round again near 98, and
    // once more near 148, each generation scaled by the signed feedback. The
    // loop's damping filter spreads every repeat over a few samples, hence
    // windows rather than exact indices.
    for (int k = 0; k < 2; ++k)
    {
        ModulationModule m;
        m.prepare (module_test::config());
        EffectChannelParams p = modtests::basic();
        p.mod.delayMs = 1.0f;   p.mod.mix = 100.0f;
        p.mod.feedback = (k == 0) ? 50.0f : -50.0f;
        m.applyParams (p, 0);
        std::vector<float> buf = modtests::impulse (512);
        module_test::render (m, buf);
        const float first  = modtests::signedPeak (buf, 44, 60);
        const float second = modtests::signedPeak (buf, 92, 112);
        const float third  = modtests::signedPeak (buf, 140, 164);
        CHECK (first > 0.9f);
        CHECK (std::fabs (second) > 0.1f && std::fabs (second) < first);
        CHECK (std::fabs (third) < std::fabs (second));        // the loop loses, never gains
        CHECK ((k == 0) ? (second > 0.0f) : (second < 0.0f));  // the feedback is signed
    }
    // reset() runs when a slot has faded to silence, so no tail may survive it -
    // not a quiet one, not a denormal one.
    {
        ModulationModule m;
        m.prepare (module_test::config());
        EffectChannelParams p = modtests::basic();
        p.mod.delayMs = 5.0f;   p.mod.depth = 40.0f;   p.mod.rateHz = 3.0f;
        p.mod.feedback = 95.0f; p.mod.mix = 100.0f;
        m.applyParams (p, 0);
        std::vector<float> excite = module_test::awkwardBlock (2048, 7);
        module_test::render (m, excite);
        CHECK (modtests::rmsTail (excite, 1024) > 1.0e-3f);    // it really was ringing
        m.reset();
        std::vector<float> quiet (4096, 0.0f);
        module_test::render (m, quiet);
        bool silent = true;
        for (float s : quiet) if (s != 0.0f) silent = false;
        CHECK (silent);
    }
}

static void testModulationDenormalFlush()
{
    using namespace spatcore::effects;
    // The block-end flush that keeps a silent channel out of denormal land must
    // not be able to cut a LIVE signal off. The case that gets it wrong is an
    // offset input that stops: the 20 Hz low cut then emits a step response that
    // decays for about a third of a second while the input is an exact zero and
    // the tap - still a whole delay behind, in the settled DC region - is an
    // exact zero too. Watching only those two, the flush fires and truncates the
    // decay to 0.0 in one sample. The artefact lands at stepSample + delay +
    // blockSize, so it gets LOUDER as the buffer gets shorter, which is why this
    // runs at two block sizes.
    const int blockSizes[2] = { 512, 64 };
    for (int k = 0; k < 2; ++k)
    {
        const int block = blockSizes[k];
        ModulationModule m;
        m.prepare (module_test::config (48000.0, block));
        EffectChannelParams p = modtests::basic();
        p.mod.delayMs = 30.0f;  p.mod.mix = 100.0f;    // 1440 samples of tap lag
        m.applyParams (p, 0);

        std::vector<float> out;
        for (int b = 0; b < 48000 / block; ++b)
        {
            std::vector<float> buf ((size_t) block, 0.5f);
            module_test::render (m, buf);
            out.insert (out.end(), buf.begin(), buf.end());
        }
        const int stepAt = (int) out.size();
        for (int b = 0; b < 48000 / block; ++b)
        {
            std::vector<float> buf ((size_t) block, 0.0f);
            module_test::render (m, buf);
            out.insert (out.end(), buf.begin(), buf.end());
        }

        // The genuine edge arrives at stepAt + 1440; everything after it is the
        // low cut's own decay, which moves by about 0.002 per sample.
        CHECK (modtests::worstStep (out, stepAt + 1445) < 0.01f);
        CHECK (std::fabs (out[(size_t) stepAt + 1600]) > 0.01f);   // and it is still running
    }
}

static void testModulationVariants()
{
    using namespace spatcore::effects;
    // A voice-count change waits for silence and the module keeps running the
    // OLD count meanwhile, so an edit taken back before the fade completed was
    // never heard. Depth is non-zero because three voices reading one static
    // delay would sum to the same sample and hide the difference. Mode rides
    // along to show it is neither a variant nor audible.
    {
        ModulationModule a, b;
        a.prepare (module_test::config());
        b.prepare (module_test::config());
        EffectChannelParams p = modtests::basic();
        p.mod.depth = 50.0f;  p.mod.rateHz = 2.0f;  p.mod.mix = 100.0f;
        a.applyParams (p, 0);
        b.applyParams (p, 0);
        EffectChannelParams three = p;
        three.mod.voices = 3;
        three.mod.mode = 1;                                    // flanger
        CHECK (b.applyParams (three, 0).variantPending);
        EffectChannelParams modeOnly = p;
        modeOnly.mod.mode = 1;
        CHECK (! a.applyParams (modeOnly, 0).variantPending);
        std::vector<float> held = module_test::awkwardBlock (1024, 3), heldB = held;
        module_test::render (a, held);
        module_test::render (b, heldB);
        CHECK (eqtests::bitEqualBlock (held, heldB));          // one voice both, mode ignored
        b.commitPendingVariant();
        CHECK (! b.applyParams (three, 0).variantPending);     // a state, not an edge
        std::vector<float> after = module_test::awkwardBlock (1024, 4), afterB = after;
        module_test::render (a, after);
        module_test::render (b, afterB);
        CHECK (! eqtests::bitEqualBlock (after, afterB));
    }
    // Through-zero delays the dry leg by the centre delay, which is latency and
    // has to be reported. At 10 ms / 48 kHz BOTH legs are delayed by 480, so an
    // impulse appears there and nowhere earlier.
    {
        ModulationModule m;
        m.prepare (module_test::config());
        EffectChannelParams p = modtests::basic();
        p.mod.delayMs = 10.0f;  p.mod.throughZero = 1;
        m.applyParams (p, 0);                                  // the first apply snaps
        CHECK (m.getLatencySamples() == 480);
        std::vector<float> buf = modtests::impulse (2048);
        module_test::render (m, buf);
        CHECK (modtests::peakIndex (buf) == 480);
        CHECK (buf[480] > 0.9f && std::fabs (buf[479]) < 0.05f);
        EffectChannelParams off = p;
        off.mod.throughZero = 0;
        CHECK (m.applyParams (off, 0).variantPending);
        CHECK (m.getLatencySamples() == 480);                  // still the running topology
        m.commitPendingVariant();
        CHECK (m.getLatencySamples() == 0);
    }
}

static void testModulationExtremes()
{
    using namespace spatcore::effects;
    // Every shape at the worst of everything else. A comb at 0.95 feedback tops
    // out near 1/(1 - 0.95) = 20x its input at resonance and the input peaks at
    // 0.7, so anything past the limit below is a runaway, not a loud day. The
    // shapes are also kept apart from one another: "bounded" alone would pass a
    // module that ignored the shape parameter, or emitted silence.
    std::vector<std::vector<float>> perShape;
    for (int shape = 0; shape <= 8; ++shape)
    {
        ModulationModule m;
        m.prepare (module_test::config (96000.0));
        EffectChannelParams p = modtests::basic();
        p.mod.shape = (std::uint8_t) shape;
        p.mod.rateHz = 10.0f;   p.mod.depth = 100.0f;   p.mod.delayMs = 30.0f;
        p.mod.feedback = 95.0f; p.mod.voices = 3;       p.mod.throughZero = 1;
        p.mod.loCutHz = 2000.0f;    p.mod.mix = 100.0f;
        m.applyParams (p, 0);
        std::vector<float> last;
        for (int block = 0; block < 8; ++block)
        {
            std::vector<float> buf = module_test::awkwardBlock (512, 60 + block);
            module_test::render (m, buf);
            CHECK (modtests::bounded (buf, 100.0f));
            last = buf;
        }
        perShape.push_back (last);
    }
    for (size_t i = 0; i < perShape.size(); ++i)
        for (size_t j = i + 1; j < perShape.size(); ++j)
            CHECK (! eqtests::bitEqualBlock (perShape[i], perShape[j]));

    // Out-of-range and not-a-number parameters land on a bound instead of in the
    // audio - the clamps are negated comparisons, so a NaN takes the low bound.
    {
        ModulationModule m;
        m.prepare (module_test::config());
        EffectChannelParams p = modtests::basic();
        const float nan = std::numeric_limits<float>::quiet_NaN();
        p.mod.rateHz = nan;     p.mod.depth = nan;      p.mod.phaseDeg = nan;
        p.mod.loCutHz = nan;    p.mod.delayMs = 1.0e9f; p.mod.feedback = 500.0f;
        p.mod.voices = 200;     p.mod.shape = 200;      p.mod.mix = 100.0f;
        m.applyParams (p, 0);
        for (int block = 0; block < 8; ++block)
        {
            std::vector<float> buf = module_test::awkwardBlock (512, 90 + block);
            module_test::render (m, buf);
            CHECK (modtests::bounded (buf, 100.0f));
        }
    }
    // Feedback is the one parameter whose LOW bound is the dangerous end, so it
    // does not take the plain clamp: -95 % is maximum INVERTED regeneration, and
    // a NaN arriving there would be the worst setting on the dial rather than
    // the safest. It lands on zero instead, so the impulse leaves once and does
    // not come round.
    {
        ModulationModule m;
        m.prepare (module_test::config());
        EffectChannelParams p = modtests::basic();
        p.mod.delayMs = 1.0f;   p.mod.mix = 100.0f;
        p.mod.feedback = std::numeric_limits<float>::quiet_NaN();
        m.applyParams (p, 0);
        std::vector<float> buf = modtests::impulse (512);
        module_test::render (m, buf);
        CHECK (modtests::signedPeak (buf, 44, 60) > 0.9f);             // the delay still works
        CHECK (std::fabs (modtests::signedPeak (buf, 92, 112)) < 0.02f);
    }
    // mix = NaN is the safe direction of the same rule: it lands on 0 and the
    // module goes bit-transparent rather than writing a NaN into the buffer.
    {
        ModulationModule m;
        m.prepare (module_test::config (48000.0, 256));
        EffectChannelParams p = modtests::basic();
        p.mod.mix = std::numeric_limits<float>::quiet_NaN();
        m.applyParams (p, 0);
        std::vector<float> buf = eqtests::makeAwkwardSignal (256);
        const std::vector<float> reference = buf;
        module_test::render (m, buf);
        CHECK (eqtests::bitEqualBlock (buf, reference));
    }
    // Determinism. The Random shape is the only place the channel's noise key is
    // read, so it doubles as the test that the key reaches the delay trajectory.
    {
        EffectChannelParams p = modtests::basic();
        p.mod.shape = 8;                                       // Random
        p.mod.rateHz = 10.0f;   p.mod.depth = 100.0f;   p.mod.mix = 100.0f;
        ModulationModule ma, mb, mc;
        ma.prepare (module_test::config (48000.0, 512, 5));
        mb.prepare (module_test::config (48000.0, 512, 5));
        mc.prepare (module_test::config (48000.0, 512, 6));
        ma.applyParams (p, 0);  mb.applyParams (p, 0);  mc.applyParams (p, 0);
        std::vector<float> a = module_test::awkwardBlock (4096, 11);
        std::vector<float> b = a, c = a;
        module_test::render (ma, a);
        module_test::render (mb, b);
        module_test::render (mc, c);
        CHECK (eqtests::bitEqualBlock (a, b));                 // same key, same audio
        int differs = 0;
        for (int i = 2048; i < 4096; ++i)
            if (! bitEqualFloat (a[(size_t) i], c[(size_t) i])) ++differs;
        CHECK (differs > 100);
    }
}

//==============================================================================
// effects/modules - Phaser (FxPhaser)
//==============================================================================

//==============================================================================
// effects/modules/PhaserModule
//==============================================================================

namespace phaser_test
{
    using namespace spatcore::effects;

    constexpr double kPi = 3.14159265358979323846;

    /** A STATIC chain: depth 0 makes the sweep factor exp2(0), which
        FastDecibels returns as exactly 1, so every stage sits on the centre
        frequency and the coefficients never move. That is what lets the phase
        laws below be pinned to a number.

        The tests that exercise the SWEEP - testPhaserSweepLaw,
        testPhaserSpreadLaw, testPhaserLfoRateAndShape - set depthOct and
        spreadOct themselves on top of this. Anything that asserts a static
        law must leave them at zero, or the notch it is looking for moves
        while it looks. */
    inline EffectChannelParams active (float centreHz, int stages, float mix)
    {
        EffectChannelParams p;
        p.phaser.bypass = 0;
        p.phaser.stages = static_cast<std::uint8_t> (stages);
        p.phaser.centreHz = centreHz;
        p.phaser.spreadOct = 0.0f;
        p.phaser.depthOct = 0.0f;
        p.phaser.rateHz = 0.3f;
        p.phaser.shape = 1;
        p.phaser.feedback = 0.0f;
        p.phaser.mix = mix;
        return p;
    }

    /** Where N first-order allpasses tuned to fc have turned the signal by half
        a turn, which is where a 50 % mix cancels. One stage turns it by
        -2*atan(tan(pi f / sr) / t) with t = tan(pi fc / sr), so N of them reach
        -pi where tan(pi f / sr) = t * tan(pi / 2N). */
    inline double notchHz (double fc, int stages, double sr)
    {
        const double t = std::tan (kPi * fc / sr);
        return sr / kPi * std::atan (t * std::tan (kPi / (2.0 * stages)));
    }

    /** The turn one allpass tuned to fc gives f, in radians. */
    inline double stagePhase (double f, double fc, double sr)
    {
        return -2.0 * std::atan (std::tan (kPi * f / sr) / std::tan (kPi * fc / sr));
    }

    /** The frequency where a whole chain - stages SPREAD over spreadOct
        octaves around centre, exactly as the module places them - has turned
        the signal by `turns` whole turns.

        `turns` 0.5 is the notch a 50 % mix cancels; `turns` 1 is where the
        chain hands the feedback path the signal back in phase, which is the
        self-oscillation frequency. The total turn falls monotonically from 0
        to -N*pi as f goes to Nyquist, so bisection finds either exactly.

        This is the closed form of notchHz() generalised: the CHECK in
        testPhaserSpreadLaw pins the two against each other at spread 0, so a
        mistake in this helper cannot quietly excuse a mistake in the module. */
    inline double chainPhaseHz (double centre, double spreadOct, int stages, double sr, double turns)
    {
        const double target = -2.0 * kPi * turns;

        auto total = [=] (double f)
        {
            double sum = 0.0;
            for (int k = 0; k < stages; ++k)
            {
                const double offset = 2.0 * static_cast<double> (k)
                                        / (static_cast<double> (stages) - 1.0) - 1.0;
                sum += stagePhase (f, centre * std::pow (2.0, spreadOct * offset), sr);
            }
            return sum;
        };

        double lo = 1.0, hi = 0.45 * sr;
        for (int i = 0; i < 200; ++i)
        {
            const double mid = 0.5 * (lo + hi);
            if (total (mid) > target) lo = mid; else hi = mid;
        }
        return 0.5 * (lo + hi);
    }

    inline std::vector<float> sine (int n, double hz, double sr, double amp)
    {
        std::vector<float> v ((size_t) n);
        for (int i = 0; i < n; ++i)
            v[(size_t) i] = (float) (amp * std::sin (2.0 * kPi * hz * (double) i / sr));
        return v;
    }

    inline float peakFrom (const std::vector<float>& v, int from)
    {
        float peak = 0.0f;
        for (int i = from; i < (int) v.size(); ++i)
            peak = std::max (peak, std::fabs (v[(size_t) i]));
        return peak;
    }

    /** The peak over one window, which is how a moving notch is caught: the
        same tone is deeply cancelled in one window and not in another. */
    inline float peakBetween (const std::vector<float>& v, int from, int to)
    {
        float peak = 0.0f;
        for (int i = from; i < to && i < (int) v.size(); ++i)
            peak = std::max (peak, std::fabs (v[(size_t) i]));
        return peak;
    }

    inline double rmsFrom (const std::vector<float>& v, int from)
    {
        double sum = 0.0;
        for (int i = from; i < (int) v.size(); ++i)
            sum += (double) v[(size_t) i] * (double) v[(size_t) i];
        return std::sqrt (sum / (double) ((int) v.size() - from));
    }

    /** How many times the level dips as the sweep carries the notch across a
        parked tone: a block RMS with hysteresis, counted over `span` samples
        from `from`. One dip per LFO cycle, so this counts LFO cycles from the
        audio alone. */
    inline int countDips (const std::vector<float>& v, int from, int span)
    {
        constexpr int blockSize = 256;
        constexpr double enter = 0.02, leave = 0.05;

        int dips = 0;
        bool below = false;

        for (int i = from; i + blockSize <= from + span; i += blockSize)
        {
            double sum = 0.0;
            for (int j = 0; j < blockSize; ++j)
            {
                const double x = v[(size_t) (i + j)];
                sum += x * x;
            }

            const double rms = std::sqrt (sum / (double) blockSize);

            if (! below && rms < enter)     { below = true; ++dips; }
            else if (below && rms > leave)  { below = false; }
        }

        return dips;
    }

    /** A fresh phaser over a sine, so each phase law costs one line. */
    inline std::vector<float> renderSine (const EffectChannelParams& p, double hz, int n,
                                          double sr = 48000.0)
    {
        PhaserModule m;
        m.prepare (module_test::config (sr, 512));
        m.applyParams (p, 0);

        std::vector<float> v = sine (n, hz, sr, 0.5);
        module_test::render (m, v);
        return v;
    }

    /** One module over one awkward block, so two of them can be compared. */
    inline std::vector<float> renderBlock (IEffectModule& m, int seed)
    {
        std::vector<float> v = module_test::awkwardBlock (256, seed);
        module_test::render (m, v);
        return v;
    }

    /** Peak over `blocks` of hot noise, or a NaN if anything non-finite came
        out - so one upper-bound check covers both runaway and NaN. */
    inline float hotPeak (const EffectChannelParams& p, int blocks)
    {
        PhaserModule m;
        m.prepare (module_test::config (48000.0, 256));
        m.applyParams (p, 0);

        float peak = 0.0f;

        for (int block = 0; block < blocks; ++block)
        {
            std::vector<float> buf = module_test::awkwardBlock (256, block + 1);
            for (auto& s : buf)
                s *= 1.4f;                          // hot, and past full scale

            module_test::render (m, buf);

            for (float s : buf)
            {
                if (! std::isfinite (s))
                    return std::numeric_limits<float>::quiet_NaN();

                peak = std::max (peak, std::fabs (s));
            }
        }

        return peak;
    }
}

static void testPhaserAllpassLaw()
{
    using namespace spatcore::effects;

    const int n = 6000;             // 2400 to settle, then 3600 samples = 75
                                    // whole periods of 1 kHz at 48 kHz
    // Every tolerance here is loose because the coefficients come through
    // std::tan, which differs by an ULP or two between platforms.

    // Four allpasses tuned to 1 kHz each turn a 1 kHz sine by -90 degrees, so
    // the chain hands it back a whole turn later, in phase: a 50 % mix of the
    // two legs is then the input itself.
    //
    // On its own this block is weak - a module that did NOTHING would pass it,
    // since in + 0.5*(in - in) is in. It is the notch check below that gives it
    // teeth, by pinning where the chain does NOT hand the signal back. Neither
    // half is worth keeping without the other.
    {
        const std::vector<float> in = phaser_test::sine (n, 1000.0, 48000.0, 0.5);
        const std::vector<float> out = phaser_test::renderSine (phaser_test::active (1000.0f, 4, 50.0f),
                                                                1000.0, n);
        float worst = 0.0f;
        for (int i = 2400; i < n; ++i)
            worst = std::max (worst, std::fabs (out[(size_t) i] - in[(size_t) i]));
        CHECK (worst < 2.0e-3f);
    }

    // Half a turn happens lower down, at
    // (sr/pi)*atan(tan(pi*1000/sr)*tan(pi/8)) = 414.70 Hz, and there the same
    // 50 % mix cancels instead. Cancelling everywhere would pass the check
    // above and fail this one; cancelling nowhere fails this one alone.
    {
        const double f = phaser_test::notchHz (1000.0, 4, 48000.0);
        CHECK (std::fabs (f - 414.70) < 0.05);

        const std::vector<float> out = phaser_test::renderSine (phaser_test::active (1000.0f, 4, 50.0f),
                                                                f, n);
        CHECK (phaser_test::peakFrom (out, 2400) < 2.0e-3f);    // 0.5 in, -48 dB out

        // The same notch at 96 kHz. The frequency barely moves (414.34 Hz), but
        // the COEFFICIENT does, so a module that had 48 kHz baked into it would
        // put its notch at 828 Hz and fail here.
        const double f96 = phaser_test::notchHz (1000.0, 4, 96000.0);
        CHECK (phaser_test::peakFrom (phaser_test::renderSine (phaser_test::active (1000.0f, 4, 50.0f),
                                                               f96, 2 * n, 96000.0), 4800) < 2.0e-3f);
    }

    // The wet leg on its own is an ALLPASS: it turns the phase and changes
    // nothing else, so a sine keeps its RMS through twelve of them. This is
    // what a coefficient that is not (t-1)/(t+1) has to get past.
    {
        const std::vector<float> in = phaser_test::sine (n, 1000.0, 48000.0, 0.5);
        const std::vector<float> out = phaser_test::renderSine (phaser_test::active (1000.0f, 12, 100.0f),
                                                                1000.0, n);
        CHECK (std::fabs (phaser_test::rmsFrom (out, 2400)
                            - phaser_test::rmsFrom (in, 2400)) < 1.0e-3);
    }
}

static void testPhaserSweepLaw()
{
    using namespace spatcore::effects;

    // The characteristic law: f_k = centre * 2^(depth*lfo). The Sine shape is
    // -cos, so the sweep STARTS at the bottom, 2^-depth, and reaches the top,
    // 2^+depth, half an LFO cycle later. One octave of depth at 1 kHz is
    // therefore 500 Hz at sample 0 and 2000 Hz at sample 240000 (rate 0.1 Hz),
    // and the notch has to be found at each end and nowhere else.
    //
    // A phaser that did not modulate at all, one that ignored depth, and one
    // that applied depth linearly rather than as a power of two all put the
    // notch somewhere this test looks and finds silence (a linear law would
    // reach 750 Hz at lfo = -1, notching at 311 Hz, not 207).

    const double sr = 48000.0;
    const double rate = 0.1;
    const int half = (int) (0.5 / rate * sr);       // samples to lfo = +1
    const int n = half + 8000;

    const double fLow  = phaser_test::notchHz (1000.0 * std::pow (2.0, -1.0), 4, sr);   // 207.17
    const double fHigh = phaser_test::notchHz (1000.0 * std::pow (2.0, +1.0), 4, sr);   // 832.37
    const double fMid  = phaser_test::notchHz (1000.0, 4, sr);                          // 414.70

    CHECK (std::fabs (fLow - 207.17) < 0.05);
    CHECK (std::fabs (fHigh - 832.37) < 0.05);

    EffectChannelParams p = phaser_test::active (1000.0f, 4, 50.0f);
    p.phaser.depthOct = 1.0f;
    p.phaser.rateHz = (float) rate;

    // Early is samples 2400..6000, by which time the chain has settled and the
    // sine LFO has moved 1.25 % of a cycle - it is still within 0.3 % of its
    // bottom, which is what makes a notch this deep hold long enough to see.
    // Late is the last 3600 samples before the top of the sweep.
    const std::vector<float> low  = phaser_test::renderSine (p, fLow, n, sr);
    const std::vector<float> high = phaser_test::renderSine (p, fHigh, n, sr);
    const std::vector<float> mid  = phaser_test::renderSine (p, fMid, n, sr);

    CHECK (phaser_test::peakBetween (low, 2400, 6000) < 0.01f);             // 0.0014 measured
    CHECK (phaser_test::peakBetween (low, half - 3600, half) > 0.25f);      // 0.458  - gone
    CHECK (phaser_test::peakBetween (high, 2400, 6000) > 0.10f);            // 0.280  - not yet
    CHECK (phaser_test::peakBetween (high, half - 3600, half) < 0.01f);     // 0.0006 - arrived

    // And the un-swept centre is notched at NEITHER end: the sweep really has
    // moved the notch away from where depth 0 would leave it.
    CHECK (phaser_test::peakBetween (mid, 2400, 6000) > 0.20f);             // 0.466
    CHECK (phaser_test::peakBetween (mid, half - 3600, half) > 0.20f);      // 0.343

    // Depth 0 is the identity of the sweep, exactly: exp2(0) is 1 to the bit,
    // so the static law of testPhaserAllpassLaw must survive a moving LFO.
    {
        EffectChannelParams flat = phaser_test::active (1000.0f, 4, 50.0f);
        flat.phaser.rateHz = 10.0f;             // as fast as it goes, and irrelevant
        CHECK (phaser_test::peakFrom (phaser_test::renderSine (flat, fMid, 6000, sr), 2400) < 2.0e-3f);
    }
}

static void testPhaserSpreadLaw()
{
    using namespace spatcore::effects;

    const double sr = 48000.0;

    // The bisection helper against the closed form, at spread 0 where they
    // describe the same chain. If this fails, nothing below means anything.
    CHECK (std::fabs (phaser_test::chainPhaseHz (1000.0, 0.0, 4, sr, 0.5)
                        - phaser_test::notchHz (1000.0, 4, sr)) < 1.0e-6);

    // Spread fans the stages over 2^(spread*(2k/(N-1) - 1)): at spread 2 and
    // four stages they sit at 250, 630, 1587 and 4000 Hz rather than all on
    // 1000. The chain still reaches half a turn exactly once, but 117 Hz lower
    // than it would unspread - and the unspread notch frequency is now passed
    // at a phase that does not cancel.
    for (int stages : { 4, 8 })
    {
        EffectChannelParams p = phaser_test::active (1000.0f, stages, 50.0f);
        p.phaser.spreadOct = 2.0f;

        const double spreadNotch = phaser_test::chainPhaseHz (1000.0, 2.0, stages, sr, 0.5);
        const double flatNotch   = phaser_test::notchHz (1000.0, stages, sr);

        CHECK (spreadNotch < flatNotch - 50.0);     // it really has moved

        // Cancelled where the spread chain says, 297.7 Hz at four stages and
        // 140.6 at eight.
        CHECK (phaser_test::peakBetween (phaser_test::renderSine (p, spreadNotch, 24000, sr),
                                         6000, 24000) < 2.0e-3f);

        // And NOT cancelled where a module that ignored spread would put it
        // (414.70 / 199.19 Hz), which is the only thing that tells the two
        // apart - both are "a notch somewhere below the centre".
        CHECK (phaser_test::peakBetween (phaser_test::renderSine (p, flatNotch, 24000, sr),
                                         6000, 24000) > 0.05f);             // 0.19 / 0.27 measured
    }
}

static void testPhaserLfoRateAndShape()
{
    using namespace spatcore::effects;

    const double sr = 48000.0;
    const double fLow = phaser_test::notchHz (500.0, 4, sr);    // the depth-1 bottom notch

    // RATE. Park a tone at the bottom of the sweep and the level dips once per
    // LFO cycle, as the notch passes over it. Four seconds of audio hold four
    // dips at 1 Hz and eight at 2 Hz: the rate parameter is read from the audio
    // itself, so a module that ignored it (or halved it) cannot pass both.
    for (double rate : { 1.0, 2.0 })
    {
        EffectChannelParams p = phaser_test::active (1000.0f, 4, 50.0f);
        p.phaser.depthOct = 1.0f;
        p.phaser.rateHz = (float) rate;

        const std::vector<float> out = phaser_test::renderSine (p, fLow, (int) (5.0 * sr), sr);
        CHECK (phaser_test::countDips (out, 4800, 4 * (int) sr) == (int) (4.0 * rate));
    }

    // SHAPE. A Square LFO is -1 for a whole half cycle, so the chain is STATIC
    // for half a second at 1 Hz and the bottom notch holds all the way through
    // it. A Sine leaves that notch within a few thousand samples. Same rate,
    // same depth, same tone: only the shape parameter differs.
    {
        EffectChannelParams p = phaser_test::active (1000.0f, 4, 50.0f);
        p.phaser.depthOct = 1.0f;
        p.phaser.rateHz = 1.0f;

        p.phaser.shape = 2;                     // LFOWaveforms::Square
        CHECK (phaser_test::peakBetween (phaser_test::renderSine (p, fLow, (int) sr, sr),
                                         2400, 20000) < 2.0e-3f);

        p.phaser.shape = 1;                     // LFOWaveforms::Sine
        CHECK (phaser_test::peakBetween (phaser_test::renderSine (p, fLow, (int) sr, sr),
                                         2400, 20000) > 0.20f);
    }
}

static void testPhaserTransparency()
{
    using namespace spatcore::effects;
    const ChainConfig cfg = module_test::config (48000.0, 256);

    // Bypassed at defaults through a slot: the buffer is not touched at all.
    {
        ModuleSlot slot;
        slot.prepare (cfg, std::make_unique<PhaserModule>());
        slot.applyParams (EffectChannelParams(), 0);
        CHECK (slot.isBypassedSettled());

        std::vector<float> buf = eqtests::makeAwkwardSignal (256);
        const std::vector<float> reference = buf;
        slot.process (buf.data(), 256);
        CHECK (eqtests::bitEqualBlock (buf, reference));
        CHECK (slot.getLatencySamples() == 0);
        CHECK (slot.nanTrips.load() == 0);
    }

    // Mix 0 while ACTIVE is bit-transparent rather than merely inaudible: the
    // module stops instead of crossfading, so the negative zeros and denormals
    // in the awkward signal come back exactly as they went in. Feedback at 90 %
    // and the widest sweep prove nothing is running underneath either.
    {
        PhaserModule m;
        m.prepare (cfg);
        EffectChannelParams p = phaser_test::active (800.0f, 8, 0.0f);
        p.phaser.feedback = 90.0f;
        p.phaser.depthOct = 4.0f;
        CHECK (! m.applyParams (p, 0).bypass);
        CHECK (m.getLatencySamples() == 0);

        for (int block = 0; block < 3; ++block)
        {
            std::vector<float> buf = (block == 0) ? eqtests::makeAwkwardSignal (256)
                                                  : module_test::awkwardBlock (256, block);
            const std::vector<float> reference = buf;
            module_test::render (m, buf);
            CHECK (eqtests::bitEqualBlock (buf, reference));
        }
    }
}

static void testPhaserResetClearsTail()
{
    using namespace spatcore::effects;

    PhaserModule m;
    m.prepare (module_test::config (48000.0, 512));
    EffectChannelParams p = phaser_test::active (300.0f, 12, 100.0f);
    p.phaser.feedback = 95.0f;
    p.phaser.depthOct = 2.0f;
    m.applyParams (p, 0);

    std::vector<float> buf = module_test::awkwardBlock (512, 5);
    module_test::render (m, buf);

    // Control: twelve allpasses at 95 % feedback are still ringing loudly after
    // the input stops, so the assertion below has something to clear.
    std::vector<float> tail (512, 0.0f);
    module_test::render (m, tail);
    CHECK (phaser_test::peakFrom (tail, 0) > 1.0e-4f);

    m.reset();

    bool silent = true;
    for (int block = 0; block < 4; ++block)
    {
        std::vector<float> after (512, 0.0f);
        module_test::render (m, after);

        for (int i = 0; i < 512; ++i)
            if (after[(size_t) i] != 0.0f)
                silent = false;
    }
    CHECK (silent);

    // prepare() again, at a different rate, on a module that has already run:
    // the device can change under a live chain. Everything the sample rate
    // reaches (the tan argument, the Nyquist clamp, the LFO and smoother
    // reference) is recomputed, and nothing carries over.
    m.prepare (module_test::config (96000.0, 512));
    m.applyParams (p, 0);

    std::vector<float> after96 = module_test::awkwardBlock (512, 6);
    module_test::render (m, after96);

    bool finite = true;
    for (float s : after96)
        finite = finite && std::isfinite (s);

    CHECK (finite);
    CHECK (phaser_test::peakFrom (after96, 0) > 0.0f);
    CHECK (phaser_test::peakFrom (after96, 0) < 8.0f);
}

static void testPhaserStagesAreAVariant()
{
    using namespace spatcore::effects;
    const ChainConfig cfg = module_test::config (48000.0, 256);

    PhaserModule held, reference;
    held.prepare (cfg);
    reference.prepare (cfg);

    EffectChannelParams six = phaser_test::active (800.0f, 6, 50.0f);
    six.phaser.depthOct = 1.0f;             // a moving sweep: the two must stay in step
    CHECK (! held.applyParams (six, 0).variantPending);
    reference.applyParams (six, 0);

    EffectChannelParams twelve = six;
    twelve.phaser.stages = 12;

    // Staged but not taken: the module keeps running six, sample for sample.
    CHECK (held.applyParams (twelve, 0).variantPending);
    CHECK (eqtests::bitEqualBlock (phaser_test::renderBlock (held, 7),
                                   phaser_test::renderBlock (reference, 7)));

    // Taken back before the fade completed: nothing pending, nothing reset.
    CHECK (! held.applyParams (six, 0).variantPending);

    // Asked again and committed at silence: twelve stages now, and twelve
    // sound different from six.
    CHECK (held.applyParams (twelve, 0).variantPending);
    held.reset();
    held.commitPendingVariant();
    CHECK (! held.applyParams (twelve, 0).variantPending);
    reference.reset();
    CHECK (! eqtests::bitEqualBlock (phaser_test::renderBlock (held, 8),
                                     phaser_test::renderBlock (reference, 8)));

    // 5 is not one of the validated counts: it degrades to the lower one, 4.
    EffectChannelParams five = six;
    five.phaser.stages = 5;
    CHECK (held.applyParams (five, 0).variantPending);
    held.reset();
    held.commitPendingVariant();

    EffectChannelParams four = six;
    four.phaser.stages = 4;
    CHECK (! held.applyParams (four, 0).variantPending);     // 5 really did mean 4
}

static void testPhaserExtremesStayFinite()
{
    using namespace spatcore::effects;

    const float centres[] = { 100.0f, 5000.0f };
    const float feedbacks[] = { -95.0f, 95.0f, 400.0f };    // 400 is out of range on purpose
    const int counts[] = { 4, 6, 8, 12 };                   // every validated count
    const int shapes[] = { 1, 2, 8 };                       // Sine, the Square edge, keyed Random

    for (float centreHz : centres)
        for (float fb : feedbacks)
            for (int stages : counts)
                for (int shape : shapes)
                {
                    EffectChannelParams p = phaser_test::active (centreHz, stages, 100.0f);
                    p.phaser.feedback = fb;
                    p.phaser.depthOct = 4.0f;       // the whole sweep, as fast as it goes
                    p.phaser.spreadOct = 3.0f;
                    p.phaser.rateHz = 10.0f;
                    p.phaser.shape = (std::uint8_t) shape;

                    // Where this number comes from, since the honest answer
                    // matters more than the number: the hot block peaks at
                    // 0.98, an allpass chain has magnitude one, and feedback is
                    // clamped to 0.95, so a loop that is merely resonating
                    // settles at 0.98/(1 - 0.95) = 19.6. The worst this grid
                    // reaches in 128 blocks is 14.7 (centre 100, feedback 95,
                    // six stages, the Square LFO - a shape that steps the whole
                    // sweep at once is what strains the coefficient
                    // interpolation hardest).
                    //
                    // It is NOT a hard bound, and should not be read as one: a
                    // modulated allpass can hand energy back to the loop, and
                    // over 2048 blocks this same grid does creep to 21.4. Read
                    // a failure here as "the sweep is pumping the loop sooner
                    // or harder than it used to", which is worth a look. The
                    // property that IS guaranteed - the ceiling - is asserted
                    // on its own below, over a run long enough for that creep.
                    // A NaN fails this too, because no comparison against a NaN
                    // is true.
                    CHECK (phaser_test::hotPeak (p, 128) < 20.0f);
                }

    // The classic self-oscillation case: maximum feedback and a tone parked on
    // the frequency where the chain turns the loop right ROUND, into positive
    // feedback. That frequency is the centre only when the stage count is a
    // multiple of four (each stage turns -90 degrees there); six stages turn
    // -540 and the resonance is elsewhere, which is why it is looked up rather
    // than assumed - and why the six-stage chain, the plan's default, would
    // otherwise be tested at the one frequency where it does NOT resonate.
    for (int stages : { 4, 6, 8, 12 })
    {
        EffectChannelParams p = phaser_test::active (1000.0f, stages, 100.0f);
        p.phaser.feedback = 95.0f;

        const double f = phaser_test::chainPhaseHz (1000.0, 0.0, stages, 48000.0, 1.0);
        const float settled = phaser_test::peakFrom (phaser_test::renderSine (p, f, 96000), 48000);

        CHECK (settled > 1.0f);     // the resonance is real: 3.6 / 5.6 / 6.9 / 8.3
        CHECK (settled < 10.5f);    // and bounded by 0.5/(1 - 0.95), the algebra's
                                    // own answer for a half-scale input - the
                                    // resonance rises steeply with stage count,
                                    // so 12 stages is the case that matters
    }

    // The guarantee, on the grid's worst corner and over eleven seconds of hot
    // audio - long enough for the modulation to pump the loop past the 19.6 the
    // algebra alone would give it. However far it creeps, it is finite and it
    // is inside the ceiling: that much is structural, not empirical. (hotPeak
    // returns a NaN if anything non-finite came out, and no comparison against
    // a NaN is true, so one bound covers both.)
    {
        EffectChannelParams p = phaser_test::active (100.0f, 6, 100.0f);
        p.phaser.feedback = 95.0f;
        p.phaser.depthOct = 4.0f;
        p.phaser.spreadOct = 3.0f;
        p.phaser.rateHz = 10.0f;
        p.phaser.shape = 2;

        CHECK (phaser_test::hotPeak (p, 2048) <= PhaserModule::kLoopCeiling);
    }

    // The ceiling itself, which nothing above reaches and which the module's
    // own header used to claim could not engage at all. It does: the house
    // awkward signal carries +-1e7 samples, and twelve stages at maximum
    // feedback put the loop straight onto it. What the ceiling promises is not
    // that it never fires but that when it does the output is bounded by it
    // exactly - 140 dB of overload in, +30 dBFS out - so nothing downstream
    // sees the loop's real excursion.
    {
        PhaserModule m;
        m.prepare (module_test::config (48000.0, 256));
        EffectChannelParams p = phaser_test::active (800.0f, 12, 100.0f);
        p.phaser.feedback = 95.0f;
        m.applyParams (p, 0);

        std::vector<float> buf = eqtests::makeAwkwardSignal (256);
        module_test::render (m, buf);

        const float peak = phaser_test::peakFrom (buf, 0);
        CHECK (peak > 16.0f);                               // it really did engage
        CHECK (peak <= PhaserModule::kLoopCeiling);         // and it held, at 32.0
    }

    // A NaN parameter must land on a bound rather than in the audio.
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        EffectChannelParams p = phaser_test::active (nan, 8, 100.0f);
        p.phaser.depthOct = nan;
        p.phaser.spreadOct = nan;
        p.phaser.rateHz = nan;
        p.phaser.feedback = 95.0f;
        CHECK (phaser_test::hotPeak (p, 8) < 20.0f);
    }
}

static void testPhaserSurvivesNonFiniteInput()
{
    using namespace spatcore::effects;
    const ChainConfig cfg = module_test::config (48000.0, 256);

    // A recursive module cannot lean on the slot's NaN guard the way a
    // memoryless one can. One non-finite sample from upstream reaches the
    // allpass history, and history is what the next sample is built from: if
    // the module clamps its OUTPUT to the loop ceiling and leaves the history
    // alone, it hands the slot a finite -32 for ever while staying internally
    // poisoned - a permanent +30 dBFS DC latch that the guard, which looks for
    // a non-finite sample, can never see. This is that regression, driven
    // through a real slot so the guard is genuinely in the picture.
    for (float poison : { std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity() })
    {
        for (float mix : { 50.0f, 100.0f })
        {
            ModuleSlot slot;
            slot.prepare (cfg, std::make_unique<PhaserModule>());

            EffectChannelParams p = phaser_test::active (800.0f, 8, mix);
            p.phaser.feedback = 30.0f;
            p.phaser.depthOct = 2.0f;
            slot.applyParams (p, 0);

            // Fade fully in first. While the slot is still crossfading it
            // holds its own copy of the DRY block and mixes the poisoned
            // sample back in itself - a documented property of the slot (a
            // single NaN can cross a fade untripped, EffectModule.h), and
            // nothing to do with the module under test. Settled, the module's
            // output IS the buffer, so what follows is the module's answer
            // alone.
            for (int block = 0; block < 64 && ! slot.isActiveSettled(); ++block)
            {
                std::vector<float> warm = module_test::awkwardBlock (256, block + 1);
                slot.process (warm.data(), 256);
            }

            CHECK (slot.isActiveSettled());

            std::vector<float> buf = module_test::awkwardBlock (256, 20);
            buf[100] = poison;
            slot.process (buf.data(), 256);

            for (float s : buf)
                CHECK (std::isfinite (s));

            // The block AFTER is the one that matters: this is where the latch
            // showed itself, every sample pinned at the ceiling for ever.
            float worst = 0.0f;
            for (int block = 0; block < 8; ++block)
            {
                std::vector<float> clean = module_test::awkwardBlock (256, 30 + block);
                slot.process (clean.data(), 256);

                for (float s : clean)
                {
                    CHECK (std::isfinite (s));
                    worst = std::max (worst, std::fabs (s));
                }
            }

            CHECK (worst < 4.0f);               // 1.4 measured: ordinary audio again

            // Nothing non-finite ever left the module, so the slot never had to
            // step in. That is the contract: a module with loop state cleans up
            // after itself, and the guard stays the backstop it was meant to be.
            CHECK (slot.nanTrips.load() == 0);
        }
    }
}

static void testPhaserDeterminism()
{
    using namespace spatcore::effects;
    const ChainConfig cfg = module_test::config (48000.0, 256, 7);

    PhaserModule a, b, other;
    a.prepare (cfg);
    b.prepare (cfg);
    other.prepare (module_test::config (48000.0, 256, 99));     // a different channel

    EffectChannelParams p = phaser_test::active (600.0f, 8, 50.0f);
    p.phaser.shape = 8;                     // the keyed random shape: the only
    p.phaser.depthOct = 3.0f;               // noise the module can reach
    p.phaser.feedback = 60.0f;
    a.applyParams (p, 0);
    b.applyParams (p, 0);
    other.applyParams (p, 0);

    // Same key, same audio to the bit; different key, different audio. Without
    // the second half, a module that dropped the per-channel key entirely - and
    // so gave every effects channel the same sweep - would pass.
    bool differs = false;

    for (int block = 0; block < 6; ++block)
    {
        const std::vector<float> fromA = phaser_test::renderBlock (a, block + 3);
        CHECK (eqtests::bitEqualBlock (fromA, phaser_test::renderBlock (b, block + 3)));
        differs = differs || ! eqtests::bitEqualBlock (fromA, phaser_test::renderBlock (other, block + 3));
    }

    CHECK (differs);
}

//==============================================================================
// effects/modules - Reverb (FxReverb)
//==============================================================================

//==============================================================================
// effects/modules/EffectReverbModule - predelay, FDN tail, tone, mix.
//==============================================================================

namespace reverb_test
{
    using namespace spatcore::effects;

    inline EffectChannelParams params (float mix, float predelayMs)
    {
        EffectChannelParams p;
        p.reverb.bypass = 0;
        p.reverb.mix = mix;
        p.reverb.predelayMs = predelayMs;
        return p;
    }

    /** Mean-square level of a window, in dB. */
    inline double windowDb (const std::vector<float>& v, int from, int to)
    {
        double sum = 0.0;
        for (int i = from; i < to; ++i)
            sum += (double) v[(size_t) i] * (double) v[(size_t) i];

        const double mean = sum / (double) (to - from);
        return 10.0 * std::log10 (mean > 1e-30 ? mean : 1e-30);
    }

    /** Level of a BAND of a window, in dB.

        Sixteen windowed single-frequency probes spread across the band, summed:
        one DFT bin of a reverb tail is far too noisy to compare against another,
        and a band average is what the ear hears anyway. The Hann window matters
        - a tail is not periodic in the window, and the skirts of a rectangular
        one would smear a 46 dB low band into a 112 dB high one. */
    inline double bandDb (const std::vector<float>& v, int from, int to,
                          double f0, double f1, double sampleRate)
    {
        const int n = to - from;
        double total = 0.0;

        for (int k = 0; k < 16; ++k)
        {
            const double f = f0 + (f1 - f0) * (double) k / 15.0;
            const double w = 6.283185307179586 * f / sampleRate;
            double re = 0.0, im = 0.0;

            for (int i = 0; i < n; ++i)
            {
                const double h = 0.5 - 0.5 * std::cos (6.283185307179586 * (double) i / (double) n);
                const double x = h * (double) v[(size_t) (from + i)];
                re += x * std::cos (w * (double) i);
                im -= x * std::sin (w * (double) i);
            }

            total += re * re + im * im;
        }

        const double mean = total / (16.0 * (double) n * (double) n);
        return 10.0 * std::log10 (mean > 1e-30 ? mean : 1e-30);
    }

    inline std::vector<float> impulseResponse (EffectReverbModule& m, int n)
    {
        std::vector<float> buf ((size_t) n, 0.0f);
        buf[0] = 1.0f;
        module_test::render (m, buf);               // one call, longer than maxBlock: chunked
        return buf;
    }
}

static void testEffectReverbIdentity()
{
    using namespace spatcore::effects;

    // Bypassed at the defaults: the slot skips the module, so the block comes
    // back bit for bit - negative zeros and denormals included.
    {
        ModuleSlot slot;
        slot.prepare (module_test::config (48000.0, 256), std::make_unique<EffectReverbModule>());

        EffectChannelParams p;                      // reverb.bypass == 1
        slot.applyParams (p, 0);
        CHECK (slot.isBypassedSettled());

        std::vector<float> buf = eqtests::makeAwkwardSignal (256);
        const std::vector<float> in = buf;
        slot.process (buf.data(), 256);
        CHECK (eqtests::bitEqualBlock (buf, in));
    }

    // ACTIVE at mix 0: the reverb runs underneath (its tail has to be there
    // when the mix comes back up) but the dry buffer is never written, so this
    // is bit-exact rather than near.
    {
        EffectReverbModule m;
        m.prepare (module_test::config (48000.0, 256));
        EffectChannelParams p = reverb_test::params (0.0f, 10.0f);
        CHECK (! m.applyParams (p, 0).bypass);

        bool identical = true;
        for (int block = 0; block < 6; ++block)
        {
            std::vector<float> buf = (block == 0) ? eqtests::makeAwkwardSignal (256)
                                                  : module_test::awkwardBlock (256, block);
            const std::vector<float> in = buf;
            m.process (buf.data(), 256);
            identical = identical && eqtests::bitEqualBlock (buf, in);
        }
        CHECK (identical);

        // ...and it really was RUNNING, not frozen: the mix comes back up onto
        // a tail that is already there. This is the whole reason mix 0 is an
        // early-out on the WRITE rather than on the block - freezing would
        // replay a stale tail the moment someone opened the control.
        EffectChannelParams up = reverb_test::params (100.0f, 10.0f);
        m.applyParams (up, 0);
        std::vector<float> tail (4096, 0.0f);       // silence in
        module_test::render (m, tail);
        CHECK (reverb_test::windowDb (tail, 0, 4096) > -60.0);      // measures -16.5
    }

    // A reverb adds a tail, not a delay: the dry component is not moved, so
    // there is nothing for the ledger to compensate.
    EffectReverbModule fresh;
    fresh.prepare (module_test::config());
    CHECK (fresh.getLatencySamples() == 0);
}

static void testEffectReverbPredelay()
{
    using namespace spatcore::effects;

    const int n = 4096;
    const ChainConfig cfg = module_test::config (48000.0, 512);

    EffectReverbModule a, b;
    a.prepare (cfg);
    b.prepare (cfg);

    // 2 ms at 48 kHz is 96 samples exactly (2.0 * 48.0), so the delayed read
    // lands ON a sample and the interpolator does nothing.
    EffectChannelParams pa = reverb_test::params (100.0f, 0.0f);
    EffectChannelParams pb = reverb_test::params (100.0f, 2.0f);
    a.applyParams (pa, 0);
    b.applyParams (pb, 0);

    const std::vector<float> ra = reverb_test::impulseResponse (a, n);
    const std::vector<float> rb = reverb_test::impulseResponse (b, n);

    // Nothing comes out before the predelay elapses, and past it B is A shifted
    // by 96 samples to the BIT: same key, so the same network doing the same
    // arithmetic on the same values, later.
    bool silentFirst = true, shifted = true;
    for (int i = 0; i < 96; ++i)
        silentFirst = silentFirst && (rb[(size_t) i] == 0.0f);
    for (int i = 96; i < n; ++i)
        shifted = shifted && bitEqualFloat (rb[(size_t) i], ra[(size_t) (i - 96)]);

    CHECK (silentFirst);
    CHECK (shifted);
    CHECK (reverb_test::windowDb (ra, 1024, n) > -80.0);   // two silent buffers would pass the above

    // The block size is a buffer size, not a parameter. Both the module and the
    // model cut a call into maxBlock chunks, and the FDN ignores the block size
    // it is prepared with, so the same signal through the same key must come out
    // the same bits whatever the host's block happens to be.
    auto renderChunked = [n] (int maxBlock)
    {
        EffectReverbModule m;
        m.prepare (module_test::config (48000.0, maxBlock));
        EffectChannelParams p = reverb_test::params (100.0f, 7.3f);   // a fractional predelay
        m.applyParams (p, 0);

        std::vector<float> buf ((size_t) n, 0.0f);
        buf[0] = 1.0f;
        for (int done = 0; done < n; )
        {
            const int chunk = (n - done) < maxBlock ? (n - done) : maxBlock;
            m.process (buf.data() + done, chunk);
            done += chunk;
        }
        return buf;
    };

    CHECK (eqtests::bitEqualBlock (renderChunked (1), renderChunked (4096)));
    CHECK (eqtests::bitEqualBlock (renderChunked (64), renderChunked (256)));
}

static void testEffectReverbDecayLaw()
{
    using namespace spatcore::effects;

    // Both multipliers at 1 collapses the three decay bands into one and every
    // line then loses exactly 60 dB per rt60, so the tail is 10^(-3t/rt60)
    // whatever path a sample took: 30 dB nominal over half a second at rt60 = 1,
    // 7.5 dB at rt60 = 4. Measured comes in a little under both (28.7 and 7.2)
    // because the feedback allpass lengthens every loop by 3-7 % without being
    // counted in the decay gain.
    auto dropDb = [] (float rt60, double sr)
    {
        EffectReverbModule m;
        m.prepare (module_test::config (sr, 512));

        EffectChannelParams p = reverb_test::params (100.0f, 0.0f);
        p.reverb.rt60 = rt60;
        p.reverb.rt60LowMult = 1.0f;
        p.reverb.rt60HighMult = 1.0f;
        m.applyParams (p, 0);

        const int n = (int) sr;                             // one second
        const int window = (int) (sr * 0.1);
        const std::vector<float> ir = reverb_test::impulseResponse (m, n);
        return reverb_test::windowDb (ir, (int) (sr * 0.20), (int) (sr * 0.20) + window)
             - reverb_test::windowDb (ir, (int) (sr * 0.70), (int) (sr * 0.70) + window);
    };

    const double drop1 = dropDb (1.0f, 48000.0);
    const double drop4 = dropDb (4.0f, 48000.0);

    CHECK (drop1 > 24.0 && drop1 < 34.0);
    CHECK (drop4 > 3.0 && drop4 < 12.0);
    CHECK (drop1 > drop4 + 10.0);                   // a longer rt60 is a slower tail

    // The same law at 96 kHz, which is also the only place the doubled delay
    // ceiling and the rate-scaled line lengths get exercised.
    CHECK (std::fabs (dropDb (1.0f, 96000.0) - drop1) < 4.0);
}

static void testEffectReverbTone()
{
    using namespace spatcore::effects;

    // The one-pole low pass on the wet, which nothing else in this file touches:
    // Identity runs at mix 0, Predelay and DecayLaw leave it at the default on
    // both sides of their comparisons, and Extremes only asks for a finite peak.
    // A build that dropped the filter, wired it as a high pass, or flipped the
    // sign of the exponent behind the coefficient would pass all of those.
    const double sr = 48000.0;
    const int n = 32768;
    const int from = 4800, to = from + 16384;       // past the first 100 ms, then 341 ms of tail

    auto tail = [sr, n] (float toneHz)
    {
        EffectReverbModule m;
        m.prepare (module_test::config (sr, 512));

        EffectChannelParams p = reverb_test::params (100.0f, 0.0f);
        p.reverb.rt60 = 2.0f;
        p.reverb.rt60LowMult = 1.0f;                // one decay band, so the only
        p.reverb.rt60HighMult = 1.0f;               // colour left is the tone filter
        p.reverb.toneHz = toneHz;
        m.applyParams (p, 0);
        return reverb_test::impulseResponse (m, n);
    };

    const std::vector<float> dark = tail (1000.0f);         // the bottom of the range
    const std::vector<float> bright = tail (20000.0f);      // the top
    const std::vector<float> dflt = tail (12000.0f);        // the default

    const double hiDark   = reverb_test::bandDb (dark,   from, to, 9000.0, 11000.0, sr);
    const double hiBright = reverb_test::bandDb (bright, from, to, 9000.0, 11000.0, sr);
    const double hiDflt   = reverb_test::bandDb (dflt,   from, to, 9000.0, 11000.0, sr);

    const double loDark   = reverb_test::bandDb (dark,   from, to, 150.0, 300.0, sr);
    const double loBright = reverb_test::bandDb (bright, from, to, 150.0, 300.0, sr);

    // Measured on this build at 48 kHz: 10 kHz is -112.7 dB at tone 1000 and
    // -94.0 at tone 20000, i.e. 18.7 dB apart (18.9 at 96 kHz). 14 leaves room
    // for a platform's exp/pow to drift and still fails a missing filter (0 dB)
    // or a high pass (negative).
    CHECK (hiBright - hiDark > 14.0);

    // It is a LOW pass: moving the cutoff by more than four octaves must not
    // move the bottom of the tail. Measured 0.26 dB apart.
    CHECK (std::fabs (loBright - loDark) < 2.0);

    // ...and the cutoff maps monotonically, so the default sits between the two.
    CHECK (hiDflt > hiDark + 10.0 && hiDflt < hiBright);
}

static void testEffectReverbWetLevel()
{
    using namespace spatcore::effects;

    // The wet make-up. It is one constant (EffectReverbModule::kWetGain) chosen
    // by ear against the FDN's own +12 dB, and every other test in this file is
    // blind to it: Identity is mix 0, Predelay and SizeVariant compare two
    // instances that would both scale, DecayLaw is a DIFFERENCE of two dB
    // windows so a constant gain cancels exactly, and Reset is a silence check.
    // So it is pinned here, on an absolute level, or it is pinned nowhere.
    const int n = 120000;                           // 2.5 s at 48 kHz
    const std::vector<float> in = module_test::awkwardBlock (n, 3);

    EffectReverbModule m;
    m.prepare (module_test::config (48000.0, 512));

    EffectChannelParams p = reverb_test::params (100.0f, 0.0f);
    p.reverb.rt60 = 1.5f;                           // everything else at its default
    m.applyParams (p, 0);

    std::vector<float> buf = in;
    module_test::render (m, buf);

    // The last second, by which time a 1.5 s tail is fully built.
    const double ratio = reverb_test::windowDb (buf, 72000, n)
                       - reverb_test::windowDb (in, 72000, n);

    // Measured -4.88 dB: a fully wet reverb sits just under the dry it replaced,
    // which is the point - a mix control has to be able to work either side of
    // the balance. The two values this constant has actually been given during
    // development land outside the window: 1.0 gives -10.90 and 0.25 gives
    // -22.94, and doubling it again to 4.0 would give +1.14.
    CHECK (ratio > -8.0 && ratio < -2.0);

    // The rate is pinned deliberately. The FDN's own tone filter is at a fixed
    // 8 kHz while its line lengths scale with the rate, so the same settings
    // measure -4.3 dB at 44.1 kHz and -8.3 at 96 - a real property of the
    // model, not something a make-up constant should be chasing.
}

static void testEffectReverbResetClearsTail()
{
    using namespace spatcore::effects;

    EffectReverbModule m;
    m.prepare (module_test::config (48000.0, 256));

    EffectChannelParams p = reverb_test::params (100.0f, 20.0f);
    p.reverb.rt60 = 8.0f;                           // the longest tail on the surface
    m.applyParams (p, 0);

    std::vector<float> excite = module_test::awkwardBlock (2048, 7);
    module_test::render (m, excite);
    CHECK (reverb_test::windowDb (excite, 1024, 2048) > -80.0);
    CHECK (m.getMeterDb() > -40.0);                 // the wet meter followed it up

    // A slot that faded out resets us before fading back in; anything left in
    // the network, the predelay ring or the tone filter would be replayed.
    m.reset();
    CHECK (m.getMeterDb() <= spatcore::dsp::FastDecibels::kMinDb);

    std::vector<float> silence (2048, 0.0f);
    module_test::render (m, silence);

    bool silent = true;
    for (int i = 0; i < 2048; ++i)
        silent = silent && (silence[(size_t) i] == 0.0f);
    CHECK (silent);
}

static void testEffectReverbSizeSpillover()
{
    using namespace spatcore::effects;

    const ChainConfig cfg = module_test::config (48000.0, 256);

    // The FIRST applyParams after prepare() builds the project's size there and
    // then. prepare() carries no parameters, so it built the default; a freshly
    // loaded show must neither fade nor spill over defaults it never meant to
    // play.
    {
        EffectReverbModule loaded, atDefault;
        loaded.prepare (cfg);
        atDefault.prepare (cfg);

        EffectChannelParams saved = reverb_test::params (100.0f, 0.0f);
        saved.reverb.size = 1.6f;                   // NOT the 1.0 prepare() guessed
        CHECK (! loaded.applyParams (saved, 0).variantPending);
        CHECK (loaded.getSpillVoices() == 0);
        CHECK (std::fabs (loaded.getActiveSize() - 1.6f) < 1.0e-5f);

        EffectChannelParams guessed = reverb_test::params (100.0f, 0.0f);
        CHECK (! atDefault.applyParams (guessed, 0).variantPending);

        // Built, not quietly dropped: 1.6 is a different network from 1.0.
        CHECK (! eqtests::bitEqualBlock (reverb_test::impulseResponse (loaded, 2048),
                                         reverb_test::impulseResponse (atDefault, 2048)));

        // Past the first set, with a tail running, a size change never asks the
        // slot for a fade: the input moves to the new size at once, and the old
        // network rings out underneath it.
        saved.reverb.size = 1.2f;
        CHECK (! loaded.applyParams (saved, 0).variantPending);
        CHECK (loaded.getSpillVoices() == 1);
        CHECK (std::fabs (loaded.getActiveSize() - 1.2f) < 1.0e-5f);

        // Handed the same set again, nothing new starts.
        loaded.applyParams (saved, 0);
        CHECK (loaded.getSpillVoices() == 1 && ! loaded.isTransitionWaiting());

        // A module that has not run since its last reset has no tail to spill,
        // so a change there is taken directly - a bypassed slot is one.
        loaded.reset();
        CHECK (loaded.getSpillVoices() == 0);
        saved.reverb.size = 0.8f;
        loaded.applyParams (saved, 0);
        CHECK (loaded.getSpillVoices() == 0);
        CHECK (std::fabs (loaded.getActiveSize() - 0.8f) < 1.0e-5f);

        // Runtime values never spill: a decay change reaches the running network.
        std::vector<float> excite = module_test::awkwardBlock (1024, 2);
        module_test::render (loaded, excite);
        saved.reverb.rt60 = 3.0f;
        loaded.applyParams (saved, 0);
        CHECK (loaded.getSpillVoices() == 0);
    }

    // THE PARTITION. At predelay 0 and fully wet the module is linear, so a
    // spillover must be exactly two reverbs summed: the old network fed the
    // input up to the change and the fading complement after it, the new one
    // fed the share fading in. A doubled or dropped input, a tail cut short or
    // a new network that did not start from silence all show up as a
    // difference against the two rendered in isolation.
    {
        const int before = 6000, n = 30000;
        const int fade = (int) (EffectReverbModule::kSpillFadeSeconds * 48000.0 + 0.5);
        std::vector<float> in = module_test::awkwardBlock (n, 21);
        for (int i = before + 3000; i < n; ++i)
            in[(size_t) i] = 0.0f;                  // then silence: the tails ring on

        EffectChannelParams p = reverb_test::params (100.0f, 0.0f);
        p.reverb.rt60 = 2.0f;

        EffectReverbModule spill;
        spill.prepare (cfg);
        spill.applyParams (p, 0);

        std::vector<float> out = in;
        spill.process (out.data(), before);
        p.reverb.size = 1.6f;
        spill.applyParams (p, 0);
        spill.process (out.data() + before, n - before);

        EffectReverbModule oldNet, newNet;
        oldNet.prepare (cfg);
        newNet.prepare (cfg);
        EffectChannelParams po = reverb_test::params (100.0f, 0.0f);
        po.reverb.rt60 = 2.0f;
        oldNet.applyParams (po, 0);
        po.reverb.size = 1.6f;
        newNet.applyParams (po, 0);

        std::vector<float> outOld = in, outNew ((size_t) n, 0.0f);
        for (int i = before; i < n; ++i)
        {
            const int k = i - before;
            const double s = k < fade ? std::sin (3.141592653589793 * (double) k / (2.0 * fade)) : 1.0;
            const float g = (float) (s * s);
            outOld[(size_t) i] = in[(size_t) i] * (1.0f - g);
            outNew[(size_t) i] = in[(size_t) i] * g;
        }
        module_test::render (oldNet, outOld);
        module_test::render (newNet, outNew);

        float worst = 0.0f, peak = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            const float d = std::fabs (out[(size_t) i] - (outOld[(size_t) i] + outNew[(size_t) i]));
            worst = d > worst ? d : worst;
            peak = std::fabs (out[(size_t) i]) > peak ? std::fabs (out[(size_t) i]) : peak;
        }
        // Measured 6.3e-7 against a peak of 0.85: rounding, and the module's
        // libm-free fade against the std::sin one here. A share dropped or a
        // tail cut short misses by four orders of magnitude.
        CHECK (peak > 0.1f);
        CHECK (worst < 1.0e-4f * peak);

        // ...and the spill is audible, not a technicality: 100 to 200 ms after
        // the change the old network still carries a real share of the output
        // (measured -18.5 dB of -15.8).
        CHECK (reverb_test::windowDb (outOld, before + 4800, before + 9600)
               > reverb_test::windowDb (out, before + 4800, before + 9600) - 12.0);

        // No dip: the 50 ms after the change are as loud as the 50 ms before
        // (measured +1.1 dB - the input is still arriving).
        CHECK (reverb_test::windowDb (out, before, before + 2400)
               > reverb_test::windowDb (out, before - 2400, before) - 3.0);
    }

    // A voice taken again starts from silence. Size 1.0 is excited and left
    // ringing under 1.6; a change back to 1.0 needs the ringing network's
    // voice, so it waits while that one fades out, then takes it. Fed silence
    // throughout, the reverb that comes back must be silent - an instance
    // reused dirty would replay the old tail at full level.
    {
        EffectReverbModule m;
        m.prepare (cfg);
        EffectChannelParams p = reverb_test::params (100.0f, 0.0f);
        p.reverb.rt60 = 8.0f;
        m.applyParams (p, 0);

        std::vector<float> excite = module_test::awkwardBlock (4800, 5);
        module_test::render (m, excite);

        std::vector<float> silence (480, 0.0f);
        p.reverb.size = 1.6f;
        m.applyParams (p, 0);                       // 1.0 rings, 1.6 takes the (silent) input
        module_test::render (m, silence);           // 10 ms: the crossfade is over
        CHECK (m.getSpillVoices() == 1);
        CHECK (reverb_test::windowDb (silence, 240, 480) > -40.0);  // the old tail, still loud

        p.reverb.size = 1.0f;
        m.applyParams (p, 0);                       // 1.0's voice is the ringing one:
        CHECK (m.isTransitionWaiting());            // the change waits...
        CHECK (m.getSpillVoices() == 1);            // ...while that voice dies

        std::fill (silence.begin(), silence.end(), 0.0f);
        module_test::render (m, silence);           // faded out inside these 10 ms
        CHECK (m.getSpillVoices() == 0);
        CHECK (m.isTransitionWaiting());            // and the change starts on a block boundary

        std::vector<float> after (4800, 0.0f);
        module_test::render (m, after);
        CHECK (! m.isTransitionWaiting());
        CHECK (std::fabs (m.getActiveSize() - 1.0f) < 1.0e-5f);

        float residue = 0.0f;
        for (int i = 0; i < 4800; ++i)
            residue = std::fabs (after[(size_t) i]) > residue ? std::fabs (after[(size_t) i]) : residue;
        CHECK (residue < 1.0e-12f);                // measured exactly 0
    }

    // Built at the capacity size and then at the size asked for, so what runs
    // is what was requested - before and after a commit.
    FdnReverbModel direct;
    ReverbParams rp;
    rp.size = 1.3f;
    direct.prepare (cfg, rp);
    CHECK (std::fabs (direct.getBuiltSize() - 1.3f) < 1.0e-5f);
    rp.size = 0.5f;
    CHECK (direct.isBuildDifferent (rp));           // asked without handing it over
    CHECK (std::fabs (direct.getBuiltSize() - 1.3f) < 1.0e-5f);
    CHECK (direct.setParams (rp));                  // pending, and not yet applied
    CHECK (std::fabs (direct.getBuiltSize() - 1.3f) < 1.0e-5f);
    direct.commitPendingVariant();
    CHECK (std::fabs (direct.getBuiltSize() - 0.5f) < 1.0e-5f);
    CHECK (! direct.isBuildDifferent (rp));
    rp.size = 0.501f;                               // quantised: the same network
    CHECK (! direct.isBuildDifferent (rp));
}

static void testEffectReverbSpilloverRelease()
{
    using namespace spatcore::effects;

    // A ringing network costs a whole second reverb, so it has to stop: once
    // its output has stayed under -96 dBFS for 50 ms, or at 30 s whatever it
    // is doing.
    const ChainConfig cfg = module_test::config (48000.0, 512);

    // A short tail: gone long before the cap.
    {
        EffectReverbModule m;
        m.prepare (cfg);
        EffectChannelParams p = reverb_test::params (100.0f, 0.0f);
        p.reverb.rt60 = 0.3f;
        p.reverb.rt60LowMult = 1.0f;
        p.reverb.rt60HighMult = 1.0f;
        m.applyParams (p, 0);

        std::vector<float> excite = module_test::awkwardBlock (4800, 9);
        module_test::render (m, excite);

        p.reverb.size = 1.5f;
        m.applyParams (p, 0);
        std::vector<float> silence (4800, 0.0f);
        module_test::render (m, silence);           // 100 ms: -20 dB, still ringing
        CHECK (m.getSpillVoices() == 1);

        std::vector<float> more (96000, 0.0f);      // 2 s
        module_test::render (m, more);
        CHECK (m.getSpillVoices() == 0);
    }

    // A tail that barely decays - 72 s in the low band - is faded at the cap.
    {
        EffectReverbModule m;
        m.prepare (cfg);
        EffectChannelParams p = reverb_test::params (100.0f, 0.0f);
        p.reverb.rt60 = 8.0f;
        p.reverb.rt60LowMult = 9.0f;
        p.reverb.crossoverLow = 500.0f;
        m.applyParams (p, 0);

        std::vector<float> excite = module_test::awkwardBlock (4800, 13);
        module_test::render (m, excite);

        p.reverb.size = 1.5f;
        m.applyParams (p, 0);

        std::vector<float> second (48000, 0.0f);
        for (int s = 0; s < 29; ++s)
        {
            std::fill (second.begin(), second.end(), 0.0f);
            module_test::render (m, second);
        }
        CHECK (m.getSpillVoices() == 1);            // 29 s: still ringing...
        CHECK (m.getMeterDb() > -60.0f);            // ...and loud (-51 dB), so quiet is not what ends it

        std::vector<float> last (52800, 0.0f);      // past 30 s, and the 5 ms fade
        module_test::render (m, last);
        CHECK (m.getSpillVoices() == 0);
    }
}

static void testEffectReverbSpilloverRapid()
{
    using namespace spatcore::effects;

    // Changes faster than the pool can spill them: a new size every 64-sample
    // block for 40 blocks. Whatever the pool does with them - wait for a
    // crossfade, steal the oldest tail - no gain may step, so the output stays
    // as smooth as the reverb of a low sine is. The sine fades in, so nothing
    // but a step inside the module can put a click in its reverb.
    const int block = 64, warm = 48000, changes = 40, n = warm + block * changes + 24000;
    const ChainConfig cfg = module_test::config (48000.0, block);
    const float sizes[] = { 1.6f, 0.7f, 1.3f, 2.0f, 0.5f, 1.0f, 1.8f };

    std::vector<float> sine ((size_t) n);
    for (int i = 0; i < n; ++i)
    {
        const double ramp = i < 4800 ? 0.5 - 0.5 * std::cos (3.141592653589793 * (double) i / 4800.0) : 1.0;
        sine[(size_t) i] = (float) (0.5 * ramp * std::sin (6.283185307179586 * 80.0 * (double) i / 48000.0));
    }

    bool sawWaiting = false;
    float lastSize = 0.0f;
    long allocations = -1;

    auto render = [&] (bool withChanges)
    {
        EffectReverbModule m;
        m.prepare (cfg);
        EffectChannelParams p = reverb_test::params (100.0f, 0.0f);
        p.reverb.rt60 = 2.0f;
        m.applyParams (p, 0);

        std::vector<float> buf = sine;
        alloc_probe::Scope probe;

        for (int b = 0; b * block < n; ++b)
        {
            const int at = b * block;
            if (withChanges && at >= warm && at < warm + block * changes)
            {
                p.reverb.size = sizes[(size_t) (b % 7)];
                m.applyParams (p, 0);
                sawWaiting = sawWaiting || m.isTransitionWaiting();
            }
            m.process (buf.data() + at, (n - at) < block ? (n - at) : block);
        }

        allocations = probe.allocations();
        lastSize = m.getActiveSize();
        return buf;
    };

    const std::vector<float> still = render (false);
    const std::vector<float> busy = render (true);
    CHECK (allocations == 0);                       // the whole storm, rebuilds included

    // Every change was eventually honoured, in order: the last one runs.
    CHECK (sawWaiting);
    CHECK (std::fabs (lastSize - sizes[(size_t) ((warm / block + changes - 1) % 7)]) < 1.0e-5f);

    // Curvature - the second difference - is what a click is made of, and
    // almost nothing for the reverb of an 80 Hz sine.
    auto curvature = [] (const std::vector<float>& v, int from, int to)
    {
        float worst = 0.0f;
        for (int i = from; i < to; ++i)
        {
            const float c = std::fabs (v[(size_t) i] - 2.0f * v[(size_t) (i - 1)] + v[(size_t) (i - 2)]);
            worst = c > worst ? c : worst;
        }
        return worst;
    };

    const int from = warm, to = warm + block * changes + 12000;
    const float cStill = curvature (still, from, to);
    const float cBusy = curvature (busy, from, to);

    float peakBusy = 0.0f;
    bool finite = true;
    for (int i = from; i < n; ++i)
    {
        finite = finite && std::isfinite (busy[(size_t) i]);
        peakBusy = std::fabs (busy[(size_t) i]) > peakBusy ? std::fabs (busy[(size_t) i]) : peakBusy;
    }

    // Measured 3.9e-5 against 1.5e-5 without the changes. Switching the input
    // hard gives 0.10, cutting a dying voice instead of fading it 0.067, and
    // starting a change during a crossfade - a gain jumping from mid-fade to
    // full - 1.7e-4, the smallest step there is to catch.
    CHECK (finite);
    CHECK (cBusy < 4.0f * cStill);

    // No runaway. (Not a level match: a room's response at 80 Hz is not flat,
    // so every size in the storm answers the same sine at its own level. The
    // partition above is what proves the input is never doubled.)
    CHECK (peakBusy < 1.0f);

    // Deterministic: the same storm twice is the same bits.
    CHECK (eqtests::bitEqualBlock (render (true), busy));

    // Through a slot: a size change asks for no fade, so the slot stays fully
    // wet across it - where the variant it used to report would have dipped
    // the whole reverb to silence and back.
    {
        ModuleSlot slot;
        slot.prepare (cfg, std::make_unique<EffectReverbModule>());
        EffectChannelParams p = reverb_test::params (100.0f, 0.0f);
        slot.applyParams (p, 0);

        std::vector<float> noise = module_test::awkwardBlock (48000, 17);
        for (int at = 0; at < 24000; at += block)
            slot.process (noise.data() + at, block);
        CHECK (slot.isActiveSettled());

        p.reverb.size = 1.7f;
        slot.applyParams (p, 0);
        CHECK (slot.isActiveSettled());

        bool settled = true;
        for (int at = 24000; at < 48000; at += block)
        {
            slot.process (noise.data() + at, block);
            settled = settled && slot.isActiveSettled();
        }
        CHECK (settled);
        CHECK (reverb_test::windowDb (noise, 24000, 26400) > reverb_test::windowDb (noise, 21600, 24000) - 3.0);
    }
}

static void testEffectReverbExtremesAndDeterminism()
{
    using namespace spatcore::effects;

    // Every corner of the surface, then values that are not on it at all: the
    // clamp has to catch a NaN before it reaches the pow() behind the decay
    // gains and turns all 48 of them into NaN.
    //
    // Every row is FULLY WET on purpose. A row at mix 0 measures the dry buffer
    // the module deliberately does not touch, so its peak is the input's and
    // its "finite" is the input's too - an earlier version of this sweep had
    // two such rows and they asserted nothing whatsoever about the reverb.
    // Mix 0 is pinned where it belongs, in testEffectReverbIdentity.
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    struct Corner { float rt60, lowMult, highMult, xLow, xHigh, diffusion, size, tone, predelay, mix, minPeak; };
    const Corner corners[] =
    {
        {  0.2f,  0.1f,  0.1f,    50.0f,  1000.0f, 0.0f,  0.5f,  1000.0f,    0.0f,  100.0f, 0.05f },
        {  8.0f,  9.0f,  9.0f,   500.0f, 10000.0f, 1.0f,  2.0f, 20000.0f,  250.0f,  100.0f, 0.30f },
        {  8.0f,  9.0f,  0.1f,    50.0f, 10000.0f, 1.0f,  2.0f, 20000.0f,  250.0f,  100.0f, 0.30f },
        { -1.0f,  1e9f, -5.0f, -100.0f,     1e6f,  5.0f, 99.0f,   -20.0f,    1e6f,  500.0f, 0.05f },
        {  qnan,  qnan,  qnan,    qnan,     qnan,  qnan,  qnan,    qnan,     qnan,  100.0f, 0.05f }
    };

    const int numCorners = (int) (sizeof (corners) / sizeof (corners[0]));
    const int n = 32768;                            // outlasts the longest predelay
    std::vector<std::vector<float>> rendered ((size_t) numCorners);

    for (int ci = 0; ci < numCorners; ++ci)
    {
        const Corner& c = corners[(size_t) ci];

        EffectReverbModule m;
        m.prepare (module_test::config (48000.0, 256));

        EffectChannelParams p = reverb_test::params (c.mix, c.predelay);
        p.reverb.rt60 = c.rt60;
        p.reverb.rt60LowMult = c.lowMult;
        p.reverb.rt60HighMult = c.highMult;
        p.reverb.crossoverLow = c.xLow;
        p.reverb.crossoverHigh = c.xHigh;
        p.reverb.diffusion = c.diffusion;
        p.reverb.size = c.size;
        p.reverb.toneHz = c.tone;

        CHECK (! m.applyParams (p, 0).variantPending);  // the first set builds the size for real

        std::vector<float> buf = module_test::awkwardBlock (4096, 11);
        const std::vector<float> in = buf;
        buf.resize ((size_t) n, 0.0f);              // excite, then let it ring out
        module_test::render (m, buf);

        // An almost lossless network fed 85 ms of noise settles well under
        // unity: the loudest corner peaks at 0.65, the quietest at 0.17.
        float peak = 0.0f;
        bool clean = true;
        for (int i = 0; i < n; ++i)
        {
            const float x = buf[(size_t) i];
            if (! std::isfinite (x))
            {
                clean = false;
                break;
            }
            if (std::fabs (x) > peak)
                peak = std::fabs (x);
        }
        CHECK (clean && peak > c.minPeak && peak < 2.0f);

        // ...and the wet path really was written, rather than the dry buffer
        // coming back untouched because a clamp sent the mix to zero.
        CHECK (! eqtests::bitEqualBlock (std::vector<float> (buf.begin(), buf.begin() + 4096), in));

        rendered[(size_t) ci] = std::move (buf);
    }

    // The strongest thing the NaN row can say. Every clamp in both classes is
    // written `if (! (v > lo)) return lo;`, so a NaN lands on the LOW bound of
    // all nine ranges it is fed to - which is exactly the all-minimum corner,
    // bit for bit. (Its mix is a real 100 rather than a tenth NaN: a NaN mix
    // leaves the buffer untouched whether the clamp is there or not, because
    // every comparison against a NaN is false, so it would prove nothing.)
    // Anything weaker than this - finite, inside a peak band - passes happily
    // on a module with no clamps at all.
    CHECK (eqtests::bitEqualBlock (rendered[(size_t) (numCorners - 1)], rendered[0]));

    // Same key, same network, same bits. A DIFFERENT key must not be the same
    // network: 32 channels sharing node 0 would comb rather than spread.
    EffectReverbModule a, b, c;
    a.prepare (module_test::config (48000.0, 512, 5));
    b.prepare (module_test::config (48000.0, 512, 5));
    c.prepare (module_test::config (48000.0, 512, 6));

    EffectChannelParams p = reverb_test::params (100.0f, 3.0f);
    a.applyParams (p, 0);
    b.applyParams (p, 0);
    c.applyParams (p, 0);

    const std::vector<float> ra = reverb_test::impulseResponse (a, 4096);
    const std::vector<float> rb = reverb_test::impulseResponse (b, 4096);
    const std::vector<float> rc = reverb_test::impulseResponse (c, 4096);

    CHECK (eqtests::bitEqualBlock (ra, rb));
    CHECK (! eqtests::bitEqualBlock (ra, rc));
}

static void testEffectReverbPresets()
{
    using namespace spatcore::effects;

    // The factory table is DATA, and data is what rots quietly. Pinned here:
    // every row is inside the surface the models clamp to, names an
    // implemented model and valid enums; the five shipped rooms keep their
    // numbers (projects store their ids); row 6 is exactly the defaults (so a
    // fresh channel is not a label over values it does not hold); and
    // expanding a preset writes the room and the model without touching taste.
    const int count = (int) ReverbType::Count;
    CHECK (count == 23);
    CHECK (findReverbPreset ((int) ReverbType::Custom) == nullptr);
    CHECK (findReverbPreset (-1) == nullptr);
    CHECK (findReverbPreset (count) == nullptr);

    // The model a stored id runs: 2 and 3 are reserved, anything unknown is
    // the FDN.
    CHECK (resolveReverbModel (0) == 0 && resolveReverbModel (1) == 1);
    CHECK (resolveReverbModel (2) == 0 && resolveReverbModel (3) == 0);
    CHECK (resolveReverbModel (4) == 4 && resolveReverbModel (5) == 5);
    CHECK (resolveReverbModel (6) == 0 && resolveReverbModel (-1) == 0 && resolveReverbModel (255) == 0);

    int rows = 0;
    for (int t = 0; t < count; ++t)
    {
        const ReverbPreset* row = findReverbPreset (t);
        if (t == (int) ReverbType::Custom)
        {
            CHECK (row == nullptr);
            continue;
        }

        CHECK (row != nullptr);
        if (row == nullptr)
            continue;

        ++rows;
        CHECK (row->name != nullptr);
        CHECK (resolveReverbModel (row->model) == (int) row->model);        // an implemented model
        CHECK (row->erProfile < (std::uint8_t) ErProfile::Count);
        CHECK (row->shimmerPitch < (std::uint8_t) ShimmerInterval::Count);
        CHECK (row->erLevelDb >= -30.0f && row->erLevelDb <= 6.0f);
        CHECK (row->rt60 >= 0.2f && row->rt60 <= 8.0f);
        CHECK (row->rt60LowMult >= 0.1f && row->rt60LowMult <= 9.0f);
        CHECK (row->rt60HighMult >= 0.1f && row->rt60HighMult <= 9.0f);
        CHECK (row->crossoverLow >= 50.0f && row->crossoverLow <= 500.0f);
        CHECK (row->crossoverHigh >= 1000.0f && row->crossoverHigh <= 10000.0f);
        CHECK (row->crossoverLow < row->crossoverHigh);
        CHECK (row->diffusion >= 0.0f && row->diffusion <= 1.0f);
        CHECK (row->size >= kReverbMinSize && row->size <= kReverbMaxSize);
        CHECK (row->predelayMs >= 0.0f && row->predelayMs <= EffectReverbModule::kMaxPredelayMs);
        CHECK (row->modRateHz >= 0.05f && row->modRateHz <= 5.0f);
        CHECK (row->modDepth >= 0.0f && row->modDepth <= 100.0f);
        CHECK (row->shimmerAmount >= 0.0f && row->shimmerAmount <= 100.0f);
    }
    CHECK (rows == count - 1);

    // The shipped rooms, frozen: model 0, no early reflections, the values
    // every existing project that stores ids 0..4 was saved with.
    struct Frozen { float rt60, lo, hi, xLo, xHi, diff, size, pre; };
    const Frozen frozen[5] = {
        { 0.6f, 1.1f, 0.5f, 200.0f, 4000.0f, 0.60f, 0.6f,  5.0f },
        { 1.2f, 1.2f, 0.6f, 180.0f, 5000.0f, 0.80f, 0.8f,  8.0f },
        { 2.4f, 1.3f, 0.4f, 200.0f, 4000.0f, 0.50f, 1.3f, 20.0f },
        { 5.0f, 1.5f, 0.3f, 150.0f, 3000.0f, 0.40f, 1.8f, 40.0f },
        { 1.8f, 0.8f, 0.9f, 300.0f, 8000.0f, 0.95f, 0.7f,  0.0f } };
    for (int t = 0; t < 5; ++t)
    {
        const ReverbPreset* row = findReverbPreset (t);
        CHECK (row != nullptr && row->model == 0 && row->erProfile == 0);
        if (row != nullptr)
            CHECK (row->rt60 == frozen[t].rt60 && row->rt60LowMult == frozen[t].lo
                   && row->rt60HighMult == frozen[t].hi && row->crossoverLow == frozen[t].xLo
                   && row->crossoverHigh == frozen[t].xHi && row->diffusion == frozen[t].diff
                   && row->size == frozen[t].size && row->predelayMs == frozen[t].pre);
    }

    // Row 6 IS the defaults, field for field, and the default type points at it.
    {
        const ReverbParams d;
        const ReverbPreset* row = findReverbPreset ((int) ReverbType::MediumHall);
        CHECK (d.type == (std::uint8_t) ReverbType::MediumHall);
        CHECK (row != nullptr);
        if (row != nullptr)
            CHECK (row->model == d.model && row->erProfile == d.erProfile && row->erLevelDb == d.erLevelDb
                   && row->predelayMs == d.predelayMs && row->rt60 == d.rt60
                   && row->rt60LowMult == d.rt60LowMult && row->rt60HighMult == d.rt60HighMult
                   && row->crossoverLow == d.crossoverLow && row->crossoverHigh == d.crossoverHigh
                   && row->diffusion == d.diffusion && row->size == d.size && row->modRateHz == d.modRateHz
                   && row->modDepth == d.modDepth && row->shimmerPitch == d.shimmerPitch
                   && row->shimmerAmount == d.shimmerAmount);
    }

    // Expanding writes the type, the model and the room, and leaves tone, mix
    // and bypass exactly as the player left them.
    ReverbParams p;
    p.toneHz = 6543.0f;
    p.mix = 42.0f;
    p.bypass = 0;
    CHECK (applyReverbPreset (p, (int) ReverbType::VocalPlate));
    CHECK (p.type == (std::uint8_t) ReverbType::VocalPlate);
    CHECK (p.model == (std::uint8_t) ReverbModel::Plate);
    CHECK (p.rt60 == 1.6f && p.size == 0.9f && p.predelayMs == 25.0f && p.modDepth == 45.0f);
    CHECK (p.toneHz == 6543.0f && p.mix == 42.0f && p.bypass == 0);

    CHECK (applyReverbPreset (p, (int) ReverbType::ShimmerFifthOctave));
    CHECK (p.model == (std::uint8_t) ReverbModel::Shimmer
           && p.shimmerPitch == (std::uint8_t) ShimmerInterval::FifthAndOctave && p.shimmerAmount == 55.0f);

    CHECK (applyReverbPreset (p, (int) ReverbType::StoneCathedral));
    CHECK (p.model == (std::uint8_t) ReverbModel::ModulatedHall && p.erProfile == (std::uint8_t) ErProfile::Cathedral
           && p.erLevelDb == -8.0f);

    // An id with no row is an answer, not a failure: the caller keeps what it
    // has, which is what makes Custom a state rather than a special case.
    const ReverbParams before = p;
    CHECK (! applyReverbPreset (p, (int) ReverbType::Custom));
    CHECK (! applyReverbPreset (p, 99));
    CHECK (p.type == before.type && p.model == before.model && p.rt60 == before.rt60 && p.size == before.size);

    // ...and every shipped row is a reverb that actually makes a sound.
    for (int t = 0; t < count; ++t)
    {
        if (findReverbPreset (t) == nullptr)
            continue;

        EffectReverbModule m;
        m.prepare (module_test::config (48000.0, 256));

        EffectChannelParams cp = reverb_test::params (100.0f, 0.0f);
        CHECK (applyReverbPreset (cp.reverb, t));               // predelay comes from the row
        CHECK (! m.applyParams (cp, 0).variantPending);

        const std::vector<float> ir = reverb_test::impulseResponse (m, 16384);

        bool finite = true;
        for (int i = 0; i < 16384; ++i)
            finite = finite && std::isfinite (ir[(size_t) i]);

        CHECK (finite);
        CHECK (reverb_test::windowDb (ir, 4096, 16384) > -70.0);
    }
}

//==============================================================================
// effects/modules/reverb - the tank primitives the reverb models share
//==============================================================================

static void testReverbDelayLineReads()
{
    using namespace spatcore::effects;

    ReverbDelayLine line;
    line.prepare (100);
    CHECK (line.isPrepared());
    CHECK (line.getMaxDelaySamples() == 100);

    // "Written d writes ago", through several wraps of a ring that is not a
    // power of two (103 slots).
    for (int i = 0; i < 1000; ++i)
        line.write ((float) i);

    CHECK (line.readInteger (1) == 999.0f);
    CHECK (line.readInteger (5) == 995.0f);
    CHECK (line.readInteger (100) == 900.0f);
    CHECK (line.readInteger (0) == line.readInteger (1));           // clamped up
    CHECK (line.readInteger (-7) == line.readInteger (1));
    CHECK (line.readInteger (5000) == line.readInteger (100));      // clamped down

    // Hermite at an integer delay IS the stored sample: every term but the
    // constant is multiplied by a zero fraction.
    std::vector<float> noise = module_test::awkwardBlock (400, 7);
    for (float x : noise)
        line.write (x);

    bool exact = true;
    for (int d = 2; d <= 100; ++d)
        exact = exact && bitEqualFloat (line.readHermite ((float) d), line.readInteger (d));
    CHECK (exact);

    // Out-of-range and non-finite delays land on the ends instead of indexing
    // wildly.
    CHECK (bitEqualFloat (line.readHermite (std::numeric_limits<float>::quiet_NaN()), line.readHermite (2.0f)));
    CHECK (bitEqualFloat (line.readHermite (-3.0f), line.readHermite (2.0f)));
    CHECK (bitEqualFloat (line.readHermite (1.0e9f), line.readHermite (100.0f)));

    // Half way between two samples it interpolates rather than picking one.
    ReverbDelayLine ramp;
    ramp.prepare (16);
    for (int i = 0; i < 16; ++i)
        ramp.write ((float) i);                     // a straight line: cubic fits it exactly
    CHECK (std::fabs (ramp.readHermite (4.5f) - 11.5f) < 1.0e-5f);   // 12 at delay 4, 11 at 5

    line.reset();
    CHECK (line.readInteger (1) == 0.0f && line.readInteger (100) == 0.0f);
}

static void testReverbDelayLineHermiteIsPassive()
{
    using namespace spatcore::effects;

    // A feedback loop multiplies whatever gain its interpolator has at every
    // frequency on every pass, so a read with |H| > 1 anywhere is a slow
    // explosion. The kernel's four weights are recovered by reading an impulse,
    // then |H(w)| is evaluated on a dense grid for fractions across [0, 1).
    auto weights = [] (float frac, double w[4])
    {
        for (int k = 0; k < 4; ++k)
        {
            ReverbDelayLine line;
            line.prepare (32);
            // Impulse at delay 10 + (k - 1): positions 9, 10, 11, 12 around the
            // read at 10 + frac.
            const int impulseDelay = 10 + (k - 1);
            for (int i = 0; i < 40; ++i)
                line.write (i == 40 - impulseDelay ? 1.0f : 0.0f);
            w[k] = (double) line.readHermite (10.0f + frac);
        }
    };

    double worst = 0.0;
    for (int fi = 0; fi < 100; ++fi)
    {
        const float frac = (float) fi * 0.01f;
        double w[4];
        weights (frac, w);

        for (int g = 0; g <= 400; ++g)
        {
            const double omega = 3.141592653589793 * (double) g / 400.0;
            double re = 0.0, im = 0.0;
            for (int k = 0; k < 4; ++k)
            {
                re += w[k] * std::cos (omega * (double) k);
                im -= w[k] * std::sin (omega * (double) k);
            }
            const double mag = std::sqrt (re * re + im * im);
            worst = mag > worst ? mag : worst;
        }
    }
    CHECK (worst <= 1.0 + 1.0e-6);

    // And the reason it is here instead of the linear read: at a fraction of
    // one half, the per-pass loss at fs/8 and fs/4.
    auto lossDb = [&] (bool hermite, double cyclesPerSample)
    {
        double w[4] = { 0.0, 0.5, 0.5, 0.0 };       // linear: the two middle taps
        if (hermite)
            weights (0.5f, w);

        const double omega = 6.283185307179586 * cyclesPerSample;
        double re = 0.0, im = 0.0;
        for (int k = 0; k < 4; ++k)
        {
            re += w[k] * std::cos (omega * (double) k);
            im -= w[k] * std::sin (omega * (double) k);
        }
        return 20.0 * std::log10 (std::sqrt (re * re + im * im));
    };

    CHECK (lossDb (true, 0.125) > -0.2);            // measures -0.07
    CHECK (lossDb (true, 0.25) > -1.3);             // measures -1.07
    CHECK (lossDb (false, 0.125) < -0.5);           // the linear read: -0.69
    CHECK (lossDb (false, 0.25) < -2.5);            // ...and -3.01
}

static void testReverbLfoSine()
{
    using namespace spatcore::effects;

    // The libm-free sine against the library's, over a cycle and past it.
    double worst = 0.0;
    for (int i = -10000; i <= 20000; ++i)
    {
        const double p = (double) i / 10000.0;
        const double err = std::fabs ((double) ReverbLfo::sin2pi (p) - std::sin (6.283185307179586 * p));
        worst = err > worst ? err : worst;
    }
    CHECK (worst < 5.0e-6);

    ReverbLfo lfo;
    lfo.prepare (48000.0);
    lfo.setRateHz (1.0f);
    lfo.setStartPhase (0.0);
    lfo.reset();

    // A quarter of a cycle at 1 Hz is 12000 samples at 48 kHz: sin 1, cos 0.
    for (int i = 0; i < 12000; ++i)
        lfo.nextSin();
    float s = 0.0f, c = 0.0f;
    lfo.nextSinCos (s, c);
    CHECK (std::fabs (s - 1.0f) < 1.0e-5f && std::fabs (c) < 1.0e-5f);

    // Quadrature holds everywhere, and the start phase is where reset() lands.
    bool unit = true;
    for (int i = 0; i < 5000; ++i)
    {
        lfo.nextSinCos (s, c);
        unit = unit && std::fabs (s * s + c * c - 1.0f) < 1.0e-5f;
    }
    CHECK (unit);

    lfo.setStartPhase (1.25);                       // wraps to a quarter
    lfo.reset();
    CHECK (std::fabs (lfo.nextSin() - 1.0f) < 1.0e-5f);

    // Same settings, same stream, to the bit.
    ReverbLfo a, b;
    a.prepare (96000.0);  b.prepare (96000.0);
    a.setRateHz (0.37f);  b.setRateHz (0.37f);
    a.setStartPhase (0.6); b.setStartPhase (0.6);
    a.reset();            b.reset();
    bool same = true;
    for (int i = 0; i < 20000; ++i)
        same = same && bitEqualFloat (a.nextSin(), b.nextSin());
    CHECK (same);

    // A negative rate holds the phase; an absurd one is clamped to 20 Hz.
    ReverbLfo held;
    held.prepare (48000.0);
    held.setRateHz (-3.0f);
    held.reset();
    const double before = held.getPhase();
    for (int i = 0; i < 100; ++i)
        held.nextSin();
    CHECK (held.getPhase() == before);

    held.setRateHz (1000.0f);
    held.reset();
    for (int i = 0; i < 600; ++i)                   // 12.5 ms: a quarter cycle at 20 Hz
        held.nextSin();                             // (unclamped, 1000 Hz would be at a half)
    CHECK (std::fabs (held.getPhase() - 0.25) < 1.0e-9);
}

//==============================================================================
// effects/modules/reverb/EarlyReflections - the profiles and their taps
//==============================================================================

static void testEarlyReflectionProfiles()
{
    using namespace spatcore::effects;

    // The tables are generated (tools/reverb/gen_er_profiles.py). Pinned here
    // is what the module relies on: every room has taps, in arrival order, no
    // earlier than 2 ms, carrying a quarter of the dry's energy; Off has none;
    // anything unknown is Off.
    CHECK (resolveErProfile (-1) == 0 && resolveErProfile (0) == 0);
    CHECK (resolveErProfile ((int) ErProfile::Cathedral) == (int) ErProfile::Cathedral);
    CHECK (resolveErProfile ((int) ErProfile::Count) == 0 && resolveErProfile (99) == 0);
    CHECK (kErProfiles[0].numTaps == 0 && kErProfiles[0].tailDelayMs == 0.0f);

    const int expectTaps[] = { 0, 16, 18, 20, 24 };
    for (int p = 1; p < (int) ErProfile::Count; ++p)
    {
        const ErProfileSpec& spec = kErProfiles[p];
        CHECK (spec.numTaps == expectTaps[p] && spec.numTaps <= ErTapSet::kMaxTaps);

        double energy = 0.0;
        bool ordered = true, early = true, hasFirst = false;
        for (int j = 0; j < spec.numTaps; ++j)
        {
            energy += (double) spec.taps[j].gain * (double) spec.taps[j].gain;
            ordered = ordered && (j == 0 || spec.taps[j].ms >= spec.taps[j - 1].ms);
            early = early && spec.taps[j].ms >= 2.0f;
            hasFirst = hasFirst || spec.taps[j].order == 1;
        }
        CHECK (std::fabs (energy - 0.25) < 0.0025);
        CHECK (ordered && early && hasFirst);
        CHECK (spec.tailDelayMs > 0.0f && spec.darkHz >= 1000.0f);
    }

    // Bigger rooms, later reflections.
    CHECK (kErRoom[15].ms < kErChamber[17].ms && kErChamber[17].ms < kErHall[19].ms
           && kErHall[19].ms < kErCathedral[23].ms);

    // Built for a channel: every tap where the table puts it, scaled by Size
    // and moved by at most 4 %; first order first and positive, higher orders
    // with signs drawn; the tail delay scaled but not jittered.
    const double sr = 48000.0;
    for (int p = 1; p < (int) ErProfile::Count; ++p)
    {
        const ErProfileSpec& spec = kErProfiles[p];

        for (float size : { 0.5f, 1.0f, 2.0f })
        {
            ErTapSet set;
            buildErTapSet (set, p, size, sr, 7u);
            CHECK (set.profile == p && set.isOn() && set.numTaps == spec.numTaps);
            CHECK (set.tailDelay == (int) ((double) spec.tailDelayMs * size * 48.0 + 0.5));
            CHECK (set.darkCoef > 0.0f && set.darkCoef < 1.0f);

            int k = 0, numFirst = 0, negatives = 0, moved = 0, maxRead = set.tailDelay;
            bool placed = true, firstPositive = true;

            for (int pass = 0; pass < 2; ++pass)
            {
                for (int j = 0; j < spec.numTaps; ++j)
                {
                    const ErTapSpec& t = spec.taps[j];
                    if ((t.order == 1) != (pass == 0))
                        continue;

                    const double nominal = (double) t.ms * size * 48.0;
                    const int d = set.delay[k];
                    placed = placed && d >= (int) (nominal * 0.96) - 1 && d <= (int) (nominal * 1.04) + 1
                                    && std::fabs (set.gain[k]) == t.gain;

                    if (pass == 0)
                    {
                        firstPositive = firstPositive && set.gain[k] > 0.0f;
                        ++numFirst;
                    }
                    else if (set.gain[k] < 0.0f)
                    {
                        ++negatives;
                    }

                    moved += d != (int) (nominal + 0.5) ? 1 : 0;
                    maxRead = d > maxRead ? d : maxRead;
                    ++k;
                }
            }

            CHECK (placed && firstPositive);
            CHECK (set.numFirst == numFirst);
            CHECK (negatives > 0 && negatives < spec.numTaps - numFirst);  // drawn, not all one way
            CHECK (moved > spec.numTaps / 2);                              // jittered, not merely rounded
            CHECK (set.maxRead == maxRead);
            CHECK (set.maxRead < erMaxReadSamples (sr));
        }
    }

    // Per channel: the same key builds the same pattern, another key another.
    ErTapSet a, b, c;
    buildErTapSet (a, 2, 1.0f, sr, 5u);
    buildErTapSet (b, 2, 1.0f, sr, 5u);
    buildErTapSet (c, 2, 1.0f, sr, 6u);
    bool same = true, differs = false;
    for (int j = 0; j < a.numTaps; ++j)
    {
        same = same && a.delay[j] == b.delay[j] && a.gain[j] == b.gain[j];
        differs = differs || a.delay[j] != c.delay[j] || a.gain[j] != c.gain[j];
    }
    CHECK (same && differs);

    // Off has nothing - no taps, no tail delay - at any size, and so does an
    // id this build does not know.
    ErTapSet off;
    buildErTapSet (off, 0, 1.7f, sr, 5u);
    CHECK (! off.isOn() && off.numTaps == 0 && off.tailDelay == 0 && off.maxRead == 0);
    buildErTapSet (off, 99, 1.0f, sr, 5u);
    CHECK (! off.isOn() && off.numTaps == 0 && off.tailDelay == 0);

    // matches(): Off is Off at any size; a room is itself at its own size only.
    CHECK (off.matches (0, 1.3f) && ! off.matches (1, 1.0f));
    CHECK (a.matches (2, 1.0f) && ! a.matches (2, 1.01f) && ! a.matches (3, 1.0f) && ! a.matches (0, 1.0f));

    // The ring the module sizes from erMaxReadSamples() holds the longest read
    // at every rate.
    ErTapSet big;
    buildErTapSet (big, (int) ErProfile::Cathedral, 2.0f, 96000.0, 5u);
    CHECK (big.maxRead < erMaxReadSamples (96000.0) && big.maxRead > erMaxReadSamples (48000.0));
}

namespace reverb_test
{
    /** Mean power over [f0, f1] in dB, from `probes` single-frequency DFTs of
        the whole buffer - for a response that ends inside it. */
    inline double meanPowerDb (const std::vector<float>& v, double f0, double f1, double sampleRate, int probes)
    {
        double total = 0.0;
        for (int k = 0; k < probes; ++k)
        {
            const double w = 6.283185307179586 * (f0 + (f1 - f0) * (double) k / (double) (probes - 1)) / sampleRate;
            double re = 0.0, im = 0.0;
            for (size_t i = 0; i < v.size(); ++i)
            {
                re += (double) v[i] * std::cos (w * (double) i);
                im -= (double) v[i] * std::sin (w * (double) i);
            }
            total += re * re + im * im;
        }
        return 10.0 * std::log10 (total / (double) probes);
    }

    /** The reflections alone. With them on, the tail's input is the ring a tail
        delay late; with them off and the predelay lengthened by exactly that,
        the tail is the same tail at the same time, so the difference of the
        two impulse responses is the reflections and nothing else - to the
        rounding. Sizes whose tail delay is a whole number of samples only. */
    inline std::vector<float> reflectionsOf (int profile, float size, float levelDb, float toneHz, int n,
                                             std::uint32_t key, spatcore::effects::ErTapSet& taps)
    {
        using namespace spatcore::effects;
        const ChainConfig cfg = module_test::config (48000.0, 256, key);

        EffectReverbModule on, off;
        on.prepare (cfg);
        off.prepare (cfg);

        EffectChannelParams p = params (100.0f, 0.0f);
        p.reverb.size = size;
        p.reverb.toneHz = toneHz;
        p.reverb.erProfile = (std::uint8_t) profile;
        p.reverb.erLevelDb = levelDb;
        on.applyParams (p, 0);
        taps = on.getActiveReflections();

        p.reverb.erProfile = 0;
        p.reverb.predelayMs = (float) taps.tailDelay / 48.0f;
        off.applyParams (p, 0);

        const std::vector<float> a = impulseResponse (on, n);
        const std::vector<float> b = impulseResponse (off, n);
        std::vector<float> d ((size_t) n);
        for (int i = 0; i < n; ++i)
            d[(size_t) i] = a[(size_t) i] - b[(size_t) i];
        return d;
    }
}

static void testEffectReverbEarlyReflections()
{
    using namespace spatcore::effects;

    const double sr = 48000.0;
    const float tone = erOnePoleCoef (20000.0f, sr);

    struct Case { int profile; float size; int n; };
    for (const Case& c : { Case { (int) ErProfile::Room, 1.0f, 4096 },
                           Case { (int) ErProfile::Hall, 1.0f, 8192 },
                           Case { (int) ErProfile::Cathedral, 2.0f, 20480 } })
    {
        ErTapSet taps;
        const std::vector<float> er = reverb_test::reflectionsOf (c.profile, c.size, 0.0f, 20000.0f, c.n, 9u, taps);
        CHECK (taps.profile == c.profile && taps.numTaps > 0);

        int firstTap = c.n, lastTap = 0;
        for (int j = 0; j < taps.numTaps; ++j)
        {
            firstTap = taps.delay[j] < firstTap ? taps.delay[j] : firstTap;
            lastTap = taps.delay[j] > lastTap ? taps.delay[j] : lastTap;
        }

        // Nothing before the first reflection, and after the last nothing but
        // two filters dying away: so the tail really moved by exactly the tail
        // delay, and the reflections are all there is between.
        float before = 0.0f, after = 0.0f;
        for (int i = 0; i < firstTap; ++i)
            before = std::fabs (er[(size_t) i]) > before ? std::fabs (er[(size_t) i]) : before;
        for (int i = lastTap + 64; i < c.n; ++i)
            after = std::fabs (er[(size_t) i]) > after ? std::fabs (er[(size_t) i]) : after;
        CHECK (before == 0.0f);
        CHECK (after < 1.0e-6f);

        // A first-order reflection is one impulse of 2 g (the wet make-up)
        // through the tone filter, where the table and the jitter put it. A
        // higher-order one goes through the dark filter first.
        bool firstLanded = true, higherDarkened = true;
        for (int j = 0; j < taps.numTaps; ++j)
        {
            bool isolated = true;
            for (int k = 0; k < taps.numTaps; ++k)
                isolated = isolated && (k == j || std::abs (taps.delay[k] - taps.delay[j]) >= 24);
            if (! isolated)
                continue;

            const float expect = 2.0f * tone * taps.gain[j] * (j < taps.numFirst ? 1.0f : taps.darkCoef);
            const bool ok = std::fabs (er[(size_t) taps.delay[j]] - expect) < 2.0e-3f;
            if (j < taps.numFirst)
                firstLanded = firstLanded && ok;
            else
                higherDarkened = higherDarkened && ok;
        }
        CHECK (firstLanded && higherDarkened);

        // At 0 dB the reflections carry the dry's energy across the band where
        // music lives (the dry here is an impulse: 0 dB everywhere).
        const double level = reverb_test::meanPowerDb (er, 200.0, 4000.0, sr, 128);
        CHECK (std::fabs (level) < 1.5);
    }

    // ER Level is a gain on the reflections and on nothing else.
    {
        ErTapSet t0, t6;
        const std::vector<float> at0 = reverb_test::reflectionsOf ((int) ErProfile::Chamber, 1.0f, 0.0f, 12000.0f, 4096, 9u, t0);
        const std::vector<float> at6 = reverb_test::reflectionsOf ((int) ErProfile::Chamber, 1.0f, -6.0f, 12000.0f, 4096, 9u, t6);
        const float g = spatcore::dsp::FastDecibels::dbToGain (-6.0f);
        float worst = 0.0f, peak = 0.0f;
        for (int i = 0; i < 4096; ++i)
        {
            worst = std::fabs (at6[(size_t) i] - g * at0[(size_t) i]) > worst ? std::fabs (at6[(size_t) i] - g * at0[(size_t) i]) : worst;
            peak = std::fabs (at0[(size_t) i]) > peak ? std::fabs (at0[(size_t) i]) : peak;
        }
        CHECK (peak > 0.1f && worst < 1.0e-5f);
    }

    // Each channel its own pattern: the module builds from the chain's key.
    {
        EffectReverbModule a, b;
        a.prepare (module_test::config (48000.0, 256, 5));
        b.prepare (module_test::config (48000.0, 256, 6));
        EffectChannelParams p = reverb_test::params (100.0f, 0.0f);
        p.reverb.erProfile = (std::uint8_t) ErProfile::Hall;
        a.applyParams (p, 0);
        b.applyParams (p, 0);
        bool differs = false;
        for (int j = 0; j < a.getActiveReflections().numTaps; ++j)
            differs = differs || a.getActiveReflections().delay[j] != b.getActiveReflections().delay[j];
        CHECK (differs);
    }

    // Unknown profile ids are Off, to the bit; a NaN level is the floor.
    {
        EffectReverbModule off, unknown, floorLevel, nanLevel;
        for (EffectReverbModule* m : { &off, &unknown, &floorLevel, &nanLevel })
            m->prepare (module_test::config (48000.0, 256));

        EffectChannelParams p = reverb_test::params (100.0f, 3.0f);
        off.applyParams (p, 0);
        p.reverb.erProfile = 77;
        unknown.applyParams (p, 0);
        CHECK (! unknown.getActiveReflections().isOn());
        CHECK (eqtests::bitEqualBlock (reverb_test::impulseResponse (off, 4096),
                                       reverb_test::impulseResponse (unknown, 4096)));

        p.reverb.erProfile = (std::uint8_t) ErProfile::Room;
        p.reverb.erLevelDb = EffectReverbModule::kMinErLevelDb;
        floorLevel.applyParams (p, 0);
        p.reverb.erLevelDb = std::numeric_limits<float>::quiet_NaN();
        nanLevel.applyParams (p, 0);
        CHECK (eqtests::bitEqualBlock (reverb_test::impulseResponse (floorLevel, 4096),
                                       reverb_test::impulseResponse (nanLevel, 4096)));
    }

    // A level change is a runtime value: it glides, it does not spill.
    {
        EffectReverbModule m;
        m.prepare (module_test::config (48000.0, 256));
        EffectChannelParams p = reverb_test::params (100.0f, 0.0f);
        p.reverb.erProfile = (std::uint8_t) ErProfile::Room;
        m.applyParams (p, 0);
        std::vector<float> excite = module_test::awkwardBlock (2048, 3);
        module_test::render (m, excite);
        p.reverb.erLevelDb = 3.0f;
        m.applyParams (p, 0);
        CHECK (m.getSpillVoices() == 0 && ! m.isTransitionWaiting());
    }

    // Settled, the block size is still only a buffer size with reflections on.
    {
        auto renderChunked = [] (int maxBlock)
        {
            EffectReverbModule m;
            m.prepare (module_test::config (48000.0, maxBlock));
            EffectChannelParams p = reverb_test::params (100.0f, 7.3f);
            p.reverb.erProfile = (std::uint8_t) ErProfile::Cathedral;
            p.reverb.size = 1.3f;
            m.applyParams (p, 0);

            std::vector<float> buf = module_test::awkwardBlock (16384, 4);
            for (int done = 0; done < 16384; )
            {
                const int chunk = (16384 - done) < maxBlock ? (16384 - done) : maxBlock;
                m.process (buf.data() + done, chunk);
                done += chunk;
            }
            return buf;
        };

        CHECK (eqtests::bitEqualBlock (renderChunked (1), renderChunked (4096)));
        CHECK (eqtests::bitEqualBlock (renderChunked (64), renderChunked (256)));
    }
}

static void testEffectReverbReflectionSpillover()
{
    using namespace spatcore::effects;

    // THE PARTITION, with reflections. A world with reflections reads its taps
    // and its tail input out of a ring, up to a quarter of a second after the
    // sample was written - so a change can only be an exact partition if the
    // old world keeps reading after it, weighted by when each sample was
    // WRITTEN. Room to Hall, Hall to Off, Off to Cathedral at Size 2, and a
    // size change inside Chamber: each must be the old world fed the input
    // written before the change plus the new one fed the rest.
    const ChainConfig cfg = module_test::config (48000.0, 256);
    const int before = 6000, n = 48000;
    const int fade = (int) (EffectReverbModule::kSpillFadeSeconds * 48000.0 + 0.5);

    struct Change { int fromEr; float fromSize; int toEr; float toSize; };
    for (const Change& c : { Change { 1, 1.0f, 3, 1.0f },
                             Change { 3, 1.0f, 0, 1.0f },
                             Change { 0, 1.0f, 4, 2.0f },
                             Change { 2, 0.7f, 2, 1.4f } })
    {
        std::vector<float> in = module_test::awkwardBlock (n, 23);
        for (int i = before + 3000; i < n; ++i)
            in[(size_t) i] = 0.0f;

        EffectChannelParams from = reverb_test::params (100.0f, 0.0f);
        from.reverb.rt60 = 2.0f;
        from.reverb.erLevelDb = 0.0f;
        from.reverb.erProfile = (std::uint8_t) c.fromEr;
        from.reverb.size = c.fromSize;
        EffectChannelParams to = from;
        to.reverb.erProfile = (std::uint8_t) c.toEr;
        to.reverb.size = c.toSize;

        EffectReverbModule spill;
        spill.prepare (cfg);
        spill.applyParams (from, 0);
        std::vector<float> out = in;
        spill.process (out.data(), before);
        spill.applyParams (to, 0);
        CHECK (spill.getSpillVoices() == 1);
        spill.process (out.data() + before, n - before);

        EffectReverbModule oldWorld, newWorld;
        oldWorld.prepare (cfg);
        newWorld.prepare (cfg);
        oldWorld.applyParams (from, 0);
        newWorld.applyParams (to, 0);

        std::vector<float> outOld = in, outNew ((size_t) n, 0.0f);
        for (int i = before; i < n; ++i)
        {
            const int k = i - before;
            const double s = k < fade ? std::sin (3.141592653589793 * (double) k / (2.0 * fade)) : 1.0;
            const float g = (float) (s * s);
            outOld[(size_t) i] = in[(size_t) i] * (1.0f - g);
            outNew[(size_t) i] = in[(size_t) i] * g;
        }
        module_test::render (oldWorld, outOld);
        module_test::render (newWorld, outNew);

        float worst = 0.0f, peak = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            const float d = std::fabs (out[(size_t) i] - (outOld[(size_t) i] + outNew[(size_t) i]));
            worst = d > worst ? d : worst;
            peak = std::fabs (out[(size_t) i]) > peak ? std::fabs (out[(size_t) i]) : peak;
        }
        CHECK (peak > 0.1f);
        CHECK (worst < 1.0e-4f * peak);
    }
}

static void testEffectReverbReflectionStorm()
{
    using namespace spatcore::effects;

    // A new profile and size every 64-sample block for 40 blocks, on the
    // reverb of a low sine: no step anywhere in the pool, nothing allocated,
    // the same bits twice, and the last change the one that runs.
    const int block = 64, warm = 48000, changes = 40, n = warm + block * changes + 24000;
    const ChainConfig cfg = module_test::config (48000.0, block);
    const int profiles[] = { 3, 0, 4, 1, 2 };
    const float sizes[] = { 1.6f, 0.7f, 1.0f };

    std::vector<float> sine ((size_t) n);
    for (int i = 0; i < n; ++i)
    {
        const double ramp = i < 4800 ? 0.5 - 0.5 * std::cos (3.141592653589793 * (double) i / 4800.0) : 1.0;
        sine[(size_t) i] = (float) (0.5 * ramp * std::sin (6.283185307179586 * 80.0 * (double) i / 48000.0));
    }

    long allocations = -1;
    int lastProfile = -1;

    // combo -1: the storm, from Room at Size 1. 0..14: one of the fifteen
    // (profile, size) pairs the storm visits, held from the first sample; 15:
    // Room at Size 1, held.
    auto render = [&] (int combo)
    {
        const bool fromRoom = combo < 0 || combo == 15;

        EffectReverbModule m;
        m.prepare (cfg);
        EffectChannelParams p = reverb_test::params (100.0f, 0.0f);
        p.reverb.rt60 = 2.0f;
        p.reverb.erLevelDb = 0.0f;
        p.reverb.erProfile = (std::uint8_t) (fromRoom ? (int) ErProfile::Room : profiles[(size_t) (combo % 5)]);
        p.reverb.size = fromRoom ? 1.0f : sizes[(size_t) (combo % 3)];
        m.applyParams (p, 0);

        std::vector<float> buf = sine;
        alloc_probe::Scope probe;

        for (int b = 0; b * block < n; ++b)
        {
            const int at = b * block;
            if (combo < 0 && at >= warm && at < warm + block * changes)
            {
                p.reverb.erProfile = (std::uint8_t) profiles[(size_t) (b % 5)];
                p.reverb.size = sizes[(size_t) (b % 3)];
                m.applyParams (p, 0);
            }
            m.process (buf.data() + at, (n - at) < block ? (n - at) : block);
        }

        allocations = probe.allocations();
        lastProfile = m.getActiveReflections().profile;
        return buf;
    };

    const std::vector<float> busy = render (-1);
    CHECK (allocations == 0);
    CHECK (lastProfile == profiles[(size_t) ((warm / block + changes - 1) % 5)]);

    auto curvature = [] (const std::vector<float>& v, int from, int to)
    {
        float worst = 0.0f;
        for (int i = from; i < to; ++i)
        {
            const float c = std::fabs (v[(size_t) i] - 2.0f * v[(size_t) (i - 1)] + v[(size_t) (i - 2)]);
            worst = c > worst ? c : worst;
        }
        return worst;
    };

    auto peakOf = [] (const std::vector<float>& v, int from, int to)
    {
        float worst = 0.0f;
        for (int i = from; i < to; ++i)
            worst = std::fabs (v[(size_t) i]) > worst ? std::fabs (v[(size_t) i]) : worst;
        return worst;
    };

    // Each room answers the sine at its own level - a reflection pattern
    // summing near-coherently at 80 Hz can double it - so the storm is held
    // against the loudest and sharpest of the rooms it passes through, not
    // against the one it started in.
    const int from = warm, to = warm + block * changes + 20000;
    float cRef = 0.0f, pRef = 0.0f;
    for (int combo = 0; combo <= 15; ++combo)
    {
        const std::vector<float> held = render (combo);
        const float c = curvature (held, from, to), pk = peakOf (held, from, to);
        cRef = c > cRef ? c : cRef;
        pRef = pk > pRef ? pk : pRef;
    }

    const float cBusy = curvature (busy, from, to);
    const float pBusy = peakOf (busy, from, n);

    // Measured: curvature 1.2e-4 against 2.4e-4 for the sharpest held room,
    // peak 1.17 against 2.14. Taps read unweighted across a change give 0.38,
    // the tail's input windowed on the output time instead of the write time
    // 0.027, and the reflection ring left unwritten 1.7e-3 - the smallest.
    bool finite = true;
    for (int i = from; i < n; ++i)
        finite = finite && std::isfinite (busy[(size_t) i]);
    CHECK (finite);
    CHECK (pBusy < 2.0f * pRef);
    CHECK (cBusy < 4.0f * cRef);

    CHECK (eqtests::bitEqualBlock (render (-1), busy));
}

//==============================================================================
// effects/modules - Multitap delay (FxDelay)
//==============================================================================

//==============================================================================
// Multitap delay. The properties pinned here are the ones a rewrite would break
// without making a noise about it: where a tap lands in each pattern mode, that
// the delay time actually MOVES when the LFO runs (and by how much), that the
// FIRST repeat ignores the feedback control and the shelves while the second one
// obeys both, that a boosting shelf inside the loop cannot push it past unity,
// that switching a shelf back on does not fire its frozen history into the loop,
// and that mix 0 hands back the input rather than a crossfade evaluated at g = 1.
//
// Every number asserted below was produced by running this module, not derived
// on paper: the arrival indices, the meter reading, the shelf ratios and the
// decay bounds all come from an actual render.
//==============================================================================

namespace delay_test
{
    using namespace spatcore::effects;

    /** One clean tap: pattern mode with a single tap, no feedback, no
        modulation, no diffusion, fully wet. The input low cut is parked at
        2 kHz deliberately - its impulse response dies within a handful of
        samples, well before the first repeat, so every amplitude asserted below
        is the delay line's arithmetic and not the filter's tail. */
    inline void singleTap (EffectChannelParams& p, float timeMs)
    {
        MultitapParams& d = p.delay;
        d.bypass = 0;         d.taps = 1;           d.tapMode = 1;     d.pattern = 0;
        d.timeMs = timeMs;    d.feedback = 0.0f;    d.feedbackTap = 0;
        d.inLoCutHz = 2000.0f;
        d.fbLoShelfDb = 0.0f; d.fbHiShelfDb = 0.0f;
        d.modRateHz = 0.1f;   d.modDepthPct = 0.0f;
        d.diffusion = 0.0f;   d.glideMs = 0.0f;     d.mix = 100.0f;

        for (int k = 0; k < 8; ++k)
            d.tapLevelDb[k] = 0.0f;
    }

    /** The input low cut's b0. An impulse meets that filter with an empty
        history, so what reaches the line at sample 0 is exactly b0 * 1, and
        every first-arrival amplitude asserted below is a multiple of it. */
    inline float lowCutB0()
    {
        return spatcore::dsp::OutputEQBiquadFilter::calculateCoefficients (1, 2000.0f, 0.0f, 0.6f, 0.7f, 48000.0).b0;
    }

    /** Renders in maxBlock-sized chunks, the way ModuleSlot::process does.
        Handing the module a 40000-sample call instead would drive its
        block-rate filter glides at 1 Hz and hide anything that depends on the
        block length - which is exactly the class of bug a delay with a
        per-sample LFO can have. */
    inline void render (IEffectModule& m, std::vector<float>& buf, int chunk = 256)
    {
        const int n = (int) buf.size();
        int offset = 0;

        while (offset < n)
        {
            const int take = (n - offset) < chunk ? (n - offset) : chunk;
            m.process (buf.data() + offset, take);
            offset += take;
        }
    }

    inline std::vector<float> impulse (IEffectModule& m, int n)
    {
        std::vector<float> buf ((size_t) n, 0.0f);
        buf[0] = 1.0f;
        render (m, buf);
        return buf;
    }

    inline int argMaxAbs (const std::vector<float>& v, int from, int to)
    {
        int best = from;
        for (int i = from; i < to; ++i)
            if (std::fabs (v[(size_t) i]) > std::fabs (v[(size_t) best]))
                best = i;
        return best;
    }

    inline int argMaxAbs (const std::vector<float>& v) { return argMaxAbs (v, 0, (int) v.size()); }

    inline float peakAbs (const std::vector<float>& v, int from, int to)
    {
        float peak = 0.0f;
        for (int i = from; i < to; ++i)
            peak = std::fabs (v[(size_t) i]) > peak ? std::fabs (v[(size_t) i]) : peak;
        return peak;
    }
}

static void testMultitapDelayNeutral()
{
    using namespace spatcore::effects;

    ChainConfig cfg = module_test::config (48000.0, 256);

    // Bypassed in a slot at defaults: the buffer is not touched at all.
    {
        ModuleSlot slot;
        slot.prepare (cfg, std::make_unique<MultitapDelayModule>());
        slot.applyParams (EffectChannelParams(), 0);
        CHECK (slot.isBypassedSettled());

        for (int block = 0; block < 4; ++block)
        {
            std::vector<float> buf = (block == 0) ? eqtests::makeAwkwardSignal (256)
                                                  : module_test::awkwardBlock (256, block);
            const std::vector<float> reference = buf;
            slot.process (buf.data(), 256);
            CHECK (eqtests::bitEqualBlock (buf, reference));
        }
        CHECK (slot.nanTrips.load() == 0);
        CHECK (slot.getLatencySamples() == 0);
    }

    // ACTIVE at mix 0, with a loud feedback setting and the time modulation
    // running underneath, and again with a NaN mix (which must clamp to the low
    // bound, not reach the audio). Bit-identical rather than close: the wet
    // sample is never mixed in, so the negative zeros and denormals come back as
    // they went in.
    for (int variant = 0; variant < 2; ++variant)
    {
        MultitapDelayModule m;
        m.prepare (cfg);

        EffectChannelParams p;
        delay_test::singleTap (p, 10.0f);
        p.delay.feedback = 90.0f;
        p.delay.modRateHz = 5.0f;
        p.delay.modDepthPct = 50.0f;
        p.delay.mix = (variant == 0) ? 0.0f : std::numeric_limits<float>::quiet_NaN();

        const ParamApplyInfo info = m.applyParams (p, 0);
        CHECK (! info.bypass);
        CHECK (! info.variantPending);          // nothing in this module needs silence
        CHECK (m.getLatencySamples() == 0);     // the dry path is not delayed

        for (int block = 0; block < 4; ++block)
        {
            std::vector<float> buf = (block == 0) ? eqtests::makeAwkwardSignal (256)
                                                  : module_test::awkwardBlock (256, block);
            const std::vector<float> reference = buf;
            module_test::render (m, buf);
            CHECK (eqtests::bitEqualBlock (buf, reference));
        }
    }
}

static void testMultitapDelayTapPlacement()
{
    using namespace spatcore::effects;
    namespace fd = spatcore::dsp::FastDecibels;

    const float b0 = delay_test::lowCutB0();
    ChainConfig cfg = module_test::config (48000.0, 256);

    // 10 ms at 48 kHz is 480 samples exactly, and an integer delay reads with
    // an interpolation weight of 0, so the arrival is b0 and it lands on
    // sample 480 and on no other.
    {
        MultitapDelayModule m;
        m.prepare (cfg);
        EffectChannelParams p;
        delay_test::singleTap (p, 10.0f);
        m.applyParams (p, 0);

        const std::vector<float> out = delay_test::impulse (m, 2000);
        CHECK (out[479] == 0.0f);                          // nothing arrives early
        CHECK (std::fabs (out[480] - b0) < 1.0e-6f);
        CHECK (delay_test::argMaxAbs (out) == 480);
    }

    // Pattern Equal is base*k, so three taps land on 480 / 960 / 1440, each
    // scaled by its own level (-2 and -4 dB are the plan's tap defaults).
    {
        MultitapDelayModule m;
        m.prepare (cfg);
        EffectChannelParams p;
        delay_test::singleTap (p, 10.0f);
        p.delay.taps = 3;
        p.delay.tapLevelDb[1] = -2.0f;
        p.delay.tapLevelDb[2] = -4.0f;
        m.applyParams (p, 0);

        const std::vector<float> out = delay_test::impulse (m, 2000);
        CHECK (std::fabs (out[480]  - b0) < 1.0e-6f);
        CHECK (std::fabs (out[960]  - b0 * fd::dbToGain (-2.0f)) < 1.0e-6f);
        CHECK (std::fabs (out[1440] - b0 * fd::dbToGain (-4.0f)) < 1.0e-6f);
    }

    // Dotted is 1.5*base*k: the same base puts the single tap on 720.
    {
        MultitapDelayModule m;
        m.prepare (cfg);
        EffectChannelParams p;
        delay_test::singleTap (p, 10.0f);
        p.delay.pattern = 1;
        m.applyParams (p, 0);
        CHECK (delay_test::argMaxAbs (delay_test::impulse (m, 2000)) == 720);
    }

    // Triplet is (2/3)*base*k: 320 samples, and it is the one pattern whose
    // tap lands EARLIER than Equal - a swapped case label would show here.
    {
        MultitapDelayModule m;
        m.prepare (cfg);
        EffectChannelParams p;
        delay_test::singleTap (p, 10.0f);
        p.delay.pattern = 2;
        m.applyParams (p, 0);
        CHECK (delay_test::argMaxAbs (delay_test::impulse (m, 2000)) == 320);
    }

    // Golden is base*phi^(k-1): 480, 776.7, 1256.7. The fractional arrivals
    // straddle two samples, so the peak lands on the nearer one.
    {
        MultitapDelayModule m;
        m.prepare (cfg);
        EffectChannelParams p;
        delay_test::singleTap (p, 10.0f);
        p.delay.pattern = 3;
        p.delay.taps = 3;
        m.applyParams (p, 0);

        const std::vector<float> out = delay_test::impulse (m, 3000);
        CHECK (delay_test::argMaxAbs (out,  700,  900) == 777);
        CHECK (delay_test::argMaxAbs (out, 1150, 1400) == 1257);
    }

    // Manual mode reads the per-tap time array and IGNORES the pattern, which
    // is set to Dotted here so that obeying it would be visible at 720.
    {
        MultitapDelayModule m;
        m.prepare (cfg);
        EffectChannelParams p;
        delay_test::singleTap (p, 10.0f);
        p.delay.tapMode = 0;
        p.delay.taps = 2;
        p.delay.tapTimeMs[0] = 5.0f;
        p.delay.tapTimeMs[1] = 12.5f;
        p.delay.pattern = 1;
        m.applyParams (p, 0);

        const std::vector<float> out = delay_test::impulse (m, 2000);
        CHECK (std::fabs (out[240] - b0) < 1.0e-6f);
        CHECK (std::fabs (out[600] - b0) < 1.0e-6f);
        CHECK (std::fabs (out[720]) < 1.0e-9f);
    }
}

static void testMultitapDelayTimeModulation()
{
    using namespace spatcore::effects;

    ChainConfig cfg = module_test::config (48000.0, 256);

    // The module's second characteristic law after tap placement, and the one
    // an inert LFO would pass every other test with: t * (1 + depth*sin), so at
    // a quarter cycle a 480-sample tap reads 720 and at three quarters it reads
    // 240. The impulses are placed where a correctly modulated read head would
    // have to have come from - 12000-720 and 36000-240 - and the echo must land
    // on the LFO extreme itself, where the delay is stationary and the arrival
    // is a single unsmeared sample.
    //
    // Run at BOTH ends of the glide range. Glide must not damp the wobble (a
    // smoother fed the modulated target is a box filter that nulls 5 and 10 Hz
    // outright at the plan's 200 ms default) and must not teleport on it either
    // (at glide 0 the smoother's teleport threshold is six samples).
    for (int variant = 0; variant < 2; ++variant)
    {
        MultitapDelayModule m;
        m.prepare (cfg);

        EffectChannelParams p;
        delay_test::singleTap (p, 10.0f);
        p.delay.modRateHz = 1.0f;               // one cycle per 48000 samples
        p.delay.modDepthPct = 50.0f;
        p.delay.glideMs = (variant == 0) ? 0.0f : 200.0f;
        m.applyParams (p, 0);

        std::vector<float> buf (40000, 0.0f);
        buf[(size_t) 11280] = 1.0f;
        buf[(size_t) 35760] = 1.0f;
        delay_test::render (m, buf);

        CHECK (delay_test::argMaxAbs (buf, 11500, 12500) - 11280 == 720);   // +50 %
        CHECK (delay_test::argMaxAbs (buf, 35500, 36500) - 35760 == 240);   // -50 %
    }

    // Depth 0 is inert to the bit at every rate: the phase still advances every
    // sample, but the multiply is left out rather than applied as 1.0.
    {
        MultitapDelayModule fast, slow;
        fast.prepare (cfg);
        slow.prepare (cfg);

        EffectChannelParams p;
        delay_test::singleTap (p, 10.0f);
        p.delay.modDepthPct = 0.0f;
        p.delay.modRateHz = 7.0f;
        fast.applyParams (p, 0);
        p.delay.modRateHz = 0.02f;
        slow.applyParams (p, 0);

        bool identical = true;
        for (int block = 0; block < 8; ++block)
        {
            std::vector<float> a = module_test::awkwardBlock (256, block + 3);
            std::vector<float> b = a;
            module_test::render (fast, a);
            module_test::render (slow, b);
            if (! eqtests::bitEqualBlock (a, b))
                identical = false;
        }
        CHECK (identical);
    }
}

static void testMultitapDelayBlockSizeInvariance()
{
    using namespace spatcore::effects;

    // Replaces a determinism check between two instances started in the same
    // process on the same block, which cannot fail. This one can: the module is
    // driven with everything moving - three taps, feedback, a 6 Hz time
    // modulation, a shelf in the loop - once in 256-sample chunks and once in
    // 64-sample chunks, and the two renders must agree TO THE BIT.
    //
    // ModuleSlot chunks at maxBlock, so a host that changes its buffer size must
    // not change the sound. It also catches the specific trap in the LFO: a
    // phase advanced once per block instead of once per sample runs numSamples
    // times too slow, which is invisible against a single reference render and
    // glaring here.
    ChainConfig cfg = module_test::config (48000.0, 256);

    MultitapDelayModule a, b;
    a.prepare (cfg);
    b.prepare (cfg);

    EffectChannelParams p;
    delay_test::singleTap (p, 13.0f);
    p.delay.taps = 3;
    p.delay.feedback = 55.0f;
    p.delay.modRateHz = 6.0f;
    p.delay.modDepthPct = 35.0f;
    p.delay.fbHiShelfDb = -6.0f;
    a.applyParams (p, 0);
    b.applyParams (p, 0);

    std::vector<float> x = module_test::awkwardBlock (4096, 3);
    std::vector<float> y = x;
    delay_test::render (a, x, 256);
    delay_test::render (b, y, 64);
    CHECK (eqtests::bitEqualBlock (x, y));
}

static void testMultitapDelayFeedbackLadder()
{
    using namespace spatcore::effects;

    const float b0 = delay_test::lowCutB0();

    const auto run = [] (float feedbackPercent, float loShelfDb, float hiShelfDb, int n)
    {
        MultitapDelayModule m;
        m.prepare (module_test::config (48000.0, 256));
        EffectChannelParams p;
        delay_test::singleTap (p, 10.0f);
        p.delay.feedback = feedbackPercent;
        p.delay.fbLoShelfDb = loShelfDb;
        p.delay.fbHiShelfDb = hiShelfDb;
        m.applyParams (p, 0);
        return delay_test::impulse (m, n);
    };

    // The line holds L[0] = b0, L[480] = f*b0, L[960] = f^2*b0, and a tap reads
    // L 480 samples after it was written. At f = 0.5 that is b0, b0/2, b0/4 on
    // samples 480, 960 and 1440.
    const std::vector<float> half = run (50.0f, 0.0f, 0.0f, 2000);
    CHECK (std::fabs (half[480]  - b0)         < 1.0e-6f);
    CHECK (std::fabs (half[960]  - 0.5f * b0)  < 1.0e-5f);
    CHECK (std::fabs (half[1440] - 0.25f * b0) < 1.0e-5f);

    // The first repeat does not move with the control, and with no feedback
    // there is no second one. A port that took the wet from after the feedback
    // multiply would fail exactly these two and nothing else.
    const std::vector<float> none = run (0.0f,  0.0f, 0.0f, 2000);
    const std::vector<float> lots = run (95.0f, 0.0f, 0.0f, 2000);
    CHECK (std::fabs (none[480] - lots[480]) < 1.0e-6f);
    CHECK (std::fabs (none[960]) < 1.0e-9f);

    // WHERE THE SHELVES SIT, which the ladder above does not pin: they are
    // strictly inside the loop, so a 24 dB high-shelf cut must leave repeat 1
    // bit-for-bit alone and take repeat 2 down. A port that moved them ahead of
    // the wet tap, or outside the loop, passes every other test in this file.
    {
        const std::vector<float> flat = run (60.0f,  0.0f,   0.0f, 2000);
        const std::vector<float> cut  = run (60.0f,  0.0f, -24.0f, 2000);
        CHECK (std::fabs (delay_test::peakAbs (flat, 470, 700)
                        - delay_test::peakAbs (cut,  470, 700)) < 1.0e-6f);
        CHECK (delay_test::peakAbs (cut, 950, 1200) < 0.5f * delay_test::peakAbs (flat, 950, 1200));
    }

    // 95 % feedback with +24 dB on both shelves is a loop gain of 15 unless the
    // ceiling divides it back down. Without the ceiling this reaches ~5e11 by
    // sample 4800; with it the loop still decays, which is the property that
    // matters rather than any hand-guessed magnitude bound.
    {
        const std::vector<float> hot = run (95.0f, 24.0f, 24.0f, 4800);
        bool finite = true;
        for (int i = 0; i < 4800; ++i)
            if (! std::isfinite (hot[(size_t) i]))
                finite = false;
        CHECK (finite);
        CHECK (delay_test::peakAbs (hot, 3800, 4800) < 0.1f * delay_test::peakAbs (hot, 0, 1000));
    }
}

static void testMultitapDelayShelfReactivation()
{
    using namespace spatcore::effects;

    // A shelf at its detent is switched OFF (shape 0), and OutputEQBiquadFilter
    // returns early in that state WITHOUT shifting its delay line - so x1/x2/
    // y1/y2 freeze holding whatever the loop was doing. Switching the shelf back
    // on therefore replays that frozen state as the filter's first output,
    // INSIDE a feedback loop.
    //
    // Measured on this module with the reset removed: the loop is at 1e-7 when
    // the shelf comes back and the stale history fires 2.1e-2 into it - five
    // orders of magnitude, recirculated at up to 0.95 a repeat. With the reset
    // the re-enable is inaudible (ratio 0.9). The bound below sits between the
    // two with a factor of 25000 of margin.
    MultitapDelayModule m;
    m.prepare (module_test::config (48000.0, 256));

    EffectChannelParams p;
    delay_test::singleTap (p, 10.0f);
    p.delay.feedback = 90.0f;
    p.delay.inLoCutHz = 20.0f;
    p.delay.fbLoShelfHz = 2000.0f;
    p.delay.fbLoShelfDb = -24.0f;
    m.applyParams (p, 0);

    for (int block = 0; block < 24; ++block)
    {
        std::vector<float> buf = module_test::awkwardBlock (256, block + 1);
        module_test::render (m, buf);
    }

    p.delay.fbLoShelfDb = 0.0f;                     // detent: the biquad freezes
    m.applyParams (p, 0);

    float tail = 0.0f;
    for (int block = 0; block < 300; ++block)
    {
        std::vector<float> buf (256, 0.0f);
        module_test::render (m, buf);
        tail = delay_test::peakAbs (buf, 0, 256);
    }

    p.delay.fbLoShelfDb = -24.0f;                   // back on, into a quiet loop
    m.applyParams (p, 0);

    float after = 0.0f;
    for (int block = 0; block < 8; ++block)
    {
        std::vector<float> buf (256, 0.0f);
        module_test::render (m, buf);
        const float pk = delay_test::peakAbs (buf, 0, 256);
        after = pk > after ? pk : after;
    }

    CHECK (after <= tail * 8.0f);
}

static void testMultitapDelayDiffusionAndMeter()
{
    using namespace spatcore::effects;
    namespace fd = spatcore::dsp::FastDecibels;

    // Diffusion is two Schroeder allpasses, so it must SMEAR the repeat without
    // changing how much of it there is: same energy to a fifth of a percent,
    // peak down by more than a third. A pair of plain delays or a mis-signed
    // allpass fails one of the two.
    {
        const auto run = [] (float diffusion)
        {
            MultitapDelayModule m;
            m.prepare (module_test::config (48000.0, 256));
            EffectChannelParams p;
            delay_test::singleTap (p, 10.0f);
            p.delay.diffusion = diffusion;
            m.applyParams (p, 0);
            return delay_test::impulse (m, 8192);
        };

        const std::vector<float> dry = run (0.0f);
        const std::vector<float> wet = run (1.0f);

        double dryEnergy = 0.0, wetEnergy = 0.0;
        for (size_t i = 0; i < dry.size(); ++i)
        {
            dryEnergy += (double) dry[i] * (double) dry[i];
            wetEnergy += (double) wet[i] * (double) wet[i];
        }
        CHECK (wetEnergy > dryEnergy * 0.98 && wetEnergy < dryEnergy * 1.02);
        CHECK (delay_test::peakAbs (wet, 0, 8192) < 0.6f * delay_test::peakAbs (dry, 0, 8192));
    }

    // getMeterDb reports the WET peak of the block just processed - before the
    // mix, which is the author's declared choice and the thing to notice if the
    // GUI ever disagrees. It is kMinDb before the repeat arrives, the low cut's
    // b0 on the block that carries it, and back to kMinDb once the line empties.
    {
        MultitapDelayModule m;
        m.prepare (module_test::config (48000.0, 256));
        CHECK (m.getMeterDb() <= fd::kMinDb + 1.0f);

        EffectChannelParams p;
        delay_test::singleTap (p, 10.0f);
        m.applyParams (p, 0);

        std::vector<float> first (256, 0.0f);
        first[0] = 1.0f;
        module_test::render (m, first);
        CHECK (m.getMeterDb() <= fd::kMinDb + 1.0f);

        std::vector<float> second (256, 0.0f);       // samples 256..511: the tap is on 480
        module_test::render (m, second);
        CHECK (std::fabs (m.getMeterDb() - fd::gainToDb (delay_test::lowCutB0())) < 0.1f);

        std::vector<float> silence (48000, 0.0f);
        delay_test::render (m, silence);
        CHECK (m.getMeterDb() <= fd::kMinDb + 1.0f);
    }
}

static void testMultitapDelayResetClearsTail()
{
    using namespace spatcore::effects;

    MultitapDelayModule m;
    m.prepare (module_test::config (48000.0, 256));
    EffectChannelParams p;
    delay_test::singleTap (p, 10.0f);
    p.delay.feedback = 90.0f;
    p.delay.diffusion = 1.0f;
    m.applyParams (p, 0);

    const std::vector<float> excited = delay_test::impulse (m, 2048);
    CHECK (delay_test::peakAbs (excited, 1024, 2048) > 1.0e-3f);   // a tail to lose

    m.reset();

    std::vector<float> silence (2048, 0.0f);
    delay_test::render (m, silence);

    bool silent = true;
    for (int i = 0; i < 2048; ++i)
        if (silence[(size_t) i] != 0.0f)
            silent = false;
    CHECK (silent);

    // Silence is the weak half of the claim. The strong half: a reset instance
    // and a fresh one must be indistinguishable to the audio. Anything reset()
    // forgets - a biquad history, an allpass write position, the LFO phase, a
    // tap smoother part way along a ramp, the block counter the smoothers are
    // indexed by - survives into the impulse response and shows up here, and
    // nowhere else in this file.
    {
        MultitapDelayModule used, fresh;
        used.prepare (module_test::config (48000.0, 256));
        fresh.prepare (module_test::config (48000.0, 256));

        EffectChannelParams q;
        delay_test::singleTap (q, 7.0f);
        q.delay.taps = 3;          q.delay.feedback = 85.0f;
        q.delay.diffusion = 0.9f;  q.delay.glideMs = 150.0f;
        q.delay.modRateHz = 3.0f;  q.delay.modDepthPct = 45.0f;
        q.delay.fbLoShelfDb = 9.0f;
        q.delay.fbHiShelfDb = -15.0f;
        used.applyParams (q, 0);
        fresh.applyParams (q, 0);

        for (int block = 0; block < 30; ++block)
        {
            std::vector<float> dirt = module_test::awkwardBlock (256, block + 61);
            module_test::render (used, dirt);
        }
        used.reset();

        std::vector<float> a (4096, 0.0f), b (4096, 0.0f);
        a[0] = 1.0f;
        b[0] = 1.0f;
        delay_test::render (used, a);
        delay_test::render (fresh, b);
        CHECK (eqtests::bitEqualBlock (a, b));
    }
}

static void testMultitapDelayInSlot()
{
    using namespace spatcore::effects;

    // The module driven the way the chain drives it: faded in, run active,
    // faded out, and reset on the audio thread when the slot settles silent.
    // Nothing else in this file exercises ModuleSlot's crossfade, its
    // silent-reset path or its NaN trip against THIS module.
    ChainConfig cfg = module_test::config (48000.0, 256);

    ModuleSlot slot;
    slot.prepare (cfg, std::make_unique<MultitapDelayModule>());

    EffectChannelParams p;
    delay_test::singleTap (p, 10.0f);
    p.delay.feedback = 70.0f;
    p.delay.diffusion = 0.6f;
    p.delay.modRateHz = 4.0f;
    p.delay.modDepthPct = 40.0f;
    p.delay.mix = 60.0f;
    slot.applyParams (p, 0);

    bool finite = true;
    for (int block = 0; block < 40; ++block)
    {
        std::vector<float> buf = module_test::awkwardBlock (256, block + 11);
        slot.process (buf.data(), 256);
        for (int i = 0; i < 256; ++i)
            if (! std::isfinite (buf[(size_t) i]))
                finite = false;
    }
    CHECK (finite);
    CHECK (slot.isActiveSettled());
    CHECK (slot.nanTrips.load() == 0);

    p.delay.bypass = 1;
    slot.applyParams (p, 0);
    for (int block = 0; block < 40; ++block)
    {
        std::vector<float> buf = module_test::awkwardBlock (256, block + 51);
        slot.process (buf.data(), 256);
    }
    CHECK (slot.isBypassedSettled());
    CHECK (slot.silentResets.load() > 0);          // the 1 MB reset really ran
    CHECK (slot.nanTrips.load() == 0);

    // Settled bypassed again: transparent to the bit, tail and all.
    {
        std::vector<float> buf = eqtests::makeAwkwardSignal (256);
        const std::vector<float> reference = buf;
        slot.process (buf.data(), 256);
        CHECK (eqtests::bitEqualBlock (buf, reference));
    }
}

static void testMultitapDelayExtremesAndRates()
{
    using namespace spatcore::effects;

    ChainConfig cfg = module_test::config (48000.0, 256);

    // Every control at an end stop, then the same with a NaN on the four a
    // broken publisher is most likely to send. The negated clamps must land
    // each of them on its low bound instead of letting it into the audio.
    //
    // The bound is that the loop DECAYS - the peak of the last block is under a
    // twentieth of the loudest block seen. An absolute magnitude bound here is
    // either vacuous or flaky; this is the property the feedback ceiling exists
    // to guarantee, and the measured margin is a factor of 200 or better.
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();

        for (int variant = 0; variant < 3; ++variant)
        {
            MultitapDelayModule m;
            m.prepare (cfg);

            EffectChannelParams p;
            MultitapParams& d = p.delay;
            d.bypass = 0;   d.taps = 8;   d.pattern = 3;      // golden: the widest spread
            d.tapMode = (variant == 1) ? 0 : 1;
            d.feedbackTap = (variant == 1) ? 4 : 0;
            d.timeMs      = (variant == 2) ? nan : 20.0f;
            d.feedback    = (variant == 2) ? nan : 95.0f;
            d.modDepthPct = (variant == 2) ? nan : 50.0f;
            d.diffusion   = (variant == 2) ? nan : 1.0f;
            d.glideMs     = (variant == 0) ? 0.0f : 2000.0f;
            d.modRateHz = 10.0f;
            d.fbLoShelfDb = 24.0f;  d.fbHiShelfDb = -24.0f;  d.mix = 100.0f;

            for (int k = 0; k < 8; ++k)
                d.tapTimeMs[k] = 3.0f * (float) (k + 1);

            m.applyParams (p, 0);

            bool finite = true;
            float worst = 0.0f, last = 0.0f;

            for (int block = 0; block < 200; ++block)
            {
                std::vector<float> buf = (block < 8) ? module_test::awkwardBlock (256, block + 1)
                                                     : std::vector<float> (256, 0.0f);
                module_test::render (m, buf);

                for (int i = 0; i < 256; ++i)
                    if (! std::isfinite (buf[(size_t) i]))
                        finite = false;

                const float peak = delay_test::peakAbs (buf, 0, 256);
                worst = peak > worst ? peak : worst;
                last = peak;
            }

            CHECK (finite);
            CHECK (last < worst * 0.05f);
        }
    }

    // prepare() at a new rate with no applyParams behind it: the tap times, the
    // buffer cap and the glide window are all recomputed against the new rate,
    // so 10 ms is still 10 ms - 960 samples at 96 kHz, not the 480 the previous
    // rate's numbers would give.
    {
        MultitapDelayModule m;
        m.prepare (module_test::config (48000.0, 256));

        EffectChannelParams p;
        delay_test::singleTap (p, 10.0f);
        p.delay.glideMs = 200.0f;
        m.applyParams (p, 0);

        m.prepare (module_test::config (96000.0, 256));
        std::vector<float> buf (4000, 0.0f);
        buf[0] = 1.0f;
        delay_test::render (m, buf);
        CHECK (delay_test::argMaxAbs (buf) == 960);
    }
}

//==============================================================================
// The effects engine's foundations: knowing when a producer has lapped you, and
// giving effect returns a slot of their own in the render-source map.
//==============================================================================

static void testSharedInputRingWrapCounter()
{
    using namespace spatcore::rt;

    SharedInputRingBuffer ring;
    ring.setSize (1024);
    CHECK (ring.getTotalWritten() == 0);

    std::vector<float> block (256);
    for (int i = 0; i < 256; ++i)
        block[(size_t) i] = 0.001f * (float) i;

    // The counter is the number of samples WRITTEN, not the position, so it
    // keeps counting past the wrap.
    for (int b = 0; b < 3; ++b)
        ring.write (block.data(), 256);
    CHECK (ring.getTotalWritten() == 768);

    for (int b = 0; b < 5; ++b)
        ring.write (block.data(), 256);
    CHECK (ring.getTotalWritten() == 2048);      // twice round a 1024 ring

    // This is the whole point: a consumer that has stopped reading can DETECT
    // that it has been lapped. The position alone cannot tell it - after a full
    // buffer the write head is back where the consumer left it, and the
    // available count reads zero, which is indistinguishable from a producer
    // that has not run at all.
    {
        SharedInputRingBuffer r2;
        r2.setSize (1024);

        int cursor = 0;
        std::uint64_t consumed = 0;
        std::vector<float> out (256);

        // Read along politely for a while.
        for (int b = 0; b < 2; ++b)
        {
            r2.write (block.data(), 256);
            consumed += (std::uint64_t) r2.readWithPosition (cursor, out.data(), 256);
        }
        CHECK (consumed == 512);
        CHECK (r2.getTotalWritten() - consumed == 0);

        // Now stall the consumer and let the producer run a whole buffer past it.
        for (int b = 0; b < 4; ++b)
            r2.write (block.data(), 256);

        const std::uint64_t behind = r2.getTotalWritten() - consumed;
        CHECK (behind == 1024);
        CHECK (behind > (std::uint64_t) (r2.getBufferSize() - 256));   // lapped, and knowable

        // The position is NOT knowable: it reads as if nothing had happened.
        CHECK (r2.getAvailableAt (cursor) == 0);
    }

    // reset() and setSize() both zero it, so a resync starts from a clean slate.
    ring.reset();
    CHECK (ring.getTotalWritten() == 0);
    ring.write (block.data(), 256);
    CHECK (ring.getTotalWritten() == 256);
    ring.setSize (512);
    CHECK (ring.getTotalWritten() == 0);
}

static void testRenderSourceMapEffectsLayout()
{
    using Map = spatcore::wfs::RenderSourceMap;
    using spatcore::wfs::SourceKind;

    // 6 inputs, two of them stereo, plus 4 effects channels.
    std::array<uint8_t, 6> types { Map::Mono, Map::Stereo, Map::Mono, Map::Stereo, Map::Mono, Map::Mono };
    Map m;
    CHECK (Map::build (types.data(), 6, 4, m));

    const int expectedFirstEffect = 6 + 2 * Map::kDerivedPerStereo;   // 16
    CHECK (m.numInputChannels == 6);
    CHECK (m.numEffectChannels == 4);
    CHECK (m.firstEffectSlot == expectedFirstEffect);
    CHECK (m.count == expectedFirstEffect + 4);

    for (int fx = 0; fx < 4; ++fx)
    {
        const auto& d = m.desc[(size_t) (m.firstEffectSlot + fx)];
        CHECK (d.kind == SourceKind::EffectReturn);
        CHECK (d.owningEffectChannel == (int16_t) fx);
        CHECK (d.owningInputChannel == -1);     // a return belongs to no input
        CHECK (d.sliceIndex == 0);
        CHECK (! d.isStereoSlice);
        CHECK (d.active);
        CHECK (d.gainLinear == 1.0f);
    }

    // Every input and derived slot is still an Input, so a consumer that keys
    // on kind cannot mistake one for the other.
    for (int i = 0; i < expectedFirstEffect; ++i)
        CHECK (m.desc[(size_t) i].kind == SourceKind::Input);

    // ADDING EFFECTS MOVES NOTHING. This is what lets an engine hold a matrix
    // row index across a channel-count change without it silently changing
    // meaning.
    {
        Map without;
        CHECK (Map::build (types.data(), 6, 0, without));
        CHECK (without.firstEffectSlot == -1);
        CHECK (without.numEffectChannels == 0);
        CHECK (without.count == expectedFirstEffect);

        for (int i = 0; i < without.count; ++i)
        {
            CHECK (m.desc[(size_t) i].owningInputChannel == without.desc[(size_t) i].owningInputChannel);
            CHECK (m.desc[(size_t) i].sliceIndex == without.desc[(size_t) i].sliceIndex);
            CHECK (m.desc[(size_t) i].isStereoSlice == without.desc[(size_t) i].isStereoSlice);
        }
        for (int i = 0; i < 6; ++i)
            CHECK (m.firstDerivedSlot[(size_t) i] == without.firstDerivedSlot[(size_t) i]);
    }

    // The two-argument overload is the old behaviour exactly.
    {
        Map twoArg;
        CHECK (Map::build (types.data(), 6, twoArg));
        CHECK (twoArg.numEffectChannels == 0);
        CHECK (twoArg.firstEffectSlot == -1);
        CHECK (twoArg.count == expectedFirstEffect);
        for (int i = 0; i < twoArg.count; ++i)
            CHECK (twoArg.desc[(size_t) i].kind == SourceKind::Input);
    }

    // The full budget fits, and one more of anything does not.
    {
        std::array<uint8_t, Map::kMaxInputChannels> full {};
        for (int i = 0; i < Map::kMaxStereoChannels; ++i)
            full[(size_t) i] = Map::Stereo;

        Map big;
        CHECK (Map::build (full.data(), Map::kMaxInputChannels, Map::kMaxEffectChannels, big));
        CHECK (big.count == Map::kMaxRenderSourceSlots);
        CHECK (big.count == Map::kMaxInputRenderSources + Map::kMaxEffectChannels);
        CHECK (big.firstEffectSlot == Map::kMaxInputRenderSources);

        Map refused;
        CHECK (! Map::build (full.data(), Map::kMaxInputChannels, Map::kMaxEffectChannels + 1, refused));
        CHECK (refused.count == 0);
        CHECK (! Map::build (full.data(), Map::kMaxInputChannels, -1, refused));
        CHECK (refused.count == 0);
    }

    // buildIdentity still means what it meant.
    {
        Map ident;
        CHECK (Map::buildIdentity (8, ident));
        CHECK (ident.count == 8);
        CHECK (ident.firstEffectSlot == -1);
    }
}

//==============================================================================
// dsp/AcousticSendMatrix - the source range the effects loop guard needs, and
// the history length the second matrix needs.
//==============================================================================

static void testAcousticSendMatrixSourceRange()
{
    using spatcore::dsp::AcousticSendMatrix;

    const double sr = 48000.0;
    const int numSamples = 256;
    const int sources = 6;
    const int nodes = 2;
    const int stride = 8;
    const int split = 4;                  // rows 0..3 are "inputs", 4..5 are "effects"

    // Levels, delays and shelves all non-trivial, so the split has to preserve
    // per-cell smoother and filter state, not just the arithmetic.
    std::vector<float> levels ((size_t) (sources * stride), 0.0f);
    std::vector<float> delays ((size_t) (sources * stride), 0.0f);
    std::vector<float> hf ((size_t) (sources * stride), 0.0f);
    for (int src = 0; src < sources; ++src)
    {
        levels[(size_t) (src * stride + 0)] = 0.2f + 0.1f * (float) src;
        delays[(size_t) (src * stride + 0)] = 1.5f * (float) src;     // fractional ms
        hf[(size_t) (src * stride + 0)] = -1.5f * (float) src;        // engages the shelf
    }

    auto makeInput = [sources, numSamples] (int block)
    {
        juce::AudioBuffer<float> in (sources, numSamples);
        for (int ch = 0; ch < sources; ++ch)
            for (int i = 0; i < numSamples; ++i)
                in.setSample (ch, i, FrDiffusion::hashNoiseBipolar (
                    (std::uint32_t) (block * numSamples + i), (std::uint32_t) (ch + 1)));
        return in;
    };

    AcousticSendMatrix whole, halves;
    whole.prepare (sr, sources, nodes);
    halves.prepare (sr, sources, nodes);

    bool identical = true;

    for (int block = 0; block < 12; ++block)
    {
        const auto in = makeInput (block);

        std::vector<float> a ((size_t) numSamples, -1.0f), b ((size_t) numSamples, -1.0f);

        whole.writeInputs (in, numSamples);
        whole.computeNodeFeed (a.data(), numSamples, 0, levels.data(), delays.data(), hf.data(), stride);
        whole.advance (numSamples);

        // The same feed, rendered as two passes: the second ACCUMULATES.
        halves.writeInputs (in, numSamples);
        halves.computeNodeFeed (b.data(), numSamples, 0, levels.data(), delays.data(), hf.data(),
                                stride, 0, split, true);
        halves.computeNodeFeed (b.data(), numSamples, 0, levels.data(), delays.data(), hf.data(),
                                stride, split, -1, false);
        halves.advance (numSamples);

        if (! eqtests::bitEqualBlock (a, b))
            identical = false;
    }

    CHECK (identical);

    // A range that touches nothing leaves the destination alone when asked to
    // accumulate, and clears it when asked to overwrite.
    {
        AcousticSendMatrix m;
        m.prepare (sr, sources, nodes);
        const auto in = makeInput (0);
        m.writeInputs (in, numSamples);

        std::vector<float> keep ((size_t) numSamples, 7.0f);
        m.computeNodeFeed (keep.data(), numSamples, 0, levels.data(), nullptr, nullptr,
                           stride, 2, 2, false);          // empty range, accumulate
        for (int i = 0; i < numSamples; ++i)
            CHECK (keep[(size_t) i] == 7.0f);

        m.computeNodeFeed (keep.data(), numSamples, 0, levels.data(), nullptr, nullptr,
                           stride, 2, 2, true);           // empty range, overwrite
        for (int i = 0; i < numSamples; ++i)
            CHECK (keep[(size_t) i] == 0.0f);
    }

    // Out-of-range bounds clamp rather than reading past the prepared sources.
    {
        AcousticSendMatrix m;
        m.prepare (sr, sources, nodes);
        const auto in = makeInput (1);
        m.writeInputs (in, numSamples);

        std::vector<float> wide ((size_t) numSamples, 0.0f), all ((size_t) numSamples, 0.0f);
        m.computeNodeFeed (wide.data(), numSamples, 0, levels.data(), nullptr, nullptr,
                           stride, -5, 1000, true);
        AcousticSendMatrix ref;
        ref.prepare (sr, sources, nodes);
        ref.writeInputs (in, numSamples);
        ref.computeNodeFeed (all.data(), numSamples, 0, levels.data(), nullptr, nullptr, stride);
        CHECK (eqtests::bitEqualBlock (wide, all));
    }
}

static void testAcousticSendMatrixHistoryLength()
{
    using spatcore::dsp::AcousticSendMatrix;

    // A shorter history is a smaller allocation, not a different sound: any
    // delay the line can actually hold must land in exactly the same place it
    // would with the default one second.
    //
    // "Can actually hold" is stricter than it looks. writeInputs fills the
    // current block before the taps read it, so the usable span is the line
    // length MINUS one block; a delay closer to the length than that reads the
    // block just written. 50 ms of history at 48 kHz is 2400 samples, so with
    // 256-sample blocks anything up to about 2144 samples (44 ms) is honest.
    const double sr = 48000.0;
    const int numSamples = 256;
    const int stride = 4;
    const float delayMs = 20.0f;                                  // 960 samples, well inside
    const int expected = (int) ((delayMs / 1000.0f) * (float) sr);

    auto arrival = [&] (double historySeconds) -> int
    {
        AcousticSendMatrix m;
        if (historySeconds > 0.0)
            m.prepare (sr, 1, 1, historySeconds);
        else
            m.prepare (sr, 1, 1);                                 // default: one second

        std::vector<float> levels ((size_t) stride, 0.0f), delays ((size_t) stride, 0.0f);
        levels[0] = 1.0f;
        delays[0] = delayMs;

        int impulseAt = -1, peakAt = -1;
        float peak = 0.0f;

        // Eight settle blocks: the per-cell smoother has a 10 ms window, so a
        // measurement taken earlier measures the glide rather than the delay.
        for (int block = 0; block < 32; ++block)
        {
            juce::AudioBuffer<float> in (1, numSamples);
            in.clear();
            if (block == 8)
            {
                in.setSample (0, 0, 1.0f);
                impulseAt = block * numSamples;
            }

            std::vector<float> out ((size_t) numSamples, 0.0f);
            m.writeInputs (in, numSamples);
            m.computeNodeFeed (out.data(), numSamples, 0, levels.data(), delays.data(), nullptr, stride);
            m.advance (numSamples);

            for (int i = 0; i < numSamples; ++i)
                if (std::fabs (out[(size_t) i]) > peak)
                {
                    peak = std::fabs (out[(size_t) i]);
                    peakAt = block * numSamples + i;
                }
        }

        CHECK (peak > 0.4f);
        return (impulseAt >= 0 && peakAt >= 0) ? peakAt - impulseAt : -1;
    };

    CHECK (std::abs (arrival (0.0) - expected) <= 1);             // default history
    CHECK (std::abs (arrival (0.05) - expected) <= 1);            // 50 ms history, same answer
    CHECK (std::abs (arrival (0.25) - expected) <= 1);            // and in between

    // A nonsensical history falls back to the default rather than producing a
    // two-sample line that nothing can read through.
    {
        AcousticSendMatrix m;
        m.prepare (sr, 1, 1, -3.0);
        CHECK (m.isPrepared());

        std::vector<float> levels ((size_t) stride, 0.0f), delays ((size_t) stride, 0.0f);
        levels[0] = 1.0f;
        delays[0] = 100.0f;                                       // 4800 samples: needs the default
        juce::AudioBuffer<float> in (1, numSamples);
        in.clear();
        std::vector<float> out ((size_t) numSamples, 0.0f);
        m.writeInputs (in, numSamples);
        m.computeNodeFeed (out.data(), numSamples, 0, levels.data(), delays.data(), nullptr, stride);
        m.advance (numSamples);
        for (int i = 0; i < numSamples; ++i)
            CHECK (std::isfinite (out[(size_t) i]));
    }
}

//==============================================================================
// effects/LoopGuard
//==============================================================================

//==============================================================================
// effects/LoopGuard.h - the per-channel effect-to-effect runaway guard.
//
// What is worth testing here is not the arithmetic, it is the promises the
// engine makes to the operator: a trip happens at a wall-clock time rather than
// after a buffer-dependent number of blocks, the feed goes away smoothly and
// completely, a signal that never sustains never trips, a guard with nothing to
// do is bit-transparent, and nothing the guard does can hold a channel down for
// ever. The release criteria are tested on the signal the guard actually
// decides on (the pre-gain feed peak) with the return peak exercised in its two
// roles: a veto that delays a release, and a veto that runs out.
//
// Requires, in SpatcoreTests.cpp:
//   #include "spatcore/effects/LoopGuard.h"
// and juce_audio_basics for juce::ScopedNoDenormals (already reached through
// spatcore/dsp/AcousticSendMatrix.h). The transparency test needs the real
// FTZ/DAZ state, because "not written" and "multiplied by 1.0f" are the same
// thing without it and different with it - and every worker item in the engine
// arms it (EffectsEngineCore::renderChannel).

namespace loopguard_test {

using spatcore::effects::LoopGuard;

static constexpr double kSr = 48000.0;

/** +12 dBFS: comfortably over the +6 dBFS ceiling. */
static constexpr float kLoudPeak = 4.0f;

/** -40 dBFS: comfortably under the ceiling less the 12 dB hysteresis. */
static constexpr float kCalmPeak = 0.01f;

/** One batch exactly as the engine runs it: measure the raw effect-to-effect
    row, observe, then attenuate in place. The row is all ones, so what comes
    back IS the gain curve the guard applied, sample by sample. */
static void runBlock (LoopGuard& g, float feedPeak, float returnPeak, int blockSize,
                      std::vector<float>& appliedOut)
{
    std::vector<float> bus (static_cast<size_t> (blockSize), 1.0f);

    g.observeBlock (feedPeak, returnPeak, blockSize);
    g.applyGain (bus.data(), blockSize);

    appliedOut.insert (appliedOut.end(), bus.begin(), bus.end());
}

static void runBlocks (LoopGuard& g, float feedPeak, float returnPeak, int blockSize,
                       int numBlocks, std::vector<float>& appliedOut)
{
    for (int i = 0; i < numBlocks; ++i)
        runBlock (g, feedPeak, returnPeak, blockSize, appliedOut);
}

/** Index of the first sample that is not exactly `value`, or -1. */
static int firstNotExactly (const std::vector<float>& applied, float value, int from = 0)
{
    for (size_t i = static_cast<size_t> (from); i < applied.size(); ++i)
        if (applied[i] != value)
            return static_cast<int> (i);

    return -1;
}

/** Index of the first sample that is exactly `value`, or -1. */
static int firstExactly (const std::vector<float>& applied, float value, int from = 0)
{
    for (size_t i = static_cast<size_t> (from); i < applied.size(); ++i)
        if (applied[i] == value)
            return static_cast<int> (i);

    return -1;
}

/** Feed the guard identical batches until the predicate holds, and return how
    many it took. Bounded by maxBlocks so a broken state machine fails the test
    instead of hanging it; -1 means the budget ran out. */
template <typename Pred>
static int blocksUntil (LoopGuard& g, float feedPeak, float returnPeak, int blockSize,
                        int maxBlocks, Pred pred)
{
    std::vector<float> bus (static_cast<size_t> (blockSize), 1.0f);

    for (int i = 0; i < maxBlocks; ++i)
    {
        std::fill (bus.begin(), bus.end(), 1.0f);
        g.observeBlock (feedPeak, returnPeak, blockSize);
        g.applyGain (bus.data(), blockSize);

        if (pred (g))
            return i + 1;
    }

    return -1;
}

/** Held down means the ramp to silence has finished, not merely started: the
    guard is mid-ramp for the first few blocks of a trip and its gain is already
    below unity there, so a predicate on "attenuating" would return the block
    the trip happened and time nothing. */
static bool isHeldDown   (const LoopGuard& g) noexcept { return g.getGain() == 0.0f; }
static bool isComingBack (const LoopGuard& g) noexcept { return g.getGain() > 0.0f; }
static bool isRestored   (const LoopGuard& g) noexcept { return ! g.isTripped(); }

/** The trip has been DECLARED - the ramp down has only just started. */
static bool hasTripped   (const LoopGuard& g) noexcept { return g.isTripped(); }

/** Blocks in `seconds`, rounded down, for the timing bounds below. */
static int blocksIn (double seconds, int blockSize) noexcept
{
    return static_cast<int> (seconds * kSr) / blockSize;
}

} // namespace loopguard_test

//==============================================================================
/** The reason the thresholds are seconds and not the plan's block count: the
    same runaway must trip at the same moment whatever the audio buffer is set
    to. At 64, 256 and 512 samples the trip lands within one block of the trip
    time, so the spread across the three is one large block. The "20 consecutive
    blocks" rule would have spread the same event over 20 x (512 - 64) samples,
    a factor of eight in how much of the runaway reaches the speakers. */
static void testLoopGuardTripTimeIsBlockSizeIndependent()
{
    using namespace loopguard_test;

    const double tripSeconds = 0.060;
    const int nominal = static_cast<int> (tripSeconds * kSr);   // 2880 samples
    const int blockSizes[3] = { 64, 256, 512 };
    int tripSample[3] = { -1, -1, -1 };

    for (int b = 0; b < 3; ++b)
    {
        LoopGuard g;
        g.prepare (kSr, true, 6.0f, tripSeconds);

        std::vector<float> applied;
        const int blocks = static_cast<int> (0.5 * kSr) / blockSizes[b];   // half a second
        runBlocks (g, kLoudPeak, kCalmPeak, blockSizes[b], blocks, applied);

        tripSample[b] = firstNotExactly (applied, 1.0f);

        CHECK (tripSample[b] >= 0);
        CHECK (g.isTripped());
        CHECK (g.getTripCount() == 1u);          // one event, not one per block

        // The trip is declared at the first batch boundary at or past the trip
        // time, so the attenuation starts within one batch either side of it.
        // The upper bound is STRICTLY one batch, and that is exactly the right
        // width: the accumulated dt can land a hair under the trip time on the
        // block that would otherwise have fired, which at 64 samples (where the
        // trip time is a whole number of blocks) puts the first attenuated
        // sample exactly on the nominal one - while a guard that genuinely
        // declared a block late would put it a full block past, and fail. An
        // assertion written `<= nominal` would pass by zero margin here and go
        // red for a rounding change rather than for a behaviour change.
        CHECK (tripSample[b] < nominal + blockSizes[b]);
        CHECK (tripSample[b] >= nominal - blockSizes[b]);
    }

    int lo = tripSample[0];
    int hi = tripSample[0];

    for (int b = 1; b < 3; ++b)
    {
        lo = tripSample[b] < lo ? tripSample[b] : lo;
        hi = tripSample[b] > hi ? tripSample[b] : hi;
    }

    CHECK (hi - lo <= 512);                      // one large block, at worst
    CHECK (hi - lo < 20 * (512 - 64));           // what the block-count rule cost
}

//==============================================================================
/** The feed has to leave smoothly and arrive at silence: a cliff clicks, and a
    one-pole that stalls a hair above zero would leave the loop running at a
    gain the operator cannot see. */
static void testLoopGuardRampToZeroIsMonotonic()
{
    using namespace loopguard_test;

    const double rampDownSeconds = 0.005;

    LoopGuard g;
    g.prepare (kSr, true, 6.0f, 0.060, 0.500, rampDownSeconds, 0.050);

    std::vector<float> applied;
    runBlocks (g, kLoudPeak, kCalmPeak, 64, 200, applied);      // 267 ms

    const int trip = firstNotExactly (applied, 1.0f);
    CHECK (trip > 0);
    CHECK (g.isTripped());

    // A ramp, not a cliff: the first attenuated sample is neither 1 nor 0.
    CHECK (applied[static_cast<size_t> (trip)] < 1.0f);
    CHECK (applied[static_cast<size_t> (trip)] > 0.0f);

    bool monotonic = true;

    for (size_t i = static_cast<size_t> (trip) + 1; i < applied.size(); ++i)
        if (applied[i] > applied[i - 1])
            monotonic = false;

    CHECK (monotonic);

    // Exactly zero, by the smoother's snap, and it stays there.
    const int zeroAt = firstExactly (applied, 0.0f, trip);
    CHECK (zeroAt > trip);
    CHECK (firstNotExactly (applied, 0.0f, zeroAt) == -1);
    CHECK (bitEqualFloat (g.getGain(), 0.0f));

    // And it takes about the time it was asked for. The ramp is quoted as a
    // completion time, so the class divides by the number of time constants a
    // one-pole needs to land on its endpoint; getting that conversion wrong is
    // the mistake this bound catches.
    const int rampSamples = zeroAt - trip;
    const int expected = static_cast<int> (rampDownSeconds * kSr);   // 240 samples
    CHECK (rampSamples >= expected / 2);
    CHECK (rampSamples <= expected * 2);
}

//==============================================================================
/** Consecutive means consecutive. A feed that peaks over the ceiling but keeps
    falling back under it is doing what loud programme material does; only a
    build-up that sustains is a runaway. */
static void testLoopGuardDipBelowCeilingNeverTrips()
{
    using namespace loopguard_test;

    LoopGuard g;
    g.prepare (kSr, true, 6.0f, 0.060);

    std::vector<float> applied;
    const int loudBlocks = static_cast<int> (0.040 * kSr) / 64;   // 40 ms, then a dip

    for (int cycle = 0; cycle < 20; ++cycle)                      // ~0.9 s in all
    {
        runBlocks (g, kLoudPeak, kCalmPeak, 64, loudBlocks, applied);
        runBlock (g, 1.0f, kCalmPeak, 64, applied);               // 0 dBFS: under +6
    }

    CHECK (firstNotExactly (applied, 1.0f) == -1);
    CHECK (! g.isTripped());
    CHECK (g.getState() == LoopGuard::State::Armed);
    CHECK (g.getTripCount() == 0u);
    CHECK (bitEqualFloat (g.getGain(), 1.0f));
}

//==============================================================================
/** The operator removes the send: the feed goes calm, and after the release
    time the guard gives the bus back, smoothly and completely. Nothing may come
    back early, or a loop that is still live would be re-armed at full level. */
static void testLoopGuardReleasesAndRampsBack()
{
    using namespace loopguard_test;

    const double releaseSeconds = 0.500;
    const double rampUpSeconds = 0.050;

    LoopGuard g;
    g.prepare (kSr, true, 6.0f, 0.060, releaseSeconds, 0.005, rampUpSeconds);

    std::vector<float> down;
    runBlocks (g, kLoudPeak, kLoudPeak, 256, 60, down);           // 320 ms of runaway
    CHECK (g.getState() == LoopGuard::State::Tripped);
    CHECK (bitEqualFloat (g.getGain(), 0.0f));

    // 400 ms of calm is not 500 ms of calm.
    std::vector<float> held;
    runBlocks (g, kCalmPeak, kCalmPeak, 256, blocksIn (0.400, 256), held);
    CHECK (firstNotExactly (held, 0.0f) == -1);
    CHECK (g.isTripped());

    std::vector<float> back;
    runBlocks (g, kCalmPeak, kCalmPeak, 256, 60, back);           // 320 ms more

    const int start = firstNotExactly (back, 0.0f);
    CHECK (start >= 0);

    bool monotonic = true;

    for (size_t i = static_cast<size_t> (start) + 1; i < back.size(); ++i)
        if (back[i] < back[i - 1])
            monotonic = false;

    CHECK (monotonic);

    const int unityAt = firstExactly (back, 1.0f, start);
    CHECK (unityAt > start);
    CHECK (firstNotExactly (back, 1.0f, unityAt) == -1);

    const int rampSamples = unityAt - start;
    const int expected = static_cast<int> (rampUpSeconds * kSr);  // 2400 samples
    CHECK (rampSamples >= expected / 2);
    CHECK (rampSamples <= expected * 2);

    CHECK (! g.isTripped());
    CHECK (g.getState() == LoopGuard::State::Armed);
    CHECK (bitEqualFloat (g.getGain(), 1.0f));
    CHECK (g.getTripCount() == 1u);
}

//==============================================================================
/** A release time shorter than the ramp down is not a reason to hand the bus
    back at half gain. The calm clock only starts once the feed has actually
    reached exactly zero, so the guard always gets the loop off the bus before
    it starts thinking about putting it back - and the operator sees a complete
    ramp down, a hold and a ramp up whatever the two times are set to. */
static void testLoopGuardReleaseWaitsForSilence()
{
    using namespace loopguard_test;

    LoopGuard g;
    g.prepare (kSr, true, 6.0f, 0.060, /*release*/ 0.002, /*down*/ 0.050, /*up*/ 0.050);

    // Stop the loud feed the moment the trip is declared, so the ramp down
    // happens entirely inside the calm phase below.
    CHECK (blocksUntil (g, kLoudPeak, kCalmPeak, 64, 400, hasTripped) > 0);

    std::vector<float> applied;
    runBlocks (g, kCalmPeak, kCalmPeak, 64, 200, applied);        // 267 ms

    const int silent = firstExactly (applied, 0.0f);
    CHECK (silent >= 0);                                          // it got there at all

    bool roseBeforeSilence = false;

    for (size_t i = 1; i < static_cast<size_t> (silent); ++i)
        if (applied[i] > applied[i - 1])
            roseBeforeSilence = true;

    CHECK (! roseBeforeSilence);                                  // no turning back early

    // And the short release time is still honoured once silence is reached.
    const int backAt = firstExactly (applied, 1.0f, silent);
    CHECK (backAt > silent);
    CHECK (! g.isTripped());
    CHECK (g.getTripCount() == 1u);
}

//==============================================================================
/** The return peak's first job. It cannot cause a release - that decision is
    taken on the feed - but it can hold one off: restoring a loop feed into a
    channel whose own output is still over the ceiling would re-pump the loop
    the same batch. */
static void testLoopGuardHotReturnDelaysRelease()
{
    using namespace loopguard_test;

    const double releaseSeconds = 0.200;

    LoopGuard g;
    g.prepare (kSr, true, 6.0f, 0.060, releaseSeconds, 0.005, 0.050);

    std::vector<float> applied;
    runBlocks (g, kLoudPeak, kLoudPeak, 128, 60, applied);        // 160 ms
    CHECK (g.isTripped());
    CHECK (bitEqualFloat (g.getGain(), 0.0f));

    // The feed is calm but the chain is still ringing over the ceiling: three
    // release times pass - well inside the veto's budget, which the next test
    // measures - and the bus stays down.
    const int heldBlocks = blocksIn (3.0 * releaseSeconds, 128);
    const int cameBack = blocksUntil (g, kCalmPeak, kLoudPeak, 128, heldBlocks, isComingBack);
    CHECK (cameBack == -1);
    CHECK (g.isTripped());
    CHECK (bitEqualFloat (g.getGain(), 0.0f));

    // The ringing dies away and the release proceeds on the feed's terms.
    const int released = blocksUntil (g, kCalmPeak, kCalmPeak, 128, 400, isComingBack);
    CHECK (released > 0);
    CHECK (released <= blocksIn (releaseSeconds + 0.100, 128));   // release + a batch

    const int restored = blocksUntil (g, kCalmPeak, kCalmPeak, 128, 400, isRestored);
    CHECK (restored > 0);
    CHECK (bitEqualFloat (g.getGain(), 1.0f));
    CHECK (g.getTripCount() == 1u);
}

//==============================================================================
/** The return peak's second job, which is to stop being able to do the first
    one. The guard never touches input-to-effect feeds, so a channel can sit
    over the ceiling for ever with no loop at all - a hot mix, a compressor's
    makeup gain, a long tail. An unbounded veto would take that channel's
    effect-to-effect sends away until someone hit Clear, which is the same
    latch the release criteria were rewritten to avoid. After the budget the
    feed decides alone: if there really is a loop it re-pumps, re-trips, and the
    backoff ladder holds it down longer each time. */
static void testLoopGuardHotReturnVetoIsBounded()
{
    using namespace loopguard_test;

    const double releaseSeconds = 0.100;

    LoopGuard g;
    g.prepare (kSr, true, 6.0f, 0.060, releaseSeconds, 0.005, 0.050);

    CHECK (blocksUntil (g, kLoudPeak, kLoudPeak, 64, 400, isHeldDown) > 0);

    // Feed dead calm, return pinned over the ceiling for as long as it likes.
    const int cameBack = blocksUntil (g, kCalmPeak, kLoudPeak, 64, 4000, isComingBack);

    CHECK (cameBack > 0);                                         // not latched for ever

    // The veto really did delay it: without one the release would have fired
    // after a single release time.
    CHECK (cameBack >= blocksIn (2.0 * releaseSeconds, 64));

    // And the budget really is bounded: veto budget + one release time, plus a
    // batch or two of slack.
    const double budget = LoopGuard::kVetoReleaseTimes * releaseSeconds;
    CHECK (cameBack <= blocksIn (budget + 2.0 * releaseSeconds, 64));

    // Coming back over a hot return is a release, not a re-trip.
    CHECK (g.getTripCount() == 1u);
}

//==============================================================================
/** A loop that comes back while the feed is being restored has to be caught on
    the way up, not after the ramp has finished: the whole point of the ladder
    below is that a re-pumping loop never rides back to unity. */
static void testLoopGuardRetripsOnTheWayBack()
{
    using namespace loopguard_test;

    const double rampUpSeconds = 0.400;        // long, so there is a ramp to catch

    LoopGuard g;
    g.prepare (kSr, true, 6.0f, 0.060, 0.100, 0.005, rampUpSeconds);

    CHECK (blocksUntil (g, kLoudPeak, kCalmPeak, 64, 400, isHeldDown) > 0);
    CHECK (blocksUntil (g, kCalmPeak, kCalmPeak, 64, 2000, isComingBack) > 0);

    CHECK (g.getState() == LoopGuard::State::Releasing);
    CHECK (g.getGain() > 0.0f);
    CHECK (g.getGain() < 1.0f);

    // Loud again, mid-ramp. A guard that only watched while Armed would ride
    // the remaining 400 ms up to unity first, so the bound is what catches it.
    const int reTrip = blocksUntil (g, kLoudPeak, kCalmPeak, 64, blocksIn (0.120, 64), isHeldDown);

    CHECK (reTrip > 0);
    CHECK (g.getTripCount() == 2u);
    CHECK (g.getState() == LoopGuard::State::Tripped);
    CHECK (bitEqualFloat (g.getGain(), 0.0f));
}

//==============================================================================
/** A guard inside the loop it watches cannot tell "the danger has gone" from
    "the guard is working", so a release into a cycle the operator has not fixed
    lets the loop build again. The exposure is bounded by making each re-trip
    hold longer: the second hold is about twice the first. Without the ladder
    the channel would burst at the trip interval indefinitely - and without the
    forget, a channel that misbehaved once this morning would carry the long
    hold into tonight's show. */
static void testLoopGuardRepeatTripBacksOffTheRelease()
{
    using namespace loopguard_test;

    const double releaseSeconds = 0.100;

    LoopGuard g;
    g.prepare (kSr, true, 6.0f, 0.060, releaseSeconds, 0.005, 0.050);

    CHECK (blocksUntil (g, kLoudPeak, kCalmPeak, 64, 400, isHeldDown) > 0);
    const int firstHold = blocksUntil (g, kCalmPeak, kCalmPeak, 64, 2000, isComingBack);
    CHECK (firstHold > 0);
    CHECK (blocksUntil (g, kCalmPeak, kCalmPeak, 64, 2000, isRestored) > 0);

    // Straight back into the same loop, before the ladder has been forgotten.
    CHECK (blocksUntil (g, kLoudPeak, kCalmPeak, 64, 400, isHeldDown) > 0);
    const int secondHold = blocksUntil (g, kCalmPeak, kCalmPeak, 64, 4000, isComingBack);
    CHECK (secondHold > 0);

    // The base hold is 0.1 s, which is 75 batches of 64 samples: both numbers
    // are real counts rather than the one block a mid-ramp predicate returns.
    CHECK (firstHold >= 60);
    CHECK (secondHold >= (firstHold * 3) / 2);
    CHECK (secondHold <= firstHold * 3);
    CHECK (g.getTripCount() == 2u);

    // Now behave. Armed, restored and calm for longer than the forget factor
    // asks for, and the ladder goes back to the bottom.
    CHECK (blocksUntil (g, kCalmPeak, kCalmPeak, 64, 4000, isRestored) > 0);

    std::vector<float> idle;
    const double idleSeconds = (LoopGuard::kBackoffForgetFactor + 2.0) * releaseSeconds;
    runBlocks (g, kCalmPeak, kCalmPeak, 64, blocksIn (idleSeconds, 64), idle);
    CHECK (firstNotExactly (idle, 1.0f) == -1);                   // still fully open

    CHECK (blocksUntil (g, kLoudPeak, kCalmPeak, 64, 400, isHeldDown) > 0);
    const int thirdHold = blocksUntil (g, kCalmPeak, kCalmPeak, 64, 4000, isComingBack);
    CHECK (thirdHold > 0);
    CHECK (thirdHold >= 60);
    CHECK (thirdHold <= (firstHold * 3) / 2);                     // the base hold again
    CHECK (g.getTripCount() == 3u);
}

//==============================================================================
/** The common case, and the one that has to be free: 32 guards on a show where
    nothing is looping. An armed guard returns before it touches the row, so the
    feed is not even multiplied by 1.0f, and an unprepared one is transparent
    rather than a crash.

    FTZ/DAZ is armed here on purpose. In the engine every worker item runs
    inside juce::ScopedNoDenormals, and with flush-to-zero on, multiplying a
    denormal by 1.0f returns 0 - so "the row was multiplied by one" and "the row
    was not written" are two different signals, and only the second one is
    bit-transparent. Without arming it this test would pass against a guard that
    had lost its fast path entirely. */
static void testLoopGuardIdleIsFreeAndTransparent()
{
    using namespace loopguard_test;

    LoopGuard g;
    g.prepare (kSr);

    const auto input = eqtests::makeAwkwardSignal (512);
    auto bus = input;

    {
        const juce::ScopedNoDenormals noDenormals;

        for (int i = 0; i < 200; ++i)                             // 2.1 s of nothing
        {
            g.observeBlock (0.25f, 0.25f, 512);                   // -12 dBFS
            g.applyGain (bus.data(), 512);
        }
    }

    // Bit-identical, negative zero and denormals included: nothing was written,
    // so nothing was flushed on the way through.
    CHECK (eqtests::bitEqualBlock (bus, input));
    CHECK (! g.isTripped());
    CHECK (g.getState() == LoopGuard::State::Armed);
    CHECK (g.getTripCount() == 0u);
    CHECK (bitEqualFloat (g.getGain(), 1.0f));

    // Nothing to do means nothing is dereferenced either.
    g.applyGain (nullptr, 512);
    g.observeBlock (0.25f, 0.25f, 0);
    CHECK (bitEqualFloat (g.getGain(), 1.0f));

    // A guard that was never prepared never trips and never touches the row.
    LoopGuard fresh;
    auto untouched = input;
    fresh.observeBlock (1000.0f, 1000.0f, 512);
    fresh.applyGain (untouched.data(), 512);
    CHECK (eqtests::bitEqualBlock (untouched, input));
    CHECK (! fresh.isTripped());
    CHECK (bitEqualFloat (fresh.getGain(), 1.0f));

    // reset() is what emergency Clear does: armed and open again, with the
    // cumulative count left alone.
    LoopGuard tripped;
    tripped.prepare (kSr, true, 6.0f, 0.060);
    std::vector<float> applied;
    runBlocks (tripped, kLoudPeak, kCalmPeak, 64, 200, applied);
    CHECK (tripped.isTripped());
    tripped.reset();
    CHECK (! tripped.isTripped());
    CHECK (bitEqualFloat (tripped.getGain(), 1.0f));
    CHECK (tripped.getTripCount() == 1u);
}

//==============================================================================
/** effectsGlobalLoopGuard = 0. An operator who is deliberately building a
    feedback bunch has to be able to turn the guard off, and off has to mean
    off: not a guard that trips silently, not a row multiplied by a unity gain,
    nothing at all. The switch is live, because the alternative is rebuilding
    the engine to change it. */
static void testLoopGuardSwitchedOffIsInert()
{
    using namespace loopguard_test;

    LoopGuard g;
    g.prepare (kSr, false, 6.0f, 0.060);
    CHECK (! g.isEnabled());

    const auto input = eqtests::makeAwkwardSignal (512);
    auto bus = input;

    {
        const juce::ScopedNoDenormals noDenormals;

        for (int i = 0; i < 200; ++i)                             // 2.1 s of runaway
        {
            g.observeBlock (kLoudPeak, kLoudPeak, 512);
            g.applyGain (bus.data(), 512);
        }
    }

    CHECK (eqtests::bitEqualBlock (bus, input));
    CHECK (! g.isTripped());
    CHECK (g.getTripCount() == 0u);
    CHECK (bitEqualFloat (g.getGain(), 1.0f));

    // Switched on mid-show, it starts watching from there.
    g.setEnabled (true);
    CHECK (g.isEnabled());
    CHECK (blocksUntil (g, kLoudPeak, kLoudPeak, 64, 400, isHeldDown) > 0);
    CHECK (g.getTripCount() == 1u);

    // Switched off while tripped, it gives the feed straight back rather than
    // leaving a bus held down that nothing is watching any more.
    g.setEnabled (false);
    CHECK (! g.isTripped());
    CHECK (bitEqualFloat (g.getGain(), 1.0f));
    CHECK (g.getTripCount() == 1u);                               // cumulative, kept

    auto stillLoud = input;
    g.observeBlock (kLoudPeak, kLoudPeak, 512);
    g.applyGain (stillLoud.data(), 512);
    CHECK (eqtests::bitEqualBlock (stillLoud, input));
}

//==============================================================================
/** The two thresholds have to tile the number line. Loud is "not at or under
    the ceiling" and calm is "at or under the ceiling less the hysteresis", so
    with the hysteresis set to zero the two meet exactly and a feed parked on
    the ceiling is calm rather than neither. Written the other way round, that
    one value would freeze both clocks and the guard would stay down for ever on
    a signal it does not consider loud. */
static void testLoopGuardZeroHysteresisHasNoDeadBand()
{
    using namespace loopguard_test;

    const float ceiling = spatcore::dsp::FastDecibels::dbToGain (6.0f);

    LoopGuard g;
    g.prepare (kSr, true, 6.0f, 0.060, 0.100, 0.005, 0.050, /*hysteresis*/ 0.0);

    // Parked exactly on the ceiling is not loud: it never trips.
    std::vector<float> applied;
    runBlocks (g, ceiling, 0.0f, 64, 200, applied);
    CHECK (firstNotExactly (applied, 1.0f) == -1);
    CHECK (! g.isTripped());

    // Over it, it trips; back on it, it is calm and releases.
    CHECK (blocksUntil (g, kLoudPeak, 0.0f, 64, 400, isHeldDown) > 0);
    const int cameBack = blocksUntil (g, ceiling, 0.0f, 64, 2000, isComingBack);
    CHECK (cameBack > 0);
    CHECK (cameBack <= blocksIn (0.300, 64));
    CHECK (g.getTripCount() == 1u);
}

//==============================================================================
/** A channel that has gone non-finite is the last one whose loop feed should be
    restored. A running max would drop the NaN (every comparison against one is
    false) and the guard would never see the block that mattered most, so the
    peak and both thresholds are written as negated comparisons. */
static void testLoopGuardNonFinitePeakTripsAndHolds()
{
    using namespace loopguard_test;

    const float nan = std::numeric_limits<float>::quiet_NaN();

    std::vector<float> block (64, 0.1f);
    block[7] = nan;
    CHECK (std::isnan (LoopGuard::peakOf (block.data(), 64)));
    CHECK (bitEqualFloat (LoopGuard::peakOf (nullptr, 64), 0.0f));

    std::vector<float> finite (64, -0.5f);
    finite[3] = 0.75f;
    CHECK (bitEqualFloat (LoopGuard::peakOf (finite.data(), 64), 0.75f));

    LoopGuard g;
    g.prepare (kSr, true, 6.0f, 0.060);

    std::vector<float> applied;
    runBlocks (g, nan, 0.0f, 64, 60, applied);                    // 80 ms of NaN
    CHECK (g.isTripped());

    // A calm return cannot talk it round while the feed is still NaN, and
    // neither can the veto budget: that budget only ever runs while the FEED is
    // calm, and a NaN feed never is.
    CHECK (blocksUntil (g, nan, 0.0f, 64, 2000, isComingBack) == -1);
    CHECK (bitEqualFloat (g.getGain(), 0.0f));
}

//==============================================================================
// effects/EffectsEngine
//==============================================================================

//==============================================================================
// effects/EffectsEngineCore.h + effects/EffectsEngine.h
//
// The engine is asserted through the ledger above everything else: a block
// written at callback n comes back at n+1, and a block routed through a second
// effects channel comes back at n+2. That one property is what the whole
// feedback design rests on, so every other test here exists to stop something
// from quietly breaking it - the worker count, a late driver, a lapped ring, a
// teardown under a live callback, the emergency Clear and the loop guard.
//
// Needs, in the includer:
//     #include "spatcore/effects/EffectsEngineCore.h"
//     #include "spatcore/effects/EffectsEngine.h"
//==============================================================================

namespace engine_test
{
    using namespace spatcore::effects;
    namespace srt = spatcore::rt;

    inline float peakOf (const std::vector<float>& v) noexcept
    {
        float p = 0.0f;
        for (float x : v)
            p = std::max (p, std::fabs (x));
        return p;
    }

    inline bool isSilent (const std::vector<float>& v) noexcept
    {
        for (float x : v)
            if (x != 0.0f)
                return false;
        return true;
    }

    inline std::vector<float> dcBlock (int n, float value)
    {
        return std::vector<float> ((size_t) n, value);
    }

    //==========================================================================
    /** A module that plants ONE non-finite sample in the middle of the block
        and leaves the last sample finite.

        That is the exact shape neither existing guard can see: ModuleSlot and
        EffectChain both test isfinite(buf[n-1]), which is the right trade for
        the recursive structures where this really happens, and blind to a
        memoryless module passing a single bad sample through. It is here to
        prove the engine scans the whole return block. */
    class MidBlockNanModule : public IEffectModule
    {
    public:
        ModuleId type() const noexcept override { return ModuleId::Trem; }
        void prepare (const ChainConfig&) override {}
        void reset() noexcept override {}

        ParamApplyInfo applyParams (const EffectChannelParams&, int) noexcept override
        {
            return { false, false };            // always active
        }

        void process (float* inout, int n) noexcept override
        {
            if (n > 2)
                inout[n / 2] = std::numeric_limits<float>::quiet_NaN();
        }

        int getLatencySamples() const noexcept override { return 0; }
    };

    inline std::unique_ptr<IEffectModule> nanFactory (ModuleId type, int, const ChainConfig&)
    {
        if (type == ModuleId::Trem)
            return std::make_unique<MidBlockNanModule>();

        return nullptr;                          // every other slot passes through
    }

    //==========================================================================
    /** A consumer, minus the audio device.

        One ring per render source with the effect returns last and contiguous,
        a feed matrix triplet, and a callback() that does the two jobs the app's
        device callback does, in the app's order: pop every return into the
        render-source row that belongs to it, write every render source into its
        ring, then let the driver run. Driving the core synchronously is what
        makes the ledger assertable exactly rather than eventually. */
    struct Rig
    {
        Rig (int numInputs, int numEffects, int blockSize, int ringBlocks = 8)
            : nIn (numInputs), nFx (numEffects), block (blockSize)
        {
            nSrc = nIn + nFx;
            firstFx = nIn;
            stride = nFx;

            for (int i = 0; i < nSrc; ++i)
            {
                auto r = std::make_unique<srt::SharedInputRingBuffer>();
                r->setSize (block * ringBlocks);
                rings.push_back (std::move (r));
            }

            levels.assign ((size_t) (nSrc * stride), 0.0f);
            delays.assign ((size_t) (nSrc * stride), 0.0f);
            hf.assign ((size_t) (nSrc * stride), 0.0f);

            silence.assign ((size_t) block, 0.0f);
            popped.assign ((size_t) nFx, std::vector<float> ((size_t) block, 0.0f));

            config.sampleRate = 48000.0;
            config.blockSize = block;
            config.numSources = nSrc;
            config.numEffects = nFx;
            config.matrixStride = stride;
            config.firstEffectSourceRow = firstFx;
            config.workerThreads = 0;
            config.returnCushionBlocks = 1;
            config.maxFeedDelaySeconds = 0.05;       // the tests ask for a block or two
            config.maxEffectDelaySeconds = 0.25;
            config.moduleFactory = nullptr;          // eleven pass-through slots
        }

        void level (int src, int fx, float linear) noexcept
        {
            levels[(size_t) (src * stride + fx)] = linear;
        }

        void delayMs (int src, int fx, float ms) noexcept
        {
            delays[(size_t) (src * stride + fx)] = ms;
        }

        bool prepare()
        {
            const bool ok = core.prepare (config, rings);
            core.setFeedMatrices (delays.data(), levels.data(), hf.data(), stride, nSrc, nFx);
            return ok;
        }

        /** One device callback. @returns batches the driver ran. */
        int callback (const std::vector<const float*>& inputs)
        {
            for (int fx = 0; fx < nFx; ++fx)
                core.pullReturn (fx, popped[(size_t) fx].data(), block);

            for (int i = 0; i < nIn; ++i)
            {
                const float* src = (i < (int) inputs.size() && inputs[(size_t) i] != nullptr)
                                       ? inputs[(size_t) i] : silence.data();
                rings[(size_t) i]->write (src, block);
            }

            for (int fx = 0; fx < nFx; ++fx)
                rings[(size_t) (firstFx + fx)]->write (popped[(size_t) fx].data(), block);

            return core.drainAvailable();
        }

        int callbackSilent()
        {
            return callback (std::vector<const float*> ((size_t) nIn, nullptr));
        }

        int callbackOn (int inputIndex, const float* data)
        {
            std::vector<const float*> in ((size_t) nIn, nullptr);
            in[(size_t) inputIndex] = data;
            return callback (in);
        }

        /** Fill every ring without letting the driver run, which is how a
            backlog and a lap are staged. */
        void writeAllRings (const float* data)
        {
            for (int i = 0; i < nSrc; ++i)
                rings[(size_t) i]->write (data != nullptr ? data : silence.data(), block);
        }

        int nIn, nFx, block, nSrc, firstFx, stride;
        std::vector<std::unique_ptr<srt::SharedInputRingBuffer>> rings;
        std::vector<float> levels, delays, hf, silence;
        std::vector<std::vector<float>> popped;
        EffectsEngineCore::Config config;
        EffectsEngineCore core;
    };
}

//==============================================================================
// THE LEDGER. One hop is one block, two hops are two, and the return cushion
// is the only thing that moves either number.
//==============================================================================

static void testEffectsEngineBlockLedger()
{
    using namespace engine_test;

    Rig rig (2, 2, 64);
    rig.level (0, 0, 1.0f);                  // input 0 -> effect 0
    rig.level (rig.firstFx + 0, 1, 1.0f);    // effect 0's RETURN -> effect 1
    CHECK (rig.prepare());
    CHECK (rig.core.getReturnCushionBlocks() == 1);

    const auto signal = module_test::awkwardBlock (64, 5);

    // Callback 0 writes the block. Nothing can have come back yet: the pull
    // happens at the TOP of the callback, before the write.
    CHECK (rig.callbackOn (0, signal.data()) == 1);
    CHECK (isSilent (rig.popped[0]));
    CHECK (isSilent (rig.popped[1]));

    // Callback 1 pops it. A send at unity with no delay is the identity, so
    // this is a bit-exact assertion and not an approximate one.
    CHECK (rig.callbackSilent() == 1);
    for (int i = 0; i < 64; ++i)
        CHECK (bitEqualFloat (rig.popped[0][(size_t) i], signal[(size_t) i]));
    CHECK (isSilent (rig.popped[1]));

    // Callback 2 pops the same audio out of the SECOND channel: effect 0's
    // return re-entered the matrix as a render source on the next batch.
    CHECK (rig.callbackSilent() == 1);
    CHECK (isSilent (rig.popped[0]));
    for (int i = 0; i < 64; ++i)
        CHECK (bitEqualFloat (rig.popped[1][(size_t) i], signal[(size_t) i]));

    // At a cushion of one the ring holds exactly one block at every pull: one
    // pops per callback and one arrives per batch. Neither counter may move.
    CHECK (rig.core.getUnderruns (0) == 0);
    CHECK (rig.core.getUnderruns (1) == 0);
    CHECK (rig.core.getReturnDiscards (0) == 0);
    CHECK (rig.core.getReturnDiscards (1) == 0);

    // And the cushion IS the latency. The design document's ledger and its
    // discard rule disagreed by one block; the ledger is the user-visible
    // claim, so the ring is primed with exactly `cushion` blocks and a pull
    // discards anything above that. Cushion 2 therefore moves every arrival by
    // exactly one block and buys one block of slack against a late driver.
    Rig cushioned (2, 2, 64);
    cushioned.config.returnCushionBlocks = 2;
    cushioned.level (0, 0, 1.0f);
    CHECK (cushioned.prepare());
    CHECK (cushioned.core.getReturnCushionBlocks() == 2);

    CHECK (cushioned.callbackOn (0, signal.data()) == 1);
    CHECK (isSilent (cushioned.popped[0]));
    CHECK (cushioned.callbackSilent() == 1);
    CHECK (isSilent (cushioned.popped[0]));           // one block later than above
    CHECK (cushioned.callbackSilent() == 1);
    for (int i = 0; i < 64; ++i)
        CHECK (bitEqualFloat (cushioned.popped[0][(size_t) i], signal[(size_t) i]));

    // The cushion holds, rather than creeping: after a hundred callbacks the
    // arrival is still two blocks, which is what the discard rule is for.
    //
    // BOTH halves of the priming, not just one. The underrun count catches a
    // ring primed with too FEW blocks; without the discard count an extra
    // priming block would be silently corrected by the first pull and cost a
    // startup glitch that no assertion here could see.
    CHECK (cushioned.core.getUnderruns (0) == 0);
    CHECK (cushioned.core.getReturnDiscards (0) == 0);
}

//==============================================================================
// THE PER-SLOT METER. Every module keeps one in a relaxed atomic it writes per
// block (gain reduction for Dynamics, output peak for the others), and the GUI
// reads it straight off the module through the core rather than from a
// telemetry copy. This proves the read reaches the RIGHT slot of the RIGHT
// chain, that a bypassed slot reads 0, and that an index off either edge
// reads 0 instead of touching memory.
//==============================================================================

static void testEffectsEngineSlotMeter()
{
    using namespace engine_test;

    constexpr int kDyn1Slot = 3;             // kSlots: dist, eq1, eq2, dyn1, dyn2, ...
    constexpr int kDyn2Slot = 4;

    Rig rig (1, 2, 64);
    rig.config.moduleFactory = &createModule;          // real modules
    rig.level (0, 0, 1.0f);                             // input 0 -> effect 0 only
    CHECK (rig.prepare());

    // Effect 0: dyn1 compressing hard, dyn2 bypassed. Effect 1: untouched.
    EffectChannelParams p;
    p.dyn[0].bypass = 0;
    p.dyn[0].compOn = 1;
    p.dyn[0].compThresholdDb = -40.0f;
    p.dyn[0].compRatio = 20.0f;
    p.dyn[0].compAttackMs = 0.1f;
    p.dyn[1].bypass = 1;
    p.revision = 1;
    rig.core.publishChannelParams (0, p);

    // Nothing has run: every meter reads its idle value.
    CHECK (rig.core.getSlotMeterDb (0, kDyn1Slot) == 0.0f);

    // A hot DC block, well above the threshold, for long enough that the
    // 0.1 ms attack has settled many times over.
    const auto hot = dcBlock (64, 0.9f);
    for (int b = 0; b < 40; ++b)
        rig.callbackOn (0, hot.data());

    const float gr = rig.core.getSlotMeterDb (0, kDyn1Slot);
    CHECK (gr < -1.0f);                                 // real gain reduction, in dB
    CHECK (gr > -60.0f);                                // and a sane amount of it

    // The bypassed instance of the SAME module on the SAME chain reads idle:
    // the read is per slot, not per module type.
    CHECK (rig.core.getSlotMeterDb (0, kDyn2Slot) == 0.0f);

    // The other chain got no signal and no compressor: idle.
    CHECK (rig.core.getSlotMeterDb (1, kDyn1Slot) == 0.0f);

    // Off either edge is 0, never a read past the arrays.
    CHECK (rig.core.getSlotMeterDb (-1, kDyn1Slot) == 0.0f);
    CHECK (rig.core.getSlotMeterDb (2, kDyn1Slot) == 0.0f);
    CHECK (rig.core.getSlotMeterDb (0, -1) == 0.0f);
    CHECK (rig.core.getSlotMeterDb (0, kNumModuleSlots) == 0.0f);
}

//==============================================================================
// DETERMINISM. An offline render and a live render have to be the same render,
// so the worker count must not be able to reach the arithmetic.
//==============================================================================

static void testEffectsEngineWorkerDeterminism()
{
    using namespace engine_test;

    const int block = 64;
    const int numBlocks = 40;

    EffectChannelParams p;
    p.trem.bypass = 0;
    p.trem.rateHz = 3.0f;
    p.trem.depthDb = 9.0f;
    p.crush.bypass = 0;
    p.crush.bits = 6.0f;
    p.crush.rateHz = 9000.0f;
    p.delay.bypass = 0;
    p.delay.timeMs = 20.0f;
    p.delay.feedback = 40.0f;
    p.delay.mix = 50.0f;
    p.revision = 1;

    std::vector<std::vector<float>> sequential, parallel;

    for (int pass = 0; pass < 2; ++pass)
    {
        Rig rig (4, 4, block);
        rig.config.workerThreads = (pass == 0) ? 0 : 3;
        rig.config.moduleFactory = &createModule;          // real modules, real state

        for (int src = 0; src < 4; ++src)
            for (int fx = 0; fx < 4; ++fx)
                rig.level (src, fx, 0.2f + 0.1f * (float) ((src + fx) % 3));

        rig.level (rig.firstFx + 0, 1, 0.4f);              // and one effect -> effect route
        rig.delayMs (1, 2, 0.5f);                          // a fractional feed delay

        CHECK (rig.prepare());

        for (int fx = 0; fx < 4; ++fx)
            rig.core.publishChannelParams (fx, p);

        std::vector<std::vector<float>> captured;

        for (int b = 0; b < numBlocks; ++b)
        {
            std::vector<std::vector<float>> inputs;
            std::vector<const float*> ptrs;
            for (int i = 0; i < 4; ++i)
            {
                inputs.push_back (module_test::awkwardBlock (block, 100 + b * 7 + i));
                ptrs.push_back (inputs.back().data());
            }

            rig.callback (ptrs);

            for (int fx = 0; fx < 4; ++fx)
                captured.push_back (rig.popped[(size_t) fx]);
        }

        if (pass == 0)
        {
            sequential = std::move (captured);
            CHECK (rig.core.getNumWorkers() == 0);
        }
        else
        {
            parallel = std::move (captured);

            // Without this the test could pass by running sequentially twice.
            CHECK (rig.core.getNumWorkers() == 3);
        }
    }

    CHECK (sequential.size() == parallel.size());

    bool identical = true;
    for (size_t b = 0; b < sequential.size() && b < parallel.size(); ++b)
        for (size_t i = 0; i < sequential[b].size(); ++i)
            if (! bitEqualFloat (sequential[b][i], parallel[b][i]))
                identical = false;

    CHECK (identical);

    // A test that compared two silent renders would pass for the wrong reason.
    float loudest = 0.0f;
    for (const auto& b : sequential)
        loudest = std::max (loudest, peakOf (b));
    CHECK (loudest > 0.01f);
}

//==============================================================================
// LATE DRIVER. Latency must never creep, on either side.
//==============================================================================

static void testEffectsEngineBacklogSkip()
{
    using namespace engine_test;

    Rig rig (1, 1, 64, 8);                   // eight blocks of ring, so no lap
    rig.config.maxSourceBacklogBlocks = 2;
    rig.level (0, 0, 1.0f);
    CHECK (rig.prepare());

    // Three whole blocks pending, each a different DC value, and nothing has
    // drained them.
    for (int b = 1; b <= 3; ++b)
    {
        const auto dc = dcBlock (64, 0.1f * (float) b);
        rig.writeAllRings (dc.data());
    }

    CHECK (rig.core.drainAvailable() == 1);              // one batch, not three
    CHECK (rig.core.getSourceSkips() == 1);
    CHECK (rig.core.getRingWraps() == 0);

    // And it kept the NEWEST block, which is the whole point: an engine that
    // worked through the backlog would be permanently two blocks late.
    std::vector<float> out ((size_t) 64, -1.0f);
    CHECK (rig.core.pullReturn (0, out.data(), 64));
    for (int i = 0; i < 64; ++i)
        CHECK (bitEqualFloat (out[(size_t) i], 0.3f));
}

static void testEffectsEngineRingWrapResync()
{
    using namespace engine_test;

    // Four blocks of ring: the depth every shared ring in the app has today.
    Rig rig (1, 1, 64, 4);
    rig.config.maxSourceBacklogBlocks = 8;               // out of the way; this is about the lap
    rig.level (0, 0, 1.0f);
    CHECK (rig.prepare());

    // Five blocks written and none read laps the consumer. The cursor's own
    // arithmetic cannot say so - it reports a plausible ONE block available,
    // which is why this needs the ring's additive counter to be visible at all.
    for (int b = 1; b <= 5; ++b)
    {
        const auto dc = dcBlock (64, 0.1f * (float) b);
        rig.writeAllRings (dc.data());
    }

    CHECK (rig.rings[0]->getAvailableAt (0) == 64);      // the aliasing, pinned
    CHECK (rig.rings[0]->getTotalWritten() == 5u * 64u);

    CHECK (rig.core.drainAvailable() == 1);
    CHECK (rig.core.getRingWraps() == 1);
    CHECK (rig.core.getSourceSkips() == 0);

    // Resynced onto the newest block rather than onto whatever the stale cursor
    // happened to point at.
    std::vector<float> out ((size_t) 64, -1.0f);
    CHECK (rig.core.pullReturn (0, out.data(), 64));
    for (int i = 0; i < 64; ++i)
        CHECK (bitEqualFloat (out[(size_t) i], 0.5f));

    // Back in step: the next block is taken normally, with no second wrap.
    const auto dc = dcBlock (64, 0.9f);
    rig.writeAllRings (dc.data());
    CHECK (rig.core.drainAvailable() == 1);
    CHECK (rig.core.getRingWraps() == 1);
}

//==============================================================================
// THE CONFIGURATION FLOORS. Two Config fields are consumer hints that know
// nothing about the block size, and both of them are unsafe below it rather
// than merely useless.
//==============================================================================

static void testEffectsEngineShortFeedHistory()
{
    using namespace engine_test;

    // Five milliseconds of history is 240 samples at 48 kHz: an ordinary
    // number to write down, and less than half a 512-sample block.
    // AcousticSendMatrix::writeInputs wraps by copying (numSamples - length)
    // floats to the START of a length-sized row, so a line shorter than the
    // block writes off the end of it - past the allocation, for the last
    // source. The engine floors the history at two blocks for that reason.
    Rig rig (2, 1, 512, 8);
    rig.config.maxFeedDelaySeconds = 0.005;
    rig.level (0, 0, 1.0f);
    CHECK (rig.prepare());

    const auto signal = module_test::awkwardBlock (512, 29);

    CHECK (rig.callbackOn (0, signal.data()) == 1);
    CHECK (isSilent (rig.popped[0]));

    // The ledger, unchanged and still bit-exact: the floor lengthens the line,
    // it does not change what a zero-delay unity send does.
    CHECK (rig.callbackSilent() == 1);
    for (int i = 0; i < 512; ++i)
        CHECK (bitEqualFloat (rig.popped[0][(size_t) i], signal[(size_t) i]));

    // Zero is the same story told by a consumer that did not fill the field in.
    Rig zero (1, 1, 256, 8);
    zero.config.maxFeedDelaySeconds = 0.0;
    zero.level (0, 0, 1.0f);
    CHECK (zero.prepare());

    const auto other = module_test::awkwardBlock (256, 31);
    CHECK (zero.callbackOn (0, other.data()) == 1);
    CHECK (zero.callbackSilent() == 1);
    for (int i = 0; i < 256; ++i)
        CHECK (bitEqualFloat (zero.popped[0][(size_t) i], other[(size_t) i]));
}

static void testEffectsEngineRefusesUndersizedRing()
{
    using namespace engine_test;

    // A ring of exactly one block can never deliver one - SharedInputRingBuffer
    // reserves a slot - so the availability gate never opens and the engine
    // would sit there for ever with batches, wraps and skips all reading zero.
    // Below TWO blocks the lap threshold (capacity - block) drops under the one
    // block legitimately in flight between a callback and its batch, so every
    // batch would declare a lap and reset every chain. Both are refused at
    // prepare(), where a consumer can see them.
    Rig tooSmall (1, 1, 64, 1);
    tooSmall.level (0, 0, 1.0f);
    CHECK (! tooSmall.prepare());
    CHECK (! tooSmall.core.isReady());

    std::vector<float> out ((size_t) 64, 7.0f);
    CHECK (! tooSmall.core.pullReturn (0, out.data(), 64));
    CHECK (isSilent (out));

    // And exactly two blocks is accepted and works, so the floor is a floor
    // rather than a margin invented for comfort.
    Rig atFloor (1, 1, 64, 2);
    atFloor.level (0, 0, 1.0f);
    CHECK (atFloor.prepare());

    const auto signal = module_test::awkwardBlock (64, 47);
    CHECK (atFloor.callbackOn (0, signal.data()) == 1);
    CHECK (atFloor.callbackSilent() == 1);
    for (int i = 0; i < 64; ++i)
        CHECK (bitEqualFloat (atFloor.popped[0][(size_t) i], signal[(size_t) i]));

    CHECK (atFloor.core.getRingWraps() == 0);
    CHECK (atFloor.core.getUnderruns (0) == 0);
}

//==============================================================================
// THE PUBLISHED MATRIX IS NOT THE CHANNEL COUNT. It bounds what may be READ
// from the matrix, and nothing else: every live channel runs every batch.
//==============================================================================

static void testEffectsEngineUnroutedChannelsStillRun()
{
    using namespace engine_test;

    // Four channels live, a matrix that claims two. The two it does not reach
    // still have to run: their chains hold the tails, and their return rings
    // are what the audio callback pops on every single callback. A sweep bounded
    // by the published count would freeze them mid-tail and underrun the
    // callback for ever, with no telemetry saying why.
    Rig rig (2, 4, 64);
    for (int fx = 0; fx < 4; ++fx)
        rig.level (0, fx, 1.0f);

    CHECK (rig.core.prepare (rig.config, rig.rings));
    rig.core.setFeedMatrices (rig.delays.data(), rig.levels.data(), rig.hf.data(),
                              rig.stride, rig.nSrc, 2);              // two, not four

    const auto signal = module_test::awkwardBlock (64, 41);

    CHECK (rig.callbackOn (0, signal.data()) == 1);
    CHECK (rig.callbackSilent() == 1);

    // The routed channels render, and render exactly.
    for (int i = 0; i < 64; ++i)
    {
        CHECK (bitEqualFloat (rig.popped[0][(size_t) i], signal[(size_t) i]));
        CHECK (bitEqualFloat (rig.popped[1][(size_t) i], signal[(size_t) i]));
    }

    // The unrouted ones render silence - not stale audio, and not a cell of
    // somebody else's matrix row.
    CHECK (isSilent (rig.popped[2]));
    CHECK (isSilent (rig.popped[3]));

    for (int b = 0; b < 20; ++b)
        rig.callbackSilent();

    // The assertion this test exists for: every channel was fed a block every
    // callback, whether the matrix reached it or not.
    for (int fx = 0; fx < 4; ++fx)
        CHECK (rig.core.getUnderruns (fx) == 0);

    // And nothing latched: publishing the full count brings them back on the
    // next batch, on the same one-block ledger as any other channel.
    rig.core.setFeedMatrices (rig.delays.data(), rig.levels.data(), rig.hf.data(),
                              rig.stride, rig.nSrc, rig.nFx);

    CHECK (rig.callbackOn (0, signal.data()) == 1);
    CHECK (rig.callbackSilent() == 1);

    for (int i = 0; i < 64; ++i)
        CHECK (bitEqualFloat (rig.popped[3][(size_t) i], signal[(size_t) i]));

    for (int fx = 0; fx < 4; ++fx)
        CHECK (rig.core.getUnderruns (fx) == 0);
}

static void testEffectsEngineNarrowStrideDoesNotAlias()
{
    using namespace engine_test;

    // computeNodeFeed indexes levels[src * stride + fx] into an array the
    // consumer sized numSources * stride, so a channel index at or above the
    // stride reads the NEXT source's row - and, on the last source, past the
    // end of the allocation. The aliasing is the observable half: at stride 2,
    // channel 2's cell for source 0 IS source 1's cell for channel 0.
    Rig rig (2, 4, 64);
    rig.stride = 2;
    rig.config.matrixStride = 2;

    rig.level (0, 0, 1.0f);                  // source 0 -> channel 0, the real route
    rig.level (1, 0, 1.0f);                  // source 1 -> channel 0, and the alias

    CHECK (rig.prepare());

    const auto signal = module_test::awkwardBlock (64, 43);

    CHECK (rig.callbackOn (0, signal.data()) == 1);
    CHECK (rig.callbackSilent() == 1);

    // Channel 0 is inside the stride and renders normally (source 1 is silent,
    // so this is still the identity).
    for (int i = 0; i < 64; ++i)
        CHECK (bitEqualFloat (rig.popped[0][(size_t) i], signal[(size_t) i]));

    // Channels 2 and 3 are outside it. A channel that read the aliased cell
    // would be carrying the input at unity right here.
    CHECK (isSilent (rig.popped[2]));
    CHECK (isSilent (rig.popped[3]));

    for (int b = 0; b < 8; ++b)
        rig.callbackSilent();

    for (int fx = 0; fx < 4; ++fx)
        CHECK (rig.core.getUnderruns (fx) == 0);
}

//==============================================================================
// MUTE AND THE LOOP-GUARD SWITCH. Both are live operator controls that the
// template's own shape would get wrong.
//==============================================================================

static void testEffectsEngineMuteKeepsChainsRunning()
{
    using namespace engine_test;

    // The reverb feed's muted branch skips its whole sweep, which is right
    // there and wrong here: the tails live inside the chains being skipped, so
    // skipping would freeze every delay, starve every return ring and resume a
    // frozen tail on unmute. Muted silences the FEED and nothing else.
    EffectChannelParams p;
    p.delay.bypass = 0;
    p.delay.timeMs = 10.0f;
    p.delay.feedback = 70.0f;
    p.delay.mix = 100.0f;
    p.revision = 1;

    Rig rig (1, 1, 64);
    rig.config.moduleFactory = &createModule;
    rig.level (0, 0, 1.0f);
    CHECK (rig.prepare());
    rig.core.publishChannelParams (0, p);

    const auto hot = module_test::awkwardBlock (64, 53);
    rig.callbackOn (0, hot.data());

    for (int b = 0; b < 12; ++b)
        rig.callbackSilent();

    CHECK (! isSilent (rig.popped[0]));               // a tail is running

    rig.core.setMuted (true);
    CHECK (rig.core.getMuted());

    rig.callbackSilent();
    const float early = peakOf (rig.popped[0]);

    for (int b = 0; b < 30; ++b)
        rig.callbackSilent();

    const float late = peakOf (rig.popped[0]);

    CHECK (early > 0.0f);                             // the chain kept producing
    CHECK (late < early);                             // and the tail moved on
    CHECK (rig.core.getUnderruns (0) == 0);           // the ring never starved

    rig.core.setMuted (false);
    CHECK (! rig.core.getMuted());

    rig.callbackOn (0, hot.data());
    rig.callbackSilent();
    CHECK (! isSilent (rig.popped[0]));
    CHECK (rig.core.getUnderruns (0) == 0);
}

static void testEffectsEngineLoopGuardSwitch()
{
    using namespace engine_test;

    Rig rig (1, 1, 64);
    rig.config.loopGuardCeilingDb = 6.0f;
    rig.config.loopGuardTripSeconds = 0.004;          // three blocks at 64/48k
    rig.config.loopGuardRampDownSeconds = 0.0005;
    rig.config.loopGuardReleaseSeconds = 10.0;        // far longer than this test
    rig.level (0, 0, 1.0f);
    rig.level (rig.firstFx + 0, 0, 1.5f);             // the runaway
    CHECK (rig.prepare());
    CHECK (rig.core.isLoopGuardEnabled());

    const auto in = module_test::awkwardBlock (64, 59);

    for (int b = 0; b < 60 && ! rig.core.isLoopGuardTripped (0); ++b)
        rig.callbackOn (0, in.data());

    CHECK (rig.core.isLoopGuardTripped (0));
    const std::uint32_t trips = rig.core.getLoopGuardTrips (0);
    CHECK (trips == 1);

    // Off, at the next batch boundary rather than by a rebuild. Turning it off
    // RE-ARMS every guard: someone switching it off mid-trip is asking for
    // their loop back now, not one release time from now.
    rig.core.setLoopGuardEnabled (false);
    CHECK (! rig.core.isLoopGuardEnabled());

    rig.callbackOn (0, in.data());
    rig.callbackOn (0, in.data());
    CHECK (! rig.core.isLoopGuardTripped (0));

    // The trip counter is a history, not a state, so switching off does not
    // rewrite it - and nothing counts while the guard is not watching.
    CHECK (rig.core.getLoopGuardTrips (0) == trips);

    // And back on: the same runaway trips it again.
    rig.core.setLoopGuardEnabled (true);
    CHECK (rig.core.isLoopGuardEnabled());

    for (int b = 0; b < 60 && ! rig.core.isLoopGuardTripped (0); ++b)
        rig.callbackOn (0, in.data());

    CHECK (rig.core.isLoopGuardTripped (0));
    CHECK (rig.core.getLoopGuardTrips (0) > trips);

    // prepare() takes the switch from Config, so a call before it is discarded
    // rather than remembered. That is worth pinning: it is the shape of bug
    // that looks like the switch being ignored at random.
    Rig fresh (1, 1, 64);
    fresh.config.loopGuardEnabled = true;
    fresh.level (0, 0, 1.0f);
    fresh.core.setLoopGuardEnabled (false);
    CHECK (fresh.prepare());
    CHECK (fresh.core.isLoopGuardEnabled());
}

//==============================================================================
// THE GATE. The audio callback degrades to silence and never waits.
//==============================================================================

static void testEffectsEngineReadyGate()
{
    using namespace engine_test;

    Rig rig (1, 2, 64);
    rig.level (0, 0, 1.0f);

    std::vector<float> out ((size_t) 64, 7.0f);

    // Before prepare: no crash, no stale buffer handed back, silence.
    CHECK (! rig.core.pullReturn (0, out.data(), 64));
    CHECK (isSilent (out));

    CHECK (rig.prepare());

    const auto signal = module_test::awkwardBlock (64, 11);
    rig.callbackOn (0, signal.data());

    std::fill (out.begin(), out.end(), 7.0f);
    CHECK (rig.core.pullReturn (0, out.data(), 64));
    CHECK (! isSilent (out));

    // Out of range is silence too, not an index into whatever is next in
    // memory.
    std::fill (out.begin(), out.end(), 7.0f);
    CHECK (! rig.core.pullReturn (99, out.data(), 64));
    CHECK (isSilent (out));

    // After release: the same, for as long as the app leaves it torn down.
    rig.core.release();
    CHECK (! rig.core.isReady());

    std::fill (out.begin(), out.end(), 7.0f);
    CHECK (! rig.core.pullReturn (0, out.data(), 64));
    CHECK (isSilent (out));

    std::fill (out.begin(), out.end(), 7.0f);
    CHECK (! rig.core.pullReturn (0, out.data(), 64));
    CHECK (isSilent (out));
}

static void testEffectsEnginePullDuringRebuild()
{
    using namespace engine_test;

    // prepare() holds the lock across the whole reallocation, which is the only
    // thing that makes a rebuild safe under a live callback. The callback must
    // come out of it promptly with silence rather than blocking on it or
    // reading a ring that is being destroyed.
    Rig rig (2, 4, 64);
    rig.level (0, 0, 1.0f);
    CHECK (rig.prepare());

    std::atomic<bool> stop { false };
    std::atomic<int> rebuilds { 0 };

    std::thread rebuilder ([&rig, &stop, &rebuilds]
    {
        // No count of its own: a rebuilder that could finish before the puller
        // was scheduled would leave the puller with nothing to race against.
        while (! stop.load (std::memory_order_relaxed))
        {
            rig.core.prepare (rig.config, rig.rings);
            rig.core.setFeedMatrices (rig.delays.data(), rig.levels.data(), rig.hf.data(),
                                      rig.stride, rig.nSrc, rig.nFx);
            rebuilds.fetch_add (1, std::memory_order_relaxed);
        }
    });

    std::vector<float> out ((size_t) 64, 0.0f);
    bool garbage = false;
    int pulls = 0;

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds (3);

    // Runs until there has been enough of both to be a race, and is bounded by
    // a clock as well so a loaded runner can never hang the suite here.
    while (std::chrono::steady_clock::now() < deadline
           && (pulls < 500 || rebuilds.load (std::memory_order_relaxed) < 20))
    {
        std::fill (out.begin(), out.end(), 7.0f);

        if (! rig.core.pullReturn (0, out.data(), 64) && ! isSilent (out))
            garbage = true;

        ++pulls;
    }

    stop.store (true, std::memory_order_relaxed);
    rebuilder.join();

    CHECK (pulls > 0);
    CHECK (rebuilds.load (std::memory_order_relaxed) > 0);
    CHECK (! garbage);                                   // a refused pull always leaves silence
}

//==============================================================================
// EMERGENCY CLEAR. Per channel it drops that chain's tails; only "all" touches
// the source history that every channel and the reverb send share.
//==============================================================================

static void testEffectsEngineRequestClear()
{
    using namespace engine_test;

    // A delay with feedback, so there is a tail to drop in the first place.
    EffectChannelParams p;
    p.delay.bypass = 0;
    p.delay.timeMs = 10.0f;
    p.delay.feedback = 70.0f;
    p.delay.mix = 100.0f;
    p.revision = 1;

    for (int mode = 0; mode < 3; ++mode)          // 0 = no clear, 1 = one channel, 2 = all
    {
        Rig rig (1, 2, 64);
        rig.config.moduleFactory = &createModule;
        rig.level (0, 0, 1.0f);
        rig.level (0, 1, 1.0f);
        CHECK (rig.prepare());

        rig.core.publishChannelParams (0, p);
        rig.core.publishChannelParams (1, p);

        const auto hot = module_test::awkwardBlock (64, 3);
        rig.callbackOn (0, hot.data());

        // Long enough for the first taps to come back: the delay is 10 ms and
        // a block is 1.33 ms.
        for (int b = 0; b < 20; ++b)
            rig.callbackSilent();

        CHECK (! isSilent (rig.popped[0]));
        CHECK (! isSilent (rig.popped[1]));

        if (mode == 1)
            rig.core.requestClear (0);
        else if (mode == 2)
            rig.core.requestClear (-1);

        // Two callbacks: one for the batch that honours it, one to pop what
        // that batch produced.
        rig.callbackSilent();
        rig.callbackSilent();

        if (mode == 0)
        {
            CHECK (rig.core.getClearCount() == 0);
            CHECK (! isSilent (rig.popped[0]));
            CHECK (! isSilent (rig.popped[1]));
        }
        else if (mode == 1)
        {
            CHECK (rig.core.getClearCount() == 1);
            CHECK (isSilent (rig.popped[0]));     // its tail is gone
            CHECK (! isSilent (rig.popped[1]));   // and nobody else's is
        }
        else
        {
            CHECK (rig.core.getClearCount() == 1);
            CHECK (isSilent (rig.popped[0]));
            CHECK (isSilent (rig.popped[1]));
        }
    }
}

static void testEffectsEngineClearAllWipesHistory()
{
    using namespace engine_test;

    // A feed delay of two blocks makes the shared source history observable:
    // what a batch renders was written two batches ago, so wiping the history
    // shows up as silence where the delayed copy would have been.
    const float twoBlocksMs = 2000.0f * 64.0f / 48000.0f;

    for (int mode = 0; mode < 3; ++mode)          // 0 = no clear, 1 = one channel, 2 = all
    {
        Rig rig (1, 1, 64);
        rig.level (0, 0, 1.0f);
        rig.delayMs (0, 0, twoBlocksMs);
        CHECK (rig.prepare());

        const auto hot = dcBlock (64, 0.5f);
        rig.callbackOn (0, hot.data());           // the only audio there will be

        if (mode == 1)
            rig.core.requestClear (0);
        else if (mode == 2)
            rig.core.requestClear (-1);

        float loudest = 0.0f;
        for (int b = 0; b < 8; ++b)
        {
            rig.callbackSilent();
            loudest = std::max (loudest, peakOf (rig.popped[0]));
        }

        if (mode == 2)
            CHECK (loudest == 0.0f);              // the history went with it
        else
            CHECK (loudest > 0.1f);               // a per-channel clear leaves it alone
    }
}

//==============================================================================
// THE LOOP GUARD. It may only ever touch the effect -> effect bus.
//==============================================================================

static void testEffectsEngineLoopGuardSparesInputRows()
{
    using namespace engine_test;

    const int block = 64;
    const int numBlocks = 90;

    std::vector<std::vector<float>> guarded, reference;
    std::uint32_t trips = 0;
    bool tripped = false;

    for (int pass = 0; pass < 2; ++pass)
    {
        Rig rig (1, 1, block);
        rig.config.loopGuardCeilingDb = 6.0f;
        rig.config.loopGuardTripSeconds = 0.004;          // three blocks at 64/48k
        rig.config.loopGuardRampDownSeconds = 0.0005;     // a completion time: gone inside a block
        rig.config.loopGuardReleaseSeconds = 10.0;        // far longer than this test runs
        rig.level (0, 0, 1.0f);

        // Pass 0 is the runaway: the channel feeds itself at a gain above one.
        // Pass 1 is the same engine with that route absent, which is exactly
        // what the guard should leave behind once it has clamped the bus.
        if (pass == 0)
            rig.level (rig.firstFx + 0, 0, 1.5f);

        CHECK (rig.prepare());

        std::vector<std::vector<float>> captured;

        for (int b = 0; b < numBlocks; ++b)
        {
            const auto in = module_test::awkwardBlock (block, 200 + b);
            rig.callbackOn (0, in.data());
            captured.push_back (rig.popped[0]);
        }

        if (pass == 0)
        {
            guarded = std::move (captured);
            trips = rig.core.getLoopGuardTrips (0);
            tripped = rig.core.isLoopGuardTripped (0);
        }
        else
        {
            reference = std::move (captured);
            CHECK (rig.core.getLoopGuardTrips (0) == 0);  // nothing to trip on
        }
    }

    CHECK (trips == 1);
    CHECK (tripped);

    // The runaway happened: the early blocks are not the reference.
    bool divergedEarly = false;
    for (size_t b = 4; b < 12 && b < guarded.size(); ++b)
        for (size_t i = 0; i < guarded[b].size(); ++i)
            if (! bitEqualFloat (guarded[b][i], reference[b][i]))
                divergedEarly = true;
    CHECK (divergedEarly);

    // And once the guard has settled the bus at zero, what is left is the
    // input contribution BIT FOR BIT. That is the assertion the two-pass feed
    // exists for: the guard scales its own scratch row, so the input rows are
    // summed identically whether it has tripped or not.
    bool tailIdentical = true;
    for (size_t b = guarded.size() - 10; b < guarded.size(); ++b)
        for (size_t i = 0; i < guarded[b].size(); ++i)
            if (! bitEqualFloat (guarded[b][i], reference[b][i]))
                tailIdentical = false;
    CHECK (tailIdentical);
}

//==============================================================================
// The non-finite guard, on the one shape the chain's own guards cannot see.
//==============================================================================

static void testEffectsEngineScansWholeReturnBlock()
{
    using namespace engine_test;

    Rig rig (1, 1, 64);
    rig.config.moduleFactory = &nanFactory;
    rig.level (0, 0, 1.0f);
    CHECK (rig.prepare());

    EffectChannelParams p;
    p.trem.bypass = 0;
    p.revision = 1;
    rig.core.publishChannelParams (0, p);

    const auto in = module_test::awkwardBlock (64, 17);

    for (int b = 0; b < 4; ++b)
    {
        rig.callbackOn (0, in.data());

        // Silence rather than a NaN on its way to a speaker, every block.
        for (int i = 0; i < 64; ++i)
            CHECK (std::isfinite (rig.popped[0][(size_t) i]));
    }

    CHECK (rig.core.getNanTrips (0) >= 1);
    CHECK (isSilent (rig.popped[0]));
}

//==============================================================================
// The thread wrapper. Bounded by an iteration count AND a clock, so it cannot
// hang whatever the scheduler does.
//
// Two of its assertions ARE timings, and deliberately: a test that never checks
// the driver ran would pass for an engine whose thread does nothing. sawAudio
// and a batch count above zero need the driver to be scheduled inside five
// seconds against a one-millisecond callback, which is a machine problem rather
// than a failure if it ever fires. Everything else here - the join, the flag,
// the silent pull afterwards - holds unconditionally.
//==============================================================================

static void testEffectsEngineThreadedSmoke()
{
    using namespace engine_test;
    using Clock = std::chrono::steady_clock;

    const int block = 128;
    const int nIn = 2, nFx = 4;
    const int nSrc = nIn + nFx;
    const int firstFx = nIn;

    std::vector<std::unique_ptr<srt::SharedInputRingBuffer>> rings;
    for (int i = 0; i < nSrc; ++i)
    {
        auto r = std::make_unique<srt::SharedInputRingBuffer>();
        r->setSize (block * 8);
        rings.push_back (std::move (r));
    }

    std::vector<float> levels ((size_t) (nSrc * nFx), 0.0f);
    for (int src = 0; src < nIn; ++src)
        for (int fx = 0; fx < nFx; ++fx)
            levels[(size_t) (src * nFx + fx)] = 0.5f;

    EffectsEngine engine;
    EffectsEngineCore::Config config;
    config.sampleRate = 48000.0;
    config.blockSize = block;
    config.numSources = nSrc;
    config.numEffects = nFx;
    config.matrixStride = nFx;
    config.firstEffectSourceRow = firstFx;
    config.workerThreads = 2;
    config.maxFeedDelaySeconds = 0.05;
    config.maxEffectDelaySeconds = 0.25;
    config.moduleFactory = nullptr;

    CHECK (engine.prepare (config, rings));
    engine.setFeedMatrices (nullptr, levels.data(), nullptr, nFx, nSrc, nFx);
    CHECK (engine.isReady());
    CHECK (engine.startThread());

    std::vector<std::vector<float>> popped ((size_t) nFx, std::vector<float> ((size_t) block, 0.0f));
    const auto input = module_test::awkwardBlock (block, 23);

    bool sawAudio = false;
    int callbacks = 0;
    const auto deadline = Clock::now() + std::chrono::seconds (5);

    // Keeps going until the driver has actually been seen, so the assertion
    // below fails for a broken engine rather than for a busy runner, and stops
    // at the deadline either way.
    while (Clock::now() < deadline && (callbacks < 200 || ! sawAudio))
    {
        for (int fx = 0; fx < nFx; ++fx)
        {
            engine.pullReturn (fx, popped[(size_t) fx].data(), block);
            if (! isSilent (popped[(size_t) fx]))
                sawAudio = true;
        }

        for (int i = 0; i < nIn; ++i)
            rings[(size_t) i]->write (input.data(), block);

        for (int fx = 0; fx < nFx; ++fx)
            rings[(size_t) (firstFx + fx)]->write (popped[(size_t) fx].data(), block);

        engine.notifyInputAvailable();
        ++callbacks;

        std::this_thread::sleep_for (std::chrono::milliseconds (1));
    }

    CHECK (callbacks > 0);
    CHECK (sawAudio);                                     // the driver really ran
    CHECK (engine.getBatchCount() > 0);

    for (int fx = 0; fx < nFx; ++fx)
        CHECK (engine.getCore().getNanTrips (fx) == 0);

    // Teardown under a live puller: the thread joins, the flag drops, and a
    // pull afterwards is silence rather than a crash.
    engine.release();
    CHECK (! engine.isThreadRunning());
    CHECK (! engine.isReady());

    std::vector<float> out ((size_t) block, 7.0f);
    CHECK (! engine.pullReturn (0, out.data(), block));
    CHECK (isSilent (out));
}

static void testStereoPassThroughIdentity()
{
    using namespace spatcore::dsp;
    PassThroughStereoDecomposer d;
    StereoDecomposerConfig cfg;
    CHECK (d.prepare (48000.0, 512, cfg));
    CHECK (d.getLatencyMs() == 0.0f);
    CHECK (d.getNumActiveSlices() == 3);   // centre + L + R

    // Bit-exact copy, several block sizes including partial blocks.
    for (int numSamples : { 512, 64, 480, 1 })
    {
        std::vector<float> left ((size_t) numSamples), right ((size_t) numSamples);
        std::vector<std::vector<float>> slices ((size_t) StereoDecomposer::kMaxSlices,
                                                std::vector<float> ((size_t) numSamples, -1.0f));
        float* slicePtrs[StereoDecomposer::kMaxSlices];
        for (int k = 0; k < StereoDecomposer::kMaxSlices; ++k)
            slicePtrs[k] = slices[(size_t) k].data();

        for (int n = 0; n < numSamples; ++n)
        {
            left[(size_t) n]  = stereoTestSample (0, n);
            right[(size_t) n] = stereoTestSample (1, n);
        }

        d.process (left.data(), right.data(), slicePtrs, numSamples);

        for (int n = 0; n < numSamples; ++n)
        {
            CHECK (slices[0][(size_t) n] == 0.0f);                // centre: silent
            CHECK (slices[1][(size_t) n] == left[(size_t) n]);    // bit-equal
            CHECK (slices[2][(size_t) n] == right[(size_t) n]);
        }
    }

    // Slice state: centre anchored at azimuth 0, pass-through slices at the
    // width extremes, full confidence.
    StereoSliceState state[StereoDecomposer::kMaxSlices];
    d.getSliceState (state);
    CHECK (state[0].active && state[0].azimuth ==  0.0f && state[0].confidence == 1.0f);
    CHECK (state[1].active && state[1].azimuth == -1.0f && state[1].confidence == 1.0f);
    CHECK (state[2].active && state[2].azimuth ==  1.0f);
    for (int k = 3; k < StereoDecomposer::kMaxSlices; ++k)
        CHECK (! state[k].active);
}

static void testStereoReconstructionInvariant()
{
    using namespace spatcore::dsp;
    PassThroughStereoDecomposer d;
    CHECK (d.prepare (96000.0, 256, {}));

    for (int block = 0; block < 16; ++block)
        checkStereoReconstruction (d, 256, block * 256);
}

static void testStereoInactiveSlotsCleared()
{
    // Covered inside checkStereoReconstruction via the -12345 sentinel prefill;
    // this test pins the property on its own so a regression names it.
    using namespace spatcore::dsp;
    PassThroughStereoDecomposer d;
    CHECK (d.prepare (48000.0, 128, {}));
    checkStereoReconstruction (d, 128, 0);
}

static void testStereoConfigChangeStability()
{
    using namespace spatcore::dsp;
    PassThroughStereoDecomposer d;
    StereoDecomposerConfig cfg;
    CHECK (d.prepare (48000.0, 256, cfg));

    for (int slices : { 2, 5, 2, 6, 3 })
    {
        cfg.activeSlices = slices;
        d.setConfig (cfg);
        checkStereoReconstruction (d, 256, slices * 1000);
    }

    // Out-of-envelope values clamp rather than break the contract.
    cfg.activeSlices = 99;
    d.setConfig (cfg);
    CHECK (d.getNumActiveSlices() <= StereoDecomposer::kMaxSlices);
    cfg.activeSlices = -3;
    d.setConfig (cfg);
    CHECK (d.getNumActiveSlices() >= 2);
    checkStereoReconstruction (d, 256, 777);

    // reset() returns to silence without reallocating (stateless here, but the
    // call must exist and be harmless for every backend).
    d.reset();
    checkStereoReconstruction (d, 256, 888);
}

int main()
{
    try
    {
        testRenderSourceMapBuild();
        testRenderSourceMapEffectsLayout();
        testSharedInputRingWrapCounter();
        testStereoPassThroughIdentity();
        testStereoReconstructionInvariant();
        testStereoInactiveSlotsCleared();
        testStereoConfigChangeStability();
        testAcousticTapNullIdentity();
        testAcousticTapDelayPlacement();
        testAcousticTapAirAbsorptionShelf();
        testAcousticTapAccumulatesAndClamps();
        testAcousticTapFastPathMatchesGeneral();
        testReverbReturnProcessorMatrixStride();
        testReverbReturnProcessorAccumulatesOntoDirect();
        testReverbReturnProcessorPerOutputDelay();
        testReverbReturnProcessorSkipBlockAdvances();
        testReverbSendMatrixStride();
        testReverbSendMatrixPerNodeDelay();
        testReverbSendMatrixSilentNodeIsSilent();
        testAcousticSendMatrixAlias();
        testAcousticSendMatrixSourceRange();
        testAcousticSendMatrixHistoryLength();
        testLockFreeRingBuffer();
        testDelayTargetSmootherDeterminism();
        testRtSnapshot();
        testRtTripleBuffer();
        testOnePoleSmoother();
        testFastDecibels();
        testLfoPhasor();
        testFractionalDelayLine();
        testDcBlocker();
        testEnvelopeFollower();
        testWaveshaperCurves();
        testChainOrderParse();
        testEffectParamsPod();
        testModuleSlotBypassFade();
        testModuleSlotVariantSwitch();
        testNaNGuard();
        testTremoloLaw();
        testBitcrusherQuantiser();
        testBitcrusherHoldRate();
        testEffectEQModuleMatchesBank();
        testResetOnFullBypass();
        testEffectModulesNeutralAtDefaults();
        testEffectModulesIdentityWhenActive();
        testChainReorderDeterminism();
        testChainLatencySum();
        testChainBypassAndMute();
        testChainNeutralAndGuarded();
        testLoopGuardTripTimeIsBlockSizeIndependent();
        testLoopGuardRampToZeroIsMonotonic();
        testLoopGuardDipBelowCeilingNeverTrips();
        testLoopGuardReleasesAndRampsBack();
        testLoopGuardReleaseWaitsForSilence();
        testLoopGuardHotReturnDelaysRelease();
        testLoopGuardHotReturnVetoIsBounded();
        testLoopGuardRetripsOnTheWayBack();
        testLoopGuardRepeatTripBacksOffTheRelease();
        testLoopGuardIdleIsFreeAndTransparent();
        testLoopGuardSwitchedOffIsInert();
        testLoopGuardZeroHysteresisHasNoDeadBand();
        testLoopGuardNonFinitePeakTripsAndHolds();
        testEffectsEngineBlockLedger();
        testEffectsEngineWorkerDeterminism();
        testEffectsEngineSlotMeter();
        testEffectsEngineBacklogSkip();
        testEffectsEngineRingWrapResync();
        testEffectsEngineShortFeedHistory();
        testEffectsEngineRefusesUndersizedRing();
        testEffectsEngineUnroutedChannelsStillRun();
        testEffectsEngineNarrowStrideDoesNotAlias();
        testEffectsEngineMuteKeepsChainsRunning();
        testEffectsEngineLoopGuardSwitch();
        testEffectsEngineReadyGate();
        testEffectsEnginePullDuringRebuild();
        testEffectsEngineRequestClear();
        testEffectsEngineClearAllWipesHistory();
        testEffectsEngineLoopGuardSparesInputRows();
        testEffectsEngineScansWholeReturnBlock();
        testEffectsEngineThreadedSmoke();
        testDistortionBypassAndIdentity();
        testDistortionShaperLaw();
        testDistortionShelves();
        testDistortionMixAlignment();
        testDistortionDcBlocker();
        testDistortionOversampling();
        testDistortionResetAndRange();
        testDynamicsBypassAndIdentity();
        testDynamicsCompressorLaw();
        testDynamicsPeakDetectorLaw();
        testDynamicsExpanderLawAndRange();
        testDynamicsKneeAndAutoMakeup();
        testDynamicsLookaheadAndDetectorDelay();
        testDynamicsHoldAndReset();
        testDynamicsVariantAndLatency();
        testDynamicsSampleRates();
        testDynamicsExtremesAndDeterminism();
        testModulationTransparency();
        testModulationThroughZeroAtMixZero();
        testModulationDelayLaw();
        testModulationLoCutPlacement();
        testModulationVoiceLevelLaw();
        testModulationFeedbackAndReset();
        testModulationDenormalFlush();
        testModulationVariants();
        testModulationExtremes();
        testPhaserAllpassLaw();
        testPhaserSweepLaw();
        testPhaserSpreadLaw();
        testPhaserLfoRateAndShape();
        testPhaserTransparency();
        testPhaserResetClearsTail();
        testPhaserStagesAreAVariant();
        testPhaserExtremesStayFinite();
        testPhaserSurvivesNonFiniteInput();
        testPhaserDeterminism();
        testEffectReverbIdentity();
        testEffectReverbPredelay();
        testEffectReverbDecayLaw();
        testEffectReverbTone();
        testEffectReverbWetLevel();
        testEffectReverbResetClearsTail();
        testEffectReverbSizeSpillover();
        testEffectReverbSpilloverRelease();
        testEffectReverbSpilloverRapid();
        testEffectReverbExtremesAndDeterminism();
        testEffectReverbPresets();
        testReverbDelayLineReads();
        testReverbDelayLineHermiteIsPassive();
        testReverbLfoSine();
        testEarlyReflectionProfiles();
        testEffectReverbEarlyReflections();
        testEffectReverbReflectionSpillover();
        testEffectReverbReflectionStorm();
        testMultitapDelayNeutral();
        testMultitapDelayTapPlacement();
        testMultitapDelayTimeModulation();
        testMultitapDelayBlockSizeInvariance();
        testMultitapDelayFeedbackLadder();
        testMultitapDelayShelfReactivation();
        testMultitapDelayDiffusionAndMeter();
        testMultitapDelayResetClearsTail();
        testMultitapDelayInSlot();
        testMultitapDelayExtremesAndRates();
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
        testTrackerQuatToHeadAngles();
        testHeadAttitudePipeline();
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
