#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <algorithm>

namespace spatcore::dsp {

//==============================================================================
/**
    Per-slice state a stereo decomposition backend publishes each block.

    POD on purpose: the caller snapshots it across threads via rt::RtSnapshot,
    and the control-rate side (slice geometry refresh, level-meter diagnostics)
    reads the copy. Azimuth is NORMALISED — mapping it to metres on the array
    (the usable-span clamp, the confidence-collapse curve) is deliberately the
    application's job, implemented once for every backend, so spatcore never
    learns about stage geometry.
*/
struct StereoSliceState
{
    float azimuth    = 0.0f;   ///< -1 = hard left .. +1 = hard right
    float gainLinear = 0.0f;   ///< rendered gain hint (meters/diagnostics)
    float confidence = 1.0f;   ///< 0..1; low collapses the slice toward the anchor
    bool  active     = false;
};

//==============================================================================
/** Backend configuration. May change live between blocks via setConfig(). */
struct StereoDecomposerConfig
{
    int   activeSlices  = 2;       ///< N, 2..kMaxSlices
    float crossoverHz   = 110.0f;  ///< below: mono to the anchor (Phase 1)
    float widthFactor   = 1.0f;    ///< 0..1, scales azimuth toward the anchor
    float centreAnchor  = 0.0f;    ///< reserved (Phase 1)

    /** Hop-phase staggering (handoff doc §6): with several stereo channels
        active, each backend offsets its STFT phase by hop * index / count so
        the FFT work of different channels never lands on the same audio
        callback. Present from the first commit so Phase 1 does not have to
        touch every construction site; the pass-through backend ignores them. */
    int staggerIndex = 0;
    int staggerCount = 1;
};

//==============================================================================
/**
    A stereo decomposition backend: two channels in, up to kMaxSlices slice
    feeds out.

    Phase 0 ships PassThroughStereoDecomposer (slice 0 = L, slice 1 = R, zero
    latency). Phase 1 adds the STFT / coherence / panning-index backend behind
    this same interface — the channel type, snapshots and render mapping must
    not change when it does.

    HARD CONTRACT on process() (handoff doc §3.5 and §8 — the reconstruction
    invariant is what makes this feature safe in front of an audience):

      - sum over k of sliceOut[k][n]  ==  left[n] + right[n]  for every n,
        within -120 dBFS, for every reachable config;
      - slice slots k >= getNumActiveSlices() are CLEARED every block, never
        left stale;
      - the output for sample n depends only on input up to n, delayed by
        exactly getLatencyMs(), constant between prepare() calls.

    Threading: prepare() allocates and is non-RT. reset(), setConfig(),
    process() and getSliceState() are RT-safe — no allocation, no locks, no
    logging, noexcept.
*/
class StereoDecomposer
{
public:
    static constexpr int kMaxSlices = 6;   ///< 5 azimuth slices + ambience bed

    virtual ~StereoDecomposer() = default;

    /** Non-RT. Allocates everything the backend will ever need for any config
        reachable from `cfg` without a re-prepare — in particular any
        activeSlices <= kMaxSlices and any crossover/width. Returns false on
        an unusable format (sampleRate/maxBlockSize <= 0). */
    virtual bool prepare (double sampleRate, int maxBlockSize,
                          const StereoDecomposerConfig& cfg) = 0;

    /** RT-safe. Flush all internal state to silence. Never reallocates. */
    virtual void reset() noexcept = 0;

    /** RT-safe, callable every block. Values outside the prepared envelope
        are clamped, not honoured. */
    virtual void setConfig (const StereoDecomposerConfig& cfg) noexcept = 0;

    /** RT-safe. `sliceOut` holds kMaxSlices write-only pointers of
        `numSamples` each; `left`/`right` never alias any `sliceOut[k]`. */
    virtual void process (const float* left, const float* right,
                          float* const* sliceOut, int numSamples) noexcept = 0;

    /** Constant between prepare() calls. 0 for pass-through; ~21 ms for the
        Phase 1 STFT backend. The render side compensates per source against
        the maximum across active sources (never negative, so it cannot fight
        the delay clamp at zero). */
    virtual float getLatencyMs() const noexcept = 0;

    virtual int getNumActiveSlices() const noexcept = 0;

    /** RT-safe. Fills exactly kMaxSlices entries. */
    virtual void getSliceState (StereoSliceState* dest) const noexcept = 0;
};

//==============================================================================
/**
    Phase 0 backend: slice 0 = left, slice 1 = right, slices 2..5 cleared.
    Zero latency; reconstruction is bit-exact by construction (L + R in,
    L + R out). This is the permanent A/B control condition the Phase 1
    decomposition must beat audibly on real program material (doc §8).
*/
class PassThroughStereoDecomposer final : public StereoDecomposer
{
public:
    bool prepare (double sampleRate, int maxBlockSize,
                  const StereoDecomposerConfig& cfg) override
    {
        if (sampleRate <= 0.0 || maxBlockSize <= 0)
            return false;

        setConfig (cfg);
        prepared = true;
        return true;
    }

    void reset() noexcept override {}   // stateless

    void setConfig (const StereoDecomposerConfig& cfg) noexcept override
    {
        config = cfg;
        config.activeSlices = juce::jlimit (2, (int) kMaxSlices, cfg.activeSlices);
    }

    void process (const float* left, const float* right,
                  float* const* sliceOut, int numSamples) noexcept override
    {
        if (numSamples <= 0)
            return;

        juce::FloatVectorOperations::copy (sliceOut[0], left,  numSamples);
        juce::FloatVectorOperations::copy (sliceOut[1], right, numSamples);

        // Inactive slots are cleared, never stale (hard contract).
        for (int k = 2; k < kMaxSlices; ++k)
            juce::FloatVectorOperations::clear (sliceOut[k], numSamples);
    }

    float getLatencyMs() const noexcept override { return 0.0f; }
    int getNumActiveSlices() const noexcept override { return 2; }

    void getSliceState (StereoSliceState* dest) const noexcept override
    {
        for (int k = 0; k < kMaxSlices; ++k)
            dest[k] = {};

        // The two pass-through slices sit at the width extremes with full
        // confidence: the application maps ±1 to the channel's stereo width.
        dest[0] = { -1.0f, 1.0f, 1.0f, true };
        dest[1] = {  1.0f, 1.0f, 1.0f, true };
    }

private:
    StereoDecomposerConfig config;
    bool prepared = false;
};

} // namespace spatcore::dsp
