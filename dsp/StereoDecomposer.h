#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <algorithm>

namespace spatcore::dsp {

//==============================================================================
/**
    Per-slice state a stereo decomposition backend publishes each block.

    POD on purpose: the caller snapshots it across threads via rt::RtSnapshot,
    and the control-rate side (slice geometry refresh, level-meter diagnostics)
    reads the copy. Azimuth is NORMALISED and stays that way — placing it on the
    stage (scaling by the channel's WIDTH IN METRES along its SPREAD AXIS, then
    the confidence-collapse curve) is deliberately the application's job,
    implemented once for every backend, so spatcore never learns about stage
    geometry or about the units the width is dialled in.
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
    float centreAnchor  = 0.0f;    ///< reserved (Phase 1)

    // Deliberately NO width field. Image width is a distance in metres, which
    // has no 0..1 form a backend could act on, and azimuth is normalised by
    // contract, so no backend needs it. Carrying one pinned at 1.0 would only
    // invite a Phase 1 implementer to re-derive the retired
    // percentage-of-array semantics from it.

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

    Slot convention (all backends): slice 0 is the CENTRE/anchor feed — it
    renders at the channel's own position, so its matrix row doubles as the
    channel's reference row (visualisation, per-channel term derivation). The
    image slices follow from index 1. Phase 0 ships
    PassThroughStereoDecomposer (slice 0 silent, slice 1 = L, slice 2 = R,
    zero latency); Phase 1 puts the crossover bass + centre content on
    slice 0, exactly as this layout anticipates — the channel type, snapshots
    and render mapping must not change when it does.

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
        activeSlices <= kMaxSlices and any crossover. Returns false on
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
    Phase 0 backend: slice 0 = centre (active but silent — it anchors the
    channel's reference row at the anchor position), slice 1 = left,
    slice 2 = right, slices 3..5 cleared. Zero latency; reconstruction is
    bit-exact by construction (0 + L + R in, L + R out). This is the permanent
    A/B control condition the Phase 1 decomposition must beat audibly on real
    program material (doc §8).
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

        // Slice 0 (centre) carries no audio in pass-through, but stays an
        // ACTIVE slot: its row renders at the anchor and is the channel's
        // reference row.
        juce::FloatVectorOperations::clear (sliceOut[0], numSamples);
        juce::FloatVectorOperations::copy (sliceOut[1], left,  numSamples);
        juce::FloatVectorOperations::copy (sliceOut[2], right, numSamples);

        // Inactive slots are cleared, never stale (hard contract).
        for (int k = 3; k < kMaxSlices; ++k)
            juce::FloatVectorOperations::clear (sliceOut[k], numSamples);
    }

    float getLatencyMs() const noexcept override { return 0.0f; }
    int getNumActiveSlices() const noexcept override { return 3; }

    void getSliceState (StereoSliceState* dest) const noexcept override
    {
        for (int k = 0; k < kMaxSlices; ++k)
            dest[k] = {};

        // Centre at the anchor (silent here, but active so its row renders),
        // then the two pass-through slices at the width extremes with full
        // confidence: the application maps ±1 to half the channel's stereo
        // width in metres, along its spread axis.
        dest[0] = {  0.0f, 1.0f, 1.0f, true };
        dest[1] = { -1.0f, 1.0f, 1.0f, true };
        dest[2] = {  1.0f, 1.0f, 1.0f, true };
    }

private:
    StereoDecomposerConfig config;
    bool prepared = false;
};

} // namespace spatcore::dsp
