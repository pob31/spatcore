#pragma once

/*
    WfsFrHostState - host-side Floor Reflection state shared by the GPU WFS
    backends (Metal + CUDA).

    JUCE-free. Owns everything FR-related that runs on the CPU side of the
    GPU algorithm, mirroring InputBufferProcessor's per-input FR machinery:

      - Per-input FR pre-filter chain (optional LowCut + optional configurable
        HighShelf, WFSBiquadFilter reused verbatim so coefficient math, clamps
        and enable-freeze semantics match the CPU reference exactly). Runs on
        the pump thread over numInputs x blockSize samples per launch (tens of
        microseconds) and fills the frIn staging buffer the kernel appends to
        the FR ring.
      - Per-(in,out) diffusion jitter (the shared Max-prototype model in
        FrDiffusionModel.h, identical to the CPU processors): rand~-style
        band-limited noise, squared, amplitude-scaled and rampsmooth'd per
        routing pair. WFS gather: advanced ONE span per launch (the kernel's
        prev->curr per-sample interpolation approximates the intra-block
        trajectory). OutputBuffer scatter: stepped truly per sample into the
        per-sample delay buffer, exactly matching the CPU scatter path.
      - FR curr-matrix computation: absolute FR delay in samples with the
        pipeline latency pre-subtracted from the ABSOLUTE delay
        (direct + extra + jitter - L), preserving the FR-vs-direct offset.

    Threading contract: setFRFilterParams / setFRDiffusion may be called from
    the 50 Hz timer thread (atomics); everything else is called by the single
    pump thread between launches. The pump applies the param atomics to the
    filters at block start (WFSBiquadFilter recalculates only on change).
*/

#include "../dsp/WFSBiquadFilter.h"
#include "../dsp/FrDiffusionModel.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace spatcore::gpu {

class WfsFrHostState
{
public:
    WfsFrHostState() = default;

    void prepare (int numInputs, int numOutputs, double sampleRate)
    {
        numIn = std::max (1, numInputs);
        numOut = std::max (1, numOutputs);

        // unique_ptr elements: InputParams holds atomics (non-movable), so the
        // vector stores pointers to keep resize/relocation legal.
        params.clear();
        params.reserve ((size_t) numIn);
        for (int i = 0; i < numIn; ++i)
            params.push_back (std::make_unique<InputParams>());

        lowCutFilters.clear();
        highShelfFilters.clear();
        lowCutFilters.resize ((size_t) numIn);
        highShelfFilters.resize ((size_t) numIn);
        for (int i = 0; i < numIn; ++i)
        {
            auto& lc = lowCutFilters[(size_t) i];
            lc.prepare (sampleRate);
            lc.setType (WFSBiquadFilter::FilterType::LowCut);
            lc.setFrequency (100.0f);

            auto& hs = highShelfFilters[(size_t) i];
            hs.prepare (sampleRate);
            hs.setType (WFSBiquadFilter::FilterType::HighShelf);
            hs.setFrequency (3000.0f);
            hs.setGainDb (-2.0f);
            hs.setSlope (0.4f);
        }

        srHz = (float) sampleRate;
        jitterStates.assign ((size_t) numIn * (size_t) numOut, FrDiffusion::State {});
        for (int in = 0; in < numIn; ++in)
            for (int out = 0; out < numOut; ++out)
                FrDiffusion::resetState (jitterStates[(size_t) in * (size_t) numOut + (size_t) out],
                                         FrDiffusion::makeKey (in, out));
        jitterSamples.assign ((size_t) numIn * (size_t) numOut, 0.0f);
        launchCounter = 0;

        // OutputBuffer (scatter) per-sample state.
        baseCurrScratch.assign ((size_t) numIn * (size_t) numOut, 0.0f);
        subStepCounter = 0;
    }

    void reset()
    {
        for (auto& f : lowCutFilters)    f.reset();
        for (auto& f : highShelfFilters) f.reset();
        for (int in = 0; in < numIn; ++in)
            for (int out = 0; out < numOut; ++out)
                FrDiffusion::resetState (jitterStates[(size_t) in * (size_t) numOut + (size_t) out],
                                         FrDiffusion::makeKey (in, out));
        std::fill (jitterSamples.begin(), jitterSamples.end(), 0.0f);
        launchCounter = 0;
        subStepCounter = 0;
        haveBase = false;
    }

    // ==== 50 Hz timer-thread setters (mirror InputBufferProcessor's) ====

    void setFRFilterParams (int inputIndex,
                            bool lowCutActive, float lowCutFreq,
                            bool highShelfActive, float highShelfFreq,
                            float highShelfGain, float highShelfSlope) noexcept
    {
        if (inputIndex < 0 || inputIndex >= (int) params.size())
            return;
        auto& p = *params[(size_t) inputIndex];
        p.lowCutActive.store (lowCutActive, std::memory_order_release);
        p.lowCutFreq.store (lowCutFreq, std::memory_order_release);
        p.highShelfActive.store (highShelfActive, std::memory_order_release);
        p.highShelfFreq.store (highShelfFreq, std::memory_order_release);
        p.highShelfGain.store (highShelfGain, std::memory_order_release);
        p.highShelfSlope.store (highShelfSlope, std::memory_order_release);
    }

    void setFRDiffusion (int inputIndex, float diffusionPercent) noexcept
    {
        if (inputIndex < 0 || inputIndex >= (int) params.size())
            return;
        // Shared zone-mapped model (FrDiffusionModel.h); publish the fraction.
        params[(size_t) inputIndex]->diffusionAmount.store (
            std::min (1.0f, std::max (0.0f, diffusionPercent * 0.01f)),
            std::memory_order_release);
    }

    // ==== Pump-thread per-launch steps ====

    /** Applies the param atomics to the per-input filters and runs the FR
        pre-filter chain over the launch block: frInFlat[in*blockSize + s] =
        highShelf(lowCut(input)) per active flags. Null input channels are
        processed as silence (keeps filter state evolution identical to a
        silent feed). The FR staging is ALWAYS written, filtered or not,
        mirroring the CPU's unconditional FR-ring write. */
    void filterBlock (const float* const* inputs, float* frInFlat, int blockSize)
    {
        for (int in = 0; in < numIn; ++in)
        {
            auto& p = *params[(size_t) in];
            const bool lcActive = p.lowCutActive.load (std::memory_order_acquire);
            const bool hsActive = p.highShelfActive.load (std::memory_order_acquire);

            auto& lc = lowCutFilters[(size_t) in];
            auto& hs = highShelfFilters[(size_t) in];

            // Apply params only while active, like the CPU setter (a disabled
            // filter's parameters stay frozen). Recalc happens only on change.
            if (lcActive)
                lc.setFrequency (p.lowCutFreq.load (std::memory_order_acquire));
            if (hsActive)
            {
                hs.setFrequency (p.highShelfFreq.load (std::memory_order_acquire));
                hs.setGainDb (p.highShelfGain.load (std::memory_order_acquire));
                hs.setSlope (p.highShelfSlope.load (std::memory_order_acquire));
            }

            const float* src = (inputs != nullptr) ? inputs[in] : nullptr;
            float* dst = frInFlat + (size_t) in * (size_t) blockSize;
            for (int s = 0; s < blockSize; ++s)
            {
                float v = (src != nullptr) ? src[s] : 0.0f;
                if (lcActive)
                    v = lc.processSample (v);
                if (hsActive)
                    v = hs.processSample (v);
                dst[s] = v;
            }
        }
    }

    /** Per-input entry point of filterBlock: runs the FR pre-filter chain for a
        SINGLE input row (input `in`), writing frInFlat[in*blockSize + s]. Split
        verbatim from the filterBlock loop body so a host worker pool can run one
        input per item. Per-input biquad state (lowCutFilters[in],
        highShelfFilters[in]) + the disjoint hFrIn row => bit-identical to the
        sequential filterBlock for any item scheduling (section-4 determinism). */
    void filterBlockForInput (int in, const float* const* inputs, float* frInFlat, int blockSize) noexcept
    {
        auto& p = *params[(size_t) in];
        const bool lcActive = p.lowCutActive.load (std::memory_order_acquire);
        const bool hsActive = p.highShelfActive.load (std::memory_order_acquire);

        auto& lc = lowCutFilters[(size_t) in];
        auto& hs = highShelfFilters[(size_t) in];

        if (lcActive)
            lc.setFrequency (p.lowCutFreq.load (std::memory_order_acquire));
        if (hsActive)
        {
            hs.setFrequency (p.highShelfFreq.load (std::memory_order_acquire));
            hs.setGainDb (p.highShelfGain.load (std::memory_order_acquire));
            hs.setSlope (p.highShelfSlope.load (std::memory_order_acquire));
        }

        const float* src = (inputs != nullptr) ? inputs[in] : nullptr;
        float* dst = frInFlat + (size_t) in * (size_t) blockSize;
        for (int s = 0; s < blockSize; ++s)
        {
            float v = (src != nullptr) ? src[s] : 0.0f;
            if (lcActive)
                v = lc.processSample (v);
            if (hsActive)
                v = hs.processSample (v);
            dst[s] = v;
        }
    }

    /** Advances the diffusion jitter by ONE span of blockSize samples per
        launch (shared Max-prototype model; advanceSpan composes exactly with
        the CPU's per-sample stepping, the kernel's prev->curr interpolation
        approximates the intra-block trajectory). Fills
        jitterSamples[in*numOut+out] with the per-pair jitter in audio samples,
        consumed by computeFrCurr(). Streams ALWAYS advance, even at
        diffusion 0, so the amplitude ramp-down decays and stream phase stays
        consistent with the CPU paths. */
    void advanceJitter (int blockSize)
    {
        for (int in = 0; in < numIn; ++in)
        {
            const float d = params[(size_t) in]->diffusionAmount.load (std::memory_order_acquire);
            const auto coeffs = FrDiffusion::computeCoeffs (d, srHz);

            const size_t base = (size_t) in * (size_t) numOut;
            for (int out = 0; out < numOut; ++out)
                jitterSamples[base + (size_t) out] = FrDiffusion::advanceSpan (
                    jitterStates[base + (size_t) out],
                    FrDiffusion::makeKey (in, out), coeffs, blockSize);
        }
        ++launchCounter;
    }

    /** Per-input entry point of advanceJitter for input `in`. `n` is vestigial
        (the model's noise is now indexed by each stream's own segment counter,
        not an external launch index); the caller's hoisted
        currentLaunchIndex()/commitJitterLaunch() protocol is kept unchanged.
        Per-pair stream state (jitterStates/jitterSamples rows for this input)
        is disjoint => item-order-invariant (section 4). */
    void advanceJitterForInput (int in, uint32_t n, int blockSize) noexcept
    {
        (void) n;
        const float d = params[(size_t) in]->diffusionAmount.load (std::memory_order_acquire);
        const auto coeffs = FrDiffusion::computeCoeffs (d, srHz);

        const size_t base = (size_t) in * (size_t) numOut;
        for (int out = 0; out < numOut; ++out)
            jitterSamples[base + (size_t) out] = FrDiffusion::advanceSpan (
                jitterStates[base + (size_t) out],
                FrDiffusion::makeKey (in, out), coeffs, blockSize);
    }

    /** The launch index every item this block passes to advanceJitterForInput
        (== the value the sequential advanceJitter would use for the whole
        block). Read once on the pump thread before the parallelFor. */
    uint32_t currentLaunchIndex() const noexcept { return launchCounter; }

    /** Commit one launch of the diffusion grain: called ONCE after the join,
        the hoisted equivalent of advanceJitter's trailing ++launchCounter. */
    void commitJitterLaunch() noexcept { ++launchCounter; }

    /** Fills the FR curr matrices for this launch (input-major [in*numOut+out]):
        frDelaysCurr in samples = clamp((directMs + extraMs - latencyMs) * srScale
        + jitterSamples, 0, maxDelaySamples); frGainsCurr = frLevels (absolute
        linear). The diffusion grain is added POST-scale in samples - matching
        the CPU paths, which add it post-smoother at the read/write position.
        Null app pointers produce zeros (FR silent). */
    void computeFrCurr (const float* delaysMs, const float* frDelaysMs, const float* frLevels,
                        float latencyMs, float srScale, float maxDelaySamples,
                        float* frDelaysCurr, float* frGainsCurr) const noexcept
    {
        const size_t matrix = (size_t) numIn * (size_t) numOut;
        for (size_t m = 0; m < matrix; ++m)
        {
            const float g = (frLevels != nullptr) ? frLevels[m] : 0.0f;
            frGainsCurr[m] = g;

            float d = 0.0f;
            if (delaysMs != nullptr && frDelaysMs != nullptr)
                d = (delaysMs[m] + frDelaysMs[m] - latencyMs) * srScale + jitterSamples[m];
            frDelaysCurr[m] = std::clamp (d, 0.0f, maxDelaySamples);
        }
    }

    /** Per-input entry point of computeFrCurr: fills the FR curr matrix rows for
        input `in` ([in*numOut, (in+1)*numOut)). Pure elementwise transform of
        the app matrices + this input's jitterSamples (written by
        advanceJitterForInput in the SAME item, so the read-after-write ordering
        holds within the lane). Disjoint rows => item-order-invariant. */
    void computeFrCurrForInput (int in,
                                const float* delaysMs, const float* frDelaysMs, const float* frLevels,
                                float latencyMs, float srScale, float maxDelaySamples,
                                float* frDelaysCurr, float* frGainsCurr) const noexcept
    {
        const size_t base = (size_t) in * (size_t) numOut;
        for (int out = 0; out < numOut; ++out)
        {
            const size_t m = base + (size_t) out;
            const float g = (frLevels != nullptr) ? frLevels[m] : 0.0f;
            frGainsCurr[m] = g;

            float d = 0.0f;
            if (delaysMs != nullptr && frDelaysMs != nullptr)
                d = (delaysMs[m] + frDelaysMs[m] - latencyMs) * srScale + jitterSamples[m];
            frDelaysCurr[m] = std::clamp (d, 0.0f, maxDelaySamples);
        }
    }

    /** OutputBuffer (scatter) variant of the FR delay: fills a PER-SAMPLE absolute
        FR delay buffer frDelaysOut[pair*blockSize + s] (pair = in*numOut+out),
        stepping the diffusion jitter TRULY PER SAMPLE for FR-active pairs —
        exactly matching the CPU OutputBufferProcessor's per-sample stepping.
        (The signal is slew-limited by the model's rampsmooth pole, so the old
        per-sample-noise scatter-hiss concern no longer applies.) Inactive pairs
        advance their stream in one O(1) advanceSpan so phase stays consistent.

        The smoothed base FR delay (direct + extra - L, no jitter) is approximated
        by a prev->curr linear ramp across the launch (basePrev in/out), consistent
        with the rest of the GPU port's delay handling. Diffusion off => jitter 0 =>
        pure base ramp (then this equals the gather path's FR delay).

        Delays are clamped to a MINIMUM of 1 sample: the write-time scatter cannot
        represent d < 1 (the head cell was just read+cleared; a same-cell write
        only re-emerges when the head wraps ~1 s later), so pipeline-compensated
        delays below the latency floor clamp to 1 (the scatter analogue of the
        gather's "below L clamps to the floor" contract).

        frGainsPrev/frGainsCurr gate the per-sample fill: pairs whose FR gain is 0
        in both matrices are skipped (the kernel's doFr gate never reads their
        rows), which removes the O(pairs*blockSize) pump-thread cost for the
        common FR-sparse case. `subBlock` is vestigial (kept so backend call
        sites are unchanged). */
    void computeFrDelaysPerSample (const float* delaysMs, const float* frDelaysMs,
                                   const float* frGainsPrev, const float* frGainsCurr,
                                   float latencyMs, float srScale, float maxDelaySamples,
                                   int blockSize, int subBlock,
                                   std::vector<float>& basePrev,   // [pairs] in/out
                                   float* frDelaysOut) noexcept    // [pairs*blockSize]
    {
        (void) subBlock;
        const int pairs = numIn * numOut;
        if ((int) basePrev.size() != pairs)
            basePrev.assign ((size_t) pairs, 1.0f);

        baseCurrScratch.resize ((size_t) pairs);
        for (int m = 0; m < pairs; ++m)
        {
            float d = 0.0f;
            if (delaysMs != nullptr && frDelaysMs != nullptr)
                d = (delaysMs[m] + frDelaysMs[m] - latencyMs) * srScale;   // base, NO jitter
            baseCurrScratch[(size_t) m] = std::clamp (d, 1.0f, maxDelaySamples);
        }
        if (! haveBase)
        {
            basePrev = baseCurrScratch;
            haveBase = true;
        }

        const float invLen = 1.0f / (float) std::max (1, blockSize);

        for (int in = 0; in < numIn; ++in)
        {
            const float dIn = params[(size_t) in]->diffusionAmount.load (std::memory_order_acquire);
            const auto coeffs = FrDiffusion::computeCoeffs (dIn, srHz);

            for (int out = 0; out < numOut; ++out)
            {
                const size_t m = (size_t) in * (size_t) numOut + (size_t) out;
                const uint32_t key = FrDiffusion::makeKey (in, out);

                // Inactive pair: advance the stream in one O(1) span (the
                // kernel's doFr gate never reads this row).
                if (frGainsPrev != nullptr && frGainsCurr != nullptr
                    && frGainsPrev[m] == 0.0f && frGainsCurr[m] == 0.0f)
                {
                    FrDiffusion::advanceSpan (jitterStates[m], key, coeffs, blockSize);
                    continue;
                }

                float* dst = frDelaysOut + m * (size_t) blockSize;
                const float bp = basePrev[m];
                const float bc = baseCurrScratch[m];
                for (int s = 0; s < blockSize; ++s)
                {
                    const float jit  = FrDiffusion::processSample (jitterStates[m], key, coeffs);
                    const float base = bp + (bc - bp) * ((float) (s + 1) * invLen);
                    dst[(size_t) s] = std::clamp (base + jit, 1.0f, maxDelaySamples);
                }
            }
        }

        for (int m = 0; m < pairs; ++m)
            basePrev[(size_t) m] = baseCurrScratch[(size_t) m];
    }

    //==========================================================================
    // Per-input (fused parallelFor) decomposition of computeFrDelaysPerSample.
    // The pump thread calls the tiny orchestration helpers around the
    // parallelFor; each item runs computeFrDelaysPerSampleForInput for one input
    // lane. The determinism argument (section-4 OB row) is now trivial: each
    // stream's noise is indexed by its own segment counter inside
    // FrDiffusion::State, and every mutable row (jitterStates/basePrev/
    // baseCurrScratch, hFrDelaySamples) is indexed by pair m = in*numOut+out,
    // disjoint per input => item-order-invariant.
    //==========================================================================

    /** Pump-thread setup before the parallelFor: sizes the shared base-delay
        scratch to `pairs`. Call once per block. */
    void beginFrDelaysPerSample (int pairs)
    {
        baseCurrScratch.resize ((size_t) pairs);
    }

    /** Consumes the prev->curr base-delay ramp bootstrap flag (true exactly on
        the first block after prepare()/reset()). Read on the pump thread and
        passed to every item; each lane copies its own baseCurrScratch rows into
        basePrev when true (the per-row form of the sequential
        `if (!haveBase) basePrev = baseCurrScratch`). */
    bool consumeFirstFrDelayBlock() noexcept
    {
        const bool first = ! haveBase;
        haveBase = true;
        return first;
    }

    /** The sub-step ordinal every item passes to
        computeFrDelaysPerSampleForInput this block. Vestigial for the model
        (streams carry their own segment counters) — kept so backend call sites
        are unchanged. */
    uint32_t currentSubStep() const noexcept { return subStepCounter; }

    /** Commit the hoisted sub-step ordinal after the join. Vestigial (see
        currentSubStep), kept so backend call sites are unchanged. */
    void commitSubSteps (int numSubBlocks) noexcept { subStepCounter += (uint32_t) numSubBlocks; }

    /** One input lane of computeFrDelaysPerSample (see the block comment above).
        `firstBlock` from consumeFirstFrDelayBlock(); `subStepBase`/`subBlock`
        are vestigial. */
    void computeFrDelaysPerSampleForInput (int in, bool firstBlock, uint32_t subStepBase,
                                           const float* delaysMs, const float* frDelaysMs,
                                           const float* frGainsPrev, const float* frGainsCurr,
                                           float latencyMs, float srScale, float maxDelaySamples,
                                           int blockSize, int subBlock,
                                           std::vector<float>& basePrev,
                                           float* frDelaysOut) noexcept
    {
        (void) subStepBase;
        (void) subBlock;
        const float invLen = 1.0f / (float) std::max (1, blockSize);
        const float dIn = params[(size_t) in]->diffusionAmount.load (std::memory_order_acquire);
        const auto coeffs = FrDiffusion::computeCoeffs (dIn, srHz);
        const size_t rowBase = (size_t) in * (size_t) numOut;

        // Base FR delay (direct + extra - L, NO jitter) for this input's pairs,
        // plus the first-block prev==curr bootstrap (per row).
        for (int out = 0; out < numOut; ++out)
        {
            const size_t m = rowBase + (size_t) out;
            float d = 0.0f;
            if (delaysMs != nullptr && frDelaysMs != nullptr)
                d = (delaysMs[m] + frDelaysMs[m] - latencyMs) * srScale;
            const float bc = std::clamp (d, 1.0f, maxDelaySamples);
            baseCurrScratch[m] = bc;
            if (firstBlock)
                basePrev[m] = bc;
        }

        for (int out = 0; out < numOut; ++out)
        {
            const size_t m = rowBase + (size_t) out;
            const uint32_t key = FrDiffusion::makeKey (in, out);

            // Inactive pair: advance the stream in one O(1) span (the kernel's
            // doFr gate never reads this row).
            if (frGainsPrev != nullptr && frGainsCurr != nullptr
                && frGainsPrev[m] == 0.0f && frGainsCurr[m] == 0.0f)
            {
                FrDiffusion::advanceSpan (jitterStates[m], key, coeffs, blockSize);
                continue;
            }

            float* dst = frDelaysOut + m * (size_t) blockSize;
            const float bp = basePrev[m];
            const float bc = baseCurrScratch[m];
            for (int s = 0; s < blockSize; ++s)
            {
                const float jit  = FrDiffusion::processSample (jitterStates[m], key, coeffs);
                const float base = bp + (bc - bp) * ((float) (s + 1) * invLen);
                dst[(size_t) s] = std::clamp (base + jit, 1.0f, maxDelaySamples);
            }
        }

        // Final basePrev update for this input's pairs (sequential trailing loop).
        for (int out = 0; out < numOut; ++out)
        {
            const size_t m = rowBase + (size_t) out;
            basePrev[m] = baseCurrScratch[m];
        }
    }

private:
    struct InputParams
    {
        std::atomic<bool>  lowCutActive { false };
        std::atomic<float> lowCutFreq { 100.0f };
        std::atomic<bool>  highShelfActive { false };
        std::atomic<float> highShelfFreq { 3000.0f };
        std::atomic<float> highShelfGain { -2.0f };
        std::atomic<float> highShelfSlope { 0.4f };
        std::atomic<float> diffusionAmount { 0.0f }; // Diffusion fraction 0..1
    };

    int numIn = 0, numOut = 0;

    std::vector<std::unique_ptr<InputParams>> params; // per input (atomics: non-movable)
    std::vector<WFSBiquadFilter> lowCutFilters;      // per input, persistent state
    std::vector<WFSBiquadFilter> highShelfFilters;   // per input, persistent state

    float srHz = 48000.0f;
    uint32_t launchCounter = 0;                      // launch ordinal (vestigial for the model)
    std::vector<FrDiffusion::State> jitterStates;    // [in*numOut+out] per-pair stream state
    std::vector<float> jitterSamples;                // [in*numOut+out] per-pair jitter (audio samples)

    // OutputBuffer (scatter) per-sample diffusion state (computeFrDelaysPerSample).
    std::vector<float> baseCurrScratch;              // [in*numOut+out] this launch's base FR delay (no jitter)
    uint32_t subStepCounter = 0;                     // sub-step ordinal (vestigial for the model)
    bool haveBase = false;                           // prev->curr base-delay ramp bootstrap
};

} // namespace spatcore::gpu

// Extraction-compat alias — app code migrates to qualified names later.
using spatcore::gpu::WfsFrHostState;
