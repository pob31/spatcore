#pragma once

#include "BinauralTypes.h"
#include "HrirSet.h"
#include "../dsp/DelayTargetSmoother.h"
#include "../dsp/WFSHighShelfFilter.h"
#include <juce_dsp/juce_dsp.h>
#include <vector>
#include <memory>
#include <cstdint>
#include <cmath>

namespace spatcore::binaural
{

/**
    Measured-HRTF renderer: uniform-partitioned overlap-save convolution of
    each source with the pre-FFT'd grid spectra of a CookedHrirSet.

    Why NOT juce::dsp::Convolution: its loader/fade machinery is built for
    occasional IR swaps, not per-block direction changes. Here a direction
    change is a pointer swap into the cooked spectra plus a one-block output
    crossfade (dual render), and the ITD — stripped from the HRIRs at load
    time — rides the same smoothed fractional delay lines as the structural
    model, applied AFTER convolution (LTI, order commutes) so one forward FFT
    serves both ears.

    Per source, per block:
      1 × FFT(2P) of [prev|curr] input  →  frequency-domain delay line (K deep)
      per ear: K complex MACs + 1 × IFFT(2P), last P samples valid
      per ear: write into a 1 s delay line, read back at the smoothed
               r/c + measured-ITD tap (DelayTargetSmoother), then the air
               shelf and 1/max(r,1) distance gain — identical distance model
               to the structural mode.

    HRIR spectra follow the NEAREST grid point (5°×10°); a change re-renders
    the block with old+new spectra and crossfades. The ITD is bilinearly
    interpolated over the four surrounding grid points every block, so the
    dominant localization cue moves continuously even between spectrum swaps.

    Set hot-swap: the message thread publishes a new CookedHrirSet under a
    SpinLock; the worker adopts it at a block boundary with the same one-block
    crossfade and parks the old set for the message thread to release
    (collectRetired) — no deallocation ever happens on the render thread.

    prepare()/reset() with the worker stopped; processSource() worker-only.
*/
class SofaHrtfRenderer
{
public:
    void prepare (double newSampleRate, int newBlockSize, int maxSources)
    {
        sampleRate = newSampleRate;
        blockSize = newBlockSize;

        int order = 1;
        while ((1 << order) < 2 * blockSize)
            ++order;
        fftSize = 1 << order;
        spectrumFloats = 2 * fftSize;
        fft = std::make_unique<juce::dsp::FFT> (order);

        maxPartitions = juce::jmax (1, (kMaxHrirTaps + blockSize - 1) / blockSize);
        delayLineLength = (int) (sampleRate * 1.0);

        const int smootherWindow = juce::jmax (2, (int) (sampleRate * 0.005));

        sources.clear();
        sources.resize ((size_t) juce::jmax (0, maxSources));
        for (auto& src : sources)
        {
            src.prevBlock.assign ((size_t) blockSize, 0.0f);
            src.fdl.assign ((size_t) maxPartitions * (size_t) spectrumFloats, 0.0f);
            src.fdlPos = 0;
            src.azIdx = src.elIdx = -1;
            src.fadeFromSet = nullptr;
            for (auto* ear : { &src.left, &src.right })
            {
                ear->delayLine.assign ((size_t) delayLineLength, 0.0f);
                ear->writePos = 0;
                ear->delaySmoother.prepare (smootherWindow);
                ear->airShelf.prepare (sampleRate);
                ear->prevGain = 0.0f;
                ear->gainInit = false;
            }
        }

        fftWork.assign ((size_t) spectrumFloats, 0.0f);
        accumNew.assign ((size_t) spectrumFloats, 0.0f);
        accumOld.assign ((size_t) spectrumFloats, 0.0f);
    }

    void reset()
    {
        for (auto& src : sources)
        {
            std::fill (src.prevBlock.begin(), src.prevBlock.end(), 0.0f);
            std::fill (src.fdl.begin(), src.fdl.end(), 0.0f);
            src.fdlPos = 0;
            src.azIdx = src.elIdx = -1;
            src.fadeFromSet = nullptr;
            for (auto* ear : { &src.left, &src.right })
            {
                std::fill (ear->delayLine.begin(), ear->delayLine.end(), 0.0f);
                ear->writePos = 0;
                ear->delaySmoother.reset();
                ear->airShelf.reset();
                ear->gainInit = false;
            }
        }
    }

    /** Message thread: publish a freshly cooked set (or nullptr to clear).
        The cooked blockSize must match prepare()'s — the app cooks per
        prepare and re-cooks on block-size changes. */
    void publishSet (std::shared_ptr<const CookedHrirSet> cooked)
    {
        const juce::SpinLock::ScopedLockType lock (setLock);
        pendingSet = std::move (cooked);
        pendingGeneration.fetch_add (1, std::memory_order_release);
    }

    /** Message thread, periodically: release sets the worker retired. */
    void collectRetired()
    {
        std::shared_ptr<const CookedHrirSet> toRelease[kRetiredSlots];
        {
            const juce::SpinLock::ScopedLockType lock (setLock);
            for (int i = 0; i < kRetiredSlots; ++i)
                toRelease[i] = std::move (retiredSets[i]);
        }
        // shared_ptr destruction (and any HrirDatabase release) happens here,
        // on the message thread, as the locals go out of scope.
    }

    /** Worker: true once a set has been adopted (engine falls back to the
        structural model otherwise). Adoption happens inside processBlockBegin. */
    bool hasActiveSet() const noexcept { return activeSet != nullptr; }

    /** Worker, once per block BEFORE any processSource call: adopt a pending
        set if one was published. Lock is tryEnter — if the message thread is
        mid-publish we just try again next block. */
    void processBlockBegin()
    {
        // Opportunistically park a finished fade guard (fades all done but a
        // previous park attempt lost the tryEnter race).
        if (fadeGuard != nullptr && fadesPending <= 0)
            retireFadeGuard();

        if (pendingGeneration.load (std::memory_order_acquire) == adoptedGeneration)
            return;

        // A previous swap's guard must be parked before we can start another
        // fade epoch: sources hold raw pointers into it, and those must be
        // cleared before the set becomes eligible for release. If the retired
        // ring is full (message thread stalled), defer adoption to a later
        // block — nothing is ever freed on this thread.
        if (fadeGuard != nullptr)
        {
            for (auto& src : sources)
                src.fadeFromSet = nullptr;   // cancel unfinished fades (rare double swap)
            fadesPending = 0;
            retireFadeGuard();
            if (fadeGuard != nullptr)
                return;
        }

        if (! setLock.tryEnter())
            return;

        std::shared_ptr<const CookedHrirSet> incoming = pendingSet;
        const auto gen = pendingGeneration.load (std::memory_order_acquire);
        setLock.exit();

        if (incoming != nullptr && incoming->blockSize != blockSize)
            incoming = nullptr;   // stale cook for another block size — ignore

        adoptedGeneration = gen;

        // Start a one-block crossfade from the old set on every source that
        // is mid-stream. The old set outlives the fades: each source holds a
        // raw pointer, and the shared_ptr parks in fadeGuard until every
        // source has faded (or the fades are cancelled), then moves to the
        // retired ring for the message thread to release.
        fadesPending = 0;
        for (auto& src : sources)
        {
            src.fadeFromSet = (activeSet != nullptr && src.azIdx >= 0) ? activeSet.get() : nullptr;
            if (src.fadeFromSet != nullptr)
                ++fadesPending;
        }
        fadeGuard = std::move (activeSet);
        activeSet = std::move (incoming);
        if (fadesPending == 0)
            retireFadeGuard();
    }

    /**
        Render one source. Same contract as StructuralHrtfRenderer::
        processSource; numSamples must equal the prepared block size.
    */
    void processSource (int sourceIdx,
                        const SourceDirection& dir,
                        float extraDelayMs,
                        float gainScale,
                        const float* input,
                        float* outL, float* outR,
                        int numSamples,
                        std::int64_t blockStart)
    {
        if (sourceIdx < 0 || sourceIdx >= (int) sources.size()
            || activeSet == nullptr || numSamples != blockSize)
            return;

        auto& src = sources[(size_t) sourceIdx];
        const CookedHrirSet& set = *activeSet;
        const HrirDatabase& db = *set.db;

        // ---- Direction → grid ------------------------------------------------
        float azDeg = juce::radiansToDegrees (dir.azRad);
        if (azDeg < 0.0f) azDeg += 360.0f;
        const float elDeg = juce::jlimit (-90.0f, 90.0f, juce::radiansToDegrees (dir.elRad));

        const int azNearest = ((int) std::lround (azDeg / HrirDatabase::kAzStepDeg)) % HrirDatabase::kNumAz;
        const int elNearest = juce::jlimit (0, HrirDatabase::kNumEl - 1,
                                            (int) std::lround ((elDeg + 90.0f) / HrirDatabase::kElStepDeg));

        const CookedHrirSet* fadeSet = src.fadeFromSet;
        const int oldAz = src.azIdx, oldEl = src.elIdx;
        const bool gridChanged = (azNearest != oldAz || elNearest != oldEl) && oldAz >= 0;
        const bool crossfade = (fadeSet != nullptr && oldAz >= 0) || gridChanged;

        // ---- ITD (bilinear over the 4 surrounding grid points) --------------
        float relL, relR;
        bilinearRelDelays (db, azDeg, elDeg, relL, relR);

        const float dist = juce::jmax (dir.distance, 0.05f);
        const float baseDelaySec = dist / kSpeedOfSound + extraDelayMs * 0.001f;
        src.left.delaySmoother.observe ((float) ((baseDelaySec + relL) * sampleRate), blockStart);
        src.right.delaySmoother.observe ((float) ((baseDelaySec + relR) * sampleRate), blockStart);

        const float distGain = (dist > 1.0f ? 1.0f / dist : 1.0f) * gainScale;
        const float airDb = dist * kAirShelfDbPerMeter;
        src.left.airShelf.setGainDb (airDb);
        src.right.airShelf.setGainDb (airDb);

        // ---- Forward FFT of [prev|curr], push into the FDL ------------------
        float* x = src.fdl.data() + (size_t) src.fdlPos * (size_t) spectrumFloats;
        std::fill (fftWork.begin(), fftWork.end(), 0.0f);
        std::copy (src.prevBlock.begin(), src.prevBlock.end(), fftWork.begin());
        std::copy (input, input + blockSize, fftWork.begin() + blockSize);
        fft->performRealOnlyForwardTransform (fftWork.data());
        std::copy (fftWork.begin(), fftWork.end(), x);
        std::copy (input, input + blockSize, src.prevBlock.begin());

        // ---- Per ear: MAC over partitions, IFFT, delay tap, shelf, gain -----
        for (int ear = 0; ear < 2; ++ear)
        {
            EarState& e = ear == 0 ? src.left : src.right;
            float* out = ear == 0 ? outL : outR;

            convolveTo (accumNew.data(), set, src, azNearest, elNearest, ear);
            fft->performRealOnlyInverseTransform (accumNew.data());

            const float* fresh = accumNew.data() + blockSize;   // last P valid

            if (crossfade)
            {
                const CookedHrirSet& oldSetRef = fadeSet != nullptr ? *fadeSet : set;
                convolveTo (accumOld.data(), oldSetRef, src, oldAz, oldEl, ear);
                fft->performRealOnlyInverseTransform (accumOld.data());
                const float* stale = accumOld.data() + blockSize;

                const float invN = 1.0f / (float) blockSize;
                for (int i = 0; i < blockSize; ++i)
                {
                    const float t = (float) i * invN;
                    e.delayLine[(size_t) e.writePos] = stale[i] + (fresh[i] - stale[i]) * t;
                    e.writePos = e.writePos + 1 < delayLineLength ? e.writePos + 1 : 0;
                }
            }
            else
            {
                for (int i = 0; i < blockSize; ++i)
                {
                    e.delayLine[(size_t) e.writePos] = fresh[i];
                    e.writePos = e.writePos + 1 < delayLineLength ? e.writePos + 1 : 0;
                }
            }

            // Read back at the smoothed delay, through shelf and gain.
            if (! e.gainInit) { e.prevGain = distGain; e.gainInit = true; }
            const float gainStart = e.prevGain, gainEnd = distGain;
            e.prevGain = distGain;

            const float maxDelay = (float) (delayLineLength - 2);
            const float invN = 1.0f / (float) blockSize;
            // writePos now points one past the block we just wrote; the
            // sample written for time (blockStart+i) sits at writePos-P+i.
            int basePos = e.writePos - blockSize;
            if (basePos < 0) basePos += delayLineLength;

            for (int i = 0; i < blockSize; ++i)
            {
                const auto d = e.delaySmoother.smoothedAt (blockStart + i);
                const float delaySamples = juce::jlimit (0.0f, maxDelay, d.delay);

                float readPos = (float) basePos + (float) i - delaySamples;
                if (readPos < 0.0f) readPos += (float) delayLineLength;

                int r1 = (int) readPos;
                if (r1 >= delayLineLength) r1 -= delayLineLength;
                int r2 = r1 + 1 < delayLineLength ? r1 + 1 : 0;
                const float frac = readPos - std::floor (readPos);

                float s = e.delayLine[(size_t) r1]
                        + frac * (e.delayLine[(size_t) r2] - e.delayLine[(size_t) r1]);
                s *= d.gain;
                s = e.airShelf.processSample (s);

                const float t = (float) i * invN;
                out[i] += s * (gainStart + (gainEnd - gainStart) * t);
            }
        }

        src.azIdx = azNearest;
        src.elIdx = elNearest;

        if (src.fadeFromSet != nullptr)
        {
            src.fadeFromSet = nullptr;
            if (--fadesPending <= 0)
                retireFadeGuard();
        }

        src.fdlPos = src.fdlPos + 1 < maxPartitions ? src.fdlPos + 1 : 0;
    }

private:
    static constexpr float kSpeedOfSound = 343.0f;
    static constexpr float kAirShelfDbPerMeter = -0.3f;
    static constexpr int kMaxHrirTaps = 4096;

    struct EarState
    {
        std::vector<float> delayLine;
        int writePos = 0;
        spatcore::dsp::DelayTargetSmoother delaySmoother;
        spatcore::dsp::WFSHighShelfFilter airShelf;
        float prevGain = 0.0f;
        bool gainInit = false;
    };

    struct SourceState
    {
        std::vector<float> prevBlock;           // previous input block (overlap-save)
        std::vector<float> fdl;                 // K × 2N frequency-domain history
        int fdlPos = 0;
        int azIdx = -1, elIdx = -1;             // current grid point (−1 = none yet)
        const CookedHrirSet* fadeFromSet = nullptr;   // non-null during a set swap
        EarState left, right;
    };

    /** accum = Σ_k X[k] · H[k] over the (clamped) partitions of `set`. */
    void convolveTo (float* accum, const CookedHrirSet& set, const SourceState& src,
                     int az, int el, int ear) const noexcept
    {
        std::fill (accum, accum + spectrumFloats, 0.0f);
        const int parts = juce::jmin (set.numPartitions, maxPartitions);

        for (int k = 0; k < parts; ++k)
        {
            // X for partition k is the spectrum pushed k blocks ago.
            int idx = src.fdlPos - k;
            if (idx < 0) idx += maxPartitions;
            const float* x = src.fdl.data() + (size_t) idx * (size_t) spectrumFloats;
            const float* h = set.spectrum (az, el, ear, k);

            for (int b = 0; b < spectrumFloats; b += 2)
            {
                const float xr = x[b], xi = x[b + 1];
                const float hr = h[b], hi = h[b + 1];
                accum[b]     += xr * hr - xi * hi;
                accum[b + 1] += xr * hi + xi * hr;
            }
        }
    }

    void bilinearRelDelays (const HrirDatabase& db, float azDeg, float elDeg,
                            float& relL, float& relR) const noexcept
    {
        const float azPos = azDeg / HrirDatabase::kAzStepDeg;
        const float elPos = (elDeg + 90.0f) / HrirDatabase::kElStepDeg;

        const int az0 = ((int) azPos) % HrirDatabase::kNumAz;
        const int az1 = (az0 + 1) % HrirDatabase::kNumAz;
        const int el0 = juce::jlimit (0, HrirDatabase::kNumEl - 1, (int) elPos);
        const int el1 = juce::jlimit (0, HrirDatabase::kNumEl - 1, el0 + 1);
        const float fa = azPos - std::floor (azPos);
        const float fe = juce::jlimit (0.0f, 1.0f, elPos - (float) el0);

        for (int ear = 0; ear < 2; ++ear)
        {
            const float d00 = db.relDelaySec[(size_t) db.delayIndex (az0, el0, ear)];
            const float d10 = db.relDelaySec[(size_t) db.delayIndex (az1, el0, ear)];
            const float d01 = db.relDelaySec[(size_t) db.delayIndex (az0, el1, ear)];
            const float d11 = db.relDelaySec[(size_t) db.delayIndex (az1, el1, ear)];
            const float d = (d00 * (1.0f - fa) + d10 * fa) * (1.0f - fe)
                          + (d01 * (1.0f - fa) + d11 * fa) * fe;
            (ear == 0 ? relL : relR) = d;
        }
    }

    void retireFadeGuard()
    {
        // Park the old set in the retired ring for the message thread to
        // release. If every slot is occupied or the lock is contended, keep
        // holding and try again later; nothing is freed on the worker.
        if (fadeGuard == nullptr)
            return;
        if (setLock.tryEnter())
        {
            for (int i = 0; i < kRetiredSlots; ++i)
            {
                if (retiredSets[i] == nullptr)
                {
                    retiredSets[i] = std::move (fadeGuard);
                    break;
                }
            }
            setLock.exit();
        }
    }

    double sampleRate = 48000.0;
    int blockSize = 0;
    int fftSize = 0;
    int spectrumFloats = 0;
    int maxPartitions = 1;
    int delayLineLength = 48000;

    std::unique_ptr<juce::dsp::FFT> fft;
    std::vector<SourceState> sources;

    // Worker-wide scratch (single render thread).
    std::vector<float> fftWork, accumNew, accumOld;

    // Set exchange: message thread publishes under setLock; worker adopts via
    // tryEnter at block boundaries; retired sets flow back for release.
    static constexpr int kRetiredSlots = 4;
    juce::SpinLock setLock;
    std::shared_ptr<const CookedHrirSet> pendingSet;               // guarded by setLock
    std::shared_ptr<const CookedHrirSet> retiredSets[kRetiredSlots]; // guarded by setLock
    std::atomic<uint32_t> pendingGeneration { 0 };
    uint32_t adoptedGeneration = 0;                       // worker-only

    std::shared_ptr<const CookedHrirSet> activeSet;       // worker-owned
    std::shared_ptr<const CookedHrirSet> fadeGuard;       // worker-owned
    int fadesPending = 0;
};

} // namespace spatcore::binaural
