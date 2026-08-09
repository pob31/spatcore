#pragma once

#include <juce_dsp/juce_dsp.h>
#include <vector>
#include <memory>
#include <cmath>

namespace spatcore::binaural
{

/**
    Measured-HRTF data for the SOFA render mode.

    Two stages, both built OFF the render thread:

    1. HrirDatabase — the load-time bake (SofaLoader.h): the measured set is
       resampled to the device rate and queried through libmysofa's
       interpolation onto a UNIFORM grid (5° azimuth × 10° elevation), each
       point stored as a pair of time-ALIGNED HRIRs plus per-ear relative
       onset delays. Aligning the HRIRs and carrying the ITD separately is
       what makes render-time crossfading between neighboring directions safe
       (raw HRIRs with differing onsets comb-filter when mixed); the ITD is
       re-applied through the same smoothed fractional delay lines the
       structural model uses.

    2. CookedHrirSet — the FFT bake (cookHrirSet below): every grid HRIR
       pre-transformed into partition spectra for a given block size, so a
       render-time direction change is a pointer swap, never an FFT of the IR.
       Re-cooked (cheap, off-thread) when the device block size changes.

    Grid convention matches BinauralTypes.h: azimuth 0 = front, positive =
    listener's right, wrapping 0..360; elevation −90..+90.
*/

struct HrirDatabase
{
    static constexpr int kNumAz = 72;    // 5° steps, wraps
    static constexpr int kNumEl = 19;    // 10° steps, −90..+90
    static constexpr float kAzStepDeg = 360.0f / (float) kNumAz;
    static constexpr float kElStepDeg = 180.0f / (float) (kNumEl - 1);

    double sampleRate = 48000.0;
    int hrirLength = 256;                // taps after alignment (zero-padded tail)

    /** Time-aligned HRIRs, [az][el][ear][tap]. */
    std::vector<float> hrirs;

    /** Per-ear onset delay relative to the earlier ear, seconds ≥ 0:
        [az][el][ear]. The earlier ear holds 0; the other holds the ITD. */
    std::vector<float> relDelaySec;

    int hrirIndex (int az, int el, int ear) const noexcept
    {
        return ((az * kNumEl + el) * 2 + ear) * hrirLength;
    }

    int delayIndex (int az, int el, int ear) const noexcept
    {
        return (az * kNumEl + el) * 2 + ear;
    }
};

struct CookedHrirSet
{
    std::shared_ptr<const HrirDatabase> db;   // keeps delays + provenance alive

    int blockSize = 0;                        // P — partition size
    int fftSize = 0;                          // N = 2P
    int numPartitions = 0;                    // K = ceil(hrirLength / P)
    int spectrumFloats = 0;                   // 2N floats per partition spectrum

    /** Partition spectra, [az][el][ear][partition][2N interleaved floats]
        in juce::dsp::FFT realOnlyForwardTransform layout. */
    std::vector<float> spectra;

    const float* spectrum (int az, int el, int ear, int partition) const noexcept
    {
        return spectra.data()
             + ((size_t) ((az * HrirDatabase::kNumEl + el) * 2 + ear) * (size_t) numPartitions
                + (size_t) partition) * (size_t) spectrumFloats;
    }
};

/** FFT-bake a database for one block size. Off-thread only (allocates). */
inline std::shared_ptr<const CookedHrirSet> cookHrirSet (std::shared_ptr<const HrirDatabase> db,
                                                         int blockSize)
{
    if (db == nullptr || blockSize < 4)
        return nullptr;

    auto cooked = std::make_shared<CookedHrirSet>();
    cooked->db = db;
    cooked->blockSize = blockSize;

    int order = 1;
    while ((1 << order) < 2 * blockSize)
        ++order;
    cooked->fftSize = 1 << order;                       // N = 2P (P a power of 2 in JUCE land)
    cooked->numPartitions = (db->hrirLength + blockSize - 1) / blockSize;
    cooked->spectrumFloats = 2 * cooked->fftSize;

    juce::dsp::FFT fft (order);
    std::vector<float> work ((size_t) cooked->spectrumFloats, 0.0f);

    const size_t total = (size_t) HrirDatabase::kNumAz * HrirDatabase::kNumEl * 2
                       * (size_t) cooked->numPartitions * (size_t) cooked->spectrumFloats;
    cooked->spectra.assign (total, 0.0f);

    auto* mutableCooked = cooked.get();
    for (int az = 0; az < HrirDatabase::kNumAz; ++az)
        for (int el = 0; el < HrirDatabase::kNumEl; ++el)
            for (int ear = 0; ear < 2; ++ear)
            {
                const float* h = db->hrirs.data() + db->hrirIndex (az, el, ear);
                for (int k = 0; k < cooked->numPartitions; ++k)
                {
                    std::fill (work.begin(), work.end(), 0.0f);
                    const int start = k * blockSize;
                    const int count = juce::jmin (blockSize, db->hrirLength - start);
                    for (int i = 0; i < count; ++i)
                        work[(size_t) i] = h[start + i];

                    fft.performRealOnlyForwardTransform (work.data());

                    float* destPtr = mutableCooked->spectra.data()
                        + ((size_t) ((az * HrirDatabase::kNumEl + el) * 2 + ear)
                               * (size_t) cooked->numPartitions + (size_t) k)
                          * (size_t) cooked->spectrumFloats;
                    std::copy (work.begin(), work.end(), destPtr);
                }
            }

    return cooked;
}

} // namespace spatcore::binaural
