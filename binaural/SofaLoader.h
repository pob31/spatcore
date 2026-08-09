#pragma once

#include "HrirSet.h"
#include <juce_core/juce_core.h>
#include <mysofa.h>
#include <vector>
#include <memory>
#include <cmath>

namespace spatcore::binaural
{

/**
    SOFA file → HrirDatabase, via the vendored libmysofa (ThirdParty/libmysofa
    + zlib inflate subset; consumers link the `spatcore-mysofa` target or
    compile the same C sources — see spatcore/CMakeLists.txt).

    Loader-thread only: allocates freely, touches the filesystem, and may take
    tens of milliseconds (mysofa_open resamples the whole set when the file
    rate differs from the device rate). Never call from the render worker.

    Bake steps per grid point (5° az × 10° el, the HrirDatabase layout):
      1. mysofa_getfilter_float — libmysofa's neighbor interpolation onto our
         direction, plus the set's Data.Delay per ear (seconds).
      2. ITD by interaural cross-correlation (±1.3 ms lag search) — robust
         against the contralateral ear's low-level creeping-wave pre-arrival,
         which fools plain threshold onset detection.
      3. Arrival decomposition: the louder (ipsilateral) ear's arrival is
         onset-detected (−14 dB re own peak, 8-sample guard); the other ear's
         arrival is that plus the xcorr ITD. Each IR is stripped by its own
         arrival, so the aligned pair mixes/crossfades without comb filtering
         and relL − relR reproduces the measured ITD exactly.
      4. The pair's common delay is discarded (propagation is re-applied at
         render time as r/c); only the differential ≥ 0 per ear survives as
         relDelaySec.
*/
namespace sofa
{

struct LoadResult
{
    std::shared_ptr<const HrirDatabase> database;   // null on failure
    juce::String status;                            // human-readable summary or error
};

/** Head-frame direction (az 0 = front, +right; el +up) → SOFA cartesian
    (x front, y left, z up), unit radius scaled by `radius`. */
inline void directionToSofaCartesian (float azRad, float elRad, float radius,
                                      float& x, float& y, float& z) noexcept
{
    const float cosEl = std::cos (elRad);
    x = radius * std::cos (azRad) * cosEl;    // front
    y = -radius * std::sin (azRad) * cosEl;   // SOFA +y = left = −(our right)
    z = radius * std::sin (elRad);            // up
}

inline LoadResult loadSofaFile (const juce::File& file, double targetSampleRate)
{
    LoadResult result;

    if (! file.existsAsFile())
    {
        result.status = "SOFA file not found: " + file.getFullPathName();
        return result;
    }

    int filterLength = 0;
    int err = 0;
    MYSOFA_EASY* easy = mysofa_open (file.getFullPathName().toRawUTF8(),
                                     (float) targetSampleRate, &filterLength, &err);
    if (easy == nullptr)
    {
        result.status = "Failed to load SOFA file (libmysofa error "
                      + juce::String (err) + "): " + file.getFileName();
        return result;
    }

    if (filterLength < 8 || filterLength > 8192)
    {
        result.status = "SOFA file has unusable filter length ("
                      + juce::String (filterLength) + "): " + file.getFileName();
        mysofa_close (easy);
        return result;
    }

    auto db = std::make_shared<HrirDatabase>();
    db->sampleRate = targetSampleRate;
    db->hrirLength = filterLength;
    db->hrirs.assign ((size_t) HrirDatabase::kNumAz * HrirDatabase::kNumEl * 2
                          * (size_t) filterLength, 0.0f);
    db->relDelaySec.assign ((size_t) HrirDatabase::kNumAz * HrirDatabase::kNumEl * 2, 0.0f);

    std::vector<float> irL ((size_t) filterLength), irR ((size_t) filterLength);

    auto peakAbs = [filterLength] (const float* ir)
    {
        float peak = 0.0f;
        for (int i = 0; i < filterLength; ++i)
            peak = std::max (peak, std::fabs (ir[i]));
        return peak;
    };

    // Onset: first sample above −14 dB re own peak, 8-sample guard ahead so
    // the attack shape survives. Reliable on the LOUD ear only — the shadowed
    // ear's creeping-wave pre-arrival sits above any usable threshold.
    auto detectOnset = [filterLength] (const float* ir, float peak)
    {
        if (peak <= 0.0f)
            return 0;
        const float threshold = peak * 0.2f;
        for (int i = 0; i < filterLength; ++i)
            if (std::fabs (ir[i]) >= threshold)
                return std::max (0, i - 8);
        return 0;
    };

    // ITD via interaural cross-correlation: lag of LEFT relative to RIGHT,
    // positive = left arrives later. ±1.3 ms search covers any human head.
    const int maxLag = juce::jmin (filterLength - 1,
                                   (int) std::ceil (0.0013 * targetSampleRate));
    auto xcorrLag = [filterLength, maxLag] (const float* l, const float* r)
    {
        int bestLag = 0;
        double bestVal = -1.0;
        for (int lag = -maxLag; lag <= maxLag; ++lag)
        {
            double acc = 0.0;
            for (int i = 0; i < filterLength; ++i)
            {
                const int j = i - lag;             // l[i] against r[i - lag]
                if (j >= 0 && j < filterLength)
                    acc += (double) l[i] * (double) r[j];
            }
            if (acc > bestVal)
            {
                bestVal = acc;
                bestLag = lag;
            }
        }
        return bestLag;
    };

    for (int az = 0; az < HrirDatabase::kNumAz; ++az)
    {
        for (int el = 0; el < HrirDatabase::kNumEl; ++el)
        {
            const float azRad = juce::degreesToRadians ((float) az * HrirDatabase::kAzStepDeg);
            const float elRad = juce::degreesToRadians (-90.0f + (float) el * HrirDatabase::kElStepDeg);

            float x, y, z;
            directionToSofaCartesian (azRad, elRad, 1.0f, x, y, z);

            float delayL = 0.0f, delayR = 0.0f;   // seconds (float easy API)
            mysofa_getfilter_float (easy, x, y, z, irL.data(), irR.data(), &delayL, &delayR);

            const float peakL = peakAbs (irL.data());
            const float peakR = peakAbs (irR.data());

            // ITD from xcorr; arrival of the loud ear from its onset; the
            // shadowed ear's arrival = loud arrival + ITD.
            const int lagLminusR = xcorrLag (irL.data(), irR.data());
            int stripL, stripR;
            if (peakL >= peakR)
            {
                stripL = detectOnset (irL.data(), peakL);
                stripR = stripL - lagLminusR;
            }
            else
            {
                stripR = detectOnset (irR.data(), peakR);
                stripL = stripR + lagLminusR;
            }
            stripL = juce::jlimit (0, filterLength - 1, stripL);
            stripR = juce::jlimit (0, filterLength - 1, stripR);

            const double totalL = (double) delayL + (double) stripL / targetSampleRate;
            const double totalR = (double) delayR + (double) stripR / targetSampleRate;
            const double common = std::min (totalL, totalR);

            db->relDelaySec[(size_t) db->delayIndex (az, el, 0)] = (float) (totalL - common);
            db->relDelaySec[(size_t) db->delayIndex (az, el, 1)] = (float) (totalR - common);

            float* dstL = db->hrirs.data() + db->hrirIndex (az, el, 0);
            float* dstR = db->hrirs.data() + db->hrirIndex (az, el, 1);
            for (int i = 0; i < filterLength; ++i)
            {
                dstL[i] = (stripL + i < filterLength) ? irL[(size_t) (stripL + i)] : 0.0f;
                dstR[i] = (stripR + i < filterLength) ? irR[(size_t) (stripR + i)] : 0.0f;
            }
        }
    }

    mysofa_close (easy);

    result.database = db;
    result.status = file.getFileName() + " — " + juce::String (filterLength) + " taps @ "
                  + juce::String (targetSampleRate / 1000.0, 1) + " kHz";
    return result;
}

} // namespace sofa
} // namespace spatcore::binaural
