#pragma once

#include "BinauralTypes.h"
#include "HeadFrame.h"
#include "StructuralHrtfRenderer.h"
#include "SofaHrtfRenderer.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <cmath>
#include <cstdint>

namespace spatcore::binaural
{

/**
    App-agnostic binaural rendering core for the Structural and Sofa modes.
    (The OrtfLegacy mode is deliberately NOT here — it stays in the app's
    BinauralProcessor, byte-for-byte, so existing projects null against it.)

    Ownership/threading model mirrors the app's binaural worker:
      - prepare()/reset() are called with the worker thread stopped.
      - processBlock() is called from one realtime worker thread, once per
        block. It never allocates and never touches app state; everything it
        needs arrives as arguments or was pre-cooked in prepare().
      - setMode()/setHeadRadius() are atomic knobs, callable from the message
        thread while the worker runs.

    "Sources" are anything with a position and a mono feed: the app passes
    inputs first, then (in preview mode) reverb-node returns. A null entry in
    `inputs` skips that source (solo gating, muted reverb) at zero cost.

    Until the SOFA renderer lands, RenderMode::Sofa falls back to the
    structural model so selecting it never produces dead air.
*/
class BinauralEngine
{
public:
    BinauralEngine() = default;

    /** Worker stopped. maxSources = inputs + reverb nodes the caller may pass. */
    void prepare (double newSampleRate, int newMaxBlockSize, int newMaxSources)
    {
        sampleRate   = newSampleRate;
        maxBlockSize = newMaxBlockSize;
        maxSources   = newMaxSources;

        structural.prepare (newSampleRate, newMaxBlockSize, newMaxSources);
        sofa.prepare (newSampleRate, newMaxBlockSize, newMaxSources);
        samplePos = 0;
        prepared = true;
    }

    /** Worker stopped. Clears all per-source DSP state. */
    void reset()
    {
        structural.reset();
        sofa.reset();
        samplePos = 0;
    }

    /** Message thread: hand a freshly cooked SOFA set to the render worker
        (nullptr clears; Sofa mode then falls back to the structural model). */
    void publishSofaSet (std::shared_ptr<const CookedHrirSet> cooked)
    {
        sofa.publishSet (std::move (cooked));
    }

    /** Message thread, periodically: release HRIR sets the worker retired. */
    void collectRetiredSofaSets()
    {
        sofa.collectRetired();
    }

    void setMode (RenderMode m) noexcept
    {
        mode.store (static_cast<int> (m), std::memory_order_release);
    }

    RenderMode getMode() const noexcept
    {
        return static_cast<RenderMode> (mode.load (std::memory_order_acquire));
    }

    void setHeadRadius (float meters) noexcept
    {
        headRadius.store (meters, std::memory_order_release);
    }

    /**
        Render one block. RT-safe: no allocation, no locks.

        pose          listener position (damped path) + head rotation (fast path)
        srcPositions  [numSources][3] world-frame positions, damped upstream
        inputs        numSources mono pointers; nullptr = skip this source
        perSourceGain optional [numSources] linear gains (nullptr = unity) —
                      used by the app for the reverb-balance trim
        extraDelayMs  global output delay offset (binauralDelay parameter)
        outL/outR     accumulators, ADDED to (caller clears)
    */
    void processBlock (const ListenerPose& pose,
                       const float (*srcPositions)[3],
                       const float* const* inputs,
                       const float* perSourceGain,
                       float extraDelayMs,
                       float* outL, float* outR,
                       int numSources, int numSamples)
    {
        juce::ScopedNoDenormals noDenormals;

        if (! prepared || numSamples <= 0)
            return;

        const float radius = headRadius.load (std::memory_order_acquire);
        numSources = juce::jmin (numSources, maxSources);

        // A library does not get to assume its caller validated. A non-finite
        // pose (tracker glitch, corrupt project file) would put NaN into the
        // ITD delay lines — where the read index is derived from the delay,
        // so a float→int conversion of NaN is undefined — and into filter and
        // convolution state, which never recovers on its own. Substituting the
        // identity pose keeps the block rendering instead of clicking to
        // silence; hosts layer their own, better-informed fallback on top.
        const ListenerPose* posePtr = &pose;
        ListenerPose fallback;
        if (! isFinitePose (pose))
            posePtr = &fallback;
        const ListenerPose& safePose = *posePtr;

        // Sofa mode renders measured HRIRs once a cooked set has been adopted;
        // it falls back to the structural model while none is loaded, so the
        // mode never produces dead air.
        bool useSofa = getMode() == RenderMode::Sofa;
        if (useSofa)
        {
            sofa.processBlockBegin();
            useSofa = sofa.hasActiveSet();
        }

        for (int i = 0; i < numSources; ++i)
        {
            if (inputs[i] == nullptr)
                continue;

            // Same for a single bad source position: mute that source rather
            // than let it poison the shared output and its own filter state.
            if (! (std::isfinite (srcPositions[i][0]) && std::isfinite (srcPositions[i][1])
                   && std::isfinite (srcPositions[i][2])))
                continue;

            const auto dir = headframe::directionInHeadFrame (safePose,
                                                              srcPositions[i][0],
                                                              srcPositions[i][1],
                                                              srcPositions[i][2]);
            const float gainScale = perSourceGain != nullptr ? perSourceGain[i] : 1.0f;

            if (useSofa)
                sofa.processSource (i, dir, extraDelayMs, gainScale,
                                    inputs[i], outL, outR, numSamples, samplePos);
            else
                structural.processSource (i, dir, radius, extraDelayMs, gainScale,
                                          inputs[i], outL, outR, numSamples, samplePos);
        }

        samplePos += numSamples;
    }

private:
    double sampleRate   = 48000.0;
    int    maxBlockSize = 512;
    int    maxSources   = 0;
    bool   prepared     = false;

    std::atomic<int>   mode       { static_cast<int> (RenderMode::Structural) };
    std::atomic<float> headRadius { 0.0875f };

    StructuralHrtfRenderer structural;
    SofaHrtfRenderer sofa;
    std::int64_t samplePos = 0;
};

} // namespace spatcore::binaural
