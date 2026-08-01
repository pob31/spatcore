#pragma once

#include "MultiChannelEQBank.h"

#include <juce_audio_basics/juce_audio_basics.h>
#include <algorithm>
#include <array>
#include <vector>

namespace spatcore::dsp {

//==============================================================================
/**
    Per-output 6-band parametric EQ processor.

    Each output channel has its own bank of 6 biquads in series. Filter state
    is independent per output; coefficient values come from per-channel
    parameter entries (typically kept in sync across the array by the GUI's
    array-propagation, but treated as independent here).

    Threading model mirrors ReverbPostProcessor:
      - prepare() / setParameters() are called from a non-audio thread (timer
        callback or prepareToPlay).
      - processBlock() runs on the audio thread.
      - The biquad's setParameters short-circuits on no-change, so pushing
        params unconditionally every tick is cheap when nothing moved.

    The per-channel filter storage and the enable flags live in the shared
    MultiChannelEQBank; this class is the WFS-facing parameter struct + JUCE
    buffer wrapper around it. The bank's push ordering contract (enable flag
    first, then the bands) is honoured in setParameters().
*/
class OutputEQProcessor
{
public:
    static constexpr int NUM_EQ_BANDS = 6;

    //==========================================================================
    struct EQBandParams
    {
        int   shape = 0;
        float freq  = 1000.0f;
        float gain  = 0.0f;
        float q     = 0.7f;
        float slope = 0.7f;
    };

    struct OutputChannelParams
    {
        bool enabled = false;
        std::array<EQBandParams, NUM_EQ_BANDS> bands {};
    };

    struct Params
    {
        std::vector<OutputChannelParams> channels;
    };

    //==========================================================================
    void prepare (double newSampleRate, int /*maxBlockSize*/, int numOutputs)
    {
        sampleRate = newSampleRate;
        numActiveChannels = std::max (0, numOutputs);

        // Resizes + prepares every filter and resets all enable flags to false.
        bank.prepare (sampleRate, numActiveChannels);
    }

    void reset()
    {
        bank.reset();
    }

    //==========================================================================
    void setParameters (const Params& newParams)
    {
        int n = std::min (static_cast<int> (newParams.channels.size()), numActiveChannels);

        for (int c = 0; c < n; ++c)
        {
            const auto& cp = newParams.channels[static_cast<size_t> (c)];

            // Enable flag FIRST: pushBandParameters forces shape 0 on a
            // disabled channel.
            bank.setChannelEnabled (c, cp.enabled);

            for (int b = 0; b < NUM_EQ_BANDS; ++b)
            {
                const auto& bp = cp.bands[static_cast<size_t> (b)];
                bank.pushBandParameters (c, b, bp.shape, bp.freq, bp.gain, bp.q, bp.slope);
            }
        }
    }

    //==========================================================================
    /** Process EQ on the WFS output buffer slice. */
    void processBlock (juce::AudioBuffer<float>& buffer, int startSample, int numSamples)
    {
        const int numCh = std::min (numActiveChannels, buffer.getNumChannels());

        for (int c = 0; c < numCh; ++c)
        {
            if (! bank.isChannelEnabled (c))
                continue;

            float* data = buffer.getWritePointer (c) + startSample;
            bank.processChannel (c, data, numSamples);
        }
    }

private:
    double sampleRate = 48000.0;
    int    numActiveChannels = 0;

    MultiChannelEQBank<NUM_EQ_BANDS> bank;
};

} // namespace spatcore::dsp

// Extraction-compat aliases — app code migrates to qualified names later.
using spatcore::dsp::OutputEQProcessor;
