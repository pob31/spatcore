#pragma once

#include <juce_core/juce_core.h>

#include <vector>

namespace spatcore
{
namespace io
{

//==============================================================================
/**
    Maps hardware channel numbers onto the COMPACTED indices a device callback
    actually uses.

    JUCE hands audioDeviceIOCallbackWithContext two arrays that contain only the
    ENABLED channels, packed with no holes: entry k is the k-th set bit of the
    device's active-channel mask, not hardware channel k. ASIO builds its buffer
    list the same way, and juce::AudioSourcePlayer compacts a second time by
    skipping null pointers. The two index spaces coincide only while the mask is
    contiguous from bit 0 — which is the normal case, and exactly the case that
    silently stops holding the moment a mask has a hole in it.

    Every consumer of a patch matrix indexes by HARDWARE channel (that is what
    the operator patched, and what the interface's connectors are labelled), so
    this type exists to translate once, up front, in audioDeviceAboutToStart:

        map.inputIndexForHw[h]   index into inputChannelData,  or -1 if hw input h is disabled
        map.outputIndexForHw[h]  index into outputChannelData, or -1 if hw output h is disabled

    It is deliberately free of juce_audio_devices so it can be unit-tested
    headlessly from a pair of hand-built BigIntegers.
*/
struct HardwareIndexMap
{
    /** Number of hardware channels addressed: one past the highest active bit
        of either mask, clamped to maxChannels. Zero when nothing is enabled. */
    int numChannels = 0;

    /** Size numChannels. Compact index into the callback's input array, or -1. */
    std::vector<int> inputIndexForHw;

    /** Size numChannels. Compact index into the callback's output array, or -1. */
    std::vector<int> outputIndexForHw;

    /** Builds the map from a device's active-channel masks.

        maxChannels bounds the addressable range: hardware channels at or above
        it are simply not represented (their compact indices still exist in the
        driver's arrays, we just never touch them). It is a policy limit, not a
        buffer size — nothing here is a fixed-size array.
    */
    static HardwareIndexMap fromMasks (const juce::BigInteger& activeInputs,
                                       const juce::BigInteger& activeOutputs,
                                       int maxChannels)
    {
        HardwareIndexMap map;

        if (maxChannels <= 0)
            return map;

        const int highestBit = juce::jmax (activeInputs.getHighestBit(),
                                           activeOutputs.getHighestBit());

        if (highestBit < 0)     // both masks empty: no device channels at all
            return map;

        map.numChannels = juce::jmin (maxChannels, highestBit + 1);
        map.inputIndexForHw.assign  ((size_t) map.numChannels, -1);
        map.outputIndexForHw.assign ((size_t) map.numChannels, -1);

        int nextInput = 0, nextOutput = 0;

        for (int hw = 0; hw < map.numChannels; ++hw)
        {
            if (activeInputs[hw])
                map.inputIndexForHw[(size_t) hw] = nextInput++;

            if (activeOutputs[hw])
                map.outputIndexForHw[(size_t) hw] = nextOutput++;
        }

        return map;
    }

    /** True when hardware index == compact index for every addressed channel,
        i.e. both masks run contiguously from bit 0 (or are empty). Diagnostics
        and tests only — the callback does not need to special-case it. */
    bool isIdentityMapping() const noexcept
    {
        for (int hw = 0; hw < numChannels; ++hw)
        {
            const int in  = inputIndexForHw[(size_t) hw];
            const int out = outputIndexForHw[(size_t) hw];

            if ((in >= 0 && in != hw) || (out >= 0 && out != hw))
                return false;
        }

        return true;
    }
};

} // namespace io
} // namespace spatcore
