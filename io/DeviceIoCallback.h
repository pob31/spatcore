#pragma once

#include <juce_audio_devices/juce_audio_devices.h>

#include "HardwareIndexMap.h"

#include <vector>

namespace spatcore
{
namespace io
{

//==============================================================================
/**
    Drives a juce::AudioSource straight from a device callback, with no channel
    cap and a HARDWARE-INDEXED buffer.

    Why this exists rather than juce::AudioAppComponent / AudioSourcePlayer:
    AudioSourcePlayer holds `float* channels[128]` (plus two more 128-entry
    arrays) and breaks out of its compaction loops once they are full, so the
    buffer handed to the source can never exceed 128 channels no matter how many
    the interface has. On a large rig every hardware channel from 128 up is
    silently unreachable — no metering, no test tone, and no actual audio.

    Two guarantees this class makes, both relied on by patch-matrix style
    routing:

      1. NO CAP. maxChannelsToAddress is a policy limit passed by the consumer
         (512 for the WFS apps), not an array bound. Nothing here is fixed-size.

      2. buffer channel h == HARDWARE channel h, inputs and outputs alike.
         The device's own arrays are compacted (entry k is the k-th enabled
         channel), so code that indexes them by hardware number is wrong the
         moment a channel mask has a hole. Translating once through
         HardwareIndexMap makes "buffer channel == the number on the
         interface's connector" true by construction.

    Buffer assembly mirrors AudioSourcePlayer exactly, generalised from compact
    to hardware indices, so for a contiguous mask starting at 0 the source sees
    a byte-identical buffer:

      - a channel that is an active output aliases the device's own output
        storage: zero-copy, the source writes in place;
      - any other addressed channel (input with no matching output, or a hole)
        gets its own preallocated scratch row: pre-filled with the input where
        there is one, cleared otherwise, and its writes are discarded;
      - every addressed channel is pre-filled before the source runs — input
        data where the input is active, silence everywhere else.

    AudioSourcePlayer's gain stage is deliberately not reproduced: it is a
    no-op at unity, which is the only gain AudioAppComponent ever set.

    Real-time contract: everything is sized in audioDeviceAboutToStart. The
    callback allocates nothing, takes no lock and never blocks. Register with
    juce::AudioDeviceManager::addAudioCallback once — the manager re-arms every
    registered callback across device changes — and remove it before this object
    or its source is destroyed (removeAudioCallback blocks until the audio
    thread has left the callback).
*/
class DeviceIoCallback : public juce::AudioIODeviceCallback
{
public:
    /** @param sourceToDrive  gets prepareToPlay / getNextAudioBlock /
                              releaseResources, exactly as from AudioSourcePlayer.
                              Must outlive this object being registered.
        @param maxChannelsToAddress  highest hardware channel count to address. */
    explicit DeviceIoCallback (juce::AudioSource& sourceToDrive,
                               int maxChannelsToAddress = 512)
        : source (sourceToDrive),
          maxChannels (juce::jmax (1, maxChannelsToAddress))
    {
    }

    //==========================================================================
    void audioDeviceAboutToStart (juce::AudioIODevice* device) override
    {
        if (device == nullptr)
            return;

        const int blockSize = juce::jmax (1, device->getCurrentBufferSizeSamples());
        const double sampleRate = device->getCurrentSampleRate();

        map = HardwareIndexMap::fromMasks (device->getActiveInputChannels(),
                                           device->getActiveOutputChannels(),
                                           maxChannels);

        // Scratch rows for the channels that do not alias device output storage,
        // plus one shared fallback row used only when a driver contradicts its
        // own active-output mask at runtime (null or short output array). That
        // row can cross-talk between such channels; they are broken either way,
        // and it keeps the normal case down to a single spare row.
        scratchRowForHw.assign ((size_t) map.numChannels, -1);
        numAddressedOutputs = 0;
        int rows = 0;

        for (int hw = 0; hw < map.numChannels; ++hw)
        {
            if (map.outputIndexForHw[(size_t) hw] >= 0)
                ++numAddressedOutputs;
            else
                scratchRowForHw[(size_t) hw] = rows++;
        }

        fallbackRow = rows++;

        scratch.setSize (rows, blockSize, false, true, true);
        scratch.clear();

        channelPtrs.assign ((size_t) map.numChannels, nullptr);
        preparedBlockSize = blockSize;
        prepared = true;

        source.prepareToPlay (blockSize, sampleRate);
    }

    void audioDeviceStopped() override
    {
        prepared = false;
        source.releaseResources();
    }

    void audioDeviceIOCallbackWithContext (const float* const* inputChannelData,
                                           int numInputChannels,
                                           float* const* outputChannelData,
                                           int numOutputChannels,
                                           int numSamples,
                                           const juce::AudioIODeviceCallbackContext&) override
    {
        if (numSamples <= 0)
            return;

        if (! prepared || map.numChannels == 0)
        {
            clearOutputs (outputChannelData, 0, numOutputChannels, 0, numSamples);
            return;
        }

        // A device handing us more samples than it declared would overrun the
        // scratch rows; clamp and silence the remainder rather than trusting it.
        const int n = juce::jmin (numSamples, preparedBlockSize);

        for (int hw = 0; hw < map.numChannels; ++hw)
        {
            const int outIndex = map.outputIndexForHw[(size_t) hw];

            float* storage = nullptr;

            if (outIndex >= 0 && outIndex < numOutputChannels
                 && outputChannelData[outIndex] != nullptr)
            {
                storage = outputChannelData[outIndex];       // zero-copy alias
            }
            else
            {
                const int row = scratchRowForHw[(size_t) hw];
                storage = scratch.getWritePointer (row >= 0 ? row : fallbackRow);
            }

            channelPtrs[(size_t) hw] = storage;

            const int inIndex = map.inputIndexForHw[(size_t) hw];

            if (inIndex >= 0 && inIndex < numInputChannels
                 && inputChannelData[inIndex] != nullptr)
                juce::FloatVectorOperations::copy (storage, inputChannelData[inIndex], n);
            else
                juce::FloatVectorOperations::clear (storage, n);
        }

        juce::AudioBuffer<float> buffer (channelPtrs.data(), map.numChannels, n);
        juce::AudioSourceChannelInfo info (&buffer, 0, n);
        source.getNextAudioBlock (info);

        // Outputs past the addressed range were never written. Compact output
        // indices are assigned in increasing hardware order, so those are
        // exactly the indices at or above numAddressedOutputs (they exist only
        // when the device has more channels than maxChannels allows us to
        // address).
        clearOutputs (outputChannelData, numAddressedOutputs, numOutputChannels, 0, n);

        if (n < numSamples)
            clearOutputs (outputChannelData, 0, numOutputChannels, n, numSamples - n);
    }

    //==========================================================================
    /** The hardware -> compact index translation currently in force. */
    const HardwareIndexMap& getChannelMap() const noexcept   { return map; }

    /** Hardware channels the callback buffer spans (0 when no device is open). */
    int getNumAddressedChannels() const noexcept             { return map.numChannels; }

private:
    static void clearOutputs (float* const* outputChannelData,
                              int firstChannel, int endChannel,
                              int startSample, int numSamples)
    {
        if (outputChannelData == nullptr || numSamples <= 0)
            return;

        for (int i = juce::jmax (0, firstChannel); i < endChannel; ++i)
            if (outputChannelData[i] != nullptr)
                juce::FloatVectorOperations::clear (outputChannelData[i] + startSample, numSamples);
    }

    juce::AudioSource& source;
    const int maxChannels;

    // Written in audioDeviceAboutToStart, read in the callback. JUCE guarantees
    // the two never overlap (the device is stopped across a prepare).
    HardwareIndexMap map;
    std::vector<int> scratchRowForHw;   // hw -> scratch row, or -1 when it aliases device output
    std::vector<float*> channelPtrs;    // rebuilt per block; no allocation (capacity is fixed here)
    juce::AudioBuffer<float> scratch;
    int fallbackRow = 0;
    int numAddressedOutputs = 0;
    int preparedBlockSize = 0;
    bool prepared = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DeviceIoCallback)
};

} // namespace io
} // namespace spatcore
