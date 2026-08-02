#pragma once

#include <juce_audio_devices/juce_audio_devices.h>

namespace spatcore
{
namespace io
{

//==============================================================================
/**
    Device open / restore policy over a juce::AudioDeviceManager the consumer
    owns (this class only holds a reference, so an app can keep its manager
    wherever it already lives).

    The policy it enforces: a spatial rig wants EVERY channel of the interface
    open, always, and it wants the channel masks it asked for to be the ones
    that get used.

    That second half is not automatic. juce::AudioDeviceManager keeps a
    `useDefaultInputChannels` / `useDefaultOutputChannels` pair that defaults to
    true, and while either is set, setAudioDeviceSetup() throws the caller's
    mask away and substitutes `range(0, numInputChansNeeded)` — a count frozen
    at whatever the last initialise() asked for. Code that builds a full mask
    and calls setAudioDeviceSetup() without clearing those flags therefore has
    no effect at all: the device reopens with the previous, usually smaller,
    channel count, and the app is left metering and patching channels the
    driver never opened. Every mutation here writes explicit masks with both
    flags cleared, which is what makes "enable all channels" actually stick.

    maxChannels bounds how many channels are ever requested (512 for the WFS
    apps — RME AoX-D class interfaces); it replaces the magic 256 that used to
    be scattered through the call sites.

    All methods are message-thread only and return JUCE's error string ("" on
    success). The class holds no state of its own beyond the reference and the
    limit, so consumers can construct one wherever they need it.
*/
class DeviceHost
{
public:
    DeviceHost (juce::AudioDeviceManager& managerToUse, int maxChannelsToOpen = 512)
        : deviceManager (managerToUse),
          maxChannels (juce::jmax (1, maxChannelsToOpen))
    {
    }

    //==========================================================================
    /** Restores a saved DEVICESETUP state, then opens every channel of whatever
        device came back. Pass selectDefaultOnFailure=false to avoid a full
        device rescan when the saved device is gone. */
    juce::String restoreFromXml (const juce::XmlElement* savedState,
                                 bool selectDefaultOnFailure)
    {
        auto error = deviceManager.initialise (maxChannels, maxChannels,
                                               savedState, selectDefaultOnFailure);

        if (error.isNotEmpty())
            return error;

        return enableAllChannels();
    }

    /** Opens a device by type + name with all its channels enabled. Used when
        no saved state is usable, so it accepts the device's own default sample
        rate and buffer size. */
    juce::String openNamedDevice (const juce::String& deviceTypeName,
                                  const juce::String& deviceName)
    {
        deviceManager.setCurrentAudioDeviceType (deviceTypeName, true);

        return setDeviceAllChannels (deviceName);
    }

    /** Selects a device within the current device type, all channels enabled,
        at the device's default sample rate and buffer size. */
    juce::String setDeviceAllChannels (const juce::String& deviceName)
    {
        auto setup = deviceManager.getAudioDeviceSetup();

        setup.inputDeviceName  = deviceName;
        setup.outputDeviceName = deviceName;

        // Let the device pick: its defaults are valid, the previous device's
        // rate and block size may not be.
        setup.sampleRate = 0;
        setup.bufferSize = 0;

        // Provisional full-width masks: the real channel counts are only
        // knowable once the device object exists, so enableAllChannels() below
        // trims them to the truth.
        applyEnableAllPolicy (setup, maxChannels, maxChannels, maxChannels);

        auto error = deviceManager.setAudioDeviceSetup (setup, true);

        if (error.isNotEmpty())
            return error;

        return enableAllChannels();
    }

    /** Enables every channel the CURRENT device has, as an explicit mask.
        No-ops when the setup already says exactly that, so calling it after a
        restore does not cost a second device open on every launch. */
    juce::String enableAllChannels()
    {
        auto* device = deviceManager.getCurrentAudioDevice();

        if (device == nullptr)
            return {};

        const auto current = deviceManager.getAudioDeviceSetup();
        auto wanted = current;

        applyEnableAllPolicy (wanted,
                              device->getInputChannelNames().size(),
                              device->getOutputChannelNames().size(),
                              maxChannels);

        // Compare the requested setup, not the device's active masks: a device
        // that declines to open everything we ask for would otherwise be
        // reopened on every call, forever.
        if (! current.useDefaultInputChannels
             && ! current.useDefaultOutputChannels
             && current.inputChannels  == wanted.inputChannels
             && current.outputChannels == wanted.outputChannels)
            return {};

        return deviceManager.setAudioDeviceSetup (wanted, true);
    }

    //==========================================================================
    // Truthful facts about what is OPEN. The UI must gate on these, not on the
    // device's channel-name counts: a channel the device lists but never opened
    // has no slot in the callback buffer, so metering it or sending a test tone
    // to it is silently impossible, and showing it as available is a lie.

    int getNumActiveInputs() const
    {
        auto* device = deviceManager.getCurrentAudioDevice();
        return device != nullptr ? device->getActiveInputChannels().countNumberOfSetBits() : 0;
    }

    int getNumActiveOutputs() const
    {
        auto* device = deviceManager.getCurrentAudioDevice();
        return device != nullptr ? device->getActiveOutputChannels().countNumberOfSetBits() : 0;
    }

    juce::BigInteger getActiveInputMask() const
    {
        auto* device = deviceManager.getCurrentAudioDevice();
        return device != nullptr ? device->getActiveInputChannels() : juce::BigInteger();
    }

    juce::BigInteger getActiveOutputMask() const
    {
        auto* device = deviceManager.getCurrentAudioDevice();
        return device != nullptr ? device->getActiveOutputChannels() : juce::BigInteger();
    }

    /** Channels the device EXPOSES (>= the active count). Diagnostics only. */
    int getNumDeviceInputs() const
    {
        auto* device = deviceManager.getCurrentAudioDevice();
        return device != nullptr ? device->getInputChannelNames().size() : 0;
    }

    int getNumDeviceOutputs() const
    {
        auto* device = deviceManager.getCurrentAudioDevice();
        return device != nullptr ? device->getOutputChannelNames().size() : 0;
    }

    int getMaxChannels() const noexcept   { return maxChannels; }

    //==========================================================================
    /** The policy itself, pulled out so it can be unit-tested without a device:
        explicit contiguous masks for the channel counts given, clamped to
        maxChannels, with both useDefault flags cleared so JUCE cannot silently
        replace them. */
    static void applyEnableAllPolicy (juce::AudioDeviceManager::AudioDeviceSetup& setup,
                                      int numDeviceInputs,
                                      int numDeviceOutputs,
                                      int maxChannels)
    {
        const int ins  = juce::jlimit (0, maxChannels, numDeviceInputs);
        const int outs = juce::jlimit (0, maxChannels, numDeviceOutputs);

        setup.inputChannels.clear();
        if (ins > 0)
            setup.inputChannels.setRange (0, ins, true);

        setup.outputChannels.clear();
        if (outs > 0)
            setup.outputChannels.setRange (0, outs, true);

        setup.useDefaultInputChannels  = false;
        setup.useDefaultOutputChannels = false;
    }

private:
    juce::AudioDeviceManager& deviceManager;
    const int maxChannels;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DeviceHost)
};

} // namespace io
} // namespace spatcore
