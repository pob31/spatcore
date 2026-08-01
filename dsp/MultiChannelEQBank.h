#pragma once

#include <algorithm>   // std::max in prepare()
#include <array>
#include <cstdint>
#include <vector>

#include "OutputEQBiquadFilter.h"

namespace spatcore::dsp {

//==============================================================================
/**
    Multi-channel bank of NumBands biquads in series, one independent chain per
    channel.

    This is the shared core of the "per-channel bank of N biquads" pattern that
    both WFS-DIY (per-output EQ) and XOA (per-speaker compensation) hand-rolled
    identically: a std::vector of std::array<OutputEQBiquadFilter, NumBands>
    plus a parallel std::vector<std::uint8_t> of per-channel enable flags
    (std::uint8_t avoids the std::vector<bool> proxy). Filter state is private
    per channel; coefficients are per channel too, even when the host GUI keeps
    them in sync across the array.

    Threading model (the production-proven "benign staleness" pattern):
      - prepare() allocates and must be called from a non-audio thread
        (prepareToPlay / device restart).
      - setChannelEnabled() / pushBandParameters() are called from the message
        thread, typically unconditionally from a timer tick. They are cheap
        because OutputEQBiquadFilter::setParameters short-circuits on
        no-change. No lock is taken: at worst the audio thread observes a torn
        coefficient set for one tick, producing a transient, never a fault.
      - processChannel() runs on the audio thread. It is noexcept and
        allocation-free.

    PARAMETER PUSH ORDERING — IMPORTANT
    pushBandParameters() forces shape 0 (OFF, pass-through) on every band while
    the channel is currently disabled, mirroring both apps. Callers must
    therefore push the enable flag FIRST and the bands SECOND:

        bank.setChannelEnabled (c, channelParams.enabled);       // 1st
        for (int b = 0; b < NumBands; ++b)                       // 2nd
            bank.pushBandParameters (c, b, shape, freq, gainDb, q, slope);

    Pushing in the other order would latch the previous enable state into the
    coefficients for one parameter tick.

    Every index accessor bounds-checks and silently no-ops when out of range,
    so a stale channel count from the message thread can never fault the audio
    thread.

    Band shape IDs are the OutputEQBiquadFilter mapping (0 = OFF, 1 = LowCut,
    2 = LowShelf, 3 = Peak/Notch, 4 = BandPass, 5 = HighShelf, 6 = HighCut,
    7 = AllPass) — a documented int mapping, never an app enum.

    NOTE ON SCOPE: this is deliberately templated on the band COUNT only, not
    on the filter type. Both current consumers use
    std::array<OutputEQBiquadFilter, 6>, and a compile-time band count keeps
    the per-channel array flat and allocation-free. A future pass may generalise
    it over the filter type (so the 4-band reverb processors, which use
    ReverbBiquadFilter with a different shape mapping, can share it too); that
    generalisation is intentionally out of scope here.
*/
template <int NumBands>
class MultiChannelEQBank
{
public:
    MultiChannelEQBank() = default;

    //==========================================================================
    /** Allocates per-channel state for `numChannels` channels and prepares every
        filter at `newSampleRate`. All enable flags are reset to false. Call from
        a non-audio thread — this allocates. */
    void prepare (double newSampleRate, int numChannels)
    {
        sampleRate = newSampleRate;

        const int n = std::max (0, numChannels);

        filters.resize (static_cast<size_t> (n));
        channelEnabled.assign (static_cast<size_t> (n), static_cast<std::uint8_t> (0));

        for (auto& bandArray : filters)
            for (auto& f : bandArray)
                f.prepare (sampleRate);
    }

    /** Clears the filter state of every band on every channel. Coefficients and
        enable flags are left untouched. */
    void reset()
    {
        for (auto& bandArray : filters)
            for (auto& f : bandArray)
                f.reset();
    }

    /** Number of channels allocated by the last prepare() call. */
    int getNumChannels() const noexcept
    {
        return static_cast<int> (filters.size());
    }

    //==========================================================================
    /** Enables or disables a whole channel. Must be pushed BEFORE that
        channel's bands — see the ordering contract in the class docs. No-op
        when `channel` is out of range. */
    void setChannelEnabled (int channel, bool enabled) noexcept
    {
        if (! isValidChannel (channel))
            return;

        channelEnabled[static_cast<size_t> (channel)]
            = enabled ? static_cast<std::uint8_t> (1) : static_cast<std::uint8_t> (0);
    }

    /** Returns false for an out-of-range channel. */
    bool isChannelEnabled (int channel) const noexcept
    {
        if (! isValidChannel (channel))
            return false;

        return channelEnabled[static_cast<size_t> (channel)] != 0;
    }

    /** Pushes one band's parameters. The shape is forced to 0 (OFF) whenever the
        channel is currently disabled, so callers MUST call setChannelEnabled()
        for the channel first. No-op when `channel` or `band` is out of range. */
    void pushBandParameters (int channel, int band, int shape, float freqHz,
                             float gainDb, float q, float slope) noexcept
    {
        if (! isValidChannel (channel) || band < 0 || band >= NumBands)
            return;

        // Disabled channel -> every band forced to OFF (shape 0).
        const int effectiveShape = channelEnabled[static_cast<size_t> (channel)] != 0
                                       ? shape : 0;

        filters[static_cast<size_t> (channel)][static_cast<size_t> (band)]
            .setParameters (effectiveShape, freqHz, gainDb, q, slope);
    }

    //==========================================================================
    /** Audio thread: runs the NumBands biquads in series, in ascending band
        order, over `numSamples` samples in place. No-op when the channel is
        disabled or out of range (each biquad additionally no-ops at shape 0). */
    void processChannel (int channel, float* samples, int numSamples) noexcept
    {
        if (! isValidChannel (channel) || samples == nullptr || numSamples <= 0)
            return;

        if (channelEnabled[static_cast<size_t> (channel)] == 0)
            return;

        auto& bandArray = filters[static_cast<size_t> (channel)];

        for (int b = 0; b < NumBands; ++b)
            bandArray[static_cast<size_t> (b)].processBlock (samples, numSamples);
    }

private:
    //==========================================================================
    bool isValidChannel (int channel) const noexcept
    {
        return channel >= 0 && channel < static_cast<int> (filters.size());
    }

    //==========================================================================
    double sampleRate = 48000.0;

    std::vector<std::array<OutputEQBiquadFilter, NumBands>> filters;
    std::vector<std::uint8_t> channelEnabled;  // uint8_t avoids vector<bool> proxy
};

} // namespace spatcore::dsp

// Extraction-compat aliases — app code migrates to qualified names later.
using spatcore::dsp::MultiChannelEQBank;
