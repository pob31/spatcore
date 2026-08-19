#pragma once

#include <array>
#include <cstdint>

namespace spatcore::wfs {

//==============================================================================
/**
    The renderer's source dimension, split from the user-visible input count.

    A *render source* is one row of the WFS delay/level/HF matrices and one
    channel of the patched input buffer. A mono input channel is exactly one
    render source. A stereo-pair input channel is SIX: its channel-primary slot
    (slice 0) plus five derived slice slots appended past the user-visible
    inputs. Derived slots are a separate source class with their own budget —
    like reverb nodes, they never appear in the audio patch matrices and are
    not user-addressable.

    Slot layout (fixed — the doc's "4 pairs x 6 feeds = 24 derived sources"
    arithmetic counts 6 per pair, 1 primary + 5 derived):

        s in [0, numInputChannels)   : channel-primary source of input s
                                       (mono: the channel; stereo: slice 0)
        s in [numInputChannels, ...) : derived slices 1..5 of stereo ordinal k,
                                       at numInputChannels + 5*k + (slice - 1)

    A stereo channel always claims all 6 slots regardless of how many slices
    the decomposition backend actually drives; unused slots are
    claimed-and-silent. Slot assignment therefore depends ONLY on the
    channel-type vector — never on the active slice count or width — so slots
    never move under a live parameter edit. Rebuilding the map is a
    config-change operation (it resizes the renderer), never 50 Hz work.
*/
struct RenderSourceDesc
{
    /** Input channel that owns this source, or -1 for an unused slot. */
    int16_t owningInputChannel = -1;

    /** 0 for a mono channel or a stereo channel's primary slot; 1..5 for the
        derived slices of a stereo channel. */
    uint8_t sliceIndex = 0;

    /** True for every source of a stereo channel, including slice 0. */
    bool isStereoSlice = false;

    /** False = claimed-and-silent (a slice slot the backend is not driving).
        Inactive sources still occupy their matrix row; the audio stage clears
        their buffer every block. */
    bool active = true;

    /** Slice position offset from the owning channel's anchor, in metres.
        Always zero for mono sources. Refreshed at control rate from the
        decomposition state — this is the 50 Hz half of the split. */
    float offsetX = 0.0f, offsetY = 0.0f, offsetZ = 0.0f;

    /** Per-slice gain applied on top of the channel's level matrix (confidence
        collapse, ambience trim). Always 1 for mono sources. */
    float gainLinear = 1.0f;
};

struct RenderSourceMap
{
    /** Mirrors the app's WFSParameterDefaults::maxInputChannels. spatcore may
        not include app headers (dep lint), so the app static_asserts the two
        constants are equal at its boundary. */
    static constexpr int kMaxInputChannels = 64;

    static constexpr int kDerivedPerStereo  = 5;
    static constexpr int kSlicesPerStereo   = kDerivedPerStereo + 1;   // 6
    static constexpr int kMaxStereoChannels = 8;
    static constexpr int kMaxRenderSources  =
        kMaxInputChannels + kMaxStereoChannels * kDerivedPerStereo;    // 104

    /** Channel type values, matching the app's inputChannelType parameter. */
    enum ChannelType : uint8_t { Mono = 0, Stereo = 1 };

    int numInputChannels = 0;
    int count = 0;                                    // total render sources
    std::array<RenderSourceDesc, kMaxRenderSources> desc {};

    /** Bar-position table: derived base slot per input channel, or -1 for a
        channel with no derived slots (mono). */
    std::array<int16_t, kMaxInputChannels> firstDerivedSlot {};

    /** Build the slot map from the channel-type vector. Pure, deterministic,
        allocation-free. Returns false (leaving `out` as an empty map) when the
        inputs exceed the fixed budgets: more than kMaxInputChannels channels,
        or more than kMaxStereoChannels stereo channels.

        Postconditions on success:
          - mono channel i maps to render source i and nothing else;
          - stereo channel i maps to source i (slice 0) plus 5 contiguous
            derived slots; derived slots are ordered by stereo ordinal, so the
            layout is independent of anything but the type vector;
          - count == numInputChannels + 5 * (number of stereo channels). */
    static bool build (const uint8_t* channelTypes, int numInputChannels,
                       RenderSourceMap& out) noexcept
    {
        out = RenderSourceMap {};

        if (channelTypes == nullptr
            || numInputChannels < 0
            || numInputChannels > kMaxInputChannels)
            return false;

        out.numInputChannels = numInputChannels;
        out.firstDerivedSlot.fill (-1);

        // Channel-primary slots.
        for (int i = 0; i < numInputChannels; ++i)
        {
            auto& d = out.desc[static_cast<size_t> (i)];
            d.owningInputChannel = static_cast<int16_t> (i);
            d.sliceIndex = 0;
            d.isStereoSlice = (channelTypes[i] == Stereo);
            d.active = true;
        }

        // Derived slots, appended in stereo-ordinal order.
        int next = numInputChannels;
        int numStereo = 0;
        for (int i = 0; i < numInputChannels; ++i)
        {
            if (channelTypes[i] != Stereo)
                continue;

            if (++numStereo > kMaxStereoChannels)
            {
                out = RenderSourceMap {};
                return false;
            }

            out.firstDerivedSlot[static_cast<size_t> (i)] = static_cast<int16_t> (next);
            for (int s = 1; s <= kDerivedPerStereo; ++s)
            {
                auto& d = out.desc[static_cast<size_t> (next++)];
                d.owningInputChannel = static_cast<int16_t> (i);
                d.sliceIndex = static_cast<uint8_t> (s);
                d.isStereoSlice = true;
                d.active = true;
            }
        }

        out.count = next;
        return true;
    }

    /** The identity map: every channel mono. Never fails for a valid count. */
    static bool buildIdentity (int numInputChannels, RenderSourceMap& out) noexcept
    {
        std::array<uint8_t, kMaxInputChannels> mono {};
        return build (mono.data(), numInputChannels, out);
    }
};

} // namespace spatcore::wfs
