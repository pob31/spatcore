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
        s in [numInputChannels, F)  : derived slices 1..5 of stereo ordinal k,
                                       at numInputChannels + 5*k + (slice - 1)
        s in [F, count)             : effect return fx, where F is
                                       firstEffectSlot = numInputChannels
                                       + 5 * numStereo

    A stereo channel always claims all 6 slots regardless of how many slices
    the decomposition backend actually drives; unused slots are
    claimed-and-silent. Slot assignment therefore depends ONLY on the
    channel-type vector — never on the active slice count or width — so slots
    never move under a live parameter edit. Rebuilding the map is a
    config-change operation (it resizes the renderer), never 50 Hz work.

    Effect returns are appended AFTER the derived region, which is what keeps
    that promise when a show gains or loses an effects channel: no input or
    derived slot ever moves, so a matrix row an engine is already using cannot
    change meaning underneath it.
*/
/** What a render source IS. An effect return is spatialised exactly like an
    input - it has a position, it feeds the reverbs, it is rendered by all four
    renderers - but it is fed by an effects chain rather than by a hardware
    input, and several consumers need to tell the two apart: a return has no
    owning input channel, no solo bit in the per-input mask, and no floor
    reflections. Before this existed those consumers keyed on
    owningInputChannel < 0, which is true of an unused slot as well, so a
    return would have rendered at full level at the world origin rather than
    failing in any visible way. */
enum class SourceKind : uint8_t { Input = 0, EffectReturn = 1 };

struct RenderSourceDesc
{
    /** Input channel that owns this source, or -1 for an unused slot and for
        every effect return. */
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

    /** Input or effect return. See SourceKind. */
    SourceKind kind = SourceKind::Input;

    /** Effects channel that owns this source, or -1 for an input source. */
    int16_t owningEffectChannel = -1;

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

    /** Every render source an INPUT can produce: one per channel plus five
        derived slices per stereo pair. */
    static constexpr int kMaxInputRenderSources =
        kMaxInputChannels + kMaxStereoChannels * kDerivedPerStereo;    // 104

    /** Capability bound on effects channels. Tied by static_assert to
        spatcore::effects::kMaxEffectChannels in the compile-check TU: wfs/ may
        not include effects/, so nothing else can tie them, and an untied pair
        would let one side grow and overrun `desc` with no compile error. */
    static constexpr int kMaxEffectChannels = 32;

    /** How many slots `desc` actually holds. */
    static constexpr int kMaxRenderSourceSlots =
        kMaxInputRenderSources + kMaxEffectChannels;                   // 136

    /** The app mirrors this in WFSParameterDefaults and asserts equality, so it
        still means "input render sources" until the app learns about effect
        returns. Phase 4 of the effects work raises it to kMaxRenderSourceSlots
        in the same commit as the app-side mirror; until then the two must not
        disagree, and there is no ordering of two separate commits that would
        keep the app building if this moved first. */
    static constexpr int kMaxRenderSources = kMaxInputRenderSources;   // 104

    /** Channel type values, matching the app's inputChannelType parameter. */
    enum ChannelType : uint8_t { Mono = 0, Stereo = 1 };

    int numInputChannels = 0;
    int numEffectChannels = 0;
    int count = 0;                                    // total render sources

    /** First effect-return slot, or -1 when there are no effects channels
        (the same convention as firstDerivedSlot). */
    int firstEffectSlot = -1;

    std::array<RenderSourceDesc, kMaxRenderSourceSlots> desc {};

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
          - effect return fx maps to the slot at firstEffectSlot + fx, after
            every input and derived slot, so adding or removing an effects
            channel moves no existing source;
          - count == numInputChannels + 5 * (number of stereo channels)
            + numEffectChannels. */
    static bool build (const uint8_t* channelTypes, int numInputChannels,
                       int numEffectChannels, RenderSourceMap& out) noexcept
    {
        out = RenderSourceMap {};

        if (channelTypes == nullptr
            || numInputChannels < 0
            || numInputChannels > kMaxInputChannels
            || numEffectChannels < 0
            || numEffectChannels > kMaxEffectChannels)
            return false;

        out.numInputChannels = numInputChannels;
        out.numEffectChannels = numEffectChannels;
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

        // Effect returns, after everything an input can claim.
        out.firstEffectSlot = (numEffectChannels > 0) ? next : -1;

        for (int fx = 0; fx < numEffectChannels; ++fx)
        {
            auto& d = out.desc[static_cast<size_t> (next++)];
            d.owningInputChannel = -1;
            d.owningEffectChannel = static_cast<int16_t> (fx);
            d.sliceIndex = 0;
            d.isStereoSlice = false;
            d.kind = SourceKind::EffectReturn;
            d.active = true;
            d.gainLinear = 1.0f;
        }

        out.count = next;
        return true;
    }

    /** The input-only map: what every caller wanted before effects channels
        existed, and still what the app asks for until it learns about them. */
    static bool build (const uint8_t* channelTypes, int numInputChannels,
                       RenderSourceMap& out) noexcept
    {
        return build (channelTypes, numInputChannels, 0, out);
    }

    /** The identity map: every channel mono, no effects. Never fails for a
        valid count. */
    static bool buildIdentity (int numInputChannels, RenderSourceMap& out) noexcept
    {
        std::array<uint8_t, kMaxInputChannels> mono {};
        return build (mono.data(), numInputChannels, 0, out);
    }
};

} // namespace spatcore::wfs
