#pragma once

#include <juce_core/juce_core.h>
#include <juce_data_structures/juce_data_structures.h>
#include <juce_graphics/juce_graphics.h>

#include <functional>

namespace spatcore
{
namespace ui
{
namespace patch
{

//==============================================================================
/**
    Colours the patch matrix draws with.

    Read through PatchMatrixConfig::paletteProvider at PAINT time, never cached:
    that is what makes a live theme switch show up on the next repaint rather
    than on the next window open.
*/
struct PatchMatrixPalette
{
    juce::Colour background;      ///< viewport fill
    juce::Colour backgroundAlt;   ///< header band, alternating fills
    juce::Colour surfaceCard;     ///< patched-cell base under the channel colour
    juce::Colour divider;         ///< grid lines and separators
    juce::Colour textPrimary;     ///< labels, selection ring
    juce::Colour textSecondary;   ///< titles, secondary labels
    juce::Colour textDisabled;    ///< overflow (unopened) columns
};

//==============================================================================
/**
    The ValueTree property names the matrix reads and writes.

    The matrix is deliberately ignorant of the host application's parameter
    schema: it is handed the two or three trees it works on plus the names of
    the five properties it touches. Everything else about the schema — how a
    channel is named, what colour a row is — arrives through the callbacks in
    PatchMatrixConfig instead of being walked here, so an app with a completely
    different tree layout can still use this component.
*/
struct PatchMatrixIds
{
    /// On the patch tree: the "r0c0,r0c1;r1c0,..." grid, rows separated by ';'.
    juce::Identifier patchData { "patchData" };

    /// On the patch tree: number of application channels (matrix rows).
    juce::Identifier rows { "rows" };

    /// On the patch tree: number of hardware columns the matrix spans.
    juce::Identifier cols { "cols" };

    /// On the patch tree: how many hardware channels the device actually
    /// OPENED. Columns at or above it are drawn dimmed and refuse patching and
    /// testing, because they have no slot in the audio callback's buffer.
    /// A value <= 0 means "unknown / no device" and disables the gating.
    juce::Identifier activeHardwareChannels { "activeHardwareChannels" };

    /// Optional, on the binaural tree: first channel of the binaural pair
    /// (1-based, -1 when disabled). Those columns are marked in the matrix so
    /// an operator does not patch over the binaural monitor outputs.
    juce::Identifier binauralOutputChannel { "binauralOutputChannel" };
};

//==============================================================================
/**
    Everything PatchMatrixComponent needs from its host application.

    Providers are invoked at the moment the value is needed, never snapshotted,
    so theme, language and UI scale changes take effect on the next repaint or
    layout without the component being rebuilt. Every one of them is
    null-checked at the call site and falls back to something sane, so a config
    with nothing but the trees filled in still produces a working matrix.
*/
struct PatchMatrixConfig
{
    //==========================================================================
    // State the matrix reads and writes.

    /// InputPatch or OutputPatch. The matrix listens to it and writes patchData.
    juce::ValueTree patchTree;

    /// The application's channel list (Inputs or Outputs). The matrix only
    /// LISTENS to it — child added/removed drives the auto-patch heuristic —
    /// and never interprets its contents; names and colours come from the
    /// callbacks below.
    juce::ValueTree channelsTree;

    /// Optional binaural section. Invalid when the host has no binaural
    /// monitoring or on an input patch.
    juce::ValueTree binauralTree;

    PatchMatrixIds ids;

    /// Highest hardware channel count the matrix will ever address; also the
    /// fallback column count when the patch tree carries no `cols`.
    int maxHardwareChannels = 512;

    //==========================================================================
    // Host queries.

    /// Number of application channels — matrix rows.
    std::function<int()> numChannelsProvider;

    /// How many hardware columns row `wfsChannel` (0-based) may hold.
    /// Null or a value < 1 means 1 — the historical strict 1:1 behaviour.
    /// A stereo-pair input row returns 2; within such a row the LOWER
    /// hardware column is the left channel by convention. The COLUMN side
    /// of the invariant is unaffected: a hardware channel still maps to at
    /// most one row, which is what protects the audio path and the
    /// unpatched-row diagnostic.
    std::function<int(int wfsChannel)> rowCapacityProvider;

    /// Called after the matrix writes patchData, so the host can re-run its
    /// own column-count policy (a patch beyond the current width must widen
    /// the matrix; removing one may narrow it).
    std::function<void()> recomputeColumns;

    /// Display name for a row, or an empty string. Row index is 0-based.
    std::function<juce::String(int)> channelNameProvider;

    /// Colour for a row (the patched-cell fill). Row index is 0-based.
    std::function<juce::Colour(int)> rowColourProvider;

    /// Legible text colour against an arbitrary background.
    std::function<juce::Colour(juce::Colour)> contrastingTextProvider;

    //==========================================================================
    // Presentation.

    std::function<PatchMatrixPalette()> paletteProvider;

    /// Localised string for a key. Returns the key itself if unresolvable.
    std::function<juce::String(const char*)> translate;

    /// Global UI scale. Read at layout AND paint time.
    std::function<float()> uiScaleProvider;

    //==========================================================================
    // Accessibility. The matrix is fully keyboard-navigable and announces what
    // the selection is doing; without these it stays silent but functional.

    /// Speak now, interrupting anything queued.
    std::function<void(const juce::String&)> announce;

    /// Speak after a short quiet period — used for hover, which would
    /// otherwise produce a torrent of announcements while the mouse moves.
    std::function<void(const juce::String&)> announceDebounced;

    /// Drop a pending debounced announcement (the mouse left the matrix).
    std::function<void()> cancelDebouncedAnnouncement;
};

} // namespace patch
} // namespace ui
} // namespace spatcore
