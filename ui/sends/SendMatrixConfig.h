#pragma once

#include <juce_core/juce_core.h>
#include <juce_graphics/juce_graphics.h>
#include <functional>

namespace spatcore
{
namespace ui
{
namespace sends
{

/** The colour roles the send matrix draws with. Read through
    SendMatrixConfig::paletteProvider at PAINT time, never cached, so a theme
    change lands on the next repaint. */
struct SendMatrixPalette
{
    juce::Colour background    { 0xFF1E1E1E };
    juce::Colour backgroundAlt { 0xFF262626 };
    juce::Colour surfaceCard   { 0xFF2C2C2C };
    juce::Colour divider       { 0xFF3A3A3A };
    juce::Colour textPrimary   { 0xFFFFFFFF };
    juce::Colour textSecondary { 0xFFB0B0B0 };
    juce::Colour textDisabled  { 0xFF707070 };
    juce::Colour selection     { 0xFFFFC107 };
    juce::Colour warning       { 0xFFE53935 };
    juce::Colour entry         { 0xFF4CAF50 };
};

/** What one row of the matrix is. Inputs come first, effects after; the
    distinction only decides the row's label prefix and whether the diagonal
    applies. */
enum class SendRowKind { Input, Effect };

/**
    Everything the send matrix needs from its host, as callbacks.

    THE WIDGET KNOWS NO SCHEMA. Unlike the patch matrix it listens to no
    ValueTree and holds no property names: the host answers every question at
    the moment it is asked, and calls refresh() when something it owns changed.
    That is what keeps a dB send matrix reusable across families whose rows
    are keyed differently - here by input permanent number and by dense effect
    index at once - without the widget ever learning either.

    PROVIDERS PULL, EVENTS PUSH. Providers (host -> widget) live here and are
    invoked when the value is needed, never snapshotted. Events (widget ->
    host) are public std::function members on the component. Every provider
    is null-checked at the call site and falls back to something sane, so a
    config with nothing filled in still produces an empty but working matrix.
*/
struct SendMatrixConfig
{
    //==========================================================================
    // Shape

    /** Rows are SOURCES: every live input, then every live effect. */
    std::function<int()> numRows;
    std::function<SendRowKind (int row)> rowKind;

    /** Columns are DESTINATIONS: every live effect. */
    std::function<int()> numColumns;

    //==========================================================================
    // Labels, colours, grouping

    std::function<juce::String (int row)> rowLabel;
    std::function<juce::Colour (int row)> rowColour;
    std::function<juce::String (int column)> columnLabel;

    /** A group id per row (0 = none). Consecutive rows in the same non-zero
        group draw under one collapsible header, so an isolated bunch of
        effects reads as one block. Inputs are never grouped. */
    std::function<int (int row)> rowGroup;
    std::function<juce::String (int group)> groupLabel;

    /** The column the host's channel selector is on, or -1. Drawn highlighted
        so the operator can find "this effect" in a 32-wide grid. */
    std::function<int()> selectedColumn;

    //==========================================================================
    // Cell state

    std::function<float (int row, int column)> cellLevelDb;
    std::function<bool  (int row, int column)> cellOn;

    /** True for a cell that can never be set - the effect->effect diagonal.
        Drawn hatched and inert. */
    std::function<bool (int row, int column)> isCellForbidden;

    /** Per-column badges the calculation engine owns: a column inside a
        feedback cycle, and a column fed by at least one input (the ENTRY
        point of its bunch). */
    std::function<bool (int column)> columnInCycle;
    std::function<bool (int column)> columnIsEntry;

    float levelMinDb = -92.0f;
    float levelMaxDb = 0.0f;
    float levelDefaultDb = 0.0f;

    //==========================================================================
    // Presentation

    std::function<SendMatrixPalette()> paletteProvider;

    /** Localise a key; returns the key itself when unresolvable. */
    std::function<juce::String (const char* key)> translate;

    /** Read at layout AND paint. */
    std::function<float()> uiScaleProvider;

    //==========================================================================
    // Accessibility

    std::function<void (const juce::String&)> announce;
    std::function<void (const juce::String&)> announceDebounced;
    std::function<void()> cancelDebouncedAnnouncement;
};

} // namespace sends
} // namespace ui
} // namespace spatcore
