#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "SendMatrixConfig.h"
#include <vector>

namespace spatcore
{
namespace ui
{
namespace sends
{

/**
    A scrollable matrix of dB sends: SOURCE rows into DESTINATION columns, a
    level and an on/off switch per cell.

    NOT THE PATCH MATRIX. That one is boolean, one-to-one on the hardware side
    and bound to a ValueTree it writes itself; this one is many-to-many, every
    cell carries a level, and it owns nothing - the host answers every question
    through SendMatrixConfig and calls refresh() when its state moved. What the
    two share is the shape: a header band, a row-header column, a cell grid
    under two scroll bars, and the Stream Deck navigation block.

    A cell draws its switch as the fill and its level as a bar inside it. Click
    toggles the switch; a vertical drag sets the level (Shift = fine); the
    wheel nudges it; the diagonal is hatched and inert. Rows in the same group
    sit under one collapsible header, so an isolated bunch of effects reads as
    one block. Column badges come from the host: a column inside a feedback
    cycle, and a column fed by an input, which is the entry point of its bunch.

    A drag emits at most one onCellLevelChanged per cell per throttle window
    (20 ms): a level write is a read-modify-write of a whole packed row on the
    host side, and sixty of them per second per cell is what a mouse produces.
*/
class SendMatrixComponent : public juce::Component,
                            private juce::ScrollBar::Listener,
                            private juce::Timer
{
public:
    explicit SendMatrixComponent (SendMatrixConfig configToUse);
    ~SendMatrixComponent() override;

    //==========================================================================
    // Events (widget -> host)

    std::function<void (int row, int column, bool on)>      onCellToggled;
    std::function<void (int row, int column, float levelDb)> onCellLevelChanged;

    /** Fired once at the start of a gesture that will write, with a name for
        the undo transaction the host may open. */
    std::function<void (const juce::String& gestureName)> onGestureStart;

    std::function<void (const juce::String&)> onStatusMessage;

    //==========================================================================
    // Host -> widget

    /** Re-read every provider. Call after any change to the host's rows,
        columns, cells or badges - the widget caches nothing between repaints
        except the row/column counts it sizes its scroll range from. */
    void refresh();

    //==========================================================================
    // Remote navigation (Stream Deck)

    void setSelectedCell (juce::Point<int> cell);
    juce::Point<int> getSelectedCell() const noexcept { return selectedCell; }

    /** Toggle the selected cell's switch. */
    void activateSelectedCell();

    /** Nudge the selected cell's level by a step (dB). */
    void nudgeSelectedLevel (float deltaDb);

    void scrollByCell (int dx, int dy);

    int getNumRows() const noexcept    { return numRows; }
    int getNumColumns() const noexcept { return numColumns; }

    //==========================================================================
    void paint (juce::Graphics&) override;
    void resized() override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseDrag (const juce::MouseEvent&) override;
    void mouseUp (const juce::MouseEvent&) override;
    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;
    void mouseWheelMove (const juce::MouseEvent&, const juce::MouseWheelDetails&) override;
    bool keyPressed (const juce::KeyPress&) override;
    void focusGained (FocusChangeType) override;
    void focusLost (FocusChangeType) override;

private:
    //==========================================================================
    // Host seams. Every one falls back to something usable, none caches.

    SendMatrixPalette palette() const;
    juce::String tr (const char* key) const;
    float uiScale() const;
    int hostRows() const;
    int hostColumns() const;
    SendRowKind rowKindOf (int row) const;
    juce::String rowLabelOf (int row) const;
    juce::Colour rowColourOf (int row) const;
    juce::String columnLabelOf (int column) const;
    int groupOf (int row) const;
    juce::String groupLabelOf (int group) const;
    int selectedColumnOf() const;
    float levelOf (int row, int column) const;
    bool onOf (int row, int column) const;
    bool forbidden (int row, int column) const;
    bool inCycle (int column) const;
    bool isEntry (int column) const;
    void announceNow (const juce::String&) const;
    void announceHover (const juce::String&) const;
    void cancelHoverAnnouncement() const;

    //==========================================================================
    // Layout

    /** The visible row list after collapsing: each entry is either a source
        row or a group header. Rebuilt by refresh() and by a collapse toggle. */
    struct VisibleRow
    {
        bool isHeader = false;
        int  group = 0;         // header: the group it names
        int  sourceRow = -1;    // row: the host row index
    };

    void rebuildVisibleRows();
    void updateScaledSizes();
    void updateScrollBars();
    void scrollToMakeVisible (juce::Point<int> cell);

    int sc (int ref) const;
    juce::Rectangle<int> gridArea() const;
    juce::Rectangle<int> cellBounds (int visibleIndex, int column) const;
    int visibleIndexOfSourceRow (int sourceRow) const;

    /** The cell under a point, in host (row, column) terms, or {-1,-1}.
        A header line reports {-1, -1} with headerHit set to its group. */
    juce::Point<int> cellAt (juce::Point<int> p, int* headerHit = nullptr) const;

    //==========================================================================
    // Drawing

    void drawHeader (juce::Graphics&);
    void drawRowHeaders (juce::Graphics&);
    void drawCells (juce::Graphics&);
    void drawCell (juce::Graphics&, juce::Rectangle<int> bounds, int row, int column);

    //==========================================================================
    // Interaction

    void toggleCell (int row, int column);
    void setCellLevel (int row, int column, float levelDb, bool immediate);
    void flushPendingLevel();
    void timerCallback() override;
    void scrollBarMoved (juce::ScrollBar*, double newRangeStart) override;
    void announceSelectedCell();
    juce::String describeCell (int row, int column) const;

    float levelForPixel (float pixelFraction) const;

    //==========================================================================
    SendMatrixConfig config;

    int numRows = 0;
    int numColumns = 0;
    std::vector<VisibleRow> visibleRows;
    std::vector<bool> collapsedGroups;   // indexed by group id

    juce::ScrollBar horizontalScroll { false };
    juce::ScrollBar verticalScroll { true };
    int scrollOffsetX = 0, scrollOffsetY = 0;
    int maxScrollX = 0, maxScrollY = 0;

    // Scaled geometry
    int cellWidth = 44, cellHeight = 28, headerHeight = 46, rowHeaderWidth = 130;
    int groupHeaderHeight = 22, scrollBarThickness = 16;

    // Pointer state
    juce::Point<int> hoveredCell { -1, -1 };
    juce::Point<int> dragCell { -1, -1 };
    float dragStartLevelDb = 0.0f;
    int dragStartY = 0;
    bool dragMoved = false;

    // Keyboard / remote
    juce::Point<int> selectedCell { -1, -1 };
    bool keyboardNavigationActive = false;

    // Throttle: one level write per cell per window
    juce::Point<int> pendingCell { -1, -1 };
    float pendingLevelDb = 0.0f;
    bool hasPending = false;
    static constexpr int kThrottleMs = 20;
    static constexpr float kNudgeDb = 1.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SendMatrixComponent)
};

} // namespace sends
} // namespace ui
} // namespace spatcore
