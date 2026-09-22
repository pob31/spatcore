#include "SendMatrixComponent.h"

#include <algorithm>
#include <cmath>

namespace spatcore
{
namespace ui
{
namespace sends
{

//==============================================================================
SendMatrixComponent::SendMatrixComponent (SendMatrixConfig configToUse)
    : config (std::move (configToUse))
{
    setWantsKeyboardFocus (true);
    setMouseClickGrabsKeyboardFocus (true);

    addAndMakeVisible (horizontalScroll);
    addAndMakeVisible (verticalScroll);
    horizontalScroll.addListener (this);
    verticalScroll.addListener (this);
    horizontalScroll.setAutoHide (false);
    verticalScroll.setAutoHide (false);

    updateScaledSizes();
    refresh();
}

SendMatrixComponent::~SendMatrixComponent()
{
    stopTimer();
    horizontalScroll.removeListener (this);
    verticalScroll.removeListener (this);
}

//==============================================================================
// Host seams

SendMatrixPalette SendMatrixComponent::palette() const
{
    return config.paletteProvider ? config.paletteProvider() : SendMatrixPalette {};
}

juce::String SendMatrixComponent::tr (const char* key) const
{
    return config.translate ? config.translate (key) : juce::String (key);
}

float SendMatrixComponent::uiScale() const
{
    return config.uiScaleProvider ? config.uiScaleProvider() : 1.0f;
}

int SendMatrixComponent::hostRows() const        { return config.numRows ? juce::jmax (0, config.numRows()) : 0; }
int SendMatrixComponent::hostColumns() const     { return config.numColumns ? juce::jmax (0, config.numColumns()) : 0; }

SendRowKind SendMatrixComponent::rowKindOf (int row) const
{
    return config.rowKind ? config.rowKind (row) : SendRowKind::Input;
}

juce::String SendMatrixComponent::rowLabelOf (int row) const
{
    return config.rowLabel ? config.rowLabel (row) : juce::String (row + 1);
}

juce::Colour SendMatrixComponent::rowColourOf (int row) const
{
    return config.rowColour ? config.rowColour (row) : palette().textSecondary;
}

juce::String SendMatrixComponent::columnLabelOf (int column) const
{
    return config.columnLabel ? config.columnLabel (column) : juce::String (column + 1);
}

int SendMatrixComponent::groupOf (int row) const
{
    return config.rowGroup ? juce::jmax (0, config.rowGroup (row)) : 0;
}

juce::String SendMatrixComponent::groupLabelOf (int group) const
{
    return config.groupLabel ? config.groupLabel (group) : (tr ("sends.group") + " " + juce::String (group));
}

int SendMatrixComponent::selectedColumnOf() const
{
    return config.selectedColumn ? config.selectedColumn() : -1;
}

float SendMatrixComponent::levelOf (int row, int column) const
{
    return config.cellLevelDb ? config.cellLevelDb (row, column) : config.levelDefaultDb;
}

bool SendMatrixComponent::onOf (int row, int column) const
{
    return config.cellOn ? config.cellOn (row, column) : false;
}

bool SendMatrixComponent::forbidden (int row, int column) const
{
    return config.isCellForbidden ? config.isCellForbidden (row, column) : false;
}

bool SendMatrixComponent::inCycle (int column) const
{
    return config.columnInCycle ? config.columnInCycle (column) : false;
}

bool SendMatrixComponent::isEntry (int column) const
{
    return config.columnIsEntry ? config.columnIsEntry (column) : false;
}

void SendMatrixComponent::announceNow (const juce::String& text) const
{
    if (config.announce)
        config.announce (text);
}

void SendMatrixComponent::announceHover (const juce::String& text) const
{
    if (config.announceDebounced)
        config.announceDebounced (text);
}

void SendMatrixComponent::cancelHoverAnnouncement() const
{
    if (config.cancelDebouncedAnnouncement)
        config.cancelDebouncedAnnouncement();
}

//==============================================================================
// Layout

int SendMatrixComponent::sc (int ref) const
{
    const float s = uiScale();
    return juce::jmax (static_cast<int> (ref * 0.65f), static_cast<int> (ref * s));
}

void SendMatrixComponent::updateScaledSizes()
{
    cellWidth          = sc (44);
    cellHeight         = sc (28);
    headerHeight       = sc (46);
    rowHeaderWidth     = sc (130);
    groupHeaderHeight  = sc (22);
    scrollBarThickness = sc (16);
}

void SendMatrixComponent::refresh()
{
    numRows = hostRows();
    numColumns = hostColumns();

    // Group ids are small (a handful of link groups); size the collapse table
    // to the largest one seen so a new group arrives expanded.
    int maxGroup = 0;
    for (int r = 0; r < numRows; ++r)
        maxGroup = juce::jmax (maxGroup, groupOf (r));
    if (static_cast<int> (collapsedGroups.size()) <= maxGroup)
        collapsedGroups.resize (static_cast<size_t> (maxGroup + 1), false);

    rebuildVisibleRows();

    if (selectedCell.x >= numColumns || selectedCell.y >= numRows)
        selectedCell = { -1, -1 };

    updateScrollBars();
    repaint();
}

void SendMatrixComponent::rebuildVisibleRows()
{
    visibleRows.clear();
    visibleRows.reserve (static_cast<size_t> (numRows + 8));

    int currentGroup = 0;
    for (int r = 0; r < numRows; ++r)
    {
        const int g = groupOf (r);

        if (g != 0 && g != currentGroup)
        {
            VisibleRow header;
            header.isHeader = true;
            header.group = g;
            visibleRows.push_back (header);
        }
        currentGroup = g;

        if (g != 0 && g < static_cast<int> (collapsedGroups.size()) && collapsedGroups[static_cast<size_t> (g)])
            continue;                       // folded away under its header

        VisibleRow row;
        row.sourceRow = r;
        visibleRows.push_back (row);
    }
}

juce::Rectangle<int> SendMatrixComponent::gridArea() const
{
    return getLocalBounds()
              .withTrimmedTop (headerHeight)
              .withTrimmedLeft (rowHeaderWidth)
              .withTrimmedRight (scrollBarThickness)
              .withTrimmedBottom (scrollBarThickness);
}

int SendMatrixComponent::visibleIndexOfSourceRow (int sourceRow) const
{
    for (size_t i = 0; i < visibleRows.size(); ++i)
        if (! visibleRows[i].isHeader && visibleRows[i].sourceRow == sourceRow)
            return static_cast<int> (i);

    return -1;
}

juce::Rectangle<int> SendMatrixComponent::cellBounds (int visibleIndex, int column) const
{
    // Rows have two heights (a group header is shorter than a cell), so the y
    // of a row is the sum of what sits above it rather than index * height.
    int y = headerHeight - scrollOffsetY;
    for (int i = 0; i < visibleIndex && i < static_cast<int> (visibleRows.size()); ++i)
        y += visibleRows[static_cast<size_t> (i)].isHeader ? groupHeaderHeight : cellHeight;

    const int h = visibleRows[static_cast<size_t> (visibleIndex)].isHeader ? groupHeaderHeight : cellHeight;
    const int x = rowHeaderWidth + column * cellWidth - scrollOffsetX;
    return { x, y, cellWidth, h };
}

juce::Point<int> SendMatrixComponent::cellAt (juce::Point<int> p, int* headerHit) const
{
    if (headerHit != nullptr)
        *headerHit = 0;

    if (! gridArea().contains (p) && ! (p.x < rowHeaderWidth && p.y >= headerHeight))
        return { -1, -1 };

    int y = headerHeight - scrollOffsetY;
    for (size_t i = 0; i < visibleRows.size(); ++i)
    {
        const auto& vr = visibleRows[i];
        const int h = vr.isHeader ? groupHeaderHeight : cellHeight;

        if (p.y >= y && p.y < y + h)
        {
            if (vr.isHeader)
            {
                if (headerHit != nullptr)
                    *headerHit = vr.group;
                return { -1, -1 };
            }

            if (p.x < rowHeaderWidth)
                return { -1, vr.sourceRow };      // the row header, no column

            const int column = (p.x - rowHeaderWidth + scrollOffsetX) / juce::jmax (1, cellWidth);
            if (column < 0 || column >= numColumns)
                return { -1, -1 };

            return { column, vr.sourceRow };
        }

        y += h;
    }

    return { -1, -1 };
}

void SendMatrixComponent::updateScrollBars()
{
    int contentH = 0;
    for (const auto& vr : visibleRows)
        contentH += vr.isHeader ? groupHeaderHeight : cellHeight;
    const int contentW = numColumns * cellWidth;

    const auto grid = gridArea();
    maxScrollX = juce::jmax (0, contentW - grid.getWidth());
    maxScrollY = juce::jmax (0, contentH - grid.getHeight());
    scrollOffsetX = juce::jlimit (0, maxScrollX, scrollOffsetX);
    scrollOffsetY = juce::jlimit (0, maxScrollY, scrollOffsetY);

    horizontalScroll.setRangeLimits (0.0, static_cast<double> (juce::jmax (contentW, 1)));
    horizontalScroll.setCurrentRange (static_cast<double> (scrollOffsetX), static_cast<double> (juce::jmax (1, grid.getWidth())),
                                      juce::dontSendNotification);
    verticalScroll.setRangeLimits (0.0, static_cast<double> (juce::jmax (contentH, 1)));
    verticalScroll.setCurrentRange (static_cast<double> (scrollOffsetY), static_cast<double> (juce::jmax (1, grid.getHeight())),
                                    juce::dontSendNotification);

    horizontalScroll.setVisible (maxScrollX > 0);
    verticalScroll.setVisible (maxScrollY > 0);
}

void SendMatrixComponent::scrollToMakeVisible (juce::Point<int> cell)
{
    if (cell.x < 0 || cell.y < 0)
        return;

    const int vi = visibleIndexOfSourceRow (cell.y);
    if (vi < 0)
        return;

    const auto grid = gridArea();
    auto b = cellBounds (vi, cell.x);

    if (b.getX() < grid.getX())
        scrollOffsetX -= grid.getX() - b.getX();
    else if (b.getRight() > grid.getRight())
        scrollOffsetX += b.getRight() - grid.getRight();

    if (b.getY() < grid.getY())
        scrollOffsetY -= grid.getY() - b.getY();
    else if (b.getBottom() > grid.getBottom())
        scrollOffsetY += b.getBottom() - grid.getBottom();

    updateScrollBars();
}

void SendMatrixComponent::resized()
{
    updateScaledSizes();

    auto area = getLocalBounds();
    horizontalScroll.setBounds (rowHeaderWidth, area.getBottom() - scrollBarThickness,
                                area.getWidth() - rowHeaderWidth - scrollBarThickness, scrollBarThickness);
    verticalScroll.setBounds (area.getRight() - scrollBarThickness, headerHeight,
                              scrollBarThickness, area.getHeight() - headerHeight - scrollBarThickness);

    updateScrollBars();
}

//==============================================================================
// Drawing

void SendMatrixComponent::paint (juce::Graphics& g)
{
    const auto pal = palette();
    g.fillAll (pal.background);

    drawCells (g);
    drawRowHeaders (g);
    drawHeader (g);

    // The corner over the row headers, under the column headers
    g.setColour (pal.backgroundAlt);
    g.fillRect (0, 0, rowHeaderWidth, headerHeight);
    g.setColour (pal.textSecondary);
    g.setFont (juce::jmax (10.0f, 12.0f * uiScale()));
    g.drawText (tr ("sends.corner"), juce::Rectangle<int> (0, 0, rowHeaderWidth, headerHeight).reduced (sc (6), 0),
                juce::Justification::centredLeft, true);
}

void SendMatrixComponent::drawHeader (juce::Graphics& g)
{
    const auto pal = palette();
    const float us = uiScale();

    juce::Rectangle<int> band (rowHeaderWidth, 0, getWidth() - rowHeaderWidth - scrollBarThickness, headerHeight);
    g.setColour (pal.backgroundAlt);
    g.fillRect (band);

    g.saveState();
    g.reduceClipRegion (band);

    const int selected = selectedColumnOf();
    const int badge = sc (7);

    for (int c = 0; c < numColumns; ++c)
    {
        juce::Rectangle<int> cell (rowHeaderWidth + c * cellWidth - scrollOffsetX, 0, cellWidth, headerHeight);
        if (cell.getRight() < band.getX() || cell.getX() > band.getRight())
            continue;

        if (c == selected)
        {
            g.setColour (pal.selection.withAlpha (0.25f));
            g.fillRect (cell);
        }

        g.setColour (pal.divider);
        g.drawVerticalLine (cell.getRight() - 1, 0.0f, static_cast<float> (headerHeight));

        // Two badges in the top corners: cycle (warning) left, entry right.
        if (inCycle (c))
        {
            g.setColour (pal.warning);
            g.fillEllipse (static_cast<float> (cell.getX() + sc (3)), static_cast<float> (sc (3)),
                           static_cast<float> (badge), static_cast<float> (badge));
        }
        if (isEntry (c))
        {
            g.setColour (pal.entry);
            g.fillEllipse (static_cast<float> (cell.getRight() - sc (3) - badge), static_cast<float> (sc (3)),
                           static_cast<float> (badge), static_cast<float> (badge));
        }

        g.setColour (c == selected ? pal.textPrimary : pal.textSecondary);
        g.setFont (juce::Font (juce::FontOptions (juce::jmax (9.0f, 11.0f * us))
                               .withStyle (c == selected ? "Bold" : "Regular")));
        g.drawFittedText (columnLabelOf (c), cell.reduced (sc (2), sc (10)), juce::Justification::centred, 2, 0.8f);
    }

    g.restoreState();

    g.setColour (pal.divider);
    g.drawHorizontalLine (headerHeight - 1, 0.0f, static_cast<float> (getWidth()));
}

void SendMatrixComponent::drawRowHeaders (juce::Graphics& g)
{
    const auto pal = palette();
    const float us = uiScale();

    juce::Rectangle<int> column (0, headerHeight, rowHeaderWidth, getHeight() - headerHeight - scrollBarThickness);
    g.setColour (pal.backgroundAlt);
    g.fillRect (column);

    g.saveState();
    g.reduceClipRegion (column);

    for (size_t i = 0; i < visibleRows.size(); ++i)
    {
        const auto& vr = visibleRows[i];
        auto b = cellBounds (static_cast<int> (i), 0).withX (0).withWidth (rowHeaderWidth);
        if (b.getBottom() < column.getY() || b.getY() > column.getBottom())
            continue;

        if (vr.isHeader)
        {
            const bool collapsed = vr.group < static_cast<int> (collapsedGroups.size())
                                && collapsedGroups[static_cast<size_t> (vr.group)];
            g.setColour (pal.surfaceCard);
            g.fillRect (b);
            g.setColour (pal.textSecondary);
            g.setFont (juce::Font (juce::FontOptions (juce::jmax (9.0f, 11.0f * us)).withStyle ("Bold")));
            g.drawText ((collapsed ? juce::String::fromUTF8 ("\xe2\x96\xb8 ") : juce::String::fromUTF8 ("\xe2\x96\xbe "))
                            + groupLabelOf (vr.group),
                        b.reduced (sc (6), 0), juce::Justification::centredLeft, true);
            continue;
        }

        const int r = vr.sourceRow;
        const auto colour = rowColourOf (r);

        // A colour swatch, then the label, as the patch matrix's row headers do
        g.setColour (colour);
        g.fillRect (b.getX() + sc (4), b.getY() + sc (6), sc (6), b.getHeight() - sc (12));

        g.setColour (r == selectedCell.y ? pal.textPrimary : pal.textSecondary);
        g.setFont (juce::jmax (9.0f, 11.0f * us));
        g.drawFittedText (rowLabelOf (r), b.reduced (sc (14), 0).withTrimmedLeft (sc (2)),
                          juce::Justification::centredLeft, 1, 0.8f);

        g.setColour (pal.divider);
        g.drawHorizontalLine (b.getBottom() - 1, 0.0f, static_cast<float> (rowHeaderWidth));
    }

    g.restoreState();

    g.setColour (pal.divider);
    g.drawVerticalLine (rowHeaderWidth - 1, static_cast<float> (headerHeight), static_cast<float> (getHeight()));
}

void SendMatrixComponent::drawCells (juce::Graphics& g)
{
    const auto grid = gridArea();
    g.saveState();
    g.reduceClipRegion (grid);

    const auto pal = palette();
    const int selected = selectedColumnOf();

    for (size_t i = 0; i < visibleRows.size(); ++i)
    {
        const auto& vr = visibleRows[i];
        auto rowBounds = cellBounds (static_cast<int> (i), 0);
        if (rowBounds.getBottom() < grid.getY() || rowBounds.getY() > grid.getBottom())
            continue;

        if (vr.isHeader)
        {
            g.setColour (pal.surfaceCard);
            g.fillRect (grid.getX(), rowBounds.getY(), grid.getWidth(), rowBounds.getHeight());
            g.setColour (pal.divider);
            g.drawHorizontalLine (rowBounds.getBottom() - 1, static_cast<float> (grid.getX()),
                                  static_cast<float> (grid.getRight()));
            continue;
        }

        for (int c = 0; c < numColumns; ++c)
        {
            auto b = cellBounds (static_cast<int> (i), c);
            if (b.getRight() < grid.getX() || b.getX() > grid.getRight())
                continue;

            if (c == selected)
            {
                g.setColour (pal.selection.withAlpha (0.08f));
                g.fillRect (b);
            }

            drawCell (g, b, vr.sourceRow, c);
        }
    }

    g.restoreState();
}

void SendMatrixComponent::drawCell (juce::Graphics& g, juce::Rectangle<int> bounds, int row, int column)
{
    const auto pal = palette();
    auto inner = bounds.reduced (sc (2));

    if (forbidden (row, column))
    {
        // The diagonal: hatched, and nothing else. An effect cannot feed itself.
        g.setColour (pal.textDisabled.withAlpha (0.35f));
        g.saveState();
        g.reduceClipRegion (inner);
        for (int x = inner.getX() - inner.getHeight(); x < inner.getRight(); x += sc (6))
            g.drawLine (static_cast<float> (x), static_cast<float> (inner.getBottom()),
                        static_cast<float> (x + inner.getHeight()), static_cast<float> (inner.getY()), 1.0f);
        g.restoreState();
        g.setColour (pal.divider);
        g.drawRect (bounds, 1);
        return;
    }

    const bool on = onOf (row, column);
    const float levelDb = juce::jlimit (config.levelMinDb, config.levelMaxDb, levelOf (row, column));
    const float frac = (config.levelMaxDb > config.levelMinDb)
                     ? (levelDb - config.levelMinDb) / (config.levelMaxDb - config.levelMinDb)
                     : 1.0f;

    const auto colour = rowColourOf (row);

    // Off: the frame only, with a faint bar so the level is still readable.
    // On: the frame filled in the row colour, the bar drawn brighter on top.
    g.setColour (on ? colour.withAlpha (0.35f) : pal.surfaceCard);
    g.fillRect (inner);

    const int barW = juce::jmax (1, static_cast<int> (std::round (inner.getWidth() * frac)));
    g.setColour (on ? colour : colour.withAlpha (0.25f));
    g.fillRect (inner.withWidth (barW));

    g.setColour (on ? colour.brighter (0.3f) : pal.divider);
    g.drawRect (inner, 1);

    // The level, when there is room for it
    if (inner.getWidth() >= sc (36))
    {
        g.setColour (on ? pal.textPrimary : pal.textDisabled);
        g.setFont (juce::jmax (8.0f, 9.5f * uiScale()));
        g.drawText (juce::String (levelDb, 0), inner, juce::Justification::centred, false);
    }

    if (hoveredCell.x == column && hoveredCell.y == row)
    {
        g.setColour (pal.textPrimary.withAlpha (0.12f));
        g.fillRect (inner);
    }

    if (keyboardNavigationActive && selectedCell.x == column && selectedCell.y == row)
    {
        g.setColour (juce::Colours::white);
        g.drawRect (bounds.reduced (1), 2);
        g.setColour (juce::Colours::black);
        g.drawRect (bounds.reduced (3), 1);
    }

    g.setColour (pal.divider);
    g.drawRect (bounds, 1);
}

//==============================================================================
// Interaction

float SendMatrixComponent::levelForPixel (float pixelFraction) const
{
    return config.levelMinDb + juce::jlimit (0.0f, 1.0f, pixelFraction) * (config.levelMaxDb - config.levelMinDb);
}

void SendMatrixComponent::toggleCell (int row, int column)
{
    if (row < 0 || column < 0 || forbidden (row, column))
        return;

    const bool next = ! onOf (row, column);

    if (onGestureStart)
        onGestureStart (tr ("sends.gesture.toggle"));
    if (onCellToggled)
        onCellToggled (row, column, next);

    announceNow (describeCell (row, column));
    repaint();
}

void SendMatrixComponent::setCellLevel (int row, int column, float levelDb, bool immediate)
{
    if (row < 0 || column < 0 || forbidden (row, column))
        return;

    levelDb = juce::jlimit (config.levelMinDb, config.levelMaxDb, levelDb);

    if (immediate)
    {
        if (onCellLevelChanged)
            onCellLevelChanged (row, column, levelDb);
        repaint();
        return;
    }

    // Coalesce: the newest value for the cell wins, one write per window.
    pendingCell = { column, row };
    pendingLevelDb = levelDb;
    hasPending = true;
    if (! isTimerRunning())
        startTimer (kThrottleMs);
    repaint();
}

void SendMatrixComponent::flushPendingLevel()
{
    if (! hasPending)
        return;

    hasPending = false;
    if (onCellLevelChanged)
        onCellLevelChanged (pendingCell.y, pendingCell.x, pendingLevelDb);
}

void SendMatrixComponent::timerCallback()
{
    flushPendingLevel();
    if (! hasPending)
        stopTimer();
}

void SendMatrixComponent::mouseDown (const juce::MouseEvent& e)
{
    grabKeyboardFocus();

    int headerGroup = 0;
    const auto cell = cellAt (e.getPosition(), &headerGroup);

    if (headerGroup != 0)
    {
        if (headerGroup < static_cast<int> (collapsedGroups.size()))
            collapsedGroups[static_cast<size_t> (headerGroup)] = ! collapsedGroups[static_cast<size_t> (headerGroup)];
        rebuildVisibleRows();
        updateScrollBars();
        repaint();
        return;
    }

    if (cell.x < 0 || cell.y < 0)
        return;

    if (e.mods.isRightButtonDown() || e.mods.isPopupMenu())
        return;

    dragCell = cell;
    dragStartLevelDb = levelOf (cell.y, cell.x);
    dragStartY = e.y;
    dragMoved = false;

    selectedCell = cell;
    keyboardNavigationActive = true;
}

void SendMatrixComponent::mouseDrag (const juce::MouseEvent& e)
{
    if (dragCell.x < 0 || dragCell.y < 0 || forbidden (dragCell.y, dragCell.x))
        return;

    const int dy = dragStartY - e.y;          // up = louder
    if (! dragMoved && std::abs (dy) < 3)
        return;

    if (! dragMoved)
    {
        dragMoved = true;
        if (onGestureStart)
            onGestureStart (tr ("sends.gesture.level"));
    }

    // One cell height of travel is 24 dB; Shift makes it 6.
    const float range = config.levelMaxDb - config.levelMinDb;
    const float perPixel = (e.mods.isShiftDown() ? 6.0f : 24.0f) / static_cast<float> (juce::jmax (1, cellHeight));
    const float next = juce::jlimit (config.levelMinDb, config.levelMaxDb,
                                     dragStartLevelDb + dy * perPixel);
    juce::ignoreUnused (range);

    setCellLevel (dragCell.y, dragCell.x, next, false);
}

void SendMatrixComponent::mouseUp (const juce::MouseEvent& e)
{
    if (dragCell.x < 0 || dragCell.y < 0)
        return;

    const auto cell = dragCell;
    dragCell = { -1, -1 };

    if (dragMoved)
    {
        flushPendingLevel();
        stopTimer();
        announceNow (describeCell (cell.y, cell.x));
        return;
    }

    if (e.mods.isLeftButtonDown() || e.mouseWasClicked())
        toggleCell (cell.y, cell.x);
}

void SendMatrixComponent::mouseMove (const juce::MouseEvent& e)
{
    const auto cell = cellAt (e.getPosition());
    if (cell != hoveredCell)
    {
        hoveredCell = cell;
        repaint();

        if (cell.x >= 0 && cell.y >= 0)
            announceHover (describeCell (cell.y, cell.x));
    }
}

void SendMatrixComponent::mouseExit (const juce::MouseEvent&)
{
    hoveredCell = { -1, -1 };
    cancelHoverAnnouncement();
    repaint();
}

void SendMatrixComponent::mouseWheelMove (const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel)
{
    const auto cell = cellAt (e.getPosition());

    // Over a cell with Ctrl/Cmd: nudge its level. Otherwise scroll, and
    // Shift turns a vertical wheel horizontal for plain mice.
    if (cell.x >= 0 && cell.y >= 0 && (e.mods.isCtrlDown() || e.mods.isCommandDown()))
    {
        if (onGestureStart)
            onGestureStart (tr ("sends.gesture.level"));
        setCellLevel (cell.y, cell.x, levelOf (cell.y, cell.x) + (wheel.deltaY > 0 ? kNudgeDb : -kNudgeDb), true);
        return;
    }

    if (wheel.deltaX != 0.0f)
        scrollOffsetX = juce::jlimit (0, maxScrollX, scrollOffsetX - static_cast<int> (wheel.deltaX * cellWidth * 3));

    if (wheel.deltaY != 0.0f)
    {
        if (e.mods.isShiftDown())
            scrollOffsetX = juce::jlimit (0, maxScrollX, scrollOffsetX - static_cast<int> (wheel.deltaY * cellWidth * 3));
        else
            scrollOffsetY = juce::jlimit (0, maxScrollY, scrollOffsetY - static_cast<int> (wheel.deltaY * cellHeight * 3));
    }

    updateScrollBars();
    repaint();
}

void SendMatrixComponent::scrollBarMoved (juce::ScrollBar* bar, double newRangeStart)
{
    if (bar == &horizontalScroll)
        scrollOffsetX = static_cast<int> (newRangeStart);
    else if (bar == &verticalScroll)
        scrollOffsetY = static_cast<int> (newRangeStart);

    repaint();
}

bool SendMatrixComponent::keyPressed (const juce::KeyPress& key)
{
    if (numRows <= 0 || numColumns <= 0)
        return false;

    const bool arrow = key == juce::KeyPress::leftKey || key == juce::KeyPress::rightKey
                    || key == juce::KeyPress::upKey   || key == juce::KeyPress::downKey;

    juce::Point<int> next = selectedCell;

    if (arrow)
    {
        if (! keyboardNavigationActive || selectedCell.x < 0 || selectedCell.y < 0)
            next = { 0, 0 };
        else if (key == juce::KeyPress::leftKey)  next.x = juce::jmax (0, selectedCell.x - 1);
        else if (key == juce::KeyPress::rightKey) next.x = juce::jmin (numColumns - 1, selectedCell.x + 1);
        else if (key == juce::KeyPress::upKey)    next.y = juce::jmax (0, selectedCell.y - 1);
        else if (key == juce::KeyPress::downKey)  next.y = juce::jmin (numRows - 1, selectedCell.y + 1);
    }
    else if (key == juce::KeyPress::homeKey)
        next = { 0, 0 };
    else if (key == juce::KeyPress::endKey)
        next = { numColumns - 1, numRows - 1 };
    else if (key == juce::KeyPress::pageUpKey)
        next.y = juce::jmax (0, selectedCell.y - 10);
    else if (key == juce::KeyPress::pageDownKey)
        next.y = juce::jmin (numRows - 1, selectedCell.y + 10);
    else if (key == juce::KeyPress::spaceKey || key == juce::KeyPress::returnKey)
    {
        activateSelectedCell();
        return true;
    }
    else if (key.getTextCharacter() == '+' || key.getTextCharacter() == '=')
    {
        nudgeSelectedLevel (key.getModifiers().isShiftDown() ? kNudgeDb * 0.1f : kNudgeDb);
        return true;
    }
    else if (key.getTextCharacter() == '-')
    {
        nudgeSelectedLevel (key.getModifiers().isShiftDown() ? -kNudgeDb * 0.1f : -kNudgeDb);
        return true;
    }
    else
        return false;

    setSelectedCell (next);
    return true;
}

void SendMatrixComponent::setSelectedCell (juce::Point<int> cell)
{
    cell.x = juce::jlimit (-1, numColumns - 1, cell.x);
    cell.y = juce::jlimit (-1, numRows - 1, cell.y);

    selectedCell = cell;
    keyboardNavigationActive = cell.x >= 0 && cell.y >= 0;

    if (keyboardNavigationActive)
    {
        scrollToMakeVisible (cell);
        announceSelectedCell();
    }

    repaint();
}

void SendMatrixComponent::activateSelectedCell()
{
    if (selectedCell.x >= 0 && selectedCell.y >= 0)
        toggleCell (selectedCell.y, selectedCell.x);
}

void SendMatrixComponent::nudgeSelectedLevel (float deltaDb)
{
    if (selectedCell.x < 0 || selectedCell.y < 0)
        return;

    if (onGestureStart)
        onGestureStart (tr ("sends.gesture.level"));

    setCellLevel (selectedCell.y, selectedCell.x, levelOf (selectedCell.y, selectedCell.x) + deltaDb, true);
    announceSelectedCell();
}

void SendMatrixComponent::scrollByCell (int dx, int dy)
{
    scrollOffsetX = juce::jlimit (0, maxScrollX, scrollOffsetX + dx * cellWidth);
    scrollOffsetY = juce::jlimit (0, maxScrollY, scrollOffsetY + dy * cellHeight);
    updateScrollBars();
    repaint();
}

void SendMatrixComponent::focusGained (FocusChangeType)
{
    repaint();
}

void SendMatrixComponent::focusLost (FocusChangeType)
{
    keyboardNavigationActive = false;
    repaint();
}

//==============================================================================
// Accessibility

juce::String SendMatrixComponent::describeCell (int row, int column) const
{
    if (row < 0 || column < 0)
        return {};

    if (forbidden (row, column))
        return rowLabelOf (row) + " " + tr ("sends.announce.into") + " " + columnLabelOf (column)
             + ", " + tr ("sends.announce.forbidden");

    return rowLabelOf (row) + " " + tr ("sends.announce.into") + " " + columnLabelOf (column)
         + ", " + (onOf (row, column) ? tr ("sends.announce.on") : tr ("sends.announce.off"))
         + ", " + juce::String (levelOf (row, column), 1) + " dB";
}

void SendMatrixComponent::announceSelectedCell()
{
    announceNow (describeCell (selectedCell.y, selectedCell.x));
}

} // namespace sends
} // namespace ui
} // namespace spatcore
