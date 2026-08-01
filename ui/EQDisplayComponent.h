#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_graphics/juce_graphics.h>
#include <juce_data_structures/juce_data_structures.h>

#include "../dsp/BiquadResponse.h"
#include "../dsp/OutputEQBiquadFilter.h"
#include "../dsp/ReverbBiquadFilter.h"

#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <map>
#include <vector>

namespace spatcore::ui {

//==============================================================================
/**
    Unified filter type for the EQ display.

    This reconciles the two different shape encodings the host applications
    use (see EQDisplayConfig::hasBandPass and shapeToFilterType below) into a
    single vocabulary.

    It is used for INTERACTION logic only — does this shape have a gain
    control, where does its marker sit vertically, which crosshair handles are
    live. It is deliberately NOT used to compute the response curve: the curve
    is designed from the RAW shape int by the very filter class the audio path
    runs (see the "Response math" section of EQDisplayComponent), so the two
    can never drift.
*/
enum class EQFilterType
{
    Off = 0,
    LowCut,      // high-pass with resonance
    LowShelf,
    PeakNotch,
    BandPass,
    AllPass,     // phase only, flat magnitude
    HighShelf,
    HighCut      // low-pass with resonance
};

//==============================================================================
/**
    Theme colours an EQDisplayComponent needs from its host application.

    spatcore may never name an app symbol, so the colours both original
    components read straight from their application colour scheme are supplied
    through EQDisplayConfig::paletteProvider instead. Six roles cover every
    scheme field either original touched:

      Role             | Where it is used                        | App source
      -----------------+-----------------------------------------+---------------------------------
      background       | full-bounds scrim painted over the graph | ColorScheme::get().background
                       | when the EQ is disabled (alpha 0.7)      |
      panelBackground  | the graph plotting area itself           | ColorScheme::get().backgroundAlt
                       | (filled opaque, first thing drawn)       |     .darker (0.3f)
      gridLine         | frequency + dB grid rules; the component | ColorScheme::get().chromeDivider
                       | applies its own alphas (0.3/0.6 vertical,|
                       | 0.4/0.8 horizontal, 0 dB emphasised)     |
      textPrimary      | response-curve stroke and the selection  | ColorScheme::get().textPrimary
                       | ring around the selected band marker     |
      textSecondary    | axis labels (Hz + dB) and the disabled   | ColorScheme::get().textSecondary
                       | "EQ OFF" caption                         |
      curveFill        | fill between the curve and the 0 dB line | ColorScheme::get().accentBlue
                       | (alpha 0.2 applied by the component)     |

    Note the .darker (0.3f) on panelBackground: the originals applied it at
    paint time, so an app factory must apply it when building the palette to
    reproduce the current look exactly. Every other role is the raw scheme
    colour — all alpha/darkening beyond that is applied internally.

    The band-marker rainbow is deliberately absent: it is theme-invariant and
    owned by the component (see EQDisplayComponent::getBandColour), because
    the band number/label/toggle/dial colours in the surrounding UI have to
    match it exactly.
*/
struct EQDisplayPalette
{
    juce::Colour background, panelBackground, gridLine, textPrimary, textSecondary, curveFill;
};

//==============================================================================
/**
    Parameter-id / range / encoding seam so one component can serve several
    different EQs (Output or speaker EQ, reverb pre-EQ, reverb post-EQ), each
    with its own ValueTree property names, Q range and shape numbering.

    slopeId
        When valid, the shelf slope (RBJ "S") is read from this property and
        clamped to [slopeMin, slopeMax].
        When INVALID, the band's Q doubles as S — the legacy WFS-DIY reverb
        behaviour, preserved so a config with no slope parameter still draws
        the shelves the way that app always drew them.

    hasBandPass
        Selects both the shape numbering AND which filter class designs the
        curve:
          true  => Output/speaker EQ order, spatcore::dsp::OutputEQBiquadFilter
                   0 Off, 1 LowCut, 2 LowShelf, 3 Peak, 4 BandPass,
                   5 HighShelf, 6 HighCut, 7 AllPass
          false => Reverb pre/post EQ order, spatcore::dsp::ReverbBiquadFilter
                   0 Off, 1 LowCut, 2 LowShelf, 3 Peak, 4 HighShelf,
                   5 HighCut, 6 BandPass
        Each display therefore draws with the exact coefficients its own audio
        path will run.

    Providers
        paletteProvider / disabledTextProvider are invoked AT PAINT TIME and
        never cached, so live theme and language switches show up on the next
        repaint. Both may be nullptr, in which case the built-in neutral
        palette and the literal "EQ OFF" are used.

    There are deliberately NO static factory functions here: a factory would
    have to name the application's parameter-ID and defaults headers, which
    spatcore is forbidden from including. Each app keeps its own
    forOutputEQ() / forReverbPreEQ() / forReverbPostEQ() / forSpeakerEQ()
    factories app-side and hands the finished config to the constructor.
*/
struct EQDisplayConfig
{
    juce::Identifier shapeId, frequencyId, gainId, qId;
    juce::Identifier slopeId;      // invalid => shelves use Q as S (legacy behaviour)

    float qMin = 0.1f, qMax = 10.0f;
    float slopeMin = 0.1f, slopeMax = 1.0f;
    bool hasBandPass = false;      // true => Output-EQ shape order, else Reverb order

    std::function<EQDisplayPalette()> paletteProvider;    // nullptr => built-in neutral palette
    std::function<juce::String()> disabledTextProvider;   // nullptr => "EQ OFF"
};

//==============================================================================
/**
    Interactive parametric EQ graph.

    Draws a log-frequency response curve summed across N bands over a
    frequency/dB grid, with one draggable colour-coded marker per band, and
    edits the bands in place:

      - drag a marker             : frequency + gain together
      - drag the selected band's  : vertical line = frequency only,
        crosshair lines             horizontal line = gain only (relative)
      - mouse wheel / magnify     : Q of the selected band (multiplicative)
      - two-finger pinch          : Q of the band nearest the pinch midpoint
      - left/right arrows         : frequency, log-ish step of freq/20
      - up/down arrows            : gain, +/-0.1 dB (gain-bearing shapes only)
      - escape                    : deselect

    Cut, band-pass and all-pass shapes have no gain control: their marker sits
    on the 0 dB line, the horizontal crosshair is not drawn and vertical drags
    and up/down arrows are ignored for them. A band whose shape is 0 (OFF)
    keeps its marker at its stored gain and keeps both crosshairs live, so the
    user can park it before switching it on.

    Response math
    -------------
    The curve is NOT computed here. Per band the raw shape int is read from
    the ValueTree and handed to the static coefficient designer of the filter
    class that config.hasBandPass selects
    (spatcore::dsp::OutputEQBiquadFilter or spatcore::dsp::ReverbBiquadFilter),
    then evaluated with spatcore::dsp::biquadMagnitudeDb. Both apps used to
    carry their own private copy of the cookbook formulas; those copies are
    retired, which is the whole point of the extraction. Band magnitudes are
    summed in dB and the total is clamped to [mindB - 6, maxdB + 6] before
    plotting.

    Coefficients are cached per band and recomputed on any ValueTree property
    change on a band child (or on the parent), on setSampleRate, and after
    every parameter write this component performs.

    Write model
    -----------
    Every parameter mutation — from every gesture — funnels through the single
    private helper setBandParameter, so the two persistence modes can never
    diverge:

      ownerPersistsValue == false (default)
          The component writes the band's ValueTree itself with
          setProperty (id, value, undoManager) and ALSO calls
          onParameterChanged if one is installed. This is the WFS-DIY reverb
          behaviour.

      ownerPersistsValue == true
          The component NEVER touches the tree; it only calls
          onParameterChanged and lets the owner persist. Required wherever the
          owner needs to see the pre-edit value — e.g. an array "relative"
          mode that computes (newValue - oldValue), which would read as zero
          if we had already written — and wherever the owner persists through
          its own scoped-undo seam. This is the WFS-DIY Output EQ behaviour,
          and the only mode XOA uses.

      In either mode the owner's (or our own) write lands on this same live
      ValueTree node, so the listener below repaints the curve.

    Undo
    ----
    setUndoManager installs the manager used both for the direct setProperty
    writes and for per-gesture transaction boundaries: one
    beginNewTransaction at the START of each user gesture (marker drag,
    crosshair drag, pinch, wheel, magnify, arrow-key nudge), so a gesture
    coalesces into a single undo step. Transactions are begun ONLY when a
    manager has been set; with no manager the component still works and simply
    writes without undo.

    Threading
    ---------
    Message thread only, like any juce::Component. The ValueTree it listens to
    must therefore also be mutated on the message thread. The coefficient
    designers it calls are pure static functions and impose no constraints.

    Injected-colour / string seam
    -----------------------------
    config.paletteProvider and config.disabledTextProvider are called at paint
    time and never cached (the hosts support live theme and language
    switching); every call site nullptr-checks and falls back to the built-in
    defaults, so a config with no providers paints sensibly and cannot crash.
*/
class EQDisplayComponent : public juce::Component,
                           private juce::ValueTree::Listener
{
public:
    //==========================================================================
    /** @param eqParentTree  parent whose child i is band i
        @param numBandsIn    number of band children to draw and edit
        @param configIn      parameter ids, ranges, shape encoding, providers
    */
    EQDisplayComponent (juce::ValueTree eqParentTree,
                        int numBandsIn,
                        const EQDisplayConfig& configIn)
        : eqTree (eqParentTree),
          numBands (juce::jmax (0, numBandsIn)),
          config (configIn)
    {
        eqTree.addListener (this);
        bandCoefficients.resize (static_cast<size_t> (numBands));
        updateAllCoefficients();

        setInterceptsMouseClicks (true, false);
        setWantsKeyboardFocus (true);
    }

    ~EQDisplayComponent() override
    {
        eqTree.removeListener (this);
    }

    //==========================================================================
    /** Sets the vertical (dB) extent of the graph. */
    void setdBRange (float newMinDb, float newMaxDb)
    {
        mindB = newMinDb;
        maxdB = newMaxDb;
        repaint();
    }

    /** Sets the sample rate the response is evaluated at. Ignored when not
        positive; recomputes every band when it actually changes.
    */
    void setSampleRate (double newSampleRate)
    {
        if (newSampleRate > 0.0 && sampleRate != newSampleRate)
        {
            sampleRate = newSampleRate;
            updateAllCoefficients();
            repaint();
        }
    }

    /** When false the graph is dimmed behind a scrim and captioned. Purely
        cosmetic: the curve is still drawn and interaction still works.
    */
    void setEQEnabled (bool enabled)
    {
        if (eqEnabled != enabled)
        {
            eqEnabled = enabled;
            repaint();
        }
    }

    bool isEQEnabled() const { return eqEnabled; }

    //==========================================================================
    void paint (juce::Graphics& g) override
    {
        // Providers are read once per paint and never cached — that is what
        // makes live theme / language switches take effect on repaint.
        const auto palette = getPalette();

        drawGrid (g, palette);
        drawResponseCurve (g, palette);
        drawBandMarkers (g, palette);

        if (! eqEnabled)
        {
            g.setColour (palette.background.withAlpha (0.7f));
            g.fillRect (getLocalBounds());

            g.setColour (palette.textSecondary);
            g.setFont (juce::FontOptions (juce::jmax (14.0f, 24.0f * paintScale())));
            g.drawText (getDisabledText(), getLocalBounds(), juce::Justification::centred);
        }
    }

    void resized() override { repaint(); }

    //==========================================================================
    // Mouse interaction (with multitouch pinch support)
    //==========================================================================
    void mouseDown (const juce::MouseEvent& e) override
    {
        int touchIndex = e.source.getIndex();

        // Track this touch point.
        TouchInfo touch;
        touch.position = e.position;
        touch.startPosition = e.position;
        activeTouches[touchIndex] = touch;

        // Second touch down => pinch gesture start.
        if (activeTouches.size() == 2)
        {
            isPinching = true;
            pinchStartDistance = getTouchDistance();
            pinchStartQ = 0.0f;

            // Target the band closest to the midpoint between the two fingers.
            auto midpoint = getTouchMidpoint();
            int centeredBand = findBandNearestToPoint (midpoint);

            if (centeredBand >= 0)
            {
                selectedBand = centeredBand;
                isDragging = false;   // cancel any single-finger drag

                beginGesture ("EQ Pinch Q");
            }

            if (selectedBand >= 0)
            {
                auto bandTree = eqTree.getChild (selectedBand);
                if (bandTree.isValid())
                    pinchStartQ = bandTree.getProperty (config.qId);
            }

            repaint();   // update selection highlight
            return;
        }

        // Single touch - normal band selection / drag.
        int clickedBand = findBandAtPosition (e.position);

        if (clickedBand >= 0)
        {
            selectedBand = clickedBand;
            isDragging = true;
            dragMode = DragMode::Both;
            dragStartPos = e.position;
            setMouseCursor (juce::MouseCursor::DraggingHandCursor);

            beginGesture ("EQ Band " + juce::String (selectedBand + 1));

            // Auto-repeat keeps drag events coming while the pointer is still.
            beginDragAutoRepeat (50);
            grabKeyboardFocus();
        }
        else
        {
            // Maybe a crosshair line of the currently selected band.
            auto crosshairMode = findCrosshairAtPosition (e.position);
            if (crosshairMode != DragMode::None)
            {
                isDragging = true;
                dragMode = crosshairMode;
                dragStartPos = e.position;

                // Remember the originals so the drag can be relative.
                auto bandTree = eqTree.getChild (selectedBand);
                dragStartFreq = static_cast<float> (static_cast<int> (bandTree.getProperty (config.frequencyId)));
                dragStartGain = bandTree.getProperty (config.gainId);

                setMouseCursor (crosshairMode == DragMode::GainOnly
                                    ? juce::MouseCursor::UpDownResizeCursor
                                    : juce::MouseCursor::LeftRightResizeCursor);

                beginGesture ("EQ Band " + juce::String (selectedBand + 1));

                beginDragAutoRepeat (50);
            }
            else
            {
                // Empty space - deselect.
                selectedBand = -1;
                isDragging = false;
                dragMode = DragMode::None;
            }
        }

        repaint();
    }

    void mouseDrag (const juce::MouseEvent& e) override
    {
        int touchIndex = e.source.getIndex();

        auto it = activeTouches.find (touchIndex);
        if (it != activeTouches.end())
            it->second.position = e.position;

        // Pinch (two fingers): pinch in => higher Q (narrower).
        if (isPinching && activeTouches.size() >= 2 && selectedBand >= 0)
        {
            float currentDistance = getTouchDistance();
            if (pinchStartDistance > 0.0f && pinchStartQ > 0.0f)
            {
                float scaleFactor = currentDistance / pinchStartDistance;
                float newQ = juce::jlimit (config.qMin, config.qMax, pinchStartQ / scaleFactor);
                setBandParameter (selectedBand, config.qId, newQ);
            }
            return;
        }

        // Normal single-finger drag.
        if (! isDragging || selectedBand < 0)
            return;

        auto bandTree = eqTree.getChild (selectedBand);
        if (! bandTree.isValid())
            return;

        int shape = bandTree.getProperty (config.shapeId);
        EQFilterType filterType = shapeToFilterType (shape);

        // Frequency (unless gain-only mode).
        if (dragMode != DragMode::GainOnly)
        {
            float newFreq;
            if (dragMode == DragMode::FrequencyOnly)
            {
                // Relative: apply the X delta to the original frequency.
                float startX = frequencyToX (dragStartFreq);
                newFreq = xToFrequency (startX + (e.position.x - dragStartPos.x));
            }
            else
            {
                newFreq = xToFrequency (e.position.x);
            }
            newFreq = juce::jlimit (20.0f, 20000.0f, newFreq);
            setBandParameter (selectedBand, config.frequencyId, static_cast<int> (newFreq));
        }

        // Gain (unless freq-only mode, or a shape with no gain control).
        if (dragMode != DragMode::FrequencyOnly && hasGainControl (filterType))
        {
            float newGain;
            if (dragMode == DragMode::GainOnly)
            {
                // Relative: apply the Y delta to the original gain.
                float startY = dBToY (dragStartGain);
                newGain = yTodB (startY + (e.position.y - dragStartPos.y));
            }
            else
            {
                newGain = yTodB (e.position.y);
            }
            newGain = juce::jlimit (mindB, maxdB, newGain);
            setBandParameter (selectedBand, config.gainId, newGain);
        }
    }

    void mouseUp (const juce::MouseEvent& e) override
    {
        activeTouches.erase (e.source.getIndex());

        if (activeTouches.size() < 2)
            isPinching = false;

        if (activeTouches.empty())
        {
            isDragging = false;
            dragMode = DragMode::None;
            setMouseCursor (juce::MouseCursor::NormalCursor);
        }
        // Selection persists for wheel / keyboard adjustment.
    }

    void mouseMove (const juce::MouseEvent& e) override
    {
        if (isDragging)
            return;

        if (findBandAtPosition (e.position) >= 0)
        {
            setMouseCursor (juce::MouseCursor::PointingHandCursor);
        }
        else
        {
            auto crosshairMode = findCrosshairAtPosition (e.position);
            if (crosshairMode == DragMode::GainOnly)
                setMouseCursor (juce::MouseCursor::UpDownResizeCursor);
            else if (crosshairMode == DragMode::FrequencyOnly)
                setMouseCursor (juce::MouseCursor::LeftRightResizeCursor);
            else
                setMouseCursor (juce::MouseCursor::NormalCursor);
        }
    }

    void mouseExit (const juce::MouseEvent&) override
    {
        if (! isDragging)
            setMouseCursor (juce::MouseCursor::NormalCursor);
    }

    void mouseWheelMove (const juce::MouseEvent&,
                         const juce::MouseWheelDetails& wheel) override
    {
        if (selectedBand < 0)
            return;

        auto bandTree = eqTree.getChild (selectedBand);
        if (! bandTree.isValid())
            return;

        beginGesture ("EQ Wheel Q");

        // Multiplicative Q adjustment, for every filter type.
        float currentQ = bandTree.getProperty (config.qId);
        float newQ = juce::jlimit (config.qMin, config.qMax,
                                   currentQ * (1.0f + wheel.deltaY * 0.5f));
        setBandParameter (selectedBand, config.qId, newQ);
    }

    void mouseMagnify (const juce::MouseEvent& e, float scaleFactor) override
    {
        // Native magnify gesture where the platform provides one.
        int targetBand = selectedBand >= 0 ? selectedBand : findBandAtPosition (e.position);
        if (targetBand < 0)
            return;

        auto bandTree = eqTree.getChild (targetBand);
        if (! bandTree.isValid())
            return;

        beginGesture ("EQ Magnify Q");

        float currentQ = bandTree.getProperty (config.qId);
        float newQ = juce::jlimit (config.qMin, config.qMax, currentQ * scaleFactor);
        setBandParameter (targetBand, config.qId, newQ);

        selectedBand = targetBand;
        repaint();
    }

    //==========================================================================
    // Keyboard interaction
    //==========================================================================
    bool keyPressed (const juce::KeyPress& key) override
    {
        if (key.getKeyCode() == juce::KeyPress::escapeKey && selectedBand >= 0)
        {
            selectedBand = -1;
            repaint();
            return true;
        }

        if (selectedBand < 0)
            return false;

        auto bandTree = eqTree.getChild (selectedBand);
        if (! bandTree.isValid())
            return false;

        EQFilterType filterType = shapeToFilterType (bandTree.getProperty (config.shapeId));
        const int keyCode = key.getKeyCode();

        if (keyCode == juce::KeyPress::leftKey || keyCode == juce::KeyPress::rightKey)
        {
            // Frequency: logarithmic-ish increment of freq / 20.
            int currentFreq = bandTree.getProperty (config.frequencyId);
            int increment = juce::jmax (1, currentFreq / 20);
            int newFreq = juce::jlimit (20, 20000,
                                        keyCode == juce::KeyPress::rightKey ? currentFreq + increment
                                                                            : currentFreq - increment);

            beginGesture ("EQ Arrow Freq");

            setBandParameter (selectedBand, config.frequencyId, newFreq);
            return true;
        }

        if ((keyCode == juce::KeyPress::upKey || keyCode == juce::KeyPress::downKey)
            && hasGainControl (filterType))
        {
            // Gain: +/-0.1 dB.
            float currentGain = bandTree.getProperty (config.gainId);
            float newGain = juce::jlimit (mindB, maxdB,
                                          currentGain + (keyCode == juce::KeyPress::upKey ? 0.1f : -0.1f));

            beginGesture ("EQ Arrow Gain");

            setBandParameter (selectedBand, config.gainId, newGain);
            return true;
        }

        return false;
    }

    //==========================================================================
    /** Index of the currently selected band, or -1 if none. */
    int getSelectedBand() const { return selectedBand; }

    /** Selects a band programmatically; out-of-range values deselect. */
    void setSelectedBand (int band)
    {
        selectedBand = (band >= 0 && band < numBands) ? band : -1;
        repaint();
    }

    /** Installs the UndoManager used for direct ValueTree writes and for
        per-gesture transaction boundaries. nullptr (the default) disables
        both; the component still edits, just without undo.
    */
    void setUndoManager (juce::UndoManager* um) { undoManagerPtr = um; }

    /** Called for every parameter change the user makes: (band, id, value).

        Always called when set, in both persistence modes — see the class doc.
        The owner may use it purely as a notification (default mode) or as the
        sole persistence path (ownerPersistsValue == true).
    */
    std::function<void (int, const juce::Identifier&, const juce::var&)> onParameterChanged;

    /** false (default): this component writes the band ValueTree itself (with
        undo) and additionally notifies onParameterChanged.
        true: this component never writes the tree — it only calls
        onParameterChanged and the owner persists. See the class doc.
    */
    bool ownerPersistsValue = false;

    /** The shared 8-colour band rainbow: red, orange, yellow, green, blue,
        purple, teal, pink, wrapping every 8 bands. Static and theme-invariant
        so band labels, toggles and dials elsewhere in the UI can match the
        markers exactly.
    */
    static juce::Colour getBandColour (int band)
    {
        static const juce::Colour colours[] = {
            juce::Colour (0xFFE74C3C),  // 1 red
            juce::Colour (0xFFE67E22),  // 2 orange
            juce::Colour (0xFFFFEB3B),  // 3 yellow
            juce::Colour (0xFF2ECC71),  // 4 green
            juce::Colour (0xFF3498DB),  // 5 blue
            juce::Colour (0xFF9B59B6),  // 6 purple
            juce::Colour (0xFF1ABC9C),  // 7 teal
            juce::Colour (0xFFE91E63),  // 8 pink
        };
        return colours[band % 8];
    }

private:
    //==========================================================================
    // Injected-seam accessors — always called at paint time, never cached.
    //==========================================================================
    /** Neutral fallback used when no paletteProvider has been installed. */
    static EQDisplayPalette getDefaultPalette()
    {
        return { juce::Colour (0xff1e1e1e),   // background
                 juce::Colour (0xff141414),   // panelBackground
                 juce::Colour (0xff505050),   // gridLine
                 juce::Colour (0xffe0e0e0),   // textPrimary
                 juce::Colour (0xff909090),   // textSecondary
                 juce::Colour (0xff3498db) }; // curveFill
    }

    EQDisplayPalette getPalette() const
    {
        return config.paletteProvider != nullptr ? config.paletteProvider()
                                                 : getDefaultPalette();
    }

    juce::String getDisabledText() const
    {
        return config.disabledTextProvider != nullptr ? config.disabledTextProvider()
                                                      : juce::String ("EQ OFF");
    }

    //==========================================================================
    /** Paint scale vs the 180px reference height. */
    float paintScale() const { return juce::jmax (0.65f, static_cast<float> (getHeight()) / 180.0f); }

    /** Cuts, band-pass and all-pass have no gain control: their marker sits on
        the 0 dB line and vertical edits are ignored.
    */
    static bool hasGainControl (EQFilterType t)
    {
        return t != EQFilterType::LowCut && t != EQFilterType::HighCut
            && t != EQFilterType::BandPass && t != EQFilterType::AllPass;
    }

    /** Opens one undo transaction per user gesture, but only when an
        UndoManager has been installed.
    */
    void beginGesture (const juce::String& transactionName)
    {
        if (undoManagerPtr != nullptr)
            undoManagerPtr->beginNewTransaction (transactionName);
    }

    //==========================================================================
    // ValueTree::Listener
    //==========================================================================
    void valueTreePropertyChanged (juce::ValueTree& tree, const juce::Identifier&) override
    {
        for (int i = 0; i < numBands; ++i)
        {
            if (tree == eqTree.getChild (i))
            {
                updateBandCoefficients (i);
                repaint();
                return;
            }
        }

        if (tree == eqTree)
        {
            updateAllCoefficients();
            repaint();
        }
    }

    void valueTreeChildAdded (juce::ValueTree&, juce::ValueTree&) override {}
    void valueTreeChildRemoved (juce::ValueTree&, juce::ValueTree&, int) override {}
    void valueTreeChildOrderChanged (juce::ValueTree&, int, int) override {}
    void valueTreeParentChanged (juce::ValueTree&) override {}

    //==========================================================================
    // Drawing
    //==========================================================================
    void drawGrid (juce::Graphics& g, const EQDisplayPalette& palette)
    {
        auto bounds = getLocalBounds().toFloat();

        g.setColour (palette.panelBackground);
        g.fillRect (bounds);

        // Frequency grid lines (logarithmic).
        const float freqLines[] = {
            20, 30, 40, 50, 60, 70, 80, 90,
            100, 200, 300, 400, 500, 600, 700, 800, 900,
            1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
            10000, 20000
        };

        const auto gridColor = palette.gridLine;
        for (float freq : freqLines)
        {
            float x = frequencyToX (freq);
            bool isMajor = (freq == 100 || freq == 1000 || freq == 10000);
            g.setColour (isMajor ? gridColor.withAlpha (0.6f) : gridColor.withAlpha (0.3f));
            g.drawVerticalLine (static_cast<int> (x), bounds.getY(), bounds.getBottom());
        }

        // Frequency labels.
        float ps = paintScale();
        g.setColour (palette.textSecondary);
        g.setFont (juce::FontOptions (juce::jmax (7.0f, 10.0f * ps)));

        const float labelFreqs[] = { 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000 };
        const char* labelTexts[] = { "20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k" };

        int freqLabelW = static_cast<int> (30 * ps);
        int freqLabelH = static_cast<int> (12 * ps);
        int freqLabelOff = static_cast<int> (15 * ps);
        for (int i = 0; i < 10; ++i)
        {
            float x = frequencyToX (labelFreqs[i]);
            g.drawText (labelTexts[i],
                        static_cast<int> (x) - freqLabelOff,
                        static_cast<int> (bounds.getBottom()) - freqLabelOff,
                        freqLabelW, freqLabelH,
                        juce::Justification::centred);
        }

        // dB grid lines, 0 dB emphasised.
        for (float dB = mindB; dB <= maxdB; dB += 6.0f)
        {
            float y = dBToY (dB);
            g.setColour (gridColor.withAlpha (std::abs (dB) < 0.1f ? 0.8f : 0.4f));
            g.drawHorizontalLine (static_cast<int> (y), bounds.getX(), bounds.getRight());

            g.setColour (palette.textSecondary);
            juce::String label = (dB > 0 ? "+" : "") + juce::String (static_cast<int> (dB));
            g.drawText (label,
                        static_cast<int> (2 * ps), static_cast<int> (y) - static_cast<int> (6 * ps),
                        static_cast<int> (25 * ps), static_cast<int> (12 * ps),
                        juce::Justification::left);
        }
    }

    void drawResponseCurve (juce::Graphics& g, const EQDisplayPalette& palette)
    {
        juce::Path responseCurve;

        const int numPoints = juce::jmax (2, getWidth());
        const float zeroY = dBToY (0.0f);

        for (int x = 0; x < numPoints; ++x)
        {
            float freq = xToFrequency (static_cast<float> (x));
            float y = dBToY (calculateTotalResponse (freq));

            if (x == 0)
                responseCurve.startNewSubPath (static_cast<float> (x), y);
            else
                responseCurve.lineTo (static_cast<float> (x), y);
        }

        // Filled area between the curve and the 0 dB line.
        juce::Path filledCurve = responseCurve;
        filledCurve.lineTo (static_cast<float> (getWidth()), zeroY);
        filledCurve.lineTo (0.0f, zeroY);
        filledCurve.closeSubPath();

        g.setColour (palette.curveFill.withAlpha (0.2f));
        g.fillPath (filledCurve);

        g.setColour (palette.textPrimary);
        g.strokePath (responseCurve, juce::PathStrokeType (2.0f));
    }

    void drawBandMarkers (juce::Graphics& g, const EQDisplayPalette& palette)
    {
        for (int band = 0; band < numBands; ++band)
        {
            auto bandTree = eqTree.getChild (band);
            if (! bandTree.isValid())
                continue;

            int shape = bandTree.getProperty (config.shapeId);
            bool isOff = (shape == 0);

            // Marker sits at frequency/gain even when the band is off.
            float freq = bandTree.getProperty (config.frequencyId);
            float gain = bandTree.getProperty (config.gainId);
            float x = frequencyToX (freq);
            float y = isOff ? dBToY (gain) : getBandMarkerPosition (band).y;

            // Band colour, darkened when OFF (like an inactive slider track).
            juce::Colour bandColour = getBandColour (band);
            if (isOff)
                bandColour = bandColour.darker (0.6f);

            bool isSelected = (selectedBand == band);
            float ps = paintScale();
            float markerSize = isSelected ? 20.0f * ps : 14.0f * ps;

            g.setColour (bandColour);
            g.fillEllipse (x - markerSize / 2, y - markerSize / 2, markerSize, markerSize);

            if (isSelected)
            {
                g.setColour (palette.textPrimary);
                float ringOff = 3.0f * ps;
                g.drawEllipse (x - markerSize / 2 - ringOff, y - markerSize / 2 - ringOff,
                               markerSize + ringOff * 2, markerSize + ringOff * 2, 2.0f * ps);

                EQFilterType filterType = shapeToFilterType (shape);
                g.setColour (bandColour.withAlpha (0.35f));

                // Vertical crosshair (frequency) — always drawn.
                g.drawLine (x, 0.0f, x, static_cast<float> (getHeight()), 1.0f);

                // Horizontal crosshair (gain) — skipped for active cut /
                // band-pass / all-pass; off bands always show both, since the
                // remembered shape is unknown.
                if (isOff || hasGainControl (filterType))
                    g.drawLine (0.0f, y, static_cast<float> (getWidth()), y, 1.0f);
            }

            // Band number.
            g.setColour (juce::Colours::black);
            g.setFont (juce::FontOptions (juce::jmax (8.0f, 13.0f * ps), juce::Font::bold));
            g.drawText (juce::String (band + 1),
                        static_cast<int> (x - markerSize / 2), static_cast<int> (y - markerSize / 2),
                        static_cast<int> (markerSize), static_cast<int> (markerSize),
                        juce::Justification::centred);
        }
    }

    //==========================================================================
    // Coordinate conversion (log frequency, linear dB)
    //==========================================================================
    float frequencyToX (float freq) const
    {
        const float minFreq = 20.0f, maxFreq = 20000.0f;
        float normalized = std::log10 (freq / minFreq) / std::log10 (maxFreq / minFreq);
        return static_cast<float> (getWidth()) * normalized;
    }

    float xToFrequency (float x) const
    {
        const float minFreq = 20.0f, maxFreq = 20000.0f;
        float normalized = x / static_cast<float> (getWidth());
        return minFreq * std::pow (maxFreq / minFreq, normalized);
    }

    float dBToY (float dB) const
    {
        float normalized = (dB - mindB) / (maxdB - mindB);
        return static_cast<float> (getHeight()) * (1.0f - normalized);
    }

    float yTodB (float y) const
    {
        float normalized = 1.0f - (y / static_cast<float> (getHeight()));
        return mindB + normalized * (maxdB - mindB);
    }

    //==========================================================================
    // Shape mapping — INTERACTION ONLY (see the EQFilterType doc).
    // The response curve never goes through this: it dispatches on the raw
    // shape int to the matching filter class instead.
    //==========================================================================
    EQFilterType shapeToFilterType (int shape) const
    {
        if (config.hasBandPass)
        {
            // Output/speaker EQ: 0=Off 1=LowCut 2=LowShelf 3=Peak 4=BandPass
            //                    5=HighShelf 6=HighCut 7=AllPass
            switch (shape)
            {
                case 0: return EQFilterType::Off;
                case 1: return EQFilterType::LowCut;
                case 2: return EQFilterType::LowShelf;
                case 3: return EQFilterType::PeakNotch;
                case 4: return EQFilterType::BandPass;
                case 5: return EQFilterType::HighShelf;
                case 6: return EQFilterType::HighCut;
                case 7: return EQFilterType::AllPass;
                default: return EQFilterType::Off;
            }
        }

        // Reverb pre/post EQ: 0=Off 1=LowCut 2=LowShelf 3=Peak 4=HighShelf
        //                     5=HighCut 6=BandPass
        switch (shape)
        {
            case 0: return EQFilterType::Off;
            case 1: return EQFilterType::LowCut;
            case 2: return EQFilterType::LowShelf;
            case 3: return EQFilterType::PeakNotch;
            case 4: return EQFilterType::HighShelf;
            case 5: return EQFilterType::HighCut;
            case 6: return EQFilterType::BandPass;
            default: return EQFilterType::Off;
        }
    }

    //==========================================================================
    // Response math — designed by the very filter class the audio path runs
    //==========================================================================
    void updateBandCoefficients (int bandIndex)
    {
        if (bandIndex < 0 || bandIndex >= numBands)
            return;

        auto& coeffs = bandCoefficients[static_cast<size_t> (bandIndex)];
        coeffs = {};   // default-constructed == inactive, flat 0 dB

        auto bandTree = eqTree.getChild (bandIndex);
        if (! bandTree.isValid())
            return;

        // The RAW shape int, in the encoding of the filter it is handed to.
        const int   shape  = bandTree.getProperty (config.shapeId);
        const float freq   = juce::jlimit (20.0f, 20000.0f, (float) bandTree.getProperty (config.frequencyId));
        const float gainDb = bandTree.getProperty (config.gainId);
        const float q      = juce::jlimit (config.qMin, config.qMax, (float) bandTree.getProperty (config.qId));

        // No slope parameter in this config => Q doubles as the shelf S,
        // preserving the legacy WFS-DIY shelf behaviour.
        const float slope = config.slopeId.isValid()
                                ? juce::jlimit (config.slopeMin, config.slopeMax,
                                                (float) bandTree.getProperty (config.slopeId))
                                : q;

        coeffs = config.hasBandPass
                     ? spatcore::dsp::OutputEQBiquadFilter::calculateCoefficients (
                           shape, freq, gainDb, q, slope, sampleRate)
                     : spatcore::dsp::ReverbBiquadFilter::calculateCoefficients (
                           shape, freq, gainDb, q, slope, sampleRate);
    }

    void updateAllCoefficients()
    {
        for (int i = 0; i < numBands; ++i)
            updateBandCoefficients (i);
    }

    float calculateTotalResponse (float frequency) const
    {
        float totalGaindB = 0.0f;

        for (const auto& coeffs : bandCoefficients)
            totalGaindB += spatcore::dsp::biquadMagnitudeDb (coeffs, frequency, sampleRate);

        return juce::jlimit (mindB - 6.0f, maxdB + 6.0f, totalGaindB);
    }

    //==========================================================================
    // Band marker positioning and hit testing
    //==========================================================================
    juce::Point<float> getBandMarkerPosition (int bandIndex) const
    {
        auto bandTree = eqTree.getChild (bandIndex);
        if (! bandTree.isValid())
            return { 0.0f, 0.0f };

        int shape = bandTree.getProperty (config.shapeId);
        float freq = bandTree.getProperty (config.frequencyId);
        float gain = bandTree.getProperty (config.gainId);

        float x = frequencyToX (freq);
        EQFilterType filterType = shapeToFilterType (shape);

        // Cuts, band-pass and all-pass park on the 0 dB line.
        float y = hasGainControl (filterType) ? dBToY (gain) : dBToY (0.0f);
        return { x, y };
    }

    /** Marker position including OFF bands, which sit at their stored gain. */
    juce::Point<float> markerPositionIncludingOff (int band) const
    {
        auto bandTree = eqTree.getChild (band);
        if (! bandTree.isValid())
            return { -1000.0f, -1000.0f };   // far outside any hit radius

        int shape = bandTree.getProperty (config.shapeId);
        float freq = bandTree.getProperty (config.frequencyId);
        float x = frequencyToX (freq);
        float y;

        if (shape == 0)
        {
            float gain = bandTree.getProperty (config.gainId);
            y = dBToY (gain);
        }
        else
        {
            y = getBandMarkerPosition (band).y;
        }

        return { x, y };
    }

    int findBandAtPosition (juce::Point<float> pos) const
    {
        const float hitRadius = 15.0f * paintScale();
        const float hitRadiusSq = hitRadius * hitRadius;

        for (int band = 0; band < numBands; ++band)
        {
            auto p = markerPositionIncludingOff (band);
            float dx = pos.x - p.x, dy = pos.y - p.y;
            if (dx * dx + dy * dy < hitRadiusSq)
                return band;
        }

        return -1;
    }

    enum class DragMode { None, Both, FrequencyOnly, GainOnly };

    DragMode findCrosshairAtPosition (juce::Point<float> pos) const
    {
        if (selectedBand < 0)
            return DragMode::None;

        auto bandTree = eqTree.getChild (selectedBand);
        if (! bandTree.isValid())
            return DragMode::None;

        int shape = bandTree.getProperty (config.shapeId);
        bool isOff = (shape == 0);
        auto marker = markerPositionIncludingOff (selectedBand);

        const float hitTolerance = 8.0f * paintScale();
        EQFilterType filterType = shapeToFilterType (shape);

        // Vertical crosshair (frequency) — always available.
        if (std::abs (pos.x - marker.x) < hitTolerance)
            return DragMode::FrequencyOnly;

        // Horizontal crosshair (gain) — off bands always, active bands only
        // when the shape has a gain control.
        if ((isOff || hasGainControl (filterType)) && std::abs (pos.y - marker.y) < hitTolerance)
            return DragMode::GainOnly;

        return DragMode::None;
    }

    //==========================================================================
    // Multitouch helpers
    //==========================================================================
    float getTouchDistance() const
    {
        if (activeTouches.size() < 2)
            return 0.0f;

        auto it = activeTouches.begin();
        auto pos1 = it->second.position;
        ++it;
        return pos1.getDistanceFrom (it->second.position);
    }

    juce::Point<float> getTouchMidpoint() const
    {
        if (activeTouches.size() < 2)
            return { 0.0f, 0.0f };

        auto it = activeTouches.begin();
        auto pos1 = it->second.position;
        ++it;
        auto pos2 = it->second.position;
        return { (pos1.x + pos2.x) * 0.5f, (pos1.y + pos2.y) * 0.5f };
    }

    /** Band nearest a point, for pinch targeting; -1 if none is close. */
    int findBandNearestToPoint (juce::Point<float> pos) const
    {
        int nearestBand = -1;
        float nearestDistSq = (std::numeric_limits<float>::max)();
        const float maxSearchRadius = 150.0f * paintScale();
        const float maxSearchRadiusSq = maxSearchRadius * maxSearchRadius;

        for (int band = 0; band < numBands; ++band)
        {
            auto p = markerPositionIncludingOff (band);
            float dx = pos.x - p.x, dy = pos.y - p.y;
            float distSq = dx * dx + dy * dy;

            if (distSq < nearestDistSq && distSq < maxSearchRadiusSq)
            {
                nearestDistSq = distSq;
                nearestBand = band;
            }
        }

        return nearestBand;
    }

    //==========================================================================
    /** THE single mutation path — every gesture funnels through here so the
        two persistence modes can never diverge. See the class doc.
    */
    void setBandParameter (int bandIndex, const juce::Identifier& paramId, const juce::var& value)
    {
        auto bandTree = eqTree.getChild (bandIndex);
        if (! bandTree.isValid())
            return;

        if (ownerPersistsValue)
        {
            // Delegate the write entirely. Pre-writing here would zero an
            // owner's relative-mode delta (it would read old == new), so
            // array members would not move; and it would bypass the owner's
            // own scoped-undo seam. The owner's write lands on this same live
            // node, so our listener still repaints.
            if (onParameterChanged)
                onParameterChanged (bandIndex, paramId, value);
        }
        else
        {
            // Write ourselves, with undo when a manager has been installed,
            // then notify (e.g. for array propagation).
            bandTree.setProperty (paramId, value, undoManagerPtr);

            if (onParameterChanged)
                onParameterChanged (bandIndex, paramId, value);
        }

        // Belt and braces: the ValueTree listener normally does this, but a
        // delegating owner may persist somewhere that does not round-trip
        // immediately, and recomputing one band is cheap and idempotent.
        updateBandCoefficients (bandIndex);
        repaint();
    }

    //==========================================================================
    // State
    //==========================================================================
    juce::ValueTree eqTree;
    int numBands;
    EQDisplayConfig config;

    float mindB = -24.0f;
    float maxdB = 24.0f;
    double sampleRate = 48000.0;
    bool eqEnabled = true;

    int selectedBand = -1;
    bool isDragging = false;
    DragMode dragMode = DragMode::None;
    juce::Point<float> dragStartPos;
    float dragStartFreq = 0.0f;   // original freq at crosshair drag start
    float dragStartGain = 0.0f;   // original gain at crosshair drag start

    std::vector<spatcore::dsp::BiquadCoefficients> bandCoefficients;

    // Multitouch tracking
    struct TouchInfo
    {
        juce::Point<float> position;
        juce::Point<float> startPosition;
    };
    std::map<int, TouchInfo> activeTouches;
    bool isPinching = false;
    float pinchStartDistance = 0.0f;
    float pinchStartQ = 0.0f;

    // Undo support
    juce::UndoManager* undoManagerPtr = nullptr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (EQDisplayComponent)
};

} // namespace spatcore::ui

// NOTE: deliberately NO global extraction-compat aliases here.
//
// The dsp/ headers emit them because app code that predates the extraction
// still names those classes unqualified and must keep compiling verbatim.
// ui/ is new code with no such history: XOA names these types qualified, and
// WFS-DIY reaches them through its own Source/gui/EQDisplayComponent.h compat
// shim, which owns the unqualified names (and layers app-specific
// EQDisplayConfig factories on top). Exporting them into the global namespace
// from here would collide with exactly that shim.
