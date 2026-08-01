#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>

namespace spatcore::ui {

//==============================================================================
/**
    Colours an EQBandToggle needs from its host application.

    The band colour itself is owned by the widget (see
    EQBandToggle::setBandColour); this struct carries only the two theme
    colours that used to be read straight from the app's colour scheme:

      offFill - fill used when the band is toggled OFF (a dark rounded rect)
      border  - outline stroked around the button in both states

    The ON fill is always the band colour, so it is deliberately absent here.
*/
struct EQBandToggleColours
{
    juce::Colour offFill, border;
};

//==============================================================================
/**
    A tiny coloured indicator button for toggling individual EQ bands on/off.

    When ON, displays a filled rounded rectangle in the band's colour.
    When OFF, displays a dark rounded rectangle. Both states are outlined
    with the border colour.

    Injected-colour seam
    --------------------
    spatcore may never name an app symbol, so the two theme colours this
    widget used to read from the application colour scheme are supplied by
    the host through the public `colourProvider` callback:

        toggle.colourProvider = []
        {
            return spatcore::ui::EQBandToggleColours {
                ColorScheme::get().sliderTrackBg,
                ColorScheme::get().buttonBorder };
        };

    The provider is invoked AT PAINT TIME, once per paintButton call, and the
    result is never cached. That is deliberate: the host applications support
    live theme switching, and re-reading the scheme on every paint is what
    makes a theme change show up on the next repaint. Providers must
    therefore be cheap - a couple of colour lookups, no allocation.

    When `colourProvider` is nullptr the widget falls back to neutral
    built-in defaults (dark grey off-fill, mid grey border), so a
    default-constructed toggle paints sensibly and can never crash.

    Threading
    ---------
    Message thread only, like any juce::Component. `colourProvider` is
    called on the message thread from within paintButton; assign it before
    the toggle becomes visible. Assigning it later is allowed but does not
    itself trigger a repaint - call repaint() if the colours have changed.

    Toggle state
    ------------
    setClickingTogglesState (true) is set in the constructor, so the button
    flips its own toggle state on click; hosts observe changes through the
    usual juce::Button::onClick / Button::Listener mechanism and read
    getToggleState().
*/
class EQBandToggle : public juce::Button
{
public:
    EQBandToggle() : juce::Button ("") { setClickingTogglesState (true); }

    //==========================================================================
    /** Sets the colour used to fill the button when the band is ON. */
    void setBandColour (juce::Colour c) { bandColour = c; repaint(); }

    /** Supplies the theme colours at paint time; nullptr => built-in
        defaults. See the class doc for the caching contract.
    */
    std::function<EQBandToggleColours()> colourProvider;

private:
    //==========================================================================
    /** Neutral fallback used when no colourProvider has been installed. */
    static EQBandToggleColours getDefaultColours()
    {
        return { juce::Colour (0xff2a2a2a), juce::Colour (0xff606060) };
    }

    juce::Colour bandColour { juce::Colours::white };

    void paintButton (juce::Graphics& g, bool /*highlighted*/, bool /*down*/) override
    {
        auto bounds = getLocalBounds().toFloat().reduced (1.0f);
        auto colours = colourProvider != nullptr ? colourProvider()
                                                 : getDefaultColours();
        bool on = getToggleState();
        g.setColour (on ? bandColour : colours.offFill);
        g.fillRoundedRectangle (bounds, 3.0f);
        g.setColour (colours.border);
        g.drawRoundedRectangle (bounds, 3.0f, 1.0f);
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (EQBandToggle)
};

} // namespace spatcore::ui

// NOTE: deliberately NO global extraction-compat aliases here — see the same
// note at the bottom of EQDisplayComponent.h. ui/ is new code, so the
// unqualified names belong to each app's own compat shim, not to spatcore.
