#pragma once

/**
    Delivers fingers from a touch panel read by the app itself into JUCE, as
    touch MouseInputSources - one per finger - so components that already
    handle touch from iOS, Windows or Linux (MouseEvent::source.isTouch(),
    getIndex()) work unchanged.

    Platform-agnostic; the device layer (macos/MacHidTouchManager on macOS)
    turns panel contacts into desktop positions and calls this.

    - A finger goes to the window under it when it lands
      (Desktop::findComponentAt), and stays with that window until it lifts,
      so a drag that leaves the window keeps its target, and detached windows
      get their own fingers.
    - Each finger gets the lowest free JUCE touch index for as long as it is
      down. Contact IDs are namespaced per device: two panels both report IDs
      0-9.
    - Events follow the sequence JUCE's own iOS peer sends: a button-less
      event where the finger lands (so it enters the component), the press,
      drags, the release, then a move off screen.

    On macOS this needs JUCE's canUseTouch() to return true (the patch in
    spatcore/patches/juce/macos/); without it JUCE drops every touch event
    without a trace. isTouchDropped() becomes true after the first finger if
    that happened, so the app can say why the panel does nothing.

    Message thread only.
*/

#include <juce_gui_basics/juce_gui_basics.h>

#include <array>
#include <map>

namespace spatcore::controllers {

class TouchInjector
{
public:
    TouchInjector() = default;
    ~TouchInjector();

    TouchInjector (const TouchInjector&) = delete;
    TouchInjector& operator= (const TouchInjector&) = delete;

    /** A finger landed at `desktopPos` (JUCE desktop coordinates, logical points). */
    void touchDown (const juce::String& deviceKey, int contactId, juce::Point<float> desktopPos);
    void touchMove (const juce::String& deviceKey, int contactId, juce::Point<float> desktopPos);
    void touchUp   (const juce::String& deviceKey, int contactId, juce::Point<float> desktopPos);

    /** Lifts every finger of one device where it last was: the device went away. */
    void releaseDevice (const juce::String& deviceKey);

    /** Lifts every finger of every device. */
    void releaseAll();

    int  getNumActiveFingers() const noexcept   { return (int) fingers.size(); }

    /** True once a finger was injected and JUCE made no touch source for it:
        the JUCE build lacks the canUseTouch() patch. */
    bool isTouchDropped() const noexcept        { return touchDropped; }

private:
    struct Finger
    {
        int touchIndex = -1;
        juce::WeakReference<juce::Component> target;   // top-level component the finger landed on
        juce::Point<float> lastPos;
    };

    using Key = std::pair<juce::String, int>;

    void send (Finger& f, juce::Point<float> desktopPos, bool buttonDown, bool offscreen = false);
    void lift (Finger& f, juce::Point<float> desktopPos);

    static constexpr int maxFingers = 32;   // JUCE asserts below 100

    std::map<Key, Finger> fingers;
    std::array<bool, maxFingers> indexInUse {};
    bool touchDropped = false;
    bool checkedOnce = false;
};

} // namespace spatcore::controllers
