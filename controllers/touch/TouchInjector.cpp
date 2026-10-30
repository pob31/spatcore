#include "TouchInjector.h"

#include <algorithm>

namespace spatcore::controllers {

TouchInjector::~TouchInjector()
{
    releaseAll();
}

void TouchInjector::touchDown (const juce::String& deviceKey, int contactId, juce::Point<float> desktopPos)
{
    JUCE_ASSERT_MESSAGE_THREAD

    const Key key { deviceKey, contactId };

    // A down for a finger already down means its lift was lost: finish it first.
    if (auto it = fingers.find (key); it != fingers.end())
        touchUp (deviceKey, contactId, it->second.lastPos);

    int index = -1;
    for (int i = 0; i < maxFingers; ++i)
        if (! indexInUse[(size_t) i]) { index = i; break; }

    if (index < 0)
        return;   // more fingers than JUCE can track: this one is ignored

    auto* leaf = juce::Desktop::getInstance().findComponentAt (desktopPos.roundToInt());
    auto* top  = leaf != nullptr ? leaf->getTopLevelComponent() : nullptr;

    indexInUse[(size_t) index] = true;
    auto& f = fingers[key];
    f.touchIndex = index;
    f.target     = top;
    f.lastPos    = desktopPos;

    send (f, desktopPos, false);   // enter, as JUCE's iOS peer does
    send (f, desktopPos, true);

    if (! checkedOnce && top != nullptr && top->getPeer() != nullptr)
    {
        checkedOnce = true;
        bool found = false;
        for (auto& source : juce::Desktop::getInstance().getMouseSources())
            if (source.isTouch() && source.getIndex() == index)
                found = true;
        touchDropped = ! found;
    }
}

void TouchInjector::touchMove (const juce::String& deviceKey, int contactId, juce::Point<float> desktopPos)
{
    JUCE_ASSERT_MESSAGE_THREAD

    if (auto it = fingers.find ({ deviceKey, contactId }); it != fingers.end())
    {
        it->second.lastPos = desktopPos;
        send (it->second, desktopPos, true);
    }
}

void TouchInjector::touchUp (const juce::String& deviceKey, int contactId, juce::Point<float> desktopPos)
{
    JUCE_ASSERT_MESSAGE_THREAD

    if (auto it = fingers.find ({ deviceKey, contactId }); it != fingers.end())
    {
        lift (it->second, desktopPos);
        fingers.erase (it);
    }
}

void TouchInjector::releaseDevice (const juce::String& deviceKey)
{
    JUCE_ASSERT_MESSAGE_THREAD

    for (auto it = fingers.begin(); it != fingers.end();)
    {
        if (it->first.first == deviceKey)
        {
            lift (it->second, it->second.lastPos);
            it = fingers.erase (it);
        }
        else
        {
            ++it;
        }
    }
}

void TouchInjector::releaseAll()
{
    if (fingers.empty())
        return;

    for (auto& [key, f] : fingers)
        lift (f, f.lastPos);
    fingers.clear();

    // Each lift above still saw the others down.
    juce::ModifierKeys::currentModifiers = juce::ModifierKeys::currentModifiers.withoutMouseButtons();
}

void TouchInjector::lift (Finger& f, juce::Point<float> desktopPos)
{
    send (f, desktopPos, false);
    send (f, desktopPos, false, true);

    if (juce::isPositiveAndBelow (f.touchIndex, maxFingers))
        indexInUse[(size_t) f.touchIndex] = false;
}

void TouchInjector::send (Finger& f, juce::Point<float> desktopPos, bool buttonDown, bool offscreen)
{
    auto* top  = f.target.get();
    auto* peer = top != nullptr ? top->getPeer() : nullptr;
    if (peer == nullptr)
        return;   // landed outside the app's windows, or the window closed

    const auto keys = juce::ModifierKeys::currentModifiers.withoutMouseButtons();
    const auto mods = buttonDown ? keys.withFlags (juce::ModifierKeys::leftButtonModifier) : keys;

    // The global button state too, as JUCE's Windows touch handler sets it:
    // PopupMenu picks an item when that state goes from down to up, and reads
    // nothing else - without this a tapped ComboBox item is never chosen. The
    // button stays down while any other finger is.
    const bool anotherFingerDown = std::any_of (fingers.begin(), fingers.end(),
                                                [&f] (const auto& kv) { return &kv.second != &f; });
    juce::ModifierKeys::currentModifiers = buttonDown || anotherFingerDown
                                              ? keys.withFlags (juce::ModifierKeys::leftButtonModifier)
                                              : keys;

    peer->handleMouseEvent (juce::MouseInputSource::InputSourceType::touch,
                            offscreen ? juce::MouseInputSource::offscreenMousePos
                                      : peer->globalToLocal (desktopPos),
                            mods,
                            buttonDown ? juce::MouseInputSource::defaultPressure : 0.0f,
                            juce::MouseInputSource::defaultOrientation,
                            juce::Time::currentTimeMillis(), {}, f.touchIndex);
}

} // namespace spatcore::controllers
