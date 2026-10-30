#pragma once

#if defined (__APPLE__)

/**
    Multitouch from external USB touch panels on macOS.

    macOS flattens an external digitizer to a single emulated pointer, so this
    reads the panel's own HID reports with IOHIDManager and hands every finger
    to JUCE as a touch MouseInputSource (through TouchInjector), the way iOS,
    Windows and Linux deliver them. Components that handle touch need no change.

    - Any Windows-style multitouch screen (usage page 0x0D, usage 0x04) is
      picked up; its report layout comes from its own descriptor
      (hid/HidDigitizerDescriptor.h), not from a list of known panels.
    - On every connection the panel is switched out of mouse-emulation mode:
      its Contact Count Maximum feature report is read (the handshake Windows
      performs) and, where the panel has one, its Input Mode is set to 2. The
      16" ILITEK panel sends no multitouch report at all until that read.
      The switch waits until no mouse button is down: switching mid-press
      leaves the emulated left button stuck down in macOS.
    - In multitouch mode macOS stops using the panel as a pointer, and the
      panel would stay in that mode until it is replugged: on quit, every
      panel that was switched is re-enumerated on USB (a software replug), so
      it comes back as the touchscreen-mouse it was. Not after a crash or a
      force-quit; a replug does it then. Fingers that land on one
      of the app's windows (the frontmost window at that point) go to JUCE as
      touches; with drivePointer, the first finger elsewhere moves and clicks
      the system cursor, so the rest of the desktop stays usable.
    - Each panel is mapped to one display: its full range onto the display's
      full bounds, refined by a TouchCalibration. The display bounds are read
      at every event, so a change of resolution or scaling needs nothing.
    - "Seize" opens the panel - and the mouse-emulation interfaces of the same
      USB device - exclusively, so macOS stops moving the cursor and stops the
      firmware's long-press right-click. Released on disable and on quit.
    - Hot-plug: a panel that goes away lifts all its fingers at once (no stuck
      touches); one that comes back gets its settings and seize state again.
    - Settings are kept per panel in a PropertiesFile, keyed by vendor ID,
      product ID and USB LocationID: identical panels report no serial number,
      so the port is what tells them apart. A panel moved to another port
      comes back as a new, unassigned panel.

    Reading needs the Input Monitoring permission. Without it the device opens
    but macOS withholds every report, so the manager asks once (the system
    prompt) and reports the state; a permission granted later takes effect
    after the app is relaunched.

    Threading: IOKit runs on a dedicated thread with its own run loop, never on
    the audio thread. Contacts reach the message thread through
    MessageManager::callAsync; everything public here is message-thread only.
*/

#include <juce_data_structures/juce_data_structures.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include "../TouchCalibration.h"
#include "../TouchInjector.h"
#include "../hid/HidContactTracker.h"

#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace spatcore::controllers {

class MacHidTouchManager
{
public:
    /** @param settingsAppName  Application name for the per-panel settings
        file (PropertiesFile app and folder name), as for EvdevTouchManager. */
    explicit MacHidTouchManager (const juce::String& settingsAppName);
    ~MacHidTouchManager();

    MacHidTouchManager (const MacHidTouchManager&) = delete;
    MacHidTouchManager& operator= (const MacHidTouchManager&) = delete;

    struct PanelSettings
    {
        /** The display the panel covers, as DisplayInfo::id. Empty for
            automatic: the only display when there is exactly one, otherwise
            none - the panel is read but its touches go nowhere until it is
            assigned (identifyDisplays(), or a display picker). */
        juce::String displayId;

        /** Settings saved before displayId existed: an index into
            Desktop::getDisplays().displays, used while displayId is empty. */
        int  displayIndex = -1;
        bool seize = false;

        /** In multitouch mode macOS no longer uses the panel as a pointer at
            all. With this on, a finger that lands outside the app's windows
            moves and clicks the system cursor instead (one finger at a time;
            needs the Accessibility permission). */
        bool drivePointer = true;

        TouchCalibration calibration;

        juce::var toVar() const;
        static PanelSettings fromVar (const juce::var&);
    };

    struct PanelInfo
    {
        juce::String key;            // "vvvv:pppp@llllllll" (hex VID, PID, LocationID)
        juce::String name;           // product string
        int  vendorId = 0, productId = 0;
        juce::uint32 locationId = 0;
        bool connected = false;      // false: known from settings only
        bool reading = false;        // opened, multitouch layout understood
        bool seized = false;
        int  declaredMaxContacts = 0;
        int  maxContactsSeen = 0;    // this session; logged to catch the 2-contact anomaly
        juce::String problem;        // empty when all is well
        PanelSettings settings;
    };

    /** Connected panels first, then panels known only from settings. */
    std::vector<PanelInfo> getPanels() const;

    /** One display, as a panel is assigned to it. The id survives a change of
        arrangement, resolution or the order macOS lists displays in: vendor,
        model and serial number from the display's EDID, plus its unit number
        when it has no serial (two identical monitors). */
    struct DisplayInfo
    {
        juce::String id;
        juce::String name;              // as macOS names it, e.g. "PLT2236"
        juce::Rectangle<int> bounds;    // JUCE desktop coordinates, logical points
        bool isMain = false;
    };
    static std::vector<DisplayInfo> getDisplays();

    /** True when touches of a panel with these settings land somewhere now. */
    bool hasDisplay (const PanelSettings&) const;

    /** While set, a finger that lands is not routed anywhere: it is reported
        here instead, with its panel and its position as a fraction of the
        panel (before calibration), and ignored until it lifts. The app shows
        a target in a different place on each display; where the finger lands
        on its panel says which display that panel covers. */
    std::function<void (const juce::String& panelKey, float x, float y)> onIdentifyTouch;

    void setPanelSettings (const juce::String& key, const PanelSettings&);

    enum class Access { granted, denied, unknown };
    static Access getInputMonitoringAccess();

    /** Synthetic cursor events (drivePointer) need the Accessibility permission. */
    static bool hasAccessibilityAccess();

    /** True once a finger was injected and JUCE dropped it: the JUCE build
        lacks spatcore's canUseTouch() patch. */
    bool isTouchDropped() const noexcept   { return injector.isTouchDropped(); }

    /** Called on the message thread when panels come, go or change state. */
    using ChangeCallback = std::function<void()>;
    int  addChangeListener (ChangeCallback);
    void removeChangeListener (int token);

    /** Diagnostics for the app's log (message thread). */
    std::function<void (const juce::String&)> onLog;

    /** Logs every finger's landing and lift: where it was on the desktop, the
        component under it, where it went and how many moves came between. */
    bool traceContacts = false;

private:
    class HidThread;
    friend class HidThread;

    // Called from the HID thread through callAsync.
    void drainContacts();
    void handleContacts (const juce::String& key, const std::vector<hid::ContactEvent>& events);
    void handlePanelState (const PanelInfo& info);
    void handlePanelGone (const juce::String& key, int maxContactsSeen);

    bool desktopPosition (const PanelSettings&, const hid::ContactEvent&, juce::Point<float>& out) const;
    bool displayArea (const PanelSettings&, juce::Rectangle<float>& out) const;

    // getDisplays() asks AppKit; touch events come 100 times a second, so the
    // list is cached and re-read at most every half second.
    mutable std::vector<DisplayInfo> cachedDisplays;
    mutable juce::uint32 cachedDisplaysTime = 0;
    /** What is frontmost at a desktop position: one of the app's JUCE views,
        the frame of one of its windows (the macOS title bar, its buttons,
        the resize edges - drawn by AppKit, not JUCE), or anything else. */
    enum class Hit { appContent, appFrame, elsewhere };
    static Hit hitTest (juce::Point<float> desktopPos);

    // The system cursor, driven by one finger outside the app's windows.
    void pointerDown (juce::Point<float>);
    void pointerMove (juce::Point<float>);
    void pointerUp   (juce::Point<float>);

    struct Trace { juce::Point<float> start, last; int moves = 0; };
    std::map<std::pair<juce::String, int>, Trace> traces;

    // A finger on the frame of one of the app's own windows, handled through
    // AppKit and JUCE directly (no permission needed for the app's own
    // windows): the close, minimise and zoom buttons fire on lift if the
    // finger is still on them; elsewhere on the title bar it drags the window;
    // on a left, right or bottom edge it resizes it. The edge zone is finger
    // sized - 14 points outside the window, 4 inside - because the macOS one
    // is a few points wide and sits on the border, out of reach of a finger.
    enum Edge { leftEdge = 1, rightEdge = 2, bottomEdge = 4 };
    struct FrameGesture
    {
        long windowNumber = 0;
        int  button = -1;                    // NSWindowButton, or -1
        int  edges = 0;                      // Edge flags for a resize, 0 for a drag
        juce::Point<float> start;            // desktop
        double originX = 0, originY = 0;     // window frame origin at the start (AppKit)
        juce::Component::SafePointer<juce::Component> window;   // JUCE top level, for a resize
        juce::Rectangle<int> startBounds;
    };
    std::map<std::pair<juce::String, int>, FrameGesture> frameGestures;
    bool edgeDown  (const std::pair<juce::String, int>&, juce::Point<float>);
    bool frameDown (const std::pair<juce::String, int>&, juce::Point<float>);
    void frameMove (const FrameGesture&, juce::Point<float>);
    void frameUp   (const FrameGesture&, juce::Point<float>);

    enum class Route { injector, frame, pointer, ignored };
    std::map<std::pair<juce::String, int>, Route> routes;   // fingers down, by (panel, contact ID)
    bool pointerFingerDown = false;
    double lastPointerUpTime = 0;
    juce::Point<float> lastPointerUpPos, lastPointerPos;
    int pointerClickCount = 0;
    bool askedForAccessibility = false;
    bool warnedTouchDropped = false;
    void loadSettings();
    void saveSettings();
    void fireChange();
    void log (const juce::String&);

    std::unique_ptr<juce::PropertiesFile> settingsFile;
    std::map<juce::String, PanelSettings> settings;   // by panel key, saved
    std::map<juce::String, PanelInfo> panels;         // by panel key, connected now
    TouchInjector injector;

    std::map<int, ChangeCallback> listeners;
    int nextListenerToken = 0;

    std::unique_ptr<HidThread> hidThread;

    JUCE_DECLARE_WEAK_REFERENCEABLE (MacHidTouchManager)
};

} // namespace spatcore::controllers

#endif
