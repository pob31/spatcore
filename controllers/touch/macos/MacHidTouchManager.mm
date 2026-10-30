#include "MacHidTouchManager.h"

#if defined (__APPLE__)

#import <AppKit/AppKit.h>
#include <ApplicationServices/ApplicationServices.h>
#include <IOKit/hid/IOHIDManager.h>
#include <IOKit/hid/IOHIDKeys.h>
#include <IOKit/hidsystem/IOHIDLib.h>

#include <mutex>
#include <thread>

namespace spatcore::controllers {

namespace
{
    const char* const kSettingsKey = "macTouchPanels";

    long intProperty (IOHIDDeviceRef device, CFStringRef key)
    {
        long v = 0;
        if (auto n = (CFNumberRef) IOHIDDeviceGetProperty (device, key))
            if (CFGetTypeID (n) == CFNumberGetTypeID())
                CFNumberGetValue (n, kCFNumberLongType, &v);
        return v;
    }

    juce::String stringProperty (IOHIDDeviceRef device, CFStringRef key)
    {
        if (auto s = (CFStringRef) IOHIDDeviceGetProperty (device, key))
            if (CFGetTypeID (s) == CFStringGetTypeID())
                return juce::String::fromCFString (s);
        return {};
    }

    juce::String errorText (IOReturn rc)
    {
        return juce::String::toHexString ((int) rc).paddedLeft ('0', 8);
    }

    // Writes `value` into a field of a report buffer (body starts after the ID).
    void writeBits (uint8_t* body, size_t bodySize, const hid::ReportField& f, uint32_t value)
    {
        for (uint32_t i = 0; i < f.bitSize; ++i)
        {
            const uint32_t bit = f.bitOffset + i;
            if (bit / 8 >= bodySize) return;
            if ((value >> i) & 1u) body[bit / 8] |=  (uint8_t) (1u << (bit & 7));
            else                   body[bit / 8] &= (uint8_t) ~(1u << (bit & 7));
        }
    }
}

//==============================================================================
/** Owns everything IOKit: the manager, the devices and their reports. Lives
    on its own thread and run loop; reaches the manager only through callAsync,
    and is reached by it only through post(). */
class MacHidTouchManager::HidThread
{
public:
    HidThread (juce::WeakReference<MacHidTouchManager> ownerRef, std::map<juce::String, bool> seizeByKey)
        : owner (std::move (ownerRef)), seizeWanted (std::move (seizeByKey))
    {
        juce::WaitableEvent ready;
        thread = std::thread ([this, &ready] { run (ready); });
        ready.wait();
    }

    ~HidThread()
    {
        post ([this]
        {
            for (auto& d : devices)
                closeDevice (*d);
            devices.clear();
            CFRunLoopStop (CFRunLoopGetCurrent());
        });
        thread.join();
    }

    void post (std::function<void()> fn)
    {
        {
            const std::lock_guard<std::mutex> lock (commandLock);
            commands.push_back (std::move (fn));
        }
        CFRunLoopSourceSignal (commandSource);
        CFRunLoopWakeUp (runLoop);
    }

    void setSeize (const juce::String& key, bool seize)
    {
        post ([this, key, seize]
        {
            seizeWanted[key] = seize;
            for (auto& d : devices)
                if (d->key == key && d->seized != seize)
                {
                    closeDevice (*d);
                    openDevice (*d);
                }
        });
    }

private:
    struct Device
    {
        HidThread* thread = nullptr;
        IOHIDDeviceRef ref = nullptr;
        juce::String key, name;
        int vendorId = 0, productId = 0;
        juce::uint32 locationId = 0;
        hid::DigitizerLayout layout;
        std::map<uint8_t, std::unique_ptr<hid::ContactTracker>> trackers;
        std::vector<uint8_t> buffer;
        std::vector<hid::ContactEvent> events;
        bool open = false, seized = false, switched = false;
        int declaredMaxContacts = 0;
        CFRunLoopTimerRef switchTimer = nullptr;
        juce::String problem;
    };

    void run (juce::WaitableEvent& ready)
    {
        runLoop = CFRunLoopGetCurrent();

        CFRunLoopSourceContext context {};
        context.info = this;
        context.perform = [] (void* info) { static_cast<HidThread*> (info)->drainCommands(); };
        commandSource = CFRunLoopSourceCreate (kCFAllocatorDefault, 0, &context);
        CFRunLoopAddSource (runLoop, commandSource, kCFRunLoopDefaultMode);

        manager = IOHIDManagerCreate (kCFAllocatorDefault, kIOHIDOptionsTypeNone);

        // Any device with a Touch Screen usage, not only those whose primary
        // usage it is: some panels list their mouse interface first.
        int page = 0x0D, usage = 0x04;
        CFNumberRef pageNum  = CFNumberCreate (kCFAllocatorDefault, kCFNumberIntType, &page);
        CFNumberRef usageNum = CFNumberCreate (kCFAllocatorDefault, kCFNumberIntType, &usage);
        const void* keys[]   = { CFSTR (kIOHIDDeviceUsagePageKey), CFSTR (kIOHIDDeviceUsageKey) };
        const void* values[] = { pageNum, usageNum };
        CFDictionaryRef matching = CFDictionaryCreate (kCFAllocatorDefault, keys, values, 2,
                                                       &kCFTypeDictionaryKeyCallBacks,
                                                       &kCFTypeDictionaryValueCallBacks);
        IOHIDManagerSetDeviceMatching (manager, matching);
        CFRelease (matching);
        CFRelease (pageNum);
        CFRelease (usageNum);

        IOHIDManagerRegisterDeviceMatchingCallback (manager, [] (void* ctx, IOReturn, void*, IOHIDDeviceRef d)
                                                    { static_cast<HidThread*> (ctx)->deviceAdded (d); }, this);
        IOHIDManagerRegisterDeviceRemovalCallback (manager, [] (void* ctx, IOReturn, void*, IOHIDDeviceRef d)
                                                   { static_cast<HidThread*> (ctx)->deviceRemoved (d); }, this);
        IOHIDManagerScheduleWithRunLoop (manager, runLoop, kCFRunLoopDefaultMode);

        // Devices are opened one by one (so one can be seized and another not);
        // the manager itself is opened too, without which reports were not
        // delivered on the bench.
        IOHIDManagerOpen (manager, kIOHIDOptionsTypeNone);

        ready.signal();
        CFRunLoopRun();

        IOHIDManagerUnscheduleFromRunLoop (manager, runLoop, kCFRunLoopDefaultMode);
        IOHIDManagerClose (manager, kIOHIDOptionsTypeNone);
        CFRelease (manager);
        CFRunLoopRemoveSource (runLoop, commandSource, kCFRunLoopDefaultMode);
        CFRelease (commandSource);
    }

    void drainCommands()
    {
        std::vector<std::function<void()>> todo;
        {
            const std::lock_guard<std::mutex> lock (commandLock);
            todo.swap (commands);
        }
        for (auto& fn : todo)
            fn();
    }

    //==========================================================================
    void deviceAdded (IOHIDDeviceRef ref)
    {
        for (auto& d : devices)
            if (d->ref == ref)
                return;

        auto d = std::make_unique<Device>();
        d->thread     = this;
        d->ref        = ref;
        d->vendorId   = (int) intProperty (ref, CFSTR (kIOHIDVendorIDKey));
        d->productId  = (int) intProperty (ref, CFSTR (kIOHIDProductIDKey));
        d->locationId = (juce::uint32) intProperty (ref, CFSTR (kIOHIDLocationIDKey));
        d->name       = stringProperty (ref, CFSTR (kIOHIDProductKey));
        d->key        = juce::String::toHexString (d->vendorId).paddedLeft ('0', 4) + ":"
                      + juce::String::toHexString (d->productId).paddedLeft ('0', 4) + "@"
                      + juce::String::toHexString ((juce::int64) d->locationId).paddedLeft ('0', 8);
        CFRetain (ref);

        if (auto data = (CFDataRef) IOHIDDeviceGetProperty (ref, CFSTR (kIOHIDReportDescriptorKey)))
            d->layout = hid::parseDigitizerDescriptor (CFDataGetBytePtr (data), (size_t) CFDataGetLength (data));

        if (d->layout.isUsable())
        {
            for (auto& r : d->layout.reports)
                d->trackers[r.reportId] = std::make_unique<hid::ContactTracker> (r);

            const long maxIn = intProperty (ref, CFSTR (kIOHIDMaxInputReportSizeKey));
            d->buffer.resize ((size_t) juce::jmax (64L, maxIn));
            d->events.reserve (32);
            openDevice (*d);
        }
        else
        {
            d->problem = "Not a multitouch layout this reader understands";
        }

        publish (*d);
        devices.push_back (std::move (d));
    }

    void deviceRemoved (IOHIDDeviceRef ref)
    {
        for (auto it = devices.begin(); it != devices.end(); ++it)
        {
            auto& d = **it;
            if (d.ref != ref)
                continue;

            // Lift every finger now: a panel that drops off USB must not leave
            // touches stuck down.
            d.events.clear();
            int maxSeen = 0;
            for (auto& [id, tracker] : d.trackers)
            {
                tracker->releaseAll (d.events);
                maxSeen = juce::jmax (maxSeen, tracker->getMaxContactsSeen());
            }
            sendEvents (d);

            cancelSwitchTimer (d);
            d.open = false;   // the device is gone: nothing to close
            CFRelease (d.ref);

            juce::MessageManager::callAsync ([o = owner, key = d.key, maxSeen]
            {
                if (auto* m = o.get())
                    m->handlePanelGone (key, maxSeen);
            });

            devices.erase (it);
            return;
        }
    }

    void openDevice (Device& d)
    {
        const bool seize = seizeWanted.count (d.key) > 0 && seizeWanted[d.key];
        const IOOptionBits options = seize ? kIOHIDOptionsTypeSeizeDevice : kIOHIDOptionsTypeNone;
        const IOReturn rc = IOHIDDeviceOpen (d.ref, options);

        d.open = rc == kIOReturnSuccess;
        d.seized = d.open && seize;
        d.switched = false;

        if (rc == kIOReturnNotPermitted)
            d.problem = "Input Monitoring permission needed";
        else if (rc == kIOReturnExclusiveAccess)
            d.problem = "Another application has this panel to itself";
        else if (rc != kIOReturnSuccess)
            d.problem = "Could not open the panel (IOKit error 0x" + errorText (rc) + ")";
        else
            d.problem = {};

        if (! d.open)
            return;

        IOHIDDeviceRegisterInputReportCallback (d.ref, d.buffer.data(), (CFIndex) d.buffer.size(),
            [] (void* ctx, IOReturn, void*, IOHIDReportType, uint32_t reportId, uint8_t* report, CFIndex length)
            {
                auto* dev = static_cast<Device*> (ctx);
                dev->thread->handleReport (*dev, reportId, report, length);
            },
            &d);

        trySwitchToMultitouch (d);
    }

    void closeDevice (Device& d)
    {
        cancelSwitchTimer (d);
        if (! d.open)
            return;

        IOHIDDeviceRegisterInputReportCallback (d.ref, d.buffer.data(), (CFIndex) d.buffer.size(), nullptr, nullptr);
        IOHIDDeviceClose (d.ref, d.seized ? kIOHIDOptionsTypeSeizeDevice : kIOHIDOptionsTypeNone);
        d.open = d.seized = false;
    }

    //==========================================================================
    // Leaving mouse-emulation mode. Switching while the emulated button is
    // down leaves it stuck down in macOS (bench, 2026-10-30), so wait until no
    // button is pressed, checking every 100 ms.
    void trySwitchToMultitouch (Device& d)
    {
        cancelSwitchTimer (d);

        if (CGEventSourceButtonState (kCGEventSourceStateHIDSystemState, kCGMouseButtonLeft)
            || CGEventSourceButtonState (kCGEventSourceStateHIDSystemState, kCGMouseButtonRight))
        {
            CFRunLoopTimerContext ctx {};
            ctx.info = &d;
            d.switchTimer = CFRunLoopTimerCreate (kCFAllocatorDefault, CFAbsoluteTimeGetCurrent() + 0.1, 0, 0, 0,
                [] (CFRunLoopTimerRef, void* info)
                {
                    auto* dev = static_cast<Device*> (info);
                    dev->thread->trySwitchToMultitouch (*dev);
                },
                &ctx);
            CFRunLoopAddTimer (runLoop, d.switchTimer, kCFRunLoopDefaultMode);
            return;
        }

        if (d.layout.contactCountMax.isPresent())
        {
            auto& f = d.layout.contactCountMax;
            std::vector<uint8_t> report (f.reportBits / 8 + 2, 0);
            report[0] = f.reportId;
            CFIndex length = (CFIndex) report.size();
            const IOReturn rc = IOHIDDeviceGetReport (d.ref, kIOHIDReportTypeFeature, f.reportId, report.data(), &length);
            if (rc == kIOReturnSuccess)
            {
                const size_t skip = d.layout.usesReportIds ? 1 : 0;
                d.declaredMaxContacts = f.field.read (report.data() + skip, (size_t) length - skip);
            }
        }

        if (d.layout.inputMode.isPresent())
        {
            auto& f = d.layout.inputMode;
            std::vector<uint8_t> report (f.reportBits / 8 + 2, 0);
            report[0] = f.reportId;
            CFIndex length = (CFIndex) report.size();
            IOHIDDeviceGetReport (d.ref, kIOHIDReportTypeFeature, f.reportId, report.data(), &length);
            const size_t skip = d.layout.usesReportIds ? 1 : 0;
            writeBits (report.data() + skip, report.size() - skip, f.field, 2);   // 2 = multi-input
            IOHIDDeviceSetReport (d.ref, kIOHIDReportTypeFeature, f.reportId, report.data(), length);
        }

        d.switched = true;
        publish (d);
    }

    void cancelSwitchTimer (Device& d)
    {
        if (d.switchTimer == nullptr)
            return;
        CFRunLoopTimerInvalidate (d.switchTimer);
        CFRelease (d.switchTimer);
        d.switchTimer = nullptr;
    }

    //==========================================================================
    void handleReport (Device& d, uint32_t reportId, const uint8_t* report, CFIndex length)
    {
        auto it = d.trackers.find ((uint8_t) reportId);
        if (it == d.trackers.end())
            return;

        const size_t skip = d.layout.usesReportIds ? 1 : 0;
        if ((size_t) length <= skip)
            return;

        d.events.clear();
        it->second->processReport (report + skip, (size_t) length - skip, d.events);
        sendEvents (d);
    }

    // Hands events to the message thread, merged the way an OS merges pointer
    // moves: while a batch waits to be delivered, a new move of a finger whose
    // last queued event is a move replaces that move. Downs and lifts are
    // never merged away. Without this a control that is slow to react (one
    // that relayouts a tab per value, say) falls further behind the finger at
    // every report - 100 a second on the ILITEK panel - and the app freezes
    // working through the backlog.
    void sendEvents (Device& d)
    {
        if (d.events.empty())
            return;

        bool post = false;
        {
            const std::lock_guard<std::mutex> lock (pendingLock);

            for (const auto& e : d.events)
            {
                bool merged = false;
                if (e.phase == hid::ContactEvent::Phase::move)
                {
                    for (auto it = pending.rbegin(); it != pending.rend(); ++it)
                    {
                        if (it->first != d.key || it->second.contactId != e.contactId)
                            continue;
                        if (it->second.phase == hid::ContactEvent::Phase::move)
                        {
                            it->second.x = e.x;
                            it->second.y = e.y;
                            merged = true;
                        }
                        break;
                    }
                }
                if (! merged)
                    pending.emplace_back (d.key, e);
            }

            if (! drainPosted)
                post = drainPosted = true;
        }

        if (post)
            juce::MessageManager::callAsync ([o = owner]
            {
                if (auto* m = o.get())
                    m->drainContacts();
            });
    }

public:
    /** Message thread: everything queued since the last drain, in order. */
    std::vector<std::pair<juce::String, hid::ContactEvent>> takePending()
    {
        const std::lock_guard<std::mutex> lock (pendingLock);
        drainPosted = false;
        std::vector<std::pair<juce::String, hid::ContactEvent>> out;
        out.swap (pending);
        return out;
    }

private:

    void publish (const Device& d)
    {
        PanelInfo info;
        info.key                 = d.key;
        info.name                = d.name;
        info.vendorId            = d.vendorId;
        info.productId           = d.productId;
        info.locationId          = d.locationId;
        info.connected           = true;
        info.reading             = d.open && d.layout.isUsable();
        info.seized              = d.seized;
        info.declaredMaxContacts = d.declaredMaxContacts;
        info.problem             = d.problem;

        juce::MessageManager::callAsync ([o = owner, info]
        {
            if (auto* m = o.get())
                m->handlePanelState (info);
        });
    }

    juce::WeakReference<MacHidTouchManager> owner;
    std::map<juce::String, bool> seizeWanted;   // HID thread only

    std::thread thread;
    CFRunLoopRef runLoop = nullptr;
    CFRunLoopSourceRef commandSource = nullptr;
    IOHIDManagerRef manager = nullptr;

    std::mutex commandLock;
    std::vector<std::function<void()>> commands;

    std::mutex pendingLock;
    std::vector<std::pair<juce::String, hid::ContactEvent>> pending;
    bool drainPosted = false;

    std::vector<std::unique_ptr<Device>> devices;   // HID thread only
};

//==============================================================================
juce::var MacHidTouchManager::PanelSettings::toVar() const
{
    auto* obj = new juce::DynamicObject();
    obj->setProperty ("displayIndex", displayIndex);
    obj->setProperty ("seize", seize);
    obj->setProperty ("drivePointer", drivePointer);
    juce::Array<juce::var> cal { calibration.a, calibration.b, calibration.c,
                                 calibration.d, calibration.e, calibration.f };
    obj->setProperty ("calibration", cal);
    return juce::var (obj);
}

MacHidTouchManager::PanelSettings MacHidTouchManager::PanelSettings::fromVar (const juce::var& v)
{
    PanelSettings s;
    if (auto* obj = v.getDynamicObject())
    {
        s.displayIndex = obj->hasProperty ("displayIndex") ? (int) obj->getProperty ("displayIndex") : -1;
        s.seize        = (bool) obj->getProperty ("seize");
        s.drivePointer = obj->hasProperty ("drivePointer") ? (bool) obj->getProperty ("drivePointer") : true;
        if (auto* cal = obj->getProperty ("calibration").getArray(); cal != nullptr && cal->size() == 6)
        {
            s.calibration.a = (*cal)[0]; s.calibration.b = (*cal)[1]; s.calibration.c = (*cal)[2];
            s.calibration.d = (*cal)[3]; s.calibration.e = (*cal)[4]; s.calibration.f = (*cal)[5];
        }
    }
    return s;
}

//==============================================================================
MacHidTouchManager::MacHidTouchManager (const juce::String& settingsAppName)
{
    juce::PropertiesFile::Options options;
    options.applicationName     = settingsAppName;
    options.filenameSuffix      = ".settings";
    options.osxLibrarySubFolder = "Application Support";
    options.folderName          = juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
                                       .getChildFile (settingsAppName).getFullPathName();
    settingsFile = std::make_unique<juce::PropertiesFile> (options);
    loadSettings();

    std::map<juce::String, bool> seizeByKey;
    for (auto& [key, s] : settings)
        seizeByKey[key] = s.seize;

    hidThread = std::make_unique<HidThread> (juce::WeakReference<MacHidTouchManager> (this), std::move (seizeByKey));
}

MacHidTouchManager::~MacHidTouchManager()
{
    hidThread.reset();   // closes (and un-seizes) every panel
    injector.releaseAll();
    if (pointerFingerDown)
        pointerUp (lastPointerPos);
}

MacHidTouchManager::Access MacHidTouchManager::getInputMonitoringAccess()
{
    switch (IOHIDCheckAccess (kIOHIDRequestTypeListenEvent))
    {
        case kIOHIDAccessTypeGranted: return Access::granted;
        case kIOHIDAccessTypeDenied:  return Access::denied;
        default:                      return Access::unknown;
    }
}

bool MacHidTouchManager::hasAccessibilityAccess()
{
    return AXIsProcessTrusted();
}

//==============================================================================
std::vector<MacHidTouchManager::PanelInfo> MacHidTouchManager::getPanels() const
{
    std::vector<PanelInfo> result;
    for (auto& [key, info] : panels)
        result.push_back (info);

    for (auto& [key, s] : settings)
        if (panels.count (key) == 0)
        {
            PanelInfo known;
            known.key = key;
            known.settings = s;
            result.push_back (known);
        }

    return result;
}

void MacHidTouchManager::setPanelSettings (const juce::String& key, const PanelSettings& s)
{
    JUCE_ASSERT_MESSAGE_THREAD

    settings[key] = s;
    saveSettings();

    if (auto it = panels.find (key); it != panels.end())
        it->second.settings = s;

    hidThread->setSeize (key, s.seize);
    fireChange();
}

int MacHidTouchManager::addChangeListener (ChangeCallback cb)
{
    listeners[++nextListenerToken] = std::move (cb);
    return nextListenerToken;
}

void MacHidTouchManager::removeChangeListener (int token)
{
    listeners.erase (token);
}

void MacHidTouchManager::fireChange()
{
    auto copy = listeners;   // listeners may add or remove themselves
    for (auto& [token, cb] : copy)
        if (cb) cb();
}

void MacHidTouchManager::log (const juce::String& message)
{
    if (onLog)
        onLog ("Touch panel: " + message);
}

//==============================================================================
void MacHidTouchManager::handlePanelState (const PanelInfo& incoming)
{
    auto info = incoming;
    if (auto it = settings.find (info.key); it != settings.end())
        info.settings = it->second;

    const auto previous = panels.find (info.key);
    const bool isNew = previous == panels.end();
    if (! isNew)
        info.maxContactsSeen = previous->second.maxContactsSeen;

    if (isNew || previous->second.problem != info.problem)
        log (info.name + " [" + info.key + "] "
             + (info.problem.isNotEmpty() ? info.problem
                                          : juce::String (info.seized ? "reading, seized" : "reading")));

    panels[info.key] = info;

    if (info.problem.isEmpty() && getInputMonitoringAccess() != Access::granted)
    {
        // The device opened, but without the permission macOS withholds its
        // reports. Ask once; the answer takes effect after a relaunch.
        if (IOHIDRequestAccess (kIOHIDRequestTypeListenEvent))
            log ("Input Monitoring granted");
        else
            log ("Input Monitoring is needed to read the panel: allow it in System Settings > "
                 "Privacy & Security > Input Monitoring, then relaunch");
    }

    if (info.settings.drivePointer && ! askedForAccessibility && ! hasAccessibilityAccess())
    {
        askedForAccessibility = true;
        NSDictionary* opts = @{ (__bridge NSString*) kAXTrustedCheckOptionPrompt: @YES };
        if (! AXIsProcessTrustedWithOptions ((__bridge CFDictionaryRef) opts))
            log ("Accessibility is needed for touches outside the app to move the cursor: allow it in "
                 "System Settings > Privacy & Security > Accessibility");
    }

    fireChange();
}

void MacHidTouchManager::handlePanelGone (const juce::String& key, int maxContactsSeen)
{
    if (auto it = panels.find (key); it != panels.end())
    {
        log (it->second.name + " [" + key + "] disconnected; most contacts at once this session: "
             + juce::String (juce::jmax (maxContactsSeen, it->second.maxContactsSeen)));
        panels.erase (it);
    }

    injector.releaseDevice (key);
    for (auto it = routes.begin(); it != routes.end();)
        it = it->first.first == key ? routes.erase (it) : std::next (it);
    for (auto it = frameGestures.begin(); it != frameGestures.end();)
        it = it->first.first == key ? frameGestures.erase (it) : std::next (it);

    fireChange();
}

//==============================================================================
bool MacHidTouchManager::desktopPosition (const PanelSettings& s, const hid::ContactEvent& e,
                                          juce::Point<float>& out) const
{
    const auto& displays = juce::Desktop::getInstance().getDisplays().displays;

    int index = s.displayIndex;
    if (index < 0)
        index = displays.size() == 1 ? 0 : -1;
    if (! juce::isPositiveAndBelow (index, displays.size()))
        return false;

    // Logical points, never native pixels; re-read at every event, so a change
    // of resolution or scaling is followed at once.
    const auto area = displays.getReference (index).totalArea.toFloat();

    float dx = 0, dy = 0;
    s.calibration.apply (e.x, e.y, dx, dy);
    out = { area.getX() + dx * area.getWidth(), area.getY() + dy * area.getHeight() };
    return true;
}

MacHidTouchManager::Hit MacHidTouchManager::hitTest (juce::Point<float> desktopPos)
{
    // JUCE's desktop is top-left based, AppKit's screen space bottom-left
    // based on the primary screen; both in points once JUCE's global scale is
    // undone.
    const auto scale = juce::Desktop::getInstance().getGlobalScaleFactor();
    NSScreen* primary = [[NSScreen screens] firstObject];
    if (primary == nil)
        return Hit::elsewhere;

    const NSPoint p = NSMakePoint (desktopPos.x * scale, primary.frame.size.height - desktopPos.y * scale);
    const NSInteger number = [NSWindow windowNumberAtPoint: p belowWindowWithWindowNumber: 0];

    for (int i = 0; i < juce::ComponentPeer::getNumPeers(); ++i)
        if (auto* peer = juce::ComponentPeer::getPeer (i))
            if (auto* view = (__bridge NSView*) peer->getNativeHandle())
                if (view.window != nil && view.window.windowNumber == number)
                {
                    const NSRect inWindow = [view convertRect: view.bounds toView: nil];
                    const NSRect onScreen = [view.window convertRectToScreen: inWindow];
                    return NSPointInRect (p, onScreen) ? Hit::appContent : Hit::appFrame;
                }

    return Hit::elsewhere;
}

void MacHidTouchManager::drainContacts()
{
    if (hidThread == nullptr)
        return;

    const auto batch = hidThread->takePending();

    // Consecutive events of one panel go through together.
    std::vector<hid::ContactEvent> run;
    for (size_t i = 0; i < batch.size(); ++i)
    {
        run.push_back (batch[i].second);
        if (i + 1 == batch.size() || batch[i + 1].first != batch[i].first)
        {
            handleContacts (batch[i].first, run);
            run.clear();
        }
    }
}

void MacHidTouchManager::handleContacts (const juce::String& key, const std::vector<hid::ContactEvent>& events)
{
    PanelSettings s;
    if (auto it = settings.find (key); it != settings.end())
        s = it->second;

    int active = 0;

    for (const auto& e : events)
    {
        const std::pair<juce::String, int> finger { key, e.contactId };
        juce::Point<float> pos;
        const bool mapped = desktopPosition (s, e, pos);

        if (e.phase == hid::ContactEvent::Phase::down)
        {
            Route route = Route::ignored;
            if (mapped)
            {
                // A window's title bar and buttons are AppKit's: JUCE never
                // sees them, so a finger there drives the cursor like one on
                // another application.
                const auto hit = hitTest (pos);
                if (edgeDown (finger, pos))
                    route = Route::frame;
                else if (hit == Hit::appContent)
                    route = Route::injector;
                else if (hit == Hit::appFrame && frameDown (finger, pos))
                    route = Route::frame;
                else if (s.drivePointer && ! pointerFingerDown)
                    route = Route::pointer;
            }

            routes[finger] = route;

            if (traceContacts)
            {
                traces[finger] = { pos, pos, 0 };
                auto* c = mapped ? juce::Desktop::getInstance().findComponentAt (pos.roundToInt()) : nullptr;
                log ("trace down #" + juce::String (e.contactId) + " panel ("
                     + juce::String (e.x, 4) + ", " + juce::String (e.y, 4) + ") desktop "
                     + pos.toString() + " -> "
                     + (route == Route::injector ? "app" : route == Route::frame ? "window frame"
                        : route == Route::pointer ? "cursor" : "ignored")
                     + (c != nullptr ? " on " + juce::String (typeid (*c).name()) + " '" + c->getName() + "'" : juce::String()));
            }
            if (route == Route::injector)      injector.touchDown (key, e.contactId, pos);
            else if (route == Route::pointer)  pointerDown (pos);
            continue;
        }

        auto it = routes.find (finger);
        if (it == routes.end())
            continue;

        if (traceContacts)
            if (auto t = traces.find (finger); t != traces.end())
            {
                if (mapped) t->second.last = pos;
                if (e.phase == hid::ContactEvent::Phase::move)
                    ++t->second.moves;
                else
                {
                    log ("trace up   #" + juce::String (e.contactId) + " after " + juce::String (t->second.moves)
                         + " moves, from " + t->second.start.toString() + " to " + t->second.last.toString());
                    traces.erase (t);
                }
            }

        if (e.phase == hid::ContactEvent::Phase::move)
        {
            if (! mapped) continue;
            if (it->second == Route::injector)      injector.touchMove (key, e.contactId, pos);
            else if (it->second == Route::pointer)  pointerMove (pos);
            else if (it->second == Route::frame)
                if (auto g = frameGestures.find (finger); g != frameGestures.end())
                    frameMove (g->second, pos);
        }
        else
        {
            if (it->second == Route::injector)
            {
                if (mapped) injector.touchUp (key, e.contactId, pos);
                else        injector.releaseDevice (key);   // display gone: lift where it was
            }
            else if (it->second == Route::pointer)
            {
                pointerUp (mapped ? pos : lastPointerPos);
            }
            else if (it->second == Route::frame)
            {
                if (auto g = frameGestures.find (finger); g != frameGestures.end())
                {
                    if (mapped) frameUp (g->second, pos);
                    frameGestures.erase (g);
                }
            }
            routes.erase (it);
        }
    }

    for (auto& [f, r] : routes)
        if (f.first == key) ++active;

    if (auto p = panels.find (key); p != panels.end() && active > p->second.maxContactsSeen)
        p->second.maxContactsSeen = active;

    if (injector.isTouchDropped() && ! warnedTouchDropped)
    {
        warnedTouchDropped = true;
        log ("JUCE dropped the injected touch: this build lacks the canUseTouch() patch "
             "(spatcore/patches/juce/macos)");
    }
}

//==============================================================================
namespace
{
    NSPoint toAppKit (juce::Point<float> desktopPos)
    {
        const auto scale = juce::Desktop::getInstance().getGlobalScaleFactor();
        NSScreen* primary = [[NSScreen screens] firstObject];
        const CGFloat height = primary != nil ? primary.frame.size.height : 0;
        return NSMakePoint (desktopPos.x * scale, height - desktopPos.y * scale);
    }

    NSWindow* windowWithNumber (long number)
    {
        for (NSWindow* w in [NSApp windows])
            if (w.windowNumber == number)
                return w;
        return nil;
    }

    NSButton* buttonUnder (NSWindow* w, NSPoint p, int& which)
    {
        for (NSWindowButton b : { NSWindowCloseButton, NSWindowMiniaturizeButton, NSWindowZoomButton })
            if (NSButton* button = [w standardWindowButton: b];
                button != nil && ! button.hidden && button.enabled)
            {
                const NSRect r = [w convertRectToScreen: [button convertRect: button.bounds toView: nil]];
                if (NSPointInRect (p, NSInsetRect (r, -6, -6)))   // a fingertip is wider than the button
                {
                    which = (int) b;
                    return button;
                }
            }
        return nil;
    }
}

bool MacHidTouchManager::frameDown (const std::pair<juce::String, int>& finger, juce::Point<float> pos)
{
    const NSPoint p = toAppKit (pos);
    const NSInteger number = [NSWindow windowNumberAtPoint: p belowWindowWithWindowNumber: 0];
    NSWindow* w = windowWithNumber ((long) number);
    if (w == nil)
        return false;

    FrameGesture g;
    g.windowNumber = (long) number;
    g.start = pos;
    g.originX = w.frame.origin.x;
    g.originY = w.frame.origin.y;

    int which = -1;
    if (buttonUnder (w, p, which) != nil)
        g.button = which;
    else if (p.y < NSMaxY ([w contentRectForFrameRect: w.frame]) || ! w.movable)
        return false;   // a resize edge, or a window that does not move: nothing to do

    frameGestures[finger] = g;
    return true;
}

bool MacHidTouchManager::edgeDown (const std::pair<juce::String, int>& finger, juce::Point<float> pos)
{
    constexpr int outside = 14, inside = 4;

    // Front to back, so a window in front owns the point.
    for (NSWindow* w in [NSApp orderedWindows])
    {
        if (! w.visible || w.miniaturized)
            continue;

        juce::ComponentPeer* peer = nullptr;
        for (int i = 0; i < juce::ComponentPeer::getNumPeers() && peer == nullptr; ++i)
            if (auto* candidate = juce::ComponentPeer::getPeer (i))
                if (auto* view = (__bridge NSView*) candidate->getNativeHandle(); view != nil && view.window == w)
                    peer = candidate;
        if (peer == nullptr)
            continue;

        auto& comp = peer->getComponent();
        const auto b = comp.getScreenBounds();   // the content area, JUCE desktop coordinates
        if (! b.expanded (outside).contains (pos.roundToInt()))
            continue;
        if (b.reduced (inside).contains (pos.roundToInt()))
            return false;   // well inside the frontmost window here: not an edge

        auto* rw = dynamic_cast<juce::ResizableWindow*> (&comp);
        if (rw == nullptr || ! rw->isResizable() || rw->isFullScreen() || rw->isKioskMode())
            return false;

        int edges = 0;
        if (pos.x < b.getX() + inside)       edges |= leftEdge;
        if (pos.x > b.getRight() - inside)   edges |= rightEdge;
        if (pos.y > b.getBottom() - inside)  edges |= bottomEdge;
        if (edges == 0)
            return false;   // above the window: its title bar, handled by frameDown

        FrameGesture g;
        g.windowNumber = (long) w.windowNumber;
        g.edges = edges;
        g.start = pos;
        g.window = &comp;
        g.startBounds = comp.getBounds();
        frameGestures[finger] = g;
        return true;
    }
    return false;
}

void MacHidTouchManager::frameMove (const FrameGesture& g, juce::Point<float> pos)
{
    if (g.edges != 0)
    {
        auto* comp = g.window.getComponent();
        if (comp == nullptr)
            return;

        const auto d = (pos - g.start).roundToInt();
        auto b = g.startBounds;
        if (g.edges & leftEdge)   b.setLeft   (b.getX() + d.x);
        if (g.edges & rightEdge)  b.setRight  (b.getRight() + d.x);
        if (g.edges & bottomEdge) b.setBottom (b.getBottom() + d.y);

        // Through the window's constrainer, as a drag of its own edge would
        // go: minimum and maximum sizes hold, the opposite edges stay put.
        if (auto* rw = dynamic_cast<juce::ResizableWindow*> (comp); rw != nullptr && rw->getConstrainer() != nullptr)
            rw->getConstrainer()->setBoundsForComponent (rw, b, false, (g.edges & leftEdge) != 0,
                                                         (g.edges & bottomEdge) != 0, (g.edges & rightEdge) != 0);
        else
            comp->setBounds (b);
        return;
    }

    if (g.button >= 0)
        return;

    if (NSWindow* w = windowWithNumber (g.windowNumber))
    {
        const auto scale = juce::Desktop::getInstance().getGlobalScaleFactor();
        const auto delta = (pos - g.start) * scale;
        [w setFrameOrigin: NSMakePoint (g.originX + delta.x, g.originY - delta.y)];
    }
}

void MacHidTouchManager::frameUp (const FrameGesture& g, juce::Point<float> pos)
{
    if (g.button < 0)
        return;

    if (NSWindow* w = windowWithNumber (g.windowNumber))
    {
        int which = -1;
        if (NSButton* button = buttonUnder (w, toAppKit (pos), which); button != nil && which == g.button)
            [button performClick: nil];
    }
}

namespace
{
    void postMouse (CGEventType type, juce::Point<float> desktopPos, int clickCount)
    {
        const auto scale = juce::Desktop::getInstance().getGlobalScaleFactor();
        const CGPoint p = CGPointMake (desktopPos.x * scale, desktopPos.y * scale);
        if (CGEventRef ev = CGEventCreateMouseEvent (nullptr, type, p, kCGMouseButtonLeft))
        {
            if (clickCount > 0)
                CGEventSetIntegerValueField (ev, kCGMouseEventClickState, clickCount);
            CGEventPost (kCGHIDEventTap, ev);
            CFRelease (ev);
        }
    }
}

void MacHidTouchManager::pointerDown (juce::Point<float> pos)
{
    const double now = juce::Time::getMillisecondCounterHiRes() / 1000.0;
    const bool repeat = now - lastPointerUpTime < [NSEvent doubleClickInterval]
                        && pos.getDistanceFrom (lastPointerUpPos) < 8.0f;
    pointerClickCount = repeat ? pointerClickCount + 1 : 1;
    pointerFingerDown = true;
    lastPointerPos = pos;

    postMouse (kCGEventMouseMoved, pos, 0);
    postMouse (kCGEventLeftMouseDown, pos, pointerClickCount);
}

void MacHidTouchManager::pointerMove (juce::Point<float> pos)
{
    lastPointerPos = pos;
    postMouse (kCGEventLeftMouseDragged, pos, pointerClickCount);
}

void MacHidTouchManager::pointerUp (juce::Point<float> pos)
{
    postMouse (kCGEventLeftMouseUp, pos, pointerClickCount);
    pointerFingerDown = false;
    lastPointerUpTime = juce::Time::getMillisecondCounterHiRes() / 1000.0;
    lastPointerUpPos = pos;
}

//==============================================================================
void MacHidTouchManager::loadSettings()
{
    auto parsed = juce::JSON::parse (settingsFile->getValue (kSettingsKey, ""));
    if (auto* obj = parsed.getDynamicObject())
        for (auto& prop : obj->getProperties())
            settings[prop.name.toString()] = PanelSettings::fromVar (prop.value);
}

void MacHidTouchManager::saveSettings()
{
    auto* obj = new juce::DynamicObject();
    for (auto& [key, s] : settings)
        obj->setProperty (key, s.toVar());
    settingsFile->setValue (kSettingsKey, juce::JSON::toString (juce::var (obj), true));
    settingsFile->saveIfNeeded();
}

} // namespace spatcore::controllers

#endif
