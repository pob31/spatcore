#pragma once

/*
    Several applications reading the same touch panels, on one computer.

    WHY. On macOS a touch panel's fingers reach no application by themselves:
    each one that wants them reads the panel (MacHidTouchManager). WFS-DIY and
    Go.dot - or any two spatcore applications - can run side by side reading
    the same panel. Which window a finger is on is already right across them
    (each asks macOS which window is frontmost at that point, and injects only
    the fingers on its own), but three things collide:

      - each moves the system cursor for a finger outside its own windows, so
        a finger on the other application's window also became a synthetic
        click from this one;
      - the first to quit put the panel back in mouse mode under the other;
      - each kept its own panel settings, so screens were placed once per app.

    THIS IS HOW THEY AGREE, with no message between them - lock files in one
    folder every reader uses, held with flock(2), which the kernel lets go of
    when a process ends however it ends:

      readers/<id>.lock  held for a reader's life: who is reading. A file whose
                         lock can be taken belonged to a reader that is gone,
                         and is removed.
      pointer.lock       the one reader that moves the cursor for a finger
                         outside every reader's windows; the next takes it when
                         the holder quits or crashes.
      restore.lock       held while a quitting reader decides whether it is the
                         last, so two quitting together cannot both think the
                         other stays.
      settings.lock      held while the shared panel settings are read and
                         written.

    `id` is the process id unless given (a test runs two readers in one
    process: flock locks belong to the open file, so two in one process
    exclude each other as two processes do).

    Plain C++17 and POSIX, no JUCE: built on macOS and Linux, nothing on
    Windows (whose windows get their own fingers from the system).
*/

#include <string>
#include <vector>

namespace spatcore::controllers {

class TouchPanelRegistry
{
public:
    /** The folder every reader of this user shares:
        ~/Library/Application Support/spatcore/touch on macOS,
        $XDG_CONFIG_HOME/spatcore/touch (or ~/.config/...) on Linux. */
    static std::string defaultFolder();

    /** Starts reading: makes the folder and takes this reader's lock. */
    explicit TouchPanelRegistry (std::string folder = defaultFolder(), std::string id = {});

    /** Stops reading: every lock let go. */
    ~TouchPanelRegistry();

    TouchPanelRegistry (const TouchPanelRegistry&) = delete;
    TouchPanelRegistry& operator= (const TouchPanelRegistry&) = delete;

    /** Whether this reader's lock is held (false only when the folder could not be used). */
    bool isReading() const noexcept     { return readerFd >= 0; }

    const std::string& id() const noexcept      { return self; }
    const std::string& folder() const noexcept  { return root; }

    /** Whether this reader moves the cursor: the pointer lock taken if it is
        free, and kept until this reader stops. Without a usable folder, true -
        a reader alone behaves as it always did. */
    bool drivesPointer();

    /** The other readers alive now, by id; the files of readers gone removed. */
    std::vector<std::string> otherReaders();

    /** Whether a reader with this id, not this one, is alive. */
    bool isOtherReader (const std::string& otherId);

    /** At quit: whether this is the last reader, so the one to put the panels
        back in mouse mode. Takes the restore lock, held until this reader
        stops, so the decision and the restoring are one step. */
    bool lastToLeave();

    /** Where the shared panel settings are kept. */
    std::string settingsPath() const;

    /** The settings lock, held for its scope. */
    class SettingsLock
    {
    public:
        explicit SettingsLock (const TouchPanelRegistry&);
        ~SettingsLock();
        SettingsLock (const SettingsLock&) = delete;
        SettingsLock& operator= (const SettingsLock&) = delete;

    private:
        int fd = -1;
    };

private:
    std::string root, self;
    int readerFd = -1;
    int pointerFd = -1;
    int restoreFd = -1;
};

} // namespace spatcore::controllers
