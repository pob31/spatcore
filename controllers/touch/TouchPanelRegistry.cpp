#include "TouchPanelRegistry.h"

#if defined (__APPLE__) || defined (__linux__)

#include <cerrno>
#include <cstdlib>
#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace spatcore::controllers {

namespace
{
    bool makeFolders (const std::string& path)
    {
        // Each component in turn: mkdir -p, without std::filesystem (macOS 10.15+ only).
        for (std::size_t at = 1; at <= path.size(); ++at)
            if (at == path.size() || path[at] == '/')
            {
                const auto part = path.substr (0, at);
                if (::mkdir (part.c_str(), 0755) != 0 && errno != EEXIST)
                    return false;
            }

        return true;
    }

    int openLockFile (const std::string& path)
    {
        return ::open (path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    }

    /** A lock taken now or not at all. */
    bool tryLock (int fd)
    {
        return fd >= 0 && ::flock (fd, LOCK_EX | LOCK_NB) == 0;
    }

    void closeFd (int& fd)
    {
        if (fd >= 0)
            ::close (fd);   // closing the file lets go of its lock
        fd = -1;
    }
}

std::string TouchPanelRegistry::defaultFolder()
{
    const char* home = std::getenv ("HOME");
    const std::string base = home != nullptr ? home : "/tmp";

   #if defined (__APPLE__)
    return base + "/Library/Application Support/spatcore/touch";
   #else
    if (const char* config = std::getenv ("XDG_CONFIG_HOME"); config != nullptr && *config != 0)
        return std::string (config) + "/spatcore/touch";
    return base + "/.config/spatcore/touch";
   #endif
}

TouchPanelRegistry::TouchPanelRegistry (std::string folderToUse, std::string idToUse)
    : root (std::move (folderToUse)), self (std::move (idToUse))
{
    if (self.empty())
        self = std::to_string (static_cast<long> (::getpid()));

    if (! makeFolders (root + "/readers"))
        return;

    readerFd = openLockFile (root + "/readers/" + self + ".lock");

    if (! tryLock (readerFd))
        closeFd (readerFd);
}

TouchPanelRegistry::~TouchPanelRegistry()
{
    closeFd (pointerFd);

    if (readerFd >= 0)
    {
        // Gone with the lock, so the next reader never has to sweep it.
        ::unlink ((root + "/readers/" + self + ".lock").c_str());
        closeFd (readerFd);
    }

    closeFd (restoreFd);
}

bool TouchPanelRegistry::drivesPointer()
{
    if (readerFd < 0)
        return true;

    if (pointerFd >= 0)
        return true;

    pointerFd = openLockFile (root + "/pointer.lock");

    if (tryLock (pointerFd))
        return true;

    closeFd (pointerFd);
    return false;
}

std::vector<std::string> TouchPanelRegistry::otherReaders()
{
    std::vector<std::string> alive;

    if (readerFd < 0)
        return alive;

    const auto folder = root + "/readers";
    DIR* listing = ::opendir (folder.c_str());

    if (listing == nullptr)
        return alive;

    while (const dirent* entry = ::readdir (listing))
    {
        const std::string name = entry->d_name;
        const std::string suffix = ".lock";

        if (name.size() <= suffix.size() || name.compare (name.size() - suffix.size(), suffix.size(), suffix) != 0)
            continue;

        const auto other = name.substr (0, name.size() - suffix.size());

        if (other == self)
            continue;

        const auto path = folder + "/" + name;
        int fd = ::open (path.c_str(), O_RDWR | O_CLOEXEC);

        if (fd < 0)
            continue;

        if (tryLock (fd))
            ::unlink (path.c_str());    // its reader is gone: the lock was free
        else
            alive.push_back (other);

        closeFd (fd);
    }

    ::closedir (listing);
    return alive;
}

bool TouchPanelRegistry::isOtherReader (const std::string& otherId)
{
    if (readerFd < 0 || otherId.empty() || otherId == self)
        return false;

    const auto path = root + "/readers/" + otherId + ".lock";
    int fd = ::open (path.c_str(), O_RDWR | O_CLOEXEC);

    if (fd < 0)
        return false;

    const auto unheld = tryLock (fd);
    closeFd (fd);
    return ! unheld;
}

bool TouchPanelRegistry::lastToLeave()
{
    if (readerFd < 0)
        return true;

    if (restoreFd < 0)
    {
        restoreFd = openLockFile (root + "/restore.lock");

        // Waited for: another reader deciding the same thing decides first.
        if (restoreFd >= 0 && ::flock (restoreFd, LOCK_EX) != 0)
            closeFd (restoreFd);
    }

    return otherReaders().empty();
}

std::string TouchPanelRegistry::settingsPath() const
{
    return root + "/panels.settings";
}

TouchPanelRegistry::SettingsLock::SettingsLock (const TouchPanelRegistry& registry)
{
    if (registry.readerFd < 0)
        return;

    fd = openLockFile (registry.root + "/settings.lock");

    if (fd >= 0 && ::flock (fd, LOCK_EX) != 0)
        closeFd (fd);
}

TouchPanelRegistry::SettingsLock::~SettingsLock()
{
    closeFd (fd);
}

} // namespace spatcore::controllers

#endif
