#pragma once

/**
    Turns a multitouch digitizer's input reports into per-contact down, move
    and up events.

    Contacts are keyed by Contact ID, never by slot: on the ILITEK panels a
    finger moves to a lower slot whenever one below it lifts (175 times in a
    45 s bench capture). A lift normally arrives once with Tip Switch = 0 at
    the last position; a contact that disappears from a complete frame without
    one is lifted anyway, so a dropped report can never leave a finger stuck
    down.

    Frames: a report whose Contact Count is non-zero starts a frame of that
    many contacts. Parallel-mode panels deliver the whole frame in one report;
    hybrid-mode panels spread it over several, the later ones with Contact
    Count 0, and the tracker waits until the frame is complete. A report that
    starts with Contact Count 0 and no frame open means no contacts at all.

    Header-only and dependency-free; not thread-safe (one tracker per device,
    fed from that device's reader thread).
*/

#include "HidDigitizerDescriptor.h"

#include <algorithm>
#include <vector>

namespace spatcore::controllers::hid {

struct ContactEvent
{
    enum class Phase { down, move, up };

    Phase phase = Phase::move;
    int   contactId = 0;
    float x = 0.0f;   // fraction of the panel's logical X range, 0..1, left to right
    float y = 0.0f;   // fraction of the panel's logical Y range, 0..1, top to bottom
};

class ContactTracker
{
public:
    explicit ContactTracker (const DigitizerReport& reportLayout) : layout (reportLayout)
    {
        pending.reserve (32);
        contacts.reserve (32);
    }

    /** Feeds one report body (the bytes after the report ID). Appends to
        `out` whatever changed once a frame is complete; appends nothing while
        a hybrid frame is still being assembled. */
    void processReport (const uint8_t* body, size_t bodySize, std::vector<ContactEvent>& out)
    {
        const bool hasCount = layout.contactCount.isPresent();
        const int count = hasCount ? layout.contactCount.read (body, bodySize) : -1;

        if (count > 0 || ! hasCount)
        {
            pending.clear();
            expected = hasCount ? count : (int) layout.fingers.size();
        }
        else if (expected <= 0)
        {
            pending.clear();   // count 0 outside a frame: nothing is touching
            expected = 0;
        }

        for (const auto& f : layout.fingers)
        {
            if ((int) pending.size() >= expected)
                break;

            Slot s;
            s.id  = f.contactId.isPresent() ? f.contactId.read (body, bodySize) : (int) pending.size();
            s.tip = f.tipSwitch.isPresent() ? f.tipSwitch.read (body, bodySize) != 0 : true;
            if (f.confidence.isPresent() && f.confidence.read (body, bodySize) == 0)
                s.tip = false;   // the panel says this is not a finger (a palm, an edge)
            s.x = f.x.readNormalised (body, bodySize);
            s.y = f.y.readNormalised (body, bodySize);

            // Without a Contact Count, only real contacts and lifts count.
            if (! hasCount && ! s.tip && find (s.id) == nullptr)
                continue;

            pending.push_back (s);
        }

        if (hasCount && (int) pending.size() < expected)
            return;   // hybrid mode: the rest of the frame is in the next report

        applyFrame (out);
        expected = 0;
    }

    /** Lifts every contact still down, at its last position: for a device
        that is going away. */
    void releaseAll (std::vector<ContactEvent>& out)
    {
        for (const auto& c : contacts)
            out.push_back ({ ContactEvent::Phase::up, c.id, c.x, c.y });
        contacts.clear();
        pending.clear();
        expected = 0;
    }

    int getNumActiveContacts() const noexcept   { return (int) contacts.size(); }
    int getMaxContactsSeen() const noexcept     { return maxContactsSeen; }

private:
    struct Slot    { int id = 0; bool tip = false; float x = 0, y = 0; };
    struct Contact { int id = 0; float x = 0, y = 0; bool seen = false; };

    Contact* find (int id)
    {
        for (auto& c : contacts)
            if (c.id == id)
                return &c;
        return nullptr;
    }

    void applyFrame (std::vector<ContactEvent>& out)
    {
        for (auto& c : contacts)
            c.seen = false;

        for (const auto& s : pending)
        {
            auto* c = find (s.id);

            if (s.tip)
            {
                if (c == nullptr)
                {
                    contacts.push_back ({ s.id, s.x, s.y, true });
                    out.push_back ({ ContactEvent::Phase::down, s.id, s.x, s.y });
                }
                else
                {
                    c->seen = true;
                    if (c->x != s.x || c->y != s.y)
                    {
                        c->x = s.x;
                        c->y = s.y;
                        out.push_back ({ ContactEvent::Phase::move, s.id, s.x, s.y });
                    }
                }
            }
            else if (c != nullptr)
            {
                out.push_back ({ ContactEvent::Phase::up, s.id, s.x, s.y });
                c->id = -1;   // removed below
            }
        }

        // Lifted with Tip = 0 (id -1), or gone from a complete frame without one.
        for (auto& c : contacts)
            if (c.id >= 0 && ! c.seen)
            {
                out.push_back ({ ContactEvent::Phase::up, c.id, c.x, c.y });
                c.id = -1;
            }

        contacts.erase (std::remove_if (contacts.begin(), contacts.end(),
                                        [] (const Contact& c) { return c.id < 0; }),
                        contacts.end());

        maxContactsSeen = std::max (maxContactsSeen, (int) contacts.size());
    }

    DigitizerReport layout;
    std::vector<Slot> pending;
    std::vector<Contact> contacts;
    int expected = 0;
    int maxContactsSeen = 0;
};

} // namespace spatcore::controllers::hid
