#pragma once

/**
    Where the fingers are in a HID multitouch digitizer's input reports, read
    from the device's own report descriptor.

    A Windows-style multitouch screen (usage page 0x0D, usage 0x04) describes
    each finger as a logical collection (0x0D/0x22) holding a Contact ID, a
    Tip Switch and X/Y, with one Contact Count per report. Panels differ in
    everything else - two ILITEK controllers from the same vendor already
    disagree on the logical range and the report length - so nothing here
    assumes a layout: offsets, sizes and ranges all come from the descriptor.

    Header-only and dependency-free (no JUCE, no OS headers), so it is unit
    tested on every platform with real descriptors as fixtures.
*/

#include <cstddef>
#include <cstdint>
#include <vector>

namespace spatcore::controllers::hid {

/** One value in a report: its bit position counted from the first byte after
    the report ID, and the logical range the descriptor declares. */
struct ReportField
{
    uint32_t bitOffset  = 0;
    uint32_t bitSize    = 0;
    int32_t  logicalMin = 0;
    int32_t  logicalMax = 0;

    bool isPresent() const noexcept { return bitSize > 0; }

    /** The raw value, sign-extended when the logical range goes negative.
        0 when the field lies beyond the bytes given. */
    int32_t read (const uint8_t* body, size_t bodySize) const noexcept
    {
        if (! isPresent() || bitSize > 32 || bitOffset + bitSize > bodySize * 8)
            return 0;

        uint64_t v = 0;
        for (uint32_t i = 0; i < bitSize; ++i)
        {
            const uint32_t bit = bitOffset + i;
            if ((body[bit >> 3] >> (bit & 7)) & 1u)
                v |= (uint64_t) 1 << i;
        }

        if (logicalMin < 0 && bitSize < 32 && ((v >> (bitSize - 1)) & 1u))
            v |= ~(uint64_t) 0 << bitSize;

        return (int32_t) (int64_t) v;
    }

    /** The value as a fraction of the logical range, clamped to 0..1. */
    float readNormalised (const uint8_t* body, size_t bodySize) const noexcept
    {
        if (logicalMax <= logicalMin)
            return 0.0f;

        const float f = (float) ((double) (read (body, bodySize) - logicalMin)
                                 / (double) (logicalMax - logicalMin));
        return f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
    }
};

/** One finger collection. X and Y are required; the rest are optional. */
struct FingerFields
{
    ReportField contactId, tipSwitch, x, y, confidence;

    bool isUsable() const noexcept { return x.isPresent() && y.isPresent(); }
};

/** An input report that carries fingers. */
struct DigitizerReport
{
    uint8_t  reportId = 0;      // 0 when the device uses no report IDs
    uint32_t bodyBits = 0;      // report size after the report ID byte
    std::vector<FingerFields> fingers;
    ReportField contactCount;
    ReportField scanTime;
};

/** A value in a feature report, with the report it lives in. */
struct FeatureField
{
    uint8_t     reportId = 0;
    uint32_t    reportBits = 0;   // size of that feature report after its ID
    ReportField field;

    bool isPresent() const noexcept { return field.isPresent(); }
};

struct DigitizerLayout
{
    bool usesReportIds = false;
    std::vector<DigitizerReport> reports;

    /** Contact Count Maximum (0x0D/0x55). Reading this feature report is the
        handshake Windows performs on every connection, and some firmware stays
        in mouse-emulation mode until it happens: the 16" ILITEK panel sends no
        multitouch report after a cold plug until it is read, and its first
        report follows the read by 7 ms. */
    FeatureField contactCountMax;

    /** Input Mode (0x0D/0x52), on panels that have one: written with 2
        (multi-input) to leave mouse mode, as Linux's hid-multitouch does. */
    FeatureField inputMode;

    bool isUsable() const noexcept { return ! reports.empty(); }

    const DigitizerReport* findReport (uint8_t reportId) const noexcept
    {
        for (auto& r : reports)
            if (r.reportId == reportId)
                return &r;
        return nullptr;
    }
};

namespace usages
{
    constexpr uint32_t make (uint16_t page, uint16_t id) noexcept { return ((uint32_t) page << 16) | id; }

    constexpr uint32_t x            = make (0x01, 0x30);
    constexpr uint32_t y            = make (0x01, 0x31);
    constexpr uint32_t finger       = make (0x0D, 0x22);
    constexpr uint32_t tipSwitch    = make (0x0D, 0x42);
    constexpr uint32_t confidence   = make (0x0D, 0x47);
    constexpr uint32_t contactId    = make (0x0D, 0x51);
    constexpr uint32_t contactCount = make (0x0D, 0x54);
    constexpr uint32_t scanTime     = make (0x0D, 0x56);
    constexpr uint32_t inputMode    = make (0x0D, 0x52);
    constexpr uint32_t contactCountMax = make (0x0D, 0x55);
}

/** Walks a report descriptor (HID 1.11, section 6.2.2) and returns every input
    report that has at least one usable finger collection. Malformed input
    yields what was understood before the fault, never a read past the end. */
inline DigitizerLayout parseDigitizerDescriptor (const uint8_t* desc, size_t size)
{
    struct Globals
    {
        uint16_t usagePage   = 0;
        int32_t  logicalMin  = 0;
        int32_t  logicalMax  = 0;
        uint32_t reportSize  = 0;
        uint32_t reportCount = 0;
        uint8_t  reportId    = 0;
    };

    struct ReportCursor { uint8_t id; uint32_t bits; };

    DigitizerLayout layout;
    Globals g;
    std::vector<Globals> globalStack;

    std::vector<uint32_t> localUsages;
    uint32_t usageMin = 0, usageMax = 0;
    bool haveUsageRange = false;

    std::vector<ReportCursor> inputCursors;  // running input bit offset per report ID
    std::vector<ReportCursor> featureCursors;
    std::vector<bool> collectionIsFinger;    // the open collections, innermost last
    FingerFields finger;
    int fingerDepth = -1;                    // collectionIsFinger index of the open finger

    auto cursorFor = [&] (uint8_t id) -> uint32_t&
    {
        for (auto& c : inputCursors)
            if (c.id == id)
                return c.bits;
        inputCursors.push_back ({ id, 0 });
        return inputCursors.back().bits;
    };

    auto featureCursorFor = [&] (uint8_t id) -> uint32_t&
    {
        for (auto& c : featureCursors)
            if (c.id == id)
                return c.bits;
        featureCursors.push_back ({ id, 0 });
        return featureCursors.back().bits;
    };

    auto reportFor = [&] (uint8_t id) -> DigitizerReport&
    {
        for (auto& r : layout.reports)
            if (r.reportId == id)
                return r;
        layout.reports.push_back ({});
        layout.reports.back().reportId = id;
        return layout.reports.back();
    };

    auto usageAt = [&] (uint32_t i) -> uint32_t
    {
        if (i < localUsages.size())
            return localUsages[i];
        if (haveUsageRange && usageMin + i <= usageMax)
            return usageMin + i;
        if (! localUsages.empty())
            return localUsages.back();
        return 0;
    };

    auto clearLocals = [&]
    {
        localUsages.clear();
        haveUsageRange = false;
    };

    size_t pos = 0;
    while (pos < size)
    {
        const uint8_t prefix = desc[pos++];

        if (prefix == 0xFE)  // long item: data size, tag, data
        {
            if (pos + 2 > size) break;
            pos += 2 + desc[pos];
            continue;
        }

        const uint32_t dataSize = (prefix & 3) == 3 ? 4 : (prefix & 3);
        if (pos + dataSize > size) break;

        uint32_t u = 0;
        for (uint32_t i = 0; i < dataSize; ++i)
            u |= (uint32_t) desc[pos + i] << (8 * i);
        int32_t s = (int32_t) u;
        if (dataSize == 1) s = (int8_t) u;
        if (dataSize == 2) s = (int16_t) u;
        pos += dataSize;

        const uint8_t type = (prefix >> 2) & 3;
        const uint8_t tag  = prefix >> 4;
        const auto withPage = [&] (uint32_t v) { return dataSize == 4 ? v : usages::make (g.usagePage, (uint16_t) v); };

        if (type == 1)  // global
        {
            switch (tag)
            {
                case 0x0: g.usagePage   = (uint16_t) u; break;
                case 0x1: g.logicalMin  = s; break;
                case 0x2: g.logicalMax  = s; break;
                case 0x7: g.reportSize  = u; break;
                case 0x8: g.reportId    = (uint8_t) u; layout.usesReportIds = true; break;
                case 0x9: g.reportCount = u; break;
                case 0xA: globalStack.push_back (g); break;
                case 0xB: if (! globalStack.empty()) { g = globalStack.back(); globalStack.pop_back(); } break;
                default:  break;  // physical range, unit, exponent: not needed
            }
        }
        else if (type == 2)  // local
        {
            switch (tag)
            {
                case 0x0: localUsages.push_back (withPage (u)); break;
                case 0x1: usageMin = withPage (u); haveUsageRange = true; break;
                case 0x2: usageMax = withPage (u); haveUsageRange = true; break;
                default:  break;
            }
        }
        else if (type == 0)  // main
        {
            if (tag == 0xA)  // collection
            {
                const bool isFinger = usageAt (0) == usages::finger && fingerDepth < 0;
                collectionIsFinger.push_back (isFinger);
                if (isFinger)
                {
                    finger = {};
                    fingerDepth = (int) collectionIsFinger.size() - 1;
                }
            }
            else if (tag == 0xC)  // end collection
            {
                if (! collectionIsFinger.empty())
                {
                    if ((int) collectionIsFinger.size() - 1 == fingerDepth)
                    {
                        if (finger.isUsable())
                            reportFor (g.reportId).fingers.push_back (finger);
                        fingerDepth = -1;
                    }
                    collectionIsFinger.pop_back();
                }
            }
            else if (tag == 0x8)  // input
            {
                const bool constant = (u & 1) != 0;
                uint32_t& bits = cursorFor (g.reportId);

                // A logical maximum written in fewer bytes than it needs reads
                // back negative; with a non-negative minimum it was meant unsigned.
                int32_t logicalMax = g.logicalMax;
                if (logicalMax < g.logicalMin && g.logicalMin >= 0 && g.reportSize < 32)
                    logicalMax = (int32_t) ((1u << g.reportSize) - 1);

                for (uint32_t i = 0; i < g.reportCount && g.reportSize > 0; ++i)
                {
                    ReportField f { bits, g.reportSize, g.logicalMin, logicalMax };
                    bits += g.reportSize;
                    if (constant) continue;

                    const uint32_t usage = usageAt (i);
                    if (fingerDepth >= 0)
                    {
                        if      (usage == usages::contactId)  finger.contactId  = f;
                        else if (usage == usages::tipSwitch)  finger.tipSwitch  = f;
                        else if (usage == usages::x)          finger.x          = f;
                        else if (usage == usages::y)          finger.y          = f;
                        else if (usage == usages::confidence) finger.confidence = f;
                    }
                    else if (usage == usages::contactCount)
                        reportFor (g.reportId).contactCount = f;
                    else if (usage == usages::scanTime)
                        reportFor (g.reportId).scanTime = f;
                }
            }

            else if (tag == 0xB)  // feature
            {
                uint32_t& bits = featureCursorFor (g.reportId);
                for (uint32_t i = 0; i < g.reportCount && g.reportSize > 0; ++i)
                {
                    ReportField f { bits, g.reportSize, g.logicalMin, g.logicalMax };
                    bits += g.reportSize;
                    if ((u & 1) != 0) continue;

                    const uint32_t usage = usageAt (i);
                    if (usage == usages::contactCountMax && ! layout.contactCountMax.isPresent())
                        layout.contactCountMax = { g.reportId, 0, f };
                    else if (usage == usages::inputMode && ! layout.inputMode.isPresent())
                        layout.inputMode = { g.reportId, 0, f };
                }
            }

            clearLocals();  // output, feature and the above all end the local state
        }
    }

    // Keep only reports that really carry fingers, with their final sizes.
    std::vector<DigitizerReport> withFingers;
    for (auto& r : layout.reports)
    {
        if (r.fingers.empty()) continue;
        r.bodyBits = cursorFor (r.reportId);
        withFingers.push_back (r);
    }
    layout.reports = std::move (withFingers);

    if (layout.contactCountMax.isPresent())
        layout.contactCountMax.reportBits = featureCursorFor (layout.contactCountMax.reportId);
    if (layout.inputMode.isPresent())
        layout.inputMode.reportBits = featureCursorFor (layout.inputMode.reportId);

    return layout;
}

} // namespace spatcore::controllers::hid
