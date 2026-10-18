#include "OSCIngestQueue.h"
#include "NetworkStringUtils.h"

#include <cstring>
#include <utility>

namespace spatcore::control::osc
{

OSCIngestQueue::OSCIngestQueue (Classifier classifierIn)
    : digitAfterPrefixBypasses (classifierIn.digitAfterPrefixBypasses)
{
    // Precompute the byte-comparable rule table once so tryClassify on
    // the receiver threads is a bounds check + memcmp per rule.
    rules.reserve (classifierIn.rules.size());
    for (const auto& r : classifierIn.rules)
    {
        Rule rule;
        rule.prefix     = r.prefix.toStdString();
        rule.len        = static_cast<int> (rule.prefix.size());
        rule.perChannel = r.firstInt32IsChannel;
        if (rule.len > 0)
            rules.push_back (std::move (rule));
    }

    for (const auto& address : classifierIn.bypassAddresses)
        if (address.isNotEmpty())
            bypassAddresses.push_back (address.toStdString());

    // Drain at a fixed cadence on the MessageManager thread. This
    // guarantees the GUI gets idle slices between batches, which
    // callAsync-on-demand draining cannot (each call piles onto the MM
    // queue and starves paint/input under flood).
    startTimer (kDrainIntervalMs);
}

OSCIngestQueue::~OSCIngestQueue()
{
    stopTimer();
}

void OSCIngestQueue::timerCallback()
{
    drainBatch();
}

void OSCIngestQueue::setDrainIntervalMs (int ms)
{
    startTimer (juce::jmax (1, ms));
}

namespace
{
    // Round up to next 4-byte boundary (OSC alignment).
    inline int alignTo4 (int v) noexcept { return (v + 3) & ~3; }
}

bool OSCIngestQueue::tryClassify (const char* data, int dataSize,
                                   juce::String& outKey) const
{
    if (data == nullptr || dataSize <= 0 || data[0] != '/')
        return false;  // not a leading-/-message; bundles fall here too

    // Find end of address (first null within bounds).
    int addrEnd = 0;
    while (addrEnd < dataSize && data[addrEnd] != '\0')
        ++addrEnd;
    if (addrEnd == 0 || addrEnd >= dataSize)
        return false;

    // Addresses the app keeps in arrival order go through the FIFO.
    for (const auto& b : bypassAddresses)
        if (addrEnd == static_cast<int> (b.size()) && std::memcmp (data, b.data(), b.size()) == 0)
            return false;

    // Match against the injected prefix set.
    const Rule* matched = nullptr;
    for (const auto& r : rules)
    {
        if (addrEnd >= r.len && std::memcmp (data, r.prefix.data(), (size_t) r.len) == 0)
        {
            matched = &r;
            break;
        }
    }
    if (matched == nullptr)
        return false;

    // OSCQuery short form like /wfs/input/7/positionX has a digit
    // immediately after the prefix. We don't try to canonicalise those
    // here — they go through the FIFO path. (Normalising would mean
    // re-emitting the standard form, which is more work than it saves.)
    if (digitAfterPrefixBypasses
        && matched->perChannel
        && matched->len < addrEnd
        && data[matched->len] >= '0' && data[matched->len] <= '9')
    {
        return false;
    }

    // For channel-less families (e.g. /wfs/config/*) key on the address.
    if (! matched->perChannel)
    {
        outKey = safeStringFromBytes (data, addrEnd);
        if (outKey.isEmpty())
            return false;  // address contained invalid UTF-8 — let FIFO handle it
        return true;
    }

    // Per-channel: read the first int32 arg.
    //   <address>\0... pad to 4 ... ",i...\0... pad to 4 ... <int32>
    const int typeTagStart = alignTo4 (addrEnd + 1);
    if (typeTagStart >= dataSize || data[typeTagStart] != ',')
        return false;

    // First arg type must be 'i' (int32) to extract the channel.
    if (typeTagStart + 1 >= dataSize || data[typeTagStart + 1] != 'i')
        return false;

    // Find end of type-tag string.
    int tagEnd = typeTagStart;
    while (tagEnd < dataSize && data[tagEnd] != '\0')
        ++tagEnd;
    if (tagEnd >= dataSize)
        return false;

    const int argStart = alignTo4 (tagEnd + 1);
    if (argStart + 4 > dataSize)
        return false;

    // Big-endian int32.
    const uint32_t bits = (static_cast<uint8_t> (data[argStart    ]) << 24)
                        | (static_cast<uint8_t> (data[argStart + 1]) << 16)
                        | (static_cast<uint8_t> (data[argStart + 2]) <<  8)
                        | (static_cast<uint8_t> (data[argStart + 3]));
    const int channelId = static_cast<int32_t> (bits);

    juce::String addr = safeStringFromBytes (data, addrEnd);
    if (addr.isEmpty())
        return false;
    outKey = addr + "|" + juce::String (channelId);
    return true;
}

void OSCIngestQueue::push (juce::MemoryBlock data,
                            juce::String senderIP,
                            int port,
                            ConnectionMode transport)
{
    juce::String coalesceKey;
    const bool coalesceable = tryClassify (
        static_cast<const char*> (data.getData()),
        static_cast<int> (data.getSize()),
        coalesceKey);

    {
        const juce::ScopedLock sl (lock);

        if (coalesceable)
        {
            if (coalesced.size() >= kCoalesceCap
                && coalesced.find (coalesceKey) == coalesced.end())
            {
                droppedTotal.fetch_add (1, std::memory_order_relaxed);
                ++droppedSinceLastReport;
                lastDropTransport = transport;
                return;
            }

            // Newest-wins. Replace any existing entry for this key.
            const auto inserted = coalesced.find (coalesceKey);
            if (inserted != coalesced.end())
                coalescedTotal.fetch_add (1, std::memory_order_relaxed);

            IngestItem item;
            item.data      = std::move (data);
            item.senderIP  = std::move (senderIP);
            item.port      = port;
            item.transport = transport;
            coalesced[std::move (coalesceKey)] = std::move (item);
        }
        else
        {
            if (fifo.size() >= fifoCap.load (std::memory_order_relaxed))
            {
                droppedTotal.fetch_add (1, std::memory_order_relaxed);
                ++droppedSinceLastReport;
                lastDropTransport = transport;
                return;
            }

            IngestItem item;
            item.data      = std::move (data);
            item.senderIP  = std::move (senderIP);
            item.port      = port;
            item.transport = transport;
            fifo.push_back (std::move (item));
        }
    }

    // No scheduling — the Timer drains at a fixed cadence regardless
    // of arrival rate.
}

void OSCIngestQueue::drainBatch()
{
    // At most maxItemsPerTick items a tick, and - when a budget is set -
    // no longer than drainBudgetMs: whichever comes first. Each item is
    // taken under the lock and dispatched outside it, so the receiver
    // thread keeps pushing and nothing taken is ever put back. Capping the
    // per-tick work is essential - without it, a flood drains for >16 ms
    // and starves the GUI.
    const int    cap     = maxItemsPerTick.load();
    const double budget  = drainBudgetMs.load();
    const double started = juce::Time::getMillisecondCounterHiRes();

    uint64_t       dropsToReport = 0;
    ConnectionMode dropTransport = ConnectionMode::UDP;

    {
        const juce::ScopedLock sl (lock);

        // The first drop at once, then once a second while they go on.
        if (droppedSinceLastReport > 0
            && (lastDropReportMs <= 0.0
                || started - lastDropReportMs >= kDropReportIntervalMs
                || droppedSinceLastReport >= kDropLogEvery))
        {
            dropsToReport = droppedSinceLastReport;
            dropTransport = lastDropTransport;
            droppedSinceLastReport = 0;
            lastDropReportMs = started;
        }
    }

    for (int taken = 0; taken < cap; ++taken)
    {
        IngestItem item;

        {
            const juce::ScopedLock sl (lock);

            // FIFO first (events / bundles are order-sensitive), then the
            // coalesced latest values, whose order does not matter.
            if (! fifo.empty())
            {
                item = std::move (fifo.front());
                fifo.pop_front();
            }
            else if (! coalesced.empty())
            {
                auto it = coalesced.begin();
                item = std::move (it->second);
                coalesced.erase (it);
            }
            else
            {
                break;
            }
        }

        if (dispatch)
        {
            dispatch (item.data, item.senderIP, item.port, item.transport);
            dispatchedTotal.fetch_add (1, std::memory_order_relaxed);
        }

        if (budget > 0.0 && juce::Time::getMillisecondCounterHiRes() - started >= budget)
            break;
    }

    if (dropsToReport > 0 && dropReport)
        dropReport (droppedTotal.load (std::memory_order_relaxed), dropTransport);
}

} // namespace spatcore::control::osc
