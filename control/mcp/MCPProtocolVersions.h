#pragma once

#include <juce_core/juce_core.h>

namespace spatcore::control::mcp::protocol
{

/** MCP spec revision this server advertises when it has no better
    information (unsupported or absent client request). */
inline constexpr const char* kLatest = "2025-06-18";

/** Revision to assume when a post-initialize HTTP request omits the
    `MCP-Protocol-Version` header. The 2025-06-18 spec mandates this
    specific fallback for backwards compatibility with clients written
    against the revision that introduced the header. */
inline constexpr const char* kFallbackWhenHeaderAbsent = "2025-03-26";

/** Revisions we accept.

    This server is wire-compatible with all three: it speaks the
    request/response half of Streamable HTTP over a single POST endpoint,
    with no protocol-level sessions, no SSE stream, and no server-initiated
    requests. Nothing in the 2025-03-26 or 2025-06-18 deltas changes how a
    server of that shape must behave, so accepting the older revisions
    costs nothing and keeps already-deployed clients working.

    Note the deliberate absence of per-connection negotiated-version state:
    because every accepted revision is handled identically, there is nothing
    to remember between requests. That matters — the transport runs a
    4-thread asio pool, so any retained negotiation state would need
    locking for no behavioural gain. */
inline bool isSupported (const juce::String& version) noexcept
{
    return version == "2025-06-18"
        || version == "2025-03-26"
        || version == "2024-11-05";
}

/** Human-readable list for error payloads. */
inline juce::String supportedList()
{
    return "2025-06-18, 2025-03-26, 2024-11-05";
}

} // namespace spatcore::control::mcp::protocol
