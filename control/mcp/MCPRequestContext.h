#pragma once

#include <juce_core/juce_core.h>
#include "MCPProtocolVersions.h"

namespace spatcore::control::mcp
{

/** Per-request transport metadata handed to the dispatcher.

    Lives in its own header rather than alongside the transport so the
    dispatcher can take one without pulling in the HTTP server's headers —
    it has no business knowing what SimpleWeb is.

    Grouped into a struct rather than passed as loose arguments so that
    adding another transport-level input later (a session id, an auth
    principal) doesn't churn the handler signature a third time. */
struct RequestContext
{
    juce::String clientIP;
    int          clientPort = 0;

    /** Value of the `MCP-Protocol-Version` request header, or empty when
        the client didn't send one. The transport has already validated
        that a non-empty value is supported, so the dispatcher can treat
        this as trusted. */
    juce::String protocolVersionHeader;

    /** Revision to assume for this request: the header when present,
        otherwise the spec-mandated fallback for header-less clients. */
    juce::String effectiveProtocolVersion() const
    {
        return protocolVersionHeader.isNotEmpty()
                   ? protocolVersionHeader
                   : juce::String (protocol::kFallbackWhenHeaderAbsent);
    }
};

} // namespace spatcore::control::mcp
