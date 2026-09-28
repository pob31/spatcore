#pragma once

#include <juce_core/juce_core.h>

namespace spatcore::control::mcp::guards
{

/**
 * Which HTTP requests the MCP endpoint answers at all.
 *
 * The server listens on this machine for AI clients that run on it, and
 * those clients never send an Origin header. A web page open in a browser on
 * the same machine can reach the port too, so the transport refuses what
 * only a browser would send:
 *   - a Host header that is not a loopback name: DNS rebinding points an
 *     attacker's host name at 127.0.0.1, and the browser then sends that name;
 *   - an Origin header from anywhere but a loopback address: every
 *     cross-origin request a page makes carries one;
 *   - a POST whose Content-Type is not application/json: a page can send
 *     text/plain, form data or multipart without asking the server first, but
 *     never application/json.
 * The loopback names are the ones the MCP SDKs accept by default:
 * 127.0.0.1, localhost and [::1], on any port (an SSH tunnel changes it).
 */

/** ":" followed by one to five digits. */
inline bool isPortSuffix (const juce::String& s)
{
    return s.length() >= 2 && s.length() <= 6 && s[0] == ':'
        && s.substring (1).containsOnly ("0123456789");
}

/** The host of "name", "name:port", "[v6]" or "[v6]:port" (brackets
 *  dropped); empty when the text is none of those. */
inline juce::String hostOfAuthority (const juce::String& authority)
{
    const auto a = authority.trim();

    if (a.startsWithChar ('['))
    {
        const int close = a.indexOfChar (']');
        if (close < 0)
            return {};

        const auto rest = a.substring (close + 1);
        if (rest.isNotEmpty() && ! isPortSuffix (rest))
            return {};

        return a.substring (1, close);
    }

    const int colon = a.indexOfChar (':');
    if (colon < 0)
        return a;

    return isPortSuffix (a.substring (colon)) ? a.substring (0, colon) : juce::String();
}

inline bool isLoopbackName (const juce::String& host)
{
    return host == "127.0.0.1" || host == "::1" || host.equalsIgnoreCase ("localhost");
}

/** A dotted-quad IPv4 address or an IPv6 address (hex digits, colons, and
 *  dots for an embedded IPv4 tail). A host name is neither, which is what
 *  matters here: DNS rebinding needs a name. */
inline bool isIpLiteral (const juce::String& host)
{
    if (host.containsChar (':'))
        return host.containsOnly ("0123456789abcdefABCDEF:.");

    juce::StringArray parts;
    parts.addTokens (host, ".", {});
    if (parts.size() != 4)
        return false;

    for (const auto& part : parts)
        if (part.isEmpty() || part.length() > 3 || ! part.containsOnly ("0123456789")
            || part.getIntValue() > 255)
            return false;

    return true;
}

/** The Host header of a request to a server bound to 127.0.0.1
 *  (loopbackOnly) or to every interface. A server on every interface also
 *  answers requests addressed to one of its IP addresses; neither answers a
 *  host name other than localhost. */
inline bool isAllowedHost (const juce::String& hostHeader, bool loopbackOnly)
{
    const auto host = hostOfAuthority (hostHeader);
    if (host.isEmpty())
        return false;

    return isLoopbackName (host) || (! loopbackOnly && isIpLiteral (host));
}

/** An Origin header value: http or https on a loopback name, any port.
 *  "null" (a sandboxed frame or a file: page) is refused like any other. */
inline bool isAllowedOrigin (const juce::String& origin)
{
    const auto o = origin.trim();

    juce::String authority;
    if (o.startsWithIgnoreCase ("http://"))
        authority = o.substring (7);
    else if (o.startsWithIgnoreCase ("https://"))
        authority = o.substring (8);
    else
        return false;

    if (authority.containsChar ('/'))
        return false;

    return isLoopbackName (hostOfAuthority (authority));
}

/** "application/json", with or without parameters such as a charset. */
inline bool isJsonContentType (const juce::String& contentType)
{
    return contentType.upToFirstOccurrenceOf (";", false, false).trim()
               .equalsIgnoreCase ("application/json");
}

} // namespace spatcore::control::mcp::guards
