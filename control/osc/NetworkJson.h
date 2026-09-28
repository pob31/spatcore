#pragma once

#include <juce_core/juce_core.h>

namespace spatcore::control::osc
{

/**
 * JSON that came in over the wire, read without letting it choose our stack
 * depth.
 *
 * Why this exists:
 *   juce::JSON's parser recurses once per array or object it opens, with no
 *   limit. A few hundred kilobytes of '[' overflow the stack of whatever
 *   thread parses them, so one request or datagram can crash the app before
 *   any validation runs. Nothing we receive legitimately nests anywhere near
 *   maxNetworkJsonDepth.
 */

/** How deep a JSON document from the network may nest before we refuse it. */
constexpr int maxNetworkJsonDepth = 64;

/** True when `text` holds more than `maxDepth` arrays and objects open at
 *  once. It scans the text the way juce::JSON reads it: strings in either
 *  quote, backslash escapes, and the end at the first NUL. That matters: if
 *  the scan took a quote for the start of a string where the parser does
 *  not, brackets the parser recurses into would go uncounted. Costs one pass
 *  over the text, no allocation. */
inline bool jsonNestsDeeperThan (const juce::String& text, int maxDepth)
{
    int depth = 0;
    juce::juce_wchar quote = 0;

    for (auto p = text.getCharPointer();;)
    {
        const auto c = p.getAndAdvance();
        if (c == 0)
            return false;

        if (quote != 0)
        {
            if (c == '\\')
            {
                if (p.getAndAdvance() == 0)
                    return false;
            }
            else if (c == quote)
            {
                quote = 0;
            }
        }
        else if (c == '"' || c == '\'')
        {
            quote = c;
        }
        else if (c == '[' || c == '{')
        {
            if (++depth > maxDepth)
                return true;
        }
        else if ((c == ']' || c == '}') && depth > 0)
        {
            --depth;
        }
    }
}

/** juce::JSON::parse for text from the network: a void var for anything that
 *  nests deeper than maxNetworkJsonDepth, as for any other text that does
 *  not parse. */
inline juce::var parseNetworkJson (const juce::String& text)
{
    if (jsonNestsDeeperThan (text, maxNetworkJsonDepth))
        return {};

    return juce::JSON::parse (text);
}

} // namespace spatcore::control::osc
