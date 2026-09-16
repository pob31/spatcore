#pragma once

#include <array>
#include <cstdint>

namespace spatcore::effects
{

/**
    The module vocabulary of an effects chain: which module types exist, which
    slots a chain has, and how an order string maps onto them.

    Both lists are APPEND-ONLY. A saved show, an OSC address and a snapshot all
    name a module by its token, and a stored chain order is a permutation of
    them, so renumbering or renaming an existing entry silently reorders every
    project that already exists. New module types and further instances of an
    existing type go on the end.
*/

/** Module types. Append only - the numbering reaches disk. */
enum class ModuleId : std::uint8_t
{
    Dist = 0,
    EQ,
    Dyn,
    Mod,
    Phaser,
    Trem,
    Reverb,
    Delay,
    Crush,
    Count
};

inline constexpr int kNumModuleTypes = 9;
static_assert (static_cast<int> (ModuleId::Count) == kNumModuleTypes,
               "kNumModuleTypes must track ModuleId::Count");

/** Slots in a chain. Nine types, with EQ and dynamics doubled. */
inline constexpr int kNumModuleSlots = 11;

/** Capability bound, mirrored by the app's maxEffectChannels. */
inline constexpr int kMaxEffectChannels = 32;

/** One chain slot: a module type, which instance of that type it is, and the
    token the order string and the wire use for it. */
struct SlotDesc
{
    ModuleId type;
    std::uint8_t instance;
    const char* token;
};

inline constexpr SlotDesc kSlots[kNumModuleSlots] =
{
    { ModuleId::Dist,   0, "dist"   },
    { ModuleId::EQ,     0, "eq1"    },
    { ModuleId::EQ,     1, "eq2"    },
    { ModuleId::Dyn,    0, "dyn1"   },
    { ModuleId::Dyn,    1, "dyn2"   },
    { ModuleId::Mod,    0, "mod"    },
    { ModuleId::Phaser, 0, "phaser" },
    { ModuleId::Trem,   0, "trem"   },
    { ModuleId::Reverb, 0, "reverb" },
    { ModuleId::Delay,  0, "delay"  },
    { ModuleId::Crush,  0, "crush"  }
};

/** A processing order: slot indices, most significant first. */
using ChainOrder = std::array<std::uint8_t, kNumModuleSlots>;

/** Slots in their declared order - the default chain. */
inline constexpr ChainOrder kDefaultOrder { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

/** True when every slot appears exactly once. The audio thread trusts this:
    a malformed order would process a slot twice and skip another. */
inline bool isValidChainOrder (const ChainOrder& order) noexcept
{
    std::uint32_t seen = 0;

    for (int i = 0; i < kNumModuleSlots; ++i)
    {
        const std::uint8_t v = order[static_cast<size_t> (i)];
        if (v >= kNumModuleSlots)
            return false;

        const std::uint32_t bit = 1u << v;
        if ((seen & bit) != 0)
            return false;

        seen |= bit;
    }

    return true;
}

namespace detail
{
    inline bool tokensEqual (const char* lowercased, const char* known) noexcept
    {
        while (*lowercased != '\0' && *known != '\0')
        {
            if (*lowercased != *known)
                return false;
            ++lowercased;
            ++known;
        }
        return *lowercased == '\0' && *known == '\0';
    }
}

/** Parse a comma-separated order string into slot indices.

    Pure and allocation-free, so the audio side could run it - though in
    practice the message thread parses once and publishes the cooked array.
    Case-insensitive, tolerant of spaces around tokens, and strict about
    everything else: exactly the 11 known tokens, each exactly once. `out` is
    left untouched unless the whole string parses, so a rejected edit keeps the
    order that was running.
*/
inline bool parseChainOrder (const char* csv, ChainOrder& out) noexcept
{
    if (csv == nullptr)
        return false;

    ChainOrder parsed {};
    bool used[kNumModuleSlots] = {};
    int count = 0;
    const char* c = csv;

    for (;;)
    {
        while (*c == ' ' || *c == '\t')
            ++c;

        char token[16];
        int length = 0;

        while (*c != '\0' && *c != ',' && *c != ' ' && *c != '\t')
        {
            if (length >= 15)
                return false;                       // longer than any known token

            char ch = *c;
            if (ch >= 'A' && ch <= 'Z')
                ch = static_cast<char> (ch - 'A' + 'a');

            token[length++] = ch;
            ++c;
        }

        token[length] = '\0';

        if (length == 0 || count >= kNumModuleSlots)
            return false;                           // empty token, or too many of them

        int match = -1;
        for (int i = 0; i < kNumModuleSlots; ++i)
        {
            if (! used[i] && detail::tokensEqual (token, kSlots[i].token))
            {
                match = i;
                break;
            }
        }

        if (match < 0)
            return false;                           // unknown token, or a duplicate

        used[match] = true;
        parsed[static_cast<size_t> (count++)] = static_cast<std::uint8_t> (match);

        while (*c == ' ' || *c == '\t')
            ++c;

        if (*c == ',')
        {
            ++c;
            continue;
        }

        if (*c == '\0')
            break;

        return false;                               // trailing garbage
    }

    if (count != kNumModuleSlots)
        return false;

    out = parsed;
    return true;
}

} // namespace spatcore::effects
