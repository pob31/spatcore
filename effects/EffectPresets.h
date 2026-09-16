#pragma once

#include "EffectParams.h"
#include <cstdint>

namespace spatcore::effects
{

/**
    Factory presets: the tables, and nothing else.

    A preset here is DATA. Selecting a type writes the expanded values into the
    tree so a saved show carries the numbers rather than a name, and editing any
    one of them flips the type to Custom. That state machine belongs to the app
    layer, which owns the tree and the undo history; a module that tried to
    infer it from its own parameter struct would have to guess whether a value
    that happens to match a row was chosen or inherited.

    Type ids reach disk. The lists are APPEND-ONLY: renumbering a row silently
    changes what every project that already exists sounds like.
*/

/** Reverb types for model 0. Custom is last and deliberately has no row: it
    means "these values came from somewhere other than the table". */
enum class ReverbType : std::uint8_t
{
    Room = 0,
    Chamber,
    Hall,
    Cathedral,
    Plate,
    Custom,
    Count
};

/** Rows in the reverb table - Custom is not one of them. */
inline constexpr int kNumReverbPresets = 5;

/**
    One reverb row.

    Eight of the reverb's twelve parameters. Tone and mix are absent on purpose:
    they are taste rather than room, and a player who has dialled a wet balance
    for a song should not lose it by auditioning another room. Bypass and model
    are absent for the same reason - the type chooses a room within a model, not
    the model itself.
*/
struct ReverbPreset
{
    const char* name;
    float rt60;             // s
    float rt60LowMult;      // x
    float rt60HighMult;     // x
    float crossoverLow;     // Hz
    float crossoverHigh;    // Hz
    float diffusion;        // 0..1
    float size;             // x
    float predelayMs;       // ms
};

/** Model 0 (FDN). Each model owns its own table: a plate algorithm's "Hall"
    would need different numbers to mean the same thing, so a shared table
    would be a promise the second model could not keep. */
inline constexpr ReverbPreset kFdnReverbPresets[kNumReverbPresets] =
{
    { "Room",      0.6f, 1.1f, 0.5f, 200.0f,  4000.0f, 0.60f, 0.6f,  5.0f },
    { "Chamber",   1.2f, 1.2f, 0.6f, 180.0f,  5000.0f, 0.80f, 0.8f,  8.0f },
    { "Hall",      2.4f, 1.3f, 0.4f, 200.0f,  4000.0f, 0.50f, 1.3f, 20.0f },
    { "Cathedral", 5.0f, 1.5f, 0.3f, 150.0f,  3000.0f, 0.40f, 1.8f, 40.0f },
    { "Plate",     1.8f, 0.8f, 0.9f, 300.0f,  8000.0f, 0.95f, 0.7f,  0.0f }
};

/** The row for a (model, type) pair, or null for Custom, an unknown type or a
    model with no table. Null is an answer, not a failure: the caller keeps the
    values it already has. */
inline const ReverbPreset* findReverbPreset (int model, int type) noexcept
{
    if (model != 0)
        return nullptr;                             // only model 0 exists in v1

    if (type < 0 || type >= kNumReverbPresets)
        return nullptr;                             // Custom, or out of range

    return &kFdnReverbPresets[type];
}

/** Expands a type into a parameter struct, leaving tone, mix, bypass and model
    as they were. Returns false and touches nothing when the type has no row. */
inline bool applyReverbPreset (ReverbParams& params, int model, int type) noexcept
{
    const ReverbPreset* preset = findReverbPreset (model, type);

    if (preset == nullptr)
        return false;

    params.type           = static_cast<std::uint8_t> (type);
    params.rt60           = preset->rt60;
    params.rt60LowMult    = preset->rt60LowMult;
    params.rt60HighMult   = preset->rt60HighMult;
    params.crossoverLow   = preset->crossoverLow;
    params.crossoverHigh  = preset->crossoverHigh;
    params.diffusion      = preset->diffusion;
    params.size           = preset->size;
    params.predelayMs     = preset->predelayMs;

    return true;
}

} // namespace spatcore::effects
