#pragma once

#include "EffectParams.h"
#include <cstdint>

namespace spatcore::effects
{

/**
    Factory presets: the tables, and nothing else.

    A preset here is DATA. Selecting one writes the expanded values into the
    tree so a saved show carries the numbers rather than a name, and editing any
    one of them flips the type to Custom. That state machine belongs to the app
    layer, which owns the tree and the undo history; a module that tried to
    infer it from its own parameter struct would have to guess whether a value
    that happens to match a row was chosen or inherited.

    Every id below reaches disk. The lists are APPEND-ONLY: renumbering a row
    silently changes what every project that already exists sounds like.
*/

//==============================================================================
// Reverb models
//==============================================================================

/** The algorithm behind the reverb module's tail. 2 and 3 are reserved (an
    SDN-style model and convolution, not built): a project that names them,
    or any id this build does not know, runs the FDN - it still makes a sound. */
enum class ReverbModel : std::uint8_t
{
    Fdn = 0,
    Plate = 1,
    SdnReserved = 2,
    IrReserved = 3,
    ModulatedHall = 4,
    Shimmer = 5,
    Count
};

/** The model a stored id actually runs. The engine, the panel and the Stream
    Deck all ask this one function, so they can never disagree about what a
    reserved or unknown id means. */
inline int resolveReverbModel (int stored) noexcept
{
    return (stored == static_cast<int> (ReverbModel::Plate)
            || stored == static_cast<int> (ReverbModel::ModulatedHall)
            || stored == static_cast<int> (ReverbModel::Shimmer))
               ? stored
               : static_cast<int> (ReverbModel::Fdn);
}

/** Early-reflection profiles, one pattern per space. Off skips the stage. */
enum class ErProfile : std::uint8_t
{
    Off = 0,
    Room,
    Chamber,
    Hall,
    Cathedral,
    Count
};

/** The shimmer lines' pitch intervals. The two-voice entries split the four
    shimmer lines two and two. An enum rather than continuous semitones: the
    musically useful shifts are a handful, and a dial stepping half a semitone
    would land between them. */
enum class ShimmerInterval : std::uint8_t
{
    OctaveUp = 0,           // +12
    FifthUp,                // +7
    FifthAndOctave,         // +7 & +12
    Twelfth,                // +19
    TwoOctaves,             // +24
    FourthUp,               // +5
    OctaveDown,             // -12
    OctaveDownAndUp,        // -12 & +12
    Count
};

/** The Size surface every model honours. */
inline constexpr float kReverbMinSize = 0.5f;
inline constexpr float kReverbMaxSize = 2.0f;

//==============================================================================
// Reverb presets
//==============================================================================

/** Preset ids - the stored effectReverbType. 0..4 are the model-0 rooms the
    module shipped with, frozen so every project that stores them keeps its
    meaning; 5 is Custom and deliberately has no row ("these values came from
    somewhere other than the table"); the rest are appended. */
enum class ReverbType : std::uint8_t
{
    Room = 0,
    Chamber,
    Hall,
    Cathedral,
    Plate,
    Custom,                 // 5 - no row
    MediumHall,             // 6 - the defaults, and the default type
    SmallRoom,
    MediumRoom,
    LargeRoom,
    LiveChamber,
    ConcertHall,            // 11
    LargeHall,
    StoneCathedral,
    LushHall,
    VocalPlate,             // 15
    BrightPlate,
    DrumPlate,
    DarkPlate,
    ShimmerOctave,          // 19
    ShimmerFifthOctave,
    ShimmerOctaveDown,
    ShimmerEthereal,        // 22
    Count
};

/**
    One preset row: the model, the early reflections and the room.

    A row OWNS fifteen of the reverb's parameters - selecting it chooses the
    model as well as the room, so "Vocal Plate" is a plate wherever it is picked
    from. Tone, mix and bypass are absent on purpose: they are taste rather than
    room, and a player who has dialled a wet balance for a song should not lose
    it by auditioning another space. Fields a row's model does not use carry the
    defaults, so switching the model by hand afterwards lands somewhere sane.
*/
struct ReverbPreset
{
    const char* name;           // null only for the Custom slot
    std::uint8_t model;         // ReverbModel
    std::uint8_t erProfile;     // ErProfile
    float erLevelDb;            // dB
    float predelayMs;           // ms
    float rt60;                 // s
    float rt60LowMult;          // x
    float rt60HighMult;         // x
    float crossoverLow;         // Hz
    float crossoverHigh;        // Hz
    float diffusion;            // 0..1
    float size;                 // x
    float modRateHz;            // Hz
    float modDepth;             // %
    std::uint8_t shimmerPitch;  // ShimmerInterval
    float shimmerAmount;        // %
};

/** Rows indexed by id. The values are starting points for listening, not
    measurements - they are meant to be tuned by ear. */
inline constexpr ReverbPreset kReverbPresets[static_cast<int> (ReverbType::Count)] =
{
    //  name                     mdl ER  ERdB  pre    rt60  lo×   hi×   xLo     xHi      diff   size  rate  depth pitch amt
    // Model 0 - the shipped rooms, frozen.
    { "Room (FDN)",              0,  0,  -6.0f,  5.0f, 0.60f, 1.10f, 0.50f, 200.0f, 4000.0f, 0.60f, 0.60f, 0.8f, 50.0f, 0, 50.0f },
    { "Chamber (FDN)",           0,  0,  -6.0f,  8.0f, 1.20f, 1.20f, 0.60f, 180.0f, 5000.0f, 0.80f, 0.80f, 0.8f, 50.0f, 0, 50.0f },
    { "Hall (FDN)",              0,  0,  -6.0f, 20.0f, 2.40f, 1.30f, 0.40f, 200.0f, 4000.0f, 0.50f, 1.30f, 0.8f, 50.0f, 0, 50.0f },
    { "Cathedral (FDN)",         0,  0,  -6.0f, 40.0f, 5.00f, 1.50f, 0.30f, 150.0f, 3000.0f, 0.40f, 1.80f, 0.8f, 50.0f, 0, 50.0f },
    { "Plate (FDN)",             0,  0,  -6.0f,  0.0f, 1.80f, 0.80f, 0.90f, 300.0f, 8000.0f, 0.95f, 0.70f, 0.8f, 50.0f, 0, 50.0f },
    { nullptr,                   0,  0,   0.0f,  0.0f, 0.00f, 0.00f, 0.00f,   0.0f,    0.0f, 0.00f, 0.00f, 0.0f,  0.0f, 0,  0.0f },  // Custom
    // FDN, with and without early reflections.
    { "Medium Hall",             0,  0,  -6.0f, 10.0f, 1.50f, 1.30f, 0.40f, 200.0f, 4000.0f, 0.50f, 1.00f, 0.8f, 50.0f, 0, 50.0f },
    { "Small Room",              0,  1,  -3.0f,  2.0f, 0.35f, 1.00f, 0.60f, 250.0f, 5000.0f, 0.70f, 0.50f, 0.8f, 50.0f, 0, 50.0f },
    { "Medium Room",             0,  1,  -4.0f,  3.0f, 0.60f, 1.10f, 0.55f, 220.0f, 4500.0f, 0.65f, 0.65f, 0.8f, 50.0f, 0, 50.0f },
    { "Large Room",              0,  2,  -5.0f,  5.0f, 0.90f, 1.15f, 0.50f, 200.0f, 4500.0f, 0.60f, 0.85f, 0.8f, 50.0f, 0, 50.0f },
    { "Live Chamber",            0,  2,  -5.0f,  3.0f, 1.40f, 1.20f, 0.65f, 180.0f, 6000.0f, 0.80f, 0.90f, 0.8f, 50.0f, 0, 50.0f },
    // Modulated Hall.
    { "Concert Hall",            4,  3,  -6.0f, 22.0f, 2.20f, 1.30f, 0.45f, 200.0f, 4000.0f, 0.65f, 1.20f, 0.5f, 30.0f, 0, 50.0f },
    { "Large Hall",              4,  3,  -7.0f, 28.0f, 3.20f, 1.40f, 0.40f, 180.0f, 3500.0f, 0.60f, 1.50f, 0.4f, 35.0f, 0, 50.0f },
    { "Stone Cathedral",         4,  4,  -8.0f, 25.0f, 6.00f, 1.50f, 0.35f, 150.0f, 3000.0f, 0.50f, 2.00f, 0.3f, 30.0f, 0, 50.0f },
    { "Lush Hall",               4,  0,  -6.0f, 30.0f, 4.50f, 1.20f, 0.50f, 180.0f, 4500.0f, 0.75f, 1.80f, 0.9f, 80.0f, 0, 50.0f },
    // Plate.
    { "Vocal Plate",             1,  0,  -6.0f, 25.0f, 1.60f, 0.90f, 0.75f, 250.0f, 7000.0f, 0.85f, 0.90f, 0.9f, 45.0f, 0, 50.0f },
    { "Bright Plate",            1,  0,  -6.0f, 10.0f, 2.20f, 0.80f, 1.00f, 300.0f, 9000.0f, 0.90f, 1.00f, 1.0f, 55.0f, 0, 50.0f },
    { "Drum Plate",              1,  0,  -6.0f,  0.0f, 1.00f, 0.70f, 0.70f, 350.0f, 6000.0f, 0.90f, 0.60f, 0.7f, 30.0f, 0, 50.0f },
    { "Dark Plate",              1,  0,  -6.0f, 15.0f, 2.60f, 1.20f, 0.35f, 200.0f, 3000.0f, 0.80f, 1.20f, 0.6f, 40.0f, 0, 50.0f },
    // Shimmer.
    { "Shimmer Octave",          5,  0,  -6.0f, 30.0f, 5.00f, 1.10f, 0.45f, 200.0f, 4500.0f, 0.75f, 1.60f, 0.6f, 50.0f, 0, 50.0f },
    { "Shimmer Fifth + Octave",  5,  0,  -6.0f, 30.0f, 6.00f, 1.10f, 0.45f, 200.0f, 4500.0f, 0.75f, 1.80f, 0.6f, 50.0f, 2, 55.0f },
    { "Shimmer Octave Down",     5,  0,  -6.0f, 20.0f, 4.00f, 1.20f, 0.40f, 200.0f, 3500.0f, 0.70f, 1.50f, 0.5f, 40.0f, 6, 40.0f },
    { "Shimmer Ethereal",        5,  0,  -6.0f, 40.0f, 8.00f, 1.00f, 0.50f, 200.0f, 5000.0f, 0.80f, 2.00f, 0.4f, 60.0f, 3, 60.0f },
};

/** The row for a preset id, or null for Custom and unknown ids. Null is an
    answer, not a failure: the caller keeps the values it already has. */
inline const ReverbPreset* findReverbPreset (int type) noexcept
{
    if (type < 0 || type >= static_cast<int> (ReverbType::Count))
        return nullptr;

    const ReverbPreset* row = &kReverbPresets[type];
    return row->name != nullptr ? row : nullptr;
}

/** Expands a preset into a parameter struct: the type and the fifteen values
    the row owns, leaving tone, mix and bypass as they were. Returns false and
    touches nothing when the id has no row. */
inline bool applyReverbPreset (ReverbParams& params, int type) noexcept
{
    const ReverbPreset* row = findReverbPreset (type);

    if (row == nullptr)
        return false;

    params.type          = static_cast<std::uint8_t> (type);
    params.model         = row->model;
    params.erProfile     = row->erProfile;
    params.erLevelDb     = row->erLevelDb;
    params.predelayMs    = row->predelayMs;
    params.rt60          = row->rt60;
    params.rt60LowMult   = row->rt60LowMult;
    params.rt60HighMult  = row->rt60HighMult;
    params.crossoverLow  = row->crossoverLow;
    params.crossoverHigh = row->crossoverHigh;
    params.diffusion     = row->diffusion;
    params.size          = row->size;
    params.modRateHz     = row->modRateHz;
    params.modDepth      = row->modDepth;
    params.shimmerPitch  = row->shimmerPitch;
    params.shimmerAmount = row->shimmerAmount;

    return true;
}

} // namespace spatcore::effects
