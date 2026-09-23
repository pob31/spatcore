#pragma once

#include "EffectsTypes.h"
#include <cstdint>
#include <type_traits>

namespace spatcore::effects
{

/**
    The parameter surface of one effects channel, as plain data.

    Everything here is a trivially copyable POD, because this is what crosses
    the thread boundary: the message thread cooks a whole EffectChannelParams
    (decibels already converted where the module wants linear, strings already
    resolved to indices) and hands it over through rt/RtTripleBuffer.h. The
    realtime side copies, it never converts, allocates or follows a pointer.

    Field names are the ValueTree identifiers minus the "effect" prefix, so the
    app's binding layer is a transcription rather than a translation.

    Defaults are the ones in the effects plan's parameter tables, and one of
    them is load-bearing: every module is BYPASSED by default. A freshly created
    effects channel is silent and transparent until someone asks for something.
*/

struct DistortionParams
{
    std::uint8_t bypass = 1;
    std::uint8_t oversample = 0;            // 0 auto, 1 off, 2 = 2x, 3 = 4x

    float driveDb = 12.0f;
    float shape = 0.5f;                     // 0 = hard clip +-0.8, 1 = tanh
    float bias = 0.0f;                      // asymmetry -> even harmonics

    float preLoShelfHz = 20.0f,    preLoShelfDb = 0.0f;
    float preHiShelfHz = 20000.0f, preHiShelfDb = 0.0f;
    float postLoShelfHz = 20.0f,    postLoShelfDb = 0.0f;
    float postHiShelfHz = 20000.0f, postHiShelfDb = 0.0f;

    float outputDb = -6.0f;
    float mix = 100.0f;                     // WET per cent
};

/** Six bands, using the OUTPUT EQ's shape ids (0 off, 1 low cut, 2 low shelf,
    3 peak, 4 band pass, 5 high shelf, 6 high cut, 7 all pass) - not the reverb
    EQ's different numbering. */
struct EqParams
{
    std::uint8_t bypass = 1;
    std::uint8_t shape[6] = { 1, 2, 3, 3, 5, 6 };
    float freqHz[6] = { 80.0f, 250.0f, 1000.0f, 4000.0f, 8000.0f, 12000.0f };
    float gainDb[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    float q[6]      = { 0.7f, 0.7f, 0.7f, 0.7f, 0.7f, 0.7f };
    float slope[6]  = { 0.7f, 0.7f, 0.7f, 0.7f, 0.7f, 0.7f };
};

/** A compressor stage followed by an expander stage, as in the prototype.

    compDetectorDelayMs is not a mistake and not the same thing as lookahead: it
    delays the DETECTOR, so the first milliseconds of a transient pass at unity
    and the gain reduction lands afterwards - the "let it through, then grab"
    behaviour of some analogue units, which a slow attack cannot reproduce
    because a slow attack starts reducing immediately. lookaheadMs delays the
    AUDIO instead, which is what a limiter needs, and costs latency. */
struct DynamicsParams
{
    std::uint8_t bypass = 1;
    std::uint8_t detector = 0;              // 0 peak, 1 RMS
    std::uint8_t autoMakeup = 0;
    std::uint8_t compOn = 1;
    std::uint8_t expOn = 0;

    float lookaheadMs = 1.0f;
    float makeupDb = 0.0f;

    float compThresholdDb = -20.0f;
    float compRatio = 4.0f;                 // 100 = limiter
    float compKneeDb = 0.0f;
    float compAttackMs = 10.0f;
    float compReleaseMs = 100.0f;
    float compDetectorDelayMs = 0.0f;       // transient pass, adds no latency
    float compScLoCutHz = 20.0f;
    float compScHiCutHz = 20000.0f;

    float expThresholdDb = -50.0f;
    float expRatio = 2.0f;                  // 100 = gate
    float expAttackMs = 10.0f;
    float expReleaseMs = 100.0f;
    float expRangeDb = -60.0f;
    float expHoldMs = 20.0f;
    float expScLoCutHz = 20.0f;
    float expScHiCutHz = 20000.0f;
};

struct ModulationParams
{
    std::uint8_t bypass = 1;
    std::uint8_t mode = 0;                  // 0 chorus, 1 flanger
    std::uint8_t voices = 2;
    std::uint8_t shape = 1;                 // LFOWaveforms shape id
    std::uint8_t throughZero = 0;

    float rateHz = 0.8f;
    float depth = 50.0f;                    // per cent of the centre delay
    float delayMs = 15.0f;
    float feedback = 0.0f;                  // signed per cent
    float phaseDeg = 0.0f;                  // spreads linked channels apart
    float loCutHz = 20.0f;
    float mix = 50.0f;
};

struct PhaserParams
{
    std::uint8_t bypass = 1;
    std::uint8_t stages = 6;                // 4, 6, 8 or 12
    std::uint8_t shape = 1;

    float centreHz = 800.0f;
    float spreadOct = 1.0f;
    float rateHz = 0.3f;
    float depthOct = 2.0f;
    float feedback = 30.0f;
    float mix = 50.0f;
};

/** Depth is in DECIBELS, so the modulation is exponential in amplitude - which
    is what the prototype does and what sounds smooth. */
struct TremoloParams
{
    std::uint8_t bypass = 1;

    float rateHz = 4.0f;
    float depthDb = 12.0f;
    float shape = 0.0f;                     // 0 sine ... 1 triangle, continuous
    float mix = 100.0f;
};

struct ReverbParams
{
    std::uint8_t bypass = 1;
    std::uint8_t model = 0;                 // ReverbModel (EffectPresets.h): 0 FDN, 1 Plate,
                                            // 4 Modulated Hall, 5 Shimmer; 2 / 3 reserved (run the FDN)
    std::uint8_t type = 6;                  // the preset id (ReverbType); 6 = Medium Hall, whose row
                                            // is exactly the defaults below. The DSP never reads it.
    std::uint8_t erProfile = 0;             // ErProfile: 0 Off, 1 Room, 2 Chamber, 3 Hall, 4 Cathedral
    std::uint8_t shimmerPitch = 0;          // ShimmerInterval: 0 = an octave up

    float predelayMs = 10.0f;
    float rt60 = 1.5f;
    float rt60LowMult = 1.3f;
    float rt60HighMult = 0.4f;
    float crossoverLow = 200.0f;
    float crossoverHigh = 4000.0f;
    float diffusion = 0.5f;
    float size = 1.0f;
    float toneHz = 12000.0f;
    float mix = 30.0f;

    float erLevelDb = -6.0f;                // early reflections vs the tail, dB
    float modRateHz = 0.8f;                 // tank modulation (Plate, Modulated Hall, Shimmer)
    float modDepth = 50.0f;                 // %
    float shimmerAmount = 50.0f;            // % of the shimmer lines' feedback that is pitch-shifted
};

struct MultitapParams
{
    std::uint8_t bypass = 1;
    std::uint8_t taps = 3;
    std::uint8_t tapMode = 1;               // 0 manual, 1 pattern
    std::uint8_t pattern = 0;               // equal / dotted / triplet / golden
    std::uint8_t feedbackTap = 0;           // 0 = last

    float timeMs = 375.0f;
    float tapTimeMs[8]  = { 375.0f, 750.0f, 1125.0f, 1500.0f, 1875.0f, 2250.0f, 2625.0f, 3000.0f };
    float tapLevelDb[8] = { 0.0f, -2.0f, -4.0f, -6.0f, -8.0f, -10.0f, -12.0f, -14.0f };

    float feedback = 30.0f;
    float inLoCutHz = 20.0f;
    float fbLoShelfHz = 200.0f,  fbLoShelfDb = 0.0f;
    float fbHiShelfHz = 4000.0f, fbHiShelfDb = -3.0f;
    float modRateHz = 0.1f;
    float modDepthPct = 0.0f;
    float diffusion = 0.0f;
    float glideMs = 200.0f;
    float mix = 35.0f;
};

struct BitcrusherParams
{
    std::uint8_t bypass = 1;
    std::uint8_t filter = 0;                // 0 hold (aliasing), 1 anti-aliased

    float bits = 8.0f;                      // fractional allowed
    float rateHz = 12000.0f;                // clamped to the device rate
    float ditherDb = -96.0f;                // -96 = off
    float mix = 100.0f;
};

/** Everything one chain needs for one block.

    `revision` is the change token: the chain re-applies parameters only when it
    moves, so a publisher that forgets to bump it will see its edit ignored.
    `order` is cooked - the string has already been parsed and validated on the
    message thread. */
struct EffectChannelParams
{
    std::uint8_t mute = 0;
    std::uint8_t chainBypass = 0;

    ChainOrder order = kDefaultOrder;

    float inputTrimLin = 1.0f;              // reserved (0 dB), not exposed in v1
    std::uint32_t revision = 0;

    DistortionParams dist;
    EqParams eq[2];
    DynamicsParams dyn[2];
    ModulationParams mod;
    PhaserParams phaser;
    TremoloParams trem;
    ReverbParams reverb;
    MultitapParams delay;
    BitcrusherParams crush;
};

static_assert (std::is_trivially_copyable_v<DistortionParams>, "effects params must be POD");
static_assert (std::is_trivially_copyable_v<EqParams>, "effects params must be POD");
static_assert (std::is_trivially_copyable_v<DynamicsParams>, "effects params must be POD");
static_assert (std::is_trivially_copyable_v<ModulationParams>, "effects params must be POD");
static_assert (std::is_trivially_copyable_v<PhaserParams>, "effects params must be POD");
static_assert (std::is_trivially_copyable_v<TremoloParams>, "effects params must be POD");
static_assert (std::is_trivially_copyable_v<ReverbParams>, "effects params must be POD");
static_assert (std::is_trivially_copyable_v<MultitapParams>, "effects params must be POD");
static_assert (std::is_trivially_copyable_v<BitcrusherParams>, "effects params must be POD");
static_assert (std::is_trivially_copyable_v<EffectChannelParams>, "effects params must be POD");

} // namespace spatcore::effects
