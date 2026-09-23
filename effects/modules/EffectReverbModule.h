#pragma once

#include "../EffectModule.h"
#include "../EffectPresets.h"
#include "../../dsp/FastDecibels.h"
#include "../../dsp/FractionalDelayLine.h"
#include "../../dsp/FrDiffusionModel.h"
#include "../../dsp/OnePoleSmoother.h"
#include "../../reverb/ReverbFDNAlgorithm.h"
#include "reverb/EarlyReflections.h"
#include "reverb/ReverbDelayLine.h"
#include "reverb/ReverbLfo.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace spatcore::effects
{

/**
    The tail generator behind the reverb module.

    The module owns predelay, tone and mix - the parts every algorithm needs in
    the same shape - and a model owns the tail itself. The plate, the SDN-style
    model and convolution are then a new class here rather than a second reverb
    module carrying its own copy of the wet path, and a project that names a
    model the build does not have still makes a sound.

    setParams RETURNS the variant flag rather than being told it, because only
    the model knows which of its parameters are build-time. prepare() allocates
    and is never realtime; everything else, commitPendingVariant() included,
    runs on the audio thread - the module rebuilds an IDLE instance there when
    a change needs another topology, so a model must reach its new topology
    without allocating.
*/
class IEffectReverbModel
{
public:
    virtual ~IEffectReverbModel() = default;

    /** Allocates. IEffectModule::prepare carries no parameters, so `initial` is
        the default set; the module's first applyParams commits any build-time
        difference immediately rather than through a fade. */
    virtual void prepare (const ChainConfig& config, const ReverbParams& initial) = 0;

    virtual void reset() noexcept = 0;

    /** True while the model is still running a value other than the one it has
        just been given, i.e. it needs silence to catch up. A STATE, not an
        edge: handed the value it is already running, it answers false. */
    virtual bool setParams (const ReverbParams& params) noexcept = 0;

    virtual void commitPendingVariant() noexcept {}

    /** Mono, in place: the predelayed input goes in, the wet tail comes out. */
    virtual void process (float* inout, int numSamples) noexcept = 0;

    /** True when `params` asks for a topology this instance is not built at.
        Asked WITHOUT handing the parameters over: the module turns such a
        change into a spillover to an idle twin, and the instance that is about
        to ring out must keep playing with the values it had. */
    virtual bool isBuildDifferent (const ReverbParams& params) const noexcept = 0;

    /** The Size the instance is built at (what process() runs). */
    virtual float getBuiltSize() const noexcept = 0;
};

//==============================================================================
/**
    Model 0: one node of the shipped FDN network, at the native device rate.

    Three decisions here are not obvious.

    The network is BUILT at a size well above the largest the parameter surface
    allows, and then immediately rebuilt at the size actually wanted. fdnSize is
    read only in prepareNode, so changing it means re-preparing, and the module
    calls commitPendingVariant() on the AUDIO thread. Preparing at 2.5 first
    leaves every delay line's vector holding capacity for more samples than any
    size in 0.5..2 can ask for, so every later rebuild is a vector::assign whose
    count fits the capacity already there - which reallocates on no standard
    library this ships against. The margin is not decorative: a line is
    floor(base * size * rateScale) plus a jitter of up to a sixteenth of itself,
    so a size of 1.9 can ask for a LONGER line than 2.0 does, and a capacity
    taken at exactly 2.0 would be a reallocation waiting for the right hash
    value. Taking 0.9375 * S * base - 1 as the shortest line a capacity build
    can leave, and 2 * 1.0625 * base as the longest any running size can ask
    for, S must clear 2.27 for every base length in the table; 2.5 does, with
    room to spare.

    Each instance gets its own node index, derived from the chain's noise key.
    Without it every effects channel would run the node-0 network - identical
    line lengths, identical tap signs - and their tails, being copies of each
    other, would sum into a comb-filtered centre image instead of spreading.

    The decay, crossover and diffusion values are handed over as they arrive
    rather than glided. The FDN recomputes 48 decay gains when any of them
    moves, which is a pow per line per band, and what a step in a decay gain
    changes is the SLOPE of a tail, not its level - there is no discontinuity to
    smooth away. The parameters that do modulate the signal sample by sample
    (predelay, tone, mix) live in the module and are smoothed there.

    A size commit rebuilds and clears the network, about 200 KiB of stores at
    48 kHz. It happens on an idle instance when someone moves the size control
    (the module spills the old tail over the new one), not per block.
*/
class FdnReverbModel final : public IEffectReverbModel
{
public:
    static constexpr float kMinSize = 0.5f;         // the plan's surface
    static constexpr float kMaxSize = 2.0f;
    static constexpr float kCapacitySize = 2.5f;    // see the class note

    void prepare (const ChainConfig& config, const ReverbParams& initial) override
    {
        sampleRate = config.sampleRate > 0.0 ? config.sampleRate : 48000.0;
        blockSamples = config.maxBlock > 0 ? config.maxBlock : 1;

        int budget = config.reverbMaxDelaySamples > 0
                       ? config.reverbMaxDelaySamples
                       : spatcore::reverb::FDNAlgorithm::MAX_DELAY_SAMPLES;
        if (budget < 4096)      budget = 4096;      // shorter than the longest line at size 1
        if (budget > (1 << 20)) budget = 1 << 20;   // a typo in a config is not a 4 GB allocation

        // The ceiling has to grow with the rate or the four longest lines all
        // clamp to the same length and the modal density collapses into a ring.
        int rateMult = static_cast<int> (std::ceil (sampleRate / 48000.0));
        if (rateMult < 1)  rateMult = 1;
        if (rateMult > 16) rateMult = 16;

        fdn = std::make_unique<spatcore::reverb::FDNAlgorithm> (budget * rateMult);
        fdn->setNodeIndexOffset (nodeOffsetFor (config.noiseKey));   // must precede prepare()

        nodeIn.setSize (1, blockSamples);
        nodeOut.setSize (1, blockSamples);
        nodeIn.clear();
        nodeOut.clear();

        setParams (initial);                        // prepareNode reads fdnSize, so values first
        buildAt (kCapacitySize);                    // claims the capacity every size can live in
        buildAt (quantiseSize (clampParam (initial.size, kMinSize, kMaxSize)));
    }

    void reset() noexcept override
    {
        if (fdn != nullptr)
            fdn->reset();
    }

    bool setParams (const ReverbParams& params) noexcept override
    {
        algoParams.rt60          = clampParam (params.rt60, 0.2f, 8.0f);
        algoParams.rt60LowMult   = clampParam (params.rt60LowMult, 0.1f, 9.0f);
        algoParams.rt60HighMult  = clampParam (params.rt60HighMult, 0.1f, 9.0f);
        algoParams.crossoverLow  = clampParam (params.crossoverLow, 50.0f, 500.0f);
        algoParams.crossoverHigh = clampParam (params.crossoverHigh, 1000.0f, 10000.0f);
        algoParams.diffusion     = clampParam (params.diffusion, 0.0f, 1.0f);
        algoParams.wetLevel      = 1.0f;            // the module owns the wet level
        algoParams.sdnScale      = 1.0f;            // not ours

        // Never hand the network a size it is not built at: fdnSize is part of
        // the FDN's own change gate, and a value it never prepares against
        // would make the next rebuild look like a no-op.
        algoParams.fdnSize = builtSize;

        pendingSize = quantiseSize (clampParam (params.size, kMinSize, kMaxSize));

        // The FDN gates on exact equality and recalculates nothing when nothing
        // moved, so an unconditional push costs a struct copy per block.
        if (fdn != nullptr)
            fdn->setParameters (algoParams);

        return pendingSize != builtSize;
    }

    void commitPendingVariant() noexcept override
    {
        if (fdn != nullptr && pendingSize != builtSize)
            buildAt (pendingSize);
    }

    void process (float* inout, int numSamples) noexcept override
    {
        if (fdn == nullptr || blockSamples <= 0)
            return;

        int done = 0;
        while (done < numSamples)
        {
            const int remaining = numSamples - done;
            const int chunk = remaining < blockSamples ? remaining : blockSamples;

            std::memcpy (nodeIn.getWritePointer (0), inout + done,
                         static_cast<size_t> (chunk) * sizeof (float));

            // ReverbAlgorithm documents nodeOutputs as cleared on entry. FDN
            // writes every sample, but honouring the contract is what keeps a
            // later algorithm that accumulates from being quietly wrong here.
            nodeOut.clear (0, 0, chunk);
            fdn->processBlock (nodeIn, nodeOut, chunk);

            std::memcpy (inout + done, nodeOut.getReadPointer (0),
                         static_cast<size_t> (chunk) * sizeof (float));
            done += chunk;
        }
    }

    /** The size the network is currently built at - the value process() is
        running, which is not the value setParams was last given while a commit
        is outstanding. */
    float getBuiltSize() const noexcept override { return builtSize; }

    bool isBuildDifferent (const ReverbParams& params) const noexcept override
    {
        return quantiseSize (clampParam (params.size, kMinSize, kMaxSize)) != builtSize;
    }

    /** Two decimals. A fader that sends 1.0000001 must not rebuild the network,
        and the shortest line only changes length every 1/337 of a size step
        anyway. */
    static float quantiseSize (float v) noexcept
    {
        return static_cast<float> (static_cast<int> (v * 100.0f + 0.5f)) * 0.01f;
    }

private:
    static float clampParam (float v, float lo, float hi) noexcept
    {
        // Negated comparisons so a NaN parameter lands on the low bound rather
        // than reaching a pow() and poisoning every decay gain in the network.
        if (! (v > lo)) return lo;
        if (v > hi)     return hi;
        return v;
    }

    /** One network per effects channel. The offset lands on the line lengths,
        the diffuser jitter and the tap signs at once. */
    static int nodeOffsetFor (std::uint32_t key) noexcept
    {
        const std::uint32_t mixed =
            spatcore::dsp::FrDiffusion::makeKey (static_cast<int> (key & 0x7FFFFFFFu),
                                                 static_cast<int> (ModuleId::Reverb));
        return static_cast<int> (mixed % 512u);
    }

    void buildAt (float size)
    {
        algoParams.fdnSize = size;
        fdn->setParameters (algoParams);            // prepareNode reads it from here
        fdn->prepare (sampleRate, blockSamples, 1);
        builtSize = size;
    }

    std::unique_ptr<spatcore::reverb::FDNAlgorithm> fdn;
    spatcore::reverb::AlgorithmParameters algoParams;

    juce::AudioBuffer<float> nodeIn, nodeOut;

    double sampleRate = 48000.0;
    int blockSamples = 0;
    float builtSize = 1.0f;
    float pendingSize = 1.0f;
};

//==============================================================================
/**
    Reverb: predelay, early reflections, a tail from the selected model, tone,
    and a wet mix.

    The wet path is predelay -> reflections + tail -> one-pole low pass ->
    make-up -> mix. The dry path is the buffer, untouched. With reflections on,
    the predelayed signal also goes into a ring that the reflection taps read,
    and the tail takes its input from the same ring a profile's tail delay
    later - see reverb/EarlyReflections.h.

    THE WET IS A SET OF WORLDS, NOT A MODEL. A world is one tail instance and
    the reflection pattern it runs with. Every tail class is built twice, in
    prepare(), and one world is ACTIVE. A change that needs another topology -
    a model of another class, another size, another reflection profile - does
    not fade the reverb out: a new world is built on the idle twin, and the
    input is partitioned between the two BY THE TIME IT WAS WRITTEN. Sound that
    arrived before the change plays out entirely in the old world - its
    reflections and its tail, with the values it had - and sound after it
    entirely in the new one, the switch smoothed by a 5 ms raised cosine on
    the write time. That is spillover, what a hardware reverb does on a program
    change: a cue that changes the room does not chop the tail that is still
    in the air. So the module never reports a variant, and the slot never
    fades it for one.

    Every world is linear and the partition is on the input, so a spillover is
    exactly the old world fed the input up to the change plus the new world
    fed the rest: no sample doubled or dropped, no gain that steps.

    - One crossfade at a time: a change that arrives within 5 ms of the last
      one waits for it.
    - At most three worlds run: the active one, one RINGING, and one DYING - a
      ringing world whose voice the next change needed, faded out over 5 ms.
      Rapid changes cost the oldest tail, never a click.
    - A ringing world goes idle once it has read its last input and its output
      has stayed under -96 dBFS for 50 ms, and is faded out after 30 s
      whatever it is doing.
    - Nothing to spill, nothing spilled: while no instance has run since the
      last reset - the first set after prepare(), a bypassed slot - a change
      takes effect at once.
    - Settled, one world with reflections off is the single-model path it
      always was, sample for sample: no ring, no crossfade, no summing with a
      zero.

    The wet leg carries a fixed gain of two, measured rather than guessed. The
    FDN's own times-four is a level CORRECTION, not headroom, and with it alone
    a fully wet reverb comes out 11 dB under the dry on white noise at the
    default decay - a mix control would then spend its whole travel making the
    reverb audible instead of balancing it. Doubling puts a 0.6 s room about
    8 dB under the dry, a 1.5 s hall 5 dB under and a 5 s cathedral level with
    it, which is a spread a mix control can work either side of. Two is exact in
    binary, so it costs the wet no accuracy. The reflection tables are
    normalised against it: at ER Level 0 dB they carry the dry's energy.

    Latency is reported as zero, and that is not a shortcut. A reverb adds a
    tail, not a delay: the dry component leaves in the same sample it arrived
    in. The predelay moves the WET only, so reporting it would have the ledger
    compensating for a delay the signal never had, and would drag every other
    channel back by it.

    Mix zero is the identity to the bit - the buffer is not touched, so negative
    zeros and denormals come out as they went in - but the reverb keeps running
    underneath. Freezing it would replay a stale tail the moment the mix came
    back up, and clearing the network every block would cost more than running
    it. The cheap way to switch a reverb off is the bypass flag, which the slot
    already honours by skipping the module entirely.

    Predelay glides rather than teleporting. A jump of a couple of hundred
    milliseconds chirps the reverb FEED while it moves, which the tail masks;
    the alternative, a mute-move-unmute envelope, puts a hole in the feed
    instead, which it does not.

    prepare() allocates. Nothing else does: an idle instance is rebuilt inside
    the capacity prepare() gave it - see FdnReverbModel - and a reflection
    pattern is a fixed-size array. About 560 KiB per module at 48 kHz (two
    networks sized for 2.5, a quarter-second predelay ring and a 0.4 s
    reflection ring), doubling with the rate.
*/
class EffectReverbModule final : public IEffectModule
{
public:
    static constexpr float kMaxPredelayMs = 250.0f;

    /** Wet make-up on top of the FDN's own +12 dB correction. A measured
        figure, not a round one - see the class note. */
    static constexpr float kWetGain = 2.0f;

    /** The input crossfade between two worlds, and a dying world's fade. */
    static constexpr double kSpillFadeSeconds = 0.005;

    /** A ringing world is idle once its wet output has stayed under
        kQuietGain (-96 dBFS) for kQuietHoldSeconds, and is faded out after
        kMaxRingSeconds whatever it is doing. */
    static constexpr float kQuietGain = 1.5849e-5f;
    static constexpr double kQuietHoldSeconds = 0.05;
    static constexpr double kMaxRingSeconds = 30.0;

    /** The reflections' level, dB against the dry. */
    static constexpr float kMinErLevelDb = -30.0f;
    static constexpr float kMaxErLevelDb = 6.0f;

    /** Tail classes, each built twice, so a change inside a class - a size -
        spills over as surely as a change between classes does. */
    static constexpr int kNumClasses = 1;
    static constexpr int kInstancesPerClass = 2;
    static constexpr int kNumInstances = kNumClasses * kInstancesPerClass;

    ModuleId type() const noexcept override { return ModuleId::Reverb; }

    void prepare (const ChainConfig& config) override
    {
        sampleRate = config.sampleRate > 0.0 ? config.sampleRate : 48000.0;
        maxBlock = config.maxBlock > 0 ? config.maxBlock : 1;
        msToSamples = static_cast<float> (sampleRate * 0.001);
        noiseKey = config.noiseKey;

        predelay.prepare (static_cast<int> (std::ceil (kMaxPredelayMs * 0.001 * sampleRate)) + 2);
        erRing.prepare (erMaxReadSamples (sampleRate) + 1);        // a delay of d reads d + 1 after the write
        wetScratch.assign (static_cast<size_t> (maxBlock), 0.0f);
        ringScratch.assign (static_cast<size_t> (maxBlock), 0.0f);
        dyingScratch.assign (static_cast<size_t> (maxBlock), 0.0f);
        for (auto& b : erScratch)
            b.assign (static_cast<size_t> (maxBlock), 0.0f);

        predelaySamples.setTimeConstant (sampleRate, kParamTauSeconds);
        predelaySamples.setSnapEpsilon (1.0e-3f);           // samples, not a unit gain
        toneCoef.setTimeConstant (sampleRate, kParamTauSeconds);
        toneCoef.setSnapEpsilon (1.0e-6f);                  // a one-pole coefficient
        wet.setTimeConstant (sampleRate, kParamTauSeconds);
        erGain.setTimeConstant (sampleRate, kParamTauSeconds);

        // w(k) = sin^2 (pi k / 2N), from the libm-free sine so a render hashes
        // the same on every platform. Its complement is the outgoing share.
        fadeLen = static_cast<int> (kSpillFadeSeconds * sampleRate + 0.5);
        if (fadeLen < 1)
            fadeLen = 1;
        fadeIn.assign (static_cast<size_t> (fadeLen), 0.0f);
        for (int k = 0; k < fadeLen; ++k)
        {
            const float s = ReverbLfo::sin2pi (static_cast<double> (k) / (4.0 * static_cast<double> (fadeLen)));
            fadeIn[static_cast<size_t> (k)] = s * s;
        }

        quietHoldSamples = static_cast<int> (kQuietHoldSeconds * sampleRate);
        maxRingSamples = static_cast<int> (kMaxRingSeconds * sampleRate);

        // Every instance, now. Building a model allocates, so the pool is
        // fixed here and a change of model only ever SELECTS from it.
        const ReverbParams defaults;
        for (int k = 0; k < kNumInstances; ++k)
        {
            instance (k).prepare (config, defaults);
            dirty[k] = true;                                // reset() below clears them all
        }

        for (auto& w : worlds)
            w = World {};

        activeW = 0;
        worlds[0].tail = firstOfClass (classFor (defaults));
        buildErTapSet (worlds[0].er, resolveErProfile (defaults.erProfile), sizeFor (defaults), sampleRate, noiseKey);
        ringingW = -1;
        dyingW = -1;
        transitionWanted = false;

        setTargets (defaults);
        snapOnNextApply = true;
        reset();
    }

    void reset() noexcept override
    {
        predelay.reset();
        erRing.reset();
        toneState = 0.0f;
        now = 0;
        lastChange = kLongAgo;

        for (int k = 0; k < kNumInstances; ++k)
            clearIfDirty (k);

        ringingW = -1;
        dyingW = -1;
        settle (worlds[activeW]);

        // Silent now, so a change still waiting for a voice has nothing left
        // to spill over: it takes effect directly.
        if (transitionWanted)
        {
            switchDirectly (wanted);
            transitionWanted = false;
        }

        predelaySamples.snap (predelaySamples.getTarget());
        toneCoef.snap (toneCoef.getTarget());
        wet.snap (wet.getTarget());
        erGain.snap (erGain.getTarget());

        meterDb.store (spatcore::dsp::FastDecibels::kMinDb, std::memory_order_relaxed);
    }

    ParamApplyInfo applyParams (const EffectChannelParams& params, int) noexcept override
    {
        const ReverbParams& r = params.reverb;

        setTargets (r);

        if (! needsTransition (r))
        {
            transitionWanted = false;               // a change taken back before it could start
            instance (worlds[activeW].tail).setParams (r);  // the runtime values
        }
        else if (snapOnNextApply || ! anyDirty())
        {
            // Nothing has run since the last reset, so there is no tail to
            // spill: the first set after prepare() - a freshly loaded show must
            // not spill over defaults it never meant to play - or a slot that
            // is bypassed and silent.
            switchDirectly (r);
            transitionWanted = false;
        }
        else
        {
            wanted = r;
            transitionWanted = ! tryStartTransition (r);
        }

        if (snapOnNextApply)
        {
            predelaySamples.snap (predelaySamples.getTarget());
            toneCoef.snap (toneCoef.getTarget());
            wet.snap (wet.getTarget());
            erGain.snap (erGain.getTarget());
            snapOnNextApply = false;
        }

        return { r.bypass != 0, false };
    }

    void process (float* inout, int numSamples) noexcept override
    {
        if (inout == nullptr || numSamples <= 0 || wetScratch.empty())
            return;

        // Sixteen recursive delay lines decay into denormals and stay there,
        // where the arithmetic costs an order of magnitude more than it does on
        // normals. Flushing them makes a quiet channel cheaper than a loud one
        // rather than dearer, and the module's own state is flushed below.
        juce::ScopedNoDenormals noDenormals;

        // A change that had to wait for the pool starts on a block boundary.
        if (transitionWanted && tryStartTransition (wanted))
            transitionWanted = false;

        float peak = 0.0f;
        int done = 0;

        while (done < numSamples)
        {
            const int remaining = numSamples - done;
            const int chunk = remaining < maxBlock ? remaining : maxBlock;
            processChunk (inout + done, chunk, peak);
            done += chunk;
        }

        // One comparison a block takes the tone filter to true zero instead of
        // letting it grind on a denormal for the rest of the show.
        if (! (std::fabs (toneState) > 1.0e-25f))
            toneState = 0.0f;

        meterDb.store (spatcore::dsp::FastDecibels::gainToDb (peak), std::memory_order_relaxed);
    }

    /** Zero: the dry signal is not delayed. See the class note. */
    int getLatencySamples() const noexcept override { return 0; }

    /** Peak WET level for the block, before the mix - the tail meter. */
    float getMeterDb() const noexcept override
    {
        return meterDb.load (std::memory_order_relaxed);
    }

    /** Worlds ringing out or fading under the active one: 0, 1 or 2. */
    int getSpillVoices() const noexcept      { return (ringingW >= 0 ? 1 : 0) + (dyingW >= 0 ? 1 : 0); }

    /** True while a change waits for the pool to free a voice. */
    bool isTransitionWaiting() const noexcept { return transitionWanted; }

    /** The Size the ACTIVE tail is built at - what the input is going to. */
    float getActiveSize() const noexcept     { return instance (worlds[activeW].tail).getBuiltSize(); }

    /** The reflection pattern the input is going to. */
    const ErTapSet& getActiveReflections() const noexcept { return worlds[activeW].er; }

private:
    static constexpr std::int64_t kLongAgo = -(static_cast<std::int64_t> (1) << 60);
    static constexpr std::int64_t kForever = static_cast<std::int64_t> (1) << 60;

    /** One tail instance and the reflections it runs with, and the stretch of
        input - by write time - that is theirs. */
    struct World
    {
        int tail = 0;                               // pool instance
        ErTapSet er;                                // Off: no taps, no tail delay
        std::int64_t start = kLongAgo;              // the input window opens here...
        std::int64_t end = kForever;                // ...and closes here, each over fadeLen
        float erGain = 1.0f;                        // frozen once the world stops being active
        float darkState = 0.0f;                     // the higher-order taps' one-pole
        int fadePos = 0;                            // dying: the output fade's position
        int age = 0;                                // ringing: samples since it stopped taking input
        int quiet = 0;                              // ringing: consecutive quiet samples
    };

    enum class Feed { Pass, Silence, Window };

    static float clampParam (float v, float lo, float hi) noexcept
    {
        if (! (v > lo)) return lo;
        if (v > hi)     return hi;
        return v;
    }

    static float sizeFor (const ReverbParams& r) noexcept
    {
        return FdnReverbModel::quantiseSize (clampParam (r.size, kReverbMinSize, kReverbMaxSize));
    }

    /** The tail class a parameter set runs on. */
    static int classFor (const ReverbParams& r) noexcept
    {
        juce::ignoreUnused (r);
        return 0;                                   // the FDN is the only class so far
    }

    static int firstOfClass (int cls) noexcept      { return cls * kInstancesPerClass; }
    static int classOfInstance (int k) noexcept     { return k / kInstancesPerClass; }

    IEffectReverbModel& instance (int k) noexcept              { return fdn[k]; }
    const IEffectReverbModel& instance (int k) const noexcept  { return fdn[k]; }

    bool needsTransition (const ReverbParams& r) const noexcept
    {
        const World& a = worlds[activeW];
        return classFor (r) != classOfInstance (a.tail)
            || instance (a.tail).isBuildDifferent (r)
            || ! a.er.matches (resolveErProfile (r.erProfile), sizeFor (r));
    }

    bool anyDirty() const noexcept
    {
        for (int k = 0; k < kNumInstances; ++k)
            if (dirty[k])
                return true;
        return false;
    }

    /** An instance that has run since it was last cleared is cleared before
        anything takes it again. */
    void clearIfDirty (int k) noexcept
    {
        if (dirty[k])
        {
            instance (k).reset();
            dirty[k] = false;
        }
    }

    /** A world that has always been the only one. */
    static void settle (World& w) noexcept
    {
        w.start = kLongAgo;
        w.end = kForever;
        w.darkState = 0.0f;
        w.fadePos = 0;
        w.age = 0;
        w.quiet = 0;
    }

    /** No tail to spill: the wanted world becomes the active one at once. */
    void switchDirectly (const ReverbParams& r) noexcept
    {
        World& a = worlds[activeW];
        const int cls = classFor (r);
        if (classOfInstance (a.tail) != cls)
        {
            clearIfDirty (a.tail);
            a.tail = firstOfClass (cls);
        }

        IEffectReverbModel& m = instance (a.tail);
        if (m.setParams (r))
            m.commitPendingVariant();

        buildErTapSet (a.er, resolveErProfile (r.erProfile), sizeFor (r), sampleRate, noiseKey);
        settle (a);
    }

    /** Starts a spillover into a new world built at `r`. False while the pool
        cannot take one yet - a crossfade still running, or the voice it needs
        still fading out - and the next block tries again. */
    bool tryStartTransition (const ReverbParams& r) noexcept
    {
        if (now < lastChange + fadeLen)
            return false;

        // The active world is about to ring, so the one ringing now has to go.
        // Its input window closed at least 5 ms ago; it keeps reading what it
        // had while its output fades.
        if (ringingW >= 0)
        {
            if (dyingW >= 0)
                return false;

            dyingW = ringingW;
            worlds[dyingW].fadePos = 0;
            ringingW = -1;
        }

        const int cls = classFor (r);
        const int busyActive = worlds[activeW].tail;
        const int busyDying = dyingW >= 0 ? worlds[dyingW].tail : -1;

        int target = -1;
        for (int k = firstOfClass (cls); k < firstOfClass (cls) + kInstancesPerClass; ++k)
        {
            if (k != busyActive && k != busyDying)
            {
                target = k;
                break;
            }
        }

        if (target < 0)
            return false;                           // its twin is the one fading out

        int slot = 0;
        while (slot == activeW || slot == dyingW)
            ++slot;

        IEffectReverbModel& m = instance (target);
        if (m.setParams (r))
            m.commitPendingVariant();               // inside the capacity prepare() gave it
        clearIfDirty (target);

        World& incoming = worlds[slot];
        incoming.tail = target;
        buildErTapSet (incoming.er, resolveErProfile (r.erProfile), sizeFor (r), sampleRate, noiseKey);
        incoming.start = now;
        incoming.end = kForever;
        incoming.darkState = 0.0f;
        incoming.fadePos = 0;
        incoming.age = 0;
        incoming.quiet = 0;

        World& outgoing = worlds[activeW];
        outgoing.end = now;
        outgoing.erGain = erGain.getCurrent();      // keeps the level it had
        outgoing.age = 0;
        outgoing.quiet = 0;

        erGain.snap (erGain.getTarget());           // the incoming window fades it in anyway
        ringingW = activeW;
        activeW = slot;
        lastChange = now;
        return true;
    }

    /** The share of the sample written at `s` that belongs to `w`. */
    float windowAt (const World& w, std::int64_t s) const noexcept
    {
        const std::int64_t a = s - w.start;
        if (a < 0)
            return 0.0f;

        const float in = a < fadeLen ? fadeIn[static_cast<size_t> (a)] : 1.0f;

        const std::int64_t b = s - w.end;
        if (b < 0)
            return in;
        if (b >= fadeLen)
            return 0.0f;

        return in * (1.0f - fadeIn[static_cast<size_t> (b)]);
    }

    /** How a world's tail is fed for a chunk starting at write time t0: its
        own window whole (pass), entirely closed (silence), or partly. */
    Feed feedFor (const World& w, std::int64_t t0, int n) const noexcept
    {
        const std::int64_t first = t0 - w.er.tailDelay;     // the oldest sample it reads
        if (first >= w.end + fadeLen)
            return Feed::Silence;
        if (first >= w.start + fadeLen && first + n <= w.end)
            return Feed::Pass;
        return Feed::Window;
    }

    /** One sample of a world's reflections, the sample at write time `s` just
        written. `windowed` weighs each tap by the world's share of what it
        reads; a settled active world skips that. */
    float reflect (World& w, std::int64_t s, bool windowed) noexcept
    {
        const ErTapSet& e = w.er;
        float first = 0.0f, higher = 0.0f;

        for (int j = 0; j < e.numTaps; ++j)
        {
            float g = e.gain[j];

            if (windowed)
            {
                const float k = windowAt (w, s - e.delay[j]);
                if (k == 0.0f)
                    continue;                       // not its sound: never even read
                g *= k;
            }

            const float v = g * erRing.readInteger (e.delay[j] + 1);
            if (j < e.numFirst)
                first += v;
            else
                higher += v;
        }

        w.darkState += e.darkCoef * (higher - w.darkState);
        return first + w.darkState;
    }

    /** 1 - exp(-2*pi*f/sr) without libm, so an offline render of the same
        session hashes the same on every platform. */
    float onePoleCoef (float cutoffHz) const noexcept
    {
        float f = cutoffHz;
        const float ceiling = static_cast<float> (sampleRate) * 0.49f;
        if (f > ceiling)
            f = ceiling;

        const float x = 6.2831853071795864f * f / static_cast<float> (sampleRate);
        return 1.0f - spatcore::dsp::FastDecibels::exp2 (-x * 1.4426950408889634f);
    }

    void setTargets (const ReverbParams& r) noexcept
    {
        predelaySamples.setTarget (clampParam (r.predelayMs, 0.0f, kMaxPredelayMs) * msToSamples);
        toneCoef.setTarget (onePoleCoef (clampParam (r.toneHz, 1000.0f, 20000.0f)));
        wet.setTarget (clampParam (r.mix, 0.0f, 100.0f) * 0.01f);
        erGain.setTarget (spatcore::dsp::FastDecibels::dbToGain (clampParam (r.erLevelDb, kMinErLevelDb, kMaxErLevelDb)));
    }

    /** The wet before tone: the active world alone when settled, every world
        while a change spills over. `x` is the predelayed input on entry. */
    void runWet (float* x, int n) noexcept
    {
        const std::int64_t t0 = now;
        World& a = worlds[activeW];

        if (ringingW < 0 && dyingW < 0 && t0 - a.er.maxRead >= a.start + fadeLen)
        {
            if (! a.er.isOn())
            {
                instance (a.tail).process (x, n);
            }
            else
            {
                float* er = erScratch[0].data();
                for (int i = 0; i < n; ++i)
                {
                    erRing.write (x[i]);
                    er[i] = erGain.next() * reflect (a, t0 + i, false);
                    x[i] = erRing.readInteger (a.er.tailDelay + 1);
                }

                instance (a.tail).process (x, n);

                for (int i = 0; i < n; ++i)
                    x[i] += er[i];
            }

            dirty[a.tail] = true;
            now += n;
            return;
        }

        // Roles: 0 active (fed in place), 1 ringing, 2 dying.
        const int roleWorld[3] = { activeW, ringingW, dyingW };
        float* const roleBuf[3] = { x, ringScratch.data(), dyingScratch.data() };

        Feed feed[3] = { Feed::Silence, Feed::Silence, Feed::Silence };
        bool windowedEr[3] = { true, true, true };
        bool ringLive = false;

        for (int r = 0; r < 3; ++r)
        {
            if (roleWorld[r] < 0)
                continue;

            const World& w = worlds[roleWorld[r]];
            feed[r] = feedFor (w, t0, n);
            windowedEr[r] = ! (w.end == kForever && t0 - w.er.maxRead >= w.start + fadeLen);
            ringLive = ringLive || w.er.isOn();
        }

        for (int i = 0; i < n; ++i)
        {
            const float xi = x[i];
            const std::int64_t s = t0 + i;

            if (ringLive)
                erRing.write (xi);

            // The active world last: it is fed in place.
            for (int r = 2; r >= 0; --r)
            {
                if (roleWorld[r] < 0)
                    continue;

                World& w = worlds[roleWorld[r]];
                const int d = w.er.tailDelay;
                float in;

                if (feed[r] == Feed::Silence)
                {
                    in = 0.0f;
                }
                else if (d == 0)
                {
                    in = feed[r] == Feed::Pass ? xi : xi * windowAt (w, s);
                }
                else if (feed[r] == Feed::Pass)
                {
                    in = erRing.readInteger (d + 1);
                }
                else
                {
                    const float k = windowAt (w, s - d);
                    in = k == 0.0f ? 0.0f : erRing.readInteger (d + 1) * k;
                }

                roleBuf[r][i] = in;

                if (w.er.isOn())
                {
                    const float g = r == 0 ? erGain.next() : w.erGain;
                    erScratch[static_cast<size_t> (r)][static_cast<size_t> (i)] = g * reflect (w, s, windowedEr[r]);
                }
            }
        }

        for (int r = 0; r < 3; ++r)
        {
            if (roleWorld[r] < 0)
                continue;

            World& w = worlds[roleWorld[r]];
            float* b = roleBuf[r];
            instance (w.tail).process (b, n);
            dirty[w.tail] = true;

            if (w.er.isOn())
            {
                const float* er = erScratch[static_cast<size_t> (r)].data();
                for (int i = 0; i < n; ++i)
                    b[i] += er[i];
            }
        }

        if (dyingW >= 0)
        {
            float* ds = roleBuf[2];
            const int pos = worlds[dyingW].fadePos;
            for (int i = 0; i < n; ++i)
            {
                const int k = pos + i;
                ds[i] *= k < fadeLen ? 1.0f - fadeIn[static_cast<size_t> (k)] : 0.0f;
            }
        }

        float ringPeak = 0.0f;
        if (ringingW >= 0)
        {
            const float* rs = roleBuf[1];
            for (int i = 0; i < n; ++i)
            {
                const float v = rs[i];
                const float magnitude = v < 0.0f ? -v : v;
                if (magnitude > ringPeak)
                    ringPeak = magnitude;
                x[i] += v;
            }
        }

        if (dyingW >= 0)
        {
            const float* ds = roleBuf[2];
            for (int i = 0; i < n; ++i)
                x[i] += ds[i];
        }

        now += n;

        if (dyingW >= 0)
        {
            worlds[dyingW].fadePos += n;
            if (worlds[dyingW].fadePos >= fadeLen)
                dyingW = -1;                        // faded out: its voice is idle
        }

        if (ringingW >= 0)
        {
            World& w = worlds[ringingW];
            const bool readsDone = t0 - w.er.maxRead >= w.end + fadeLen;

            w.age += n;
            w.quiet = (readsDone && ringPeak * kWetGain < kQuietGain) ? w.quiet + n : 0;

            if (w.quiet >= quietHoldSamples)
            {
                ringingW = -1;                      // rung out: its voice is idle
            }
            else if (w.age >= maxRingSamples && dyingW < 0)
            {
                dyingW = ringingW;
                worlds[dyingW].fadePos = 0;
                ringingW = -1;
            }
        }
    }

    void processChunk (float* inout, int n, float& peak) noexcept
    {
        float* w = wetScratch.data();

        // Write then read: readLinear(0) is the sample just written, so a
        // predelay of zero hands the model the input sample itself.
        for (int i = 0; i < n; ++i)
        {
            predelay.write (inout[i]);
            w[i] = predelay.readLinear (predelaySamples.next());
        }

        runWet (w, n);

        for (int i = 0; i < n; ++i)
        {
            toneState += toneCoef.next() * (w[i] - toneState);

            const float wetSample = kWetGain * toneState;
            const float magnitude = wetSample < 0.0f ? -wetSample : wetSample;
            if (magnitude > peak)
                peak = magnitude;

            const float g = wet.next();

            if (g >= 1.0f)
                inout[i] = wetSample;
            else if (g > 0.0f)
                inout[i] = inout[i] + g * (wetSample - inout[i]);
            // g <= 0: the dry sample, not touched at all - a crossfade at zero
            // would still turn a negative zero into a positive one.
        }
    }

    FdnReverbModel fdn[kInstancesPerClass];
    bool dirty[kNumInstances] = {};

    World worlds[3];
    int activeW = 0;
    int ringingW = -1;
    int dyingW = -1;
    std::int64_t now = 0;                           // the write time of the next sample
    std::int64_t lastChange = kLongAgo;
    bool transitionWanted = false;
    ReverbParams wanted;

    spatcore::dsp::FractionalDelayLine predelay;
    ReverbDelayLine erRing;
    std::vector<float> wetScratch, ringScratch, dyingScratch;
    std::array<std::vector<float>, 3> erScratch;
    std::vector<float> fadeIn;
    spatcore::dsp::OnePoleSmoother predelaySamples, toneCoef, wet, erGain;

    double sampleRate = 48000.0;
    float msToSamples = 48.0f;
    std::uint32_t noiseKey = 0;
    int maxBlock = 0;
    int fadeLen = 1;
    int quietHoldSamples = 2400;
    int maxRingSamples = 1440000;
    float toneState = 0.0f;
    bool snapOnNextApply = true;

    std::atomic<float> meterDb { spatcore::dsp::FastDecibels::kMinDb };
};

} // namespace spatcore::effects
