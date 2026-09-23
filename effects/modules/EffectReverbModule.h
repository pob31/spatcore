#pragma once

#include "../EffectModule.h"
#include "../EffectPresets.h"
#include "../../dsp/FastDecibels.h"
#include "../../dsp/FractionalDelayLine.h"
#include "../../dsp/FrDiffusionModel.h"
#include "../../dsp/OnePoleSmoother.h"
#include "../../reverb/ReverbFDNAlgorithm.h"
#include "reverb/ReverbLfo.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <algorithm>
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
    Reverb: predelay, a tail from the selected model, tone, and a wet mix.

    The wet path is predelay -> tail -> one-pole low pass -> make-up -> mix.
    The dry path is the buffer, untouched.

    THE TAIL IS A POOL, NOT A MODEL. Every tail class is built twice, in
    prepare(). One instance is ACTIVE and takes the input. A change that needs
    another topology - a model of another class, another size - does not fade
    the reverb out: the idle twin is rebuilt at the new topology, the input is
    crossfaded onto it over 5 ms, and the instance that was active RINGS OUT
    with the values it had, fed silence, until its tail is gone. That is
    spillover, what a hardware reverb does on a program change: a cue that
    changes the room does not chop the tail that is still in the air. So the
    module never reports a variant, and the slot never fades it for one.

    - The input is PARTITIONED, never doubled or dropped: the incoming instance
      takes w(k) of it and the outgoing one 1 - w(k), w a raised cosine. Every
      gain in the pool moves continuously, so no transition can click,
      whatever arrives when.
    - One crossfade at a time. A change that arrives during one waits for it -
      5 ms at most - while the input carries on moving across.
    - At most three instances run: the active one, one RINGING, and one DYING,
      a ringing instance whose voice the next change needed, faded out over
      5 ms. Rapid changes cost the oldest tail, never a click.
    - A ringing instance goes idle once its output has stayed under -96 dBFS
      for 50 ms, and is faded out after 30 s whatever it is doing.
    - Nothing to spill, nothing spilled: while no instance has run since the
      last reset - the first set after prepare(), a bypassed slot - a change
      takes effect at once.
    - Settled, with one instance and no transition, the path is the
      single-model path it always was, sample for sample: no crossfade, no
      summing with a zero.

    The wet leg carries a fixed gain of two, measured rather than guessed. The
    FDN's own times-four is a level CORRECTION, not headroom, and with it alone
    a fully wet reverb comes out 11 dB under the dry on white noise at the
    default decay - a mix control would then spend its whole travel making the
    reverb audible instead of balancing it. Doubling puts a 0.6 s room about
    8 dB under the dry, a 1.5 s hall 5 dB under and a 5 s cathedral level with
    it, which is a spread a mix control can work either side of. Two is exact in
    binary, so it costs the wet no accuracy.

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
    the capacity prepare() gave it - see FdnReverbModel. About 480 KiB per
    module at 48 kHz (two networks sized for 2.5, plus a quarter-second
    predelay ring), doubling with the rate.
*/
class EffectReverbModule final : public IEffectModule
{
public:
    static constexpr float kMaxPredelayMs = 250.0f;

    /** Wet make-up on top of the FDN's own +12 dB correction. A measured
        figure, not a round one - see the class note. */
    static constexpr float kWetGain = 2.0f;

    /** The input crossfade onto an incoming instance, and a dying one's fade. */
    static constexpr double kSpillFadeSeconds = 0.005;

    /** A ringing instance is idle once its wet output has stayed under
        kQuietGain (-96 dBFS) for kQuietHoldSeconds, and is faded out after
        kMaxRingSeconds whatever it is doing. */
    static constexpr float kQuietGain = 1.5849e-5f;
    static constexpr double kQuietHoldSeconds = 0.05;
    static constexpr double kMaxRingSeconds = 30.0;

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

        predelay.prepare (static_cast<int> (std::ceil (kMaxPredelayMs * 0.001 * sampleRate)) + 2);
        wetScratch.assign (static_cast<size_t> (maxBlock), 0.0f);
        ringScratch.assign (static_cast<size_t> (maxBlock), 0.0f);
        dyingScratch.assign (static_cast<size_t> (maxBlock), 0.0f);

        predelaySamples.setTimeConstant (sampleRate, kParamTauSeconds);
        predelaySamples.setSnapEpsilon (1.0e-3f);           // samples, not a unit gain
        toneCoef.setTimeConstant (sampleRate, kParamTauSeconds);
        toneCoef.setSnapEpsilon (1.0e-6f);                  // a one-pole coefficient
        wet.setTimeConstant (sampleRate, kParamTauSeconds);

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

        active = firstOfClass (classFor (defaults));
        ringing = -1;
        dying = -1;
        transitionWanted = false;

        setTargets (defaults);
        snapOnNextApply = true;
        reset();
    }

    void reset() noexcept override
    {
        predelay.reset();
        toneState = 0.0f;

        for (int k = 0; k < kNumInstances; ++k)
            clearIfDirty (k);

        ringing = -1;
        dying = -1;
        xfadePos = fadeLen;

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

        meterDb.store (spatcore::dsp::FastDecibels::kMinDb, std::memory_order_relaxed);
    }

    ParamApplyInfo applyParams (const EffectChannelParams& params, int) noexcept override
    {
        const ReverbParams& r = params.reverb;

        setTargets (r);

        if (! needsTransition (r))
        {
            transitionWanted = false;               // a change taken back before it could start
            instance (active).setParams (r);        // the runtime values
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

    /** Instances ringing out or fading under the active one: 0, 1 or 2. */
    int getSpillVoices() const noexcept      { return (ringing >= 0 ? 1 : 0) + (dying >= 0 ? 1 : 0); }

    /** True while a change waits for the pool to free a voice. */
    bool isTransitionWaiting() const noexcept { return transitionWanted; }

    /** The Size the ACTIVE instance is built at - what the input is going to. */
    float getActiveSize() const noexcept     { return instance (active).getBuiltSize(); }

private:
    static float clampParam (float v, float lo, float hi) noexcept
    {
        if (! (v > lo)) return lo;
        if (v > hi)     return hi;
        return v;
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
        return classFor (r) != classOfInstance (active) || instance (active).isBuildDifferent (r);
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

    /** No tail to spill: the wanted topology becomes the active one at once. */
    void switchDirectly (const ReverbParams& r) noexcept
    {
        const int cls = classFor (r);
        if (classOfInstance (active) != cls)
        {
            clearIfDirty (active);
            active = firstOfClass (cls);
        }

        IEffectReverbModel& m = instance (active);
        if (m.setParams (r))
            m.commitPendingVariant();
    }

    /** Starts a spillover onto an idle instance built at `r`. False while the
        pool cannot take one yet - a crossfade still running, or the voice it
        needs still fading out - and the next block tries again. */
    bool tryStartTransition (const ReverbParams& r) noexcept
    {
        if (xfadePos < fadeLen)
            return false;

        // The active instance is about to ring, so the one ringing now has to
        // go. Fed silence since its own crossfade ended, only its output needs
        // the fade.
        if (ringing >= 0)
        {
            if (dying >= 0)
                return false;

            dying = ringing;
            dyingPos = 0;
            ringing = -1;
        }

        const int cls = classFor (r);
        int target = -1;
        for (int k = firstOfClass (cls); k < firstOfClass (cls) + kInstancesPerClass; ++k)
        {
            if (k != active && k != dying)
            {
                target = k;
                break;
            }
        }

        if (target < 0)
            return false;                           // its twin is the one fading out

        IEffectReverbModel& m = instance (target);
        if (m.setParams (r))
            m.commitPendingVariant();               // inside the capacity prepare() gave it
        clearIfDirty (target);

        ringing = active;
        ringAge = 0;
        ringQuiet = 0;
        active = target;
        xfadePos = 0;
        return true;
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
    }

    /** The tail: the active instance alone when settled, the pool while a
        change spills over. */
    void runTail (float* w, int n) noexcept
    {
        if (ringing < 0 && dying < 0 && xfadePos >= fadeLen)
        {
            instance (active).process (w, n);
            dirty[active] = true;
            return;
        }

        const bool crossfading = xfadePos < fadeLen;
        float* rs = ringScratch.data();
        float* ds = dyingScratch.data();

        // The partition: the incoming instance takes w * fadeIn, the outgoing
        // one the rest - and silence once the crossfade is over.
        if (crossfading)
        {
            for (int i = 0; i < n; ++i)
            {
                const int k = xfadePos + i;
                const float g = k < fadeLen ? fadeIn[static_cast<size_t> (k)] : 1.0f;
                rs[i] = w[i] * (1.0f - g);
                w[i] = w[i] * g;
            }
        }
        else if (ringing >= 0)
        {
            std::fill (rs, rs + n, 0.0f);
        }

        if (ringing >= 0)
        {
            instance (ringing).process (rs, n);
            dirty[ringing] = true;
        }

        if (dying >= 0)
        {
            std::fill (ds, ds + n, 0.0f);
            instance (dying).process (ds, n);
            dirty[dying] = true;

            for (int i = 0; i < n; ++i)
            {
                const int k = dyingPos + i;
                ds[i] *= k < fadeLen ? 1.0f - fadeIn[static_cast<size_t> (k)] : 0.0f;
            }
        }

        instance (active).process (w, n);
        dirty[active] = true;

        float ringPeak = 0.0f;
        if (ringing >= 0)
        {
            for (int i = 0; i < n; ++i)
            {
                const float v = rs[i];
                const float magnitude = v < 0.0f ? -v : v;
                if (magnitude > ringPeak)
                    ringPeak = magnitude;
                w[i] += v;
            }
        }

        if (dying >= 0)
            for (int i = 0; i < n; ++i)
                w[i] += ds[i];

        if (crossfading)
            xfadePos = (fadeLen - xfadePos) > n ? xfadePos + n : fadeLen;

        if (dying >= 0)
        {
            dyingPos += n;
            if (dyingPos >= fadeLen)
                dying = -1;                         // faded out: idle, cleared when next taken
        }

        if (ringing >= 0)
        {
            ringAge += n;
            ringQuiet = (! crossfading && ringPeak * kWetGain < kQuietGain) ? ringQuiet + n : 0;

            if (ringQuiet >= quietHoldSamples)
            {
                ringing = -1;                       // rung out: idle, cleared when next taken
            }
            else if (ringAge >= maxRingSamples && dying < 0)
            {
                dying = ringing;
                dyingPos = 0;
                ringing = -1;
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

        runTail (w, n);

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

    int active = 0;
    int ringing = -1;
    int dying = -1;
    int xfadePos = 0;
    int dyingPos = 0;
    int ringAge = 0;
    int ringQuiet = 0;
    bool transitionWanted = false;
    ReverbParams wanted;

    spatcore::dsp::FractionalDelayLine predelay;
    std::vector<float> wetScratch, ringScratch, dyingScratch;
    std::vector<float> fadeIn;
    spatcore::dsp::OnePoleSmoother predelaySamples, toneCoef, wet;

    double sampleRate = 48000.0;
    float msToSamples = 48.0f;
    int maxBlock = 0;
    int fadeLen = 1;
    int quietHoldSamples = 2400;
    int maxRingSamples = 1440000;
    float toneState = 0.0f;
    bool snapOnNextApply = true;

    std::atomic<float> meterDb { spatcore::dsp::FastDecibels::kMinDb };
};

} // namespace spatcore::effects
