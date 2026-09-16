#pragma once

#include "EffectsTypes.h"
#include "EffectParams.h"
#include "../dsp/OnePoleSmoother.h"
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace spatcore::effects
{

/**
    What a module is prepared with.

    A struct rather than an argument list because phases 2 and 3 add fields to
    it (the reverb module's delay budget, the delay module's buffer cap, a noise
    key per channel), and every module signature would otherwise churn each
    time. The app's EffectsEngine::Config maps onto this.
*/
struct ChainConfig
{
    double sampleRate = 48000.0;
    int maxBlock = 512;
    int reverbMaxDelaySamples = 16384;      // the FDN's budget, per instance
    double maxEffectDelaySeconds = 5.0;     // sizes the delay module's buffer
    std::uint32_t noiseKey = 1;             // per-channel seed for keyed noise
};

/** Slot and chain fades. A TIME CONSTANT, not a completion time: a fade covers
    63 % in one tau and arrives exactly after roughly nine. */
inline constexpr float kFadeTauSeconds = 0.005f;

/** In-module glides for continuous values (gain, mix, depth). */
inline constexpr float kParamTauSeconds = 0.010f;

/** What a module reports back when it is handed a new parameter set.

    variantPending is a STATE, not an edge: "I am holding a change that needs
    the output to be silent first". A module that is handed the value it is
    already running reports false again, so an edit made and then taken back
    before the fade completes cancels itself, with no audible dip and no reset. */
struct ParamApplyInfo
{
    bool bypass = true;
    bool variantPending = false;
};

/**
    One processing module.

    A module owns its own corner of EffectChannelParams and picks it out itself
    (p.trem, p.eq[instance], ...) rather than being handed a pre-selected
    struct. That is what keeps ModuleSlot and EffectChain free of any knowledge
    of module types: they route audio and run fades, and adding a module type
    touches the factory and nothing else.

    applyParams also answers two questions the slot cannot work out on its own:
    whether the module wants to be bypassed, and whether the change needs the
    audio to be silent first. The second is for parameters that cannot be
    interpolated - a different reverb size, a different chorus mode, a different
    decimation filter. The module stages such a change as pending and KEEPS
    RUNNING THE OLD ONE; the slot fades out, calls reset() and then
    commitPendingVariant() while the output is silent, and fades back in. A
    change that is revoked before the fade completes costs nothing.

    prepare() may allocate and is never called on the audio thread. reset(),
    applyParams(), process() and the getters are realtime: noexcept, no
    allocation, no locks.
*/
class IEffectModule
{
public:
    virtual ~IEffectModule() = default;

    virtual ModuleId type() const noexcept = 0;

    /** Allocates and sizes everything. Not realtime-safe. */
    virtual void prepare (const ChainConfig& config) = 0;

    /** Drop audio state (tails, filter memory, phases); keep parameters, and
        snap any smoother to its target so the next block starts settled. */
    virtual void reset() noexcept = 0;

    /** Take the parameters that belong to this module and this instance. */
    virtual ParamApplyInfo applyParams (const EffectChannelParams& params, int instance) noexcept = 0;

    /** Called by the slot at silence, after reset(), when applyParams reported
        a variant change. Modules with no variant parameters need not override. */
    virtual void commitPendingVariant() noexcept {}

    /** Mono, in place, realtime. */
    virtual void process (float* inout, int numSamples) noexcept = 0;

    /** Reported, never compensated (the effects plan's latency ledger). */
    virtual int getLatencySamples() const noexcept = 0;

    /** Optional meter value for the GUI - gain reduction, level, whatever suits
        the module. Read across threads, so back it with a relaxed atomic. */
    virtual float getMeterDb() const noexcept { return 0.0f; }
};

/**
    One slot of a chain: a module, its bypass crossfade, and the safety net.

    Three properties the rest of the engine relies on:

    1. A settled slot does no arithmetic. Bypassed, the buffer is not touched at
       all; active, the module's output goes straight through. A crossfade of
       dry against wet at g = 1 is NOT the identity in floating point - it turns
       -0.0 into +0.0, for one - so "bypassed is bit-transparent" would be a
       claim rather than a fact if the fade ran all the time.

    2. A module that has faded to silence is reset once, so its tail cannot
       reappear when it is switched back on, and it is then skipped entirely
       until it is wanted again.

    3. A module that produces a non-finite sample is silenced, reset and
       counted, rather than being allowed to poison the rest of the chain, the
       return ring and eventually the speakers. The test is the last sample of
       the block: cheap, and enough for the recursive structures (filters,
       feedback loops) where this actually happens - once they go bad they stay
       bad. A memoryless module could pass a single NaN through untripped, which
       is why the chain checks its own output too.

    A slot with no module is a legitimate pass-through: the chain has 11 slots
    from the start, and the module types that are not implemented yet simply
    have none.
*/
class ModuleSlot
{
public:
    ModuleSlot() = default;

    /** Takes ownership. A null module makes this a pass-through slot. */
    void prepare (const ChainConfig& config, std::unique_ptr<IEffectModule> newModule)
    {
        module = std::move (newModule);
        maxBlock = config.maxBlock > 0 ? config.maxBlock : 1;
        dry.assign (static_cast<size_t> (maxBlock), 0.0f);

        fade.setTimeConstant (config.sampleRate, kFadeTauSeconds);
        fade.snap (0.0f);                       // every module starts bypassed
        bypassWanted = true;
        variantPending = false;

        if (module != nullptr)
            module->prepare (config);
    }

    void reset() noexcept
    {
        if (module != nullptr)
        {
            module->reset();

            if (variantPending)
            {
                module->commitPendingVariant();
                variantPending = false;
            }
        }

        fade.snap (bypassWanted ? 0.0f : 1.0f);
    }

    void applyParams (const EffectChannelParams& params, int instance) noexcept
    {
        if (module == nullptr)
            return;

        const ParamApplyInfo info = module->applyParams (params, instance);

        bypassWanted = info.bypass;
        variantPending = info.variantPending;

        fade.setTarget ((bypassWanted || variantPending) ? 0.0f : 1.0f);
    }

    void process (float* inout, int numSamples) noexcept
    {
        if (module == nullptr || inout == nullptr || numSamples <= 0)
            return;

        int offset = 0;
        while (offset < numSamples)
        {
            const int remaining = numSamples - offset;
            const int chunk = remaining < maxBlock ? remaining : maxBlock;
            processChunk (inout + offset, chunk);
            offset += chunk;
        }
    }

    bool hasModule() const noexcept        { return module != nullptr; }
    IEffectModule* getModule() noexcept    { return module.get(); }

    bool isBypassedSettled() const noexcept { return fade.isSettled() && fade.getCurrent() == 0.0f; }
    bool isActiveSettled() const noexcept   { return fade.isSettled() && fade.getCurrent() == 1.0f; }
    float getFadeGain() const noexcept      { return fade.getCurrent(); }

    int getLatencySamples() const noexcept
    {
        return (module != nullptr && ! bypassWanted) ? module->getLatencySamples() : 0;
    }

    float getMeterDb() const noexcept
    {
        return module != nullptr ? module->getMeterDb() : 0.0f;
    }

    std::atomic<std::uint32_t> nanTrips { 0 };
    std::atomic<std::uint32_t> silentResets { 0 };

private:
    void commitVariantAtSilence() noexcept
    {
        module->commitPendingVariant();
        variantPending = false;
        fade.setTarget (bypassWanted ? 0.0f : 1.0f);
    }

    void runModule (float* buf, int n) noexcept
    {
        module->process (buf, n);

        if (! std::isfinite (buf[n - 1]))
        {
            std::memset (buf, 0, static_cast<size_t> (n) * sizeof (float));
            module->reset();
            nanTrips.fetch_add (1, std::memory_order_relaxed);
        }
    }

    void processChunk (float* buf, int n) noexcept
    {
        // Settled and silent: commit anything waiting on silence, then either
        // leave the buffer completely alone or start fading back in.
        if (isBypassedSettled())
        {
            if (variantPending)
            {
                module->reset();
                commitVariantAtSilence();
            }

            if (fade.getTarget() == 0.0f)
                return;
        }

        // Settled and fully wet: the module's output, untouched by any fade.
        if (isActiveSettled() && fade.getTarget() == 1.0f)
        {
            runModule (buf, n);
            return;
        }

        std::memcpy (dry.data(), buf, static_cast<size_t> (n) * sizeof (float));
        runModule (buf, n);

        for (int i = 0; i < n; ++i)
        {
            const float g = fade.next();

            if (g <= 0.0f)
                buf[i] = dry[static_cast<size_t> (i)];
            else if (g < 1.0f)
                buf[i] = dry[static_cast<size_t> (i)] + g * (buf[i] - dry[static_cast<size_t> (i)]);
            // g >= 1: the wet sample already in the buffer
        }

        if (isBypassedSettled())
        {
            module->reset();
            silentResets.fetch_add (1, std::memory_order_relaxed);

            if (variantPending)
                commitVariantAtSilence();
        }
    }

    std::unique_ptr<IEffectModule> module;
    std::vector<float> dry;
    spatcore::dsp::OnePoleSmoother fade;

    bool bypassWanted = true;
    bool variantPending = false;
    int maxBlock = 0;
};

} // namespace spatcore::effects
