#pragma once

#include "EffectModule.h"
#include "modules/TremoloModule.h"
#include "modules/BitcrusherModule.h"
#include "modules/EffectEQModule.h"
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace spatcore::effects
{

/** How a chain obtains its modules. Swappable so tests can drive a chain with
    instrumented stand-ins, and so a consumer can supply its own module set. */
using ModuleFactory = std::unique_ptr<IEffectModule> (*) (ModuleId type, int instance, const ChainConfig& config);

/** The module set shipped so far. Types that are not implemented yet return
    null, which the chain treats as a pass-through slot - so the slot layout,
    the chain order and the whole parameter surface are already final while the
    remaining modules are still being written. */
inline std::unique_ptr<IEffectModule> createModule (ModuleId type, int instance, const ChainConfig& config)
{
    (void) instance;        // the shipped modules are all single-instance so far
    (void) config;

    switch (type)
    {
        case ModuleId::Trem:   return std::make_unique<TremoloModule>();
        case ModuleId::Crush:  return std::make_unique<BitcrusherModule>();
        case ModuleId::EQ:     return std::make_unique<EffectEQModule>();

        case ModuleId::Dist:
        case ModuleId::Dyn:
        case ModuleId::Mod:
        case ModuleId::Phaser:
        case ModuleId::Reverb:
        case ModuleId::Delay:
        case ModuleId::Count:
        default:               return nullptr;
    }
}

/**
    One effects channel's processing chain: eleven slots in a user-chosen order.

    The order is data, not structure. Reordering does not move any module or
    disturb its state; it changes which slot the audio visits next, and the
    switch happens at a block boundary under a short mute so the discontinuity
    is covered rather than heard. A module keeps its tail across a reorder,
    which is what an operator expects when they drag a reverb behind a delay.

    Parameters are re-applied only when `revision` moves. The message thread
    publishes a whole EffectChannelParams per tick through the triple buffer;
    diffing on one integer keeps the per-block cost at a comparison when nothing
    has changed, which is the normal case.

    Chain bypass and mute are separate on purpose. Bypass means "take this
    channel out of the signal path": the slots are skipped and reset, so nothing
    rings on when it comes back. Mute means "silence the output": the modules
    keep running, so a reverb tail that is muted and unmuted has moved on rather
    than resuming where it stopped. Both fade, and both do nothing at all once
    settled, so neither costs anything in the state an operator leaves it in.

    The chain checks its own output for non-finite samples on top of the
    per-slot check, because a slot's last-sample test can miss a memoryless
    module passing a single NaN straight through. Whatever gets past the slots
    is caught before it reaches the return ring.

    Not copyable or movable (the atomics see to that), so an engine holds these
    by pointer: std::vector<std::unique_ptr<EffectChain>>, never a vector of
    chains.
*/
class EffectChain
{
public:
    EffectChain() = default;

    /** Allocates every module and the crossfade scratch. Not realtime-safe. */
    void prepare (const ChainConfig& config, ModuleFactory factory = &createModule)
    {
        maxBlock = config.maxBlock > 0 ? config.maxBlock : 1;
        dry.assign (static_cast<size_t> (maxBlock), 0.0f);

        for (int s = 0; s < kNumModuleSlots; ++s)
            slots[static_cast<size_t> (s)].prepare (config,
                                                    factory != nullptr
                                                        ? factory (kSlots[s].type, kSlots[s].instance, config)
                                                        : nullptr);

        for (auto* envelope : { &reorderEnvelope, &bypassEnvelope, &muteEnvelope })
        {
            envelope->setTimeConstant (config.sampleRate, kFadeTauSeconds);
            envelope->snap (1.0f);
        }

        currentOrder = kDefaultOrder;
        pendingOrder = kDefaultOrder;
        reorderPending = false;
        chainBypassWanted = false;
        muteWanted = false;
        slotsResetWhileBypassed = false;
        primed = false;
        lastRevision = 0;
    }

    void reset() noexcept
    {
        for (auto& slot : slots)
            slot.reset();

        if (reorderPending)
        {
            currentOrder = pendingOrder;
            reorderPending = false;
        }

        reorderEnvelope.snap (1.0f);
        bypassEnvelope.snap (chainBypassWanted ? 0.0f : 1.0f);
        muteEnvelope.snap (muteWanted ? 0.0f : 1.0f);
        slotsResetWhileBypassed = false;
    }

    void process (float* inout, int numSamples, const EffectChannelParams& params) noexcept
    {
        if (inout == nullptr || numSamples <= 0)
            return;

        if (! primed || params.revision != lastRevision)
        {
            applyParams (params);
            primed = true;
            lastRevision = params.revision;
        }

        int offset = 0;
        while (offset < numSamples)
        {
            const int remaining = numSamples - offset;
            const int chunk = remaining < maxBlock ? remaining : maxBlock;
            processChunk (inout + offset, chunk);
            offset += chunk;
        }
    }

    /** Reported, never compensated: the effects return is deliberately left out
        of the calculation engine's latency alignment, or every dry input would
        be delayed to match it. */
    int getLatencySamples() const noexcept
    {
        if (chainBypassWanted)
            return 0;

        int total = 0;
        for (const auto& slot : slots)
            total += slot.getLatencySamples();

        return total;
    }

    const ChainOrder& getCurrentOrder() const noexcept { return currentOrder; }
    bool isReorderPending() const noexcept             { return reorderPending; }

    ModuleSlot& getSlot (int slot) noexcept             { return slots[static_cast<size_t> (slot)]; }
    const ModuleSlot& getSlot (int slot) const noexcept { return slots[static_cast<size_t> (slot)]; }

    /** Non-finite blocks that got past the per-slot guards. */
    std::atomic<std::uint32_t> nanTrips { 0 };

    EffectChain (const EffectChain&) = delete;
    EffectChain& operator= (const EffectChain&) = delete;

private:
    void applyParams (const EffectChannelParams& params) noexcept
    {
        for (int s = 0; s < kNumModuleSlots; ++s)
            slots[static_cast<size_t> (s)].applyParams (params, static_cast<int> (kSlots[s].instance));

        // An invalid order is ignored rather than obeyed: the message thread
        // validates before publishing, so anything else is a bug upstream and
        // the running order is the safer answer.
        if (isValidChainOrder (params.order))
        {
            if (params.order != currentOrder)
            {
                pendingOrder = params.order;
                reorderPending = true;
            }
            else
            {
                reorderPending = false;
            }
        }

        chainBypassWanted = params.chainBypass != 0;
        muteWanted = params.mute != 0;

        reorderEnvelope.setTarget (reorderPending ? 0.0f : 1.0f);
        bypassEnvelope.setTarget (chainBypassWanted ? 0.0f : 1.0f);
        muteEnvelope.setTarget (muteWanted ? 0.0f : 1.0f);
    }

    void resetSlots() noexcept
    {
        for (auto& slot : slots)
            slot.reset();
    }

    void applyMute (float* buf, int n) noexcept
    {
        if (muteEnvelope.isSettled())
        {
            if (muteEnvelope.getCurrent() == 0.0f)
                std::memset (buf, 0, static_cast<size_t> (n) * sizeof (float));

            return;                                  // settled open: no arithmetic
        }

        for (int i = 0; i < n; ++i)
            buf[i] *= muteEnvelope.next();
    }

    void processChunk (float* buf, int n) noexcept
    {
        // The swap happens at a block boundary while the mute envelope is down.
        if (reorderPending && reorderEnvelope.isSettled() && reorderEnvelope.getCurrent() == 0.0f)
        {
            currentOrder = pendingOrder;             // module state is untouched
            reorderPending = false;
            reorderEnvelope.setTarget (1.0f);
        }

        if (bypassEnvelope.isSettled() && bypassEnvelope.getCurrent() == 0.0f)
        {
            if (! slotsResetWhileBypassed)
            {
                resetSlots();
                slotsResetWhileBypassed = true;
            }

            // Silent anyway, so a pending reorder can simply happen.
            if (reorderPending)
            {
                currentOrder = pendingOrder;
                reorderPending = false;
                reorderEnvelope.snap (1.0f);
            }

            applyMute (buf, n);                      // the dry signal, untouched otherwise
            return;
        }

        slotsResetWhileBypassed = false;

        const bool crossfading = ! (bypassEnvelope.isSettled() && bypassEnvelope.getCurrent() == 1.0f);
        if (crossfading)
            std::memcpy (dry.data(), buf, static_cast<size_t> (n) * sizeof (float));

        for (int k = 0; k < kNumModuleSlots; ++k)
            slots[static_cast<size_t> (currentOrder[static_cast<size_t> (k)])].process (buf, n);

        if (! std::isfinite (buf[n - 1]))
        {
            std::memset (buf, 0, static_cast<size_t> (n) * sizeof (float));
            resetSlots();
            nanTrips.fetch_add (1, std::memory_order_relaxed);
        }

        if (crossfading)
        {
            for (int i = 0; i < n; ++i)
            {
                const float g = bypassEnvelope.next();

                if (g <= 0.0f)
                    buf[i] = dry[static_cast<size_t> (i)];
                else if (g < 1.0f)
                    buf[i] = dry[static_cast<size_t> (i)] + g * (buf[i] - dry[static_cast<size_t> (i)]);
            }
        }

        if (! (reorderEnvelope.isSettled() && reorderEnvelope.getCurrent() == 1.0f))
        {
            for (int i = 0; i < n; ++i)
                buf[i] *= reorderEnvelope.next();
        }

        applyMute (buf, n);
    }

    std::array<ModuleSlot, kNumModuleSlots> slots;
    ChainOrder currentOrder = kDefaultOrder;
    ChainOrder pendingOrder = kDefaultOrder;

    spatcore::dsp::OnePoleSmoother reorderEnvelope, bypassEnvelope, muteEnvelope;
    std::vector<float> dry;

    bool reorderPending = false;
    bool chainBypassWanted = false;
    bool muteWanted = false;
    bool slotsResetWhileBypassed = false;
    bool primed = false;
    std::uint32_t lastRevision = 0;
    int maxBlock = 0;
};

} // namespace spatcore::effects
