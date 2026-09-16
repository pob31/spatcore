#pragma once

#include "../dsp/FastDecibels.h"
#include "../dsp/OnePoleSmoother.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace spatcore::effects
{

/**
    Per-channel runaway guard on one effects channel's effect-to-effect feed.

    Effects channels are allowed to feed each other and the design deliberately
    does not limit the loop gain: bunches of channels feeding one another is
    something an operator builds on purpose. A loop whose gain exceeds unity
    nevertheless builds without bound, and with 32 channels in a sends grid one
    can be created by accident while reaching for something else. This class is
    what stops a building loop from reaching the speakers.

    It attenuates ONLY the effect-to-effect part of a channel's feed. That part
    can be isolated because the engine renders it into the channel's scratch row
    on its own (AcousticSendMatrix::computeNodeFeed with srcBegin/srcEnd) before
    accumulating the input-to-effect part on top with clearDest = false. An
    input-to-effect feed cannot run away, so the guard must never touch one: a
    trip has to cost the operator their loop, never their inputs.

    Per batch, per channel, inside the worker item:

        const float feedPeak = LoopGuard::peakOf (scratchRow, n);   // BEFORE the gain
        guard.observeBlock (feedPeak, lastReturnPeak, n);
        guard.applyGain (scratchRow, n);

    Call the two in that order with the same numSamples. observeBlock moves the
    timers and decides what the gain is heading for, applyGain is what actually
    advances the ramp.

    WHAT THE CALLER OWES THIS CLASS, and what breaks when it does not:

      - THE FEED PEAK MUST COME FROM peakOf(). A hand-rolled running max
        (`if (a > peak) peak = a;`) drops a NaN, because every comparison
        against one is false, so a channel that has gone non-finite reads as a
        peak of exactly 0 and the guard sees the calmest channel in the room.
        peakOf() IS the NaN policy; a guard fed by anything else does not have
        one, however carefully the state machine below is written.
      - observeBlock() AND applyGain() ARE A PAIR: every batch, in that order,
        same length. applyGain is what advances the ramp, and the release clock
        is gated on the feed having actually reached silence (isSettledAt (0)),
        so a batch that observes without applying freezes the ramp part-way down
        and that channel never releases. Skipping both is fine (a stopped
        engine); skipping one is not.
      - reset() BELONGS TO THE DRIVER THREAD, not to a worker item. The engine
        calls it from honourClearRequests(), which runs inside processBatch()
        before the sweep forks, so no worker can be inside observeBlock() or
        applyGain() on the same guard while it happens.
      - THE RAMP SECONDS ARE COMPLETION TIMES, not time constants. The plan
        quotes the fade as "5 ms away, 50 ms back" and that is exactly what
        these arguments mean; the conversion to a one-pole tau happens in here
        (kTausToSettle). A caller that passes a tau gets a ramp nine times too
        slow, with no diagnostic.

    WHY THE FEED PEAK IS TAKEN BEFORE THE GAIN. The guard sits inside the loop
    it is watching, so anything measured downstream of its own attenuation
    describes the guard rather than the loop. The pre-gain peak keeps meaning
    one thing whether the guard is engaged or not: how hard the other effects
    channels are pushing this one. Measuring after the multiply would collapse
    the moment the guard acted, the guard would read that as calm, release, and
    the pair would oscillate at the ramp time.

    WHY THE THRESHOLDS ARE TIMES AND NOT BLOCK COUNTS. The plan specifies "more
    than 20 consecutive blocks", which is 27 ms at a 64-sample buffer and 213 ms
    at 512, or 232 ms once the machine is also at 44.1 kHz: the same
    user-visible event behaving eight or nine times differently depending on
    settings the operator may not know they changed, and eight or nine times as
    much of the runaway reaching the speakers on the machine with the bigger
    buffer. Everything here is in seconds, converted against the sample rate at
    prepare, so a trip lands at the same wall-clock point to within one block at
    any buffer size.

    WHY THE RELEASE IS NOT ON THE RETURN PEAK. The plan releases when the RETURN
    peak stays 12 dB under the ceiling for 500 ms. That pairing does not work,
    because the return sits on the far side of the chain from the actuator:

      - a channel whose chain sustains itself (a delay at unity feedback, a long
        or frozen reverb) keeps its return hot with the effect-to-effect bus
        already silenced, so the release never fires and the guard stays latched
        on a channel whose loop the operator removed minutes ago. Only Clear
        gets it back;
      - a channel with a memoryless chain (an EQ) has its return collapse the
        instant the bus is attenuated, so the release fires after exactly the
        release time whether or not the loop is still dangerous.

    So the return peak measures the chain's decay time rather than the loop, and
    the two thresholds sit on different buses with a chain's gain between them,
    which is not hysteresis in any useful sense. Both decisions are therefore
    taken on the same signal, the pre-gain effect-to-effect feed peak: trip
    above the ceiling for the trip time, release below the ceiling minus the
    hysteresis for the release time. One bus, one pair of thresholds, a real
    hysteresis band between them.

    The return peak is still used, in the safe direction only: it can DELAY a
    release, never cause one. While the channel's own output is above the
    ceiling, restoring its loop feed would re-pump the loop immediately, so the
    calm timer is held at zero until the return comes back under.

    THE VETO IS BOUNDED, and the bound is not a detail. The guard deliberately
    never touches input-to-effect feeds, so a channel can sit permanently over
    the ceiling with no loop anywhere near it: compressor makeup, distortion
    drive, a long reverb tail, a hot mix. An unbounded return veto would hold
    such a channel's effect-to-effect sends down for the rest of the show, badge
    lit, with no way back but Clear - which is the latching failure this class
    was written to get away from, coming back in through the veto. So the veto
    gets a budget of kVetoReleaseTimes release times of otherwise-calm,
    fully-attenuated feed, after which the feed decides alone. If there really
    is a loop, restoring it re-pumps it, the guard trips again within the trip
    time, and the backoff ladder doubles the hold: bounded and audible, which is
    the trade this whole class already makes.

    WHAT IT STILL CANNOT DO. The guard is inside the loop, so no measurement it
    can take distinguishes "the danger has gone" from "the guard is working": as
    soon as it attenuates, the loop decays and the feed goes calm. A release
    into a cycle the operator has not fixed therefore lets the loop build again,
    and the guard trips again. That limit cycle is bounded but audible, so each
    re-trip doubles the hold time (1x, 2x, 4x of the release time), and the
    ladder is forgotten once the channel has been calm and fully restored for a
    while. A channel that keeps tripping ends up held down for four release
    times out of every cycle, its trip count climbs and the GUI badge stays lit,
    which is the signal that a human has to remove the send. The guard buys time
    and headroom; it is not a loop-gain limiter (plan item 9b).

    NO LIBM ON THE AUDIO PATH. std::exp is reached exactly where a ramp STARTS -
    trip() and beginRelease(), once per state change - and from prepare(), which
    is not realtime. reset() deliberately does not touch the coefficient: it
    snaps, and whichever ramp starts next sets its own, so an emergency Clear
    across 32 guards makes no library call at all. dB to linear goes through
    FastDecibels rather than std::pow, so both thresholds are bit-identical on
    every platform and two machines trip on the same sample.

    NO ATOMICS, BY CONSTRUCTION. The engine keeps one guard per channel in a
    plain std::vector, which EffectChain and RtTripleBuffer cannot manage
    because their atomics make them non-movable. Nothing here is atomic and
    nothing here allocates, so a vector of guards resizes on a channel-count
    change like any other POD. A guard is touched by exactly one thread at a
    time: the worker item that owns its channel, with AudioParallelFor's fork
    and join supplying the happens-before edge between batches. getTripCount()
    is a plain counter for the driver to mirror into the engine's telemetry
    block after the join; the GUI reads that atomic, never this.
*/
class LoopGuard
{
public:
    /** Armed: watching, feed at unity. Tripped: ramping to, or held at,
        silence. Releasing: ramping back, and watching for a re-trip the whole
        way up. */
    enum class State : std::uint8_t
    {
        Armed = 0,
        Tripped,
        Releasing
    };

    //==========================================================================
    /** Plan defaults (section 2.2): +6 dBFS peak ceiling, 5 ms away, 500 ms of
        calm, 50 ms back. The trip time is 60 ms because that is what the plan's
        own venue check measures ("the loop guard trips within ~60 ms"); the 20
        blocks it specifies instead spans 27 ms to 232 ms depending on the buffer
        and the sample rate, and 60 ms sits inside that spread. */
    static constexpr float  kDefaultCeilingDb       = 6.0f;
    static constexpr double kDefaultTripSeconds     = 0.060;
    static constexpr double kDefaultReleaseSeconds  = 0.500;
    static constexpr double kDefaultRampDownSeconds = 0.005;
    static constexpr double kDefaultRampUpSeconds   = 0.050;
    static constexpr double kDefaultHysteresisDb    = 12.0;

    /** How close to its endpoint the ramp has to get before it takes it
        exactly: -80 dB on a unit gain. It is the reason "settled" can mean
        bit-exactly 0 or bit-exactly 1 rather than nearly, which in turn is what
        lets an armed guard skip the buffer entirely and a held one memset. */
    static constexpr float kSnapEpsilon = 1.0e-4f;

    /** ln (1 / kSnapEpsilon). A one-pole covers 63 % of its distance in one
        time constant and lands on its endpoint after this many, so a ramp
        quoted as a completion time converts to a time constant by dividing.
        The same arithmetic as EffectModule's kFadeTauSeconds. */
    static constexpr double kTausToSettle = 9.210340371976184;

    /** Hold-time ladder for a loop the operator has not fixed: 1x, 2x, 4x. */
    static constexpr int kMaxBackoffSteps = 3;

    /** Armed, restored and calm for this many release times forgets the ladder,
        so a channel tripped twice this morning does not carry a 2 s hold into
        tonight's show. */
    static constexpr double kBackoffForgetFactor = 4.0;

    /** How long a hot return may hold a release off: this many release times of
        calm, fully attenuated feed, after which the feed decides alone. Without
        a bound, a channel that is merely loud - and has no loop at all - loses
        its effect-to-effect sends until someone hits Clear. */
    static constexpr double kVetoReleaseTimes = 4.0;

    //==========================================================================
    LoopGuard() noexcept
    {
        gain.setSnapEpsilon (kSnapEpsilon);
        gain.snap (1.0f);
    }

    /** Allocation-free, so re-preparing a live engine at a batch boundary is
        safe. Zeroes the trip count; reset() does not.

        It is not click-free, though: like reset(), it snaps a held feed back to
        unity in one sample. Re-preparing a live engine is a rebuild, and a
        rebuild is already a discontinuity on every other bus in the engine.

        There is no maxBlock argument: the guard buffers nothing, so there is
        nothing to size.

        THE ORDER OF THE FIRST TWO ARGUMENTS IS LOAD-BEARING. guardEnabled sits
        second because that is where EffectsEngineCore's call site passes
        Config::loopGuardEnabled. Drop it and every argument after it shifts one
        place - and since bool, float and double all convert to one another
        silently, such a call compiles clean at /W4 while turning a 0.25 s trip
        time into a 6 s one. The deleted overload below is what makes that shift
        a compile error rather than six seconds of runaway.

        @param newSampleRate     what the timers are counted against
        @param guardEnabled      effectsGlobalLoopGuard: false makes observeBlock
                                 and applyGain no-ops, for the operator who is
                                 building a feedback bunch on purpose
        @param ceilingDb         peak dBFS the effect-to-effect feed may reach
        @param newTripSeconds    continuous time above the ceiling before the
                                 feed is ramped away; 0 trips on the first block
        @param newReleaseSeconds continuous calm before it is restored
        @param rampDownSeconds   COMPLETION time of the ramp to silence, not a
                                 time constant
        @param rampUpSeconds     completion time of the ramp back
        @param hysteresisDb      how far under the ceiling counts as calm */
    void prepare (double newSampleRate,
                  bool   guardEnabled      = true,
                  float  ceilingDb         = kDefaultCeilingDb,
                  double newTripSeconds    = kDefaultTripSeconds,
                  double newReleaseSeconds = kDefaultReleaseSeconds,
                  double rampDownSeconds   = kDefaultRampDownSeconds,
                  double rampUpSeconds     = kDefaultRampUpSeconds,
                  double hysteresisDb      = kDefaultHysteresisDb) noexcept
    {
        sampleRate = newSampleRate > 0.0 ? newSampleRate : 48000.0;
        invSampleRate = 1.0 / sampleRate;
        enabled = guardEnabled;

        // FastDecibels rather than std::pow: libm-free and identical on every
        // platform, so two machines trip at the same sample.
        ceilingLinear = spatcore::dsp::FastDecibels::dbToGain (ceilingDb);

        // A negative hysteresis would put the release threshold ABOVE the
        // ceiling, which is a latch in the other direction, so it clamps to
        // zero. Zero itself is legal and means "release as soon as the feed is
        // no longer loud": the calm test is <= and the loud test is ! (<=), so
        // between them they tile the number line and no peak is neither.
        const float hysteresis = hysteresisDb > 0.0 ? static_cast<float> (hysteresisDb) : 0.0f;
        releaseLinear = ceilingLinear * spatcore::dsp::FastDecibels::dbToGain (-hysteresis);

        tripSeconds = newTripSeconds > 0.0 ? newTripSeconds : 0.0;
        releaseSeconds = newReleaseSeconds > 0.0 ? newReleaseSeconds : 0.0;

        rampDownTau = tauForSettleTime (rampDownSeconds);
        rampUpTau = tauForSettleTime (rampUpSeconds);

        prepared = true;
        reset();
        tripCount = 0;

        // Once, off the audio path, so the smoother never holds a coefficient
        // nobody set. trip() and beginRelease() each set their own.
        gain.setTimeConstant (sampleRate, rampUpTau);
    }

    /** A TRIP-WIRE, not an API. Anything but a bool in the second position is
        the old parameter list arriving at a signature that has grown one, and
        every argument in it converts silently. Deleting this template turns
        that into a compile error naming this line; write the switch out
        (prepare (sr, true, ...)). */
    template <typename NotABool, typename... Rest,
              typename = std::enable_if_t<! std::is_same_v<std::decay_t<NotABool>, bool>>>
    void prepare (double newSampleRate, NotABool secondArgument, Rest... rest) = delete;

    /** The live effectsGlobalLoopGuard switch: realtime-safe and allocation
        free, so the driver can flip it at a batch boundary without a rebuild.

        Turning the guard OFF re-arms it. A guard that is no longer watching
        must not leave a feed held down, so it snaps back to unity exactly as
        Clear does. That step is deliberate: an operator switching the guard off
        mid-trip is asking for their loop back now. */
    void setEnabled (bool shouldBeEnabled) noexcept
    {
        if (shouldBeEnabled == enabled)
            return;

        enabled = shouldBeEnabled;

        if (! enabled)
            reset();
    }

    bool isEnabled() const noexcept { return enabled; }

    /** Armed again with the feed at full gain and no trip in progress: what the
        emergency Clear does to every guard. The trip count survives, because
        telemetry counters are cumulative and are never reset on read.

        Snapping a held feed straight back to unity is a full-scale step, and it
        is the one transition here that is not click-free. That is deliberate:
        Clear resets every chain and wipes the shared feed history in the same
        batch, so there is nothing left on the bus for the step to click
        against.

        No setTimeConstant, and so no std::exp: whichever ramp starts next sets
        its own coefficient. That is what makes a 32-channel Clear libm-free on
        the driver thread. */
    void reset() noexcept
    {
        state = State::Armed;
        timerSeconds = 0.0;
        armedSeconds = 0.0;
        vetoSeconds = 0.0;
        consecutiveTrips = 0;

        gain.setSnapEpsilon (kSnapEpsilon);
        gain.snap (1.0f);
    }

    //==========================================================================
    /** Advance the timers by one batch and decide what the gain is heading for.

        @param feedPeak   absolute peak of this channel's effect-to-effect feed
                          row for this batch, taken BEFORE applyGain and with
                          peakOf() rather than a plain running max
        @param returnPeak absolute peak of this channel's return. The engine has
                          not run the chain yet when this is called, so it is
                          last batch's value; one batch of lag on a veto that
                          can only ever delay a release changes nothing. The
                          feed it is compared against is older still - it was
                          built from returns that left the ring a batch earlier
                          again, which is the plan's n+2 for an effect-to-effect
                          route - so the two describe instants a batch apart.
                          Harmless for a veto; they are not simultaneous.
        @param numSamples the batch length

        A non-finite peak counts as loud for the trip test and as not calm for
        both release tests, so a channel producing NaN trips and stays tripped:
        every comparison against a NaN is false, and each test is written so
        that the false branch is the failure-safe one. */
    void observeBlock (float feedPeak, float returnPeak, int numSamples) noexcept
    {
        if (! enabled || ! prepared || numSamples <= 0)
            return;

        const double dt = static_cast<double> (numSamples) * invSampleRate;

        // The ramp back has landed on exactly 1: the guard is out of the way.
        if (state == State::Releasing && isSettledAt (1.0f))
            state = State::Armed;

        if (state == State::Tripped)
        {
            // The feed decides, the return can only veto, and only for so long.
            // isSettledAt(0) holds the clock at zero until the feed has
            // actually been taken away, so a release time shorter than the ramp
            // down cannot give back a bus that is still on its way to silence.
            const bool feedCalm = feedPeak <= releaseLinear;      // NaN: not calm
            const bool ready = feedCalm && isSettledAt (0.0f);
            const bool returnCalm = returnPeak < ceilingLinear;   // NaN: not calm

            if (! ready)
                vetoSeconds = 0.0;
            else if (! returnCalm)
                vetoSeconds += dt;

            const bool calm = ready && (returnCalm || vetoSeconds >= vetoBudgetSeconds());

            timerSeconds = calm ? timerSeconds + dt : 0.0;

            if (calm && timerSeconds >= activeReleaseSeconds())
                beginRelease();

            return;
        }

        // Armed and Releasing both watch for a trip: a loop that comes back
        // while the feed is being restored is caught on the way up.
        if (! (feedPeak <= ceilingLinear))
        {
            armedSeconds = 0.0;
            timerSeconds += dt;

            if (timerSeconds >= tripSeconds)
                trip();

            return;
        }

        // Consecutive means consecutive: one block under the ceiling and the
        // build-up was not a build-up.
        timerSeconds = 0.0;

        if (state == State::Armed && isSettledAt (1.0f))
        {
            armedSeconds += dt;

            if (armedSeconds >= kBackoffForgetFactor * releaseSeconds)
            {
                consecutiveTrips = 0;
                armedSeconds = 0.0;
            }
        }
        else
        {
            armedSeconds = 0.0;
        }
    }

    /** Apply the guard's gain to the effect-to-effect feed row in place, one
        multiply per sample while the ramp is moving.

        Two fast paths, and the first is why an engine full of guards that have
        never seen a loud block costs nothing: an armed guard returns before it
        touches the buffer at all, so the row is not merely multiplied by 1.0f,
        it is not written. That distinction is real rather than pedantic,
        because every worker item runs inside juce::ScopedNoDenormals: with
        FTZ/DAZ armed, a denormal multiplied by 1.0f comes back as zero, so
        "multiply by one" is NOT the identity on the audio path. The second path
        memsets a fully held row rather than multiplying it by zero, which is
        the same silence for less work.

        A null row or a non-positive length is a no-op: a guard with nothing to
        do must not care what it was handed.

        The `enabled` test is belt and braces, and deliberately so: a switched
        off guard is always settled at unity today (prepare and setEnabled both
        re-arm), so the fast path below would return anyway and no test can tell
        the two apart. It stays because the failure it prevents - attenuating a
        feed while the operator has the guard switched off - is the one this
        class must never commit, and a future setEnabled that stopped re-arming
        would otherwise introduce it silently. */
    void applyGain (float* fxToFxBus, int numSamples) noexcept
    {
        if (! enabled || ! prepared || fxToFxBus == nullptr || numSamples <= 0)
            return;

        if (isSettledAt (1.0f))
            return;

        if (isSettledAt (0.0f))
        {
            std::memset (fxToFxBus, 0, static_cast<std::size_t> (numSamples) * sizeof (float));
            return;
        }

        for (int i = 0; i < numSamples; ++i)
            fxToFxBus[i] *= gain.next();
    }

    //==========================================================================
    /** Absolute peak of a block, with a non-finite sample surviving as the peak
        instead of being dropped. A plain running max loses a NaN, because
        `v > peak` is false for one, and the state machine would then never see
        the block that mattered most. */
    static float peakOf (const float* block, int numSamples) noexcept
    {
        float peak = 0.0f;

        if (block == nullptr)
            return peak;

        for (int i = 0; i < numSamples; ++i)
        {
            const float v = std::fabs (block[i]);

            if (! (v <= peak))
            {
                if (std::isnan (v))
                    return v;               // nothing later can beat it

                peak = v;
            }
        }

        return peak;
    }

    //==========================================================================
    State getState() const noexcept { return state; }

    /** True while the guard is holding this feed down or bringing it back, so
        the GUI badge stays lit for the whole event rather than going out the
        moment the release starts. */
    bool isTripped() const noexcept { return state != State::Armed; }

    /** The gain currently applied to the effect-to-effect feed: exactly 1 when
        armed, exactly 0 while held, in between while ramping. */
    float getGain() const noexcept { return gain.getCurrent(); }

    /** Cumulative since prepare(). PLAIN, not atomic, and read on the thread
        that drives the guard: the engine mirrors it into its telemetry block
        after the parallel join, and the GUI reads that. */
    std::uint32_t getTripCount() const noexcept { return tripCount; }

private:
    //==========================================================================
    static float tauForSettleTime (double settleSeconds) noexcept
    {
        // tau <= 0 makes the smoother jump, which is the right reading of a
        // ramp time of zero.
        return settleSeconds > 0.0 ? static_cast<float> (settleSeconds / kTausToSettle) : 0.0f;
    }

    bool isSettledAt (float value) const noexcept
    {
        return gain.getTarget() == value && gain.isSettled();
    }

    double activeReleaseSeconds() const noexcept
    {
        const int steps = consecutiveTrips > 0 ? consecutiveTrips - 1 : 0;
        const int capped = steps < kMaxBackoffSteps - 1 ? steps : kMaxBackoffSteps - 1;

        return releaseSeconds * static_cast<double> (1 << capped);
    }

    /** Against the BASE release time, not the backed-off one: how long a chain
        takes to stop ringing is a property of the chain, not of how many times
        this channel has already tripped. */
    double vetoBudgetSeconds() const noexcept
    {
        return kVetoReleaseTimes * releaseSeconds;
    }

    void trip() noexcept
    {
        state = State::Tripped;
        timerSeconds = 0.0;
        armedSeconds = 0.0;
        vetoSeconds = 0.0;
        ++tripCount;

        if (consecutiveTrips < kMaxBackoffSteps)
            ++consecutiveTrips;

        // One std::exp per state change, not per block: the two ramps have
        // different lengths and OnePoleSmoother holds one coefficient.
        gain.setTimeConstant (sampleRate, rampDownTau);
        gain.setTarget (0.0f);
    }

    void beginRelease() noexcept
    {
        state = State::Releasing;
        timerSeconds = 0.0;
        vetoSeconds = 0.0;

        gain.setTimeConstant (sampleRate, rampUpTau);
        gain.setTarget (1.0f);
    }

    //==========================================================================
    double sampleRate = 48000.0;
    double invSampleRate = 1.0 / 48000.0;

    float ceilingLinear = 1.0f;
    float releaseLinear = 1.0f;
    double tripSeconds = kDefaultTripSeconds;
    double releaseSeconds = kDefaultReleaseSeconds;
    float rampDownTau = 0.0f;
    float rampUpTau = 0.0f;
    bool prepared = false;
    bool enabled = true;

    State state = State::Armed;
    spatcore::dsp::OnePoleSmoother gain;

    /** Loud time while armed or releasing, calm time while tripped. The two
        never run at once, so one accumulator says what the state machine is
        waiting for. */
    double timerSeconds = 0.0;

    /** Armed, restored and calm: counts towards forgetting the backoff. */
    double armedSeconds = 0.0;

    /** Held at silence with a calm feed and a return still over the ceiling:
        counts towards the veto's budget, and goes back to zero the moment the
        feed is loud again or the ramp is not settled at silence. */
    double vetoSeconds = 0.0;

    int consecutiveTrips = 0;
    std::uint32_t tripCount = 0;
};

} // namespace spatcore::effects
