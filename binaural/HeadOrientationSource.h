#pragma once

#include "BinauralTypes.h"
#include "../rt/RtSnapshot.h"
#include <juce_core/juce_core.h>

namespace spatcore::binaural
{

/**
    A source of head attitude for the binaural renderer — the fast path that
    bypasses the 50 Hz damped parameter pipeline.

    Contract:
      - getOrientation() is RT-safe (POD copy, lock bounded) and is called by
        the render worker once per block. Implementations publish through
        spatcore::rt::RtSnapshot<HeadOrientation> from ONE non-RT thread
        (a device poll thread, the message thread, ...).
      - Angles are offsets from the facing-origin baseline (BinauralTypes.h);
        a tracker's "zero" is calibrated by looking at the stage.
      - valid == false in the returned HeadOrientation means the source has
        nothing trustworthy (device unplugged, no data yet) — the consumer
        falls back to the manual orientation parameters, slewed.
      - Angles are always FINITE. A source that computes a NaN/infinite
        attitude reports valid == false instead; publishOrientation() below
        enforces this for every source that uses it.
      - getSourceId() is a stable identifier for persistence ("manual" is
        reserved; hardware sources use e.g. "usb:<serial>"). Display names
        are free-form.

    "Manual orientation" is represented by the ABSENCE of an active source
    (the manual yaw/pitch/roll already travel in the 50 Hz RT snapshot), so
    implementations of this interface are always real external sources.
*/
class HeadOrientationSource
{
public:
    virtual ~HeadOrientationSource() = default;

    virtual juce::String getSourceId() const = 0;
    virtual juce::String getDisplayName() const = 0;
    virtual bool isConnected() const = 0;

    /** RT-safe. Called by the render worker once per block. */
    virtual HeadOrientation getOrientation() const noexcept = 0;

    /** Calibrate "this attitude is zero" — the user faces the stage and asks
        for the current pose to become the facing-origin reference. Sources
        store the inverse of the current attitude and compose it with every
        subsequent report (rotation composition, NOT per-angle subtraction:
        rotations don't commute). Which SIDE it composes on is the source's
        own business — a camera cancels a world-side offset, a head-mounted
        sensor cancels a body-side one. Called from the message thread; sources
        that need no calibration ignore it. */
    virtual void setZero() {}

    /** Message thread. Called when this source becomes, or stops being, the
        selected one.

        Sources owning an EXCLUSIVE device open it in activate() and release it
        in deactivate(), so an unselected tracker never holds a webcam the user
        wants elsewhere. Returning false refuses activation (the source logs
        the reason) and the caller falls back to manual orientation.

        Sources fed by a SHARED device that must already be running BEFORE any
        selection — a USB dongle whose stream is what populates the tracker
        list in the first place — leave both as no-ops; that device's lifetime
        is owned elsewhere and is not tied to selection. Both must be
        idempotent. */
    virtual bool activate() { return true; }
    virtual void deactivate() {}
};

/** Publish/acquire base most sources will want: poll thread publishes,
    render worker acquires. */
class SnapshotHeadOrientationSource : public HeadOrientationSource
{
public:
    HeadOrientation getOrientation() const noexcept override
    {
        return snapshot.acquire();
    }

protected:
    /** Call from the ONE producing thread only.

        Non-finite attitudes are refused here — the single choke point every
        source publishes through, so no driver can put NaN on the render
        thread by forgetting to check (see isFiniteAttitude). The refusal is
        published as "nothing trustworthy" rather than dropped, so the
        consumer's existing invalid-source fallback handles it.

        This is a backstop, not a licence to skip validating upstream: by the
        time a bad value reaches here, a source's own smoothing/calibration
        state may already be poisoned, which only that source can repair. */
    void publishOrientation (const HeadOrientation& o) noexcept
    {
        if (o.valid && ! isFiniteAttitude (o))
        {
            snapshot.publish ({});
            return;
        }
        snapshot.publish (o);
    }

private:
    spatcore::rt::RtSnapshot<HeadOrientation> snapshot;
};

} // namespace spatcore::binaural
