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
    /** Call from the ONE producing thread only. */
    void publishOrientation (const HeadOrientation& o) noexcept
    {
        snapshot.publish (o);
    }

private:
    spatcore::rt::RtSnapshot<HeadOrientation> snapshot;
};

} // namespace spatcore::binaural
