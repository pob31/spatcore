#pragma once

#include <cmath>
#include <cstdint>

namespace spatcore::binaural
{

/**
    Shared types for the binaural rendering module.

    Coordinate conventions (load-bearing — everything in this module assumes them):

    World (origin) frame — the app's storage frame:
      +x = stage right, +y = upstage (away from audience), +z = up.
      An orientation angle θ faces direction (sin θ, −cos θ) in the x/y plane
      (θ = 0 faces −y, i.e. toward the audience).

    Head frame:
      +x = out of the RIGHT ear, +y = facing direction, +z = up.
      Azimuth az = atan2(x, y): 0 = straight ahead, positive = source to the
      listener's right. Elevation el = atan2(z, hypot(x, y)): positive = above.
      This is the vertical-polar convention used by SOFA HRTF sets.

    The total head rotation is composed as
        R_head = R_baseline(listenerAngle) · R_offset(yaw, pitch, roll)
    where R_baseline is the "facing the origin" orientation implied by the
    listener placement, and yaw/pitch/roll are offsets from that baseline
    (manual controls or a head tracker calibrated by looking at the stage).
    All-zero offsets therefore reproduce the legacy facing-origin behavior.
*/

/** Which algorithm renders the binaural pair. Values are persisted — keep stable. */
enum class RenderMode : int
{
    OrtfLegacy = 0,   // two virtual capsules, delay-and-sum (pre-HRTF behavior)
    Structural = 1,   // parametric spherical-head model (ITD + shadow + elevation)
    Sofa       = 2,   // measured HRIRs from a SOFA file
};

/**
    Head attitude sample from an orientation source (manual UI or a tracker).
    Trivially copyable — published through spatcore::rt::RtSnapshot.

    Angles are radians, offsets from the facing-origin baseline (see above).
    `valid == false` means the source has nothing trustworthy (tracker
    unplugged / not yet delivering) and the consumer should fall back to the
    manual values.

    The position fields are reserved for future 6DOF trackers; no current
    source fills them and no consumer reads them.
*/
struct HeadOrientation
{
    float yawRad   = 0.0f;
    float pitchRad = 0.0f;
    float rollRad  = 0.0f;
    bool  valid    = false;

    float posX = 0.0f, posY = 0.0f, posZ = 0.0f;   // reserved (6DOF)
    bool  hasPosition = false;                     // reserved (6DOF)
};

/**
    Is this attitude usable as a rotation?

    Head attitude arrives from untrusted producers — the tracker plugin ABI
    hands over RAW angles, and the manual parameters come from a project file
    — so it can be NaN or infinite (a degenerate face box divides by a
    vanishing eye distance). Non-finite angles build a non-orthonormal
    rotation matrix, and from there NaN spreads into delay lines, IIR state
    and convolution history, where it stays: the damage outlives the bad
    frame that caused it.

    Note the positive-logic form. `a < limit` style tests are useless against
    NaN (every comparison with NaN is false), which is exactly how this class
    of bug slips through.
*/
inline bool isFiniteAttitude (const HeadOrientation& o) noexcept
{
    return std::isfinite (o.yawRad) && std::isfinite (o.pitchRad) && std::isfinite (o.rollRad);
}

/**
    Listener pose in the world frame, built per block by the render worker:
    damped position from the slow (50 Hz) parameter path, rotation from the
    fast orientation path.
*/
struct ListenerPose
{
    float x = 0.0f, y = 0.0f, z = 0.0f;

    /** Head→world rotation, row-major 3x3: world = R · head. */
    float R[9] = { 1.0f, 0.0f, 0.0f,
                   0.0f, 1.0f, 0.0f,
                   0.0f, 0.0f, 1.0f };
};

/** Is this pose usable? Same reasoning as isFiniteAttitude, one hop later:
    by here the angles have become a rotation matrix, and a single non-finite
    entry makes every direction derived from it non-finite. */
inline bool isFinitePose (const ListenerPose& p) noexcept
{
    if (! (std::isfinite (p.x) && std::isfinite (p.y) && std::isfinite (p.z)))
        return false;
    for (int i = 0; i < 9; ++i)
        if (! std::isfinite (p.R[i]))
            return false;
    return true;
}

/** A source direction expressed in the head frame (vertical-polar). */
struct SourceDirection
{
    float azRad    = 0.0f;   // 0 = front, positive = right
    float elRad    = 0.0f;   // positive = above
    float distance = 0.0f;   // meters from head center
};

} // namespace spatcore::binaural
