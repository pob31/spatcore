#pragma once

#include "BinauralTypes.h"
#include <cmath>

namespace spatcore::binaural
{

/**
    Head-frame rotation math. Pure functions, no state, RT-safe.

    Baseline: a listener placed at world angle α — position (d·sin α, −d·cos α),
    facing the origin — has head basis (columns = head axes in world coords)

        R_baseline(α) = [ cos α  −sin α  0 ]
                        [ sin α   cos α  0 ]
                        [ 0       0      1 ]

    i.e. right ear along (cos α, sin α), facing (−sin α, cos α), up (0,0,1).
    This matches the legacy virtual-capsule geometry ("listener's right
    direction at angle α is (cos α, sin α)").

    Offsets (see BinauralTypes.h): positive yaw = turn right, positive
    pitch = look up, positive roll = right ear down. Intrinsic composition
    yaw → pitch → roll:

        R_offset = Rz(−yaw) · Rx(pitch) · Ry(roll)
        R_head   = R_baseline(α) · R_offset
*/
namespace headframe
{

/** Row-major 3x3 multiply: out = a · b. Aliasing-safe (works via temporaries). */
inline void multiply (const float a[9], const float b[9], float out[9]) noexcept
{
    float r[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r[i * 3 + j] = a[i * 3 + 0] * b[0 * 3 + j]
                         + a[i * 3 + 1] * b[1 * 3 + j]
                         + a[i * 3 + 2] * b[2 * 3 + j];
    for (int i = 0; i < 9; ++i)
        out[i] = r[i];
}

/** Offset rotation from yaw/pitch/roll (radians): Rz(−yaw)·Rx(pitch)·Ry(roll). */
inline void yawPitchRollToMatrix (float yaw, float pitch, float roll, float R[9]) noexcept
{
    const float cy = std::cos (yaw),   sy = std::sin (yaw);
    const float cp = std::cos (pitch), sp = std::sin (pitch);
    const float cr = std::cos (roll),  sr = std::sin (roll);

    // Rz(−yaw)
    const float rz[9] = {  cy,  sy, 0.0f,
                          -sy,  cy, 0.0f,
                          0.0f, 0.0f, 1.0f };
    // Rx(pitch)
    const float rx[9] = { 1.0f, 0.0f, 0.0f,
                          0.0f,  cp,  -sp,
                          0.0f,  sp,   cp };
    // Ry(roll)
    const float ry[9] = {  cr, 0.0f,  sr,
                          0.0f, 1.0f, 0.0f,
                          -sr, 0.0f,  cr };

    float tmp[9];
    multiply (rx, ry, tmp);
    multiply (rz, tmp, R);
}

/** Inverse of yawPitchRollToMatrix: recover the angles from the offset matrix.

    With R = Rz(−yaw)·Rx(pitch)·Ry(roll) the entries are
        R[7] = sin(pitch)
        R[1] =  sin(yaw)·cos(pitch)   R[4] = cos(yaw)·cos(pitch)
        R[6] = −cos(pitch)·sin(roll)  R[8] = cos(pitch)·cos(roll)
    At |pitch| → 90° yaw and roll degenerate (gimbal lock); we then attribute
    the whole rotation to yaw and report roll = 0, which is the stable choice
    for head tracking (looking straight up/down is already an edge case).
*/
inline void matrixToYawPitchRoll (const float R[9], float& yaw, float& pitch, float& roll) noexcept
{
    const float sinPitch = R[7] < -1.0f ? -1.0f : (R[7] > 1.0f ? 1.0f : R[7]);
    pitch = std::asin (sinPitch);

    if (std::abs (sinPitch) > 0.9999f)
    {
        yaw = std::atan2 (-R[3], R[0]);
        roll = 0.0f;
        return;
    }

    yaw  = std::atan2 (R[1], R[4]);
    roll = std::atan2 (-R[6], R[8]);
}

/** Transpose of an orthonormal rotation = its inverse. Aliasing-safe. */
inline void transpose (const float R[9], float out[9]) noexcept
{
    const float t[9] = { R[0], R[3], R[6],
                         R[1], R[4], R[7],
                         R[2], R[5], R[8] };
    for (int i = 0; i < 9; ++i)
        out[i] = t[i];
}

/** Head attitude from a hardware tracker's body->world quaternion.

    Source convention (headtracker PROTOCOL.md 1.3): Hamilton quaternion in
    w,x,y,z order, unit norm, rotating body -> world, in a body frame with
    X forward (nose), Y left, Z up.

    Target convention (BinauralTypes.h): x = out of the right ear, y = facing,
    z = up, with R_offset = Rz(-yaw)*Rx(pitch)*Ry(roll).

    The two frames are the same physical triad with relabelled axes, so they
    differ by one fixed change of basis. Writing the head basis in body
    coordinates -- right ear = (0,-1,0), facing = (1,0,0), up = (0,0,1) -- gives
    the columns of C, where v_body = C * v_head:

            [  0  1  0 ]
        C = [ -1  0  0 ]  = Rz(-90 deg)
            [  0  0  1 ]

    and therefore R_offset = C^T * R(q) * C. Carried out on a row-major R(q)
    with entries r0..r8 that whole product is a signed index shuffle:

                    [  r4  -r3  -r5 ]
        R_offset =  [ -r1   r0   r2 ]
                    [ -r7   r6   r8 ]

    Equivalently -- and this is what testTrackerQuatToHeadAngles pins -- the
    result is the source's intrinsic Z-Y-X Euler angles with yaw and pitch
    negated and roll kept: the source's +yaw is a turn to the LEFT and its
    +pitch is nose DOWN, while both conventions agree that +roll is right ear
    down. The matrix route is used here rather than those three sign flips
    because it is mechanically checkable against the derivation above and
    avoids a second trigonometric round trip.

    Non-finite or degenerate input yields (0,0,0) rather than NaN: a tracker
    frame can pass its CRC and still carry a NaN or infinity, and every
    consumer latches filter and calibration state from this output.
*/
inline void trackerQuatToYawPitchRoll (float qw, float qx, float qy, float qz,
                                       float& yaw, float& pitch, float& roll) noexcept
{
    const float n2 = qw * qw + qx * qx + qy * qy + qz * qz;

    // Both halves matter: positive logic because `n2 < eps` is false for NaN,
    // and isfinite because an infinite component makes n2 infinite, which
    // passes a lower-bound test and then gives inf * (1/sqrt(inf)) = NaN.
    if (! (std::isfinite (n2) && n2 > 1.0e-12f))
    {
        yaw = pitch = roll = 0.0f;
        return;
    }

    const float inv = 1.0f / std::sqrt (n2);
    const float w = qw * inv, x = qx * inv, y = qy * inv, z = qz * inv;

    // Hamilton body->world rotation matrix, row-major r0..r8.
    const float r0 = 1.0f - 2.0f * (y * y + z * z);
    const float r1 = 2.0f * (x * y - w * z);
    const float r2 = 2.0f * (x * z + w * y);
    const float r3 = 2.0f * (x * y + w * z);
    const float r4 = 1.0f - 2.0f * (x * x + z * z);
    const float r5 = 2.0f * (y * z - w * x);
    const float r6 = 2.0f * (x * z - w * y);
    const float r7 = 2.0f * (y * z + w * x);
    const float r8 = 1.0f - 2.0f * (x * x + y * y);

    const float offset[9] = {  r4, -r3, -r5,
                              -r1,  r0,  r2,
                              -r7,  r6,  r8 };

    matrixToYawPitchRoll (offset, yaw, pitch, roll);
}

/** Baseline rotation for a listener placed at world angle alphaRad, facing the origin. */
inline void baselineMatrix (float alphaRad, float R[9]) noexcept
{
    const float ca = std::cos (alphaRad), sa = std::sin (alphaRad);
    R[0] = ca;   R[1] = -sa;  R[2] = 0.0f;
    R[3] = sa;   R[4] = ca;   R[5] = 0.0f;
    R[6] = 0.0f; R[7] = 0.0f; R[8] = 1.0f;
}

/** R_head = R_baseline(alphaRad) · R_offset. */
inline void composeWithBaseline (float alphaRad, const float offset[9], float out[9]) noexcept
{
    float base[9];
    baselineMatrix (alphaRad, base);
    multiply (base, offset, out);
}

/** Source direction in the head frame: v_head = Rᵀ·(p_src − p_listener). */
inline SourceDirection directionInHeadFrame (const ListenerPose& pose,
                                             float srcX, float srcY, float srcZ) noexcept
{
    const float vx = srcX - pose.x;
    const float vy = srcY - pose.y;
    const float vz = srcZ - pose.z;

    // Rᵀ·v — R is orthonormal, so the transpose is the inverse.
    const float hx = pose.R[0] * vx + pose.R[3] * vy + pose.R[6] * vz;
    const float hy = pose.R[1] * vx + pose.R[4] * vy + pose.R[7] * vz;
    const float hz = pose.R[2] * vx + pose.R[5] * vy + pose.R[8] * vz;

    SourceDirection d;
    d.distance = std::sqrt (hx * hx + hy * hy + hz * hz);
    d.azRad    = std::atan2 (hx, hy);
    d.elRad    = std::atan2 (hz, std::sqrt (hx * hx + hy * hy));
    return d;
}

} // namespace headframe
} // namespace spatcore::binaural
