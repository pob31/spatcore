/*
    HeadTrackPluginApi.h — C ABI for out-of-process-grade head-tracker plugins.

    A head-tracker plugin is a shared library (wfs_headtrack.dll /
    libwfs_headtrack.so / libwfs_headtrack.dylib) that owns whatever heavy
    machinery a tracker needs — camera capture, neural inference, vendor SDKs —
    and hands the app a stream of head attitudes. The app dlopens it, resolves
    the symbols below by name, and never links any of those dependencies.

    This mirrors the GPU vendor-plugin split (spatcore/gpu/plugin) but fixes
    its two known weaknesses, because a tracker plugin is built out-of-tree
    with a different toolchain than the app:

      1. VERSIONED. wfs_headtrack_abi_version() must equal WFS_HEADTRACK_ABI or
         the app refuses the plugin. A stale library fails loudly instead of
         silently running old behavior.
      2. POD ONLY. Nothing with a vtable, no STL types, no allocation crossing
         the boundary; the app never frees plugin memory (destroy through
         wfs_headtrack_destroy). Structs carry structSize so fields can be
         appended within an ABI version.

    Angle conventions (identical to spatcore/binaural/BinauralTypes.h):
      +yaw   = turn right      +pitch = look up      +roll = right ear down
    Radians. The plugin reports RAW attitude in these conventions — no
    smoothing, no calibration: the app owns filtering (spatcore/dsp/
    OneEuroFilter.h) and zero-calibration so they can be tuned without
    rebuilding the plugin.
*/

#ifndef WFS_HEADTRACK_PLUGIN_API_H
#define WFS_HEADTRACK_PLUGIN_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Bump when the meaning of existing fields/functions changes. Appending a
    field guarded by structSize does NOT need a bump. */
#define WFS_HEADTRACK_ABI 1

#if defined(_WIN32)
  #define WFS_HEADTRACK_API __declspec(dllexport)
#else
  #define WFS_HEADTRACK_API __attribute__((visibility("default")))
#endif

/** One head attitude sample. */
typedef struct WfsHeadtrackPose
{
    uint32_t structSize;      /* sizeof(WfsHeadtrackPose) as the plugin sees it */
    double   timestampMs;     /* plugin monotonic clock, for rate/latency diagnostics */
    float    yawRad;
    float    pitchRad;
    float    rollRad;
    float    confidence;      /* 0..1, tracker-defined quality */
    int32_t  faceDetected;    /* 0 = nothing tracked this frame (angles meaningless) */
} WfsHeadtrackPose;

/** Creation parameters. Zero/absent values mean "plugin default". */
typedef struct WfsHeadtrackConfig
{
    uint32_t structSize;
    int32_t  cameraIndex;     /* 0 = first camera */
    int32_t  width;           /* requested capture size, 0 = default (640) */
    int32_t  height;          /* 0 = default (480) */
    int32_t  fps;             /* requested rate, 0 = default (60) */
} WfsHeadtrackConfig;

/** Pose delivery. Called from the plugin's own worker thread — ONE thread for
    the lifetime of a start()/stop() pair, so the app can publish straight into
    a single-producer snapshot without locking. Must return promptly: it sits
    in the tracker's latency path. */
typedef void (*WfsHeadtrackPoseCb)(const WfsHeadtrackPose* pose, void* userData);

/** Returns WFS_HEADTRACK_ABI of the build. Resolve and check this FIRST. */
WFS_HEADTRACK_API int32_t wfs_headtrack_abi_version(void);

/** Short human-readable tracker name ("Webcam (FaceMesh)"), for the UI. */
WFS_HEADTRACK_API const char* wfs_headtrack_name(void);

/** Create an instance. Returns NULL on failure — call wfs_headtrack_last_error
    (NULL handle accepted) for the reason. Does not open the camera yet. */
WFS_HEADTRACK_API void* wfs_headtrack_create(const WfsHeadtrackConfig* config);

/** Open the device and begin delivering poses. Returns 1 on success, 0 on
    failure (see last_error). Idempotent. */
WFS_HEADTRACK_API int32_t wfs_headtrack_start(void* handle, WfsHeadtrackPoseCb cb, void* userData);

/** Stop delivery and release the device. Blocks until the worker thread has
    joined, so no callback can arrive after this returns. Idempotent. */
WFS_HEADTRACK_API void wfs_headtrack_stop(void* handle);

/** Destroy an instance (implies stop). The app must NEVER free the handle. */
WFS_HEADTRACK_API void wfs_headtrack_destroy(void* handle);

/** Last error for this instance, or the last global error when handle is NULL.
    Valid until the next call on the same handle. Never NULL. */
WFS_HEADTRACK_API const char* wfs_headtrack_last_error(void* handle);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* WFS_HEADTRACK_PLUGIN_API_H */
