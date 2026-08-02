// SpatcoreIoCompileCheck.cpp - compile-check + archive anchor for the
// spatcore-io static library (CMake build only).
//
// spatcore's io/ layer is header-only, so the target has no other source of its
// own. This TU
//   1. guarantees the archive has at least one public symbol on every platform
//      (MSVC's librarian otherwise warns LNK4221 for every empty object and
//      some ar/ranlib implementations reject empty archives);
//   2. compile-checks the public header surface under the exact optimization/FP
//      flags in cmake/SpatcoreCompileFlags.cmake, so a header regression fails
//      spatcore's own build instead of the first consumer's.
//
// Why a SEPARATE target/TU from SpatcoreAudioCompileCheck.cpp: this layer links
// juce_audio_devices, which drags in the platform driver backends (ASIO/WASAPI,
// CoreAudio, ALSA/JACK). spatcore-audio must stay linkable in headless
// renderers and DAW-plugin DSP, where the host owns the device and those
// backends have no business being present, so the device layer is opt-in
// (SPATCORE_IO) and separate. The dependency only ever runs io/ -> juce, never
// io/ -> audio/ or the reverse.
//
// NOT part of the WFS-DIY app build: the .jucer / vcxproj file lists do not
// reference this file - the app compiles spatcore sources directly through
// Projucer.

// io/ - device host + uncapped hardware-indexed callback + test signals. The
// host application owns the AudioDeviceManager and the AudioSource; nothing
// here names an app symbol.
#include "HardwareIndexMap.h"
#include "DeviceHost.h"
#include "DeviceIoCallback.h"
#include "TestSignalGenerator.h"

namespace spatcore
{
    // Referenced by nothing at runtime; exists so the archive is never empty.
    int spatcoreIoCompileCheckAnchor() noexcept { return 1; }
}
