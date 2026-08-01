// SpatcoreUiCompileCheck.cpp - compile-check + archive anchor for the
// spatcore-ui static library (CMake build only).
//
// spatcore's ui/ layer is header-only: the shared widgets are implicitly-inline
// class definitions, so the target has no other source of its own. This TU
//   1. guarantees the archive has at least one public symbol on every
//      platform (MSVC's librarian otherwise warns LNK4221 for every empty
//      object and some ar/ranlib implementations reject empty archives);
//   2. compile-checks the public widget header surface under the exact
//      optimization/FP flags in cmake/SpatcoreCompileFlags.cmake, so a header
//      regression fails spatcore's own build instead of the first consumer's.
//
// Why a SEPARATE target/TU from SpatcoreAudioCompileCheck.cpp: spatcore-audio
// deliberately does not link juce_gui_basics (the audio layer must stay usable
// in headless renderers and DAW-plugin DSP), so these headers cannot be
// compiled there. The dependency only ever runs ui/ -> dsp/, never the reverse:
// EQDisplayComponent draws its curves with the JUCE-free ../dsp/ biquad
// headers, which is why spatcore-ui needs no link to spatcore-audio.
//
// NOT part of the WFS-DIY app build: the .jucer / vcxproj file lists do not
// reference this file - the app compiles spatcore sources directly through
// Projucer.

// ui/ - shared EQ widgets. The host application injects its palette and its
// localised strings through the config structs (paletteProvider /
// colourProvider / disabledTextProvider), so nothing here names an app symbol.
#include "EQBandToggle.h"
#include "EQDisplayComponent.h"

namespace spatcore
{
    // Referenced by nothing at runtime; exists so the archive is never empty.
    int spatcoreUiCompileCheckAnchor() noexcept { return 1; }
}
