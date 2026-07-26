# spatcore — shared real-time spatial-audio core extracted from WFS-DIY.
# Design: docs/architecture/core-boundary-proposal-audio.md (§3 dependency
# rules) + core-boundary-proposal-control.md.
#
# Targets
#   spatcore-audio    STATIC  rt/ dsp/ wfs/ reverb/ gpu/   (header-only from
#                             the consumer's point of view on Windows/Linux;
#                             the gpu backend .cpps self-guard to empty TUs —
#                             see SpatcoreAudioCompileCheck.cpp; Metal .mm
#                             carries real code on Apple)
#   spatcore-control  STATIC  control/{osc,state,mcp}
#
# Consumer-provides-JUCE contract: this file does NOT add_subdirectory(JUCE).
# The consumer must make the juce::* module targets available BEFORE adding
# spatcore (add_subdirectory of a JUCE checkout, or find_package(JUCE)), and
# must juce_add_module() the vendored juce_simpleweb module. See
# tests/standalone/CMakeLists.txt for the reference wiring used inside the
# WFS-DIY repo (ThirdParty/JUCE + ThirdParty/juce_simpleweb).
#
# The WFS-DIY app itself does NOT use these targets: it stays Projucer-built
# and compiles the spatcore sources directly (same compiler, same flags —
# zero bit-exact risk). These targets serve spatcore's own tests and the
# future CMake-native consumers (XOA, Tight-WFS).

cmake_minimum_required(VERSION 3.22)

project(spatcore VERSION 0.1.1 LANGUAGES C CXX)

# Metal backends (.mm) are Objective-C++ — and on Apple, JUCE's module sources
# are .mm too, attached to whichever target links the module, INCLUDING the
# consumer's own app target. CMake loads a language's compile rules only into
# the directory scope that enables it (and below), so enabling OBJCXX here
# covers spatcore's targets but cannot cover the consumer's. With the
# Makefile/Ninja generators the consumer must enable OBJCXX at its top level
# BEFORE add_subdirectory(spatcore), or generation fails with the cryptic
# "Missing variable is: CMAKE_OBJCXX_COMPILE_OBJECT". Fail at configure with
# instructions instead. (The Xcode generator infers languages per file and
# needs none of this.)
if(APPLE)
    if(NOT CMAKE_GENERATOR STREQUAL "Xcode" AND NOT DEFINED CMAKE_OBJCXX_COMPILE_OBJECT)
        message(FATAL_ERROR
            "spatcore: OBJCXX is not enabled in the consumer's top-level "
            "CMakeLists. On Apple with the \"${CMAKE_GENERATOR}\" generator, add\n"
            "    if(APPLE)\n"
            "        enable_language(OBJCXX)\n"
            "    endif()\n"
            "after project() and BEFORE add_subdirectory(spatcore), so the "
            "Objective-C++ compile rules exist in every directory that compiles "
            ".mm sources (JUCE module sources are .mm on Apple and attach to the "
            "consumer's own targets). See tests/standalone/CMakeLists.txt.")
    endif()
    enable_language(OBJCXX)
endif()

option(SPATCORE_STANDALONE_TESTS
    "Build spatcore's own unit-test console app (tests/)" OFF)

# Hardware controller device layer (SpaceMouse / Stream Deck+ / Lightpad /
# Linux evdev touch). OFF-able because headless consumers may not want the
# HID / BLOCKS dependencies at all.
option(SPATCORE_CONTROLLERS
    "Build spatcore-controllers (needs hidapi headers + roli_blocks_basics module)" ON)

# Control plane (OSC + ValueTree state + MCP server). OFF-able because a
# pure-audio consumer (or an audio-only build/example) may not want the
# juce_simpleweb dependency the MCP transport needs.
option(SPATCORE_CONTROL
    "Build spatcore-control (needs the juce_simpleweb module)" ON)

# ---------------------------------------------------------------------------
# Consumer-provides-JUCE contract
# ---------------------------------------------------------------------------
if(NOT TARGET juce::juce_core)
    message(FATAL_ERROR
        "spatcore: no juce::juce_core target. The consumer provides JUCE: "
        "add_subdirectory(<JUCE checkout>) (or find_package(JUCE)) BEFORE "
        "add_subdirectory(spatcore). Inside the WFS-DIY repo, build via "
        "spatcore/tests/standalone/CMakeLists.txt, which wires up "
        "ThirdParty/JUCE.")
endif()

if(SPATCORE_CONTROL AND NOT TARGET juce_simpleweb)
    message(FATAL_ERROR
        "spatcore: no juce_simpleweb target (needed by spatcore-control's "
        "MCPTransport). The consumer must register the vendored module with "
        "juce_add_module(<path>/juce_simpleweb) after adding JUCE, or configure "
        "with -DSPATCORE_CONTROL=OFF. Inside the WFS-DIY repo the module lives "
        "at ThirdParty/juce_simpleweb; see spatcore/tests/standalone/"
        "CMakeLists.txt or Plugin/CMakeLists.txt.")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/SpatcoreCompileFlags.cmake")

# Public include root: the directory CONTAINING spatcore/, so consumer code
# includes headers exactly like the app does: "spatcore/rt/RtSnapshot.h".
# Until the Phase 6 repo split this is the WFS-DIY repo root; the dependency
# lint (tools/validation/spatcore_dep_lint.py) enforces that no spatcore file
# reaches back into the app tree through it. After the split it is the
# submodule mount parent.
get_filename_component(SPATCORE_INCLUDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# ---------------------------------------------------------------------------
# spatcore-audio
# ---------------------------------------------------------------------------
# The Cuda*/Hip* .cpps are in every consumer's TU list exactly like in the
# app's vcxproj: they self-guard on WFS_GPU_NATIVE / WFS_GPU_PLUGINS /
# WFS_GPU_HIP and compile to empty TUs here (vendor code is compiled only
# into the per-vendor dlopen plugins by tools/gpu/build-gpu-plugins.*).
# gpu/plugin/GpuVendorPlugin.cpp is deliberately NOT listed — it is the
# plugin entry-point TU, compile="0" in the app project too.
add_library(spatcore-audio STATIC
    SpatcoreAudioCompileCheck.cpp
    gpu/CudaWfsBackend.cpp
    gpu/CudaObBackend.cpp
    gpu/CudaIrBackend.cpp
    gpu/CudaFdnBackend.cpp
    gpu/CudaSdnBackend.cpp
    gpu/HipWfsBackend.cpp
    gpu/HipObBackend.cpp
    gpu/HipIrBackend.cpp
    gpu/HipFdnBackend.cpp
    gpu/HipSdnBackend.cpp
)

if(APPLE)
    target_sources(spatcore-audio PRIVATE
        gpu/MetalWfsBackend.mm
        gpu/MetalObBackend.mm
        gpu/MetalIrBackend.mm
        gpu/MetalFdnBackend.mm
        gpu/MetalSdnBackend.mm
    )
endif()

target_include_directories(spatcore-audio PUBLIC "${SPATCORE_INCLUDE_ROOT}")

# GPU build-flag surface, mirroring the app's exporters (WFS-DIY.jucer):
#   Windows/Linux: WFS_GPU_NATIVE=1 WFS_GPU_PLUGINS=1 (backends dlopen'd from
#                  per-vendor plugin DLLs; no GPU runtime linked here)
#   macOS:         WFS_GPU_NATIVE=1 (Metal backend compiled in-process)
# PUBLIC on purpose: consumer TUs that include spatcore headers must see the
# same preprocessor state as the library (core-boundary doc §5.1 R6).
if(APPLE)
    target_compile_definitions(spatcore-audio PUBLIC WFS_GPU_NATIVE=1)
else()
    target_compile_definitions(spatcore-audio PUBLIC WFS_GPU_NATIVE=1 WFS_GPU_PLUGINS=1)
endif()

# JUCE module deps as measured from spatcore includes (2026-07):
#   juce_audio_basics  rt/dsp/wfs/reverb buffers (AudioBuffer, jmin/jmax, ...)
#   juce_dsp           reverb IR convolution FFT (ReverbIRAlgorithm[GPU].h)
#   juce_audio_formats reverb/ReverbEngine.h IR file loading
#                      (AudioFormatManager/Reader; the header relies on the
#                      includer providing the module header first - hygiene
#                      pass pending, see SpatcoreAudioCompileCheck.cpp)
#   juce_events        rt/ReverbDiagnostics.h
#   juce_core          rt/RtSnapshot.h (SpinLock) + transitive base
# No juce_osc on the audio side.
target_link_libraries(spatcore-audio PUBLIC
    juce::juce_audio_basics
    juce::juce_dsp
    juce::juce_audio_formats
    juce::juce_events
    juce::juce_core)

# The five gpu/Metal*Backend.mm above call into Metal.framework, and NO JUCE
# module pulls it in (juce_core supplies Foundation/Accelerate/QuartzCore, not
# Metal - the app supplies it by hand via the .jucer extraLinkerFlags). Without
# this, any consumer that odr-uses a Metal backend fails to link with undefined
# _MTLCreateSystemDefaultDevice / _OBJC_CLASS_$_MTLCompileOptions.
#
# It has been latent rather than broken only because spatcore-audio is STATIC:
# nothing referenced those objects, so the linker never extracted them from the
# archive. PUBLIC so consumers (offline-render, spatcore-tests) inherit it.
if(APPLE)
    target_link_libraries(spatcore-audio PUBLIC "-framework Metal")
endif()

# Consumers may be shared objects (plugin formats); PIC is free on Win/mac
# and required on Linux (see Plugin/CMakeLists.txt for the ld error text).
set_target_properties(spatcore-audio PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    # The self-guarded gpu TUs are empty on Windows by design; silence the
    # librarian's LNK4221 "no public symbols" warning for them.
    set_target_properties(spatcore-audio PROPERTIES STATIC_LIBRARY_OPTIONS "/IGNORE:4221")
endif()

spatcore_apply_compile_flags(spatcore-audio)

# ---------------------------------------------------------------------------
# spatcore-control
# ---------------------------------------------------------------------------
if(SPATCORE_CONTROL)
add_library(spatcore-control STATIC
    control/osc/OSCIngestQueue.cpp
    control/osc/OSCRateLimiter.cpp
    control/osc/OSCReceiverWithSenderIP.cpp
    control/osc/OSCTCPReceiver.cpp
    control/osc/TrackingIngestQueue.cpp
    control/state/TreeParameterStore.cpp
    control/state/XmlPersistence.cpp
    control/mcp/MCPChangeRecords.cpp
    control/mcp/MCPDispatcher.cpp
    control/mcp/MCPPromptRegistry.cpp
    control/mcp/MCPResourceRegistry.cpp
    control/mcp/MCPTierEnforcement.cpp
    control/mcp/MCPToolRegistry.cpp
    control/mcp/MCPTransport.cpp
)

target_include_directories(spatcore-control PUBLIC "${SPATCORE_INCLUDE_ROOT}")

# Disable juce_simpleweb's TLS/SSL paths, mirroring the app + DAW plugins:
# the MCP transport speaks plain HTTP/WebSocket, and leaving TLS on adds an
# OpenSSL runtime dependency (see Plugin/CMakeLists.txt for the full story).
# PUBLIC because the juce_simpleweb module sources are compiled into whichever
# target links them — every TU must agree on this flag.
target_compile_definitions(spatcore-control PUBLIC SIMPLEWEB_SECURE_SUPPORTED=0)

# JUCE module deps as measured from spatcore includes (2026-07):
#   juce_core             all of control/
#   juce_osc              osc/OSC{RateLimiter,ReceiverWithSenderIP,TCPReceiver}.h
#                         (+ OSCParser/OSCSerializer use juce::OSC* types)
#   juce_events           mcp/MCP{Dispatcher.cpp,TierEnforcement.h}, osc/OSCIngestQueue.h
#   juce_data_structures  state/{TreeParameterStore,XmlPersistence}.h, osc/TrackingIngestQueue.h
#   juce_simpleweb        mcp/MCPTransport.h (HTTP/WebSocket MCP transport)
#   juce_cryptography     included unconditionally by juce_simpleweb.h
#                         (WSCrypto websocket handshake), which declares no
#                         module dependencies of its own
target_link_libraries(spatcore-control PUBLIC
    juce::juce_core
    juce::juce_osc
    juce::juce_events
    juce::juce_data_structures
    juce::juce_cryptography
    juce_simpleweb)

set_target_properties(spatcore-control PROPERTIES POSITION_INDEPENDENT_CODE ON)

spatcore_apply_compile_flags(spatcore-control)
endif()  # SPATCORE_CONTROL

# ---------------------------------------------------------------------------
# spatcore-controllers
# ---------------------------------------------------------------------------
# Hardware controller device layer. Header-only apart from the Linux evdev
# touch .cpps, which self-guard on __linux__ and compile to empty TUs on
# Windows/macOS (same pattern as the gpu backend .cpps above).
#
# Consumer-provides contracts (mirroring juce_simpleweb above):
#   roli_blocks_basics          JUCE module for the ROLI Lightpad BLOCKS
#                               protocol (lightpad/LightpadDevice.h). The
#                               consumer registers it with juce_add_module().
#                               Inside the WFS-DIY repo it lives at
#                               ThirdParty/roli_blocks_basics.
#   SPATCORE_HIDAPI_INCLUDE_DIR Directory that contains hidapi/hidapi.h
#                               (headers only - the hid_* implementation is
#                               compiled by the final app, e.g. WFS-DIY's
#                               ThirdParty/hidapi/HidApiPlatform.c). Inside
#                               the WFS-DIY repo: ThirdParty/hidapi.
if(SPATCORE_CONTROLLERS)
    if(NOT TARGET roli_blocks_basics)
        message(FATAL_ERROR
            "spatcore: no roli_blocks_basics target (needed by "
            "spatcore-controllers' LightpadDevice). The consumer must register "
            "the module with juce_add_module(<path>/roli_blocks_basics) after "
            "adding JUCE, or configure with -DSPATCORE_CONTROLLERS=OFF. Inside "
            "the WFS-DIY repo the module lives at ThirdParty/roli_blocks_basics; "
            "see spatcore/tests/standalone/CMakeLists.txt.")
    endif()

    if(NOT SPATCORE_HIDAPI_INCLUDE_DIR)
        message(FATAL_ERROR
            "spatcore: SPATCORE_HIDAPI_INCLUDE_DIR is not set (spatcore-"
            "controllers' HID drivers include \"hidapi/hidapi.h\"). Point it at "
            "a directory containing hidapi/hidapi.h, or configure with "
            "-DSPATCORE_CONTROLLERS=OFF. Inside the WFS-DIY repo: "
            "ThirdParty/hidapi; see spatcore/tests/standalone/CMakeLists.txt.")
    endif()

    if(NOT EXISTS "${SPATCORE_HIDAPI_INCLUDE_DIR}/hidapi/hidapi.h")
        message(FATAL_ERROR
            "spatcore: SPATCORE_HIDAPI_INCLUDE_DIR "
            "(${SPATCORE_HIDAPI_INCLUDE_DIR}) does not contain hidapi/hidapi.h.")
    endif()

    add_library(spatcore-controllers STATIC
        controllers/SpatcoreControllersCompileCheck.cpp
        controllers/touch/linux/EvdevTouchDevice.cpp
        controllers/touch/linux/EvdevTouchManager.cpp
    )

    # Stage the hidapi headers into a private include tree rather than putting
    # the consumer's directory itself on the include path. An official
    # libusb/hidapi checkout has a VERSION file at its root; on case-insensitive
    # filesystems (macOS) libc++'s internal `#include <version>` resolves to it
    # and every TU in every target inheriting this PUBLIC include dir explodes.
    # Staging keeps the documented contract ("a directory containing
    # hidapi/hidapi.h" — a trimmed vendored copy and a full hidapi checkout both
    # work) while guaranteeing nothing else leaks onto the include path.
    set(_spatcore_hidapi_stage "${CMAKE_CURRENT_BINARY_DIR}/hidapi-include")
    file(COPY "${SPATCORE_HIDAPI_INCLUDE_DIR}/hidapi"
         DESTINATION "${_spatcore_hidapi_stage}"
         FILES_MATCHING PATTERN "*.h")

    target_include_directories(spatcore-controllers PUBLIC
        "${SPATCORE_INCLUDE_ROOT}"
        "${_spatcore_hidapi_stage}")

    # JUCE module deps as measured from spatcore/controllers includes (2026-07):
    #   juce_graphics         streamdeck button/LCD rendering + JPEG packing,
    #                         page/binding colours, lightpad LED colours
    #   juce_gui_basics       touch/linux/EvdevTouchManager (Desktop /
    #                         ComponentPeer touch injection; Linux-only code
    #                         but the dependency is declared unconditionally)
    #   juce_data_structures  touch/linux/EvdevTouchManager settings
    #                         (PropertiesFile)
    #   juce_events           device drivers' MessageManager::callAsync + Timer
    #   juce_core             threads, strings, everything
    #   roli_blocks_basics    lightpad/LightpadDevice.h (BLOCKS protocol)
    target_link_libraries(spatcore-controllers PUBLIC
        juce::juce_gui_basics
        juce::juce_graphics
        juce::juce_data_structures
        juce::juce_events
        juce::juce_core
        roli_blocks_basics)

    # Linux evdev touch needs libudev for enumeration/hotplug (the app's
    # Linux exporter declares pkg-config libudev the same way).
    if(UNIX AND NOT APPLE)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(UDEV REQUIRED IMPORTED_TARGET libudev)
        target_link_libraries(spatcore-controllers PUBLIC PkgConfig::UDEV)
    endif()

    set_target_properties(spatcore-controllers PROPERTIES POSITION_INDEPENDENT_CODE ON)

    if(MSVC)
        # The evdev .cpps are empty TUs on Windows by design; silence LNK4221.
        set_target_properties(spatcore-controllers PROPERTIES STATIC_LIBRARY_OPTIONS "/IGNORE:4221")
    endif()

    spatcore_apply_compile_flags(spatcore-controllers)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(SPATCORE_STANDALONE_TESTS)
    add_subdirectory(tests)
endif()
