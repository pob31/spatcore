# spatcore — minimal consumer example.
#
# A CI-tested reference wiring that exercises spatcore's CMake-native consumer
# contract standalone (no parent app repo). It builds spatcore-audio, whose
# Metal .mm backends compile on macOS — the exact path (the top-level OBJCXX
# requirement) that regressed the first real consumer (XOA) before this guard
# existed — plus spatcore-ui, so the header-only GUI widgets get compiled on all
# three platforms too. It fetches only JUCE, so spatcore's CI stays
# self-contained and robust; the control/controllers wiring (juce_simpleweb,
# hidapi) is exercised by the downstream app CIs (XOA, Tight-WFS), which build
# the full stack.
#
# Build (from the spatcore repo root):
#   cmake -S examples/minimal-app -B examples/minimal-app/build
#   cmake --build examples/minimal-app/build --config Debug
# Reuse a local JUCE checkout instead of fetching:
#   -DFETCHCONTENT_SOURCE_DIR_JUCE=<path-to-JUCE>

cmake_minimum_required(VERSION 3.22)

project(spatcore-minimal-app VERSION 0.2.0 LANGUAGES C CXX)

get_filename_component(SPATCORE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

# spatcore's consumer helpers — the same module real apps use.
list(APPEND CMAKE_MODULE_PATH "${SPATCORE_ROOT}/cmake")
include(SpatcoreConsumer)

# Enable OBJCXX at the top level on Apple, before JUCE is added (the whole
# point of this example — regression-tests the consumer OBJCXX contract).
spatcore_enable_consumer_languages()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# JUCE — the consumer provides it; fetched here, pinned to the version the apps
# use. Override with -DFETCHCONTENT_SOURCE_DIR_JUCE to reuse a local checkout.
FetchContent_Declare(JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        e18f7f506c0b96f2c738a0bcd7fe6467a5005ad8   # JUCE 9.0.1
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(JUCE)

# Audio + UI: both depend on nothing but JUCE itself. Turning control and
# controllers OFF keeps this example's dependency set to JUCE alone (they need
# juce_simpleweb / roli_blocks_basics / hidapi from the consumer).
#
# spatcore-ui is ON so the 3-OS CI actually compiles the shared widget headers
# on Windows, macOS and Linux — the same reason spatcore-audio is here. It costs
# the Linux job nothing: the workflow already installs the X11/freetype/
# fontconfig dev packages, because JUCE's juceaide links juce_gui_basics.
#
# spatcore-io is ON for the same reason, and it is the one target whose JUCE
# dependency is platform driver code (ASIO/WASAPI, CoreAudio, ALSA), so
# compiling it on all three OSes here is the only guard against a backend-
# specific break. The Linux job already installs libasound2-dev.
set(SPATCORE_CONTROL     OFF)
set(SPATCORE_CONTROLLERS OFF)
set(SPATCORE_UI          ON)
set(SPATCORE_IO          ON)
add_subdirectory("${SPATCORE_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/spatcore")

juce_add_console_app(spatcore-minimal-app)
target_sources(spatcore-minimal-app PRIVATE main.cpp)
target_link_libraries(spatcore-minimal-app PRIVATE spatcore-audio spatcore-ui spatcore-io)
spatcore_apply_compile_flags(spatcore-minimal-app)
