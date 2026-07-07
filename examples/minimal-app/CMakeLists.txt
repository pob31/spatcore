# spatcore — minimal consumer example.
#
# A CI-tested reference wiring that exercises spatcore's CMake-native consumer
# contract standalone (no parent app repo). It builds spatcore-audio, whose
# Metal .mm backends compile on macOS — the exact path (the top-level OBJCXX
# requirement) that regressed the first real consumer (XOA) before this guard
# existed. It fetches only JUCE, so spatcore's CI stays self-contained and
# robust; the control/controllers wiring (juce_simpleweb, hidapi) is exercised
# by the downstream app CIs (XOA, Tight-WFS), which build the full stack.
#
# Build (from the spatcore repo root):
#   cmake -S examples/minimal-app -B examples/minimal-app/build
#   cmake --build examples/minimal-app/build --config Debug
# Reuse a local JUCE checkout instead of fetching:
#   -DFETCHCONTENT_SOURCE_DIR_JUCE=<path-to-JUCE>

cmake_minimum_required(VERSION 3.22)

project(spatcore-minimal-app VERSION 0.1.1 LANGUAGES C CXX)

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
    GIT_TAG        2cdfca8feb300fb424002ba2c2751569e5bacb64   # JUCE 8.0.14
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(JUCE)

# Audio only: spatcore-audio depends on nothing but JUCE. Turning control and
# controllers OFF keeps this example's dependency set to JUCE alone.
set(SPATCORE_CONTROL     OFF)
set(SPATCORE_CONTROLLERS OFF)
add_subdirectory("${SPATCORE_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/spatcore")

juce_add_console_app(spatcore-minimal-app)
target_sources(spatcore-minimal-app PRIVATE main.cpp)
target_link_libraries(spatcore-minimal-app PRIVATE spatcore-audio)
spatcore_apply_compile_flags(spatcore-minimal-app)
