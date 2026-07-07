# Standalone build of the spatcore unit tests from inside the WFS-DIY repo.
#
# This is the entry point CI/dev configures TODAY: it supplies the JUCE
# checkout and the vendored juce_simpleweb module from the host repo's
# ThirdParty/, then adds spatcore with SPATCORE_STANDALONE_TESTS=ON.
# After the Phase 6 repo split, the spatcore repository's own top-level
# CMakeLists (with its own JUCE pins) replaces this shim.
#
# Configure/build/run (Windows, VS-bundled cmake, from the repo root):
#   "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" ^
#         -S spatcore/tests/standalone -B spatcore/tests/standalone/build ^
#         -G "Visual Studio 18 2026"
#   cmake --build spatcore/tests/standalone/build --config Release
#   spatcore\tests\standalone\build\spatcore\tests\spatcore-tests_artefacts\Release\spatcore-tests.exe
# (exit code 0 = all tests passed)

cmake_minimum_required(VERSION 3.22)

project(spatcore-standalone-tests VERSION 0.1.1 LANGUAGES C CXX)

# spatcore's consumer helpers (OBJCXX enable, juce_simpleweb fixups, ...) — the
# same module apps use, so this reference wiring exercises the contract too.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
include(SpatcoreConsumer)
spatcore_enable_consumer_languages()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

# Consumer-provides contract: JUCE first, then the simpleweb module (with its
# stub + OpenSSL-strip fixups) via the helper.
add_subdirectory("${REPO_ROOT}/ThirdParty/JUCE" "${CMAKE_CURRENT_BINARY_DIR}/juce" EXCLUDE_FROM_ALL)
spatcore_add_juce_simpleweb("${REPO_ROOT}/ThirdParty/juce_simpleweb")

# Controllers side: the ROLI BLOCKS module for the Lightpad driver, and the
# hidapi HEADERS for the HID drivers. The tests compile the controller sources
# but never link a HID backend (hid_* impls live in the final app), so expose
# the headers rather than building the hidapi lib.
juce_add_module("${REPO_ROOT}/ThirdParty/roli_blocks_basics")
set(SPATCORE_HIDAPI_INCLUDE_DIR "${REPO_ROOT}/ThirdParty/hidapi")

set(SPATCORE_STANDALONE_TESTS ON)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../.." "${CMAKE_CURRENT_BINARY_DIR}/spatcore")
