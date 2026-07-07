# SpatcoreConsumer.cmake — helpers for CMake-native spatcore consumers.
#
# Encapsulates the fiddly, drift-prone wiring every consumer (XOA, Tight-WFS,
# the minimal example, and post-split WFS-DIY if it ever goes CMake) must do to
# satisfy spatcore's consumer-provides-dependencies contract. Centralizing it
# here means a fix lands once and every consumer inherits it on a spatcore pin
# bump — instead of being copy-pasted into each app's CMakeLists and drifting.
#
# Usage (in the consumer's TOP-LEVEL CMakeLists, after project()):
#
#     list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/spatcore/cmake")
#     include(SpatcoreConsumer)
#     spatcore_enable_consumer_languages()              # BEFORE add_subdirectory(JUCE)
#
#     add_subdirectory(ThirdParty/JUCE juce EXCLUDE_FROM_ALL)   # consumer owns JUCE
#     spatcore_add_juce_simpleweb(ThirdParty/juce_simpleweb)
#     juce_add_module(ThirdParty/roli_blocks_basics)            # plain — no fixups
#     spatcore_add_hidapi(ThirdParty/hidapi)                    # sets HIDAPI_INCLUDE_DIR
#     add_subdirectory(spatcore spatcore-build)
#
# Then link spatcore-audio / spatcore-control / spatcore-controllers as usual,
# and call spatcore_apply_compile_flags(<your target>) (SpatcoreCompileFlags is
# included transitively by spatcore's own CMakeLists; add its dir to
# CMAKE_MODULE_PATH too if you want to call it before add_subdirectory(spatcore)).

# ---------------------------------------------------------------------------
# spatcore_enable_consumer_languages()
#
# Enable the extra languages spatcore needs on the current platform, in the
# CALLER's directory scope. This MUST be a macro, not a function: enable_language
# populates a language's compile rules only into the directory scope that runs
# it (and below), and on Apple the consumer's OWN targets compile .mm sources
# (spatcore's Metal backends AND JUCE's module .mm), so the rules must exist in
# the consumer's top-level scope. Calling this from a function would enable the
# language in the function's throwaway scope and the consumer would still hit
# "Missing variable is: CMAKE_OBJCXX_COMPILE_OBJECT" at generate time. A macro is
# textually inlined into the caller, so enable_language runs exactly where it must.
# (The Xcode generator infers languages per file and needs none of this, but the
# macro is harmless there.)
macro(spatcore_enable_consumer_languages)
    if(APPLE)
        enable_language(OBJCXX)
    endif()
endmacro()

# ---------------------------------------------------------------------------
# spatcore_add_juce_simpleweb(<path-to-juce_simpleweb-module>)
#
# juce_add_module(juce_simpleweb) plus the two consumer-side fixups spatcore-
# control's MCP transport needs:
#   1. juce_simpleweb's .cpps were authored for Projucer and #include a global
#      "JuceHeader.h"; put spatcore's minimal stub on the module's interface
#      include path (shipped next to this file so consumers don't maintain one).
#   2. the module header declares libssl,libcrypto[,z] as link libs
#      unconditionally, but spatcore-control builds it with
#      SIMPLEWEB_SECURE_SUPPORTED=0 (plain HTTP/WS, no TLS) — strip them so no
#      OpenSSL SDK is required.
function(spatcore_add_juce_simpleweb simpleweb_path)
    if(NOT EXISTS "${simpleweb_path}/juce_simpleweb.h")
        message(FATAL_ERROR
            "spatcore_add_juce_simpleweb: no juce_simpleweb.h under "
            "'${simpleweb_path}'. Pass the path to the vendored juce_simpleweb "
            "module directory.")
    endif()

    juce_add_module("${simpleweb_path}")

    # Compile the module TLS-off everywhere it is compiled. JUCE injects a
    # module's .cpp into EVERY target that links it (spatcore-control AND, via
    # PUBLIC propagation, the consumer's own app/exe), so relying on spatcore-
    # control's PUBLIC SIMPLEWEB_SECURE_SUPPORTED=0 alone leaves the app target's
    # copy compiling asio's OpenSSL paths — unresolved SSL symbols at link
    # (upstream juce_simpleweb defaults secure ON). An INTERFACE define on the
    # module makes "no TLS" travel with it to every compiling target.
    target_compile_definitions(juce_simpleweb INTERFACE SIMPLEWEB_SECURE_SUPPORTED=0)

    target_include_directories(juce_simpleweb INTERFACE
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/JuceHeaderStub")

    get_target_property(_sw_libs juce_simpleweb INTERFACE_LINK_LIBRARIES)
    if(_sw_libs)
        list(REMOVE_ITEM _sw_libs libssl libcrypto z)
        set_target_properties(juce_simpleweb PROPERTIES
            INTERFACE_LINK_LIBRARIES "${_sw_libs}")
    endif()
endfunction()

# ---------------------------------------------------------------------------
# spatcore_add_hidapi(<path-to-hidapi-checkout>)
#
# Build hidapi as a static lib for spatcore-controllers' HID drivers and export
# SPATCORE_HIDAPI_INCLUDE_DIR to the caller. Accepts either a full libusb/hidapi
# checkout or a trimmed vendored copy — anything with hidapi/hidapi.h at its root
# and a CMakeLists (spatcore-controllers stages the headers privately, so a
# checkout's stray root files never poison the include path).
#
# Applies the settings consumers were all repeating: static only, no hidtest,
# hidraw backend on Linux, and the CMake >= 4 policy floor hidapi's old
# cmake_minimum_required(3.1.3...) trips. All of these are scoped to this
# function, so they don't leak into the consumer's own configuration.
function(spatcore_add_hidapi hidapi_path)
    if(NOT EXISTS "${hidapi_path}/hidapi/hidapi.h")
        message(FATAL_ERROR
            "spatcore_add_hidapi: no hidapi/hidapi.h under '${hidapi_path}'.")
    endif()

    set(BUILD_SHARED_LIBS OFF)
    set(HIDAPI_BUILD_HIDTEST OFF)
    set(HIDAPI_WITH_LIBUSB OFF)          # Linux: hidraw backend only
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5) # hidapi's minimum predates CMake 4's floor

    add_subdirectory("${hidapi_path}" "${CMAKE_BINARY_DIR}/hidapi" EXCLUDE_FROM_ALL)

    set(SPATCORE_HIDAPI_INCLUDE_DIR "${hidapi_path}" PARENT_SCOPE)
endfunction()
