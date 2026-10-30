#!/usr/bin/env bash
# Apply spatcore's JUCE patches for one platform to a consumer's JUCE checkout.
#
#   spatcore/tools/juce-patches/apply-juce-patches.sh <path/to/JUCE> <platform>
#
# <platform> names a folder under spatcore/patches/juce/ (today: macos).
# Every *.patch in it is applied with 'git apply' inside the JUCE checkout.
# Safe to run again: a patch that is already applied is skipped. A patch that
# applies neither way stops the script, which means it needs updating for the
# JUCE version the consumer pins.
#
# A submodule checkout of JUCE puts the patched files back, so consumers run
# this after every 'git submodule update' that touches JUCE, and in CI before
# the build.
#
# What each patch is for:
#   macos/juce-canUseTouch-macos.patch
#       detail::MouseInputSourceList::canUseTouch() returns false on macOS, so
#       ComponentPeer::handleMouseEvent (InputSourceType::touch, ...) is dropped
#       without a trace. controllers/touch/ injects one touch source per finger
#       from USB touch panels and needs it to be true. Retire it once JUCE
#       handles touch on macOS itself.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <path/to/JUCE> <platform>" >&2
    exit 2
fi

JUCE_DIR="$(cd "$1" && pwd)"
PATCH_DIR="$(cd "$(dirname "$0")/../../patches/juce" && pwd)/$2"

if [[ ! -f "$JUCE_DIR/modules/juce_core/juce_core.h" ]]; then
    echo "No JUCE checkout at $JUCE_DIR (initialise the submodule first)." >&2
    exit 1
fi

if [[ ! -d "$PATCH_DIR" ]]; then
    echo "No JUCE patches for platform '$2' ($PATCH_DIR)." >&2
    exit 1
fi

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)

cd "$JUCE_DIR"
for p in "${patches[@]}"; do
    name="$(basename "$p")"
    if git apply --check "$p" 2>/dev/null; then
        git apply "$p"
        echo "applied: $name"
    elif git apply --check --reverse "$p" 2>/dev/null; then
        echo "skipped (already applied): $name"
    else
        echo "FAILED: $name does not apply to this JUCE checkout - update the patch for this JUCE version" >&2
        exit 1
    fi
done
