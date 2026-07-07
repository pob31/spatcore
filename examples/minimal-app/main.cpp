/*
  ==============================================================================

    spatcore — minimal consumer example.

    A compile-and-link smoke test that exercises the CMake-native consumer
    contract end to end. Its real job is to make spatcore's own CI *build*
    spatcore-audio (whose Metal .mm backends are compiled on macOS) and
    spatcore-control against a JUCE + juce_simpleweb it fetched itself — the
    exact path that broke the first real consumer (XOA) before this example
    existed to guard it.

    Released under the same terms as spatcore.

  ==============================================================================
*/

#include "spatcore/rt/RtSnapshot.h"

int main()
{
    // A message-thread -> RT hand-off round trip, so main.cpp actually
    // compiles a spatcore header (with its JUCE dependency) rather than
    // merely linking the libraries.
    spatcore::rt::RtSnapshot<int> snapshot;
    snapshot.publish (7);
    return snapshot.acquire() == 7 ? 0 : 1;
}
