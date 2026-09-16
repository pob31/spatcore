#pragma once

#include "../dsp/AcousticSendMatrix.h"

namespace spatcore::reverb {

/**
    Compatibility alias. The send matrix was promoted to
    spatcore::dsp::AcousticSendMatrix so the effects channels can reuse it
    (effects-channels plan, section 2.2): same type, same code, same tests.
    ReverbFeedThread holds one BY VALUE, so this must stay a `using` - never a
    wrapper or a derived class.
*/
using ReverbSendMatrix = spatcore::dsp::AcousticSendMatrix;

} // namespace spatcore::reverb

// Extraction-compat alias - app code migrates to qualified names later.
using spatcore::reverb::ReverbSendMatrix;
