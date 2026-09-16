#pragma once

#include "../EffectModule.h"
#include "../../dsp/MultiChannelEQBank.h"

namespace spatcore::effects
{

/**
    Six-band parametric EQ - the output EQ, reused rather than rewritten.

    The user already has this model on every output and on the reverb sends, and
    it is the one they know; a second EQ with different shape numbering would be
    a trap rather than a feature. So this wraps MultiChannelEQBank<6> with one
    channel, which brings the shared coefficient maths, the clamping, and the
    property that a band set to shape 0 is a genuine bypass rather than a filter
    at unity.

    Shape ids are therefore the OUTPUT EQ's (0 off, 1 low cut, 2 low shelf,
    3 peak, 4 band pass, 5 high shelf, 6 high cut, 7 all pass), which differ
    from the reverb EQ's numbering. Anything mapping a GUI control to a band
    must use this one.

    The bank's channel is enabled for the module's whole life and the slot owns
    bypassing, so the enable-then-bands push order the bank documents is
    satisfied trivially: the enable never changes.

    Worth knowing when reading the tests: the DEFAULT bands are not an identity.
    Band 1 is a low cut at 80 Hz and band 6 a high cut at 12 kHz, so an
    un-bypassed EQ at defaults audibly does something. The identity is every
    shape set to 0.

    Zero latency, no variant parameters, no allocation after prepare().
*/
class EffectEQModule : public IEffectModule
{
public:
    static constexpr int kNumBands = 6;

    ModuleId type() const noexcept override { return ModuleId::EQ; }

    void prepare (const ChainConfig& config) override
    {
        bank.prepare (config.sampleRate, 1);
        bank.setChannelEnabled (0, true);
    }

    void reset() noexcept override { bank.reset(); }

    ParamApplyInfo applyParams (const EffectChannelParams& params, int instance) noexcept override
    {
        const EqParams& e = params.eq[instance & 1];

        // Enable first, then the bands - pushing the other way round would latch
        // the previous enable state into the coefficients for one tick.
        bank.setChannelEnabled (0, true);

        for (int b = 0; b < kNumBands; ++b)
            bank.pushBandParameters (0, b,
                                     static_cast<int> (e.shape[b]),
                                     e.freqHz[b],
                                     e.gainDb[b],
                                     e.q[b],
                                     e.slope[b]);

        return { e.bypass != 0, false };
    }

    void process (float* inout, int numSamples) noexcept override
    {
        bank.processChannel (0, inout, numSamples);
    }

    int getLatencySamples() const noexcept override { return 0; }

private:
    spatcore::dsp::MultiChannelEQBank<kNumBands> bank;
};

} // namespace spatcore::effects
