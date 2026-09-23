#pragma once

#include "../../EffectModule.h"
#include "../../EffectParams.h"

namespace spatcore::effects
{

/** Size to two decimals - the step every model builds at. A fader that sends
    1.0000001 must not rebuild a network, and no line changes length by less
    than a hundredth of a size step anyway. */
inline float quantiseReverbSize (float v) noexcept
{
    return static_cast<float> (static_cast<int> (v * 100.0f + 0.5f)) * 0.01f;
}

/**
    The tail generator behind the reverb module.

    The module owns predelay, tone and mix - the parts every algorithm needs in
    the same shape - and a model owns the tail itself. The plate, the SDN-style
    model and convolution are then one more class behind this interface rather
    than a second reverb module carrying its own copy of the wet path, and a
    project that names a model the build does not have still makes a sound.

    setParams RETURNS the variant flag rather than being told it, because only
    the model knows which of its parameters are build-time. prepare() allocates
    and is never realtime; everything else, commitPendingVariant() included,
    runs on the audio thread - the module rebuilds an IDLE instance there when
    a change needs another topology, so a model must reach its new topology
    without allocating.
*/
class IEffectReverbModel
{
public:
    virtual ~IEffectReverbModel() = default;

    /** Allocates. IEffectModule::prepare carries no parameters, so `initial` is
        the default set; the module's first applyParams commits any build-time
        difference immediately rather than through a fade. */
    virtual void prepare (const ChainConfig& config, const ReverbParams& initial) = 0;

    virtual void reset() noexcept = 0;

    /** Takes the runtime values at once, and answers true while the model is
        still BUILT at a topology other than the one it has just been given -
        commitPendingVariant() rebuilds it. A STATE, not an edge: handed the
        value it is already built at, it answers false. */
    virtual bool setParams (const ReverbParams& params) noexcept = 0;

    /** Rebuilds at the pending topology, from silence. Audio thread: inside
        the capacity prepare() allocated. */
    virtual void commitPendingVariant() noexcept {}

    /** Mono, in place: the predelayed input goes in, the wet tail comes out. */
    virtual void process (float* inout, int numSamples) noexcept = 0;

    /** True when `params` asks for a topology this instance is not built at.
        Asked WITHOUT handing the parameters over: the module turns such a
        change into a spillover to an idle twin, and the instance that is about
        to ring out must keep playing with the values it had. */
    virtual bool isBuildDifferent (const ReverbParams& params) const noexcept = 0;

    /** The Size the instance is built at (what process() runs). */
    virtual float getBuiltSize() const noexcept = 0;
};

} // namespace spatcore::effects
