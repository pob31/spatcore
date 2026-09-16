#pragma once

#include "../EffectModule.h"
#include "../../dsp/DcBlocker.h"
#include "../../dsp/FastDecibels.h"
#include "../../dsp/FractionalDelayLine.h"
#include "../../dsp/OnePoleSmoother.h"
#include "../../dsp/OutputEQBiquadFilter.h"
#include "../../dsp/Waveshaper.h"

#include <juce_dsp/juce_dsp.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace spatcore::effects
{

/**
    Solid-state clipping through to tube-like saturation, from the user's gen~
    prototype.

    The stage order is the prototype's: pre low shelf, pre high shelf, drive in
    decibels, a crossfade between a hard clip at +-0.8 and a tanh, output gain,
    post low shelf, post high shelf, and a mix against the completely untouched
    input. Several things inside it are deliberately NOT the prototype's, and
    each avoids something the prototype does by accident.

    THE SHELVES are the shared RBJ pair of OutputEQBiquadFilter (shapes 2 and 5
    at slope 0.7), not a port of the prototype's codebox. The prototype pushes
    its gain inlet through dbtoa and then applies A = 10^(gain/40) to the
    already-linear result, so its shelves can never cut and are never flat: a
    nominal 0 dB is +1.00 dB and +40 dB is +100 dB. Its alpha expression is also
    missing a pair of brackets, evaluating sqrt((A + 1/A)/S + 1) where RBJ wants
    sqrt((A + 1/A)(1/S - 1) + 2), which widens the corner by about 16 % even at
    the 0 dB operating point. Both are corrected here. The high shelf runs at
    slope 0.7 like the low one rather than at the prototype's 0, because under
    the corrected alpha a slope of 0 asks for the square root of 2 - (A + 1/A),
    which is negative for every boost and would hand the filter a NaN.

    A shelf asked for exactly 0 dB is switched to shape 0, which is a TRUE
    bypass rather than a biquad running at unity. That is what lets the default
    settings cost no filtering at all and keeps an untouched signal untouched.
    Its state is dropped on the way out, because a frozen x[n-1] would step the
    signal the next time that shelf is dialled back in.

    THE BIAS is an addition: the prototype feeds tanh the unbiased signal and
    has no asymmetry anywhere, so it can only ever make odd harmonics.
    Waveshaper::blend subtracts tanh(bias) to keep the curve through the origin,
    and that tanh must not be evaluated per sample. So the shaper's two controls
    are taken as a linear ramp across the block, between the value the block
    starts at and the value it ends at, and the ramp for tanh(bias) is built
    from the same two points. The block's end value becomes the next block's
    start value, which costs exactly one std::tanh per block and leaves no step
    at the boundary; a ramp that restarted from a stale tanh would inject a
    discontinuity the size of the bias move straight into the audio.

    THE SHAPE CONTROL reads the opposite way to the prototype's "waveform",
    which is a percentage of the CLIP leg. Here 0 is the hard clip and 1 the
    tanh, which is the published parameter surface.

    THE DC BLOCKER is switched IN by a non-zero bias with the tanh leg actually
    in use. With no bias, or with the shape control sitting on the hard clip,
    the curve is odd-symmetric and cannot make DC, so a channel that never asks
    for one never pays for it. It is NOT switched back out when the bias comes
    off, though. A 5 Hz high pass is not transparent to low-frequency material
    - at 50 Hz it still turns the phase by 5.7 degrees - so taking it out of
    live audio in one sample is a step rather than a no-op: an ordinary move of
    the shape or bias fader broke the second difference of a 0.5-peak 50 Hz
    sine from a 4.7e-5 baseline to 8.5e-2, 1800 times the curvature that signal
    has of its own, which is heard as a click. Switching it IN is free of that,
    because a cleared blocker's first output is its input exactly. So once
    engaged it stays in circuit until the state is dropped anyway - at silence,
    at a reset, at a variant commit, or on the way into mix 0 - which is the
    only moment its one-sample history is not a piece of live audio.

    OVERSAMPLING is the other addition (the prototype hard clips at base rate).
    The factor is a VARIANT parameter because changing it changes the reported
    latency, so it waits for the slot to reach silence. Both oversamplers are
    built in prepare() and only SELECTED at commit time: commitPendingVariant()
    runs on the audio thread, so it must not be able to allocate. Auto is 4x at
    or below 48 kHz, 2x at 88.2 and 96 kHz, and off from 176.4 kHz up.

    The oversampler is the module's only latency, and the dry leg of the mix is
    delayed to match it. Without that a partial mix would sum a delayed copy
    against an undelayed one and comb, and the module's reported latency would
    describe only half of its own output.

    MIX 0 IS TRANSPARENT AT THE LATENCY THE MODULE REPORTS. With the
    oversampler off that is the plain identity: the block is handed back
    untouched, not multiplied by one, so negative zeros and denormals survive
    it. With the oversampler ON there is no untouched block to hand back. Every
    other mix setting delivers the dry leg through the alignment delay, so
    returning x[n] at mix 0 while mix 0.01 returns x[n-5] would shift the whole
    channel five samples through time the instant the mix smoother landed on
    zero - measured at 57 % of peak amplitude on a 1 kHz sine, and again on the
    way back off zero - and would leave getLatencySamples() reporting a delay on
    a block it had just passed through undelayed. The idle path therefore keeps
    running the alignment line AND returning its output; what it skips is the
    shelves, the shaper and the oversampler. Their tails are dropped once on the
    way in, so coming back off zero cannot replay a moment from before. The
    alignment line is the one thing not dropped, because it holds nothing but
    raw input and clearing it would punch a hole the width of the latency into
    the dry leg.

    THE SHAPER'S TWO CONTROLS RAMP BETWEEN BLOCK ENDPOINTS and the shelves are
    retuned once a block, so while either is MOVING the render depends on the
    block size the host hands over - measured 2.8e-2 on a 0.4 signal over a full
    shape glide, one 1024-sample call against 8 x 128. Settled, it does not.
    That is what the one std::tanh a block buys; the sibling modules, whose
    controls need no transcendental, step everything per sample instead.

    LATENCY only changes inside commitPendingVariant(), with one exception: the
    first applyParams() after prepare() snaps rather than glides, and picks up
    the auto factor's latency along with everything else. A host reading the
    chain's latency must read it after that first apply, not between prepare()
    and it.

    No allocation after prepare().
*/
class DistortionModule : public IEffectModule
{
public:
    static constexpr int kNumShelves = 4;       // pre lo, pre hi, post lo, post hi

    ModuleId type() const noexcept override { return ModuleId::Dist; }

    void prepare (const ChainConfig& config) override
    {
        sampleRate = config.sampleRate > 0.0 ? config.sampleRate : 48000.0;
        maxBlock = config.maxBlock > 0 ? config.maxBlock : 1;

        driveGain.setTimeConstant (sampleRate, kParamTauSeconds);
        outputGain.setTimeConstant (sampleRate, kParamTauSeconds);
        wet.setTimeConstant (sampleRate, kParamTauSeconds);
        shapeSm.setTimeConstant (sampleRate, kParamTauSeconds);
        biasSm.setTimeConstant (sampleRate, kParamTauSeconds);
        biasSm.setSnapEpsilon (1.0e-6f);            // bias spans +-0.5, so 1e-4 would be a visible step

        for (int i = 0; i < kNumShelves; ++i)
        {
            shelfHz[i].setTimeConstant (sampleRate, kParamTauSeconds);
            shelfHz[i].setSnapEpsilon (0.5f);       // hertz
            shelfDb[i].setTimeConstant (sampleRate, kParamTauSeconds);
            shelfDb[i].setSnapEpsilon (1.0e-3f);    // decibels, and it must reach 0 exactly
            shelf[i].prepare (sampleRate);
        }

        dcBlocker.prepare (sampleRate);             // 5 Hz, well below anything musical
        dry.assign (static_cast<std::size_t> (maxBlock), 0.0f);

        using OS = juce::dsp::Oversampling<float>;
        os2x = std::make_unique<OS> (static_cast<std::size_t> (1), static_cast<std::size_t> (1),
                                     OS::filterHalfBandPolyphaseIIR, false, true);
        os4x = std::make_unique<OS> (static_cast<std::size_t> (1), static_cast<std::size_t> (2),
                                     OS::filterHalfBandPolyphaseIIR, false, true);
        os2x->initProcessing (static_cast<std::size_t> (maxBlock));
        os4x->initProcessing (static_cast<std::size_t> (maxBlock));

        // Integer latency is asked of the oversampler, so this rounding is a
        // formality rather than a compensation of its own.
        latency2x = static_cast<int> (std::lround (os2x->getLatencyInSamples()));
        latency4x = static_cast<int> (std::lround (os4x->getLatencyInSamples()));

        dryAlign.prepare (latency4x > latency2x ? latency4x : latency2x);

        runningLog2 = 0;
        pendingLog2 = 0;
        latencySamples.store (0, std::memory_order_relaxed);
        snapOnNextApply = true;

        reset();
    }

    void reset() noexcept override
    {
        clearAudioState();

        driveGain.snap (driveGain.getTarget());
        outputGain.snap (outputGain.getTarget());
        wet.snap (wet.getTarget());
        shapeSm.snap (shapeSm.getTarget());
        biasSm.snap (biasSm.getTarget());
        tanhBias = std::tanh (biasSm.getCurrent());

        for (int i = 0; i < kNumShelves; ++i)
        {
            shelfHz[i].snap (shelfHz[i].getTarget());
            shelfDb[i].snap (shelfDb[i].getTarget());
        }

        meterDb.store (spatcore::dsp::FastDecibels::kMinDb, std::memory_order_relaxed);
        idle = true;                                // there is nothing left to drop
    }

    ParamApplyInfo applyParams (const EffectChannelParams& params, int) noexcept override
    {
        namespace fd = spatcore::dsp::FastDecibels;
        const DistortionParams& d = params.dist;

        driveGain.setTarget (fd::dbToGain (clamp (d.driveDb, 0.0f, 40.0f)));
        shapeSm.setTarget (clamp (d.shape, 0.0f, 1.0f));
        biasSm.setTarget (clamp (d.bias, -0.5f, 0.5f));
        outputGain.setTarget (fd::dbToGain (clamp (d.outputDb, -24.0f, 12.0f)));
        wet.setTarget (clamp (d.mix, 0.0f, 100.0f) * 0.01f);

        shelfHz[0].setTarget (clamp (d.preLoShelfHz, 20.0f, 2000.0f));
        shelfDb[0].setTarget (clamp (d.preLoShelfDb, -24.0f, 24.0f));
        shelfHz[1].setTarget (clamp (d.preHiShelfHz, 1000.0f, 20000.0f));
        shelfDb[1].setTarget (clamp (d.preHiShelfDb, -24.0f, 24.0f));
        shelfHz[2].setTarget (clamp (d.postLoShelfHz, 20.0f, 2000.0f));
        shelfDb[2].setTarget (clamp (d.postLoShelfDb, -24.0f, 24.0f));
        shelfHz[3].setTarget (clamp (d.postHiShelfHz, 1000.0f, 20000.0f));
        shelfDb[3].setTarget (clamp (d.postHiShelfDb, -24.0f, 24.0f));

        pendingLog2 = factorLog2For (d.oversample);

        if (snapOnNextApply)
        {
            // First set after prepare/reset: take the values, do not glide to
            // them, or every project load would ramp its effects in.
            driveGain.snap (driveGain.getTarget());
            outputGain.snap (outputGain.getTarget());
            wet.snap (wet.getTarget());
            shapeSm.snap (shapeSm.getTarget());
            biasSm.snap (biasSm.getTarget());
            tanhBias = std::tanh (biasSm.getCurrent());

            for (int i = 0; i < kNumShelves; ++i)
            {
                shelfHz[i].snap (shelfHz[i].getTarget());
                shelfDb[i].snap (shelfDb[i].getTarget());
            }

            runningLog2 = pendingLog2;
            latencySamples.store (latencyFor (runningLog2), std::memory_order_relaxed);
            snapOnNextApply = false;
        }

        return { d.bypass != 0, pendingLog2 != runningLog2 };
    }

    void commitPendingVariant() noexcept override
    {
        runningLog2 = pendingLog2;
        latencySamples.store (latencyFor (runningLog2), std::memory_order_relaxed);
        clearAudioState();
    }

    void process (float* inout, int numSamples) noexcept override
    {
        if (inout == nullptr || numSamples <= 0)
            return;

        // A host, or a direct caller, may hand over more than the block this
        // was sized for. Chunking here is what keeps process() free of any
        // allocation and keeps the oversampler inside its own buffers.
        int offset = 0;
        while (offset < numSamples)
        {
            const int remaining = numSamples - offset;
            const int chunk = remaining < maxBlock ? remaining : maxBlock;
            processChunk (inout + offset, chunk);
            offset += chunk;
        }
    }

    int getLatencySamples() const noexcept override
    {
        return latencySamples.load (std::memory_order_relaxed);
    }

    /** Output peak of the last block, for a level readout beside the drive. */
    float getMeterDb() const noexcept override
    {
        return meterDb.load (std::memory_order_relaxed);
    }

private:
    static float clamp (float v, float lo, float hi) noexcept
    {
        // Negated comparisons so a NaN parameter lands on the low bound rather
        // than propagating into the audio.
        if (! (v > lo)) return lo;
        if (v > hi)     return hi;
        return v;
    }

    /** Runs a per-sample smoother forward by a whole block. Settled is the
        common case and costs one comparison; it is only while a control is
        actually moving that this walks. */
    static void advance (spatcore::dsp::OnePoleSmoother& s, int n) noexcept
    {
        if (s.isSettled())
            return;

        for (int i = 0; i < n; ++i)
            s.next();
    }

    int factorLog2For (std::uint8_t oversample) const noexcept
    {
        switch (oversample)
        {
            case 1: return 0;                       // off
            case 2: return 1;                       // 2x
            case 3: return 2;                       // 4x
            default: break;                         // 0 = auto, and anything unrecognised
        }

        if (sampleRate <= 50000.0)  return 2;       // 44.1 and 48 kHz
        if (sampleRate <= 120000.0) return 1;       // 88.2 and 96 kHz
        return 0;                                   // 176.4 kHz and up
    }

    int latencyFor (int log2Factor) const noexcept
    {
        if (log2Factor == 1) return latency2x;
        if (log2Factor == 2) return latency4x;
        return 0;
    }

    /** Everything the WET leg carries. The alignment line is deliberately not
        in here: it holds raw input rather than any tail of the effect, and it
        is still feeding the dry leg at mix 0, where clearing it would punch a
        hole the width of the latency into an otherwise transparent path. */
    void clearWetState() noexcept
    {
        for (int i = 0; i < kNumShelves; ++i)
            shelf[i].reset();

        dcBlocker.reset();
        dcEngaged = false;

        if (os2x != nullptr) os2x->reset();
        if (os4x != nullptr) os4x->reset();
    }

    void clearAudioState() noexcept
    {
        clearWetState();
        dryAlign.reset();
    }

    /** A shelf at exactly 0 dB becomes shape 0, which OutputEQBiquadFilter
        treats as a true bypass. The reset on the way out stops a frozen
        history from popping the next time the shelf is asked to do something. */
    void tuneShelf (int index, int shelfShape) noexcept
    {
        const float hz = shelfHz[index].getCurrent();
        const float dB = shelfDb[index].getCurrent();
        const int want = (dB == 0.0f) ? 0 : shelfShape;

        if (want == 0 && shelf[index].getShape() != 0)
            shelf[index].reset();

        // q is inert on a shelf: OutputEQBiquadFilter's shelf coefficients are
        // built from the slope alone, so 0.7 here is a placeholder, not a Q.
        shelf[index].setParameters (want, hz, dB, 0.7f, 0.7f);
    }

    /** The waveshaper, over whichever rate it is running at. The two controls
        arrive as the block's endpoints and are walked linearly between them. */
    static void shapeBlock (float* data, int count,
                            float s0, float s1,
                            float b0, float b1,
                            float t0, float t1) noexcept
    {
        if (count <= 0)
            return;

        const float inv = 1.0f / static_cast<float> (count);

        // Pure hard clip earns its own loop: it is the only path with no libm
        // call in it, so it renders identically on every platform, and it skips
        // the tanh that dominates the cost of the other one.
        if (s0 == 0.0f && s1 == 0.0f)
        {
            for (int i = 0; i < count; ++i)
                data[i] = spatcore::dsp::Waveshaper::hardClip (data[i]);

            return;
        }

        const float ds = (s1 - s0) * inv;
        const float db = (b1 - b0) * inv;
        const float dt = (t1 - t0) * inv;

        float s = s0, b = b0, t = t0;

        for (int i = 0; i < count; ++i)
        {
            s += ds;
            b += db;
            t += dt;
            data[i] = spatcore::dsp::Waveshaper::blend (data[i], s, b, t);
        }
    }

    void processChunk (float* buf, int n) noexcept
    {
        const int align = latencySamples.load (std::memory_order_relaxed);

        // Mix 0 is the identity, and an identity has to be exact: the input
        // comes back rather than being crossfaded against itself. What the
        // module owes at mix 0 is the identity AT THE LATENCY IT REPORTS, so
        // with the oversampler on the alignment line keeps running and keeps
        // delivering - skipping it would hand back x[n] where mix 0.01 hands
        // back x[n - align], a jump of the whole channel through time. With no
        // oversampler there is no latency to honour and the block is not
        // touched at all, negative zeros and denormals included.
        if (wet.isSettled() && wet.getCurrent() == 0.0f)
        {
            if (! idle)
            {
                clearWetState();            // the wet tail goes; the dry line does not
                idle = true;
            }

            float peak = 0.0f;

            for (int i = 0; i < n; ++i)
            {
                float y = buf[i];

                if (align > 0)
                {
                    dryAlign.write (y);
                    y = dryAlign.readInteger (align);
                    buf[i] = y;
                }

                const float a = std::fabs (y);
                if (a > peak)
                    peak = a;
            }

            // The meter is an OUTPUT reading, and at mix 0 the output is the
            // dry signal. Freezing it at whatever the wet leg last managed
            // would leave the GUI showing a level the module is not making.
            meterDb.store (spatcore::dsp::FastDecibels::gainToDb (peak), std::memory_order_relaxed);
            return;
        }

        idle = false;

        float inPeak = 0.0f;

        for (int i = 0; i < n; ++i)
        {
            const float x = buf[i];
            dry[static_cast<std::size_t> (i)] = x;
            const float a = std::fabs (x);
            if (a > inPeak)
                inPeak = a;
        }

        // The biquads are retuned once per block, to the value the block
        // STARTS on: recomputing coefficients per sample is what the prototype
        // does and it is far too expensive here, and tuning to the value the
        // block ENDS on would run the whole block at a setting it has not
        // reached yet. The smoothers are walked once the block is filtered.
        tuneShelf (0, 2 /* LowShelf */);
        tuneShelf (1, 5 /* HighShelf */);
        shelf[0].processBlock (buf, n);
        shelf[1].processBlock (buf, n);

        // The drive is linear, so it is applied at the base rate: putting it
        // inside the oversampled section would buy nothing and would make the
        // glide rate depend on the oversampling factor.
        for (int i = 0; i < n; ++i)
            buf[i] *= driveGain.next();

        const float s0 = shapeSm.getCurrent();
        advance (shapeSm, n);
        const float s1 = shapeSm.getCurrent();

        const float b0 = biasSm.getCurrent();
        advance (biasSm, n);
        const float b1 = biasSm.getCurrent();
        const float t0 = tanhBias;
        const float t1 = std::tanh (b1);            // the one std::tanh of the bias per block
        tanhBias = t1;

        // The pointer is checked rather than assumed: applyParams() can set
        // runningLog2 from its snap branch, so a caller that skipped prepare()
        // would dereference a null unique_ptr here. clearWetState() already
        // guards the same two pointers, and the base-rate leg is the right
        // fallback - it is what oversampling off runs.
        juce::dsp::Oversampling<float>* os = nullptr;

        if (runningLog2 == 2)       os = os4x.get();
        else if (runningLog2 == 1)  os = os2x.get();

        if (os != nullptr)
        {
            float* channels[1] = { buf };
            juce::dsp::AudioBlock<float> block (channels, static_cast<std::size_t> (1),
                                                static_cast<std::size_t> (n));

            juce::dsp::AudioBlock<float> up = os->processSamplesUp (block);

            // An oversampler that has not been given its buffers hands back an
            // empty block; asking that for a channel pointer is a null read.
            if (up.getNumSamples() > 0)
                shapeBlock (up.getChannelPointer (0), static_cast<int> (up.getNumSamples()),
                            s0, s1, b0, b1, t0, t1);

            os->processSamplesDown (block);
        }
        else
        {
            shapeBlock (buf, n, s0, s1, b0, b1, t0, t1);
        }

        // Only the biased tanh leg can make DC, so a channel that never asks
        // for bias never pays for the blocker: with no bias, or with the shape
        // control sitting on the hard clip, the curve is odd-symmetric and the
        // blocker would be a 5 Hz high pass charged for nothing.
        //
        // But it only ever switches ON here. A cleared blocker's first output
        // is its input exactly, so engaging costs no discontinuity; taking a
        // 5 Hz high pass back OUT of live audio does, because it is not
        // transparent to low-frequency material - it turns the phase of a 50 Hz
        // tone by 5.7 degrees, and dropping that in one sample is a step of
        // some 5 % of amplitude, i.e. a click on an ordinary fader move. It is
        // retired by clearWetState() instead, which runs at silence, at a
        // reset, at a variant commit and on the way into mix 0 - the moments
        // when its history is not a piece of live audio.
        if ((b0 != 0.0f || b1 != 0.0f) && (s0 != 0.0f || s1 != 0.0f))
            dcEngaged = true;

        if (dcEngaged)
            dcBlocker.processBlock (buf, n);

        for (int i = 0; i < n; ++i)
            buf[i] *= outputGain.next();

        tuneShelf (2, 2 /* LowShelf */);
        tuneShelf (3, 5 /* HighShelf */);
        shelf[2].processBlock (buf, n);
        shelf[3].processBlock (buf, n);

        for (int i = 0; i < kNumShelves; ++i)
        {
            advance (shelfHz[i], n);
            advance (shelfDb[i], n);
        }

        float outPeak = 0.0f;

        for (int i = 0; i < n; ++i)
        {
            const float w = wet.next();
            const float raw = dry[static_cast<std::size_t> (i)];

            float d = raw;

            if (align > 0)
            {
                dryAlign.write (raw);
                d = dryAlign.readInteger (align);
            }

            float y;

            if (w >= 1.0f)      y = buf[i];         // fully wet, untouched by any crossfade
            else if (w > 0.0f)  y = d + w * (buf[i] - d);
            else                y = d;

            buf[i] = y;
            const float a = std::fabs (y);
            if (a > outPeak)
                outPeak = a;
        }

        meterDb.store (spatcore::dsp::FastDecibels::gainToDb (outPeak), std::memory_order_relaxed);

        // A silent input whose tail has already gone is the cheapest moment to
        // drop the filter state. Left alone, four biquads and an oversampler
        // decay into denormals, where a SILENT channel costs more than a loud
        // one; this is two comparisons a block and needs no FTZ to work.
        //
        // Deliberately NOT guarded by an "already flushed" flag. The clear is a
        // few kilobytes of memset against a block that has just run four
        // biquads and, on the tanh leg, one std::tanh per oversampled sample.
        // Measured over 4000 silent 512-sample blocks at 4x, adding the flag
        // moved nothing (77.1 ms against 78.1 ms; 59.6 against 60.1 on the clip
        // leg), and the silent channel was already cheaper than the loud one it
        // is supposed to be keeping up with (103 ms and 69 ms). A flag every
        // clear path has to keep in sync is not worth a result inside the
        // noise.
        if (inPeak == 0.0f && outPeak < 1.0e-20f)
            clearAudioState();
    }

    double sampleRate = 48000.0;
    int maxBlock = 512;

    spatcore::dsp::OnePoleSmoother driveGain, outputGain, wet, shapeSm, biasSm;
    spatcore::dsp::OnePoleSmoother shelfHz[kNumShelves], shelfDb[kNumShelves];
    spatcore::dsp::OutputEQBiquadFilter shelf[kNumShelves];
    spatcore::dsp::DcBlocker dcBlocker;
    spatcore::dsp::FractionalDelayLine dryAlign;

    std::unique_ptr<juce::dsp::Oversampling<float>> os2x, os4x;
    std::vector<float> dry;

    float tanhBias = 0.0f;
    int latency2x = 0, latency4x = 0;
    int runningLog2 = 0, pendingLog2 = 0;
    bool dcEngaged = false;
    bool snapOnNextApply = true;
    bool idle = true;

    std::atomic<int> latencySamples { 0 };
    std::atomic<float> meterDb { spatcore::dsp::FastDecibels::kMinDb };
};

} // namespace spatcore::effects
