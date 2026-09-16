#pragma once

#include "../EffectModule.h"
#include "../../dsp/DelayTargetSmoother.h"
#include "../../dsp/FastDecibels.h"
#include "../../dsp/FractionalDelayLine.h"
#include "../../dsp/LFOWaveforms.h"
#include "../../dsp/LfoPhasor.h"
#include "../../dsp/OnePoleSmoother.h"
#include "../../dsp/OutputEQBiquadFilter.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

namespace spatcore::effects
{

/**
    Multitap delay: up to eight taps off one line, with a shelved feedback loop.

    The signal topology is the user's gen~ prototype, which is a single tap with
    feedback. The taps, the pattern modes, the per-tap levels, the diffusion and
    the glide are the plan's additions arranged around it. Where the prototype
    defines the path it is followed exactly, because that is the sound the user
    has already listened to:

    - The DRY is tapped BEFORE the input low cut, so the low cut shapes only
      what enters the line. Raising it thins the repeats and leaves the direct
      signal untouched.

    - The WET is tapped from the line ahead of the feedback gain and ahead of
      both shelves. The line is written at unity, so THE FIRST REPEAT IS AT FULL
      LEVEL WHATEVER THE FEEDBACK CONTROL SAYS, and the shelves colour only the
      repeats after it. Taking the wet from after the feedback multiply is the
      easy mistake, and it makes the whole effect vanish at low feedback
      settings where the prototype is still perfectly audible.

    - The two shelves sit INSIDE the loop, low then high, after the feedback
      multiply, so their curve compounds once per repeat rather than once in
      total.

    Two faults in the prototype's shelves are deliberately not reproduced: it
    runs the gain through dbtoa before a codebox that reads the result as
    decibels again (which leaves a shelf that can only ever boost, whatever the
    control says), and its alpha expression is missing a pair of brackets. Both
    shelves here are the shared OutputEQBiquadFilter at slope 0.7, which is the
    law the plan published and the one every other filter in the engine obeys.

    WHY THE FEEDBACK CEILING MOVES. The shelves live inside the loop, so a
    boosting shelf multiplies the loop gain: +12 dB on each shelf with the
    feedback at its nominal 0.95 maximum is a loop gain of 15, which is not a
    long delay but a siren. The feedback target is therefore divided by the
    worst-case shelf magnitude, max(0, loDb) + max(0, hiDb) in decibels, so the
    loop gain can never exceed 0.95 at any frequency. That bound is deliberately
    crude: it assumes both plateaus land on the same frequency, which they only
    do when the high shelf is set below the low shelf. Buying safety with a
    little feedback range at boosting settings is the right trade in a room with
    thirty-two of these feeding a speaker array.

    WHY NOTHING HERE IS A VARIANT. The buffer is sized once from
    ChainConfig::maxEffectDelaySeconds and never resized, so the one change that
    would truly need silence cannot happen. A tap-count change is carried by the
    per-tap level smoothers, which glide a dropped tap to zero instead of
    cutting it; a pattern or tap-mode change is carried by the per-tap
    DelayTargetSmoother, which glides or, for a big jump, runs its own
    mute-move-unmute envelope; a feedback-tap change crossfades between the old
    and the new read over one parameter glide. Making any of them a variant
    would fade the slot to silence and reset it, which throws away the delay
    tail the operator is listening to.

    WHY THE GLIDE WINDOW WAITS. glideMs sets the DelayTargetSmoother's window,
    and changing that window means re-preparing the smoother, which forgets
    where the read head currently is. Doing that mid-move would step the delay
    by the whole smoothing lag. The new window is therefore adopted at the first
    block where every tap is already sitting on the delay the bootstrap is about
    to hand it, which is the only moment the swap is inaudible.

    WHERE THE MODULATION SITS. Time modulation is multiplicative,
    t * (1 + depth * sin(2 pi rate t)), and it multiplies the SMOOTHER'S OUTPUT,
    one fresh value per sample. It is deliberately NOT folded into the target
    the smoother chases, which is the obvious reading of the plan's "through the
    same smoother" and is wrong three times over:

    - The smoother would have to decide whether a wobble is a teleport. A tap at
      375 ms wobbling 50 % at 10 Hz moves some three thousand samples between
      one 256-sample block and the next, and DelayTargetSmoother's threshold is
      three glide windows - six samples at glide 0. It would restart a
      mute-move-unmute envelope on every single block, for ever.

    - Glide and modulation would fight. The smoother is a box filter one glide
      window wide, and a box filter nulls outright at every multiple of its own
      reciprocal: at the plan's 200 ms default that is 5 Hz, 10 Hz, 15 Hz. The
      two mod controls would read as dead at exactly the settings the plan
      documents.

    - The prototype writes the modulated time straight to the delay-time inlet
      with no smoothing of any kind. The smoother is ours, and it belongs on the
      part the operator moves, not on the LFO.

    Nothing is given up by moving it: a sine is continuous and its phase
    advances one sample at a time, so the modulated delay is exactly as smooth
    as the base it multiplies, and the read head still cannot jump. The phase
    advances on every sample whether or not the depth is zero, so switching
    modulation on does not depend on when; only the cosine call is skipped at
    depth 0, and the multiply is then left out altogether so the smoothed delay
    comes through bit-identical.

    IDENTITY AND DENORMALS. Mix 0 leaves the buffer untouched sample by sample,
    so it is bit-transparent including negative zeros; the line keeps running
    underneath, because a delay that stopped filling would open a silent hole
    the length of its own delay time the moment the mix came back up. Diffusion
    0 skips both allpasses entirely, and their buffers are cleared once on the
    way down so the material they hold cannot reappear later. What is written
    into the line is flushed to true zero below 1e-20, and when a whole block
    passes with no input and no output the three filters are cleared, so a
    silent channel costs less than a loud one instead of more.

    Latency is reported as zero: the dry path is not delayed and the wet path IS
    the effect.
*/
class MultitapDelayModule : public IEffectModule
{
public:
    static constexpr int kMaxTaps = 8;

    ModuleId type() const noexcept override { return ModuleId::Delay; }

    void prepare (const ChainConfig& config) override
    {
        sampleRate = config.sampleRate > 0.0 ? config.sampleRate : 48000.0;
        samplesPerMs = static_cast<float> (sampleRate / 1000.0);

        const int block = config.maxBlock > 0 ? config.maxBlock : 1;

        // Negated comparison first so a NaN cap lands on the floor rather than
        // asking for a buffer of NaN samples.
        double seconds = config.maxEffectDelaySeconds;
        if (! (seconds > 0.05)) seconds = 0.05;
        if (seconds > 20.0)     seconds = 20.0;

        maxDelaySamples = static_cast<int> (seconds * sampleRate);
        if (maxDelaySamples < 2)
            maxDelaySamples = 2;

        maxDelayF  = static_cast<float> (maxDelaySamples);
        maxDelayMs = static_cast<float> (maxDelaySamples * 1000.0 / sampleRate);

        // The glide window is a SAMPLE count and the tap times are clamped
        // against a cap that has just moved, so both were computed against the
        // old rate. Recompute them here rather than waiting for the next
        // applyParams: the app happens to send one, but a module may not depend
        // on that, and reset() below adopts pendingWindow as it stands. Without
        // this a 200 ms glide set at 48 kHz runs as 50 ms after a move to
        // 192 kHz, until something else happens to touch the parameter.
        setGlideWindow (glideMs);
        baseTimeMs = clamp (baseTimeMs, 1.0f, maxDelayMs);

        for (int k = 0; k < kMaxTaps; ++k)
            manualMs[k] = clamp (manualMs[k], 1.0f, maxDelayMs);

        line.prepare (maxDelaySamples + block);

        inLoCut.prepare (sampleRate);
        fbLoShelf.prepare (sampleRate);
        fbHiShelf.prepare (sampleRate);
        lfo.prepare (sampleRate);

        // Two short, mutually inharmonic allpasses: long enough to smear a
        // repeat, short enough not to read as a second echo.
        diffuserA.prepare (static_cast<int> (0.00731 * sampleRate));
        diffuserB.prepare (static_cast<int> (0.01123 * sampleRate));

        // The tap smoothers are prepared by reset() below, onto the window this
        // prepare() has just recomputed. Doing it here as well would only
        // prepare them onto the outgoing one.
        for (int k = 0; k < kMaxTaps; ++k)
            tapGain[k].setTimeConstant (sampleRate, kParamTauSeconds);

        feedbackGain.setTimeConstant (sampleRate, kParamTauSeconds);
        wetMix.setTimeConstant (sampleRate, kParamTauSeconds);
        diffusion.setTimeConstant (sampleRate, kParamTauSeconds);
        fbBlend.setTimeConstant (sampleRate, kParamTauSeconds);

        // The five filter controls glide once per block, because a biquad only
        // solves its coefficients once per block; their time constant is set
        // against the block rate the first time a block arrives.
        lastBlockLen = 0;

        inLoCutHz.setSnapEpsilon (0.5f);            // hertz
        loShelfHz.setSnapEpsilon (0.5f);
        hiShelfHz.setSnapEpsilon (0.5f);
        loShelfDb.setSnapEpsilon (0.01f);           // decibels
        hiShelfDb.setSnapEpsilon (0.01f);

        reset();
    }

    void reset() noexcept override
    {
        line.reset();
        inLoCut.reset();
        fbLoShelf.reset();
        fbHiShelf.reset();
        lfo.reset();
        clearDiffusers();

        // reset() is the one moment a window swap costs nothing: everything is
        // about to be silent anyway.
        runningWindow = pendingWindow;

        for (int k = 0; k < kMaxTaps; ++k)
        {
            tapSmoother[k].prepare (runningWindow);
            tapGain[k].snap (tapGain[k].getTarget());
        }

        feedbackGain.snap (feedbackGain.getTarget());
        wetMix.snap (wetMix.getTarget());
        diffusion.snap (diffusion.getTarget());
        inLoCutHz.snap (inLoCutHz.getTarget());
        loShelfHz.snap (loShelfHz.getTarget());
        loShelfDb.snap (loShelfDb.getTarget());
        hiShelfHz.snap (hiShelfHz.getTarget());
        hiShelfDb.snap (hiShelfDb.getTarget());

        fbFrom = fbTo = fbPending;
        fbBlend.snap (1.0f);

        blockStart = 0;
        meterDb.store (spatcore::dsp::FastDecibels::kMinDb, std::memory_order_relaxed);
        snapOnNextApply = true;
    }

    ParamApplyInfo applyParams (const EffectChannelParams& params, int) noexcept override
    {
        const MultitapParams& d = params.delay;

        numTaps      = clampInt (static_cast<int> (d.taps), 1, kMaxTaps);
        manualTiming = (d.tapMode == 0);
        pattern      = clampInt (static_cast<int> (d.pattern), 0, 3);

        baseTimeMs = clamp (d.timeMs, 1.0f, maxDelayMs);

        for (int k = 0; k < kMaxTaps; ++k)
        {
            manualMs[k] = clamp (d.tapTimeMs[k], 1.0f, maxDelayMs);

            // A tap past the active count glides to silence instead of being
            // cut, so changing the count is a fade and not a click.
            tapGain[k].setTarget (k < numTaps
                                      ? spatcore::dsp::FastDecibels::dbToGain (clamp (d.tapLevelDb[k], -60.0f, 0.0f))
                                      : 0.0f);
        }

        userFeedback = clamp (d.feedback, 0.0f, 95.0f) * 0.01f;
        if (userFeedback > kFeedbackCeiling)
            userFeedback = kFeedbackCeiling;

        inLoCutHz.setTarget (clamp (d.inLoCutHz, 20.0f, 2000.0f));
        loShelfHz.setTarget (clamp (d.fbLoShelfHz, 20.0f, 2000.0f));
        loShelfDb.setTarget (clamp (d.fbLoShelfDb, -24.0f, 24.0f));
        hiShelfHz.setTarget (clamp (d.fbHiShelfHz, 1000.0f, 20000.0f));
        hiShelfDb.setTarget (clamp (d.fbHiShelfDb, -24.0f, 24.0f));

        lfo.setRateHz (clamp (d.modRateHz, 0.02f, 10.0f));
        modDepth = clamp (d.modDepthPct, 0.0f, 50.0f) * 0.01f;

        diffusion.setTarget (clamp (d.diffusion, 0.0f, 1.0f));
        wetMix.setTarget (clamp (d.mix, 0.0f, 100.0f) * 0.01f);

        setGlideWindow (d.glideMs);

        // 0 means "the last active tap", and so does any selection past the
        // active count, which is what a saved show holds after the count is
        // lowered.
        const int wanted = clampInt (static_cast<int> (d.feedbackTap), 0, kMaxTaps);
        fbPending = (wanted == 0 || wanted > numTaps) ? (numTaps - 1) : (wanted - 1);

        updateFeedbackTarget();

        if (snapOnNextApply)
        {
            for (int k = 0; k < kMaxTaps; ++k)
                tapGain[k].snap (tapGain[k].getTarget());

            inLoCutHz.snap (inLoCutHz.getTarget());
            loShelfHz.snap (loShelfHz.getTarget());
            loShelfDb.snap (loShelfDb.getTarget());
            hiShelfHz.snap (hiShelfHz.getTarget());
            hiShelfDb.snap (hiShelfDb.getTarget());
            diffusion.snap (diffusion.getTarget());
            wetMix.snap (wetMix.getTarget());

            // Not the redundant repeat of the call above that it looks like.
            // updateFeedbackTarget() takes max(current, target) on each shelf,
            // and the two shelf smoothers have just been snapped, so the first
            // call saw a `current` that no longer exists. Without this the
            // feedback would snap to a ceiling computed from a stale boost.
            updateFeedbackTarget();
            feedbackGain.snap (feedbackGain.getTarget());

            runningWindow = pendingWindow;
            for (int k = 0; k < kMaxTaps; ++k)
                tapSmoother[k].prepare (runningWindow);

            fbFrom = fbTo = fbPending;
            fbBlend.snap (1.0f);
            snapOnNextApply = false;
        }

        return { d.bypass != 0, false };
    }

    void process (float* inout, int numSamples) noexcept override
    {
        if (inout == nullptr || numSamples <= 0 || ! line.isPrepared())
            return;

        //----------------------------------------------------------------- block
        if (numSamples != lastBlockLen)
        {
            const double blockRate = sampleRate / static_cast<double> (numSamples);
            inLoCutHz.setTimeConstant (blockRate, kParamTauSeconds);
            loShelfHz.setTimeConstant (blockRate, kParamTauSeconds);
            loShelfDb.setTimeConstant (blockRate, kParamTauSeconds);
            hiShelfHz.setTimeConstant (blockRate, kParamTauSeconds);
            hiShelfDb.setTimeConstant (blockRate, kParamTauSeconds);
            lastBlockLen = numSamples;
        }

        const float cutHz = inLoCutHz.next();
        const float loHz  = loShelfHz.next();
        const float loDb  = loShelfDb.next();
        const float hiHz  = hiShelfHz.next();
        const float hiDb  = hiShelfDb.next();

        // Q 0.6 is the prototype's fixed alpha = sin(w)/1.2 on the input cut.
        inLoCut.setParameters (1, cutHz, 0.0f, 0.6f, 0.7f);

        // A shelf at its detent is switched OFF rather than run at unity: a
        // 0 dB biquad is only unity in exact arithmetic, and this one sits in a
        // feedback loop where the error would compound once per repeat.
        //
        // Switching one back ON has to start from a clean history.
        // OutputEQBiquadFilter::processSample returns early while the shape is
        // 0 WITHOUT shifting its delay line, and setParameters re-solves the
        // coefficients but never touches it, so x1/x2/y1/y2 still hold whatever
        // the loop was doing at the moment the shelf was switched off. Replaying
        // that as the first output of a filter INSIDE a feedback loop is a step
        // of the old loop amplitude, recirculated at up to 0.95 a repeat, on a
        // channel feeding a speaker array. The block-silence flush below cannot
        // help: it only fires on a channel that has gone quiet, which is exactly
        // the case this is not.
        const int loShape = isFlat (loDb) ? 0 : 2;
        const int hiShape = isFlat (hiDb) ? 0 : 5;

        if (loShape != 0 && fbLoShelf.getShape() == 0) fbLoShelf.reset();
        if (hiShape != 0 && fbHiShelf.getShape() == 0) fbHiShelf.reset();

        fbLoShelf.setParameters (loShape, loHz, loDb, 0.7f, 0.7f);
        fbHiShelf.setParameters (hiShape, hiHz, hiDb, 0.7f, 0.7f);

        updateFeedbackTarget();

        // The smoother is handed the NOMINAL tap time. The modulation is a
        // per-sample multiply on its output, further down - see the class
        // comment for why it is not folded in here.
        float rawTarget[kMaxTaps];
        for (int k = 0; k < kMaxTaps; ++k)
        {
            float d = tapTimeMs (k) * samplesPerMs;
            if (! (d > 1.0f))    d = 1.0f;
            if (d > maxDelayF)   d = maxDelayF;
            rawTarget[k] = d;
        }

        adoptGlideWindowIfStill (rawTarget);

        for (int k = 0; k < kMaxTaps; ++k)
            tapSmoother[k].observe (rawTarget[k], blockStart);

        float fixedDelay[kMaxTaps];
        bool  steady[kMaxTaps];
        for (int k = 0; k < kMaxTaps; ++k)
        {
            fixedDelay[k] = rawTarget[k];
            steady[k] = tapSmoother[k].isSteadyFrom (blockStart, fixedDelay[k]);
        }

        if (fbBlend.isSettled() && fbFrom != fbTo)
            fbFrom = fbTo;

        if (fbPending != fbTo && fbFrom == fbTo)
        {
            fbTo = fbPending;
            fbBlend.snap (0.0f);
            fbBlend.setTarget (1.0f);
        }

        // Read the taps that sound, plus any that are still fading out and
        // whichever two the feedback crossfade is between.
        int readCount = numTaps;
        for (int k = numTaps; k < kMaxTaps; ++k)
            if (tapGain[k].getCurrent() != 0.0f)
                readCount = k + 1;
        if (fbFrom + 1 > readCount) readCount = fbFrom + 1;
        if (fbTo   + 1 > readCount) readCount = fbTo + 1;

        const bool diffusing = (diffusion.getCurrent() > 0.0f) || (diffusion.getTarget() > 0.0f);

        // modDepth is a block-rate parameter, so the branch is hoisted: at
        // depth 0 the delay is handed to the line exactly as the smoother
        // produced it, with no multiply and no re-clamp to round it.
        const bool modulating = modDepth > 0.0f;

        bool  sourceActive = false;
        float wetPeak = 0.0f;

        //---------------------------------------------------------------- sample
        for (int i = 0; i < numSamples; ++i)
        {
            // The phase advances on EVERY sample, whatever the depth, so
            // switching modulation on does not depend on when. sin(2 pi p) is
            // written with the shared waveform, whose Sine is -cos(2 pi p): a
            // quarter turn makes the two the same function, and the delay then
            // starts at its nominal time rather than at an extreme.
            const float phase = lfo.nextPhase();
            float modScale = 1.0f;

            if (modulating)
            {
                float sinPhase = phase + 0.25f;
                if (sinPhase >= 1.0f)
                    sinPhase -= 1.0f;

                modScale = 1.0f + modDepth
                    * spatcore::dsp::LfoPhasor::shapeValue (spatcore::dsp::LFOWaveforms::Sine, sinPhase);
            }

            const float in = inout[i];
            const float hp = inLoCut.processSample (in);

            float raw[kMaxTaps];
            for (int k = 0; k < readCount; ++k)
            {
                float d, envelope;

                if (steady[k])
                {
                    d = fixedDelay[k];
                    envelope = 1.0f;
                }
                else
                {
                    const auto s = tapSmoother[k].smoothedAt (blockStart + i);
                    d = s.delay;
                    envelope = s.gain;
                }

                // t * (1 + depth * sin(2 pi rate t)), the plan's law, applied to
                // the smoothed delay. The prototype lets the modulated time fall
                // through its own 1 ms floor and relies on gen~'s internal clamp;
                // ours re-clamps into the buffer instead, which is the one
                // declared divergence on this path.
                if (modulating)
                {
                    d *= modScale;
                    if (! (d > 1.0f))       d = 1.0f;
                    else if (d > maxDelayF) d = maxDelayF;
                }

                // The line is read before it is written this sample, so a tap
                // meant to be d samples back asks for d - 1.
                const float r = line.readLinear (d - 1.0f);
                raw[k] = (envelope >= 1.0f) ? r : r * envelope;
            }

            float fbSource;
            if (fbFrom == fbTo)
            {
                fbSource = raw[fbTo];
            }
            else
            {
                const float b = fbBlend.next();
                fbSource = raw[fbFrom] + b * (raw[fbTo] - raw[fbFrom]);
            }

            // Gain first, then low shelf, then high shelf: the prototype's order,
            // which matters because a moving gain does not commute with a biquad.
            float recirculated = fbSource * feedbackGain.next();
            recirculated = fbLoShelf.processSample (recirculated);
            recirculated = fbHiShelf.processSample (recirculated);

            float lineIn = hp + recirculated;
            if (std::fabs (lineIn) < kTinyFloor)
                lineIn = 0.0f;

            line.write (lineIn);

            float wet = 0.0f;
            for (int k = 0; k < readCount; ++k)
                wet += raw[k] * tapGain[k].next();

            const float dif = diffusion.next();
            if (dif > 0.0f)
            {
                const float g = dif * kDiffuserMaxG;
                const float smeared = diffuserB.process (diffuserA.process (wet, g), g);

                // Blended rather than switched: at a small depth the allpasses
                // contribute a small amount of themselves, so turning diffusion
                // up from zero cannot step.
                wet += dif * (smeared - wet);
            }

            if (std::fabs (wet) < kTinyFloor)
                wet = 0.0f;

            if (wet > wetPeak)       wetPeak = wet;
            else if (-wet > wetPeak) wetPeak = -wet;

            if (in != 0.0f || lineIn != 0.0f)
                sourceActive = true;

            const float w = wetMix.next();
            if (w >= 1.0f)
                inout[i] = wet;
            else if (w > 0.0f)
                inout[i] = in + w * (wet - in);
            // w <= 0: the dry sample already in the buffer, untouched
        }

        blockStart += numSamples;

        // Nothing went in and nothing came out for a whole block: the filters
        // hold only tails below 1e-20, so clearing them is inaudible and stops
        // the recursions parking in denormal territory.
        if (! sourceActive && wetPeak == 0.0f)
        {
            inLoCut.reset();
            fbLoShelf.reset();
            fbHiShelf.reset();
        }

        if (! diffusing && ! diffusersIdle)
            clearDiffusers();
        else if (diffusing)
            diffusersIdle = false;

        meterDb.store (spatcore::dsp::FastDecibels::gainToDb (wetPeak), std::memory_order_relaxed);
    }

    int getLatencySamples() const noexcept override { return 0; }

    float getMeterDb() const noexcept override
    {
        return meterDb.load (std::memory_order_relaxed);
    }

private:
    //==========================================================================
    /** Schroeder allpass: flat magnitude, so it smears a repeat without
        colouring it. At g = 0 it degenerates to a plain delay, which is why the
        caller must skip it rather than run it at zero. */
    struct Allpass
    {
        void prepare (int len)
        {
            buffer.assign (static_cast<size_t> (len > 1 ? len : 1), 0.0f);
            pos = 0;
        }

        void clear() noexcept
        {
            std::fill (buffer.begin(), buffer.end(), 0.0f);
            pos = 0;
        }

        float process (float x, float g) noexcept
        {
            const float held = buffer[static_cast<size_t> (pos)];
            const float fed  = x + g * held;
            buffer[static_cast<size_t> (pos)] = fed;

            if (++pos >= static_cast<int> (buffer.size()))
                pos = 0;

            return held - g * fed;
        }

        std::vector<float> buffer;
        int pos = 0;
    };

    //==========================================================================
    static float clamp (float v, float lo, float hi) noexcept
    {
        // Negated comparisons so a NaN parameter lands on the low bound rather
        // than propagating into the audio.
        if (! (v > lo)) return lo;
        if (v > hi)     return hi;
        return v;
    }

    static int clampInt (int v, int lo, int hi) noexcept
    {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    static bool isFlat (float dB) noexcept { return std::fabs (dB) < 0.01f; }

    /** The tap's nominal time before modulation, already inside the buffer. */
    float tapTimeMs (int k) const noexcept
    {
        if (manualTiming)
            return manualMs[k];

        const float count = static_cast<float> (k + 1);
        float ms;

        switch (pattern)
        {
            case 1:  ms = 1.5f * baseTimeMs * count;            break;   // dotted
            case 2:  ms = (2.0f / 3.0f) * baseTimeMs * count;   break;   // triplet
            case 3:  ms = baseTimeMs * kGolden[k];              break;   // golden
            default: ms = baseTimeMs * count;                   break;   // equal
        }

        return clamp (ms, 1.0f, maxDelayMs);
    }

    /** Divides the feedback by the worst-case shelf magnitude so the loop gain
        cannot reach 1 at any frequency. max(current, target) on each shelf
        covers the whole of a glide, not just its endpoint. */
    void updateFeedbackTarget() noexcept
    {
        const float lo = std::max (loShelfDb.getCurrent(), loShelfDb.getTarget());
        const float hi = std::max (hiShelfDb.getCurrent(), hiShelfDb.getTarget());
        const float boostDb = (lo > 0.0f ? lo : 0.0f) + (hi > 0.0f ? hi : 0.0f);

        float g = userFeedback;

        if (boostDb > 0.0f)
        {
            const float ceiling = kFeedbackCeiling / spatcore::dsp::FastDecibels::dbToGain (boostDb);
            if (g > ceiling)
                g = ceiling;
        }

        feedbackGain.setTarget (g);
    }

    /** Re-prepares the per-tap smoothers onto a new glide window.

        Re-preparing forgets where the read head is, and the bootstrap
        observation then puts it on the raw target. That is only free while
        every tap is ALREADY sitting on the raw target it is about to be handed
        - the steady case. Anywhere else the read head would jump by the
        smoothing lag, which at a 200 ms window is a long way. A tap part way
        through a teleport envelope is never a good moment either, whatever the
        numbers say. The modulation is not in these numbers: it multiplies the
        smoother's output rather than its target, so a running LFO never delays
        the swap. */
    void adoptGlideWindowIfStill (const float* rawTarget) noexcept
    {
        if (pendingWindow == runningWindow)
            return;

        for (int k = 0; k < kMaxTaps; ++k)
        {
            const auto here = tapSmoother[k].smoothedAt (blockStart);
            if (here.gain < 1.0f)
                return;
            if (std::fabs (here.delay - rawTarget[k]) > kWindowSwapSlackSamples)
                return;
        }

        for (int k = 0; k < kMaxTaps; ++k)
            tapSmoother[k].prepare (pendingWindow);

        runningWindow = pendingWindow;
    }

    /** Stores the glide in MILLISECONDS as well as samples, so prepare() can
        rebuild the sample count against a new rate instead of carrying the old
        rate's number forward. */
    void setGlideWindow (float ms) noexcept
    {
        glideMs = clamp (ms, 0.0f, 2000.0f);
        pendingWindow = clampInt (static_cast<int> (glideMs * samplesPerMs), 2, 1 << 20);
    }

    void clearDiffusers() noexcept
    {
        diffuserA.clear();
        diffuserB.clear();
        diffusersIdle = true;
    }

    //==========================================================================
    static constexpr float kFeedbackCeiling = 0.95f;
    static constexpr float kDiffuserMaxG    = 0.7f;
    static constexpr float kTinyFloor       = 1.0e-20f;

    /** How far the read head may move when the glide window is swapped. A
        twentieth of a sample is the same order as the interpolation error the
        line already carries, so it cannot be heard. */
    static constexpr float kWindowSwapSlackSamples = 0.05f;

    /** phi^0 .. phi^7, so the golden pattern costs a lookup rather than a
        chain of multiplies whose error grows with the tap index. */
    static constexpr float kGolden[kMaxTaps] =
    {
        1.0f, 1.61803399f, 2.61803399f, 4.23606798f,
        6.85410197f, 11.09016994f, 17.94427191f, 29.03444185f
    };

    double sampleRate = 48000.0;
    float samplesPerMs = 48.0f;
    int maxDelaySamples = 240000;
    float maxDelayF = 240000.0f;
    float maxDelayMs = 5000.0f;

    spatcore::dsp::FractionalDelayLine line;
    spatcore::dsp::OutputEQBiquadFilter inLoCut, fbLoShelf, fbHiShelf;
    spatcore::dsp::LfoPhasor lfo;
    spatcore::dsp::DelayTargetSmoother tapSmoother[kMaxTaps];
    spatcore::dsp::OnePoleSmoother tapGain[kMaxTaps];
    spatcore::dsp::OnePoleSmoother feedbackGain, wetMix, diffusion, fbBlend;
    spatcore::dsp::OnePoleSmoother inLoCutHz, loShelfHz, loShelfDb, hiShelfHz, hiShelfDb;

    Allpass diffuserA, diffuserB;

    std::int64_t blockStart = 0;
    int lastBlockLen = 0;

    int numTaps = 3;
    int pattern = 0;
    bool manualTiming = false;
    float baseTimeMs = 375.0f;
    float manualMs[kMaxTaps] = { 375.0f, 750.0f, 1125.0f, 1500.0f, 1875.0f, 2250.0f, 2625.0f, 3000.0f };
    float modDepth = 0.0f;
    float userFeedback = 0.0f;

    float glideMs = 200.0f;        // the plan's default, kept in ms for prepare()
    int pendingWindow = 9600;      // the same window in samples at 48 kHz
    int runningWindow = 9600;
    int fbFrom = 0, fbTo = 0, fbPending = 0;

    bool diffusersIdle = true;
    bool snapOnNextApply = true;
    std::atomic<float> meterDb { spatcore::dsp::FastDecibels::kMinDb };
};

} // namespace spatcore::effects
