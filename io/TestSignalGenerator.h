#pragma once

#include <juce_audio_basics/juce_audio_basics.h>

#include <atomic>
#include <cmath>

namespace spatcore
{
namespace io
{

//==============================================================================
/**
    Output test-signal generator for patch verification: the operator clicks a
    hardware output and hears which speaker it is.

    Signal types:
      - PinkNoise    continuous, Gardner/Kellett 7-pole
      - Tone         continuous sine, 20 Hz - 20 kHz
      - Sweep        log sweep 20 Hz -> 20 kHz over 1 s, then 3 s of silence
      - DiracPulse   5 ms burst repeating every second

    HEARING PROTECTION: the continuous signals (pink noise and tone) always
    ramp up over 500 ms, and the ramp restarts every time the signal starts or
    the signal type changes. A rig under test can be pointed at someone's head
    at full system gain, so this is a safety property, not a nicety — do not
    shorten it, and give any new continuous signal type the same ramp. The
    transient types (sweep, pulse) carry their own envelopes and start at unity.

    Injected AFTER the spatial processing, straight into the hardware output
    buffer, and it REPLACES the samples on its target channel rather than mixing
    into them, so the operator hears the test signal alone.

    Thread model: the setters are message-thread and publish through atomics;
    renderNextBlock() owns the generator state and runs on the audio thread;
    prepare() runs with the device stopped.
*/
class TestSignalGenerator
{
public:
    enum class SignalType
    {
        Off,
        PinkNoise,
        Tone,
        Sweep,
        DiracPulse
    };

    TestSignalGenerator() = default;

    //==========================================================================
    // Setup (message thread / device start).
    //==========================================================================

    /** Must be called before renderNextBlock, and again whenever the device
        sample rate changes — the tone's phase increment is derived from it. */
    void prepare (double newSampleRate, int maxBlockSize)
    {
        juce::ignoreUnused (maxBlockSize);

        sampleRate = newSampleRate > 0.0 ? newSampleRate : 48000.0;

        for (int i = 0; i < 7; ++i)
            pinkNoiseState[i] = 0.0f;

        random = juce::Random (deterministicSeed != 0 ? deterministicSeed
                                                      : juce::Time::currentTimeMillis());

        phase = 0.0f;

        // Without this the tone is silent until something happens to call
        // setFrequency(): phaseIncrement starts at zero, so sin() never
        // advances. It also has to be recomputed here rather than only in
        // setFrequency(), or the tone comes back at the wrong pitch after a
        // sample-rate change.
        phaseIncrement = frequency / static_cast<float> (sampleRate);

        sweepPosition = 0.0f;
        pulsePosition = 0.0f;
        fadePosition.store (0.0f);
    }

    void setSignalType (SignalType type)
    {
        const SignalType oldType = currentType.load();

        if (oldType != type)
        {
            currentType.store (type);

            phase = 0.0f;
            sweepPosition = 0.0f;
            pulsePosition = 0.0f;

            // Restart the 500 ms protective ramp for the continuous signals;
            // the transient ones envelope themselves.
            if (type == SignalType::PinkNoise || type == SignalType::Tone)
                fadePosition.store (0.0f);
            else
                fadePosition.store (1.0f);
        }
    }

    /** Tone frequency, clamped to 20 Hz - 20 kHz. */
    void setFrequency (float hz)
    {
        frequency = juce::jlimit (20.0f, 20000.0f, hz);
        phaseIncrement = frequency / static_cast<float> (sampleRate);
    }

    /** Output level in dB. */
    void setLevel (float dB)
    {
        levelLinear.store (juce::Decibels::decibelsToGain (dB));
    }

    /** Target HARDWARE output channel, or -1 to stop. */
    void setOutputChannel (int channel)
    {
        const int oldChannel = targetChannel.load();
        targetChannel.store (channel);

        const SignalType type = currentType.load();

        // Restart the ramp on every start, so the operator never gets a step
        // into full level.
        if (channel >= 0 && oldChannel < 0)
            if (type == SignalType::PinkNoise || type == SignalType::Tone)
                fadePosition.store (0.0f);

        // Moving to another channel restarts the transient signals, so the
        // operator hears them from the beginning on the new speaker.
        if (type == SignalType::Sweep && channel >= 0 && channel != oldChannel)
        {
            sweepPosition = 0.0f;
            phase = 0.0f;
        }

        if (type == SignalType::DiracPulse && channel >= 0 && channel != oldChannel)
            pulsePosition = 0.0f;
    }

    /** Hold mode: keep the signal running after the mouse is released. */
    void setHoldEnabled (bool hold)   { holdEnabled.store (hold); }

    /** Seeds the pink-noise RNG for reproducible renders (offline tests).
        Zero, the default, seeds from the wall clock instead. Takes effect at
        the next prepare(). */
    void setDeterministicSeed (juce::int64 seed) noexcept   { deterministicSeed = seed; }

    //==========================================================================
    bool isActive() const
    {
        return targetChannel.load() >= 0 && currentType.load() != SignalType::Off;
    }

    SignalType getSignalType() const   { return currentType.load(); }
    float getLevelDb() const           { return juce::Decibels::gainToDecibels (levelLinear.load()); }
    float getFrequency() const         { return frequency; }
    bool isHoldEnabled() const         { return holdEnabled.load(); }
    int getOutputChannel() const       { return targetChannel.load(); }

    /** Stops all signal generation (thread-safe). */
    void reset()
    {
        targetChannel.store (-1);
        currentType.store (SignalType::Off);
        fadePosition.store (0.0f);
        holdEnabled.store (false);
    }

    //==========================================================================
    // Audio thread. REPLACES the target channel's samples.
    //==========================================================================
    void renderNextBlock (juce::AudioBuffer<float>& outputBuffer,
                          int startSample,
                          int numSamples)
    {
        const int channel = targetChannel.load();
        const SignalType signalType = currentType.load();
        const float level = levelLinear.load();

        if (channel < 0 || channel >= outputBuffer.getNumChannels()
             || signalType == SignalType::Off)
            return;

        auto* channelData = outputBuffer.getWritePointer (channel, startSample);
        float currentFade = fadePosition.load();
        const float fadeStep = 1.0f / (fadeDuration * static_cast<float> (sampleRate));

        for (int i = 0; i < numSamples; ++i)
        {
            float sample = 0.0f;

            switch (signalType)
            {
                case SignalType::PinkNoise:
                    sample = generatePinkNoise();
                    break;

                case SignalType::Tone:
                    sample = std::sin (phase * juce::MathConstants<float>::twoPi);
                    phase += phaseIncrement;
                    if (phase >= 1.0f)
                        phase -= 1.0f;
                    break;

                case SignalType::Sweep:
                    sample = generateSweep();
                    break;

                case SignalType::DiracPulse:
                    sample = (pulsePosition < pulseDuration) ? pulseAmplitude : 0.0f;
                    pulsePosition += 1.0f / static_cast<float> (sampleRate);
                    if (pulsePosition >= pulseDuration + pulseGap)
                        pulsePosition = 0.0f;
                    break;

                case SignalType::Off:
                default:
                    sample = 0.0f;
                    break;
            }

            sample *= level * currentFade;

            // Replace, not mix: the operator must hear the test signal alone.
            channelData[i] = sample;

            if (currentFade < 1.0f)
                currentFade = juce::jmin (1.0f, currentFade + fadeStep);
        }

        fadePosition.store (currentFade);
    }

private:
    //==========================================================================
    float generatePinkNoise()
    {
        // Gardner/Kellett 7-pole approximation of a 1/f spectrum.
        const float white = random.nextFloat() * 2.0f - 1.0f;

        pinkNoiseState[0] = 0.99886f * pinkNoiseState[0] + white * 0.0555179f;
        pinkNoiseState[1] = 0.99332f * pinkNoiseState[1] + white * 0.0750759f;
        pinkNoiseState[2] = 0.96900f * pinkNoiseState[2] + white * 0.1538520f;
        pinkNoiseState[3] = 0.86650f * pinkNoiseState[3] + white * 0.3104856f;
        pinkNoiseState[4] = 0.55000f * pinkNoiseState[4] + white * 0.5329522f;
        pinkNoiseState[5] = -0.7616f * pinkNoiseState[5] - white * 0.0168980f;

        const float pink = pinkNoiseState[0] + pinkNoiseState[1] + pinkNoiseState[2]
                         + pinkNoiseState[3] + pinkNoiseState[4] + pinkNoiseState[5]
                         + pinkNoiseState[6] + white * 0.5362f;

        pinkNoiseState[6] = white * 0.115926f;

        return pink * 0.11f;    // approximate normalisation
    }

    float generateSweep()
    {
        if (sweepPosition < sweepDuration)
        {
            const float t = sweepPosition / sweepDuration;

            const float logStart = std::log (sweepStartHz);
            const float logEnd   = std::log (sweepEndHz);
            const float currentFreq = std::exp (logStart + t * (logEnd - logStart));

            const float sample = std::sin (phase * juce::MathConstants<float>::twoPi);

            phase += currentFreq / static_cast<float> (sampleRate);
            if (phase >= 1.0f)
                phase -= 1.0f;

            sweepPosition += 1.0f / static_cast<float> (sampleRate);

            return sample;
        }

        sweepPosition += 1.0f / static_cast<float> (sampleRate);

        if (sweepPosition >= (sweepDuration + sweepGap))
        {
            sweepPosition = 0.0f;
            phase = 0.0f;
        }

        return 0.0f;    // silence during the gap
    }

    //==========================================================================
    // Control state (message thread <-> audio thread).
    std::atomic<int> targetChannel { -1 };
    std::atomic<float> fadePosition { 0.0f };
    std::atomic<bool> holdEnabled { false };
    std::atomic<SignalType> currentType { SignalType::Off };
    std::atomic<float> levelLinear { 0.01f };   // -40 dB

    // Generator state (audio thread; frequency/sampleRate are also read there
    // after a message-thread write, as in the original app implementation).
    double sampleRate = 48000.0;
    float frequency = 1000.0f;
    juce::int64 deterministicSeed = 0;

    float pinkNoiseState[7] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    juce::Random random;

    float phase = 0.0f;
    float phaseIncrement = 1000.0f / 48000.0f;

    float sweepPosition = 0.0f;
    static constexpr float sweepDuration = 1.0f;
    static constexpr float sweepGap = 3.0f;
    static constexpr float sweepStartHz = 20.0f;
    static constexpr float sweepEndHz = 20000.0f;

    float pulsePosition = 0.0f;
    static constexpr float pulseDuration = 0.005f;   // 5 ms burst
    static constexpr float pulseGap = 1.0f;          // 1 s between pulses
    static constexpr float pulseAmplitude = 2.0f;

    // 500 ms protective ramp — see the class comment before changing.
    static constexpr float fadeDuration = 0.5f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (TestSignalGenerator)
};

} // namespace io
} // namespace spatcore
