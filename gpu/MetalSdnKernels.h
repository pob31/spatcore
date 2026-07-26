#pragma once

/*
    MSL kernel source for the native GPU SDN reverb backend, embedded as a
    string literal (compiled at prepare() time by MetalSdnBackend, like the
    other native kernels). KEEP IN SYNC with CudaSdnKernels.h — the two strings
    implement the byte-identical algorithm; only the language scaffolding differs.

    TWO kernels, bit-identical to each other and selected per block by the host
    (mirroring CudaSdnKernels.h):

    sdn_process       — the original lockstep. ONE threadgroup, one THREAD per
                        node (N <= MAX_NODES = 32), one barrier per sample. That
                        barrier must stay mem_device: thread n writes
                        delayLines[pathIndex(n,i)] at sample s and thread i reads
                        it at s+1, which is cross-thread DEVICE traffic that
                        mem_threadgroup does NOT order. Correct for ANY geometry,
                        including single-sample inter-node delays, but it occupies
                        exactly one GPU core. Retained as the fallback.

    sdn_process_nodes — grid = numNodes threadgroups (so the work spreads across
                        cores), 32 lanes each, one lane per path. Requires the
                        host to launch it in chunks of C samples where every
                        effective delay is >= C (SdnHostConfig::
                        chunkSamplesForBlock); inside such a window nodes are
                        mutually independent, so no cross-threadgroup barrier is
                        needed. A chunk boundary is a DISPATCH boundary — on
                        Metal a serial compute encoder orders successive
                        dispatchThreadgroups and makes each one's side effects
                        visible to the next, which is the analogue of a CUDA
                        kernel-launch boundary. See MetalSdnBackend.mm.

    Both process [sampleOffset, sampleOffset + chunkSamples) of the block. All
    sample-derived quantities stay BLOCK-relative — ringWritePos advances once
    per block, and the crossfade term is mix + crossfadeRate * s — so s must
    remain the absolute in-block index, not the in-chunk one.

    A sample-exact port of the synchronous SDNAlgorithm::processBlock
    (spatcore/reverb/ReverbSDNAlgorithm.h). The network is recurrent and couples
    nodes within a block (an inter-node delay can be shorter than the block).
    A node owns its diffuser + tone/DC output state; the per-path delay lines and
    3-band decay-filter state live in device buffers indexed by pathIndex.

    Per sample (CPU op order preserved exactly):
      1. read incoming from every path i->n at the snapshot read head
         (ringWritePos + s - delay), with the crossfade dual-tap when a path's
         mix < 1; Householder X = (2/(N-1)) * sum(incoming).
      2. output = sum_i (X - incoming[i]) accumulated in path order (independent
         of, and before, the decay filter).
      3. diffuse the node input through 2 allpass stages (gated diffusion>1e-4,
         state frozen when off).
      4. for every outgoing path n->i: signal = (X - incoming[i]) + diffused/N,
         3-band decay filter, write at the shared write head (ringWritePos + s).
      5. tone one-pole LPF -> DC blocker -> * sdnOutputGain -> outputs[n][s].
      BARRIER (mem_device) so this sample's cross-node writes are visible before
      the next sample's reads. One barrier suffices: every delay >= 1, so a
      sample never reads a cell written this same sample.

    The single ring write head and modulo-MAX_DELAY_SAMPLES addressing match
    SDNAlgorithm exactly; the host advances ringWritePos by blockSize per launch.
    Parameters/geometry are applied stepwise on the pump thread (no in-kernel
    ramp); the crossfade mix is the block-start snapshot, advanced host-side.
*/

static const char* const kSdnProcessKernelSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint NUM_DIFFUSERS = 2u;

struct SdnParams
{
    uint  numNodes;
    uint  numPaths;
    uint  blockSize;
    uint  maxDelaySamples;   // per-path delay-line stride (8192)
    uint  ringWritePos;      // block-start write head (shared by all paths)
    uint  maxDiffLen;        // per-stage diffuser-ring stride
    float diffusionCoeff;    // diffusion * 0.5
    float toneCoeff;         // one-pole 8 kHz LPF
    float lowCoeff;          // 3-band crossover (low)
    float highCoeff;         // 3-band crossover (high)
    float dcPole;            // 0.9995
    float sdnOutputGain;     // (1 + 18/N) * 0.25
    float inputDistribution; // 1/N
    float crossfadeRate;     // global; 1/(sr*0.01) while crossfading
    uint  sampleOffset;      // first in-block sample of this chunk
    uint  chunkSamples;      // samples in this chunk (== blockSize when unchunked)
};

// This struct exists TWICE (here and SdnParamsGpu in MetalSdnBackend.mm) and it
// travels through setBytes, so a field-order or size mismatch is not a compile
// error anywhere — it is silent garbage. HIP shipped exactly that bug. 16
// consecutive 4-byte scalars pack with no padding: 64 bytes, align 4.
static_assert (sizeof (SdnParams) == 64,
               "SdnParams must match SdnParamsGpu in MetalSdnBackend.mm (16 x 4-byte scalars)");

inline uint sdnPathIndex (uint from, uint to, uint n)
{
    uint idx = from * (n - 1u);
    idx += (to > from) ? (to - 1u) : to;
    return idx;
}

kernel void sdn_process (
    constant SdnParams&   p                 [[buffer(0)]],
    const device float*   inputs            [[buffer(1)]],   // [numNodes][blockSize]
    device float*         outputs           [[buffer(2)]],   // [numNodes][blockSize]
    device float*         delayLines        [[buffer(3)]],   // [numPaths][maxDelaySamples]
    const device int*     delayLength       [[buffer(4)]],   // [numPaths]
    const device int*     targetDelayLength [[buffer(5)]],   // [numPaths]
    const device float*   crossfadeMix      [[buffer(6)]],   // [numPaths]
    const device float*   gainLow           [[buffer(7)]],   // [numPaths]
    const device float*   gainMid           [[buffer(8)]],   // [numPaths]
    const device float*   gainHigh          [[buffer(9)]],   // [numPaths]
    device float*         decayLowState     [[buffer(10)]],  // [numPaths]
    device float*         decayHighState    [[buffer(11)]],  // [numPaths]
    const device int*     diffuserDelays    [[buffer(12)]],  // [numNodes*2]
    device float*         diffRings         [[buffer(13)]],  // [numNodes*2*maxDiffLen]
    device int*           diffWritePos      [[buffer(14)]],  // [numNodes*2]
    device float*         toneState         [[buffer(15)]],  // [numNodes]
    device float*         dcState           [[buffer(16)]],  // [numNodes*2] (x1, y1)
    uint nodeId [[thread_position_in_threadgroup]])
{
    const uint N = p.numNodes;
    if (nodeId >= N)
        return;
    const uint n = nodeId;

    const uint   MAXD = p.maxDelaySamples;
    const int    iMAXD = (int) MAXD;
    const float  invN = p.inputDistribution;
    const float  twoOverNm1 = 2.0f / (float) (N - 1u);
    const float  c = p.diffusionCoeff;
    const bool   doDiffuse = (c > 0.0001f);

    // Per-node state -> registers.
    float toneSt = toneState[n];
    float dcX1   = dcState[n * 2u + 0u];
    float dcY1   = dcState[n * 2u + 1u];
    int   dfWp0  = diffWritePos[n * NUM_DIFFUSERS + 0u];
    int   dfWp1  = diffWritePos[n * NUM_DIFFUSERS + 1u];
    const int diffLen0 = diffuserDelays[n * NUM_DIFFUSERS + 0u];
    const int diffLen1 = diffuserDelays[n * NUM_DIFFUSERS + 1u];
    device float* dr0 = diffRings + (ulong) (n * NUM_DIFFUSERS + 0u) * p.maxDiffLen;
    device float* dr1 = diffRings + (ulong) (n * NUM_DIFFUSERS + 1u) * p.maxDiffLen;

    for (uint sc = 0; sc < p.chunkSamples; ++sc)
    {
        const uint s = p.sampleOffset + sc;   // block-relative (see header note)
        const int base = (int) (p.ringWritePos + s);

        // 1. read incoming from every path i->n; Householder sum.
        float incoming[32];  // per-node scratch; size >= MAX_NODES (reverb maxReverbChannels)
        float sumIn = 0.0f;
        uint  k = 0u;
        for (uint i = 0u; i < N; ++i)
        {
            if (i == n) continue;
            const uint pidx = sdnPathIndex (i, n, N);
            const float mix = crossfadeMix[pidx];
            float val;
            if (mix >= 1.0f)
            {
                int rp = (base - delayLength[pidx]) % iMAXD;
                if (rp < 0) rp += iMAXD;
                val = delayLines[(ulong) pidx * MAXD + (uint) rp];
            }
            else
            {
                int orp = (base - delayLength[pidx]) % iMAXD;
                if (orp < 0) orp += iMAXD;
                int nrp = (base - targetDelayLength[pidx]) % iMAXD;
                if (nrp < 0) nrp += iMAXD;
                const float oldS = delayLines[(ulong) pidx * MAXD + (uint) orp];
                const float newS = delayLines[(ulong) pidx * MAXD + (uint) nrp];
                const float m = fmin (1.0f, mix + p.crossfadeRate * (float) s);
                val = oldS * (1.0f - m) + newS * m;
            }
            incoming[k] = val;
            sumIn += val;
            ++k;
        }
        const uint inCount = k;
        const float X = twoOverNm1 * sumIn;

        // 2. output = sum of scattered (X - incoming[i]), in path order.
        float output = 0.0f;
        for (uint i = 0u; i < inCount; ++i)
            output += (X - incoming[i]);

        // 3. diffuse the node input (2 serial allpass stages, gated).
        float diffused = inputs[n * p.blockSize + s];
        if (doDiffuse)
        {
            { float delayed = dr0[(uint) dfWp0]; float v = diffused - c * delayed;
              dr0[(uint) dfWp0] = v; if (++dfWp0 >= diffLen0) dfWp0 = 0; diffused = delayed + c * v; }
            { float delayed = dr1[(uint) dfWp1]; float v = diffused - c * delayed;
              dr1[(uint) dfWp1] = v; if (++dfWp1 >= diffLen1) dfWp1 = 0; diffused = delayed + c * v; }
        }

        // 4. write outgoing paths n->i with the 3-band decay filter.
        const uint writePos = (uint) (base % iMAXD);
        uint outIdx = 0u;
        for (uint i = 0u; i < N; ++i)
        {
            if (i == n) continue;
            const uint pidx = sdnPathIndex (n, i, N);
            const float signal = (X - incoming[outIdx]) + diffused * invN;

            float lowS  = decayLowState[pidx];
            float highS = decayHighState[pidx];
            lowS  += p.lowCoeff  * (signal - lowS);
            highS += p.highCoeff * (signal - highS);
            const float low  = lowS;
            const float high = signal - highS;
            const float mid  = highS - lowS;
            const float outSig = low * gainLow[pidx] + mid * gainMid[pidx] + high * gainHigh[pidx];
            decayLowState[pidx]  = lowS;
            decayHighState[pidx] = highS;

            delayLines[(ulong) pidx * MAXD + writePos] = outSig;
            ++outIdx;
        }

        // 5. tone LPF -> DC blocker -> output gain.
        toneSt += p.toneCoeff * (output - toneSt);
        const float toned = toneSt;
        const float dcOut = toned - dcX1 + p.dcPole * dcY1;
        dcX1 = toned;
        dcY1 = dcOut;
        outputs[n * p.blockSize + s] = dcOut * p.sdnOutputGain;

        threadgroup_barrier (mem_flags::mem_device);
    }

    // Persist per-node state.
    toneState[n] = toneSt;
    dcState[n * 2u + 0u] = dcX1;
    dcState[n * 2u + 1u] = dcY1;
    diffWritePos[n * NUM_DIFFUSERS + 0u] = dfWp0;
    diffWritePos[n * NUM_DIFFUSERS + 1u] = dfWp1;
}

/*
    Node-parallel twin. grid = numNodes threadgroups, 32 lanes each.

    BIT-EXACT vs sdn_process. The two accumulations that fix the result -- sumIn
    over the incoming vector and the Householder output sum -- keep their exact
    path order and stay on a single lane. What is parallelised is the MEMORY
    traffic on either side of them: the N-1 gather loads and the N-1 scatter
    read-modify-writes, one path per lane. Anything recursive across samples
    (diffusers, tone, DC blocker) is inherently serial and stays on lane 0, which
    publishes its result through threadgroup memory.

    Two barriers per sample, both intra-threadgroup and therefore cheap:
      A  after the cooperative gather, before lane 0 reads all of sIncoming
      B  after lane 0 publishes sX/sDiffused, before the others read them
    mem_threadgroup is the right scope for BOTH: they only publish threadgroup
    data. The gather's device READS need no fence (a read is not a side effect
    anyone must observe), and the scatter's device WRITES are read by other
    threadgroups only in a LATER chunk, which the dispatch boundary orders.
    No third barrier is needed to protect sIncoming from the next iteration:
    lane 0's full-vector read happens before B, and every other lane rewrites
    only its own slot after B.

    The barriers sit OUTSIDE the `t < inCount` guards on purpose -- at N = 32,
    lane 31 does no work but must still reach A and B or the threadgroup hangs.
*/
kernel void sdn_process_nodes (
    constant SdnParams&   p                 [[buffer(0)]],
    const device float*   inputs            [[buffer(1)]],
    device float*         outputs           [[buffer(2)]],
    device float*         delayLines        [[buffer(3)]],
    const device int*     delayLength       [[buffer(4)]],
    const device int*     targetDelayLength [[buffer(5)]],
    const device float*   crossfadeMix      [[buffer(6)]],
    const device float*   gainLow           [[buffer(7)]],
    const device float*   gainMid           [[buffer(8)]],
    const device float*   gainHigh          [[buffer(9)]],
    device float*         decayLowState     [[buffer(10)]],
    device float*         decayHighState    [[buffer(11)]],
    const device int*     diffuserDelays    [[buffer(12)]],
    device float*         diffRings         [[buffer(13)]],
    device int*           diffWritePos      [[buffer(14)]],
    device float*         toneState         [[buffer(15)]],
    device float*         dcState           [[buffer(16)]],
    uint n [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]])
{
    const uint N = p.numNodes;
    if (n >= N)
        return;                       // threadgroup-uniform: whole group leaves, barrier-safe
    const uint inCount = N - 1u;

    // Threadgroup, not a per-lane array: the lockstep's float incoming[32] is
    // indexed by a runtime-bounded loop, so it cannot live in registers.
    threadgroup float sIncoming[32];
    threadgroup float sX;
    threadgroup float sDiffused;

    const uint   MAXD = p.maxDelaySamples;
    const int    iMAXD = (int) MAXD;
    const float  invN = p.inputDistribution;
    const float  twoOverNm1 = 2.0f / (float) (N - 1u);
    const float  c = p.diffusionCoeff;
    const bool   doDiffuse = (c > 0.0001f);

    // Slot t of the incoming vector is source node (t < n ? t : t + 1), and the
    // scatter walks the destinations in that same order, so one index serves
    // both -- exactly the k / outIdx correspondence of the lockstep.
    const uint peer = (t < n) ? t : t + 1u;

    float toneSt = toneState[n];
    float dcX1   = dcState[n * 2u + 0u];
    float dcY1   = dcState[n * 2u + 1u];
    int   dfWp0  = diffWritePos[n * NUM_DIFFUSERS + 0u];
    int   dfWp1  = diffWritePos[n * NUM_DIFFUSERS + 1u];
    const int diffLen0 = diffuserDelays[n * NUM_DIFFUSERS + 0u];
    const int diffLen1 = diffuserDelays[n * NUM_DIFFUSERS + 1u];
    device float* dr0 = diffRings + (ulong) (n * NUM_DIFFUSERS + 0u) * p.maxDiffLen;
    device float* dr1 = diffRings + (ulong) (n * NUM_DIFFUSERS + 1u) * p.maxDiffLen;

    for (uint sc = 0; sc < p.chunkSamples; ++sc)
    {
        const uint s = p.sampleOffset + sc;   // block-relative (see header note)
        const int base = (int) (p.ringWritePos + s);

        // -- gather: one incoming path per lane -----------------------------
        if (t < inCount)
        {
            const uint pidx = sdnPathIndex (peer, n, N);
            const float mix = crossfadeMix[pidx];
            float val;
            if (mix >= 1.0f)
            {
                int rp = (base - delayLength[pidx]) % iMAXD;
                if (rp < 0) rp += iMAXD;
                val = delayLines[(ulong) pidx * MAXD + (uint) rp];
            }
            else
            {
                int orp = (base - delayLength[pidx]) % iMAXD;
                if (orp < 0) orp += iMAXD;
                int nrp = (base - targetDelayLength[pidx]) % iMAXD;
                if (nrp < 0) nrp += iMAXD;
                const float oldS = delayLines[(ulong) pidx * MAXD + (uint) orp];
                const float newS = delayLines[(ulong) pidx * MAXD + (uint) nrp];
                const float m = fmin (1.0f, mix + p.crossfadeRate * (float) s);
                val = oldS * (1.0f - m) + newS * m;
            }
            sIncoming[t] = val;
        }
        threadgroup_barrier (mem_flags::mem_threadgroup);            // barrier A

        // -- serial core: summation order is the bit-exactness contract ------
        if (t == 0u)
        {
            float sumIn = 0.0f;
            for (uint i = 0u; i < inCount; ++i)
                sumIn += sIncoming[i];
            const float X = twoOverNm1 * sumIn;

            float output = 0.0f;
            for (uint i = 0u; i < inCount; ++i)
                output += (X - sIncoming[i]);

            float diffused = inputs[n * p.blockSize + s];
            if (doDiffuse)
            {
                { float delayed = dr0[(uint) dfWp0]; float v = diffused - c * delayed;
                  dr0[(uint) dfWp0] = v; if (++dfWp0 >= diffLen0) dfWp0 = 0; diffused = delayed + c * v; }
                { float delayed = dr1[(uint) dfWp1]; float v = diffused - c * delayed;
                  dr1[(uint) dfWp1] = v; if (++dfWp1 >= diffLen1) dfWp1 = 0; diffused = delayed + c * v; }
            }

            toneSt += p.toneCoeff * (output - toneSt);
            const float toned = toneSt;
            const float dcOut = toned - dcX1 + p.dcPole * dcY1;
            dcX1 = toned;
            dcY1 = dcOut;
            outputs[n * p.blockSize + s] = dcOut * p.sdnOutputGain;

            sX = X;
            sDiffused = diffused;
        }
        threadgroup_barrier (mem_flags::mem_threadgroup);            // barrier B

        // -- scatter: one outgoing path per lane -----------------------------
        if (t < inCount)
        {
            const uint writePos = (uint) (base % iMAXD);
            const uint pidx = sdnPathIndex (n, peer, N);
            const float signal = (sX - sIncoming[t]) + sDiffused * invN;

            float lowS  = decayLowState[pidx];
            float highS = decayHighState[pidx];
            lowS  += p.lowCoeff  * (signal - lowS);
            highS += p.highCoeff * (signal - highS);
            const float low  = lowS;
            const float high = signal - highS;
            const float mid  = highS - lowS;
            const float outSig = low * gainLow[pidx] + mid * gainMid[pidx] + high * gainHigh[pidx];
            decayLowState[pidx]  = lowS;
            decayHighState[pidx] = highS;

            delayLines[(ulong) pidx * MAXD + writePos] = outSig;
        }
    }

    if (t == 0u)
    {
        toneState[n] = toneSt;
        dcState[n * 2u + 0u] = dcX1;
        dcState[n * 2u + 1u] = dcY1;
        diffWritePos[n * NUM_DIFFUSERS + 0u] = dfWp0;
        diffWritePos[n * NUM_DIFFUSERS + 1u] = dfWp1;
    }
}
)MSL";
