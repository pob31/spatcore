#pragma once

/*
    CUDA kernel source for the native GPU SDN reverb backend (NVRTC-compiled at
    prepare() time by CudaSdnBackend, like the other native kernels — no .cu
    build step, no nvcc).

    KEEP IN SYNC with MetalSdnKernels.h — the two strings implement the
    byte-identical algorithm; only the language scaffolding differs. The full
    algorithm/threading notes live in the Metal twin.

    TWO kernels, bit-identical to each other and selected per block by the host:

    sdn_process       — the original lockstep. grid = 1, block = numNodes,
                        threadIdx.x = node, one __syncthreads() per sample (which
                        orders global memory within the block, the analogue of
                        Metal's mem_device barrier). Correct for ANY geometry,
                        including inter-node delays of a single sample, but it
                        occupies exactly one SM and its per-thread incoming[]
                        scratch is dynamically indexed, so it lands in local
                        (global-backed) memory. Retained as the fallback.

    sdn_process_nodes — grid = numNodes (one block per node, so the work spreads
                        across SMs), block = 32. Requires the host to launch it
                        in chunks of C samples where every effective delay is
                        >= C (SdnHostConfig::chunkSamplesForBlock); inside such
                        a window nodes are mutually independent, so the missing
                        cross-block barrier is not needed. A chunk boundary is a
                        kernel-launch boundary.

    Both process [sampleOffset, sampleOffset + chunkSamples) of the block. All
    sample-derived quantities stay BLOCK-relative — ringWritePos advances once
    per block, and the crossfade term is mix + crossfadeRate * s — so s must
    remain the absolute in-block index, not the in-chunk one.
*/

static const char* const kSdnProcessKernelSource = R"CUDA(

#define NUM_DIFFUSERS 2u

struct SdnParams
{
    unsigned int numNodes;
    unsigned int numPaths;
    unsigned int blockSize;
    unsigned int maxDelaySamples;
    unsigned int ringWritePos;
    unsigned int maxDiffLen;
    float diffusionCoeff;
    float toneCoeff;
    float lowCoeff;
    float highCoeff;
    float dcPole;
    float sdnOutputGain;
    float inputDistribution;
    float crossfadeRate;
    unsigned int sampleOffset;   // first in-block sample of this chunk
    unsigned int chunkSamples;   // samples in this chunk (== blockSize when unchunked)
};

__device__ inline unsigned int sdnPathIndex (unsigned int from, unsigned int to, unsigned int n)
{
    unsigned int idx = from * (n - 1u);
    idx += (to > from) ? (to - 1u) : to;
    return idx;
}

extern "C" __global__ void sdn_process (
    const SdnParams           p,
    const float* __restrict__ inputs,             // [numNodes][blockSize]
    float* __restrict__       outputs,            // [numNodes][blockSize]
    float* __restrict__       delayLines,         // [numPaths][maxDelaySamples]
    const int* __restrict__   delayLength,        // [numPaths]
    const int* __restrict__   targetDelayLength,  // [numPaths]
    const float* __restrict__ crossfadeMix,       // [numPaths]
    const float* __restrict__ gainLow,            // [numPaths]
    const float* __restrict__ gainMid,            // [numPaths]
    const float* __restrict__ gainHigh,           // [numPaths]
    float* __restrict__       decayLowState,      // [numPaths]
    float* __restrict__       decayHighState,     // [numPaths]
    const int* __restrict__   diffuserDelays,     // [numNodes*2]
    float* __restrict__       diffRings,          // [numNodes*2*maxDiffLen]
    int* __restrict__         diffWritePos,       // [numNodes*2]
    float* __restrict__       toneState,          // [numNodes]
    float* __restrict__       dcState)            // [numNodes*2]
{
    const unsigned int N = p.numNodes;
    const unsigned int nodeId = threadIdx.x;
    if (nodeId >= N)
        return;
    const unsigned int n = nodeId;

    const unsigned int MAXD = p.maxDelaySamples;
    const int   iMAXD = (int) MAXD;
    const float invN = p.inputDistribution;
    const float twoOverNm1 = 2.0f / (float) (N - 1u);
    const float c = p.diffusionCoeff;
    const bool  doDiffuse = (c > 0.0001f);

    float toneSt = toneState[n];
    float dcX1   = dcState[n * 2u + 0u];
    float dcY1   = dcState[n * 2u + 1u];
    int   dfWp0  = diffWritePos[n * NUM_DIFFUSERS + 0u];
    int   dfWp1  = diffWritePos[n * NUM_DIFFUSERS + 1u];
    const int diffLen0 = diffuserDelays[n * NUM_DIFFUSERS + 0u];
    const int diffLen1 = diffuserDelays[n * NUM_DIFFUSERS + 1u];
    float* dr0 = diffRings + (unsigned long long) (n * NUM_DIFFUSERS + 0u) * p.maxDiffLen;
    float* dr1 = diffRings + (unsigned long long) (n * NUM_DIFFUSERS + 1u) * p.maxDiffLen;

    for (unsigned int sc = 0; sc < p.chunkSamples; ++sc)
    {
        const unsigned int s = p.sampleOffset + sc;   // block-relative (see header note)
        const int base = (int) (p.ringWritePos + s);

        float incoming[32];  // per-node scratch; size >= MAX_NODES (reverb maxReverbChannels)
        float sumIn = 0.0f;
        unsigned int k = 0u;
        for (unsigned int i = 0u; i < N; ++i)
        {
            if (i == n) continue;
            const unsigned int pidx = sdnPathIndex (i, n, N);
            const float mix = crossfadeMix[pidx];
            float val;
            if (mix >= 1.0f)
            {
                int rp = (base - delayLength[pidx]) % iMAXD;
                if (rp < 0) rp += iMAXD;
                val = delayLines[(unsigned long long) pidx * MAXD + (unsigned int) rp];
            }
            else
            {
                int orp = (base - delayLength[pidx]) % iMAXD;
                if (orp < 0) orp += iMAXD;
                int nrp = (base - targetDelayLength[pidx]) % iMAXD;
                if (nrp < 0) nrp += iMAXD;
                const float oldS = delayLines[(unsigned long long) pidx * MAXD + (unsigned int) orp];
                const float newS = delayLines[(unsigned long long) pidx * MAXD + (unsigned int) nrp];
                const float m = fminf (1.0f, mix + p.crossfadeRate * (float) s);
                val = oldS * (1.0f - m) + newS * m;
            }
            incoming[k] = val;
            sumIn += val;
            ++k;
        }
        const unsigned int inCount = k;
        const float X = twoOverNm1 * sumIn;

        float output = 0.0f;
        for (unsigned int i = 0u; i < inCount; ++i)
            output += (X - incoming[i]);

        float diffused = inputs[n * p.blockSize + s];
        if (doDiffuse)
        {
            { float delayed = dr0[(unsigned int) dfWp0]; float v = diffused - c * delayed;
              dr0[(unsigned int) dfWp0] = v; if (++dfWp0 >= diffLen0) dfWp0 = 0; diffused = delayed + c * v; }
            { float delayed = dr1[(unsigned int) dfWp1]; float v = diffused - c * delayed;
              dr1[(unsigned int) dfWp1] = v; if (++dfWp1 >= diffLen1) dfWp1 = 0; diffused = delayed + c * v; }
        }

        const unsigned int writePos = (unsigned int) (base % iMAXD);
        unsigned int outIdx = 0u;
        for (unsigned int i = 0u; i < N; ++i)
        {
            if (i == n) continue;
            const unsigned int pidx = sdnPathIndex (n, i, N);
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

            delayLines[(unsigned long long) pidx * MAXD + writePos] = outSig;
            ++outIdx;
        }

        toneSt += p.toneCoeff * (output - toneSt);
        const float toned = toneSt;
        const float dcOut = toned - dcX1 + p.dcPole * dcY1;
        dcX1 = toned;
        dcY1 = dcOut;
        outputs[n * p.blockSize + s] = dcOut * p.sdnOutputGain;

        __syncthreads();
    }

    toneState[n] = toneSt;
    dcState[n * 2u + 0u] = dcX1;
    dcState[n * 2u + 1u] = dcY1;
    diffWritePos[n * NUM_DIFFUSERS + 0u] = dfWp0;
    diffWritePos[n * NUM_DIFFUSERS + 1u] = dfWp1;
}

/*
    Node-parallel twin. grid = numNodes, block = 32 (one warp per node).

    BIT-EXACT vs sdn_process. The two accumulations that fix the result --
    sumIn over the incoming vector and the Householder output sum -- keep their
    exact path order and stay on a single thread. What is parallelised is the
    MEMORY traffic on either side of them: the N-1 gather loads and the N-1
    scatter read-modify-writes, one path per thread. Anything recursive across
    samples (diffusers, tone, DC blocker) is inherently serial and stays on
    thread 0, which publishes its result through shared memory.

    Two barriers per sample, both intra-block and therefore cheap:
      A  after the cooperative gather, before thread 0 reads all of sIncoming
      B  after thread 0 publishes sX/sDiffused, before the others read them
    No third barrier is needed to protect sIncoming from the next iteration:
    past B, thread t touches only sIncoming[t] -- its own slot, which only it
    rewrites -- and thread 0's full-vector read is fenced by the next A.
*/
extern "C" __global__ void sdn_process_nodes (
    const SdnParams           p,
    const float* __restrict__ inputs,
    float* __restrict__       outputs,
    float* __restrict__       delayLines,
    const int* __restrict__   delayLength,
    const int* __restrict__   targetDelayLength,
    const float* __restrict__ crossfadeMix,
    const float* __restrict__ gainLow,
    const float* __restrict__ gainMid,
    const float* __restrict__ gainHigh,
    float* __restrict__       decayLowState,
    float* __restrict__       decayHighState,
    const int* __restrict__   diffuserDelays,
    float* __restrict__       diffRings,
    int* __restrict__         diffWritePos,
    float* __restrict__       toneState,
    float* __restrict__       dcState)
{
    const unsigned int N = p.numNodes;
    const unsigned int n = blockIdx.x;
    if (n >= N)
        return;                       // whole block leaves together: barrier-safe
    const unsigned int t = threadIdx.x;
    const unsigned int inCount = N - 1u;

    // Shared, not a per-thread array: the lockstep's float incoming[32] is
    // indexed by a runtime-bounded loop, so it cannot live in registers and the
    // compiler spills it to local (global-backed) memory -- N-1 stores plus
    // 2*(N-1) loads per node per sample, on top of the gather itself.
    __shared__ float sIncoming[32];
    __shared__ float sX;
    __shared__ float sDiffused;

    const unsigned int MAXD = p.maxDelaySamples;
    const int   iMAXD = (int) MAXD;
    const float invN = p.inputDistribution;
    const float twoOverNm1 = 2.0f / (float) (N - 1u);
    const float c = p.diffusionCoeff;
    const bool  doDiffuse = (c > 0.0001f);

    // Slot t of the incoming vector is source node (t < n ? t : t + 1), and the
    // scatter walks the destinations in that same order, so one index serves
    // both -- exactly the k / outIdx correspondence of the lockstep.
    const unsigned int peer = (t < n) ? t : t + 1u;

    float toneSt = toneState[n];
    float dcX1   = dcState[n * 2u + 0u];
    float dcY1   = dcState[n * 2u + 1u];
    int   dfWp0  = diffWritePos[n * NUM_DIFFUSERS + 0u];
    int   dfWp1  = diffWritePos[n * NUM_DIFFUSERS + 1u];
    const int diffLen0 = diffuserDelays[n * NUM_DIFFUSERS + 0u];
    const int diffLen1 = diffuserDelays[n * NUM_DIFFUSERS + 1u];
    float* dr0 = diffRings + (unsigned long long) (n * NUM_DIFFUSERS + 0u) * p.maxDiffLen;
    float* dr1 = diffRings + (unsigned long long) (n * NUM_DIFFUSERS + 1u) * p.maxDiffLen;

    for (unsigned int sc = 0; sc < p.chunkSamples; ++sc)
    {
        const unsigned int s = p.sampleOffset + sc;   // block-relative (see header note)
        const int base = (int) (p.ringWritePos + s);

        // -- gather: one incoming path per thread ---------------------------
        if (t < inCount)
        {
            const unsigned int pidx = sdnPathIndex (peer, n, N);
            const float mix = crossfadeMix[pidx];
            float val;
            if (mix >= 1.0f)
            {
                int rp = (base - delayLength[pidx]) % iMAXD;
                if (rp < 0) rp += iMAXD;
                val = delayLines[(unsigned long long) pidx * MAXD + (unsigned int) rp];
            }
            else
            {
                int orp = (base - delayLength[pidx]) % iMAXD;
                if (orp < 0) orp += iMAXD;
                int nrp = (base - targetDelayLength[pidx]) % iMAXD;
                if (nrp < 0) nrp += iMAXD;
                const float oldS = delayLines[(unsigned long long) pidx * MAXD + (unsigned int) orp];
                const float newS = delayLines[(unsigned long long) pidx * MAXD + (unsigned int) nrp];
                const float m = fminf (1.0f, mix + p.crossfadeRate * (float) s);
                val = oldS * (1.0f - m) + newS * m;
            }
            sIncoming[t] = val;
        }
        __syncthreads();                                            // barrier A

        // -- serial core: summation order is the bit-exactness contract ------
        if (t == 0u)
        {
            float sumIn = 0.0f;
            for (unsigned int i = 0u; i < inCount; ++i)
                sumIn += sIncoming[i];
            const float X = twoOverNm1 * sumIn;

            float output = 0.0f;
            for (unsigned int i = 0u; i < inCount; ++i)
                output += (X - sIncoming[i]);

            float diffused = inputs[n * p.blockSize + s];
            if (doDiffuse)
            {
                { float delayed = dr0[(unsigned int) dfWp0]; float v = diffused - c * delayed;
                  dr0[(unsigned int) dfWp0] = v; if (++dfWp0 >= diffLen0) dfWp0 = 0; diffused = delayed + c * v; }
                { float delayed = dr1[(unsigned int) dfWp1]; float v = diffused - c * delayed;
                  dr1[(unsigned int) dfWp1] = v; if (++dfWp1 >= diffLen1) dfWp1 = 0; diffused = delayed + c * v; }
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
        __syncthreads();                                            // barrier B

        // -- scatter: one outgoing path per thread ---------------------------
        if (t < inCount)
        {
            const unsigned int writePos = (unsigned int) (base % iMAXD);
            const unsigned int pidx = sdnPathIndex (n, peer, N);
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

            delayLines[(unsigned long long) pidx * MAXD + writePos] = outSig;
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
)CUDA";
