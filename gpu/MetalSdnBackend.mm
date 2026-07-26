/*
    MetalSdnBackend implementation (Objective-C++).

    The MSL kernel source lives in MetalSdnKernels.h as a string literal,
    compiled at prepare() into TWO pipeline states. The shared host math
    (geometry->delays, decay gains, crossfade state) lives in SdnHostConfig;
    this file owns the Metal buffers and the dispatch.

    All buffers use shared storage (Apple Silicon unified memory). Persistent
    state (per-path delay lines, decay-filter states, diffuser rings + write
    positions, tone/DC state) lives in device buffers across launches; reset
    zeroes them and the shared ring write head. The per-path delays/gains and the
    crossfade mix are uploaded whenever a param/geometry change or an in-flight
    crossfade makes them stale.

    Dispatch — two mappings, chosen per block (mirrors CudaSdnBackend.cpp):

      sdn_process (lockstep)  1 threadgroup x numNodes threads, per-sample
                              mem_device barrier. Correct for ANY geometry but
                              occupies exactly one GPU core. The fallback.

      sdn_process_nodes       numNodes threadgroups x 32 lanes, so the work
                              spreads across cores. Valid only within a chunk of
                              C samples where every effective delay is >= C
                              (SdnHostConfig::chunkSamplesForBlock); the host
                              launches ceil(blockSize/C) dispatches per block.

    Both mappings are bit-identical, so falling back costs only the core spread.
    WFS_SDN_NODE_PARALLEL=0 forces the lockstep; WFS_SDN_TRACE=1 logs mapping
    changes. Both are read in prepare() — never getenv on the pump thread.
*/

#include "MetalSdnBackend.h"

#if WFS_GPU_NATIVE

#include "MetalSdnKernels.h"
#include "SdnHostConfig.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace spatcore::gpu {

namespace
{
// Host mirror of the kernel-side SdnParams - layout must match exactly.
// KEEP IN SYNC with SdnParams in MetalSdnKernels.h (which static_asserts 64 too):
// this travels through setBytes, so a mismatch is silent garbage, not a compile
// error. HIP shipped exactly that bug.
struct SdnParamsGpu
{
    uint32_t numNodes;
    uint32_t numPaths;
    uint32_t blockSize;
    uint32_t maxDelaySamples;
    uint32_t ringWritePos;
    uint32_t maxDiffLen;
    float    diffusionCoeff;
    float    toneCoeff;
    float    lowCoeff;
    float    highCoeff;
    float    dcPole;
    float    sdnOutputGain;
    float    inputDistribution;
    float    crossfadeRate;
    uint32_t sampleOffset;   // first in-block sample of this chunk
    uint32_t chunkSamples;   // samples in this chunk (== blockSize when unchunked)
};
static_assert (sizeof (SdnParamsGpu) == 64, "SdnParamsGpu/SdnParams layout drift");

// Above this many chunks the per-dispatch overhead outweighs the core spread,
// and the lockstep (one dispatch, any geometry) is the better mapping.
constexpr int kMaxSdnChunksPerBlock = 8;
} // namespace

struct MetalSdnBackend::Impl
{
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pso = nil;        // sdn_process       (lockstep)
    id<MTLComputePipelineState> psoNodes = nil;   // sdn_process_nodes (grid = numNodes)

    // Params live on the HOST and go out via setBytes (by copy) per dispatch --
    // NOT in an MTLBuffer. A shared buffer mutated between dispatchThreadgroups
    // calls would hand every chunk the LAST written sampleOffset/chunkSamples,
    // because encode time != execute time: the CPU finishes all its writes
    // before commit, and the GPU reads the buffer afterwards.
    SdnParamsGpu params {};

    id<MTLBuffer> bInputs = nil, bOutputs = nil;
    id<MTLBuffer> bDelayLines = nil;
    id<MTLBuffer> bDelayLength = nil, bTargetDelayLength = nil, bCrossfadeMix = nil;
    id<MTLBuffer> bGainLow = nil, bGainMid = nil, bGainHigh = nil;
    id<MTLBuffer> bDecayLowState = nil, bDecayHighState = nil;
    id<MTLBuffer> bDiffuserDelays = nil, bDiffRings = nil, bDiffWritePos = nil;
    id<MTLBuffer> bToneState = nil, bDcState = nil;

    int numNodes = 0, numPaths = 0, blockSize = 0;
    double sampleRate = 0.0;
    uint32_t ringWritePos = 0;
    bool needUpload = true;   // re-upload the dynamic per-path buffers next launch

    bool nodeParallel = true;   // WFS_SDN_NODE_PARALLEL=0 forces the lockstep
    bool trace = false;         // WFS_SDN_TRACE=1 logs each mapping change
    bool tracedNodes = false;
    int  tracedChunk = -1;

    SdnHostConfig cfg;        // pump-thread owned after prepare

    // Parameter staging (engine thread -> pump thread). The members always hold
    // the latest set values; paramsDirty just flags a change to consume.
    std::mutex paramMutex;
    std::atomic<bool> paramsDirty { false };
    float pRt60 = 1.5f, pLowMult = 1.3f, pHighMult = 0.5f;
    float pXLow = 200.0f, pXHigh = 4000.0f, pDiffusion = 0.5f, pSdnScale = 1.0f;

    // Geometry staging (engine thread -> pump thread).
    std::mutex geomMutex;
    std::atomic<bool> geometryDirty { false };
    std::vector<float> stagedXyz;     // count * 3
    int stagedCount = 0;
    std::vector<SdnHostConfig::NodePos> posScratch;

    std::atomic<bool> resetRequested { false };
};

// deviceIndex accepted for API uniformity but ignored on macOS (system default device).
MetalSdnBackend::MetalSdnBackend (int deviceIndex) : impl (std::make_unique<Impl>()) { (void) deviceIndex; }
MetalSdnBackend::~MetalSdnBackend() { release(); }

bool MetalSdnBackend::prepare (int numNodes, int blockSize, double sampleRate)
{
    release();
    auto& m = *impl;

    m.numNodes = std::min (std::max (1, numNodes), SdnHostConfig::MAX_NODES);
    m.blockSize = std::max (1, blockSize);
    m.sampleRate = sampleRate;
    m.cfg.prepare (m.numNodes, sampleRate);
    m.numPaths = m.cfg.numPaths;
    m.ringWritePos = 0;
    m.needUpload = true;
    m.nodeParallel = true;    // re-decided below; release() must not leave it latched off
    m.tracedNodes = false;
    m.tracedChunk = -1;

    @autoreleasepool
    {
        m.device = MTLCreateSystemDefaultDevice();
        if (m.device == nil)
        {
            lastError = "No Metal device available";
            return false;
        }
        deviceName = std::string (m.device.name.UTF8String) + " (Metal)";

        NSError* err = nil;

        // SAFE math, deliberately - do not drop this to speed the kernel up.
        //
        // Metal defaults fast math to ON, and that is not a wash here: it lets the
        // compiler reassociate the two summation loops in sdn_process_nodes (clean
        // reductions over a threadgroup array) differently from the lockstep's
        // gather-interleaved accumulation, so the two mappings stop agreeing
        // bit-for-bit. Measured on an M4 Pro: with fast math they diverge (~1e-7,
        // the same order as the CPU/GPU delta); with safe math they are
        // bit-identical, verified by Experiments/metal-sdn-test.
        //
        // Bit-exactness across the pair matters because the host switches mapping
        // MID-SESSION (a session opens pre-geometry on the lockstep and moves to
        // node-parallel once geometry arrives), and because CUDA/HIP are bit-exact
        // across the same pair for free - NVRTC does not reassociate without
        // --use_fast_math. Cost here is ~2% on the node-parallel path
        // (1.254 -> 1.277 ms at 32 nodes x 256, against a 5.33 ms budget).
        MTLCompileOptions* copts = [[MTLCompileOptions alloc] init];
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if (@available (macOS 15.0, *))
            copts.mathMode = MTLMathModeSafe;
        else
#endif
        {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            copts.fastMathEnabled = NO;      // pre-15 spelling of the same thing
#pragma clang diagnostic pop
        }

        id<MTLLibrary> lib = [m.device newLibraryWithSource:
                                  [NSString stringWithUTF8String: kSdnProcessKernelSource]
                                                    options: copts
                                                      error: &err];
#if ! __has_feature(objc_arc)
        [copts release];   // the app builds these .mm without ARC; the harness with it
#endif
        if (lib == nil)
        {
            lastError = std::string ("SDN kernel compile failed: ")
                        + (err ? err.localizedDescription.UTF8String : "?");
            return false;
        }

        m.pso = [m.device newComputePipelineStateWithFunction:
                     [lib newFunctionWithName: @"sdn_process"]
                                                        error: &err];
        if (m.pso == nil)
        {
            lastError = std::string ("Pipeline state failed (sdn_process): ")
                        + (err ? err.localizedDescription.UTF8String : "?");
            return false;
        }

        if (m.pso.maxTotalThreadsPerThreadgroup < (NSUInteger) m.numNodes)
        {
            lastError = "Threadgroup too small for the SDN node count";
            release();
            return false;
        }

        // Node-parallel twin. A failure here is NOT fatal: the lockstep is
        // bit-identical and correct for any geometry, so fall back rather than
        // refuse to prepare. maxTotalThreadsPerThreadgroup is a PER-PIPELINE
        // property (a register-heavy kernel can sit below the device max), so it
        // must be checked on this pipeline, not on m.pso -- exceeding it at
        // dispatch throws on the pump thread, i.e. a hard crash in audio.
        id<MTLFunction> fnNodes = [lib newFunctionWithName: @"sdn_process_nodes"];
        if (fnNodes != nil)
            m.psoNodes = [m.device newComputePipelineStateWithFunction: fnNodes error: &err];

        if (m.psoNodes == nil || m.psoNodes.maxTotalThreadsPerThreadgroup < 32)
            m.nodeParallel = false;

        m.queue = [m.device newCommandQueue];

        const int N = m.numNodes;
        const int P = std::max (1, m.numPaths);
        const int D = SdnHostConfig::NUM_DIFFUSERS;
        const size_t maxDelay = (size_t) SdnHostConfig::MAX_DELAY_SAMPLES;
        const size_t maxDiff  = (size_t) m.cfg.maxDiffLen;

        auto shared = MTLResourceStorageModeShared;
        auto mkF = [&](size_t floats) { return [m.device newBufferWithLength: floats * sizeof (float) options: shared]; };
        auto mkI = [&](size_t ints)   { return [m.device newBufferWithLength: ints  * sizeof (int)   options: shared]; };

        m.bInputs  = mkF ((size_t) N * m.blockSize);
        m.bOutputs = mkF ((size_t) N * m.blockSize);

        m.bDelayLines        = mkF ((size_t) P * maxDelay);
        m.bDelayLength       = mkI ((size_t) P);
        m.bTargetDelayLength = mkI ((size_t) P);
        m.bCrossfadeMix      = mkF ((size_t) P);
        m.bGainLow  = mkF ((size_t) P);
        m.bGainMid  = mkF ((size_t) P);
        m.bGainHigh = mkF ((size_t) P);
        m.bDecayLowState  = mkF ((size_t) P);
        m.bDecayHighState = mkF ((size_t) P);

        m.bDiffuserDelays = mkI ((size_t) N * D);
        m.bDiffRings      = mkF ((size_t) N * D * maxDiff);
        m.bDiffWritePos   = mkI ((size_t) N * D);
        m.bToneState = mkF ((size_t) N);
        m.bDcState   = mkF ((size_t) N * 2);

        if (! (m.bInputs && m.bOutputs && m.bDelayLines && m.bDelayLength
               && m.bTargetDelayLength && m.bCrossfadeMix && m.bGainLow && m.bGainMid
               && m.bGainHigh && m.bDecayLowState && m.bDecayHighState && m.bDiffuserDelays
               && m.bDiffRings && m.bDiffWritePos && m.bToneState && m.bDcState))
        {
            lastError = "Metal buffer allocation failed";
            release();
            return false;
        }

        // Upload the static config + initial dynamic config.
        memcpy (m.bDiffuserDelays.contents, m.cfg.diffuserDelays.data(), (size_t) N * D * sizeof (int));
        memcpy (m.bDelayLength.contents,       m.cfg.delayLength.data(),       (size_t) P * sizeof (int));
        memcpy (m.bTargetDelayLength.contents, m.cfg.targetDelayLength.data(), (size_t) P * sizeof (int));
        memcpy (m.bCrossfadeMix.contents,      m.cfg.crossfadeMix.data(),      (size_t) P * sizeof (float));
        memcpy (m.bGainLow.contents,  m.cfg.gainLow.data(),  (size_t) P * sizeof (float));
        memcpy (m.bGainMid.contents,  m.cfg.gainMid.data(),  (size_t) P * sizeof (float));
        memcpy (m.bGainHigh.contents, m.cfg.gainHigh.data(), (size_t) P * sizeof (float));

        // Zero the persistent state.
        memset (m.bDelayLines.contents,    0, m.bDelayLines.length);
        memset (m.bDecayLowState.contents,  0, m.bDecayLowState.length);
        memset (m.bDecayHighState.contents, 0, m.bDecayHighState.length);
        memset (m.bDiffRings.contents,    0, m.bDiffRings.length);
        memset (m.bDiffWritePos.contents, 0, m.bDiffWritePos.length);
        memset (m.bToneState.contents, 0, m.bToneState.length);
        memset (m.bDcState.contents,   0, m.bDcState.length);

        m.params = SdnParamsGpu { (uint32_t) N, (uint32_t) m.numPaths, (uint32_t) m.blockSize,
                                  (uint32_t) SdnHostConfig::MAX_DELAY_SAMPLES, 0u,
                                  (uint32_t) m.cfg.maxDiffLen,
                                  m.cfg.diffusionCoeff, m.cfg.toneCoeff, m.cfg.lowCoeff,
                                  m.cfg.highCoeff, SdnHostConfig::DC_POLE, m.cfg.sdnOutputGain,
                                  m.cfg.inputDistribution, m.cfg.crossfadeRate,
                                  0u, (uint32_t) m.blockSize };

        // Warm up BOTH pipelines, then restore pristine state. The first dispatch
        // of a pipeline pays driver setup + page faulting, and that is exactly the
        // burst of underruns seen right after an engine (re)start. Both matter,
        // not just the one that runs first: a session opens pre-geometry with
        // every delay at 1 (so chunkSamplesForBlock collapses to 1 and the
        // lockstep is chosen) and only switches to the node-parallel mapping once
        // real geometry arrives -- which would otherwise pay a SECOND first-launch
        // stall mid-session, with audio running.
        //
        // Warming the node-parallel kernel here races on the delay lines (the
        // pre-geometry delays are far below any safe chunk window), which is
        // harmless and deliberate: every value it touches is zeroed immediately
        // below, and the point is to fault in the code path, not to compute.
        if (N >= 2)
        {
            SdnParamsGpu warm = m.params;
            warm.sampleOffset = 0u;
            warm.chunkSamples = 1u;

            id<MTLCommandBuffer> cb = [m.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

            [enc setComputePipelineState: m.pso];
            [enc setBytes: &warm length: sizeof (warm) atIndex: 0];
            [enc setBuffer: m.bInputs            offset: 0 atIndex: 1];
            [enc setBuffer: m.bOutputs           offset: 0 atIndex: 2];
            [enc setBuffer: m.bDelayLines        offset: 0 atIndex: 3];
            [enc setBuffer: m.bDelayLength       offset: 0 atIndex: 4];
            [enc setBuffer: m.bTargetDelayLength offset: 0 atIndex: 5];
            [enc setBuffer: m.bCrossfadeMix      offset: 0 atIndex: 6];
            [enc setBuffer: m.bGainLow           offset: 0 atIndex: 7];
            [enc setBuffer: m.bGainMid           offset: 0 atIndex: 8];
            [enc setBuffer: m.bGainHigh          offset: 0 atIndex: 9];
            [enc setBuffer: m.bDecayLowState     offset: 0 atIndex: 10];
            [enc setBuffer: m.bDecayHighState    offset: 0 atIndex: 11];
            [enc setBuffer: m.bDiffuserDelays    offset: 0 atIndex: 12];
            [enc setBuffer: m.bDiffRings         offset: 0 atIndex: 13];
            [enc setBuffer: m.bDiffWritePos      offset: 0 atIndex: 14];
            [enc setBuffer: m.bToneState         offset: 0 atIndex: 15];
            [enc setBuffer: m.bDcState           offset: 0 atIndex: 16];
            [enc dispatchThreadgroups: MTLSizeMake (1, 1, 1)
                  threadsPerThreadgroup: MTLSizeMake ((NSUInteger) N, 1, 1)];

            if (m.psoNodes != nil)   // bindings persist across setComputePipelineState
            {
                [enc setComputePipelineState: m.psoNodes];
                [enc setBytes: &warm length: sizeof (warm) atIndex: 0];
                [enc dispatchThreadgroups: MTLSizeMake ((NSUInteger) N, 1, 1)
                      threadsPerThreadgroup: MTLSizeMake (32, 1, 1)];
            }

            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            // Restore pristine first-launch state, so the first audible block is
            // bit-identical to a prepare() that never warmed up.
            memset (m.bDelayLines.contents,     0, m.bDelayLines.length);
            memset (m.bDecayLowState.contents,  0, m.bDecayLowState.length);
            memset (m.bDecayHighState.contents, 0, m.bDecayHighState.length);
            memset (m.bDiffRings.contents,      0, m.bDiffRings.length);
            memset (m.bDiffWritePos.contents,   0, m.bDiffWritePos.length);
            memset (m.bToneState.contents,      0, m.bToneState.length);
            memset (m.bDcState.contents,        0, m.bDcState.length);
            memset (m.bOutputs.contents,        0, m.bOutputs.length);
            m.ringWritePos = 0;
        }
    }

    // Escape hatch + trace. Read HERE, never in processBlock: that runs on the
    // reverb pump thread, where getenv is not something to call per block.
    if (const char* e = std::getenv ("WFS_SDN_NODE_PARALLEL"))
        m.nodeParallel = m.nodeParallel && (std::string (e) != "0");
    m.trace = (std::getenv ("WFS_SDN_TRACE") != nullptr);

    ready = true;
    lastError.clear();
    return true;
}

void MetalSdnBackend::setGeometry (const float* xyz, int count) noexcept
{
    auto& m = *impl;
    std::lock_guard<std::mutex> lock (m.geomMutex);
    m.stagedXyz.assign (xyz, xyz + (size_t) std::max (0, count) * 3);
    m.stagedCount = std::max (0, count);
    m.geometryDirty.store (true, std::memory_order_release);
}

void MetalSdnBackend::setParameters (float rt60, float rt60LowMult, float rt60HighMult,
                                     float crossoverLow, float crossoverHigh,
                                     float diffusion, float sdnScale) noexcept
{
    auto& m = *impl;
    std::lock_guard<std::mutex> lock (m.paramMutex);
    m.pRt60 = rt60; m.pLowMult = rt60LowMult; m.pHighMult = rt60HighMult;
    m.pXLow = crossoverLow; m.pXHigh = crossoverHigh; m.pDiffusion = diffusion;
    m.pSdnScale = sdnScale;
    m.paramsDirty.store (true, std::memory_order_release);
}

void MetalSdnBackend::requestReset() noexcept
{
    impl->resetRequested.store (true, std::memory_order_release);
}

bool MetalSdnBackend::processBlock (const float* const* inputs, float* const* outputs)
{
    if (! ready)
        return false;

    auto& m = *impl;
    const int N = m.numNodes;

    // With 0-1 nodes the SDN cannot scatter — pass through on the host, exactly
    // like SDNAlgorithm::processBlock (N==1 copies in->out, N==0 is silence).
    if (N < 2)
    {
        for (int n = 0; n < N; ++n)
            if (outputs[n] != nullptr)
            {
                if (inputs[n] != nullptr)
                    memcpy (outputs[n], inputs[n], (size_t) m.blockSize * sizeof (float));
                else
                    memset (outputs[n], 0, (size_t) m.blockSize * sizeof (float));
            }
        return true;
    }

    const int P = std::max (1, m.numPaths);

    @autoreleasepool
    {
        const auto t0 = std::chrono::steady_clock::now();

        if (m.resetRequested.exchange (false, std::memory_order_acq_rel))
        {
            memset (m.bDelayLines.contents,    0, m.bDelayLines.length);
            memset (m.bDecayLowState.contents,  0, m.bDecayLowState.length);
            memset (m.bDecayHighState.contents, 0, m.bDecayHighState.length);
            memset (m.bDiffRings.contents,    0, m.bDiffRings.length);
            memset (m.bDiffWritePos.contents, 0, m.bDiffWritePos.length);
            memset (m.bToneState.contents, 0, m.bToneState.length);
            memset (m.bDcState.contents,   0, m.bDcState.length);
            m.ringWritePos = 0;
        }

        // Consume staged geometry/params: recompute the config (CPU order —
        // delays first, then decay gains, then diffusion) and flag an upload.
        const bool geomDirty = m.geometryDirty.exchange (false, std::memory_order_acquire);
        const bool parDirty  = m.paramsDirty.exchange (false, std::memory_order_acquire);
        if (geomDirty || parDirty)
        {
            float rt60, lowMult, highMult, xLow, xHigh, diffusion, scale;
            {
                std::lock_guard<std::mutex> lock (m.paramMutex);
                rt60 = m.pRt60; lowMult = m.pLowMult; highMult = m.pHighMult;
                xLow = m.pXLow; xHigh = m.pXHigh; diffusion = m.pDiffusion;
                scale = m.pSdnScale;
            }

            if (geomDirty)
            {
                std::lock_guard<std::mutex> lock (m.geomMutex);
                m.posScratch.resize ((size_t) m.stagedCount);
                for (int i = 0; i < m.stagedCount; ++i)
                    m.posScratch[(size_t) i] = { m.stagedXyz[(size_t) (i * 3 + 0)],
                                                 m.stagedXyz[(size_t) (i * 3 + 1)],
                                                 m.stagedXyz[(size_t) (i * 3 + 2)] };
                m.cfg.recalcDelays (m.posScratch.data(), m.stagedCount, scale);
            }
            else
            {
                m.cfg.recalcDelaysFromStored (scale);
            }

            m.cfg.recalcDecay (rt60, lowMult, highMult, xLow, xHigh);
            m.cfg.setDiffusion (diffusion);
            m.needUpload = true;
        }

        // Upload the dynamic per-path buffers if stale (recompute or an in-flight
        // crossfade), plus the scalar coefficients in the param block.
        if (m.needUpload)
        {
            memcpy (m.bDelayLength.contents,       m.cfg.delayLength.data(),       (size_t) P * sizeof (int));
            memcpy (m.bTargetDelayLength.contents, m.cfg.targetDelayLength.data(), (size_t) P * sizeof (int));
            memcpy (m.bCrossfadeMix.contents,      m.cfg.crossfadeMix.data(),      (size_t) P * sizeof (float));
            memcpy (m.bGainLow.contents,  m.cfg.gainLow.data(),  (size_t) P * sizeof (float));
            memcpy (m.bGainMid.contents,  m.cfg.gainMid.data(),  (size_t) P * sizeof (float));
            memcpy (m.bGainHigh.contents, m.cfg.gainHigh.data(), (size_t) P * sizeof (float));

            m.params.diffusionCoeff = m.cfg.diffusionCoeff;
            m.params.lowCoeff = m.cfg.lowCoeff;
            m.params.highCoeff = m.cfg.highCoeff;
            m.params.sdnOutputGain = m.cfg.sdnOutputGain;
            m.params.crossfadeRate = m.cfg.crossfadeRate;
            m.needUpload = false;
        }

        // Inputs -> flat shared buffer (silence for missing channels).
        float* inFlat = (float*) m.bInputs.contents;
        for (int n = 0; n < N; ++n)
        {
            if (inputs[n] != nullptr)
                memcpy (inFlat + (size_t) n * m.blockSize, inputs[n], (size_t) m.blockSize * sizeof (float));
            else
                memset (inFlat + (size_t) n * m.blockSize, 0, (size_t) m.blockSize * sizeof (float));
        }

        // Block-start write head.
        m.params.ringWritePos = m.ringWritePos;

        // Pick the mapping for THIS block. The node-parallel kernel needs a chunk
        // window inside which no node reads a cell another node writes, and a
        // chunk boundary costs a dispatch, so it only pays while the geometry
        // keeps the window wide. A degenerate or not-yet-received geometry
        // collapses the window towards 1 sample; the lockstep handles that in one
        // dispatch and is bit-identical, so falling back costs only the spread.
        int chunk = m.blockSize;
        bool useNodes = m.nodeParallel && N >= 2 && m.psoNodes != nil;
        if (useNodes)
        {
            chunk = m.cfg.chunkSamplesForBlock (m.blockSize);
            const int chunks = (m.blockSize + chunk - 1) / chunk;
            if (chunks > kMaxSdnChunksPerBlock)
            {
                useNodes = false;
                chunk = m.blockSize;
            }
        }

        if (m.trace && (useNodes != m.tracedNodes || chunk != m.tracedChunk))
        {
            m.tracedNodes = useNodes; m.tracedChunk = chunk;
            std::fprintf (stderr, "[sdn] mapping=%s chunk=%d/%d minDelay=%d\n",
                          useNodes ? "node-parallel" : "lockstep", chunk, m.blockSize,
                          m.cfg.numPaths > 0 ? *std::min_element (m.cfg.delayLength.begin(),
                                                                  m.cfg.delayLength.begin() + m.cfg.numPaths) : -1);
        }

        id<MTLCommandBuffer> cb = [m.queue commandBuffer];
        // SERIAL encoder, and that is load-bearing: Metal orders successive
        // dispatchThreadgroups on a serial encoder and makes each one's side
        // effects visible to the next, which is exactly the launch boundary the
        // node-parallel chunk contract requires (memoryBarrierWithScope: is
        // "allowed, but ignored" on a serial encoder -- it has nothing to add).
        // NEVER switch this to computeCommandEncoderWithDispatchType:
        // MTLDispatchTypeConcurrent: chunk k+1's gather would race chunk k's
        // scatter and the SDN would go quietly, non-deterministically wrong.
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState: useNodes ? m.psoNodes : m.pso];
        [enc setBuffer: m.bInputs            offset: 0 atIndex: 1];
        [enc setBuffer: m.bOutputs           offset: 0 atIndex: 2];
        [enc setBuffer: m.bDelayLines        offset: 0 atIndex: 3];
        [enc setBuffer: m.bDelayLength       offset: 0 atIndex: 4];
        [enc setBuffer: m.bTargetDelayLength offset: 0 atIndex: 5];
        [enc setBuffer: m.bCrossfadeMix      offset: 0 atIndex: 6];
        [enc setBuffer: m.bGainLow           offset: 0 atIndex: 7];
        [enc setBuffer: m.bGainMid           offset: 0 atIndex: 8];
        [enc setBuffer: m.bGainHigh          offset: 0 atIndex: 9];
        [enc setBuffer: m.bDecayLowState     offset: 0 atIndex: 10];
        [enc setBuffer: m.bDecayHighState    offset: 0 atIndex: 11];
        [enc setBuffer: m.bDiffuserDelays    offset: 0 atIndex: 12];
        [enc setBuffer: m.bDiffRings         offset: 0 atIndex: 13];
        [enc setBuffer: m.bDiffWritePos      offset: 0 atIndex: 14];
        [enc setBuffer: m.bToneState         offset: 0 atIndex: 15];
        [enc setBuffer: m.bDcState           offset: 0 atIndex: 16];

        const MTLSize tgCount = useNodes ? MTLSizeMake ((NSUInteger) N, 1, 1)
                                         : MTLSizeMake (1, 1, 1);
        const MTLSize tgSize  = useNodes ? MTLSizeMake (32, 1, 1)
                                         : MTLSizeMake ((NSUInteger) N, 1, 1);

        for (int off = 0; off < m.blockSize; off += chunk)
        {
            m.params.sampleOffset = (uint32_t) off;
            m.params.chunkSamples = (uint32_t) std::min (chunk, m.blockSize - off);
            // BY COPY, per dispatch. A shared MTLBuffer mutated between dispatches
            // would give EVERY dispatch the last-written value (see Impl::params).
            [enc setBytes: &m.params length: sizeof (SdnParamsGpu) atIndex: 0];
            [enc dispatchThreadgroups: tgCount threadsPerThreadgroup: tgSize];
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status == MTLCommandBufferStatusError)
        {
            lastError = std::string ("GPU SDN launch failed: ")
                        + (cb.error ? cb.error.localizedDescription.UTF8String : "?");
            ready = false;
            return false;
        }

        const float* outFlat = (const float*) m.bOutputs.contents;
        for (int n = 0; n < N; ++n)
            if (outputs[n] != nullptr)
                memcpy (outputs[n], outFlat + (size_t) n * m.blockSize, (size_t) m.blockSize * sizeof (float));

        // Advance the crossfade (host-side, like the CPU post-block loop) and the
        // shared ring write head for the next launch.
        if (m.cfg.advanceCrossfades (m.blockSize))
            m.needUpload = true;
        m.ringWritePos = (m.ringWritePos + (uint32_t) m.blockSize)
                         % (uint32_t) SdnHostConfig::MAX_DELAY_SAMPLES;

        lastLaunchMs = std::chrono::duration<double, std::milli> (
                           std::chrono::steady_clock::now() - t0).count();
    }
    return true;
}

void MetalSdnBackend::release() noexcept
{
    auto& m = *impl;
    @autoreleasepool
    {
        m.bInputs = m.bOutputs = nil;
        m.bDelayLines = m.bDelayLength = m.bTargetDelayLength = m.bCrossfadeMix = nil;
        m.bGainLow = m.bGainMid = m.bGainHigh = nil;
        m.bDecayLowState = m.bDecayHighState = nil;
        m.bDiffuserDelays = m.bDiffRings = m.bDiffWritePos = nil;
        m.bToneState = m.bDcState = nil;
        m.psoNodes = nil;
        m.pso = nil;
        m.queue = nil;
        m.device = nil;
    }
    ready = false;
}

} // namespace spatcore::gpu

#endif // WFS_GPU_NATIVE
