# spatcore

The shared spatial-audio core extracted from [WFS-DIY](https://github.com/pob31/WFS-DIY):
a JUCE-based real-time engine for wave-field synthesis and related spatial
rendering, plus the control plane that drives it. Built to be consumed as a git
submodule by multiple apps (WFS-DIY, and the planned XOA and Tight-WFS).

## Layout

| Directory | Contents |
|---|---|
| `rt/` | Realtime primitives: lock-free rings, fork-join pool, RT thread priority (MMCSS/mach/SCHED_FIFO), `RtSnapshot<T>` message→RT hand-off |
| `dsp/` | Biquads, smoothers, speed limiter, tracking filter, FR diffusion model, LFO waveforms, level detectors |
| `wfs/` | WFS renderers: CPU gather/scatter processors + the GPU-pipeline renderer wrappers |
| `reverb/` | Reverb engine + FDN / SDN / IR algorithms (CPU and GPU variants), pre/post processing, feed thread |
| `gpu/` | Multi-vendor GPU compute: CUDA / HIP / Metal backends for 5 kernel families, runtime kernel compilation (NVRTC/hipRTC), async pipeline, device manager, vendor-plugin factory, JUCE-free host work pool |
| `control/` | `osc/` wire codec + transports + ingest queues, `state/` ValueTree parameter store + XML persistence, `mcp/` MCP server core (JSON-RPC transport, dispatcher, tool registry, tier enforcement) |
| `controllers/` | Hardware controller device layer: SpaceMouse, Stream Deck+, ROLI Lightpad, evdev touch |
| `tools/` | `gpu/` per-vendor plugin build scripts, `codegen/` CSV→MCP tool generator core |
| `tests/` | Standalone unit tests + the wiring to build them |
| `docs/` | The architecture analyses and boundary decisions this extraction was built from |

## Consuming

spatcore builds as CMake static libraries (`spatcore-audio`, `spatcore-control`,
`spatcore-controllers`). The consumer provides the third-party dependencies —
each contract fails loudly if unmet:

- **JUCE**: `juce::*` module targets must exist before `add_subdirectory(spatcore)`
  (bring your own JUCE via `add_subdirectory`).
- **juce_simpleweb** (GPLv3): `juce_add_module()` it yourself — needed by the MCP
  transport in `spatcore-control`.
- **hidapi**: set `SPATCORE_HIDAPI_INCLUDE_DIR` (headers only; you compile the
  platform implementation) — needed by `spatcore-controllers`
  (`SPATCORE_CONTROLLERS=OFF` drops the requirement).
- **roli_blocks_basics**: `juce_add_module()` — Lightpad support, also under
  `SPATCORE_CONTROLLERS`.

Compile flags: `cmake/SpatcoreCompileFlags.cmake` pins the optimization and
floating-point flags the bit-exactness gates were baselined with.

`tests/standalone/` shows a complete reference wiring (it currently resolves the
dependencies from a consumer checkout's `ThirdParty/`; self-contained pins are a
known follow-up).

## Guarantees and gates

The extraction was performed move-by-move under bit-exactness gates that live in
the consumer repo (WFS-DIY `tools/validation/`): offline render hashes for the
CPU and GPU paths, kernel-source hash manifest (the `*Kernels.h` string headers
are byte-frozen contracts compiled at runtime by NVRTC/hipRTC), control-plane
replay suites, and a dependency lint enforcing that nothing in spatcore
references a consumer. GPU backends are validated on Windows/Linux × CUDA/HIP
and macOS × Metal.

## License

Core code follows the consumer projects' licensing; note `juce_simpleweb`
(required by the MCP transport) is GPLv3 — see the consumer-provides contract
above.
