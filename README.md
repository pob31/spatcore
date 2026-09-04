# spatcore

The shared spatial-audio core extracted from [WFS-DIY](https://github.com/pob31/WFS-DIY):
a JUCE-based real-time engine for wave-field synthesis and related spatial
rendering, plus the control plane that drives it. Built to be consumed as a git
submodule by multiple apps (WFS-DIY, and the planned XOA and Tight-WFS).

## Layout

| Directory | Contents |
|---|---|
| `rt/` | Realtime primitives: lock-free rings, fork-join pool, RT thread priority (MMCSS/mach/SCHED_FIFO), `RtSnapshot<T>` message→RT hand-off |
| `dsp/` | Biquads + their shared magnitude-response math, multi-channel EQ bank and the 6-band per-output EQ processor, smoothers, speed limiter, tracking filter, FR diffusion model, LFO waveforms, level detectors |
| `wfs/` | WFS renderers: CPU gather/scatter processors + the GPU-pipeline renderer wrappers |
| `reverb/` | Reverb engine + FDN / SDN / IR algorithms (CPU and GPU variants), pre/post processing, feed thread |
| `gpu/` | Multi-vendor GPU compute: CUDA / HIP / Metal backends for 5 kernel families, runtime kernel compilation (NVRTC/hipRTC), async pipeline, device manager, vendor-plugin factory, JUCE-free host work pool |
| `control/` | `osc/` wire codec + transports + ingest queues, `state/` ValueTree parameter store + XML persistence, `mcp/` MCP server core (JSON-RPC transport, dispatcher, tool registry, tier enforcement) |
| `controllers/` | Hardware controller device layer: SpaceMouse, Stream Deck+, ROLI Lightpad, evdev touch |
| `ui/` | Shared GUI widgets: the interactive EQ curve display (ValueTree-backed, undoable) and the per-band toggle. Palette and localised strings are injected by the consuming app through provider callbacks, so spatcore stays app-agnostic |
| `io/` | Audio device layer: open/restore policy that makes explicit channel masks stick, a device callback with no channel cap whose buffer is indexed by hardware channel, and the output test-signal generator (500 ms protective ramp) |
| `tools/` | `gpu/` per-vendor plugin build scripts, `codegen/` CSV→MCP tool generator core |
| `tests/` | Standalone unit tests + the wiring to build them |
| `docs/` | The architecture analyses and boundary decisions this extraction was built from |

## Consuming

spatcore builds as CMake static libraries (`spatcore-audio`, `spatcore-control`,
`spatcore-controllers`, `spatcore-ui`, `spatcore-io`). The consumer provides the third-party
dependencies — each contract fails loudly if unmet:

- **JUCE**: `juce::*` module targets must exist before `add_subdirectory(spatcore)`
  (bring your own JUCE via `add_subdirectory`).
- **OBJCXX on Apple**: with the Makefile/Ninja generators, call
  `enable_language(OBJCXX)` in your **top-level** CMakeLists before adding
  spatcore — JUCE module sources are `.mm` on Apple and compile inside *your*
  targets, and CMake only loads a language's compile rules into the enabling
  directory scope and below. spatcore fails at configure with instructions if
  this is missing. (The Xcode generator doesn't need it.)
- **juce_simpleweb** (GPLv3): `juce_add_module()` it yourself — needed by the MCP
  transport in `spatcore-control`.
- **hidapi**: set `SPATCORE_HIDAPI_INCLUDE_DIR` to a directory containing
  `hidapi/hidapi.h` (headers only; you compile the platform implementation) —
  needed by `spatcore-controllers` (`SPATCORE_CONTROLLERS=OFF` drops the
  requirement). A trimmed vendored copy and a full libusb/hidapi checkout both
  work: spatcore stages the headers into a private include tree, so stray files
  in that directory (e.g. hidapi's root `VERSION` file, which would otherwise
  shadow libc++'s `<version>` on macOS) never reach the include path.
- **roli_blocks_basics**: `juce_add_module()` — Lightpad support, also under
  `SPATCORE_CONTROLLERS`.

`spatcore-ui` needs no extra dependency (its JUCE modules ship with JUCE), but
it does link `juce_gui_basics` — which `spatcore-audio` deliberately never does,
so the audio layer stays usable headless. `SPATCORE_UI=OFF` drops it entirely.

`spatcore-io` is separate for the same reason and links `juce_audio_devices`,
which pulls in the platform driver backends (ASIO/WASAPI, CoreAudio, ALSA/JACK).
An app that opens its own device wants it; a DAW plugin or an offline renderer,
where the host owns the device, sets `SPATCORE_IO=OFF`.

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

**GPL-3.0-or-later** — see [`LICENSE`](LICENSE).

spatcore was extracted from WFS-DIY, which is GPL-3, and it is consumed by
WFS-DIY, XOA, Tight-WFS and Go.dot, all of which are GPL-3. Its dependencies are
compatible with that and in two cases require it: JUCE is used under the AGPLv3
path, Tracktion Engine (where a consumer links it) under the GPLv3 path, and
`juce_simpleweb`, which the MCP transport needs, is GPLv3 — see the
consumer-provides contract above.

Stating this in a file rather than leaving it to "whatever the consumer is"
matters for one practical reason: a repository with no licence grants nobody any
rights, including a future consumer that wants to pin it.
