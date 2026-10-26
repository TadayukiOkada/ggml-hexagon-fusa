# ggml-hexagon-fusa

An out-of-tree build of the ggml Hexagon (HTP) backend for
[llama.cpp](https://github.com/ggml-org/llama.cpp), maintained for
**Hexagon v73 NSPs running functional-safety (FuSa) DSP firmware**.

It builds `libggml-hexagon.so` (the host side, loaded with
`GGML_BACKEND_DL`) and `libggml-htp-v73.so` (the DSP side) from one
source tree, against a pinned, unmodified llama.cpp submodule. The two
libraries always have to come from the same build.

> Status: work in progress. Built and tested on an HTP v73 FuSa board and a
> QCS9075 (v73) development kit; interfaces may still change.
>
> "FuSa" names the DSP firmware this backend runs alongside. The backend itself
> is not developed or assessed to any functional-safety standard, and this
> project is not affiliated with or endorsed by Qualcomm.

## Scope

- **Primary target:** HTP v73 with FuSa firmware. This is where
  optimization and long-run reliability work happens.
- **Other generations (v75, v79, v81):** kept building and checked from
  time to time. If one breaks it gets fixed, but it is not tuned.
- **Runtime:** plain llama.cpp tools (`llama-cli`, `llama-server`,
  `llama-mtmd-cli`, `llama-bench`, ...), and optionally NeMo-Speech.cpp
  speech recognition (below).

## Building

Clone with the submodules (`git submodule update --init`). `git clone --recursive`
also works but fetches the submodules of NeMo-Speech.cpp, which this build does not use.
The presets read the SDK locations from the environment:

| Preset | For | Environment |
|---|---|---|
| `arm64-oe-linux-v73-fusa` | v73 on FuSa firmware: skel from an older SDK, both yield options, SWIV CRC, skel checks | `OE_TOOLCHAIN_ROOT`, `HEXAGON_SDK_ROOT`/`HEXAGON_TOOLS_ROOT` (host side, SDK 6.x), `HEXAGON_HTP_SDK_ROOT`/`HEXAGON_HTP_TOOLS_ROOT`/`HEXAGON_HTP_TOOLV` (skel), `SWIV_BUILD_UTILITY`, optionally `GGML_HEXAGON_FUSA_SHELL` |
| `arm64-oe-linux-v73-fusa-asr` | the same, plus speech recognition (below) | as above |
| `arm64-oe-linux-release` | upstream defaults on an OpenEmbedded image | `OE_TOOLCHAIN_ROOT`, `HEXAGON_SDK_ROOT`, `HEXAGON_TOOLS_ROOT` |
| `arm64-linux-release` | upstream defaults, clang cross build (the `arm64-linux` toolchain container) | `HEXAGON_SDK_ROOT`, `HEXAGON_TOOLS_ROOT` |
| `arm64-android-release` | upstream defaults on Android (the `arm64-android` toolchain container) | `ANDROID_NDK_ROOT`, `HEXAGON_SDK_ROOT`, `HEXAGON_TOOLS_ROOT` |

For the FuSa skel, use the Hexagon tools bundled with SDK 5.5.x (8.7.06;
`HEXAGON_HTP_TOOLS_ROOT=$HEXAGON_HTP_SDK_ROOT/tools/HEXAGON_Tools/8.7.06`,
`HEXAGON_HTP_TOOLV=toolv87`). Tools 8.6.05 build a skel that loads and runs
but drifts in single-token gemma decoding (KL divergence against the CPU
about 8x higher). 8.7.10 emits the same code as 8.7.06; 19.0.x also works
but changes all numerics slightly.

```sh
cmake --preset arm64-oe-linux-v73-fusa -B build
cmake --build build -j
scripts/package.sh build pkg          # flat directory with run.sh and BUILD_INFO.txt
scripts/package.sh --split build pkg  # bin/ + lib/, the layout of an upstream install
```

`libggml-hexagon.so` and `libggml-htp-v*.so` embed one build ID; the host
aborts when the skel comes from another build. `hexagon-build-id.txt` in the
build directory and `BUILD_INFO.txt` in a package carry it, and every session
logs it (`ggml-hex: HTP0 build ...`).

## Testing

- `test-backend-ops-hex` is upstream `test-backend-ops` with `tests/patches/*.patch`
  applied at configure time (extra cases for this backend).
- `scripts/device-tests.sh <package> <out> [--ops] [--long] [--bench MODEL]` runs
  on the target: per-op tests against the CPU, the long single ops with
  `GGML_HEXAGON_PROFILE=1`, and `llama-bench`.
- `scripts/op-budget.py <out>/long_*.log` lists the longest ops against a per-op
  budget (default 25 ms, half of a 50 ms self-test period).
- `scripts/check-skel.sh` checks a skel for the SWIV CRC section and, given the
  firmware shell, for imports the shell does not export. FuSa builds run it.
- CI builds the Android and Linux presets for all HTP generations.

With `GGML_HEXAGON_PROFILE=1`, each batch also logs a `profile-resource` line
with VTCM/HMX maxima since the session opened (not per batch).
`response-max-us` (release request to release) is the delay a waiting self-test
or peer sees; `hold-max-us` also counts holds nobody asked to end, so it can be
as long as a whole batch.

`GGML_HEXAGON_IO_TRACE=<new-file.jsonl>` enables startup-only host I/O
diagnostics. The file is created exclusively, flushed after each record, and
limited to 128 MiB; a `truncated` record means the capture is incomplete.
If the file cannot be created or written, a warning is logged and tracing
stops; inference continues. Disabled mode adds no file I/O or device
synchronization. Recording changes host timing, so use separate
uninstrumented runs for performance comparisons.

Tensor records contain names, buffer usage, handler request lengths and known
completed host `memcpy` calls. Repack internals are unmeasured (`null`), except
for the explicitly identified output staging copy in the 2D get handler.
CPU-only copies, buffer-clear `memset` and physical DMA/DRAM traffic are outside
this capture. Initialization and lazy weight loads are included; buffer usage
and application lifecycle markers are needed to separate them from inference.

Queue records pair successful batch submissions with received DSP responses
(`HTP_STATUS_OK=1`). `batch_submit.queued_ops` counts published descriptors;
`graph_dispatch.queued_ops` counts enqueue attempts before queue fusion. These
are neither individual FastRPC invocation counts nor DSP execution durations.

Optional entry points, found with `ggml_backend_reg_get_proc_address()` on the
Hexagon registry:

| Name | Type | Returns |
|---|---|---|
| `ggml_backend_hexagon_support_reason` | `const char * (*)(ggml_backend_dev_t, const ggml_tensor *)` | why `supports_op` rejects the op, from the same checks (below) |
| `ggml_backend_buffer_metadata_epoch` | `uint64_t (*)(ggml_backend_buffer_t)` | a value that changes on every buffer reset, 0 for a non-Hexagon buffer. A tensor bound outside the graph allocator must be initialized again when it changes |
| `ggml_backend_buffer_metadata_count` | `size_t (*)(ggml_backend_buffer_t)` | tensor metadata entries held by the buffer (bounded across resets), `SIZE_MAX` for a non-Hexagon buffer |

Support reasons: `supported`, `explicit_opfilter` (`GGML_HEXAGON_OPFILTER`),
`unsupported_op` (no kernel for the op or its variant), `unsupported_dtype`,
`unsupported_shape`, `unsupported_shape_alignment`, `unsupported_stride_layout`,
`unsupported_params` (op parameters, such as a RoPE mode), `unsupported_sinks`,
`unsupported_weight_layout` (repacked or viewed quantized weights),
`unsupported_kernel` (no kernel configuration for the shape), `vtcm_limit`,
`op_budget` (estimated time over the per-op budget, left to the CPU; MUL_MAT,
from the FLOP rate and src0 bytes, with the measured rate of the kernel when src0
is not a weight, such as attention keys; SOFT_MAX by element count, in builds
without `GGML_HEXAGON_HTP_YIELD_IN_OPS`, which otherwise runs it in rounds) and
`unknown`. Reasons are recorded for MUL_MAT, MUL_MAT_ID, FLASH_ATTN_EXT, ROPE,
SOFT_MAX, the binary and unary ops, GLU, GET_ROWS, SET_ROWS, CPY, CONT, ARGSORT
and TOP_K; other ops report `unknown` when rejected.

## Speech recognition

`GGML_HEXAGON_ASR=ON` (the `arm64-oe-linux-v73-fusa-asr` preset) also builds
[NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) streaming speech
recognition with its encoder on the HTP. NeMo-Speech.cpp is a pinned, unmodified
submodule like llama.cpp: configure exports it into the build directory and
applies `asr/patches/*.patch`.

| Patch | Changes |
|---|---|
| `0001-hexagon-asr-build` | `NEMO_SPEECH_HEXAGON_ASR`: shared libraries, the pristine ggml of an external llama.cpp, no in-tree Hexagon backend |
| `0002-runtime-hexagon-backend` | loads `libggml-hexagon.so` explicitly (`--backend hexagon`, `--backend-lib`), per-role tensor storage (upstream's zero-copy mmap binding only for static weights, on buffer types that accept host memory, so not on the HTP), explicit CPU placement of named nodes, tensors bound outside the graph allocator kept valid across buffer resets (`ggml_backend_buffer_metadata_epoch`), and an optional schedule trace (`NEMO_SPEECH_SCHEDULE_TRACE`) with each node's backend and `ggml_backend_hexagon_support_reason` |
| `0003-asr-backend-placement` | CLI and C API options for the backend, front end, position encoding and decoder placement |
| `0004-asr-backend-tests` | `test_backend_storage` (storage roles, partial quantized uploads, graph reuse, support reasons) and an RNNT stage replay test |
| `0005-asr-http-demo` | admission control and revision metadata for the optional HTTP/WebSocket server |

NeMo is built as a separate CMake project with its own shared ggml (CPU backend
linked in) from the same llama.cpp submodule, and SentencePiece is built from
source at the commit NeMo pins, unless `GGML_HEXAGON_ASR_SENTENCEPIECE_LIB` and
`GGML_HEXAGON_ASR_SENTENCEPIECE_INCLUDE_DIR` name a static library for the target
(the build otherwise needs network access).

```sh
cmake --preset arm64-oe-linux-v73-fusa-asr -B build
cmake --build build -j
scripts/package.sh --split build pkg
pkg/run-asr.sh nemotron-3.5-asr-streaming-0.6b.q8_0.gguf input.wav --language ja-JP --json
ASR_DEVICE=HTP1 pkg/run-asr.sh ...    # the encoder on the second CDSP
```

NeMo's own ggml libraries go to `nemo-speech/` with `nemo-speech` (flat
package), or to `lib/nemo-speech/` with `nemo-speech` in `bin/` (`--split`): the
ggml of the llama.cpp tools loads its backends as modules, NeMo's has the CPU
backend linked in, under the same library names. Both load the same
`libggml-hexagon.so` and skels. `--asr` makes a flat package with speech
recognition only.

Separate processes can use the HTP at the same time, each with its own DSP
session: for example a llama.cpp tool on `HTP0` and `ASR_DEVICE=HTP1`, or both on
one CDSP, where the sessions hand VTCM and HMX to each other at op boundaries
through the DSP resource manager, as they do for the firmware's self-test.
To move the NeMo pin, update the submodule and refresh the patches until
configure applies them, then rerun `test_backend_storage` and recognition on
the target.

## SenseVoice

`GGML_HEXAGON_SENSEVOICE=ON` builds the official llama.cpp implementation from
[QwenAudio/SenseVoice](https://github.com/QwenAudio/SenseVoice), pinned as the
unmodified `SenseVoice/` submodule. Like NeMo, configure exports the source into
the build directory and applies `sensevoice/patches/*.patch`. The runtime uses
the same pinned llama.cpp/ggml as the backend, with private shared libraries.

```sh
git submodule update --init llama.cpp SenseVoice
cmake --preset arm64-oe-linux-v73-fusa-sensevoice -B build
cmake --build build -j
scripts/package.sh --split build pkg
pkg/run-sensevoice.sh sensevoice-small-f16.gguf input.wav --threads 8
SENSEVOICE_DEVICE=HTP1 pkg/run-sensevoice.sh sensevoice-small-f16.gguf input.wav
pkg/run.sh llama-funasr-sensevoice -m sensevoice-small-f16.gguf -a input.wav --backend cpu
```

The wrapper explicitly loads the packaged `libggml-hexagon.so`; unavailable HTP
libraries or devices fail with an error. `--trace-schedule` prints each graph
node's backend and rejection reason. All weights except the query embeddings
have persistent HTP storage across segments, and the encoder and CTC head run
on HTP as one graph split, within the backend's shape and FuSa operation-budget
checks. Audio preprocessing, query embeddings and optional VAD use the CPU.
F16 query embeddings are converted to float instead of being read as F32.

With HTP, attention uses flash attention with F16 keys and values: it releases
VTCM between output blocks, so the segment length is not bounded by the FuSa
self-test period, and the scores (segment length squared) are never stored.
The CTC head computes the best token of each frame in chunks of 256 frames, so
its logits (25055 per frame) are not kept for the whole segment either.
On the CPU (`--backend cpu`) the graph and results are those of the upstream
implementation.

The default flat package keeps SenseVoice and its ggml libraries in
`sensevoice/`; `--split` puts executables in `bin/` and private libraries in
`lib/sensevoice/`. `scripts/package.sh --sensevoice build pkg` creates a flat
SenseVoice-only package. All layouts include `run-sensevoice.sh`, source pins,
patch hashes and third-party licenses, and reuse the host backend and skels
from the same build. Both speech runtimes may be enabled together.

CPU validation needs no Hexagon SDK or downloaded model. The reference test
creates a small deterministic model and compares F32/F16 embedding inference
against an independent NumPy SAN-M/CTC implementation; the backend test also
checks persistent weights across graph shapes and the padded, masked attention
against a direct computation. Packaging tests exercise all
three layouts and argument/library routing.

```sh
cmake -S sensevoice -B build-sensevoice-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-sensevoice-cpu -j 2
ctest --test-dir build-sensevoice-cpu/sensevoice-build --output-on-failure
# Python tests require numpy and PyYAML (CI pins 1.26.4 and 6.0.2).
python3 sensevoice/tests/test_cli.py --binary build-sensevoice-cpu/sensevoice-build/bin/llama-funasr-sensevoice --llama llama.cpp
python3 sensevoice/tests/test_package.py
```

CI builds the Android/Linux ARM64 integration and runs these CPU tests. HTP
execution and real speech accuracy still require validation on the target:
compare `--ids` output with CPU for the same model and audio, inspect
`--trace-schedule`, and repeat with different segment lengths and `--vad`.

## Busy-waiting response reads

Some FastRPC kernel drivers (seen on the FuSa board's guest) leave a DSP signal
that arrives before the host starts waiting in the signaled state. From then on
every blocking `dspqueue_read()` on that queue spins on a CPU core until the
response arrives, for the rest of the queue's life; results stay correct. The
backend measures the thread CPU time of its blocking response reads and, once
they turn out to be spinning, re-creates the session's dspqueue at the next
point with no batches outstanding. The DSP side switches to the new queue and
keeps VTCM, HMX, mappings and threads. `GGML_HEXAGON_QRECOVER=0` turns this off.

`GGML_HEXAGON_OPPOLL=2` avoids blocking waits altogether: it polls without
blocking, spins for up to 100 microseconds per batch response and then sleeps
50 microseconds after each empty read. The sleep can add latency (about 2 ms
per wait on the FuSa board's guest).

## Relation to upstream

The backend sources start from upstream `ggml/src/ggml-hexagon/`. The
`vendor` branch holds unmodified upstream snapshots; `main` merges them
and carries the changes for this target. `VENDOR.md` lists the imported
upstream commits and the steps to import a newer one: update `vendor`
with `scripts/vendor-update.sh`, merge it, move the submodule to the same
commit, then build all presets and run the device tests before committing.

## License

MIT, see [LICENSE](LICENSE). The backend is derived from llama.cpp /
ggml by the ggml authors.

`asr/patches/` modify NeMo-Speech.cpp and are under its Apache License 2.0.
Packages made by `scripts/package.sh` carry the licenses of their contents in
`licenses/`.
