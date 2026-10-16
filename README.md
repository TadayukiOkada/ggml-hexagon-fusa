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
scripts/package.sh --split build pkg  # bin/ + lib/, the layout of an upstream install (adds speech recognition, below)
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
`op_budget` (estimated time over the per-op budget, left to the CPU) and
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
| `0002-runtime-hexagon-backend` | loads `libggml-hexagon.so` explicitly (`--backend hexagon`, `--backend-lib`), per-role tensor storage, explicit CPU placement of named nodes, tensors bound outside the graph allocator kept valid across buffer resets (`ggml_backend_buffer_metadata_epoch`), and an optional schedule trace (`NEMO_SPEECH_SCHEDULE_TRACE`) with each node's backend and `ggml_backend_hexagon_support_reason` |
| `0003-asr-backend-placement` | CLI and C API options for the backend, front end, position encoding and decoder placement |
| `0004-asr-backend-tests` | `test_backend_storage` (storage roles, partial quantized uploads, graph reuse, support reasons) and an RNNT stage replay test |

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

`--split` puts `nemo-speech` in `bin/` next to the llama.cpp tools and NeMo's own
ggml libraries in `lib/nemo-speech/`: the ggml of the llama.cpp tools loads its
backends as modules, NeMo's has the CPU backend linked in, under the same
library names. Both load the same `bin/libggml-hexagon.so` and skels in `lib/`.
`--asr` makes a flat package with speech recognition only.

Separate processes can use the HTP at the same time, each with its own DSP
session: for example a llama.cpp tool on `HTP0` and `ASR_DEVICE=HTP1`, or both on
one CDSP, where the sessions hand VTCM and HMX to each other at op boundaries
through the DSP resource manager, as they do for the firmware's self-test.
To move the NeMo pin, update the submodule and refresh the patches until
configure applies them, then rerun `test_backend_storage` and recognition on
the target.

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
