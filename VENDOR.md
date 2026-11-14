# Upstream imports

`ggml/src/ggml-hexagon/` and `ggml/include/ggml-hexagon.h` come from
[llama.cpp](https://github.com/ggml-org/llama.cpp). The `vendor` branch
holds the unmodified upstream files; `main` merges it. The `llama.cpp`
submodule always points at the same commit as the latest import.

To import a newer upstream release (and new NeMo-Speech.cpp / SenseVoice pins):

```sh
scripts/llama-release.sh import v1.0.0 ../ggml-hexagon-fusa-import <work dir> [--nemo v0.3.0] [--sensevoice <rev>]
# a conflict stops it (merge in the worktree, or a patch series in <work dir>/rebase-*/repo):
# resolve, then run the same command again until it prints "import ready"
scripts/release-build.sh <out dir>             # in the worktree: every local build and CPU check
# device regression: scripts/target-suite.sh on the targets, scripts/verdict.py on the results
# then commit the pins, the patch series and a row below
```

`import` runs `scripts/vendor-update.sh`, merges `vendor` into `import/llama-<tag>` without
squashing, moves the pins, and moves `tests/patches/`, `asr/patches/` and `sensevoice/patches/`
with `scripts/rebase-patches.sh` (which checks that each series still adds and removes the same
lines and keeps its braces balanced). Its `review-ops` step lists the ops the HTP newly takes and
whether the op budget (`htp-budget.cpp`) or a yield point covers each: on functional-safety
firmware an op that holds VTCM/HMX past the self-test period stops the DSP.

| Date | llama.cpp commit | Notes |
|---|---|---|
| 2026-10-09 | `3d65c90d0` | Initial import |
| 2026-10-26 | `caa8191df` | Upstream rebuilt the host side (`htp-fusion`/`htp-kparams`/`htp-repack`/`htp-opnode`); main's commits ported on top in order, each with a "Port to llama.cpp caa8191df" note. Op budget moved to `htp-budget.cpp`. Merge without squashing, so the vendor merge stays the merge-base |
| 2026-10-27 | `7aca0d2ea` | `GGML_HEXAGON_VTCM_SIZE`; one conflict |
| 2026-10-28 | `8642541b2` (v0.8.0) | Hexagon backend unchanged from `7aca0d2ea`; only the llama.cpp submodule pin changes, no vendor merge needed |
| 2026-11-04 | `cd959bae9` (v0.9.0) | New HTP ops CONV_2D, CONV_2D_DW, UPSCALE, TOP_K_MASK, LIGHTNING_INDEXER, DSV4_HC_PRE/POST and a tiled transposed copy. CONV_2D_DW and the transposed copy release VTCM between rounds; the others are in the op budget, fitted to one shape each on v73 with margin (LIGHTNING_INDEXER took 38 ms at 64 heads x 16k keys, so large ones stay on the CPU). IM2COL keeps its rounds on upstream's block split and gains the KH = 1 gather path. NeMo-Speech.cpp moves to v0.3.0 (`20999cc`) |
| 2026-11-13 | `b1b770d90` (v1.0.0) | Upstream counts threads and HVX contexts apart (`n_threads`/`n_hvx`, `htp_thread_t`, `ctx->hvx_queue`), keeps tensor state as flag bits in `tensor->extra` and tiles weights in place when a graph first uses them, fuses the post-norm residual (`RMS_NORM_MUL_ADD`), adds Q5_0/Q5_1 weights, CONV_3D and an absolute skel path. 28 conflicts; decisions in "v1.0.0 import" below |

Upstream fixes taken into `main` ahead of an import. When an import brings
the commit, its test patch no longer applies (configure stops): delete it.

| Upstream commit | In `main` as | Test patch to delete at the import |
|---|---|---|

The DSP side includes a few ggml headers by paths relative to
`ggml/src/ggml-hexagon/htp`. They are symlinks into the submodule
(`ggml/src/ggml-common.h`, `ggml/src/ggml-impl.h`, `ggml/include/ggml.h`,
`ggml/include/gguf.h`), so they always match the pinned llama.cpp.

## SenseVoice runtime

`SenseVoice/` pins QwenAudio/SenseVoice at
`586bbdb2b052f4fe19322581825228b6810a9fb3`. Its official `runtime/llama.cpp`
implementation is exported and patched by `sensevoice/CMakeLists.txt`, without
changing the submodule. It uses this repository's pinned llama.cpp/ggml instead
of downloading another revision. When updating the SenseVoice pin, refresh
`sensevoice/patches/`, run the CPU reference and packaging tests documented in
README.md, then validate recognition and schedule placement on the target HTP.

## v1.0.0 import: what was kept, replaced and removed

| Local change | Upstream v1.0.0 | Decision |
|---|---|---|
| Rounds that release VTCM/HMX on request inside long ops, the op budget, yield between ops | none | Kept, with upstream's names. `CONV_3D` (new on the HTP) is in the op budget at the CONV_2D rate, not timed on v73 |
| Per-tensor metadata: pool reused after a buffer reset, views sharing the metadata of their source, covered ranges of a multipart weight upload | No per-tensor metadata: flag bits in `tensor->extra`, weights tiled at graph time | Removed. `ggml_backend_buffer_metadata_count` / `_epoch` answer 0 / 1 for a Hexagon buffer; they can go once `asr/patches` stop asking |
| Write to a weight that is already tiled | Copies standard blocks over the tiled layout | Kept as `ggml_hexagon_tensor_untile_for_write` (standard layout restored, tiled again by the next graph) |
| Residual ADD fused into RMS_NORM+MUL by a second pass over the rows | `HTP_OP_RMS_NORM_MUL_ADD`, one pass in HVX registers, broadcast weight only | Upstream's. The scalar MUL after the residual (`mul_src3`) rides on it |
| Count of matmuls taking an RMS_NORM+MUL output (norm absorbed into MUL_MAT_NX) | none | Kept, in host-only flag bits 8-15 |
| Fusions GLU into MUL_MAT_NX, norm into MUL_MAT_NX, norm into ROPE, logit softcap, GDN state rows | none | Kept. `GGML_HEXAGON_OPFUSION` bits moved to 20-25 (upstream took 13) |
| Q4_0 tile layout for vrmpy, Q8_0/low-bit KV kernels, multi-head decode, D-cache range table, GLU gate table | Q5_0/Q5_1 weight tiles and kernels next to them | Kept; upstream's Q5 kernels use the standard nibble layout and are untouched |
| Reject reasons of `supports_op` | `max_vmem` check | Both; new reason `max_vmem` |
| Host buffers off | On by default | Off up to v73 (`GGML_HEXAGON_HOSTBUF=1` turns them on); not measured on v73 |
| Build ID check, dspqueue recovery, DSP state query, I/O trace | none | Kept |

For the next import: upstream's renames were mechanical in every DSP file except `main.c`,
`work-queue.[ch]` and `htp-ctx.h`; the conflicts came from main's rounds sitting around the
`work_queue_run` calls and from the host files (`ggml-hexagon.cpp`, `htp-fusion.cpp`,
`htp-kparams.cpp`). The skel is now opened by absolute path first (the directory of
`libggml-hexagon.so` or `GGML_HEXAGON_LIBHTP_DIR`), then through the DSP search path.
