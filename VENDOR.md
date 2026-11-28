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
| 2026-11-14 | `c6c0ec6d5` | Four upstream changes after v1.0.0: buffer allocation and mapping rework (#32144), SSM_SCAN (#31921), IQ2_XXS and NVFP4 weights (#31954), the unmasked FA tail fix (#32243). Four conflicts: mapping follows upstream's `mmap_buffers` (the foreign-buffer check moved to `enqueue_op`); the IQ2_XXS decode table goes through `hmx_mm_2d_vtcm_setup`, so it is copied again after VTCM is released inside the op; `hmx_mm_nx_2d_f32` takes upstream's `rows_per_chunk` and keeps the single output of the fused GLU; `op_matmul_nx_impl` keeps both sides. SSM_SCAN stays on the CPU under the op budget (no release point, not timed on v73). One break merged without a conflict: upstream's `src2_bytes` truncation shrank the VTCM staging area of a streamed src2 (fixed in `ce00b99`). Fused GLU is not applied to tiled F16 weights (`94da20a`) |
| 2026-11-26 | `171527522` (v1.1.0) | Upstream ggml-hexagon 2.0 foundation: INT-HMX W8A8 matmul and flash attention, automatic F32-to-F16 graph lowering (`htp-lower.cpp`), generic DMA pipelines (`dma-proxy.[ch]`), 32x4 weight tiling, new `work-queue.c` (`htp-dcache.[ch]` removed), `HTP_FUSE_MUL_MAT_NX_GLU`, `HTP_FUSE_GDN_STATE`, and `TIMESTEP_EMBEDDING`. 30 conflicts; decisions in "v1.1.0 import" below |

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

## v1.1.0 import: what was kept, replaced and removed

| Local change | Upstream v1.1.0 (ggml-hexagon 2.0) | Decision |
|---|---|---|
| Rounds that release VTCM/HMX on request inside long ops (`HTP_YIELD_IN_OPS`), op budget (`htp-budget.cpp`), yield between ops | Generic DMA pipelines (`dma-proxy.[ch]`), INT-HMX W8A8 matmul/FA, new `work-queue.c` | Kept and re-integrated into 2.0 `matmul-ops.c` (pipelined/sync FP16 & INT-HMX paths, `dma_proxy_flush` inside `htp_vtcm_yield`), `flash-attn-ops.c`, `get-rows-ops.c`, `binary-ops.c`. Added `TIMESTEP_EMBEDDING`, `MUL_MAT_ID` budget, and GDN replay `n_pend` to `htp-budget.cpp` |
| Runtime LoRA fusion (`lora-ops.c`, `HTP_OP_MUL_MAT_LORA`, `HTP_OP_LORA_ADD`) | none | Kept and adapted to 2.0 (`hmx-mm-kernels-tiled-fp.h`, `hvx_flat_store_*_f32`, `resolve_fp16_output_col_chunk_fn`; skipped in `htp-lower.cpp` F32-to-F16 lowering) |
| Mixed/low-bit KV cache (`Q4_0`/`Q4_1`/`Q5_0`/`Q5_1`/`F8_E4M3`/`IQ4_NL`, `kv-cache-quant.h`, `hmx-kv-prep.h`, `hmx-q8-prep.h`) and multi-head decode (`fa_mh_run`) | W8A8 FA and `fa_phase_o_store` | Kept; `fa_mh_run` updated to 2.0 `fa_q_load_args_t` and `fa_phase_o_store` |
| Separable HVX bilinear `upscale-ops.c` | Basic upscale | Kept |
| Local `MUL_MAT_NX`+GLU fusion, GDN state-rows fusion, `htp-dcache.[ch]`, local Q4_0 tile layout, local `work-queue.c` | Upstream `HTP_FUSE_MUL_MAT_NX_GLU`, `HTP_FUSE_GDN_STATE`, 32x4 tiling, new `work-queue.c`, removed `htp-dcache.[ch]` | Replaced with upstream 2.0 implementations |
| Local small fusions (`NORM_MUL_MAT_NX`, `RMS_NORM_ROPE`, `RMS_NORM_MUL_ADD_MUL`, `MUL_MAT_POST`, compact-activation MTP kernels) | Automatic F32-to-F16 graph lowering (`htp-lower.cpp`), INT-HMX W8A8 | Removed for clean 2.0 adoption (revisit only if KPI regresses vs baseline) |
| none | `softplus` as `log(1 + exp(x))` (unary ops and `SSM_SCAN`) | New in this import: `hvx_vec_softplus_f32_inline` moved to `hvx-utils.h` as `max(x, 0) + log1p(exp(-|x|))` and used by both |
| none | `hvx_scale_offset_f16` in qf32 below v79 | New in this import: a scalar loop with exact F16/F32 conversion below v79 (`hvx-scale.h`); unary ops give F16 `SCALE` 4 MiB rounds instead of 32 MiB there |
| none | INT-HMX W8A8 rounding through the `0x4B400000` / `0x4B400080` qf32 magic bias, and post-GLU projections on F16 HMX by default | Upstream `ca123ba76` (after v1.1.0) backported: signed round-to-nearest (`hvx_vec_i32_from_qf32_rnd`, `hmx_qf32_to_i32_zp128`) on every generation. Below v81 the bias had rounded 0 to 1, so masked lanes leaked into the product. The fork's own rounding helper is gone; `hmx-fa-w8a8-kernels.h` keeps the split of the HMX deep load into 64 KiB ranges. The default `MM_SELECT=3` is upstream's pure W8A8. The W8A8+F16 hybrid that upstream dropped is kept as `MM_SELECT=4` (`htp_mm_is_post_glu` in `htp-kparams.cpp`; README "Prefill matmul precision"): pure W8A8 prefills 6-15% faster on the EVK, but mean KLD against the CPU rose from 0.0054 to 0.0154 on Qwen3.5-0.8B Q4_0 |
| none | `htp_lower_graph` restarts its scan of the touched nodes after every expansion and solves every dirty node on each pass | Kept as the reference; the transactions run in `htp-lower-fast.inc` (included by `htp-lower.cpp`) with the same lowered roots and kernel params: only nodes whose roots changed or whose inputs may have become selected roots are visited, in upstream's order; solve results are reused; a transaction ends as soon as an unsolved node can no longer change (all its roots fixed, or no node that can still act is connected to it through lowerable roots, found with cuts in the node order). `GGML_HEXAGON_LOWER_VERIFY=1` runs upstream's loops too and aborts on any difference (`test-hexagon-lower`, `device-tests.sh --lower`); `GGML_HEXAGON_LOWER_FAST=0` runs upstream's loops alone. SenseVoice's 70-layer graph: 7 ms instead of 1.1 s |
| Host buffers off up to v73, build ID check, dspqueue recovery, DSP state query, I/O trace, SWIV/SDK 5.5.7 | Host buffers on by default | Kept |

For the next import: check every new upstream DSP path against `htp-budget.cpp` and the yield points
(`review-ops`); upstream's HVX flash attention split over KV blocks (`can_split_kv`, decode) has no
yield point and no budget entry yet. `ca123ba76` is backported: from an import that includes it,
`hmx-int-utils.h` and `hmx-mm-kernels-tiled-int.h` should match upstream again. Grep new kernels for
qf32 magic-bias rounding (`0x4B400000`): it is wrong below v81. The fork's changes to
`htp-lower.cpp` are the LoRA exclusion in `can_lower` and the call into `htp-lower-fast.inc`, which
relies on how `can_lower`, `is_selected_root` and `htp_kparams_solve` depend on the lowered flags:
after a change to `htp_lower_graph` or the solvers, run `test-hexagon-lower` and the kit models with
`GGML_HEXAGON_LOWER_VERIFY=1`.
