# Upstream imports

`ggml/src/ggml-hexagon/` and `ggml/include/ggml-hexagon.h` come from
[llama.cpp](https://github.com/ggml-org/llama.cpp). The `vendor` branch
holds the unmodified upstream files; `main` merges it. The `llama.cpp`
submodule always points at the same commit as the latest import.

To import a newer upstream release (and new NeMo-Speech.cpp / SenseVoice pins):

```sh
scripts/llama-release.sh import v0.9.0 ../ggml-hexagon-fusa-import <work dir> [--nemo v0.3.0] [--sensevoice <rev>]
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
