# Upstream imports

`ggml/src/ggml-hexagon/` and `ggml/include/ggml-hexagon.h` come from
[llama.cpp](https://github.com/ggml-org/llama.cpp). The `vendor` branch
holds the unmodified upstream files; `main` merges it. The `llama.cpp`
submodule always points at the same commit as the latest import.

To import a newer upstream:

```sh
scripts/vendor-update.sh <commit>
git merge vendor
git -C llama.cpp checkout <commit> && git add llama.cpp
# build and test, then add a row below
```

| Date | llama.cpp commit | Notes |
|---|---|---|
| 2026-10-09 | `3d65c90d0` | Initial import |

Upstream fixes taken into `main` ahead of an import. When an import brings
the commit, its test patch no longer applies (configure stops): delete it.

| Upstream commit | In `main` as | Test patch to delete at the import |
|---|---|---|
| `d4d82d67f` hexagon: fix IM2COL patch-embed DMA ring overflow (#30189) | cherry-pick | `tests/patches/0011-im2col-exact-tiling-many-channels.patch` |

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
