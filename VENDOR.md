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

The DSP side includes a few ggml headers by paths relative to
`ggml/src/ggml-hexagon/htp`. They are symlinks into the submodule
(`ggml/src/ggml-common.h`, `ggml/src/ggml-impl.h`, `ggml/include/ggml.h`,
`ggml/include/gguf.h`), so they always match the pinned llama.cpp.
