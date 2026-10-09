# ggml-hexagon-fusa

An out-of-tree build of the ggml Hexagon (HTP) backend for
[llama.cpp](https://github.com/ggml-org/llama.cpp), maintained for
**Hexagon v73 NSPs running functional-safety (FuSa) DSP firmware**.

It builds `libggml-hexagon.so` (the host side, loaded with
`GGML_BACKEND_DL`) and `libggml-htp-v73.so` (the DSP side) from one
source tree, against a pinned, unmodified llama.cpp submodule. The two
libraries always have to come from the same build.

> Status: work in progress. Nothing here is usable yet.

## Scope

- **Primary target:** HTP v73 with FuSa firmware. This is where
  optimization and long-run reliability work happens.
- **Other generations (v75, v79, v81):** kept building and checked from
  time to time. If one breaks it gets fixed, but it is not tuned.
- **Runtime:** plain llama.cpp tools (`llama-cli`, `llama-server`,
  `llama-mtmd-cli`, `llama-bench`, ...).

## Relation to upstream

The backend sources start from upstream `ggml/src/ggml-hexagon/`. The
`vendor` branch holds unmodified upstream snapshots; `main` merges them
and carries the changes for this target. `VENDOR.md` lists the imported
upstream commits.

## License

MIT, see [LICENSE](LICENSE). The backend is derived from llama.cpp /
ggml by the ggml authors.
