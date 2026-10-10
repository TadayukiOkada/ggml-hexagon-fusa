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

## Scope

- **Primary target:** HTP v73 with FuSa firmware. This is where
  optimization and long-run reliability work happens.
- **Other generations (v75, v79, v81):** kept building and checked from
  time to time. If one breaks it gets fixed, but it is not tuned.
- **Runtime:** plain llama.cpp tools (`llama-cli`, `llama-server`,
  `llama-mtmd-cli`, `llama-bench`, ...).

## Building

Clone with the submodule (`git clone --recursive`, or `git submodule update --init`).
The presets read the SDK locations from the environment:

| Preset | For | Environment |
|---|---|---|
| `arm64-oe-linux-v73-fusa` | v73 on FuSa firmware: skel from an older SDK, both yield options, SWIV CRC, skel checks | `OE_TOOLCHAIN_ROOT`, `HEXAGON_SDK_ROOT`/`HEXAGON_TOOLS_ROOT` (host side, SDK 6.x), `HEXAGON_HTP_SDK_ROOT`/`HEXAGON_HTP_TOOLS_ROOT`/`HEXAGON_HTP_TOOLV` (skel), `SWIV_BUILD_UTILITY`, optionally `GGML_HEXAGON_FUSA_SHELL` |
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
