#!/usr/bin/env bash
# The local build gate of a release (T0): every build and CPU check an import needs before any
# device test, each step logged to <out>/<step>.log and recorded as one JSON line in
# <out>/results.jsonl. Stops at the first failing step (later steps are recorded as not run).
#
#   scripts/release-build.sh <out dir> [--only fusa,all,svcpu,offline] [--tests ON|OFF]
#
#   fusa     preset arm64-oe-linux-v73-fusa-asr with SenseVoice and both HTTP servers: skel with
#            the FuSa SDK/tools (HEXAGON_HTP_SDK_ROOT, HEXAGON_HTP_TOOLS_ROOT, HEXAGON_HTP_TOOLV,
#            SWIV_BUILD_UTILITY, GGML_HEXAGON_FUSA_SHELL as the preset needs), then the split and
#            flat packages (pkg/, flat/, with SHA256SUMS) and the skel check
#   all      preset arm64-oe-linux-release with SenseVoice for every default HTP generation, with
#            ALL_HEXAGON_SDK_ROOT / ALL_HEXAGON_TOOLS_ROOT (a newer SDK), and its split package
#   svcpu    SenseVoice on the host CPU: CTest, the NumPy reference and the packaging tests
#   offline  tests/release (scripts/verdict.py on synthetic results)
#
# --tests OFF builds the FuSa preset as released (no test programs); default ON.
# CCACHE_DISABLE=1 is set: a release build must not reuse objects from another tree.
set -uo pipefail

top=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?usage: $0 <out dir> [--only fusa,all,svcpu,offline] [--tests ON|OFF]}
shift
only=fusa,all,svcpu,offline tests=ON
while [ $# -gt 0 ]; do
    case $1 in
        --only) only=${2:?}; shift ;;
        --tests) tests=${2:?}; shift ;;
        *) echo "release-build: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
mkdir -p "$out"; out=$(cd "$out" && pwd)
export CCACHE_DISABLE=1
jobs=${RELEASE_BUILD_JOBS:-$(nproc)}
results=$out/results.jsonl
failed=""

step() { # name command...: run unless an earlier step failed, log, record
    local name=$1; shift
    local log=$out/$name.log t0 t1 rc
    if [ -n "$failed" ]; then
        printf '{"step":"%s","status":"NOT_RUN","after":"%s"}\n' "$name" "$failed" >> "$results"
        return 1
    fi
    t0=$(date -u +%FT%TZ)
    echo "release-build: $name"
    ( cd "$top" && "$@" ) > "$log" 2>&1
    rc=$?
    t1=$(date -u +%FT%TZ)
    printf '{"step":"%s","status":"%s","rc":%d,"start":"%s","end":"%s","log":"%s"}\n' \
        "$name" "$([ $rc -eq 0 ] && echo PASS || echo FAIL)" "$rc" "$t0" "$t1" "$name.log" >> "$results"
    if [ $rc -ne 0 ]; then
        failed=$name
        echo "release-build: $name FAILED (rc=$rc), see $log" >&2
        tail -20 "$log" >&2
        return 1
    fi
}
want() { [[ ",$only," == *",$1,"* ]]; }
sums() { (cd "$1" && find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS); }

{
    echo "source: $(git -C "$top" rev-parse HEAD) $(git -C "$top" status --porcelain --untracked-files=no | wc -l) modified tracked files"
    git -C "$top" submodule status
    echo "date: $(date -u +%FT%TZ) tests=$tests only=$only"
} > "$out/source.txt"
: > "$results"

if want fusa; then
    rm -rf "$out/build-fusa" "$out/pkg" "$out/flat"
    step fusa-configure cmake --preset arm64-oe-linux-v73-fusa-asr -B "$out/build-fusa" \
        -DGGML_HEXAGON_SENSEVOICE=ON -DGGML_HEXAGON_ASR_HTTP=ON -DGGML_HEXAGON_SENSEVOICE_HTTP=ON \
        -DLLAMA_BUILD_TESTS=$tests
    step fusa-build cmake --build "$out/build-fusa" -j "$jobs"
    step fusa-package-split scripts/package.sh --split "$out/build-fusa" "$out/pkg"
    step fusa-package-flat scripts/package.sh "$out/build-fusa" "$out/flat"
    # the FuSa build itself runs scripts/check-skel.sh (CRC section, firmware shell imports)
    if [ -z "$failed" ]; then
        sums "$out/pkg"; sums "$out/flat"
        { grep -m1 -o 'build ID[^,]*' "$out/build-fusa.log" 2>/dev/null
          sha256sum "$out/flat/libggml-htp-v73.so" "$out/flat/libggml-hexagon.so"; } > "$out/fusa-artifacts.txt"
    fi
fi

if want all; then
    rm -rf "$out/build-all" "$out/pkg-all"
    step all-configure env HEXAGON_SDK_ROOT="${ALL_HEXAGON_SDK_ROOT:?ALL_HEXAGON_SDK_ROOT not set}" \
        HEXAGON_TOOLS_ROOT="${ALL_HEXAGON_TOOLS_ROOT:?ALL_HEXAGON_TOOLS_ROOT not set}" \
        cmake --preset arm64-oe-linux-release -B "$out/build-all" -DGGML_HEXAGON_SENSEVOICE=ON
    step all-build env HEXAGON_SDK_ROOT="$ALL_HEXAGON_SDK_ROOT" HEXAGON_TOOLS_ROOT="$ALL_HEXAGON_TOOLS_ROOT" \
        cmake --build "$out/build-all" -j "$jobs"
    step all-package scripts/package.sh --split "$out/build-all" "$out/pkg-all"
fi

if want svcpu; then
    rm -rf "$out/build-sv-cpu"
    step svcpu-configure cmake -S sensevoice -B "$out/build-sv-cpu" -G Ninja -DCMAKE_BUILD_TYPE=Release
    step svcpu-build cmake --build "$out/build-sv-cpu" -j "$jobs"
    step svcpu-ctest ctest --test-dir "$out/build-sv-cpu/sensevoice-build" --output-on-failure
    step svcpu-cli python3 sensevoice/tests/test_cli.py \
        --binary "$out/build-sv-cpu/sensevoice-build/bin/llama-funasr-sensevoice" --llama llama.cpp
    step svcpu-package python3 sensevoice/tests/test_package.py
fi

if want offline; then
    step offline-tests python3 -m unittest discover -s tests/release -v
fi

pass=$(grep -c '"status":"PASS"' "$results"); total=$(wc -l < "$results")
echo "release-build: $pass/$total steps passed; $results"
[ -z "$failed" ]
