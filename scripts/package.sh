#!/usr/bin/env bash
# Collect a build into one flat directory for the target: tools, libraries,
# backend modules and HTP skels. Host binaries lose their debug info (the
# toolchain's crt objects carry build-machine paths); skels are copied as
# built, since stripping would invalidate an embedded SWIV CRC.
#
# usage: scripts/package.sh [--split] <build dir> <output dir>
#
# --split puts tools and backend modules in bin/ and the other libraries and
# the skels in lib/, the layout of an upstream llama.cpp install (the QDC test
# scripts in llama.cpp/scripts/snapdragon/qdc expect it).
#
# PACKAGE_FORBIDDEN (extended regex) adds patterns that must not appear in
# any packaged file. The source dir, build dir and $HOME are always checked.
set -euo pipefail

split=0
if [ "${1:-}" = --split ]; then split=1; shift; fi
build=$(cd "${1:?usage: $0 [--split] <build dir> <output dir>}" && pwd)
out=${2:?usage: $0 [--split] <build dir> <output dir>}
top=$(cd "$(dirname "$0")/.." && pwd)

cache=$build/CMakeCache.txt
[ -f "$cache" ] || { echo "package: no CMakeCache.txt in $build" >&2; exit 2; }
strip=$(sed -n 's/^CMAKE_STRIP:FILEPATH=//p' "$cache")
# CMAKE_STRIP can be the host's strip when the toolchain file does not set one: then try
# llvm-strip (any target) and the aarch64 binutils strip.
strip_debug() {
    for s in "$strip" llvm-strip $(compgen -c llvm-strip- 2>/dev/null | sort -u) aarch64-linux-gnu-strip; do
        command -v "$s" > /dev/null 2>&1 && "$s" --strip-debug "$1" 2>/dev/null && return 0
    done
    echo "package: no strip could read $1" >&2; return 1
}

rm -rf "$out"; mkdir -p "$out"
for f in "$build"/bin/*; do
    name=$(basename "$f")
    case $name in
        test-*) [[ $name == test-backend-ops || $name == test-backend-ops-hex ]] || continue ;;
    esac
    [ -f "$f" ] || continue
    cp -L "$f" "$out/$name"
done
cp "$build"/ggml/src/ggml-hexagon/libggml-htp-v*.so "$out/"

for f in "$out"/*; do
    case $(basename "$f") in libggml-htp-v*.so) continue ;; esac
    if head -c 4 "$f" | grep -q $'\x7fELF'; then strip_debug "$f" || exit 1; fi
done

if [ $split -eq 1 ]; then
    mkdir -p "$out/bin" "$out/lib"
    for f in "$out"/*; do
        [ -f "$f" ] || continue
        case $(basename "$f") in
            libggml-cpu*.so|libggml-hexagon.so) mv "$f" "$out/bin/" ;;  # loaded from the executable's directory
            lib*.so*) mv "$f" "$out/lib/" ;;
            *) mv "$f" "$out/bin/" ;;
        esac
    done
    cat > "$out/run.sh" <<'RUN'
#!/bin/sh
# usage: ./run.sh <tool> [args...]
d=$(cd "$(dirname "$0")" && pwd)
tool=$1; shift
ADSP_LIBRARY_PATH="$d/lib;${ADSP_LIBRARY_PATH:-/usr/lib/rfsa/adsp;/usr/lib/dsp/cdsp;/usr/lib/dsp/cdsp1;/dsp}" \
LD_LIBRARY_PATH="$d/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" exec "$d/bin/$tool" "$@"
RUN
else
cat > "$out/run.sh" <<'RUN'
#!/bin/sh
# usage: ./run.sh <tool> [args...]   (e.g. ./run.sh llama-cli -m model.gguf -dev HTP0 -ngl 99)
# The skels are loaded from this directory, ahead of ADSP_LIBRARY_PATH. When that is
# unset, the usual firmware library directories follow; boards that keep them
# elsewhere should export ADSP_LIBRARY_PATH (';' separated) first.
d=$(cd "$(dirname "$0")" && pwd)
tool=$1; shift
ADSP_LIBRARY_PATH="$d;${ADSP_LIBRARY_PATH:-/usr/lib/rfsa/adsp;/usr/lib/dsp/cdsp;/usr/lib/dsp/cdsp1;/dsp}" \
LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" exec "$d/$tool" "$@"
RUN
fi
chmod +x "$out/run.sh"

{
    echo "build-id:   $(cat "$build/hexagon-build-id.txt")"
    echo "llama.cpp:  $(git -C "$top/llama.cpp" rev-parse HEAD)"
    for v in GGML_HEXAGON_FUSA GGML_HEXAGON_HTP_ARCHS GGML_HEXAGON_TOOLV GGML_HEXAGON_HTP_YIELD_BETWEEN_OPS GGML_HEXAGON_HTP_YIELD_IN_CPY GGML_HEXAGON_HTP_YIELD_IN_OPS; do
        printf '%-36s %s\n' "$v" "$(sed -n "s/^$v:[A-Z]*=//p" "$cache")"
    done
    echo
    (cd "$out" && find . -type f ! -name BUILD_INFO.txt | sort | xargs sha256sum)
} > "$out/BUILD_INFO.txt"

# A path counts only where it starts: not inside a longer path such as a toolchain's own build path.
re_quote() { printf '%s' "$1" | sed 's/[][\.^$*+?(){}|]/\\&/g'; }
pattern="(^|[^A-Za-z0-9_./-])($(re_quote "$top")|$(re_quote "$build")|$(re_quote "${HOME%/}"))/"
[ -n "${PACKAGE_FORBIDDEN:-}" ] && pattern="$pattern|$PACKAGE_FORBIDDEN"
bad=0
for f in $(find "$out" -type f); do
    if hits=$(strings -a "$f" | grep -E "$pattern" | sort -u) && [ -n "$hits" ]; then
        echo "package: $(basename "$f") contains build-machine paths:" >&2
        echo "$hits" | head -5 | sed 's/^/  /' >&2
        bad=1
    fi
done
[ $bad -eq 0 ] || exit 1
echo "package: $(find "$out" -type f | wc -l) files in $out"
