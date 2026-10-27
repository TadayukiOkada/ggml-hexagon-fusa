#!/usr/bin/env bash
# Collect a build into one flat directory for the target: tools, libraries,
# backend modules and HTP skels. Host binaries lose their debug info (the
# toolchain's crt objects carry build-machine paths); skels are copied as
# built, since stripping would invalidate an embedded SWIV CRC.
#
# usage: scripts/package.sh [--split | --asr | --sensevoice] <build dir> <output dir>
#
# --split puts tools and backend modules in bin/ and the other libraries and
# the skels in lib/, the layout of an upstream llama.cpp install (the QDC test
# scripts in llama.cpp/scripts/snapdragon/qdc expect it).
#
# A GGML_HEXAGON_ASR build adds NeMo-Speech.cpp speech recognition and run-asr.sh:
# nemo-speech with NeMo's own ggml libraries in nemo-speech/ (flat) or with the
# tools in bin/ and the libraries in lib/nemo-speech/ (--split). The ggml of the
# llama.cpp tools loads its backends as modules and NeMo's has the CPU backend
# linked in, under the same library names, so they cannot share a directory.
# Both use the same libggml-hexagon.so and skels.
#
# --asr packages only the speech recognition, flat.
#
# PACKAGE_FORBIDDEN (extended regex) adds patterns that must not appear in
# any packaged file. The source dir, build dir and $HOME are always checked.
set -euo pipefail

split=0 asr=0 sensevoice_only=0
if [ "${1:-}" = --split ]; then split=1; shift; elif [ "${1:-}" = --asr ]; then asr=1; shift; elif [ "${1:-}" = --sensevoice ]; then sensevoice_only=1; shift; fi
build=$(cd "${1:?usage: $0 [--split | --asr | --sensevoice] <build dir> <output dir>}" && pwd)
out=${2:?usage: $0 [--split | --asr | --sensevoice] <build dir> <output dir>}
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
strip_dir() {
    for f in "$1"/*; do
        case $(basename "$f") in libggml-htp-v*.so) continue ;; esac
        [ -f "$f" ] && head -c 4 "$f" | grep -q $'\x7fELF' && { strip_debug "$f" || exit 1; }
    done
    return 0
}

nemo=$build/asr/nemo-build/bin
with_asr=0
[ -x "$nemo/nemo-speech" ] && with_asr=1
if [ $asr -eq 1 ] && [ $with_asr -eq 0 ]; then
    echo "package: no $nemo/nemo-speech (configure with GGML_HEXAGON_ASR=ON)" >&2; exit 2
fi

sv=$build/sensevoice/sensevoice-build/bin
with_sensevoice=0
[ -x "$sv/llama-funasr-sensevoice" ] && [ $asr -eq 0 ] && with_sensevoice=1
if [ $sensevoice_only -eq 1 ] && [ $with_sensevoice -eq 0 ]; then
    echo "package: no SenseVoice runtime (configure with GGML_HEXAGON_SENSEVOICE=ON)" >&2; exit 2
fi
copy_sensevoice() {  # <executables> <private ggml libraries>
    for f in "$sv"/llama-funasr-sensevoice "$sv"/test-sensevoice-backend "$sv"/sensevoice-server; do [ -f "$f" ] && cp -L "$f" "$1/"; done
    for f in "$sv"/lib*.so*; do [ -f "$f" ] && cp -L "$f" "$2/"; done
    strip_dir "$1"; strip_dir "$2"
}

# NeMo: nemo-speech (and its storage test) with NeMo's own ggml libraries.
copy_nemo() {  # <dir for executables> <dir for libraries>
    for f in "$nemo"/nemo-speech "$nemo"/test_backend_storage; do [ -f "$f" ] && cp -L "$f" "$1/"; done
    for f in "$nemo"/lib*.so*; do [ -f "$f" ] && cp -L "$f" "$2/"; done
    strip_dir "$1"; strip_dir "$2"
}

rm -rf "$out"; mkdir -p "$out"
if [ $sensevoice_only -eq 1 ]; then
    with_asr=0
    copy_sensevoice "$out" "$out"
    cp "$build"/bin/libggml-hexagon.so "$out/"
elif [ $asr -eq 1 ]; then
    copy_nemo "$out" "$out"
    cp "$build"/bin/libggml-hexagon.so "$out/"
else
    for f in "$build"/bin/*; do
        name=$(basename "$f")
        case $name in
            test-*) [[ $name == test-backend-ops || $name == test-backend-ops-hex || $name == test-hexagon-metadata ]] || continue ;;
        esac
        [ -f "$f" ] || continue
        cp -L "$f" "$out/$name"
    done
fi
cp "$build"/ggml/src/ggml-hexagon/libggml-htp-v*.so "$out/"
strip_dir "$out"

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
    if [ $with_asr -eq 1 ]; then
        mkdir -p "$out/lib/nemo-speech"
        copy_nemo "$out/bin" "$out/lib/nemo-speech"
    fi
    if [ $with_sensevoice -eq 1 ]; then
        mkdir -p "$out/lib/sensevoice"
        copy_sensevoice "$out/bin" "$out/lib/sensevoice"
    fi
    cat > "$out/run.sh" <<'RUN'
#!/bin/sh
# usage: ./run.sh <tool> [args...]
# nemo-speech and test_backend_storage use NeMo's own ggml in lib/nemo-speech;
# all tools share bin/libggml-hexagon.so and the skels in lib/.
d=$(cd "$(dirname "$0")" && pwd)
tool=$1; shift
lib=$d/lib
case $tool in
    nemo-speech|test_backend_storage) lib=$d/lib/nemo-speech:$d/lib ;;
    llama-funasr-sensevoice|test-sensevoice-backend|sensevoice-server) lib=$d/lib/sensevoice:$d/lib ;;
esac
ADSP_LIBRARY_PATH="$d/lib;${ADSP_LIBRARY_PATH:-/usr/lib/rfsa/adsp;/usr/lib/dsp/cdsp;/usr/lib/dsp/cdsp1;/dsp}" \
LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" exec "$d/bin/$tool" "$@"
RUN
    sv_exe='$d/bin' sv_libs='$d/lib/sensevoice'
    exe='$d/bin' libs='$d/lib/nemo-speech' skels='$d/lib' hex='$d/bin'
else
cat > "$out/run.sh" <<'RUN'
#!/bin/sh
# usage: ./run.sh <tool> [args...]   (e.g. ./run.sh llama-cli -m model.gguf -dev HTP0 -ngl 99)
# The skels are loaded from this directory, ahead of ADSP_LIBRARY_PATH. When that is
# unset, the usual firmware library directories follow; boards that keep them
# elsewhere should export ADSP_LIBRARY_PATH (';' separated) first.
# nemo-speech and test_backend_storage live in nemo-speech/ with NeMo's own ggml;
# all tools share libggml-hexagon.so and the skels here.
d=$(cd "$(dirname "$0")" && pwd)
tool=$1; shift
exe=$d lib=$d
case $tool in
    nemo-speech|test_backend_storage) exe=$d/nemo-speech lib=$d/nemo-speech:$d ;;
    llama-funasr-sensevoice|test-sensevoice-backend|sensevoice-server)
        if [ -d "$d/sensevoice" ]; then exe=$d/sensevoice lib=$d/sensevoice:$d; fi ;;
esac
ADSP_LIBRARY_PATH="$d;${ADSP_LIBRARY_PATH:-/usr/lib/rfsa/adsp;/usr/lib/dsp/cdsp;/usr/lib/dsp/cdsp1;/dsp}" \
LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" exec "$exe/$tool" "$@"
RUN
    sv_exe='$d' sv_libs='$d'
    if [ $with_sensevoice -eq 1 ] && [ $sensevoice_only -eq 0 ]; then
        mkdir -p "$out/sensevoice"
        copy_sensevoice "$out/sensevoice" "$out/sensevoice"
        sv_exe='$d/sensevoice' sv_libs='$d/sensevoice'
    fi
    exe='$d' libs='$d' skels='$d' hex='$d'
    if [ $with_asr -eq 1 ] && [ $asr -eq 0 ]; then
        mkdir -p "$out/nemo-speech"
        copy_nemo "$out/nemo-speech" "$out/nemo-speech"
        exe='$d/nemo-speech' libs='$d/nemo-speech'
    fi
fi
chmod +x "$out/run.sh"

if [ $with_asr -eq 1 ]; then
    {
        cat <<'RUN'
#!/bin/sh
# usage: ./run-asr.sh <model.gguf> <input.wav> [nemo-speech transcribe options]
#   (e.g. ./run-asr.sh nemotron-asr.gguf input.wav --language ja-JP --json)
# Streaming recognition with the encoder on the HTP and the audio front end and
# position encoding on the CPU. ASR_DEVICE picks the HTP (default HTP0 = the
# first CDSP; HTP1 = the second). The skels are loaded ahead of ADSP_LIBRARY_PATH,
# as in run.sh.
d=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 2 ] || { echo "usage: $0 <model.gguf> <input.wav> [options]" >&2; exit 2; }
model=$1 audio=$2; shift 2
dev=${ASR_DEVICE:-HTP0}
RUN
        echo "exe=\"$exe\" libs=\"$libs\" skels=\"$skels\" hex=\"$hex\""
        cat <<'RUN'
GGML_HEXAGON_DEVICES=${GGML_HEXAGON_DEVICES:-$dev} \
ADSP_LIBRARY_PATH="$skels;${ADSP_LIBRARY_PATH:-/usr/lib/rfsa/adsp;/usr/lib/dsp/cdsp;/usr/lib/dsp/cdsp1;/dsp}" \
LD_LIBRARY_PATH="$libs:$skels${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" exec "$exe/nemo-speech" transcribe "$audio" --model "$model" \
    --backend hexagon --backend-lib "$hex/libggml-hexagon.so" --device "$dev" --frontend cpu --stream --no-batching \
    --no-warmup --asr.batching.state_arena_slots 1 --asr.backend.threads 4 --asr.backend.position_backend cpu \
    --asr.streaming.rnnt_right_context 3 "$@"
RUN
    } > "$out/run-asr.sh"
    chmod +x "$out/run-asr.sh"
fi

if [ $with_sensevoice -eq 1 ]; then
    {
        cat <<'RUN'
#!/bin/sh
# usage: ./run-sensevoice.sh <model.gguf> <audio.wav> [SenseVoice options]
d=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 2 ] || { echo "usage: $0 <model.gguf> <audio.wav> [options]" >&2; exit 2; }
model=$1 audio=$2; shift 2
dev=${SENSEVOICE_DEVICE:-HTP0}
RUN
        echo "exe=\"$sv_exe\" libs=\"$sv_libs\" skels=\"$skels\" hex=\"$hex\""
        cat <<'RUN'
GGML_HEXAGON_DEVICES=${GGML_HEXAGON_DEVICES:-$dev} \
ADSP_LIBRARY_PATH="$skels;${ADSP_LIBRARY_PATH:-/usr/lib/rfsa/adsp;/usr/lib/dsp/cdsp;/usr/lib/dsp/cdsp1;/dsp}" \
LD_LIBRARY_PATH="$libs:$skels${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" exec "$exe/llama-funasr-sensevoice" \
    -m "$model" -a "$audio" --backend hexagon --backend-lib "$hex/libggml-hexagon.so" --device "$dev" "$@"
RUN
    } > "$out/run-sensevoice.sh"
    chmod +x "$out/run-sensevoice.sh"
fi

# Licenses of what the package contains: this backend, llama.cpp/ggml and the
# notices llama.cpp ships, and with speech recognition NeMo-Speech.cpp and SentencePiece.
lic=$out/licenses; mkdir -p "$lic"
cp "$top/LICENSE" "$lic/LICENSE"
cp "$top/llama.cpp/LICENSE" "$lic/llama.cpp-LICENSE"
for f in "$top"/llama.cpp/licenses/*; do [ -f "$f" ] && cp "$f" "$lic/llama.cpp-$(basename "$f")"; done
if [ $with_asr -eq 1 ]; then
    for f in LICENSE NOTICE THIRD_PARTY_NOTICES.md; do cp "$top/NeMo-Speech.cpp/$f" "$lic/NeMo-Speech.cpp-$f"; done
    echo "NeMo-Speech.cpp is built with the changes in asr/patches of ggml-hexagon-fusa." > "$lic/NeMo-Speech.cpp-MODIFIED"
    spm=$build/asr/sentencepiece/src/sentencepiece
    if [ -f "$spm/LICENSE" ]; then
        cp "$spm/LICENSE" "$lic/sentencepiece-LICENSE"
        for d in "$spm"/third_party/*/; do
            [ -f "$d/LICENSE" ] && cp "$d/LICENSE" "$lic/sentencepiece-$(basename "$d")-LICENSE"
        done
    else
        echo "package: add the licenses of the SentencePiece library given to the build to $lic" >&2
    fi
fi

# The HTTP servers of what is packaged: nemo-speech built with GGML_HEXAGON_ASR_HTTP,
# sensevoice-server. Both use cpp-httplib (MIT, the LICENSE llama.cpp vendors).
asr_http=0 sv_http=0
[ $with_asr -eq 1 ] && [ "$(sed -n 's/^GGML_HEXAGON_ASR_HTTP:BOOL=//p' "$cache")" = ON ] && asr_http=1
[ $with_sensevoice -eq 1 ] && [ -x "$sv/sensevoice-server" ] && sv_http=1
if [ $asr_http -eq 1 ] || [ $sv_http -eq 1 ]; then
    cp "$top/llama.cpp/vendor/cpp-httplib/LICENSE" "$lic/cpp-httplib-LICENSE"
fi
if [ $sv_http -eq 1 ]; then
    # sensevoice-server uses NeMo's standalone JSON parser (no NeMo ggml).
    for f in LICENSE NOTICE THIRD_PARTY_NOTICES.md; do cp "$top/NeMo-Speech.cpp/$f" "$lic/NeMo-Speech.cpp-$f"; done
fi

if [ $with_sensevoice -eq 1 ]; then
    cp "$top/SenseVoice/LICENSE" "$lic/SenseVoice-LICENSE"
    sed -n '/^This software is available as a choice/,$p' \
        "$top/SenseVoice/runtime/llama.cpp/funasr-common/miniaudio.h" > "$lic/miniaudio-LICENSE"
    echo "SenseVoice runtime is built with sensevoice/patches and sensevoice/include of ggml-hexagon-fusa." > "$lic/SenseVoice-MODIFIED"
fi

# The gateway runs on the browser's host PC; keep it separate from target tools.
if [ $asr_http -eq 1 ] || [ $sv_http -eq 1 ]; then
    demo=$out/demos/asr-web
    mkdir -p "$demo/static"
    for f in gateway.py native.py requirements.txt profiles.example.json README.md; do
        cp "$top/demos/asr-web/$f" "$demo/"
    done
    cp "$top"/demos/asr-web/static/* "$demo/static/"
    cat > "$out/run-asr-web.sh" <<'EOF'
#!/bin/sh
# Run on the host PC with the Python dependencies installed.
set -eu
base=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${PYTHON:-python3}" "$base/demos/asr-web/gateway.py" "$@"
EOF
    chmod +x "$out/run-asr-web.sh"
fi

{
    echo "build-id:   $(cat "$build/hexagon-build-id.txt")"
    echo "llama.cpp:  $(git -C "$top/llama.cpp" rev-parse HEAD)"
    if [ $with_asr -eq 1 ]; then
        echo "NeMo-Speech.cpp: $(git -C "$top/NeMo-Speech.cpp" rev-parse HEAD) + asr/patches"
        (cd "$top" && sha256sum asr/patches/*.patch)
    fi
    if [ $with_sensevoice -eq 1 ]; then
        echo "SenseVoice: $(git -C "$top/SenseVoice" rev-parse HEAD) + sensevoice/patches"
        (cd "$top" && sha256sum sensevoice/patches/*.patch sensevoice/include/*.h sensevoice/runtime/CMakeLists.txt)
    fi
    for v in GGML_HEXAGON_SENSEVOICE_HTTP GGML_HEXAGON_ASR_HTTP GGML_HEXAGON_SENSEVOICE GGML_HEXAGON_ASR GGML_HEXAGON_FUSA GGML_HEXAGON_HTP_ARCHS GGML_HEXAGON_TOOLV GGML_HEXAGON_HTP_YIELD_BETWEEN_OPS GGML_HEXAGON_HTP_YIELD_IN_CPY GGML_HEXAGON_HTP_YIELD_IN_OPS; do
        printf '%-36s %s\n' "$v" "$(sed -n "s/^$v:[A-Z]*=//p" "$cache")"
    done
    echo
    (cd "$out" && find . -type f ! -name BUILD_INFO.txt | sort | xargs sha256sum)
} > "$out/BUILD_INFO.txt"

# A path counts only where it starts: not inside a longer path such as a toolchain's own build path.
re_quote() { printf '%s' "$1" | sed 's/[][\.^$*+?(){}|]/\\&/g'; }
paths="$(re_quote "$top")|$(re_quote "$build")"
home=${HOME%/}
[ -n "$home" ] && paths="$paths|$(re_quote "$home")"  # not when HOME is / (a container without a user entry)
pattern="(^|[^A-Za-z0-9_./-])($paths)/"
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
