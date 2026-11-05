#!/bin/sh
# The target part of a release regression: device-tests.sh and compare-packages.sh in a fixed
# order, on a new and an old package with the test kit. POSIX sh (busybox). Run it on the target.
#
#   scripts/target-suite.sh --evk  <new pkg> <old pkg> <kit> <out> [group]...
#   scripts/target-suite.sh --fusa <new pkg> <old pkg> <kit> <out> [group]...
#
# groups (default: all of the mode's, in this order)
#   ops      device-tests.sh --ops (every supported op against the CPU)          --evk only
#   long     device-tests.sh --long --leak (single ops of tens of ms), probed    --evk only
#   compare  compare-packages.sh leak llm bench prefill asr sv servers, one step at a time;
#            prefill, asr and servers probed on --evk
#   extra    300 s NeMo CLI input and a Gemma prefill of about 2k tokens (new package), probed
#            on --evk; on --fusa only the 300 s input, unprobed
#
# --fusa refuses ops and long: on functional-safety firmware a single op that holds VTCM/HMX past
# the self-test period stops the DSP, and those groups contain such ops on purpose. The
# VTCM/HMX probe (tools/vtcmlat, PROBE=<dir with vtcmlat and its skel>) runs on --evk only.
#
# <out>/steps.txt gets one line per group step: "<group> <step> rc=<n> <start> <end>"; the
# first failing step stops the suite ("STOP ..." line, exit 1). Results stay in <out>; judge them
# on the host with scripts/verdict.py. STEPS overrides the compare steps, TEST_PAUSE the pause
# after a failure in device-tests.sh. Nothing is killed with SIGKILL.
set -u

mode=${1:-}
case $mode in --evk|--fusa) ;; *) echo "usage: $0 --evk|--fusa <new> <old> <kit> <out> [group]..." >&2; exit 2 ;; esac
new=$(cd "${2:?}" && pwd); old=${3:?}; kit=$(cd "${4:?}" && pwd); out=${5:?}; shift 5
[ "$old" = - ] || old=$(cd "$old" && pwd)
mkdir -p "$out"; out=$(cd "$out" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
groups=${*:-$([ $mode = --evk ] && echo ops long compare extra || echo compare extra)}
steps=$out/steps.txt
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

for g in $groups; do
    case $mode:$g in
        --fusa:ops|--fusa:long) echo "target-suite: $g is not allowed on functional-safety firmware" >&2; exit 2 ;;
        *:ops|*:long|*:compare|*:extra) ;;
        *) echo "target-suite: unknown group $g" >&2; exit 2 ;;
    esac
done
if [ $mode = --evk ] && [ -n "${PROBE:-}" ] && [ ! -x "$PROBE/vtcmlat" ]; then
    echo "target-suite: PROBE=$PROBE has no vtcmlat" >&2; exit 2
fi

probe_loop() { # name pid: probe in 60 s chunks while pid runs (a probe holds VTCM/HMX only briefly)
    i=0
    while kill -0 "$2" 2> /dev/null; do
        LD_LIBRARY_PATH="$PROBE:${LD_LIBRARY_PATH:-}" ADSP_LIBRARY_PATH="$PROBE;${ADSP_LIBRARY_PATH:-}" \
            "$PROBE/vtcmlat" -d cdsp -p 50000 -H 800 -n 1200 -P -20 -t 2000000 -o "$out/probe/$1_$i.csv" \
            > "$out/probe/$1_$i.csv.log" 2>&1
        i=$((i + 1))
    done
}

run() { # group step command...: run, probe when asked, record; stop the suite on failure
    g=$1; s=$2; shift 2
    t0=$(now)
    if [ $mode = --evk ] && [ -n "${PROBE:-}" ] && [ "${probe:-0}" = 1 ]; then
        mkdir -p "$out/probe"
        "$@" > "$out/$g-$s.out" 2>&1 &
        pid=$!
        probe_loop "$g-$s" $pid
        wait $pid; rc=$?
    else
        "$@" > "$out/$g-$s.out" 2>&1; rc=$?
    fi
    echo "$g $s rc=$rc $t0 $(now)" >> "$steps"
    if [ $rc -ne 0 ]; then
        echo "STOP $g $s rc=$rc $(now)" >> "$steps"
        exit 1
    fi
}

echo "suite $mode new=$(head -1 "$new/BUILD_INFO.txt") old=$([ "$old" = - ] && echo - || head -1 "$old/BUILD_INFO.txt") $(now)" >> "$steps"
for g in $groups; do
    case $g in
    ops)
        probe=0 run ops ops sh "$here/device-tests.sh" "$new" "$out/ops" --ops ;;
    long)
        probe=1 run long long sh "$here/device-tests.sh" "$new" "$out/long" --long --leak ;;
    compare)
        for s in ${STEPS:-leak llm bench prefill asr sv servers}; do
            case $s in prefill|asr|servers) p=1 ;; *) p=0 ;; esac
            probe=$p run compare "$s" sh "$here/compare-packages.sh" "$new" "$old" "$kit" "$out/compare" "$s"
            if grep -Eq 'rc=-?[1-9]|alive=(nemo|sv)' "$out/compare/summary.txt"; then
                echo "STOP compare $s: failure in the summary $(now)" >> "$steps"; exit 1
            fi
        done ;;
    extra)
        probe=1 run extra asr300 env ASR_DEVICE=HTP0 timeout 1200 "$new/run-asr.sh" \
            "$kit/models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf" "$kit/audio/l300.wav" --language ja-JP --json
        if [ $mode = --evk ] && [ -f "$kit/models/gemma-4-E2B_q4_0-it.gguf" ]; then
            head -c 9000 "$kit/prompts/long_prompt.txt" > "$out/prompt2k.txt"
            probe=1 run extra gemma2k timeout 600 "$new/run.sh" llama-completion -m "$kit/models/gemma-4-E2B_q4_0-it.gguf" \
                -ngl 99 -dev HTP0 -no-cnv --temp 0 -n 4 -c 4096 -f "$out/prompt2k.txt"
        fi ;;
    esac
done
echo "DONE $(now)" >> "$steps"
