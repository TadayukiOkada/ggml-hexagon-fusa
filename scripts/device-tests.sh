#!/bin/sh
# Run the device-side checks on a package made by scripts/package.sh (flat layout).
# POSIX sh, so it also runs under busybox. Logs go to <out dir>; nothing is killed
# with SIGKILL (a killed session can leave the DSP holding its resources).
#
# usage: scripts/device-tests.sh <package dir> <out dir> [--ops] [--long] [--bench MODEL.gguf]...
#
#   --ops     test-backend-ops-hex against the CPU, one process per op, on HTP0, with
#             GGML_HEXAGON_PROFILE=1: run scripts/op-budget.py --fail on test_*.log from a
#             device without the self-test before running --ops on functional-safety firmware
#   --long    single ops that run for tens of ms (GGML_TEST_LONG_OPS=1), checked against the
#             CPU once each with GGML_HEXAGON_PROFILE=1; feed long_*.log to scripts/op-budget.py.
#             Some of them hold VTCM/HMX for over 100 ms in one op: do not run this on
#             functional-safety firmware, whose NSP self-test then times out and stops the DSP.
#             Measure op times on a device without the self-test.
#   --bench   llama-bench pp512, pp2048 and tg128 on HTP0 for each model
#
# Environment: ADSP_LIBRARY_PATH as the board needs it (run.sh puts the package first),
# TEST_PAUSE seconds between processes after a failure or a model run (default 20).
set -u

pkg=$(cd "${1:?usage: $0 <package dir> <out dir> [--ops] [--long] [--bench MODEL]...}" && pwd)
out=${2:?usage: $0 <package dir> <out dir> [--ops] [--long] [--bench MODEL]...}
shift 2
mkdir -p "$out"
out=$(cd "$out" && pwd)
pause=${TEST_PAUSE:-20}

ops=0; long=0; models=""
while [ $# -gt 0 ]; do
    case $1 in
        --ops) ops=1 ;;
        --long) long=1 ;;
        --bench) shift; models="$models ${1:?--bench needs a model}" ;;
        *) echo "device-tests: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

run() { # timeout log tool args...
    t=$1; log=$2; shift 2
    timeout -s TERM "$t" "$pkg/run.sh" "$@" < /dev/null > "$log" 2>&1
}

summary=$out/summary.txt
echo "package: $(head -1 "$pkg/BUILD_INFO.txt")" | tee "$summary"
echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$summary"

if [ $ops -eq 1 ]; then
    run 300 "$out/support.log" test-backend-ops-hex support -b HTP0
    grep -aoE '^  [A-Z_0-9]+\(' "$out/support.log" | tr -d ' (' | sort -u > "$out/ops.txt"
    echo "ops: $(wc -l < "$out/ops.txt")" | tee -a "$summary"
    for op in $(cat "$out/ops.txt"); do
        GGML_HEXAGON_PROFILE=1 run 3600 "$out/test_$op.log" test-backend-ops-hex test -b HTP0 -j 1 -o "$op"
        rc=$?
        echo "test $op rc=$rc $(grep -aE '[0-9]+/[0-9]+ tests passed' "$out/test_$op.log" | tail -1 | tr -s ' ')" | tee -a "$summary"
        [ $rc -ne 0 ] && sleep "$pause"
    done
fi

if [ $long -eq 1 ]; then
    # The single ops added by tests/patches/0002 and friends.
    for spec in \
        "cont CONT ne=\[(1024,(768|1152),1,1|32,32,(768|1152),1|128,8,(512,4|4096,1))\]" \
        "lmhead MUL_MAT m=(151936|262144)," \
        "attn32 MUL_MAT type_a=f32,type_b=f32,m=(512|1024|128),n=(504|1004)," \
        "attnact MUL_MAT type=f32,m=(512|1024)," \
        "softmax SOFT_MAX ne=\[(512,128,16,4|4096,512,16,1|8192,512,16,1)\]" \
        "setrows SET_ROWS ne=\[1,(1048576|65536)," \
        "argsort ARGSORT ne=\[(151936|262144),1,1,1\]" \
        "repeat REPEAT ne=\[(1,4096,1,1|128,16,512,1)\]" \
        "fa FLASH_ATTN_EXT hsk=128,hsv=128,nh=8,nr23=\[2,1\],kv=(512|4096|8192|16384),nb=512,"
    do
        set -- $spec
        GGML_TEST_LONG_OPS=1 GGML_HEXAGON_PROFILE=1 run 1800 "$out/long_$1.log" test-backend-ops-hex test -b HTP0 -j 1 -o "$2" -p "$3"
        rc=$?
        echo "long $1 rc=$rc" | tee -a "$summary"
        [ $rc -ne 0 ] && sleep "$pause"
    done
fi

for m in $models; do
    name=$(basename "$m" .gguf)
    run 1800 "$out/bench_$name.md" llama-bench -m "$m" -p 512,2048 -n 128 -r 3 -dev HTP0 -ngl 99 -o md
    rc=$?
    echo "bench $name rc=$rc" | tee -a "$summary"
    grep -E '\| *(pp|tg)[0-9]+ ' "$out/bench_$name.md" | tee -a "$summary"
    sleep "$pause"
done

echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$summary"
