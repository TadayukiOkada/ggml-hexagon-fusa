#!/bin/sh
# Run the device-side checks on a package made by scripts/package.sh (flat layout).
# POSIX sh, so it also runs under busybox. Logs go to <out dir>; nothing is killed
# with SIGKILL (a killed session can leave the DSP holding its resources).
#
# usage: scripts/device-tests.sh <package dir> <out dir> [--ops] [--long] [--leak] [--rows] [--lower] [--bench MODEL.gguf]...
#
#   --leak    test-hexagon-metadata on HTP0: graphs allocated again and again with views of a
#             weight, of a cache and scratch views must not grow the backend's tensor metadata
#             (seconds; scripts/rss-soak.py checks a running server end to end)
#   --rows    test-hexagon-mm-rows on HTP0: each row of a quantized mat-mul with two to four
#             activation rows must have the same bits as the row computed alone (speculative
#             decoding verifies draft tokens in such batches); short ops, fine on any firmware
#   --lower   test-hexagon-lower on HTP0 (both fast modes): random small graphs whose lowering to
#             F16 must be the same with the fork's fast transactions as with upstream's loops; short ops
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
# TEST_PAUSE seconds between processes after a failure or a model run (default 20),
# LONG_ONLY=name,... runs only those --long cases. To time ops the op budget leaves to the CPU
# (to fit its estimates), run --long with GGML_HEXAGON_OP_BUDGET_MS=0, on a device without the
# self-test only.
set -u

pkg=$(cd "${1:?usage: $0 <package dir> <out dir> [--ops] [--long] [--leak] [--rows] [--lower] [--bench MODEL]...}" && pwd)
out=${2:?usage: $0 <package dir> <out dir> [--ops] [--long] [--leak] [--rows] [--lower] [--bench MODEL]...}
shift 2
mkdir -p "$out"
out=$(cd "$out" && pwd)
pause=${TEST_PAUSE:-20}

ops=0; long=0; leak=0; rows=0; lower=0; models=""
while [ $# -gt 0 ]; do
    case $1 in
        --ops) ops=1 ;;
        --long) long=1 ;;
        --leak) leak=1 ;;
        --rows) rows=1 ;;
        --lower) lower=1 ;;
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
    # the fused-graph tests of tests/patches have names of their own, which the support run does not list
    # LORA_FUSION_UNFUSED: the LORA_FUSION cases with the chains left unfused (GGML_HEXAGON_LORA_UNFUSED),
    # the adapter weights flat on the flat-weight matmuls, as when a fused op does not fit
    fused="F16_DIV_PRECISION FA_SINK_NORMALIZATION GDN_DECAY LORA_FUSION LORA_FUSION_UNFUSED MUL_MAT_ADD_RESIDUAL
        MUL_MAT_GLU MUL_MAT_SOFTCAP NORM_MUL_MAT RMS_NORM_ADD_SCALE RMS_NORM_ROPE"
    { grep -aoE '^  [A-Z_0-9]+\(' "$out/support.log" | tr -d ' ('; printf '%s\n' $fused; } | sort -u > "$out/ops.txt"
    echo "ops: $(wc -l < "$out/ops.txt")" | tee -a "$summary"
    ops_bad=0
    for op in $(cat "$out/ops.txt"); do
        o=${op%_UNFUSED}; unfused=0; [ "$o" != "$op" ] && unfused=1
        GGML_HEXAGON_LORA_UNFUSED=$unfused GGML_HEXAGON_PROFILE=1 run 3600 "$out/test_$op.log" test-backend-ops-hex test -b HTP0 -j 1 -o "$o"
        rc=$?
        echo "test $op rc=$rc $(grep -aE '[0-9]+/[0-9]+ tests passed' "$out/test_$op.log" | tail -1 | tr -s ' ')" | tee -a "$summary"
        if [ $rc -ne 0 ]; then
            if [ $rc -lt 0 ] || [ $rc -gt 128 ] || [ $rc -eq 124 ] || [ $rc -eq 126 ] || [ $rc -eq 127 ]; then
                ops_bad=1
            elif ! grep -q "Failing tests:" "$out/test_$op.log" &&
                 ! grep -q "no test case ran: the selected test cases are not supported" "$out/test_$op.log"; then
                ops_bad=1
            fi
            sleep "$pause"
        fi
    done
    [ "$ops_bad" -eq 0 ] || exit 1
fi

if [ $long -eq 1 ]; then
    # The single ops added by tests/patches/0002 and friends.
    for spec in \
        "cont CONT ne=\[(1024,(768|1152),1,1|32,32,(768|1152),1|128,8,(512,4|4096,1|16384,1))\]" \
        "lmhead MUL_MAT m=(151936|262144)," \
        "attn32 MUL_MAT type_a=f32,type_b=f32,m=(512|1024|128),n=(504|1004)," \
        "attnact MUL_MAT type=f32,m=(512|1024)," \
        "softmax SOFT_MAX ne=\[(512,128,16,4|4096,512,16,1|8192,512,16,1)\]" \
        "setrows SET_ROWS ne=\[1,(1048576|65536)," \
        "binadd ADD ne=\[4096,(1|4096),1,1\],nr=\[1,(4096|1),1,1\]" \
        "binmul MUL ne=\[(4096,4096|1,1),1,1\],nr=\[(1,1|4096,4096),1,1\]" \
        "argsort ARGSORT ne=\[(151936|262144),1,1,1\]" \
        "repeat REPEAT ne=\[(1,4096,1,1|128,16,512,1)\]" \
        "fa FLASH_ATTN_EXT hsk=128,hsv=128,nh=8,nr23=\[2,1\],kv=(512|4096|8192|16384),nb=512," \
        "convdw CONV_2D_DW ne_input=\[(4096,1,1024,1|64,3000,256,1)\]" \
        "conv2d CONV_2D ne_input=\[128,128,256,1\]" \
        "upscale UPSCALE ne=\[(512,512,64,1|1152,16,16,1)\]" \
        "topkmask TOP_K_MASK n_kv=32768," \
        "indexer LIGHTNING_INDEXER hsk=128,nh=64,kv=16384,"
    do
        set -- $spec
        case ",${LONG_ONLY:-$1}," in *",$1,"*) ;; *) continue ;; esac
        GGML_TEST_LONG_OPS=1 GGML_HEXAGON_PROFILE=1 run 1800 "$out/long_$1.log" test-backend-ops-hex test -b HTP0 -j 1 -o "$2" -p "$3"
        rc=$?
        echo "long $1 rc=$rc" | tee -a "$summary"
        [ $rc -ne 0 ] && sleep "$pause"
    done
fi

if [ $leak -eq 1 ]; then
    run 300 "$out/metadata.log" test-hexagon-metadata HTP0 64
    rc=$?
    echo "metadata rc=$rc $(tail -1 "$out/metadata.log")" | tee -a "$summary"
    [ $rc -ne 0 ] && sleep "$pause"
fi

if [ $rows -eq 1 ]; then
    run 600 "$out/mm_rows.log" test-hexagon-mm-rows HTP0
    rc=$?
    echo "mm-rows rc=$rc $(tail -1 "$out/mm_rows.log")" | tee -a "$summary"
    [ $rc -ne 0 ] && sleep "$pause"
fi

if [ $lower -eq 1 ]; then
    for mode in 1 2; do
        run 600 "$out/lower_$mode.log" test-hexagon-lower HTP0 300 $mode
        rc=$?
        echo "lower mode $mode rc=$rc $(tail -1 "$out/lower_$mode.log")" | tee -a "$summary"
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
