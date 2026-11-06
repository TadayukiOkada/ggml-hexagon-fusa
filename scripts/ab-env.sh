#!/bin/sh
# A/B of a runtime setting on one package: the same models with environment A ("old") and B
# ("new"), e.g. an experimental kernel flag off and on. POSIX sh; nothing is killed with SIGKILL.
# Summary lines use the compare-packages.sh format, so scripts/verdict.py --compare judges them
# (outputs must match; speed from the means of the balanced runs).
#
#   scripts/ab-env.sh <pkg> <kit> <out> <env A> <env B> <step>...
#   e.g. scripts/ab-env.sh pkg kit res GGML_HEXAGON_FA_Q8_PREP=0 GGML_HEXAGON_FA_Q8_PREP=1 check bench profile
#
#   ops      test-backend-ops-hex for $OPS (default FLASH_ATTN_EXT) matching $OPS_PATTERN, A and B
#   check    llama-completion, 64 tokens greedy, from a 664-token and a 12.7k-token prompt, A and B
#   bench    llama-bench pp512 and tg32 at each depth, in the order A, B, B, A (-r $REPS)
#   profile  GGML_HEXAGON_PROFILE=1 llama-completion of the 12.7k prompt, A and B, probed with
#            tools/vtcmlat when PROBE=<dir> (devices without the FuSa self-test only); the
#            profile-op lines go to profile_<A|B>_<model>.log.gz for scripts/op-budget.py
#
# Environment: MODELS (kit model file names, default Qwen3.5-0.8B-Q4_0.gguf), DEPTHS (default
# "4096 12288"), KV (llama.cpp KV options, default "-ctk q8_0 -ctv q8_0"), REPS (default 5),
# ADSP_LIBRARY_PATH as the target needs.
set -u

pkg=$(cd "${1:?usage: $0 <pkg> <kit> <out> <env A> <env B> <step>...}" && pwd)
kit=$(cd "${2:?}" && pwd)
out=${3:?}; envA=${4:?}; envB=${5:?}; shift 5
mkdir -p "$out"; out=$(cd "$out" && pwd)
summary=$out/summary.txt
models=${MODELS:-Qwen3.5-0.8B-Q4_0.gguf}
depths=${DEPTHS:-4096 12288}
kv=${KV:--ctk q8_0 -ctv q8_0}
reps=${REPS:-5}
opts="-dev HTP0 -ngl 99 -fa on"

log() { echo "$*" | tee -a "$summary"; }
md5() { md5sum < "$1" | cut -c1-12; }
envof() { if [ "$1" = old ]; then echo "$envA"; else echo "$envB"; fi; }
same() { # key value version
    if [ "$3" = old ]; then echo "$2" > "$out/.$1"; echo ""; return; fi
    if [ "$(cat "$out/.$1" 2>/dev/null)" = "$2" ]; then echo same; else echo DIFFERENT; fi
}
probe_loop() { # name pid
    i=0
    while kill -0 "$2" 2> /dev/null; do
        LD_LIBRARY_PATH="$PROBE:${LD_LIBRARY_PATH:-}" ADSP_LIBRARY_PATH="$PROBE;${ADSP_LIBRARY_PATH:-}" \
            "$PROBE/vtcmlat" -d cdsp -p 50000 -H 800 -n 1200 -P -20 -t 2000000 -o "$out/probe/$1_$i.csv" \
            > "$out/probe/$1_$i.csv.log" 2>&1
        i=$((i + 1))
    done
}

log "pkg: $(head -1 "$pkg/BUILD_INFO.txt")"
log "old: A $envA"
log "new: B $envB"
log "kv: $kv  models: $models  depths: $depths"
head -c 2600 "$kit/prompts/long_prompt.txt" > "$out/prompt664.txt"
for step in "$@"; do
    log "## $step $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    case $step in
    ops)
        for v in old new; do for o in ${OPS:-FLASH_ATTN_EXT}; do
            env $(envof $v) timeout 3600 "$pkg/run.sh" test-backend-ops-hex test -b HTP0 -j 1 -o "$o" ${OPS_PATTERN:+-p "$OPS_PATTERN"} \
                < /dev/null > "$out/ops_${v}_$o.log" 2>&1
            log "ops $v $o rc=$? $(grep -aE '[0-9]+/[0-9]+ tests passed' "$out/ops_${v}_$o.log" | tail -1 | tr -s ' ')"
            sleep 5; done; done ;;
    check)
        for v in old new; do for m in $models; do for p in prompt664 prompt12k; do
            f=$out/$p.txt; [ $p = prompt12k ] && f=$kit/prompts/prompt12k.txt
            n=${m%.gguf}
            env $(envof $v) timeout 900 "$pkg/run.sh" llama-completion -m "$kit/models/$m" $opts $kv -c 16384 \
                -no-cnv --temp 0 -s 1 -n 64 -f "$f" < /dev/null > "$out/check_${v}_${n}_$p.out" 2> "$out/check_${v}_${n}_$p.err"
            rc=$?; h=$(md5 "$out/check_${v}_${n}_$p.out")
            log "llm $v $n@$p rc=$rc md5=$h $(same "c_${n}_$p" "$h" $v) $(grep -a ' eval time' "$out/check_${v}_${n}_$p.err" | grep -v 'prompt eval' | tail -1 | sed 's/.*(//; s/)//')"
            sleep 5; done; done; done ;;
    bench)
        for m in $models; do n=${m%.gguf}; for d in $depths; do i=0
            for v in old new new old; do i=$((i + 1))
                env $(envof $v) timeout 1800 "$pkg/run.sh" llama-bench -m "$kit/models/$m" $opts $kv -p 512 -n 32 -d "$d" \
                    -r "$reps" -o md > "$out/bench_${n}_d${d}_${i}_$v.md" 2> "$out/bench_${n}_d${d}_${i}_$v.err"
                log "bench $v $n@d$d rc=$? $(grep -E '\| *(pp|tg)[0-9]+' "$out/bench_${n}_d${d}_${i}_$v.md" | awk -F'|' '{t = $(NF-2); gsub(/ /, "", t); split(t, a, "@"); v = $(NF-1); gsub(/^ +| +$/, "", v); printf "%s %s  ", a[1], v}')"
                sleep 10; done; done; done ;;
    profile)
        for v in old new; do for m in $models; do n=${m%.gguf}; tag=$([ $v = old ] && echo A || echo B)
            env $(envof $v) GGML_HEXAGON_PROFILE=1 timeout 1800 "$pkg/run.sh" llama-completion -m "$kit/models/$m" $opts $kv \
                -c 16384 -no-cnv --temp 0 -n 8 -lv 5 -f "$kit/prompts/prompt12k.txt" < /dev/null \
                > "$out/profile_${tag}_$n.log" 2>&1 &
            pid=$!
            if [ -n "${PROBE:-}" ]; then mkdir -p "$out/probe"; probe_loop "profile_${tag}_$n" $pid; fi
            wait $pid; rc=$?
            log "profile $v $n rc=$rc $(grep -ac 'profile-op FLASH_ATTN_EXT' "$out/profile_${tag}_$n.log") FA ops"
            gzip -f "$out/profile_${tag}_$n.log"
            sleep 5; done; done ;;
    *) log "unknown step $step"; exit 2 ;;
    esac
done
rm -f "$out"/.c_*
