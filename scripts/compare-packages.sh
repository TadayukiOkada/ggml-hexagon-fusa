#!/bin/sh
# Compare a new package with an older one on the target, step by step, on a fixed set of models,
# audio and prompts (the test kit, see tests/test-kit.txt). POSIX sh; nothing is killed with SIGKILL.
#
# usage: scripts/compare-packages.sh <new pkg> <old pkg | -> <kit dir> <out dir> <step>...
#
#   leak     test-hexagon-metadata on HTP0 (new only)
#   llm      llama-completion, 64 tokens greedy, Qwen3-0.6B and Qwen3.5-0.8B: text and speed, old vs new
#   bench    llama-bench pp512 / tg128 on the same two models, old vs new
#   prefill  Qwen3-0.6B, a 12.7k-token prompt, flash attention on and off (new only)
#   asr      run-asr.sh (streaming) on HTP0 and HTP1, old vs new, and a 60 s clip (new only)
#   sv       run-sensevoice.sh --json on HTP0 and HTP1, 4 inputs, old vs new
#   servers  nemo-speech serve and sensevoice-server on the same HTP0, back-to-back requests
#            checked with scripts/rss-soak.py (new only)
#
# Packages are made by scripts/package.sh (flat or --split; both have run.sh, run-asr.sh and
# run-sensevoice.sh). "-" as the old package skips the old runs. Set ADSP_LIBRARY_PATH as the
# target needs (run.sh puts the package first). Each line of <out>/summary.txt is one result;
# "DIFFERENT" marks an output that differs from the old package's.
set -u

new=$(cd "${1:?usage: $0 <new pkg> <old pkg|-> <kit> <out> <step>...}" && pwd)
old=${2:?}; [ "$old" = - ] || old=$(cd "$old" && pwd)
kit=$(cd "${3:?}" && pwd)
out=${4:?}; shift 4
mkdir -p "$out"; out=$(cd "$out" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
summary=$out/summary.txt
M=$kit/models A=$kit/audio
Q3=$M/Qwen3-0.6B-Q4_0.gguf Q35=$M/Qwen3.5-0.8B-Q4_0.gguf
NM=$M/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf SM=$M/sensevoice-small-q8.gguf

log() { echo "$*" | tee -a "$summary"; }
md5() { md5sum < "$1" | cut -c1-12; }
asr_text() { python3 -c "import json,sys,hashlib; t=json.load(open(sys.argv[1])).get('text',''); print(hashlib.md5(t.encode()).hexdigest()[:12])" "$1" 2>/dev/null || echo none; }
versions() { if [ "$old" = - ]; then echo new; else echo old new; fi; }
pkg() { if [ "$1" = old ]; then echo "$old"; else echo "$new"; fi; }
same() { # name value: compare with the old package's value of the same name
    if [ "$old" = - ] || [ "${3:-}" = old ]; then echo "$2" > "$out/.$1"; echo ""; return; fi
    if [ "$(cat "$out/.$1" 2>/dev/null)" = "$2" ]; then echo same; else echo DIFFERENT; fi
}
ready() { python3 - "$@" <<'PY'
import sys, time, urllib.request
for _ in range(180):
    try:
        for u in sys.argv[1:]: urllib.request.urlopen(u, timeout=2).read()
        sys.exit(0)
    except Exception: time.sleep(1)
sys.exit(1)
PY
}

log "new: $(head -1 "$new/BUILD_INFO.txt")"
[ "$old" = - ] || log "old: $(head -1 "$old/BUILD_INFO.txt")"
for step in "$@"; do
    log "## $step $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    case $step in
    leak)
        timeout 120 "$new/run.sh" test-hexagon-metadata HTP0 64 > "$out/leak.log" 2>&1
        log "leak rc=$? $(tail -1 "$out/leak.log")" ;;
    llm)
        for v in $(versions); do for m in Q3 Q35; do eval f=\$$m
            timeout 180 "$(pkg $v)/run.sh" llama-completion -m "$f" -ngl 99 -dev HTP0 -no-cnv --temp 0 -s 1 -n 64 \
                -p "The capital of France is" -lv 5 < /dev/null > "$out/llm_${v}_$m.out" 2> "$out/llm_${v}_$m.err"
            rc=$?; h=$(md5 "$out/llm_${v}_$m.out")
            log "llm $v $m rc=$rc md5=$h $(same llm_$m "$h" $v) $(grep -a ' eval time' "$out/llm_${v}_$m.err" | grep -v 'prompt eval' | tail -1 | sed 's/.*(//; s/)//')"
            sleep 5; done; done ;;
    bench)
        for v in $(versions); do for m in Q3 Q35; do eval f=\$$m
            timeout 900 "$(pkg $v)/run.sh" llama-bench -m "$f" -p 512 -n 128 -r 3 -dev HTP0 -ngl 99 -o md \
                > "$out/bench_${v}_$m.md" 2> "$out/bench_${v}_$m.err"
            log "bench $v $m rc=$? $(grep -E '\| *(pp|tg)[0-9]+ ' "$out/bench_${v}_$m.md" | awk -F'|' '{gsub(/ /,"",$8); gsub(/^ +| +$/,"",$9); printf "%s %s  ", $8, $9}')"
            sleep 10; done; done ;;
    prefill)
        for fa in on off; do
            timeout 900 "$new/run.sh" llama-completion -m "$Q3" -ngl 99 -dev HTP0 -no-cnv --temp 0 -n 4 -c 13312 -fa $fa \
                -f "$kit/prompts/prompt12k.txt" < /dev/null > "$out/prefill_$fa.out" 2> "$out/prefill_$fa.err"
            log "prefill fa=$fa rc=$? $(grep -a 'prompt eval time' "$out/prefill_$fa.err" | tail -1 | sed 's/.*= *//')"
            sleep 10; done ;;
    asr)
        for v in $(versions); do for d in HTP0 HTP1; do for w in ja_second input; do
            ASR_DEVICE=$d timeout 300 "$(pkg $v)/run-asr.sh" "$NM" "$A/$w.wav" --language ja-JP --json \
                > "$out/asr_${v}_${d}_$w.json" 2> "$out/asr_${v}_${d}_$w.err"
            rc=$?; h=$(asr_text "$out/asr_${v}_${d}_$w.json")
            log "asr $v $d $w rc=$rc text=$h $(same asr_${d}_$w "$h" $v)"; sleep 3; done; done; done
        ASR_DEVICE=HTP0 timeout 600 "$new/run-asr.sh" "$NM" "$A/l060.wav" --language ja-JP --json \
            > "$out/asr_new_HTP0_l060.json" 2> "$out/asr_new_HTP0_l060.err"
        log "asr new HTP0 l060 rc=$? text=$(asr_text "$out/asr_new_HTP0_l060.json")" ;;
    sv)
        for v in $(versions); do for d in HTP0 HTP1; do for n in input input30 browser-captured short100; do
            SENSEVOICE_DEVICE=$d timeout 120 "$(pkg $v)/run-sensevoice.sh" "$SM" "$A/$n.wav" --json --threads 4 \
                > "$out/sv_${v}_${d}_$n.json" 2> "$out/sv_${v}_${d}_$n.err"
            rc=$?; h=$(md5 "$out/sv_${v}_${d}_$n.json")
            log "sv $v $d $n rc=$rc md5=$h $(same sv_${d}_$n "$h" $v)"; sleep 2; done; done; done ;;
    servers)
        hex=$new/bin/libggml-hexagon.so; [ -f "$hex" ] || hex=$new/libggml-hexagon.so
        cat > "$out/nemo.yaml" <<YAML
http:
  enabled: true
  host: 127.0.0.1
  port: 18080
  threads: 4
  max-upload-mb: 16
  playground: false
asr:
  enabled: true
  backend:
    name: hexagon
    library_path: $hex
    device: HTP0
    frontend: cpu
    position_backend: cpu
    diagnostics: false
    threads: 4
  batching:
    enabled: false
    state_arena_slots: 1
  streaming:
    rnnt_right_context: 3
YAML
        GGML_HEXAGON_DEVICES=HTP0 "$new/run.sh" nemo-speech serve --config "$out/nemo.yaml" --asr-model "$NM" --no-warmup > "$out/nemo_server.log" 2>&1 < /dev/null &
        npid=$!
        GGML_HEXAGON_DEVICES=HTP0 "$new/run.sh" sensevoice-server -m "$SM" --backend hexagon --backend-lib "$hex" --device HTP0 \
            --threads 4 --port 18081 > "$out/sv_server.log" 2>&1 < /dev/null &
        spid=$!
        ready http://127.0.0.1:18080/v1/models http://127.0.0.1:18081/ready; log "servers ready rc=$?"
        timeout 1500 python3 "$here/rss-soak.py" --pid $npid --url http://127.0.0.1:18080/v1/audio/transcriptions --multipart \
            --field language=ja-JP --field response_format=json -n 24 --warmup 4 --max-kib 1000000 --csv "$out/soak_nemo.csv" \
            "$A/s60_0.wav" "$A/s60_1.wav" > "$out/soak_nemo.txt" 2>&1 &
        n1=$!
        timeout 1500 python3 "$here/rss-soak.py" --pid $spid --url http://127.0.0.1:18081/api/transcribe --pcm -n 600 --warmup 100 \
            --csv "$out/soak_sv.csv" "$A/s30_0.wav" "$A/s30_1.wav" "$A/s30_2.wav" "$A/s30_3.wav" > "$out/soak_sv.txt" 2>&1
        log "sv soak rc=$? $(head -1 "$out/soak_sv.txt")"
        wait $n1; log "nemo soak rc=$? $(head -1 "$out/soak_nemo.txt") (RSS of NeMo is recorded, not gated)"
        kill -TERM $npid $spid; i=0
        while [ $i -lt 30 ] && { kill -0 $npid 2> /dev/null || kill -0 $spid 2> /dev/null; }; do sleep 1; i=$((i + 1)); done
        log "servers stopped after ${i}s alive=$(kill -0 $npid 2> /dev/null && echo nemo)$(kill -0 $spid 2> /dev/null && echo sv)" ;;
    *) log "unknown step $step"; exit 2 ;;
    esac
done
rm -f "$out"/.llm_* "$out"/.asr_* "$out"/.sv_*
