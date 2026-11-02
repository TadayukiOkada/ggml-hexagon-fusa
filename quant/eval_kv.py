#!/usr/bin/env python3
"""Compare KV cache types against an F16 cache of the same model, on the CPU.

Saves the F16 logits once, then runs llama-perplexity with --kl-divergence for each
K/V pair, with the attention rotation on and off. One JSON record per run.

usage: quant/eval_kv.py --tool bin/llama-perplexity --model M.gguf --text wiki.test.raw \\
           --out DIR [--chunks 128] [--threads 8] [--pairs q8_0:q8_0 q4_0:q4_0] [--no-rotation-runs]
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

PAIRS = ["q8_0:q8_0", "q5_1:q5_1", "q5_0:q5_0", "q4_1:q4_1", "q4_0:q4_0",
         "q8_0:q5_0", "q8_0:q4_0", "q5_0:q4_0"]

NUM = r"([-+0-9.eE]+|nan|-?inf)"
STATS = {
    "ppl": rf"Mean PPL\(Q\)\s*:\s*{NUM}\s*±\s*{NUM}",
    "ppl_base": rf"Mean PPL\(base\)\s*:\s*{NUM}\s*±\s*{NUM}",
    "ln_ppl_ratio": rf"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*{NUM}\s*±\s*{NUM}",
    "kld": rf"Mean\s+KLD:\s*{NUM}\s*±\s*{NUM}",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse(log):
    out = {}
    for name, pattern in STATS.items():
        m = re.search(pattern, log)
        if m:
            out[name], out[name + "_unc"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"Maximum KLD:\s*" + NUM, log)
    if m:
        out["kld_max"] = float(m.group(1))
    return out


def run(cmd, env, log_path):
    start = time.time()
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    with open(log_path, errors="replace") as f:
        return rc, f.read(), time.time() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--pairs", nargs="*", default=PAIRS)
    ap.add_argument("--no-rotation-runs", action="store_true", help="skip the rotation-off runs")
    ap.add_argument("--keep-logits", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base_kld = os.path.join(args.out, "base.kld")
    common = [args.tool, "-m", args.model, "-f", args.text, "--chunks", str(args.chunks),
              "-c", str(args.ctx), "-t", str(args.threads), "-ngl", "0", "-fa", "on"]
    meta = {"model_sha256": sha256(args.model), "model": os.path.basename(args.model),
            "text_sha256": sha256(args.text), "chunks": args.chunks, "ctx": args.ctx,
            "env": "host-cpu"}

    env = dict(os.environ)
    env.pop("LLAMA_ATTN_ROT_DISABLE", None)
    rc, log, secs = run(common + ["-ctk", "f16", "-ctv", "f16", "--kl-divergence-base", base_kld],
                        env, os.path.join(args.out, "base.log"))
    if rc != 0:
        sys.exit(f"base run failed, rc {rc}: see {args.out}/base.log")

    records = 0
    failed = 0
    for pair in args.pairs:
        k, v = pair.split(":")
        for rotation in ("hadamard", "none")[: 1 if args.no_rotation_runs else 2]:
            run_env = dict(env)
            if rotation == "none":
                run_env["LLAMA_ATTN_ROT_DISABLE"] = "1"
            name = f"k_{k}_v_{v}_rot_{rotation}"
            cmd = common + ["-ctk", k, "-ctv", v, "--kl-divergence-base", base_kld, "--kl-divergence"]
            rc, log, secs = run(cmd, run_env, os.path.join(args.out, name + ".log"))
            stats = parse(log)
            ok = rc == 0 and "kld" in stats
            failed += not ok
            record = dict(meta, kv={"k": k, "v": v, "rotation": rotation}, rc=rc, ok=ok,
                          seconds=round(secs, 1), cmd=" ".join(cmd), results=stats)
            with open(os.path.join(args.out, name + ".json"), "w") as f:
                json.dump(record, f, indent=1)
            records += 1
            print(f"{name}: rc={rc} {stats}", flush=True)

    if not args.keep_logits:
        os.remove(base_kld)
    print(f"{records} records, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
