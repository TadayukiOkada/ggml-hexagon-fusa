#!/usr/bin/env python3
"""Balanced cache-type benchmark using one upstream tool/backend session.

Calls libllama-bench-impl's unmodified entry point and saves each call's stdout,
stderr, command and library hash. F16 / forward matrix / reversed matrix / F16
order balances cache types over time. Use an outer process timeout.
"""
import argparse
import json
import os
from pathlib import Path
import time

from eval_kv import InProcessTool, sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="HTP0")
    ap.add_argument("--gpu-layers", type=int, default=99)
    ap.add_argument("--key-type", default="q8_0", help="K type of the matrix between the F16 reference runs")
    ap.add_argument("--values", nargs="+", default=["q8_0", "q5_0", "q4_0", "q4_1", "iq4_nl", "f8_e4m3"])
    ap.add_argument("--depths", nargs="+", type=int, default=[4096, 12288])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=32)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tool = InProcessTool(args.library, "_Z11llama_benchiPPc")
    meta = {"library_sha256": sha256(args.library), "evaluator_sha256": sha256(__file__)}
    for model in args.models:
        model_hash = sha256(model)
        for order in (1, 2, 3, 4):
            values = ["f16"] if order in (1, 4) else args.values if order == 2 else list(reversed(args.values))
            k = "f16" if order in (1, 4) else args.key_type
            tag = Path(model).stem + f"-{order}"
            cmd = ["llama-bench", "-m", model, "-dev", args.device, "-ngl", str(args.gpu_layers), "-fa", "on",
                   "-ctk", k, "-ctv", ",".join(values), "-p", str(args.prompt), "-n", str(args.tokens),
                   "-d", ",".join(map(str, args.depths)), "-r", str(args.reps), "-o", "json", "--progress"]
            t0 = time.monotonic()
            rc = tool(cmd, dict(os.environ), out / (tag + ".json"), stderr_path=out / (tag + ".err"))
            record = dict(meta, model_sha256=model_hash, cmd=cmd, rc=rc, seconds=time.monotonic() - t0)
            (out / (tag + ".meta.json")).write_text(json.dumps(record, indent=2))
            if rc != 0:
                raise RuntimeError(f"benchmark failed rc={rc}: {tag}")
            rows = json.loads((out / (tag + ".json")).read_text())
            expected = len(values) * len(args.depths) * (int(args.prompt > 0) + int(args.tokens > 0))
            if len(rows) != expected or not all(row.get("avg_ts", 0) > 0 for row in rows):
                raise RuntimeError(f"incomplete/invalid benchmark: {tag}, expected {expected} rows")
            print(f"{tag}: {len(rows)} cases, {record['seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
