#!/usr/bin/env python3
"""Compare KV cache types against an F16 cache of the same model/backend.

Saves the F16 logits once, then runs llama-perplexity with --kl-divergence for each
K/V pair, with the attention rotation on and off. One JSON record per run.

usage: quant/eval_kv.py --tool bin/llama-perplexity --model M.gguf --text wiki.test.raw \\
           --out DIR [--chunks 128] [--threads 8] [--pairs q8_0:q8_0 q4_0:q4_0] [--no-rotation-runs]
"""
import argparse
import ctypes
from contextlib import ExitStack
import hashlib
import json
import math
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


class InProcessTool:
    """Call an unmodified tool entry point while retaining the backend registry.

    The Linux shared tool's C++ entry point has an int/char** ABI. This mode is
    sequential and version-pinned; an absent entry/flush symbol fails immediately.
    It avoids opening a new FastRPC session for every cache pair. Environment paths
    (LD_LIBRARY_PATH, ADSP_LIBRARY_PATH) must be set before starting Python.
    """
    def __init__(self, path, entry_symbol):
        path = os.path.abspath(path)
        self.lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        self.entry = getattr(self.lib, entry_symbol)
        self.entry.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.entry.restype = ctypes.c_int
        # Resolve through this dependency graph. Packaged .so/.so.0 names can be
        # distinct files; dlopening both would create a second logger/registry.
        self.common = self.lib
        mainlog = getattr(self.common, "_Z15common_log_mainv")
        mainlog.restype = ctypes.c_void_p
        self.log = mainlog()
        self.flush = getattr(self.common, "_Z16common_log_flushP10common_log")
        self.flush.argtypes = [ctypes.c_void_p]
        self.flush.restype = None
        self.libc = ctypes.CDLL(None)
        self.libc.fflush.argtypes = [ctypes.c_void_p]
        backend = self.lib
        backend.ggml_backend_reg_count.restype = ctypes.c_size_t
        backend.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
        if backend.ggml_backend_reg_count() == 0:
            backend.ggml_backend_load_all_from_path(os.path.dirname(path).encode())

    def __call__(self, cmd, env, log_path, stderr_path=None):
        self.flush(self.log)
        self.libc.fflush(None)
        sys.stdout.flush()
        sys.stderr.flush()
        saved = [os.dup(1), os.dup(2)]
        old_rotation = os.environ.get("LLAMA_ATTN_ROT_DISABLE")
        try:
            if "LLAMA_ATTN_ROT_DISABLE" in env:
                os.environ["LLAMA_ATTN_ROT_DISABLE"] = env["LLAMA_ATTN_ROT_DISABLE"]
            else:
                os.environ.pop("LLAMA_ATTN_ROT_DISABLE", None)
            with ExitStack() as stack:
                log = stack.enter_context(open(log_path, "w"))
                err = stack.enter_context(open(stderr_path, "w")) if stderr_path else log
                os.dup2(log.fileno(), 1)
                os.dup2(err.fileno(), 2)
                argv = (ctypes.c_char_p * (len(cmd) + 1))(*[os.fsencode(x) for x in cmd], None)
                rc = self.entry(len(cmd), argv)
                self.flush(self.log)
                self.libc.fflush(None)
            return rc
        finally:
            for fd, original in zip((1, 2), saved):
                os.dup2(original, fd)
                os.close(original)
            if old_rotation is None:
                os.environ.pop("LLAMA_ATTN_ROT_DISABLE", None)
            else:
                os.environ["LLAMA_ATTN_ROT_DISABLE"] = old_rotation


class InProcessPerplexity(InProcessTool):
    def __init__(self, path):
        super().__init__(path, "_Z16llama_perplexityiPPc")


def run(cmd, env, log_path, timeout, in_process=None):
    start = time.time()
    if in_process is not None:
        rc = in_process(cmd, env, log_path)
    else:
        with open(log_path, "w") as log:
            try:
                rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                rc = 124
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
    ap.add_argument("--gpu-layers", type=int, default=0)
    ap.add_argument("--device", help="e.g. HTP0; omitted for the default CPU evaluation")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per process")
    ap.add_argument("--null", action="store_true", help="include F16:F16 control against the saved F16 logits")
    ap.add_argument("--build-info", help="packaged BUILD_INFO.txt for backend/build provenance")
    ap.add_argument("--in-process-library", help="Linux libllama-perplexity-impl.so; one registry/session across pairs; use an outer process timeout")
    ap.add_argument("--log-verbosity", type=int, help="upstream log verbosity, e.g. 5 for cache sizes and rotation metadata")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base_kld = os.path.join(args.out, "base.kld")
    common = [args.tool, "-m", args.model, "-f", args.text, "--chunks", str(args.chunks),
              "-c", str(args.ctx), "-t", str(args.threads), "-ngl", str(args.gpu_layers), "-fa", "on"]
    if args.device:
        common += ["-dev", args.device]
    if args.log_verbosity is not None:
        common += ["-lv", str(args.log_verbosity)]
    meta = {"model_sha256": sha256(args.model), "model": os.path.basename(args.model),
            "text_sha256": sha256(args.text), "chunks": args.chunks, "ctx": args.ctx,
            "env": args.device or ("cpu" if args.gpu_layers == 0 else "default-gpu"),
            "gpu_layers": args.gpu_layers, "tool_sha256": sha256(args.tool),
            "evaluator_sha256": sha256(__file__)}
    if args.build_info:
        with open(args.build_info) as f:
            meta["build_info"] = f.read()
        meta["build_info_sha256"] = sha256(args.build_info)
    in_process = InProcessPerplexity(args.in_process_library) if args.in_process_library else None
    if args.in_process_library:
        meta["in_process_library_sha256"] = sha256(args.in_process_library)

    env = dict(os.environ)
    env.pop("LLAMA_ATTN_ROT_DISABLE", None)
    rc, log, secs = run(common + ["-ctk", "f16", "-ctv", "f16", "--kl-divergence-base", base_kld],
                        env, os.path.join(args.out, "base.log"), args.timeout, in_process)
    if rc != 0:
        sys.exit(f"base run failed, rc {rc}: see {args.out}/base.log")
    # "Mean PPL(base)" of a --kl-divergence run comes from the saved log probabilities, which are
    # clipped 16 below the top logit, so it reads low. Compare PPL with this estimate instead.
    m = re.search(r"Final estimate: PPL = " + NUM + r" \+/- " + NUM, log)
    base_final = {"ppl_base_final": float(m.group(1)), "ppl_base_final_unc": float(m.group(2))} if m else {}
    if not base_final or not math.isfinite(base_final["ppl_base_final"]) or not os.path.isfile(base_kld):
        sys.exit(f"base run has no finite PPL or saved logits: see {args.out}/base.log")

    records = 0
    failed = 0
    for pair in (["f16:f16"] if args.null else []) + args.pairs:
        k, v = pair.split(":")
        for rotation in ("hadamard", "none")[: 1 if args.no_rotation_runs else 2]:
            run_env = dict(env)
            if rotation == "none":
                run_env["LLAMA_ATTN_ROT_DISABLE"] = "1"
            name = f"k_{k}_v_{v}_rot_{rotation}"
            cmd = common + ["-ctk", k, "-ctv", v, "--kl-divergence-base", base_kld, "--kl-divergence"]
            rc, log, secs = run(cmd, run_env, os.path.join(args.out, name + ".log"), args.timeout, in_process)
            stats = dict(parse(log), **base_final)
            ok = rc == 0 and all(k in stats and math.isfinite(stats[k]) for k in ("kld", "kld_unc", "ppl"))
            if "ppl_base_final" in stats and "ppl" in stats:
                stats["ppl_change_pct"] = 100 * (stats["ppl"] / stats["ppl_base_final"] - 1)
                stats["quality_gate"] = bool(ok and stats["kld"] + 2 * stats.get("kld_unc", 0) < 0.01
                                             and stats["ppl_change_pct"] <= 1)
            # A small F16 null error measures HTP arithmetic + baseline-logit encoding,
            # rather than cache quantization. Record it, do not subtract it silently.
            rotations = re.findall(r"n_rot_([kv]) = (\d+)", log)
            failed += not ok
            record = dict(meta, kv={"k": k, "v": v, "rotation": rotation}, rc=rc, ok=ok,
                          seconds=round(secs, 1), cmd=cmd, results=stats, actual_rotation=dict(rotations))
            with open(os.path.join(args.out, name + ".json"), "w") as f:
                json.dump(record, f, indent=1)
            records += 1
            print(f"{name}: rc={rc} {stats}", flush=True)
            if re.search(r"(?:buffer mapping failed|fastrpc_mmap failed)", log):
                sys.exit("DSP mapping failed; stop without launching more processes; baseline retained")

    if not args.keep_logits:
        os.remove(base_kld)
    print(f"{records} records, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
