#!/usr/bin/env python3
"""Build/run the KV tile and cache-write oracles on the v73 ISA simulator.

SDK/tool locations are explicit arguments, never embedded in the repository.
The compile database is from the HTP ExternalProject, not the ARM host build.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import random
import shlex
import struct
import subprocess


def golden(repo, out, cc):
    libpath = out / "quants-ref.so"
    subprocess.run([cc, "-O2", "-DNDEBUG", "-fPIC", "-shared",
                    str(repo / "llama.cpp/ggml/src/ggml-quants.c"),
                    "-I" + str(repo / "llama.cpp/ggml/include"),
                    "-I" + str(repo / "llama.cpp/ggml/src"), "-o", str(libpath), "-lm"], check=True)
    # Unused quantizers reference ggml logging/type helpers. Lazy binding leaves
    # those unused functions unresolved; the six reference converters are standalone.
    lib = ctypes.CDLL(str(libpath), mode=os.RTLD_LAZY)
    rng = random.Random(717)
    blob, count = bytearray(), 0
    for name, typ, size in [("q4_0",2,18),("q4_1",3,20),("q5_0",6,22),
                            ("q5_1",7,24),("f8_e4m3",43,32),("iq4_nl",20,18)]:
        fn = getattr(lib, "quantize_row_" + name + "_ref")
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        for n in (32,64,96,128,256,576):
            for mode in range(12):
                scale = (0,1e-8,1e-4,.1,1,10,100)[mode % 7]
                vals = [rng.uniform(-scale, scale) for _ in range(n)]
                if mode == 7: vals = [1.0] * n
                if mode == 8: vals = [-.25] * n
                if mode == 9: vals = [(-1 if i % 2 else 1) * 2 for i in range(n)]
                if mode == 10: vals = [(i % 32 - 16) / 16 for i in range(n)]
                if mode == 11: vals = [(-1 if i % 2 else 1) * 2 ** ((-i) % 23) for i in range(n)]
                src = (ctypes.c_float * n)(*vals)
                dst = (ctypes.c_uint8 * (n // 32 * size))()
                fn(src, dst, n)
                blob += struct.pack("<III", typ, n, len(dst)) + bytes(src) + bytes(dst)
                count += 1
    (out / "kv-write-golden.bin").write_bytes(struct.pack("<I", count) + blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdk", required=True, type=Path)
    ap.add_argument("--tools", required=True, type=Path, help="HEXAGON_Tools version directory (contains Tools/)")
    ap.add_argument("--compile-commands", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--host-cc", default="cc")
    ap.add_argument("--exhaustive", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    out, sdk = args.out.resolve(), args.sdk.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tools = args.tools.resolve() / "Tools"
    db = json.loads(args.compile_commands.read_text())
    entry = next(x for x in db if x["file"].endswith("/flash-attn-ops.c"))
    flags = entry.get("arguments") or shlex.split(entry["command"])
    flags = [x for x in flags if x != "-flto"]
    golden(repo, out, args.host_cc)
    (out / "q6ss.cfg").write_text(f"{tools}/lib/iss/qtimer.so --csr_base=0xFC900000 --irq_p=1 --freq=19200000 --cnttid=1\n{tools}/lib/iss/l2vic.so 32 0xab010000\n")
    (out / "osam.cfg").write_text(f"{sdk}/rtos/qurt/computev73/debugger/lnx64/qurt_model.so\n")
    runner = next((sdk / "libs/run_main_on_hexagon/ship").glob("hexagon_*_v73/run_main_on_hexagon_sim"))
    for name in ("test-kv-prep", "test-kv-write"):
        obj, so = out / (name + ".o"), out / (name + ".so")
        cmd = list(flags)
        cmd[cmd.index("-o") + 1] = str(obj)
        cmd[cmd.index("-c") + 1] = str(Path(__file__).with_name(name + ".c"))
        cmd += ["-Wno-unused-function", "-Wno-unused-variable"]
        if args.exhaustive:
            cmd += ["-DKV_PREP_EXHAUSTIVE=1"]
        subprocess.run(cmd, check=True)
        subprocess.run([str(tools / "bin/hexagon-clang"), "-mv73", "-shared", "-G0", "-fpic",
                        "-Wl,-Bsymbolic", "-Wl,--no-threads", str(obj), "-o", str(so), "-lc"], check=True)
        cmd = [str(tools / "bin/hexagon-sim"), "-mv73na_1", "--simulated_returnval", "--usefs", str(out),
               "--pmu_statsfile", str(out / (name + ".pmu.txt")), "--cosim_file", str(out / "q6ss.cfg"),
               "--l2tcm_base", "0xd800", "--rtos", str(out / "osam.cfg"),
               str(sdk / "rtos/qurt/computev73/sdksim_bin/runelf.pbn"), "--", str(runner), "--" + str(so), "100"]
        (out / (name + ".command.json")).write_text(json.dumps(cmd, indent=2))
        with (out / (name + ".log")).open("w") as log:
            subprocess.run(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=True)
        if "PASS " not in (out / (name + ".log")).read_text(errors="replace"):
            raise RuntimeError(f"{name}: simulator exited without a PASS marker")
        print(name + " PASS", flush=True)


if __name__ == "__main__":
    main()
