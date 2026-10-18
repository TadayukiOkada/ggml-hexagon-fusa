#!/usr/bin/env python3
"""Per-op HTP time against a budget, from GGML_HEXAGON_PROFILE=1 logs.

Functional-safety firmware runs a periodic NSP self-test that needs VTCM and
HMX. A single op that holds them longer than the budget can make the self-test
miss its deadline (yield points are only between ops and inside some copies).
This groups the profile-op lines by op, shapes and types, and reports count,
mean, p95 and max time, flagging groups whose max exceeds the budget with !!.
Ops that release VTCM/HMX inside the op when built with
GGML_HEXAGON_HTP_YIELD_IN_CPY / _IN_OPS / _BETWEEN_OPS (--yielding, see YIELDING) are marked
~~ instead: their op time is not the time the resources are held, at least on
the paths that yield. For builds without those options pass --yielding "".

usage: scripts/op-budget.py [--budget-ms 25] [--top 30] [--fail] LOG...
       (reads stdin when no LOG is given)
"""

import argparse
import math
import re
import sys
from collections import defaultdict

# ggml-hex: HTP0 profile-op CONT|src_transposed -> out|768:1024 -> 768:1024|f32 -> f32|...|usec 10629 cycles ...
LINE = re.compile(
    r"(?P<dev>HTP\d+(?::\d+)?) profile-op (?P<op>[A-Z_0-9+]+)\|(?P<fields>.*)\|usec (?P<usec>\d+) cycles (?P<cycles>\d+)"
)


# Ops that release VTCM/HMX on request inside the op in builds with GGML_HEXAGON_HTP_YIELD_IN_CPY
# (CONT, CPY) and GGML_HEXAGON_HTP_YIELD_IN_OPS (flash attention, IM2COL, GET_ROWS, PAD, unary ops,
# SOFT_MAX, SET_ROWS, multi-row sorts, the HMX 2D matmul), plus fence waits with GGML_HEXAGON_HTP_YIELD_BETWEEN_OPS.
# "OP:tag" only counts the op with that tag: the kernel
# path for matmuls and flash attention (e.g. hmx-tiled), otherwise multi-row or single-row.
YIELDING = ["CONT", "CPY", "FENCE_WAIT", "FLASH_ATTN_EXT", "IM2COL", "GET_ROWS", "PAD", "MUL_MAT:hmx-tiled",
            "SOFT_MAX", "SET_ROWS", "ARGSORT:multi-row", "TOP_K:multi-row",
            "SCALE", "CLAMP", "LEAKY_RELU", "SQR", "SQRT", "NEG", "EXP", "SIGMOID", "SILU", "GELU",
            "GELU_QUICK", "GELU_ERF", "SOFTPLUS", "TANH", "ABS", "LOG", "RELU", "STEP",
            "NORM", "RMS_NORM", "RMS_NORM_MUL", "L2_NORM", "TRI"]


def parse(stream, groups):
    for line in stream:
        m = LINE.search(line)
        if not m or m["op"] == "OPBATCH":
            continue
        fields = m["fields"].split("|")
        dims = fields[1] if len(fields) > 1 else ""
        types = fields[2] if len(fields) > 2 else ""
        groups[(m["dev"], m["op"], dims, types, tag(m["op"], fields, dims))].append(int(m["usec"]))


def tag(op, fields, dims):
    """Kernel path of matmuls and flash attention, else whether the first source has several rows."""
    last = fields[-1] if len(fields) > 4 else ""
    if (op.startswith("MUL_MAT") or op == "FLASH_ATTN_EXT") and " vtcm " in last:
        return last.split(" vtcm ")[0]
    ne = [int(x) for x in re.findall(r"\d+", dims.split(" ")[0])]
    return "multi-row" if len(ne) > 1 and math.prod(ne[1:]) > 1 else "single-row"


def is_yielding(op, tg, yielding):
    return op in yielding or f"{op}:{tg}" in yielding


def p95(values):
    s = sorted(values)
    return s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--budget-ms", type=float, default=25.0,
                    help="per-op budget in ms (default 25, half of a 50 ms self-test period)")
    ap.add_argument("--top", type=int, default=30, help="rows to print, longest max first (0 = all)")
    ap.add_argument("--yielding", default=",".join(YIELDING),
                    help="ops that yield inside the op, marked ~~ and not counted for --fail "
                         "(default: those that do with GGML_HEXAGON_HTP_YIELD_IN_CPY, _IN_OPS and _BETWEEN_OPS)")
    ap.add_argument("--fail", action="store_true", help="exit with 1 when any other op exceeds the budget")
    args = ap.parse_args()

    groups = defaultdict(list)
    if args.logs:
        for path in args.logs:
            with open(path, errors="replace") as f:
                parse(f, groups)
    else:
        parse(sys.stdin, groups)

    if not groups:
        print("op-budget: no profile-op lines (run with GGML_HEXAGON_PROFILE=1)", file=sys.stderr)
        return 2

    budget_us = args.budget_ms * 1000.0
    rows = sorted(groups.items(), key=lambda kv: max(kv[1]), reverse=True)
    yielding = {o.strip() for o in args.yielding.split(",") if o.strip()}
    over = [r for r in rows if max(r[1]) > budget_us and not is_yielding(r[0][1], r[0][4], yielding)]
    over_y = [r for r in rows if max(r[1]) > budget_us and is_yielding(r[0][1], r[0][4], yielding)]

    print(f"{'':2}{'max ms':>9} {'p95 ms':>9} {'mean ms':>9} {'count':>7}  op  dims  types  tag")
    for (dev, op, dims, types, tg), us in rows[: args.top or None]:
        flag = ("~~" if is_yielding(op, tg, yielding) else "!!") if max(us) > budget_us else ""
        print(f"{flag:2}{max(us) / 1000:9.2f} {p95(us) / 1000:9.2f} {sum(us) / len(us) / 1000:9.2f} {len(us):7d}  "
              f"{dev} {op}  {dims}  {types}  {tg}")
    print(f"\n{len(over)} of {len(rows)} op groups exceed {args.budget_ms:g} ms"
          f" (plus {len(over_y)} of the yielding ops {','.join(sorted(yielding))})")
    return 1 if args.fail and over else 0


if __name__ == "__main__":
    sys.exit(main())
