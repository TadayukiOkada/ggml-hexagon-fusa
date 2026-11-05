#!/usr/bin/env python3
"""PASS / REVIEW / FAIL for the target test runs of an upstream import.

  --ops DIR       device-tests.sh --ops output. Every op listed by the support run (ops.txt) must
                  have exactly one result row with a pass count; every failing case must match
                  tests/known-failures.txt (op, case regex, largest error); an op that fails without
                  a list of failing cases (crash, timeout, signal) fails. --leak in the same summary
                  must pass. An empty or partial summary fails.
  --long DIR      device-tests.sh --long output: every long step rc=0 with a log that has profile
                  lines, then scripts/op-budget.py --fail on long_*.log (no op over the budget that
                  does not yield inside).
  --compare F     compare-packages.sh summary.txt. Each "## step" must have all its result lines
                  (counts below), each with rc=0; a server left running fails; llama-bench new below
                  old by more than --speed-drop fails (the means when a pair was run several times,
                  e.g. old/new/new/old); an output that differs from the old package's is REVIEW.
  --probe-dir D   tools/vtcmlat CSVs (header t_us,lat_us) under D: every CSV must have samples,
                  no failed (negative) sample and a maximum below --probe-ms (the FuSa self-test
                  stops the DSP at 50 ms); its .log, when present, must report no failure. No CSV
                  at all fails.
  --probe F       older summary lines "probe <name> <n> samples, max=<us> us, timeouts=<n>".
  --accept F      reviewed differences: lines "<regex> <reason>"; a REVIEW finding that a regex
                  matches becomes "accepted" (printed with the reason). Only for decisions a reviewer
                  recorded for this release; a FAIL is never accepted. Unused lines are reported.
  --json OUT      also write the findings as JSON (level, findings with kind/check/message).

Exit 0 PASS, 3 REVIEW, 1 FAIL. Prints one line per finding.

usage: scripts/verdict.py [--ops DIR] [--long DIR] [--compare F]... [--probe-dir D]... [--probe F]...
                          [--known tests/known-failures.txt] [--speed-drop 0.03] [--probe-ms 25]
                          [--json OUT]
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RC = re.compile(r"\brc=(-?\d+)\b")

# Result lines of one compare-packages.sh step, per package version run (old and new, or new
# only), plus lines for the new package only.
COMPARE_LINES = {
    "leak":    (0, 1),
    "llm":     (2, 0),
    "bench":   (2, 0),
    "bench-abba": (4, 0),
    "prefill": (0, 2),
    "asr":     (4, 1),
    "sv":      (8, 0),
    "servers": (0, 4),
    "asr300":  (0, 1),
}


def load_known(path):
    known = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        op, pattern, limit, reason = line.split(None, 3)
        known.append({"op": op, "re": re.compile(pattern), "limit": None if limit == "any" else float(limit),
                      "reason": reason, "hits": 0})
    return known


def failing_cases(log):
    """(listed failing cases or None, {case: error}); error is inf for mismatches without a number."""
    text = ANSI.sub("", open(log, errors="replace").read())
    errs = {}
    for m in re.finditer(r"^\[[A-Z0-9_]+\] (.*?)\s+([A-Z0-9_]+\(.*\)): FAIL$", text, re.M):
        e = re.match(r"ERR = ([0-9.eE+-]+)", m.group(1))
        errs[m.group(2)] = float(e.group(1)) if e else math.inf
    listed = None
    if "Failing tests:" in text:
        tail = text.split("Failing tests:", 1)[1]
        listed = [l.strip() for l in tail.splitlines() if re.match(r"^  [A-Z0-9_]+\(", l)]
    return listed, errs


def rc_of(line):
    m = RC.search(line)
    return int(m.group(1)) if m else None


def check_ops(d, known, out):
    summary = os.path.join(d, "summary.txt")
    if not os.path.isfile(summary):
        out.fail(f"ops: no {summary}")
        return
    rows = [l.rstrip("\n") for l in open(summary, errors="replace") if l.startswith(("test ", "metadata "))]
    ops_txt = os.path.join(d, "ops.txt")
    expected = [l.strip() for l in open(ops_txt)] if os.path.isfile(ops_txt) else []
    expected = [o for o in expected if o]
    if not expected:
        out.fail(f"ops: no supported ops listed in {ops_txt} (support run failed?)")
    seen = {}
    for row in rows:
        if row.startswith("metadata "):
            if rc_of(row) != 0:
                out.fail(f"ops: {row.strip()}")
            continue
        op = row.split()[1]
        seen[op] = seen.get(op, 0) + 1
        rc = rc_of(row)
        if rc is None:
            out.fail(f"ops: {op} without rc: {row}")
            continue
        if rc == 0:
            if not re.search(r"\b\d+/\d+ tests passed", row):
                out.fail(f"ops: {op} rc=0 without a pass count (did the test run?): {row}")
            continue
        if rc < 0 or rc > 128 or rc in (124, 126, 127):
            out.fail(f"ops: {op} rc={rc} (signal, timeout or not runnable)")
            continue
        listed, errs = failing_cases(os.path.join(d, f"test_{op}.log"))
        if not listed:
            out.fail(f"ops: {op} rc={rc} without a list of failing cases (crash or timeout)")
            continue
        for case in listed:
            err = errs.get(case, math.inf)
            k = next((k for k in known if k["op"] == op and k["re"].search(case)
                      and (k["limit"] is None or err <= k["limit"])), None)
            if k:
                k["hits"] += 1
            else:
                out.fail(f"ops: {case} ERR {err:g} not in the known failures")
    for op, n in sorted(seen.items()):
        if n > 1:
            out.fail(f"ops: {op} has {n} result rows (mixed runs in one directory?)")
    missing = [o for o in expected if o not in seen]
    if missing:
        out.fail(f"ops: {len(missing)} of {len(expected)} supported ops have no result: {' '.join(missing[:10])}")
    extra = [o for o in seen if expected and o not in expected]
    if extra:
        out.fail(f"ops: results for ops the support run did not list: {' '.join(extra[:10])}")
    for k in known:
        if k["hits"]:
            out.note(f"ops: known {k['op']} {k['re'].pattern}: {k['hits']} case(s) ({k['reason']})")
        elif k["op"] in seen:
            out.note(f"ops: known {k['op']} {k['re'].pattern} no longer fails: remove it from the list?")
    out.note(f"ops: {len(seen)} ops checked")


def check_long(d, out):
    summary = os.path.join(d, "summary.txt")
    steps = [l for l in open(summary, errors="replace") if l.startswith("long ")] if os.path.isfile(summary) else []
    if not steps:
        out.fail(f"long: no long results in {summary}")
    for line in steps:
        name, rc = line.split()[1], rc_of(line)
        if rc != 0:
            out.fail(f"long: {line.strip()}")
        log = os.path.join(d, f"long_{name}.log")
        text = ANSI.sub("", open(log, errors="replace").read()) if os.path.isfile(log) else ""
        if "backends passed" not in text:
            out.fail(f"long: {name}: {log} has no test result")
        elif "profile-op" not in text:
            out.note(f"long: {name}: no op ran on the HTP (left to the CPU by the op budget)")
    logs = sorted(glob.glob(os.path.join(d, "long_*.log")))
    if not logs:
        out.fail(f"long: no long_*.log in {d}")
        return
    r = subprocess.run([sys.executable, os.path.join(HERE, "op-budget.py"), "--fail", *logs],
                       capture_output=True, text=True)
    (out.fail if r.returncode else out.note)(f"long: op-budget.py --fail rc={r.returncode} on {len(logs)} logs")


def check_compare(path, drop, out):
    if not os.path.isfile(path):
        out.fail(f"compare: no {path}")
        return
    lines = [l.rstrip("\n") for l in open(path, errors="replace")]
    two = any(re.match(r"old: \S", l) for l in lines)
    steps = {}
    cur = None
    bench = {}
    for line in lines:
        s = line.strip()
        m = re.match(r"## (\S+)", s)
        if m:
            cur = m.group(1)
            if cur in steps:
                out.fail(f"compare: step {cur} appears twice in {path} (merge resumed runs explicitly)")
            steps[cur] = 0
            continue
        if cur is None or not s or re.match(r"(new|old):", s):
            continue
        if s.startswith("unknown step"):
            out.fail(f"compare: {s}")
            continue
        steps[cur] += 1
        rc = rc_of(s)
        if s.startswith("servers stopped"):
            if not re.search(r"alive=$", s):
                out.fail(f"compare: {s}")
            continue
        if rc is None:
            out.fail(f"compare: no rc in {s}")
        elif rc != 0:
            out.fail(f"compare: {s}")
        if " DIFFERENT" in s:
            out.review(f"compare: {s}")
        m = re.match(r"bench (old|new) (\S+) rc=0\b(.*)", s)
        if m:
            values = re.findall(r"(pp\d+|tg\d+) ([0-9.]+)", m.group(3))
            if not values:
                out.fail(f"compare: bench without values: {s}")
            for test, value in values:
                bench.setdefault((m.group(2), test, m.group(1)), []).append(float(value))
    if not steps:
        out.fail(f"compare: no steps in {path}")
    for step, n in steps.items():
        if step not in COMPARE_LINES:
            continue
        per_version, new_only = COMPARE_LINES[step]
        want = per_version * (2 if two else 1) + new_only
        if n < want:
            out.fail(f"compare: step {step} has {n} of {want} result lines (interrupted?)")
        elif n > want and not step.startswith("bench"):
            out.fail(f"compare: step {step} has {n} result lines, expected {want}")
    for (model, test, v), new in sorted(bench.items()):
        if v != "new":
            if (model, test, "new") not in bench:
                out.fail(f"compare: bench {model} {test} has no new result")
            continue
        old = bench.get((model, test, "old"))
        if old is None:
            if two:
                out.fail(f"compare: bench {model} {test} has no old result")
            continue
        o, n = sum(old) / len(old), sum(new) / len(new)
        if o <= 0 or not math.isfinite(o) or not math.isfinite(n):
            out.fail(f"compare: bench {model} {test} invalid values {old} {new}")
            continue
        change = n / o - 1
        reps = f" (means of {len(old)}/{len(new)})" if len(old) > 1 or len(new) > 1 else ""
        msg = f"compare: bench {model} {test} {o:g} -> {n:g} ({change:+.1%}){reps}"
        (out.fail if change < -drop else out.note)(msg)


PROBE_LOG = re.compile(r"lat_us min (-?\d+) .* max (-?\d+) \| >5ms (\d+) >50ms (\d+) failed (\d+)")


def probe_csvs(d):
    for c in sorted(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)):
        with open(c, errors="replace") as f:
            if f.readline().strip() == "t_us,lat_us":
                yield c


def check_probe_dir(d, limit_ms, out):
    """vtcmlat CSVs (t_us,lat_us) under d; their .log, when present, adds the failed-sample count."""
    csvs = list(probe_csvs(d))
    if not csvs:
        out.fail(f"probe: no vtcmlat CSV in {d}")
        return
    worst = 0
    for c in csvs:
        name = os.path.relpath(c, d)[:-len(".csv")]
        lat = []
        for line in open(c, errors="replace").readlines()[1:]:
            parts = line.strip().split(",")
            try:
                lat.append(int(parts[1]))
            except (IndexError, ValueError):
                out.fail(f"probe: {name}: bad line {line.strip()!r}")
        if not lat:
            out.fail(f"probe: {name}: no samples")
            continue
        mn, mx = min(lat), max(lat)
        if mn < 0:
            out.fail(f"probe: {name}: {sum(1 for v in lat if v < 0)} failed samples")
        if mx >= limit_ms * 1000:
            out.fail(f"probe: {name}: max {mx / 1000:.1f} ms >= {limit_ms:g} ms")
        if os.path.isfile(c + ".log"):
            m = PROBE_LOG.search(open(c + ".log", errors="replace").read())
            if not m:
                out.fail(f"probe: {name}: log without a latency summary (probe did not finish?)")
            elif int(m.group(5)) or int(m.group(4)):
                out.fail(f"probe: {name}: log reports failed {m.group(5)}, >50ms {m.group(4)}")
        worst = max(worst, mx)
    out.note(f"probe: {len(csvs)} CSVs in {d}, max {worst / 1000:.1f} ms")


def check_probe(path, limit_ms, out):
    n = 0
    for line in open(path):
        m = re.match(r"probe (\S+):? (?:(\d+) samples, )?.*max=(-?\d+) ?us(?:, timeouts=(\d+))?", line)
        if not m:
            continue
        n += 1
        samples, us, timeouts = m.group(2), int(m.group(3)), m.group(4)
        bad = us < 0 or us >= limit_ms * 1000 or (samples is not None and int(samples) == 0) or (timeouts not in (None, "0"))
        (out.fail if bad else out.note)(f"probe: {m.group(1)} max {us / 1000:.1f} ms" +
                                        (f", {samples} samples, timeouts {timeouts}" if samples else ""))
    if n == 0:
        out.fail(f"probe: no probe lines in {path}")


class Out:
    def __init__(self, accept=()):
        self.level = 0  # 0 pass, 3 review, 1 fail
        self.findings = []
        self.accept = list(accept)  # [regex, reason, hits]

    def _add(self, kind, msg):
        self.findings.append({"kind": kind, "check": msg.split(":", 1)[0], "message": msg})

    def note(self, msg):
        print(f"  ok      {msg}")
        self._add("ok", msg)

    def review(self, msg):
        a = next((a for a in self.accept if a[0].search(msg)), None)
        if a:
            a[2] += 1
            print(f"  accept  {msg} ({a[1]})")
            self._add("accepted", f"{msg} ({a[1]})")
            return
        print(f"  REVIEW  {msg}")
        self._add("review", msg)
        if self.level == 0:
            self.level = 3

    def fail(self, msg):
        print(f"  FAIL    {msg}")
        self._add("fail", msg)
        self.level = 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ops")
    ap.add_argument("--long")
    ap.add_argument("--compare", action="append", default=[])
    ap.add_argument("--probe-dir", action="append", default=[])
    ap.add_argument("--probe", action="append", default=[])
    ap.add_argument("--known", default=os.path.join(HERE, "..", "tests", "known-failures.txt"))
    ap.add_argument("--speed-drop", type=float, default=0.03)
    ap.add_argument("--probe-ms", type=float, default=25.0)
    ap.add_argument("--accept")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    accept = []
    if a.accept:
        for line in open(a.accept):
            line = line.strip()
            if line and not line.startswith("#"):
                pattern, _, reason = line.partition(" ")
                accept.append([re.compile(pattern), reason.strip() or "accepted", 0])
    out = Out(accept)
    if not (a.ops or a.long or a.compare or a.probe_dir or a.probe):
        out.fail("nothing to check")
    if a.ops:
        check_ops(a.ops, load_known(a.known), out)
    if a.long:
        check_long(a.long, out)
    for f in a.compare:
        check_compare(f, a.speed_drop, out)
    for d in a.probe_dir:
        check_probe_dir(d, a.probe_ms, out)
    for f in a.probe:
        check_probe(f, a.probe_ms, out)
    for pattern, reason, hits in out.accept:
        if not hits:
            out.note(f"accept: {pattern.pattern} matched nothing ({reason})")
    verdict = {0: "PASS", 3: "REVIEW", 1: "FAIL"}[out.level]
    print(verdict)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"verdict": verdict, "args": vars(a), "findings": out.findings}, f, indent=1)
    return out.level


if __name__ == "__main__":
    sys.exit(main())
