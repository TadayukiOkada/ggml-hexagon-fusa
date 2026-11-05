#!/usr/bin/env python3
"""Offline tests of scripts/verdict.py on synthetic target results (no device needed).

  python3 -m unittest discover -s tests/release -v
"""

import contextlib
import io
import os
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import verdict  # noqa: E402

PASS, FAIL, REVIEW = 0, 1, 3

KNOWN = "SUB type=f16 1e-6 f16 rounding\nPOOL_2D k0=128,.*p0=(64|128), any padding\n"


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(textwrap.dedent(text).lstrip("\n"))


def compare_summary(two=True, drop_line=None, edit=None):
    lines = ["new: build abc", "old: build def" if two else "old: "]
    versions = ["old", "new"] if two else ["new"]
    lines.append("## leak 2026-11-04T00:00:00Z")
    lines.append("leak rc=0 OK")
    lines.append("## llm 2026-11-04T00:00:00Z")
    for v in versions:
        for m in ("Q3", "Q35"):
            lines.append(f"llm {v} {m} rc=0 md5=aaaa {'same' if v == 'new' and two else ''} 12 ms per token")
    lines.append("## bench 2026-11-04T00:00:00Z")
    for v in versions:
        for m in ("Q3", "Q35"):
            lines.append(f"bench {v} {m} rc=0 pp512 1000.0  tg128 50.0  ")
    lines.append("## prefill 2026-11-04T00:00:00Z")
    lines += ["prefill fa=on rc=0 1000 ms", "prefill fa=off rc=0 2000 ms"]
    lines.append("## asr 2026-11-04T00:00:00Z")
    for v in versions:
        for d in ("HTP0", "HTP1"):
            for w in ("ja_second", "input"):
                lines.append(f"asr {v} {d} {w} rc=0 text=bbbb {'same' if v == 'new' and two else ''}")
    lines.append("asr new HTP0 l060 rc=0 text=cccc")
    lines.append("## sv 2026-11-04T00:00:00Z")
    for v in versions:
        for d in ("HTP0", "HTP1"):
            for n in ("input", "input30", "browser-captured", "short100"):
                lines.append(f"sv {v} {d} {n} rc=0 md5=dddd {'same' if v == 'new' and two else ''}")
    lines.append("## servers 2026-11-04T00:00:00Z")
    lines += ["servers ready rc=0", "sv soak rc=0 600 ok", "nemo soak rc=0 24 ok (RSS of NeMo is recorded, not gated)",
              "servers stopped after 1s alive="]
    if drop_line:
        lines = [l for l in lines if not l.startswith(drop_line)]
    if edit:
        lines = [edit(l) for l in lines]
    return "\n".join(lines) + "\n"


class VerdictTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.known = os.path.join(self.d, "known.txt")
        write(self.known, KNOWN)

    def tearDown(self):
        self.tmp.cleanup()

    def run_verdict(self, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            level = verdict.main(["--known", self.known, *args])
        self.output = buf.getvalue()
        return level

    # --ops

    def ops_dir(self, rows, ops=("ADD", "SUB"), logs=None):
        d = os.path.join(self.d, "ops")
        write(os.path.join(d, "ops.txt"), "".join(o + "\n" for o in ops))
        write(os.path.join(d, "summary.txt"), "package: x\n" + "".join(r + "\n" for r in rows))
        for name, text in (logs or {}).items():
            write(os.path.join(d, f"test_{name}.log"), text)
        return d

    def test_ops_pass(self):
        d = self.ops_dir(["test ADD rc=0 10/10 tests passed", "test SUB rc=0 4/4 tests passed", "metadata rc=0 OK"])
        self.assertEqual(self.run_verdict("--ops", d), PASS, self.output)

    def test_ops_known_failure(self):
        log = """
        [SUB] ERR = 0.0000005 > 0.0000001   SUB(type=f16,ne=[1,1,1,1]): FAIL
        Failing tests:
          SUB(type=f16,ne=[1,1,1,1])
        """
        d = self.ops_dir(["test ADD rc=0 10/10 tests passed", "test SUB rc=1 3/4 tests passed"], logs={"SUB": log})
        self.assertEqual(self.run_verdict("--ops", d), PASS, self.output)

    def test_ops_unknown_failure(self):
        log = """
        [ADD] ERR = 0.5 > 0.0000001   ADD(type=f32,ne=[1,1,1,1]): FAIL
        Failing tests:
          ADD(type=f32,ne=[1,1,1,1])
        """
        d = self.ops_dir(["test ADD rc=1 9/10 tests passed", "test SUB rc=0 4/4 tests passed"], logs={"ADD": log})
        self.assertEqual(self.run_verdict("--ops", d), FAIL, self.output)

    def test_ops_empty_support_list(self):
        d = self.ops_dir([], ops=())
        self.assertEqual(self.run_verdict("--ops", d), FAIL, self.output)

    def test_ops_missing_op(self):
        d = self.ops_dir(["test ADD rc=0 10/10 tests passed"])
        self.assertEqual(self.run_verdict("--ops", d), FAIL, self.output)
        self.assertIn("1 of 2 supported ops have no result", self.output)

    def test_ops_signal_and_not_runnable(self):
        for rc in ("-11", "127", "124"):
            d = self.ops_dir([f"test ADD rc={rc} ", "test SUB rc=0 4/4 tests passed"])
            self.assertEqual(self.run_verdict("--ops", d), FAIL, rc)

    def test_ops_rc0_without_count(self):
        d = self.ops_dir(["test ADD rc=0 ", "test SUB rc=0 4/4 tests passed"])
        self.assertEqual(self.run_verdict("--ops", d), FAIL, self.output)

    def test_ops_duplicate_rows(self):
        d = self.ops_dir(["test ADD rc=0 10/10 tests passed", "test ADD rc=0 10/10 tests passed",
                          "test SUB rc=0 4/4 tests passed"])
        self.assertEqual(self.run_verdict("--ops", d), FAIL, self.output)

    def test_ops_metadata_fail(self):
        d = self.ops_dir(["test ADD rc=0 10/10 tests passed", "test SUB rc=0 4/4 tests passed", "metadata rc=1 grew"])
        self.assertEqual(self.run_verdict("--ops", d), FAIL, self.output)

    # --long

    def long_dir(self, rows, logs):
        d = os.path.join(self.d, "long")
        write(os.path.join(d, "summary.txt"), "".join(r + "\n" for r in rows))
        for name, text in logs.items():
            write(os.path.join(d, f"long_{name}.log"), text)
        return d

    PROFILE = "ggml-hex: HTP0 profile-op {op}|a -> b|f32 -> f32|x|usec {usec} cycles 1\n2/2 backends passed\n"

    def test_long_pass_and_cpu(self):
        d = self.long_dir(["long fa rc=0", "long argsort rc=0"],
                          {"fa": self.PROFILE.format(op="FLASH_ATTN_EXT", usec=90000), "argsort": "2/2 backends passed\n"})
        self.assertEqual(self.run_verdict("--long", d), PASS, self.output)
        self.assertIn("left to the CPU", self.output)

    def test_long_over_budget(self):
        d = self.long_dir(["long lu rc=0"], {"lu": self.PROFILE.format(op="LIGHTNING_INDEXER", usec=90000)})
        self.assertEqual(self.run_verdict("--long", d), FAIL, self.output)

    def test_long_crashed(self):
        d = self.long_dir(["long fa rc=139"], {"fa": "Segmentation fault\n"})
        self.assertEqual(self.run_verdict("--long", d), FAIL, self.output)

    def test_long_empty(self):
        d = self.long_dir([], {})
        self.assertEqual(self.run_verdict("--long", d), FAIL, self.output)

    # --compare

    def compare(self, text):
        p = os.path.join(self.d, "summary.txt")
        write(p, text)
        return self.run_verdict("--compare", p)

    def test_compare_pass(self):
        self.assertEqual(self.compare(compare_summary()), PASS, self.output)

    def test_compare_new_only(self):
        self.assertEqual(self.compare(compare_summary(two=False)), PASS, self.output)

    def test_compare_rc127_old(self):
        s = compare_summary(edit=lambda l: l.replace("llm old Q3 rc=0", "llm old Q3 rc=127"))
        self.assertEqual(self.compare(s), FAIL, self.output)

    def test_compare_signal(self):
        s = compare_summary(edit=lambda l: l.replace("prefill fa=off rc=0", "prefill fa=off rc=-6"))
        self.assertEqual(self.compare(s), FAIL, self.output)

    def test_compare_different_is_review(self):
        s = compare_summary(edit=lambda l: l.replace("sv new HTP1 input rc=0 md5=dddd same", "sv new HTP1 input rc=0 md5=eeee DIFFERENT"))
        self.assertEqual(self.compare(s), REVIEW, self.output)

    def test_compare_interrupted_step(self):
        self.assertEqual(self.compare(compare_summary(drop_line="sv new HTP1")), FAIL, self.output)
        self.assertIn("step sv has", self.output)

    def test_compare_server_alive(self):
        s = compare_summary(edit=lambda l: l.replace("alive=", "alive=nemo"))
        self.assertEqual(self.compare(s), FAIL, self.output)

    def test_compare_speed_drop(self):
        s = compare_summary(edit=lambda l: l.replace("bench new Q35 rc=0 pp512 1000.0", "bench new Q35 rc=0 pp512 960.0"))
        self.assertEqual(self.compare(s), FAIL, self.output)

    def test_compare_speed_abba_means(self):
        # old/new/new/old: old 1000 and 980, new 990 and 985 -> -0.25%, within 3%
        s = compare_summary()
        s += "## bench 2026-11-04T01:00:00Z\n"
        s = s.replace("## bench 2026-11-04T01:00:00Z\n", "")
        s = s.replace("bench old Q35 rc=0 pp512 1000.0  tg128 50.0  \n",
                      "bench old Q35 rc=0 pp512 1000.0  tg128 50.0  \nbench old Q35 rc=0 pp512 980.0  tg128 50.0  \n")
        s = s.replace("bench new Q35 rc=0 pp512 1000.0  tg128 50.0  \n",
                      "bench new Q35 rc=0 pp512 990.0  tg128 50.0  \nbench new Q35 rc=0 pp512 985.0  tg128 50.0  \n")
        self.assertEqual(self.compare(s), PASS, self.output)
        self.assertIn("means of 2/2", self.output)

    def test_compare_bench_without_values(self):
        s = compare_summary(edit=lambda l: "bench new Q3 rc=0 " if l.startswith("bench new Q3 ") else l)
        self.assertEqual(self.compare(s), FAIL, self.output)

    def test_compare_duplicate_step(self):
        s = compare_summary() + "## leak 2026-11-04T02:00:00Z\nleak rc=0 OK\n"
        self.assertEqual(self.compare(s), FAIL, self.output)

    def test_compare_empty(self):
        self.assertEqual(self.compare("new: x\n"), FAIL, self.output)

    # --probe-dir

    def probe(self, name, lat, log=None):
        p = os.path.join(self.d, "probe", name + ".csv")
        write(p, "t_us,lat_us\n" + "".join(f"{i * 50000},{v}\n" for i, v in enumerate(lat)))
        if log is not None:
            write(p + ".log", log)
        return os.path.dirname(p)

    LOG = "dom=cdsp\nlat_us min 9 p50 17 p90 600 p99 8000 max {mx} | >5ms 2 >50ms 0 failed {failed}\n"

    def test_probe_pass(self):
        d = self.probe("asr_0", [10, 200, 12000], self.LOG.format(mx=12000, failed=0))
        self.probe("asr_1", [10, 20])
        self.assertEqual(self.run_verdict("--probe-dir", d), PASS, self.output)

    def test_probe_over_limit(self):
        d = self.probe("asr_0", [10, 26000])
        self.assertEqual(self.run_verdict("--probe-dir", d), FAIL, self.output)

    def test_probe_failed_sample(self):
        d = self.probe("asr_0", [10, -1])
        self.assertEqual(self.run_verdict("--probe-dir", d), FAIL, self.output)

    def test_probe_log_reports_failure(self):
        d = self.probe("asr_0", [10, 20], self.LOG.format(mx=20, failed=3))
        self.assertEqual(self.run_verdict("--probe-dir", d), FAIL, self.output)

    def test_probe_empty_csv(self):
        d = self.probe("asr_0", [])
        self.assertEqual(self.run_verdict("--probe-dir", d), FAIL, self.output)

    def test_probe_no_csv(self):
        os.makedirs(os.path.join(self.d, "none"))
        self.assertEqual(self.run_verdict("--probe-dir", os.path.join(self.d, "none")), FAIL, self.output)

    def test_probe_ignores_other_csv(self):
        d = self.probe("asr_0", [10, 20])
        write(os.path.join(d, "soak_sv.csv"), "request,rss_kib\n1,100\n")
        self.assertEqual(self.run_verdict("--probe-dir", d), PASS, self.output)

    # general

    def test_nothing_to_check(self):
        self.assertEqual(self.run_verdict(), FAIL)

    def test_json(self):
        d = self.ops_dir(["test ADD rc=0 10/10 tests passed", "test SUB rc=0 4/4 tests passed"])
        j = os.path.join(self.d, "v.json")
        self.run_verdict("--ops", d, "--json", j)
        import json
        data = json.load(open(j))
        self.assertEqual(data["verdict"], "PASS")
        self.assertTrue(any(f["check"] == "ops" for f in data["findings"]))


if __name__ == "__main__":
    unittest.main()
