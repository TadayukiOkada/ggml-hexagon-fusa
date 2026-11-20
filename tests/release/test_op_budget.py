#!/usr/bin/env python3
"""Offline tests of the yield count and exemptions in scripts/op-budget.py, on synthetic profile-op lines.

  python3 -m unittest discover -s tests/release -v
"""

import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "..", "scripts", "op-budget.py")

MM = ("0.01 D ggml-hex: HTP0 profile-op MUL_MAT|w x a -> o|1024:3584 x 1024:512 -> 3584:512|iq2_xxs x f32 -> f32|"
      "66:264 x 4:4096 -> 4:14336|hmx-tiled nhvx 4 vtcm 4725376|usec {usec} cycles 100 start 1 mhz 1470.8{tail}\n")
FA = ("0.02 D ggml-hex: HTP0 profile-op FLASH_ATTN_EXT|q x k x v -> o|256:512:8:1 x 256:1024:2:1 -> 256:8:512:1|"
      "f32 x f16 x f16 -> f32|4:8 x 2:8 -> 4:8|hmx-pipe nhvx 4 vtcm 8283776|usec {usec} cycles 100 start 1 mhz 1460.6{tail}\n")


def run(text, *args):
    return subprocess.run([sys.executable, SCRIPT, *args], input=text, capture_output=True, text=True)


class OpBudgetYieldsTest(unittest.TestCase):
    def test_sums_yields_per_group(self):
        text = MM.format(usec=900, tail=" yields 2") + MM.format(usec=800, tail=" yields 1") + \
            FA.format(usec=500, tail=" yields 0")
        r = run(text, "--fail", "--min-yields", "MUL_MAT:hmx-tiled=3", "--min-yields", "MUL_MAT=1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("3 yields inside ops", r.stdout)
        self.assertIn("ok: 3 yields inside MUL_MAT:hmx-tiled", r.stdout)
        mm_row = [line for line in r.stdout.splitlines() if " MUL_MAT " in line][0]
        self.assertEqual(mm_row.split()[4], "3")  # max, p95, mean, count, yields

    def test_yields_after_pmu(self):
        r = run(MM.format(usec=900, tail=" pmu [1,2,3,4,5,6,7,8] yields 4"), "--fail", "--min-yields", "MUL_MAT=4")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_missing_yields_fail(self):
        text = MM.format(usec=900, tail=" yields 0") + FA.format(usec=500, tail=" yields 5")
        r = run(text, "--fail", "--min-yields", "MUL_MAT=1")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("MISSING: 0 yields inside MUL_MAT", r.stdout)
        # without --fail the shortfall is only reported
        self.assertEqual(run(text, "--min-yields", "MUL_MAT=1").returncode, 0)

    def test_old_log_without_yields(self):
        text = MM.format(usec=900, tail="")
        r = run(text, "--fail")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("not reported by this build", r.stdout)
        self.assertEqual([line for line in r.stdout.splitlines() if " MUL_MAT " in line][0].split()[4], "-")
        # an expectation cannot be met by a log that does not report yields
        self.assertEqual(run(text, "--fail", "--min-yields", "MUL_MAT=1").returncode, 1)

    def test_exempt(self):
        text = MM.format(usec=90000, tail=" yields 0")
        self.assertEqual(run(text, "--fail", "--yielding", "").returncode, 1)
        r = run(text, "--fail", "--yielding", "", "--exempt", r"MUL_MAT:hmx-tiled=^1024:3584\sx\s1024:512\s")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("1 group(s) over the budget", r.stdout)
        self.assertTrue([l for l in r.stdout.splitlines() if l.startswith("xx")], r.stdout)
        # another tag or shape does not match
        self.assertEqual(run(text, "--fail", "--yielding", "", "--exempt", "MUL_MAT:hvx-tiled=.").returncode, 1)
        self.assertEqual(run(text, "--fail", "--yielding", "", "--exempt", "MUL_MAT=^2048:").returncode, 1)
        self.assertEqual(run(text, "--exempt", "MUL_MAT").returncode, 2)
        self.assertEqual(run(text, "--exempt", "MUL_MAT=(").returncode, 2)

    def test_bad_spec(self):
        self.assertEqual(run(MM.format(usec=1, tail=""), "--min-yields", "MUL_MAT").returncode, 2)


if __name__ == "__main__":
    unittest.main()
