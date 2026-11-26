"""Offline tests for device-tests.sh, target-suite.sh, and llama-release.sh."""

import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "scripts")


def write(path, text, executable=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    if executable:
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IRUSR)


class SuiteAndImportTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="suite-import-test-")

    def tearDown(self):
        shutil.rmtree(self.d)

    def _make_fake_pkg(self, pkg_dir, mode="pass"):
        os.makedirs(pkg_dir, exist_ok=True)
        write(os.path.join(pkg_dir, "BUILD_INFO.txt"), "build-id: fake\n")
        run_sh = f"""#!/bin/sh
tool=$1; shift
if [ "$tool" = "test-backend-ops-hex" ]; then
    sub=$1
    if [ "$sub" = "support" ]; then
        echo "  ADD(type=f32): SUPPORTED"
        echo "  SIN(type=f32): NOT SUPPORTED"
        exit 0
    fi
    op=""
    while [ $# -gt 0 ]; do
        if [ "$1" = "-o" ]; then op=$2; break; fi
        shift
    done
    if [ "$op" = "SIN" ]; then
        echo "0/0 tests passed"
        echo "no test case ran: the selected test cases are not supported"
        exit 1
    fi
    if [ "{mode}" = "crash" ] && [ "$op" = "ADD" ]; then
        echo "Segmentation fault"
        exit 139
    fi
    if [ "{mode}" = "unknown_fail" ] && [ "$op" = "ADD" ]; then
        echo "ggml-hex: HTP0 profile-op ADD|1 -> 1|f32 -> f32|x|usec 10 cycles 1"
        echo "[ADD] ERR = 0.5 > 0.0000001   ADD(type=f32,ne=[1,1,1,1]): FAIL"
        echo "Failing tests:"
        echo "  ADD(type=f32,ne=[1,1,1,1])"
        echo "9/10 tests passed"
        exit 1
    fi
    echo "ggml-hex: HTP0 profile-op $op|1 -> 1|f32 -> f32|x|usec 10 cycles 1"
    echo "10/10 tests passed"
    echo "2/2 backends passed"
    exit 0
fi
if [ "$tool" = "test-hexagon-metadata" ]; then
    echo "OK, 64 graphs on HTP0"
    exit 0
fi
exit 0
"""
        write(os.path.join(pkg_dir, "run.sh"), run_sh, executable=True)

    def _stage_scripts(self, base):
        sdir = os.path.join(base, "scripts")
        tdir = os.path.join(base, "tests")
        os.makedirs(sdir, exist_ok=True)
        for name in ("device-tests.sh", "target-suite.sh", "verdict.py", "op-budget.py"):
            shutil.copy2(os.path.join(SCRIPTS, name), os.path.join(sdir, name))
        write(os.path.join(tdir, "known-failures.txt"), "")
        write(os.path.join(tdir, "known-over-budget.txt"), "")
        return sdir

    def test_device_tests_ops_detects_crash_and_accepts_unsupported(self):
        pkg_ok = os.path.join(self.d, "pkg_ok")
        self._make_fake_pkg(pkg_ok, mode="pass")
        out_ok = os.path.join(self.d, "out_ok")
        env = dict(os.environ, TEST_PAUSE="0")
        r_ok = subprocess.run(
            ["sh", os.path.join(SCRIPTS, "device-tests.sh"), pkg_ok, out_ok, "--ops"],
            env=env, capture_output=True, text=True,
        )
        self.assertEqual(r_ok.returncode, 0, r_ok.stdout + r_ok.stderr)

        pkg_crash = os.path.join(self.d, "pkg_crash")
        self._make_fake_pkg(pkg_crash, mode="crash")
        out_crash = os.path.join(self.d, "out_crash")
        r_crash = subprocess.run(
            ["sh", os.path.join(SCRIPTS, "device-tests.sh"), pkg_crash, out_crash, "--ops"],
            env=env, capture_output=True, text=True,
        )
        self.assertNotEqual(r_crash.returncode, 0)

    def test_target_suite_stops_on_ops_verdict_failure(self):
        sdir = self._stage_scripts(self.d)
        pkg = os.path.join(self.d, "pkg")
        kit = os.path.join(self.d, "kit")
        out = os.path.join(self.d, "res")
        os.makedirs(kit, exist_ok=True)
        self._make_fake_pkg(pkg, mode="unknown_fail")
        env = dict(os.environ, TEST_PAUSE="0")
        r = subprocess.run(
            ["sh", os.path.join(sdir, "target-suite.sh"), "--evk", pkg, "-", kit, out, "ops", "long"],
            env=env, capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        with open(os.path.join(out, "steps.txt"), encoding="utf-8") as f:
            steps = f.read()
        self.assertIn("STOP ops: verdict failed", steps)
        self.assertNotIn("long ", steps)

    def test_llama_release_review_ops_exits_3_on_review(self):
        repo = os.path.join(self.d, "repo")
        sub = os.path.join(repo, "llama.cpp")
        os.makedirs(repo, exist_ok=True)
        os.makedirs(sub, exist_ok=True)

        def git(cwd, *args):
            return subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                cwd=cwd, check=True, capture_output=True, text=True,
            ).stdout.strip()

        git(sub, "init", "-q")
        kparams = os.path.join(sub, "ggml", "src", "ggml-hexagon", "htp-kparams.cpp")
        write(kparams, "case GGML_OP_ADD: return HTP_OP_ADD;\n")
        git(sub, "add", ".")
        git(sub, "commit", "-q", "-m", "old")
        old_rev = git(sub, "rev-parse", "HEAD")

        write(kparams, "case GGML_OP_ADD: return HTP_OP_ADD;\ncase GGML_OP_NEW_OP: return HTP_OP_NEW_OP;\n")
        git(sub, "add", ".")
        git(sub, "commit", "-q", "-m", "new")
        new_rev = git(sub, "rev-parse", "HEAD")

        git(repo, "init", "-q")
        write(os.path.join(repo, "ggml", "src", "ggml-hexagon", "htp-budget.cpp"), "case GGML_OP_ADD:\n")
        write(os.path.join(repo, "scripts", "op-budget.py"), "\"ADD\"\n")
        shutil.copy2(os.path.join(SCRIPTS, "llama-release.sh"), os.path.join(repo, "scripts", "llama-release.sh"))
        git(repo, "add", ".")
        git(repo, "commit", "-q", "-m", "super")

        r_review = subprocess.run(
            ["bash", os.path.join(repo, "scripts", "llama-release.sh"), "review-ops", old_rev, new_rev, repo],
            cwd=repo, capture_output=True, text=True,
        )
        self.assertEqual(r_review.returncode, 3, r_review.stdout + r_review.stderr)
        self.assertIn("NEW_OP", r_review.stdout)

        write(os.path.join(repo, "ggml", "src", "ggml-hexagon", "htp-budget.cpp"),
              "case GGML_OP_ADD:\ncase GGML_OP_NEW_OP:\n")
        r_ok = subprocess.run(
            ["bash", os.path.join(repo, "scripts", "llama-release.sh"), "review-ops", old_rev, new_rev, repo],
            cwd=repo, capture_output=True, text=True,
        )
        self.assertEqual(r_ok.returncode, 0, r_ok.stdout + r_ok.stderr)


if __name__ == "__main__":
    unittest.main()
