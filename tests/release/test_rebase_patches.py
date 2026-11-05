#!/usr/bin/env python3
"""Offline tests of scripts/rebase-patches.sh on a toy superproject with one submodule.

  python3 -m unittest discover -s tests/release -v
"""

import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "..", "scripts", "rebase-patches.sh")
ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
           GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")


def sh(*args, cwd=None, check=True):
    r = subprocess.run(args, cwd=cwd, env=ENV, capture_output=True, text=True)
    if check and r.returncode:
        raise AssertionError(f"{args} rc={r.returncode}\n{r.stdout}\n{r.stderr}")
    return r


class RebasePatchesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.up = os.path.join(t, "up")
        os.makedirs(self.up)
        sh("git", "init", "-q", "-b", "main", cwd=self.up)
        self.write(self.up, "a.c", "int f(void) {\n    return 1;\n}\n\nint g(void) {\n    return 2;\n}\n")
        sh("git", "add", "-A", cwd=self.up)
        sh("git", "commit", "-qm", "c1", cwd=self.up)
        self.c1 = sh("git", "rev-parse", "HEAD", cwd=self.up).stdout.strip()

        self.top = os.path.join(t, "top")
        os.makedirs(self.top)
        sh("git", "init", "-q", "-b", "main", cwd=self.top)
        sh("git", "-c", "protocol.file.allow=always", "submodule", "add", "-q", self.up, "sub", cwd=self.top)
        os.makedirs(os.path.join(self.top, "patches"))
        sh("git", "commit", "-qm", "top", cwd=self.top)

        # the patch: a test case added after f()
        sub = os.path.join(self.top, "sub")
        self.write(sub, "a.c", "int f(void) {\n    return 1;\n}\n\nint h(void) {\n    return 3;\n}\n\nint g(void) {\n    return 2;\n}\n")
        patch = sh("git", "diff", cwd=sub).stdout
        sh("git", "checkout", "-q", "--", "a.c", cwd=sub)
        self.write(self.top, "patches/0001-h.patch", patch)
        self.work = os.path.join(t, "work")

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def write(d, name, text):
        with open(os.path.join(d, name), "w") as f:
            f.write(text)

    def upstream(self, text):
        self.write(self.up, "a.c", text)
        sh("git", "commit", "-qam", "c2", cwd=self.up)
        rev = sh("git", "rev-parse", "HEAD", cwd=self.up).stdout.strip()
        sh("git", "-c", "protocol.file.allow=always", "fetch", "-q", "origin", cwd=os.path.join(self.top, "sub"))
        return rev

    def run_script(self, *args, check=True):
        return sh("bash", SCRIPT, *args, cwd=self.top, check=check)

    def test_clean_rebase(self):
        new = self.upstream("// header\nint f(void) {\n    return 1;\n}\n\nint g(void) {\n    return 2;\n}\n")
        self.run_script("start", "sub", "patches", new, self.work)
        r = self.run_script("export", self.work)
        self.assertIn("same lines as before", r.stdout)
        patch = open(os.path.join(self.top, "patches/0001-h.patch")).read()
        self.assertIn("+int h(void) {", patch)

    def test_union_conflict(self):
        # upstream adds its own function at the same place: both additions stay
        new = self.upstream("int f(void) {\n    return 1;\n}\n\nint u(void) {\n    return 4;\n}\n\nint g(void) {\n    return 2;\n}\n")
        r = self.run_script("start", "sub", "patches", new, self.work, check=False)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertNotEqual(self.run_script("continue", self.work, check=False).returncode, 0)
        self.run_script("continue", self.work, "--union")
        r = self.run_script("export", self.work)
        a = open(os.path.join(self.work, "repo", "a.c")).read()
        self.assertEqual(a, "int f(void) {\n    return 1;\n}\n\nint u(void) {\n    return 4;\n}\n\n"
                            "int h(void) {\n    return 3;\n}\n\nint g(void) {\n    return 2;\n}\n")

    def test_brace_check(self):
        # a resolution that leaves a function unclosed is caught at export
        new = self.upstream("int f(void) {\n    return 1;\n}\n\nint u(void) {\n    return 4;\n}\n\nint g(void) {\n    return 2;\n}\n")
        self.run_script("start", "sub", "patches", new, self.work, check=False)
        self.run_script("continue", self.work, "--union")
        repo = os.path.join(self.work, "repo")
        a = open(os.path.join(repo, "a.c")).read().replace("int h(void) {\n    return 3;\n}\n", "int h(void) {\n    return 3;\n")
        self.write(repo, "a.c", a)
        sh("git", "commit", "-q", "--amend", "-a", "--no-edit", cwd=repo)
        r = self.run_script("export", self.work, check=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("brace check", r.stderr)

    def test_not_applying_to_pin(self):
        self.write(self.top, "patches/0002-bad.patch", "--- a/a.c\n+++ b/a.c\n@@ -1,1 +1,1 @@\n-nothing\n+else\n")
        new = self.upstream("// header\nint f(void) {\n    return 1;\n}\n\nint g(void) {\n    return 2;\n}\n")
        r = self.run_script("start", "sub", "patches", new, self.work, check=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not apply to the pinned", r.stderr)


if __name__ == "__main__":
    unittest.main()
