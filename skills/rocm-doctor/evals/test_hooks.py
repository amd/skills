"""Tests for `hooks.py`, the stub `rocm` the skill evals replay.

Run by `cargo test -p xtask` (see `the_eval_hook_tests_pass` in
`xtask/src/eval_fixtures.rs`), as `python -I test_hooks.py`. Stdlib only.

What they pin is the promise the docs make about the stub: it replays the
recorded answer for `--version`, `fix --json` and `diagnose`, refuses every
other command with exit code 2, and leaves PATH and the temporary directory as
it found them. It never runs anything (it only prints canned text); the refusal
of unrecorded `diagnose` options is about fidelity.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("eval_hooks", HERE / "hooks.py")
hooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hooks)

CASE = types.SimpleNamespace(id="rocm-current-cli-carries-on")
RECORDED = json.loads(
    (HERE / "fixtures" / CASE.id / "replies.json").read_text(encoding="utf-8")
)


def rocm(*args):
    """Run whatever `rocm` PATH resolves to, the way the agent's shell would."""
    found = shutil.which("rocm")
    assert found is not None, "no `rocm` on PATH"
    return subprocess.run([found, *args], capture_output=True, text=True, timeout=60)


def git_bash():
    """Git for Windows' bash, or None.

    A bare `bash` on a Windows runner can be the WSL launcher in System32 (it
    exits 1 when no distribution is installed), so find Git's own, starting from
    where `git` is: `<root>/cmd/git.exe` or `<root>/bin/git.exe` gives
    `<root>/bin/bash.exe`.
    """
    roots = []
    git = shutil.which("git")
    if git:
        roots.append(Path(git).resolve().parent.parent)
    for var in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        if os.environ.get(var):
            roots.append(Path(os.environ[var]) / "Git")
            roots.append(Path(os.environ[var]) / "Programs" / "Git")
    for root in roots:
        for relative in ("bin/bash.exe", "usr/bin/bash.exe"):
            candidate = root / relative
            if candidate.is_file():
                return str(candidate)
    return None


class Replay(unittest.TestCase):
    def setUp(self):
        self.saved_path = os.environ["PATH"]
        self.ctx = {}
        hooks.setup(None, CASE, self.ctx)
        self.directory = Path(self.ctx["_replay"][0])

    def tearDown(self):
        hooks.teardown(None, CASE, self.ctx)
        os.environ["PATH"] = self.saved_path

    def test_a_replayed_command_prints_the_recorded_reply_and_exits_0(self):
        for args, key in [
            (["--version"], "version"),
            (["fix", "--json"], "fix_json"),
            (["diagnose", "--json"], "diagnose_json"),
            (["diagnose", "--symptom", "x"], "diagnose_text"),
        ]:
            with self.subTest(args=args):
                done = rocm(*args)
                self.assertEqual(done.returncode, RECORDED[key]["rc"])
                self.assertEqual(done.returncode, 0)
                self.assertEqual(
                    done.stdout.replace("\r\n", "\n"), RECORDED[key]["stdout"]
                )

    def test_git_bash_on_windows_finds_the_stub_not_a_real_rocm(self):
        # Git Bash does not look for `rocm.cmd`; without the extensionless
        # launcher it would run a real `rocm` further along PATH.
        if os.name != "nt":
            self.skipTest("Git Bash on Windows only")
        bash = git_bash()
        if bash is None:
            self.skipTest(
                "no Git for Windows bash found next to `git` or in Program Files"
            )
        done = subprocess.run(
            [bash, "-c", "rocm --version"], capture_output=True, text=True, timeout=60
        )
        detail = (
            f"bash={bash} rc={done.returncode} "
            f"stdout={done.stdout!r} stderr={done.stderr!r}"
        )
        self.assertEqual(done.returncode, 0, detail)
        self.assertEqual(done.stdout, RECORDED["version"]["stdout"], detail)

    def test_every_other_command_is_refused_with_exit_code_2(self):
        for args in [
            ["fix", "some-id"],
            ["install"],
            ["serve"],
            ["diagnose", "--report"],
            ["diagnose", "--send"],
            ["diagnose", "--model", "foo"],
            ["diagnose", "--distro", "Ubuntu"],
            ["diagnose", "--model=foo"],
            ["diagnose", "--distro=Ubuntu"],
            ["diagnose", "--json", "--send=me@example.invalid"],
        ]:
            with self.subTest(args=args):
                done = rocm(*args)
                self.assertEqual(done.returncode, 2, done.stderr)
                self.assertEqual(done.stdout, "")
                self.assertIn("runs nothing else", done.stderr)

    def test_teardown_restores_path_and_removes_the_directory(self):
        self.assertTrue(self.directory.is_dir())
        self.assertEqual(os.environ["PATH"].split(os.pathsep)[0], str(self.directory))
        hooks.teardown(None, CASE, self.ctx)
        self.assertEqual(os.environ["PATH"], self.saved_path)
        self.assertFalse(self.directory.exists())
        self.assertNotIn("_replay", self.ctx)
        # A second teardown is harmless.
        hooks.teardown(None, CASE, self.ctx)
        self.assertEqual(os.environ["PATH"], self.saved_path)


class Overlap(unittest.TestCase):
    def test_a_second_replay_while_one_is_active_is_refused(self):
        saved = os.environ["PATH"]
        first, second = {}, {}
        hooks.setup(None, CASE, first)
        try:
            with self.assertRaises(RuntimeError):
                hooks.setup(None, CASE, second)
            self.assertNotIn("_replay", second)
        finally:
            hooks.teardown(None, CASE, first)
            os.environ["PATH"] = saved
        # The refusal did not leave the hook stuck: a later case can replay.
        again = {}
        hooks.setup(None, CASE, again)
        hooks.teardown(None, CASE, again)
        self.assertEqual(os.environ["PATH"], saved)


class NoFixture(unittest.TestCase):
    def test_a_case_with_no_directory_gets_no_rocm_and_an_untouched_path(self):
        before = os.environ["PATH"]
        ctx = {}
        self.assertEqual(
            hooks.setup(None, types.SimpleNamespace(id="no-such"), ctx), {}
        )
        self.assertEqual(os.environ["PATH"], before)
        self.assertEqual(ctx, {})
        hooks.teardown(None, types.SimpleNamespace(id="no-such"), ctx)
        self.assertEqual(os.environ["PATH"], before)


if __name__ == "__main__":
    unittest.main()
