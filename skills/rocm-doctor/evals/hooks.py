"""Eval plumbing: put a recorded `rocm` on PATH for the cases that have one.

The grading machines carry no `rocm` binary, so an agent following the skill
stops at the install offer and never reads a diagnosis. For a case that has a
directory under `fixtures/`, `setup` writes a stub `rocm` into a temporary
directory, puts it first on PATH, and `teardown` takes it away again. A case
with no directory runs exactly as it did before: no `rocm` on PATH.

The stub replays recorded output and nothing else. It answers `--version`,
`fix --json` and `diagnose`, and `examine --json` for a case whose recording has
one, and refuses every other command with exit code 2.
It only ever prints text read from `replies.json`, so whatever an agent asks it,
it cannot change a machine or send anything: that comes from the allowlist, not
from the refusal of `diagnose` options below, which is there for fidelity (see
`refused`). The recordings are generated from the real catalog by
`cargo xtask eval-fixtures`, and CI fails when they go stale; do not edit them
by hand.

Harness contract, as skillscope v0.1.0 calls it (`skillscope/behavior.py`):
`run()` loops over a skill's cases one at a time in one process (`for case in
skill_cases`), and `run_case()` calls `setup()` before the agent starts and
`teardown()` in a `finally` after grading. The value `setup()` returns only
feeds template variables for the prompt (`case_ctx.update(...)`); it is not an
environment. The agent is started with `env=claude_env()`, which is a copy of
`os.environ` taken at that moment (`skillscope/agent.py`), so the only way to put
a `rocm` in front of the agent is to prepend to the process-global `PATH`. That
is safe only while cases run serially in one process, which is what the harness
does; `setup()` raises if a replay is already active, so a harness that overlaps
cases fails loudly instead of leaking one case's directory into another.

Stdlib only. No network, no subprocess, nothing written outside the temporary
directory.
"""

import os
import shlex
import shutil
import stat
import sys
import tempfile
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPLIES = "replies.json"

STUB = r"""import json
import sys
from pathlib import Path

replies = json.loads(
    (Path(__file__).resolve().parent / "replies.json").read_text(encoding="utf-8")
)
args = sys.argv[1:]
# Fidelity only, not safety (the allowlist below already makes the stub inert).
# The one recorded `diagnose` answer is for the symptom the case was recorded
# with. A real CLI would answer a different one for `--distro` or `--model`, and
# would prepare or send a report for `--report` or `--send`; replaying the same
# text for those would show the agent something the real CLI never prints, so the
# stub says it has no recording instead. Matched on the name alone, so
# `--model=foo` is refused as `--model foo` is.
refused = {"--report", "--send", "--distro", "--model"}


def reply(entry):
    sys.stdout.buffer.write(entry["stdout"].encode("utf-8"))
    sys.stderr.buffer.write(entry["stderr"].encode("utf-8"))
    sys.exit(entry["rc"])


if args in (["--version"], ["-V"], ["version"]):
    reply(replies["version"])
elif args == ["fix", "--json"]:
    reply(replies["fix_json"])
elif args[:1] == ["diagnose"] and not refused & {a.split("=")[0] for a in args}:
    reply(replies["diagnose_json" if "--json" in args else "diagnose_text"])
elif args == ["examine", "--json"] and "examine_json" in replies:
    reply(replies["examine_json"])
sys.stderr.write(
    "error: this evaluation replays only `--version`, `fix --json`, `diagnose` "
    "and, where recorded, `examine --json`; it runs nothing else\n"
)
sys.exit(2)
"""


def _write_launcher(directory: Path) -> None:
    """A `rocm` the agent's shell can find, which runs the stub.

    On Windows the agent's shell may be cmd, PowerShell (both find `rocm.cmd`
    through PATHEXT) or Git Bash (which does not look for `.cmd` and would run a
    real `rocm` further along PATH), so both a `rocm.cmd` and an extensionless
    `rocm` script are written.
    """
    python = sys.executable
    launcher = directory / "rocm"
    rocm_py = shlex.quote(str(directory / "rocm.py"))
    # Bytes, not write_text: on Windows text mode would write CRLF, which a
    # POSIX shell reads as part of the interpreter name (`/bin/sh<CR>`).
    launcher.write_bytes(
        f'#!/bin/sh\nexec {shlex.quote(python)} {rocm_py} "$@"\n'.encode()
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    if os.name == "nt":
        (directory / "rocm.cmd").write_text(
            f'@echo off\r\n"{python}" "%~dp0rocm.py" %*\r\n',
            encoding="utf-8",
        )


_active = None  # the replay directory while a case is running, else None


def setup(workspace, case, ctx):
    global _active
    source = FIXTURES / case.id / REPLIES
    if not source.is_file():
        return {}
    if _active is not None:
        raise RuntimeError(
            "a replay is already active: this hook assumes the harness runs "
            "one case at a time in one process"
        )
    directory = Path(tempfile.mkdtemp(prefix="rocm-replay-"))
    shutil.copyfile(source, directory / REPLIES)
    (directory / "rocm.py").write_text(STUB, encoding="utf-8")
    _write_launcher(directory)
    _active = str(directory)
    ctx["_replay"] = (str(directory), os.environ.get("PATH", ""))
    os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
    return {}


def teardown(workspace, case, ctx):
    global _active
    replay = ctx.pop("_replay", None)
    if replay is None:
        return
    _active = None
    directory, saved_path = replay
    os.environ["PATH"] = saved_path
    shutil.rmtree(directory, ignore_errors=True)
