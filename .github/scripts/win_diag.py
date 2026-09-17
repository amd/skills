"""What this Windows runner actually provides, measured rather than assumed.

Two failures here have each survived a fix aimed at a theory: `WinError 2`
launching the CLI, and `sqlite3.OperationalError` opening inspect's sample
buffer. Both fixes were reasoned from how Windows is supposed to behave. This
asks the machine.
"""

from __future__ import annotations

import asyncio
import asyncio.subprocess as sp
import glob
import os
import shutil
import subprocess


def resolved() -> str | None:
    found = shutil.which("claude")
    print(f"which('claude')  -> {found!r}")
    print(f"PATHEXT          -> {os.environ.get('PATHEXT')}")
    pattern = os.path.expandvars(r"%APPDATA%\npm\claude*")
    print(f"npm shims        -> {sorted(glob.glob(pattern))}")
    return found


def buffer_dir() -> None:
    """The sample buffer path, to test the MAX_PATH reading of the sqlite error."""
    from inspect_ai._util.appdirs import inspect_data_dir

    base = str(inspect_data_dir("samplebuffer"))
    # <base>/<64-char sha256>/<timestamp 25><task name><22-char id>.eval.<pid>.db
    fixed = len(base) + 1 + 64 + 1 + 25 + 22 + len(".eval.12345.db")
    print(f"samplebuffer     -> {base}")
    print(f"fixed overhead   -> {fixed} chars, leaving {260 - fixed} for the task name")
    for task in (
        "behavioral-local-ai-use",
        "behavioral-lemonade-router-builder",
        "behavioral-tracelens-analysis-orchestrator",
    ):
        total = fixed + len(task)
        print(f"   {task:44s} {total:4d} {'OVER 260' if total > 260 else 'ok'}")


def launches(found: str | None) -> None:
    """Sync works today under the legacy engine; async is what inspect uses."""
    if not found:
        print("no claude on PATH; nothing to launch")
        return
    r = subprocess.run([found, "--version"], capture_output=True, text=True)
    print(f"sync  Popen      -> rc={r.returncode} {(r.stdout or r.stderr)[:60]!r}")

    async def go() -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                found, "--version", stdout=sp.PIPE, stderr=sp.PIPE
            )
            out, err = await proc.communicate()
            print(f"async exec       -> rc={proc.returncode} {(out or err).decode()[:60]!r}")
        except Exception as exc:  # noqa: BLE001 -- the failure is the result
            print(f"async exec       -> {type(exc).__name__}: {exc}")

        try:
            proc = await asyncio.create_subprocess_exec(
                "cmd.exe", "/c", found, "--version", stdout=sp.PIPE, stderr=sp.PIPE
            )
            out, err = await proc.communicate()
            print(f"async via cmd.exe-> rc={proc.returncode} {(out or err).decode()[:60]!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"async via cmd.exe-> {type(exc).__name__}: {exc}")

    asyncio.run(go())


if __name__ == "__main__":
    found = resolved()
    print()
    buffer_dir()
    print()
    launches(found)
