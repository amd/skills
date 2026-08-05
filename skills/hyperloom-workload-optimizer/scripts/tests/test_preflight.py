# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Regression tests for the IR-1 VRAM gate in ../preflight.py.

Each case installs a fake amd-smi that reports a busy GPU in a different output
shape, then asserts the gate blocks the launch. Before the fail-closed rewrite,
an unexpected shape or a non-zero probe exit let a GPU holding ~140 GiB pass as
idle, so these cases are the reproduction for that bug.

Run standalone; no pytest or third-party dependency required:

    python3 scripts/tests/test_preflight.py
"""

from __future__ import annotations

import importlib.util
import os
import stat
import sys
import tempfile
from pathlib import Path

BUSY_MIB = 143360
SCRIPT = Path(__file__).resolve().parents[1] / "preflight.py"


def _load_preflight():
    spec = importlib.util.spec_from_file_location("preflight_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = _load_preflight()


def _install_fake_smi(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _emit(payload: str) -> str:
    quoted = payload.replace("'", "'\\''")
    return f"printf '%s' '{quoted}'"


CASES: list[tuple[str, str | None, bool, dict[str, str]]] = [
    (
        "expected shape, GPU busy",
        _emit(
            '[{"gpu": 0, "mem_usage": {"total_vram": {"value": 196592, "unit": "MB"},'
            ' "used_vram": {"value": %d, "unit": "MB"}}}]' % BUSY_MIB
        ),
        False,
        {},
    ),
    (
        "expected shape, GPU idle",
        _emit('[{"gpu": 0, "mem_usage": {"used_vram": {"value": 283, "unit": "MB"}}}]'),
        True,
        {},
    ),
    (
        "top-level object instead of list, GPU busy",
        _emit('{"gpu_0": {"mem_usage": {"used_vram": {"value": %d, "unit": "MB"}}}}' % BUSY_MIB),
        False,
        {},
    ),
    (
        "gpu list wrapped in a top-level container key, GPU busy",
        _emit(
            '{"gpu_data": [{"gpu": 0, "mem_usage": {"used_vram":'
            ' {"value": %d, "unit": "MB"}}}]}' % BUSY_MIB
        ),
        False,
        {},
    ),
    (
        "gpu list wrapped in a top-level container key, GPU idle",
        _emit(
            '{"gpu_data": [{"gpu": 0, "mem_usage": {"used_vram":'
            ' {"value": 283, "unit": "MB"}}}]}'
        ),
        True,
        {},
    ),
    (
        "wrapped gpu list alongside a sibling metadata key, GPU idle",
        _emit(
            '{"gpu_data": [{"gpu": 0, "mem_usage": {"used_vram":'
            ' {"value": 283, "unit": "MB"}}}],'
            ' "metadata": {"version": "26.2.2"}}'
        ),
        True,
        {},
    ),
    (
        "scalar used_vram instead of value/unit map, GPU busy",
        _emit('[{"gpu": 0, "mem_usage": {"used_vram": %d}}]' % BUSY_MIB),
        False,
        {},
    ),
    (
        "container renamed mem_usage -> mem, GPU busy",
        _emit('[{"gpu": 0, "mem": {"used_vram": {"value": %d, "unit": "MB"}}}]' % BUSY_MIB),
        False,
        {},
    ),
    (
        "used_vram key absent (renamed vram_used), GPU busy",
        _emit('[{"gpu": 0, "mem_usage": {"vram_used": {"value": %d, "unit": "MB"}}}]' % BUSY_MIB),
        False,
        {},
    ),
    (
        "no used-VRAM field at all",
        _emit('[{"gpu": 0, "mem_usage": {"total_vram": {"value": 196592, "unit": "MB"}}}]'),
        False,
        {},
    ),
    (
        "amd-smi exits non-zero (driver error)",
        'echo "Unable to communicate with the amdgpu driver" >&2\nexit 1',
        False,
        {},
    ),
    (
        "warning banner before JSON, GPU busy",
        'echo "WARNING: amdgpu version mismatch"\n'
        + _emit('[{"gpu": 0, "mem_usage": {"used_vram": {"value": %d, "unit": "MB"}}}]' % BUSY_MIB),
        False,
        {},
    ),
    (
        "used_vram explicitly null",
        _emit('[{"gpu": 0, "mem_usage": {"used_vram": null}}]'),
        False,
        {},
    ),
    (
        "empty GPU list",
        _emit("[]"),
        False,
        {},
    ),
    (
        "GiB unit, GPU busy",
        _emit('[{"gpu": 0, "mem_usage": {"used_vram": {"value": 140, "unit": "GiB"}}}]'),
        False,
        {},
    ),
    (
        "MB unit just under the MiB ceiling",
        _emit('[{"gpu": 0, "mem_usage": {"used_vram": {"value": 500, "unit": "MB"}}}]'),
        True,
        {},
    ),
    (
        "unknown unit",
        _emit('[{"gpu": 0, "mem_usage": {"used_vram": {"value": 12, "unit": "furlongs"}}}]'),
        False,
        {},
    ),
    (
        "second GPU unreadable while first is idle",
        _emit(
            '[{"gpu": 0, "mem_usage": {"used_vram": {"value": 100, "unit": "MB"}}},'
            ' {"gpu": 1, "mem_usage": {}}]'
        ),
        False,
        {},
    ),
    (
        "no probe tool on PATH",
        None,
        False,
        {},
    ),
    (
        "no probe tool on PATH with explicit override",
        None,
        True,
        {"IR1_ALLOW_UNVERIFIED_VRAM": "1"},
    ),
    (
        "unreadable probe with explicit override",
        'exit 1',
        True,
        {"IR1_ALLOW_UNVERIFIED_VRAM": "1"},
    ),
]


def run_case(body: str | None, extra_env: dict[str, str]) -> bool:
    saved_env = dict(os.environ)
    with tempfile.TemporaryDirectory() as tmp:
        fake_dir = Path(tmp)
        if body is not None:
            _install_fake_smi(fake_dir, "amd-smi", body)
        try:
            # Isolate PATH so only the fake probe (if any) is discoverable.
            os.environ["PATH"] = str(fake_dir)
            os.environ.pop("IR1_ALLOW_UNVERIFIED_VRAM", None)
            os.environ["IR1_VRAM_LIMIT_MIB"] = "500"
            os.environ.update(extra_env)
            return preflight.check_vram()
        finally:
            os.environ.clear()
            os.environ.update(saved_env)


def main() -> int:
    failures = 0
    for name, body, expected_pass, extra_env in CASES:
        actual_pass = run_case(body, extra_env)
        ok = actual_pass == expected_pass
        if not ok:
            failures += 1
        want = "PASS" if expected_pass else "BLOCK"
        got = "PASS" if actual_pass else "BLOCK"
        print(f"[{'ok' if ok else 'FAIL'}] {name}: want {want}, got {got}")
    print()
    print(f"{len(CASES) - failures}/{len(CASES)} cases behaved as expected")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
