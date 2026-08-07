# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""IR-1 launcher gate: refuse to start `optimize` unless the GPUs are idle.

Checks, in order: MODEL_PATH resolves to a model directory, torch sees the
GPUs, no foreign serving process is running, and every visible GPU holds less
than IR1_VRAM_LIMIT_MIB of VRAM.

The VRAM check is fail-closed. When VRAM cannot be read -- no amd-smi/rocm-smi,
a probe that exits non-zero, unparseable output, or a reading missing for any
visible GPU -- the gate blocks the launch instead of assuming the GPU is free.
Set IR1_ALLOW_UNVERIFIED_VRAM=1 to downgrade an unreadable probe to a warning
after confirming by hand that the GPUs are idle.

Environment:
    MODEL_PATH                  Required. Model directory checked for config.json.
    IR1_VRAM_LIMIT_MIB          Per-GPU VRAM ceiling in MiB. Default 500.
    IR1_ALLOW_UNVERIFIED_VRAM   Set to 1 to proceed when VRAM is unreadable.

Exit status:
    0   Gate passed; safe to launch.
    1   Gate blocked; do not launch.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

DEFAULT_VRAM_LIMIT_MIB = 500
PROBE_TIMEOUT_SEC = 30

FOREIGN_SERVING_PATTERNS = (
    "hyperloom.inference_optimizer.cli",
    "Magpie",
    "sglang.launch_server",
    "vllm.entrypoints",
)

# amd-smi 26.x (ROCm 7.x) wraps the GPU list in a top-level container key
# instead of emitting a bare list.
GPU_LIST_KEYS = ("gpu_data", "gpus", "data")

# amd-smi nests the reading differently across releases, so try each container
# and key spelling rather than assuming one shape.
MEM_CONTAINER_KEYS = ("mem_usage", "mem", "memory", "vram")
USED_VRAM_KEYS = ("used_vram", "vram_used", "used_memory", "used")

# Vendor tools report MB (10^6) while the IR-1 ceiling is MiB (2^20).
UNIT_TO_MIB = {
    "b": 1 / (1024 * 1024),
    "kb": 1000 / (1024 * 1024),
    "kib": 1 / 1024,
    "mb": 1000 * 1000 / (1024 * 1024),
    "mib": 1.0,
    "gb": 1000 * 1000 * 1000 / (1024 * 1024),
    "gib": 1024.0,
}


class VramUnreadable(Exception):
    """The VRAM probe could not produce a reading for every visible GPU."""


def _fail(message: str) -> None:
    print(f"IR-1 BLOCK: {message}", file=sys.stderr)


def _warn(message: str) -> None:
    print(f"IR-1 WARNING: {message}", file=sys.stderr)


def check_model_path() -> bool:
    path = os.environ.get("MODEL_PATH", "").strip()
    if not path:
        _fail("MODEL_PATH is empty; re-run the Phase 2 'Persist the plan' step")
        return False
    if not os.path.isdir(path):
        _fail(f"MODEL_PATH is not a directory: {path}")
        return False
    if not os.path.isfile(os.path.join(path, "config.json")):
        _fail(f"MODEL_PATH has no config.json: {path}")
        return False
    print(f"model_path_ok={path}")
    return True


def check_torch_gpus() -> bool:
    try:
        import torch
    except Exception as exc:
        _fail(f"torch is not importable ({type(exc).__name__}); run install.sh first")
        return False
    try:
        available = torch.cuda.is_available()
        count = torch.cuda.device_count() if available else 0
    except Exception as exc:
        _fail(f"torch GPU probe raised {type(exc).__name__}: {str(exc)[:200]}")
        return False
    print(f"torch_cuda_available={available} torch_cuda_device_count={count}")
    if not available or count == 0:
        _fail("torch sees no GPU; check ROCm, /dev/kfd and /dev/dri")
        return False
    return True


def check_foreign_processes() -> bool:
    # Report the matched pattern and pid only; a serving cmdline can carry tokens.
    found = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                raw = handle.read()
        except OSError:
            continue
        text = raw.replace(b"\0", b" ").decode("utf-8", "ignore")
        if not text:
            continue
        for pattern in FOREIGN_SERVING_PATTERNS:
            if pattern in text:
                found.append((pid, pattern))
                break
    for pid, pattern in found:
        print(f"foreign_serving_process pid={pid} matched={pattern}")
    if found:
        _fail(f"{len(found)} foreign serving process(es) still hold the GPUs")
        return False
    print("foreign_serving_processes=0")
    return True


def _to_mib(value: object, unit: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise VramUnreadable(f"unsupported VRAM value type: {type(value).__name__}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise VramUnreadable(f"VRAM value is not numeric: {value!r}") from None
    key = str(unit or "mib").strip().lower()
    if key not in UNIT_TO_MIB:
        raise VramUnreadable(f"unknown VRAM unit: {unit!r}")
    return number * UNIT_TO_MIB[key]


def _extract_used_vram(entry: object) -> float:
    if not isinstance(entry, dict):
        raise VramUnreadable(f"GPU entry is {type(entry).__name__}, expected an object")
    containers = [entry]
    for key in MEM_CONTAINER_KEYS:
        nested = entry.get(key)
        if isinstance(nested, dict):
            containers.append(nested)
    for container in containers:
        for key in USED_VRAM_KEYS:
            if key not in container:
                continue
            reading = container[key]
            if isinstance(reading, dict):
                if "value" not in reading:
                    raise VramUnreadable(f"{key} object has no 'value' field")
                return _to_mib(reading.get("value"), reading.get("unit"))
            return _to_mib(reading, "mib")
    raise VramUnreadable(f"no used-VRAM field found; keys={sorted(entry)[:8]}")


def _run_probe(argv: list[str]) -> str:
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=PROBE_TIMEOUT_SEC
        )
    except subprocess.TimeoutExpired:
        raise VramUnreadable(f"{argv[0]} timed out after {PROBE_TIMEOUT_SEC}s") from None
    except OSError as exc:
        raise VramUnreadable(f"{argv[0]} could not be executed: {exc}") from None
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[:200]
        raise VramUnreadable(f"{argv[0]} exited {completed.returncode}: {detail}")
    if not completed.stdout.strip():
        raise VramUnreadable(f"{argv[0]} produced no output")
    return completed.stdout


def _parse_json(payload: str, tool: str) -> object:
    try:
        return json.loads(payload)
    except ValueError as exc:
        raise VramUnreadable(f"{tool} output is not valid JSON: {exc}") from None


def _iter_gpu_entries(parsed: object, tool: str) -> list[tuple[object, object]]:
    if isinstance(parsed, list):
        return list(enumerate(parsed))
    if isinstance(parsed, dict):
        # A known container key wins over the per-key scan below, so a sibling
        # metadata object is not mistaken for a GPU entry.
        for key in GPU_LIST_KEYS:
            nested = parsed.get(key)
            if isinstance(nested, list):
                return list(enumerate(nested))
        entries = [(key, value) for key, value in parsed.items() if isinstance(value, dict)]
        if entries:
            return entries
    raise VramUnreadable(f"{tool} output has no recognizable GPU list")


def read_vram_usage() -> list[tuple[object, float]]:
    if shutil.which("amd-smi"):
        tool = "amd-smi"
        payload = _run_probe(["amd-smi", "metric", "-m", "--json"])
    elif shutil.which("rocm-smi"):
        tool = "rocm-smi"
        payload = _run_probe(["rocm-smi", "--showmeminfo", "vram", "--json"])
    else:
        raise VramUnreadable("neither amd-smi nor rocm-smi is on PATH")

    entries = _iter_gpu_entries(_parse_json(payload, tool), tool)
    readings: list[tuple[object, float]] = []
    for device, entry in entries:
        try:
            readings.append((device, _extract_used_vram(entry)))
        except VramUnreadable as exc:
            raise VramUnreadable(f"{tool} gpu {device}: {exc}") from None
    if not readings:
        raise VramUnreadable(f"{tool} reported no GPUs")
    return readings


def check_vram() -> bool:
    limit = int(os.environ.get("IR1_VRAM_LIMIT_MIB", DEFAULT_VRAM_LIMIT_MIB))
    allow_unverified = os.environ.get("IR1_ALLOW_UNVERIFIED_VRAM", "").strip() == "1"
    try:
        readings = read_vram_usage()
    except VramUnreadable as exc:
        if allow_unverified:
            _warn(
                f"VRAM unreadable ({exc}); proceeding because "
                "IR1_ALLOW_UNVERIFIED_VRAM=1. Confirm the GPUs are idle by hand."
            )
            return True
        _fail(
            f"VRAM unreadable ({exc}). A busy GPU cannot be ruled out, so the "
            "launch is blocked. Fix the probe, or set "
            "IR1_ALLOW_UNVERIFIED_VRAM=1 after confirming the GPUs are idle."
        )
        return False

    over_limit = []
    for device, mib in readings:
        marker = "OVER_LIMIT" if mib > limit else "ok"
        print(f"gpu {device}: used_vram_mib={mib:.0f} ({marker})")
        if mib > limit:
            over_limit.append(device)
    if over_limit:
        _fail(f"GPU(s) {over_limit} hold more than {limit} MiB; stop them first")
        return False
    return True


def main() -> int:
    checks = (
        check_model_path,
        check_torch_gpus,
        check_foreign_processes,
        check_vram,
    )
    passed = True
    for check in checks:
        if not check():
            passed = False
    print(f"IR1_RESULT={'PASS' if passed else 'BLOCK'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
