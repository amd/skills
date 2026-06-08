#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Detect whether this machine is a tunable AMD APU and report support level.

This is the first script the `apu-memory-tuner` skill runs. It is read-only.
It answers a single question: "can this skill help you on this machine?"

It prints a human-readable summary and, with `--json`, emits a structured
record the agent can parse to drive the next steps.

Exit codes:
  0 = supported; the rest of the skill can proceed.
  2 = not an AMD APU, or APU generation we don't have tuning recipes for.
  3 = AMD APU detected, but a hard prerequisite is missing (e.g. Linux
      kernel too old). The agent should surface the reason and stop.

The detection is intentionally best-effort. Failing to read one source does
not abort the script; it just leaves the corresponding field as `None` /
`"unknown"`. The agent should treat empty fields as "needs user
confirmation", not "broken".
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from _sanitize import sanitize_process_output

# Mapping from LLVM gfx target to the marketing-friendly generation bucket
# the rest of the skill keys off. RDNA3.5 (gfx115x) is the only generation
# the source ROCm doc describes shared-memory tuning for, so it is the only
# generation marked "supported" here. Older APUs still expose the same TTM
# knob but without an officially supported tuning recipe.
GFX_TO_GENERATION: dict[str, str] = {
    "gfx1150": "rdna35",
    "gfx1151": "rdna35",
    "gfx1152": "rdna35",
    "gfx1103": "rdna3",
    "gfx1102": "rdna3",
    "gfx1100": "rdna3",
    "gfx1036": "rdna2",
    "gfx1035": "rdna2",
    "gfx1034": "rdna2",
    "gfx1033": "rdna2",
}

# Kernel version floors for the Linux gate. The authoritative, up-to-date
# matrix (per distribution + per ROCm release) lives at:
#   https://rocm.docs.amd.com/en/latest/how-to/system-optimization/rdna3-5.html
# Always cross-check that page before bumping these constants. Strix Halo
# (gfx1151) requires the KFD fixes referenced there; without them queue
# creation and memory checks misbehave.
LINUX_KERNEL_MIN_MAINLINE = (6, 18, 4)
LINUX_KERNEL_MIN_UBUNTU_HWE = (6, 17, 0)
LINUX_KERNEL_MIN_UBUNTU_OEM = (6, 14, 0)


@dataclass
class Detection:
    os_family: str = "unknown"          # linux | windows | other
    os_version: str = ""
    cpu_vendor: str = "unknown"
    cpu_model: str = ""
    gpu_name: str = ""
    gfx_target: str = ""                # e.g. gfx1151
    generation: str = "unknown"         # rdna35 | rdna3 | rdna2 | older | not-amd-apu | unknown
    is_apu: bool | None = None
    total_ram_gb: float | None = None
    kernel_version: str = ""            # Linux only
    kernel_supported: bool | None = None  # Linux only
    amd_ttm_present: bool | None = None   # Linux only
    adrenalin_version: str = ""           # Windows only
    supported: bool = False
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _run(cmd: list[str], timeout: float = 5.0) -> tuple[int, str, str]:
    """Run a command; return (exit, stdout, stderr). Never raises."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return (
            r.returncode,
            sanitize_process_output(r.stdout),
            sanitize_process_output(r.stderr),
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return 127, "", ""


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _total_ram_gb_linux() -> float | None:
    txt = _read_text("/proc/meminfo")
    m = re.search(r"^MemTotal:\s+(\d+)\s+kB", txt, re.MULTILINE)
    if not m:
        return None
    return round(int(m.group(1)) / (1024 * 1024), 2)


def _total_ram_gb_windows() -> float | None:
    # `wmic` is deprecated but still ships everywhere; PowerShell CIM is the
    # modern path. Try CIM first, fall back to wmic, then GlobalMemoryStatus
    # via ctypes as a last resort.
    rc, out, _ = _run([
        "powershell", "-NoProfile", "-Command",
        "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
    ], timeout=8)
    if rc == 0 and out.strip().isdigit():
        return round(int(out.strip()) / (1024 ** 3), 2)

    rc, out, _ = _run(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"])
    if rc == 0:
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                return round(int(line) / (1024 ** 3), 2)

    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return round(stat.ullTotalPhys / (1024 ** 3), 2)
    except (OSError, AttributeError, ImportError):
        pass
    return None


def _cpu_info_linux() -> tuple[str, str]:
    """Return (vendor, model) from /proc/cpuinfo."""
    txt = _read_text("/proc/cpuinfo")
    vendor = ""
    model = ""
    for line in txt.splitlines():
        if not vendor and line.startswith("vendor_id"):
            vendor = line.split(":", 1)[1].strip()
        elif not model and line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
        if vendor and model:
            break
    vendor_short = "amd" if "AMD" in vendor else ("intel" if "Intel" in vendor else vendor.lower())
    return vendor_short, model


def _cpu_info_windows() -> tuple[str, str]:
    rc, out, _ = _run([
        "powershell", "-NoProfile", "-Command",
        "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name",
    ], timeout=8)
    name = out.strip() if rc == 0 else ""
    vendor = "amd" if "AMD" in name else ("intel" if "Intel" in name else "unknown")
    return vendor, name


def _gfx_target_from_rocminfo() -> tuple[str, str]:
    """Return (gfx_target, gpu_name) from rocminfo if available."""
    if shutil.which("rocminfo") is None:
        return "", ""
    rc, out, _ = _run(["rocminfo"], timeout=10)
    if rc != 0:
        return "", ""
    gfx = ""
    name = ""
    in_gpu_agent = False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Device Type:"):
            in_gpu_agent = "GPU" in s
        if in_gpu_agent and s.startswith("Marketing Name:") and not name:
            name = s.split(":", 1)[1].strip()
        if in_gpu_agent and s.startswith("Name:") and s.endswith("gfx") is False:
            val = s.split(":", 1)[1].strip()
            if val.startswith("gfx") and not gfx:
                gfx = val
        if gfx and name:
            break
    return gfx, name


def _gfx_target_from_sysfs() -> tuple[str, str]:
    """Best-effort fallback when rocminfo isn't installed.

    The amdgpu driver exposes the LLVM target string at
    /sys/class/drm/card*/device/llvm_gfx_target on recent kernels.
    """
    try:
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
            target_path = card / "device" / "llvm_gfx_target"
            if target_path.exists():
                target = target_path.read_text().strip()
                name_path = card / "device" / "product_name"
                name = name_path.read_text().strip() if name_path.exists() else ""
                if target.startswith("gfx"):
                    return target, name
    except OSError:
        pass
    return "", ""


def _gpu_info_windows() -> tuple[str, str]:
    """Return (gfx_target_guess, gpu_name) for the AMD adapter on Windows.

    Windows does not expose the LLVM gfx target the way the Linux amdgpu
    driver does; we approximate by mapping the marketing name to a generation
    bucket. The mapping is intentionally conservative; when in doubt we
    return an empty gfx target and let the user confirm.
    """
    rc, out, _ = _run([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like '*AMD*' -or $_.Name -like '*Radeon*' } | Select-Object -ExpandProperty Name",
    ], timeout=8)
    if rc != 0 or not out.strip():
        return "", ""
    name = out.strip().splitlines()[0].strip()
    lname = name.lower()
    # Heuristic mapping for known APU marketing names. Any miss falls through
    # to "unknown gfx target", which downstream code treats as "ask the user".
    if "ryzen ai max" in lname or "strix halo" in lname:
        return "gfx1151", name
    if "radeon 880m" in lname or "radeon 890m" in lname or "strix" in lname:
        return "gfx1150", name
    if "radeon 780m" in lname or "radeon 760m" in lname or "phoenix" in lname or "hawk point" in lname:
        return "gfx1103", name
    return "", name


def _kernel_version_linux() -> str:
    return platform.release()


def _parse_kernel_tuple(v: str) -> tuple[int, int, int]:
    """Extract (major, minor, patch) from a Linux kernel release string.

    Examples:
      "6.17.0-19-generic"   -> (6, 17, 0)
      "6.18.4"              -> (6, 18, 4)
      "6.14.0-1018-oem"     -> (6, 14, 0)
    """
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", v)
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _kernel_supported(release: str) -> bool:
    """Apply the version matrix from the source ROCm doc.

    We accept the kernel if it meets ANY of:
      - Mainline >= 6.18.4
      - Ubuntu HWE >= 6.17.0 (HWE kernels carry the backport)
      - Ubuntu OEM >= 6.14.0 (OEM kernels carry the backport)
    """
    tup = _parse_kernel_tuple(release)
    if tup >= LINUX_KERNEL_MIN_MAINLINE:
        return True
    if "generic" in release and tup >= LINUX_KERNEL_MIN_UBUNTU_HWE:
        return True
    if "oem" in release and tup >= LINUX_KERNEL_MIN_UBUNTU_OEM:
        return True
    return False


def _adrenalin_version_windows() -> str:
    """Best-effort probe of the AMD Adrenalin driver version from the registry.

    Adrenalin writes its version under HKLM\\SOFTWARE\\AMD\\CN\\<...>; the
    exact key drifts across releases, so we shell out to PowerShell rather
    than encode a registry path that will rot.
    """
    rc, out, _ = _run([
        "powershell", "-NoProfile", "-Command",
        "(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like '*AMD*' -or $_.Name -like '*Radeon*' } | Select-Object -First 1).DriverVersion",
    ], timeout=8)
    return out.strip() if rc == 0 else ""


def detect() -> Detection:
    d = Detection()
    sysname = platform.system().lower()
    d.os_version = platform.platform()

    if sysname == "linux":
        d.os_family = "linux"
        d.cpu_vendor, d.cpu_model = _cpu_info_linux()
        d.total_ram_gb = _total_ram_gb_linux()
        d.kernel_version = _kernel_version_linux()
        d.kernel_supported = _kernel_supported(d.kernel_version)
        d.amd_ttm_present = shutil.which("amd-ttm") is not None
        gfx, gpu = _gfx_target_from_rocminfo()
        if not gfx:
            gfx, gpu = _gfx_target_from_sysfs()
        d.gfx_target, d.gpu_name = gfx, gpu
    elif sysname == "windows":
        d.os_family = "windows"
        d.cpu_vendor, d.cpu_model = _cpu_info_windows()
        d.total_ram_gb = _total_ram_gb_windows()
        d.gfx_target, d.gpu_name = _gpu_info_windows()
        d.adrenalin_version = _adrenalin_version_windows()
    else:
        d.os_family = "other"
        d.reasons.append(
            f"Unsupported OS family: {sysname}. This skill targets Linux and Windows."
        )
        return d

    # Generation classification
    if d.gfx_target in GFX_TO_GENERATION:
        d.generation = GFX_TO_GENERATION[d.gfx_target]
    elif d.gfx_target.startswith("gfx10"):
        d.generation = "older"
    elif d.cpu_vendor != "amd":
        d.generation = "not-amd-apu"
    else:
        d.generation = "unknown"

    # An APU has the GPU and CPU on the same package. We can't read that
    # directly without DMI; use a proxy: AMD CPU + integrated Radeon string,
    # or a known APU gfx target.
    if d.generation in {"rdna35", "rdna3", "rdna2"}:
        d.is_apu = True
    elif "radeon" in d.gpu_name.lower() and d.cpu_vendor == "amd":
        d.is_apu = True
    else:
        d.is_apu = False

    # Decide overall supported flag and surface reasons.
    if d.cpu_vendor != "amd":
        d.reasons.append("CPU is not AMD; this skill only tunes AMD APUs.")
    elif not d.is_apu:
        d.reasons.append(
            "No AMD APU detected. This skill is APU-only (it does not tune "
            "discrete Radeon GPUs)."
        )
    elif d.generation == "rdna35":
        d.supported = True
    elif d.generation in {"rdna3", "rdna2"}:
        d.supported = True
        d.notes.append(
            f"Detected {d.generation.upper()} APU. Tuning will work, but the "
            "official AMD recipe targets RDNA3.5; recommended GTT/VRAM splits "
            "may be conservative."
        )
    else:
        d.reasons.append(
            f"AMD GPU detected ({d.gpu_name or 'unknown'}) but the generation "
            "could not be classified. Run `rocminfo` (Linux) or share the "
            "Adapter name (Windows) so the skill can confirm."
        )

    if d.os_family == "linux" and d.supported and d.kernel_supported is False:
        d.supported = False
        d.reasons.append(
            f"Linux kernel {d.kernel_version} is below the minimum required "
            f"for RDNA3.5 ({'.'.join(map(str, LINUX_KERNEL_MIN_MAINLINE))} "
            f"mainline, {'.'.join(map(str, LINUX_KERNEL_MIN_UBUNTU_HWE))} "
            "Ubuntu HWE, or "
            f"{'.'.join(map(str, LINUX_KERNEL_MIN_UBUNTU_OEM))} Ubuntu OEM). "
            "Upgrade the kernel before tuning."
        )

    if d.os_family == "linux" and d.supported and d.amd_ttm_present is False:
        d.notes.append(
            "`amd-ttm` is not on PATH. Apply step will install it on demand "
            "via `pipx install amd-debug-tools`."
        )

    return d


def _print_human(d: Detection) -> None:
    print("APU memory tuner -- platform detection")
    print("-" * 40)
    print(f"OS:                {d.os_family} ({d.os_version})")
    print(f"CPU:               {d.cpu_model or 'unknown'} (vendor: {d.cpu_vendor})")
    print(f"GPU:               {d.gpu_name or 'unknown'}")
    print(f"GFX target:        {d.gfx_target or 'unknown'}")
    print(f"Generation:        {d.generation}")
    print(f"APU:               {d.is_apu}")
    print(f"Total RAM:         {d.total_ram_gb if d.total_ram_gb is not None else 'unknown'} GB")
    if d.os_family == "linux":
        print(f"Kernel:            {d.kernel_version} (supported: {d.kernel_supported})")
        print(f"amd-ttm on PATH:   {d.amd_ttm_present}")
    if d.os_family == "windows":
        print(f"Adrenalin driver:  {d.adrenalin_version or 'unknown'}")
    print()
    print(f"Supported by skill: {'YES' if d.supported else 'NO'}")
    for reason in d.reasons:
        print(f"  - {reason}")
    for note in d.notes:
        print(f"  note: {note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human summary.",
    )
    args = parser.parse_args(argv)

    d = detect()

    if args.json:
        print(json.dumps(asdict(d), indent=2))
    else:
        _print_human(d)

    if not d.supported:
        # Distinguish "wrong hardware" (exit 2) from "right hardware, missing
        # prereq" (exit 3) so the agent can pick the right next step.
        if any("kernel" in r.lower() for r in d.reasons):
            return 3
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
