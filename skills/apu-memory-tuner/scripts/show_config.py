#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Report the current shared-vs-dedicated memory split on an AMD APU.

Read-only. Safe to run at any time. Used by `apu-memory-tuner` both as the
"show me where I am" step and as the post-change verification step.

On Linux it reads:
  - /sys/module/ttm/parameters/pages_limit  (the GTT/shared cap)
  - /proc/meminfo                            (total RAM, for percentages)
  - dmesg lines about amdgpu VRAM            (the firmware carve-out)
  - rocminfo Pool sizes                      (sanity check via the runtime)

On Windows it reads:
  - Win32_VideoController.AdapterRAM         (the BIOS carve-out, capped at
                                              4 GiB by the WDDM API)
  - dxdiag /t output                         (Dedicated + Shared Memory)
  - Win32_ComputerSystem.TotalPhysicalMemory (total RAM)

All probes are best-effort. Missing fields are reported as `unknown` rather
than failing, because the user typically only needs one of the numbers to
make a decision.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from _sanitize import sanitize_process_output

# 4 KiB is the page size assumed by the amdgpu/TTM accounting on every
# platform we target. The kernel param is in pages, the user thinks in GB.
PAGE_SIZE_BYTES = 4096

# Windows' Win32_VideoController.AdapterRAM is a 32-bit field, so it caps at
# 2^32 - 1 == 4294967295 bytes (~4 GiB). Anything at or above this is a
# WDDM truncation artifact, not a real measurement; we surface that to the
# user instead of pretending it is the real reservation.
WIN_ADAPTER_RAM_CAP = 4_294_967_295


@dataclass
class Config:
    os_family: str = "unknown"
    total_ram_gb: float | None = None
    dedicated_vram_gb: float | None = None
    shared_gpu_gb: float | None = None
    ttm_pages_limit: int | None = None      # Linux only
    ttm_source_file: str = ""               # Linux only
    rocminfo_global_gb: float | None = None  # Linux only
    notes: list[str] | None = None


def _run(cmd: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
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


def _ttm_pages_limit() -> tuple[int | None, str]:
    """Return (pages, source_path) for the current TTM pages_limit on Linux.

    The kernel parameter lives at /sys/module/ttm/parameters/pages_limit.
    The persisted override (if amd-ttm has set it) is in
    /etc/modprobe.d/ttm.conf. We prefer the live value because that is what
    the running kernel is actually enforcing.
    """
    live = _read_text("/sys/module/ttm/parameters/pages_limit").strip()
    if live.isdigit():
        return int(live), "/sys/module/ttm/parameters/pages_limit"
    persisted = _read_text("/etc/modprobe.d/ttm.conf")
    m = re.search(r"pages_limit\s*=\s*(\d+)", persisted)
    if m:
        return int(m.group(1)), "/etc/modprobe.d/ttm.conf"
    return None, ""


def _total_ram_gb_linux() -> float | None:
    txt = _read_text("/proc/meminfo")
    m = re.search(r"^MemTotal:\s+(\d+)\s+kB", txt, re.MULTILINE)
    return round(int(m.group(1)) / (1024 * 1024), 2) if m else None


def _vram_carveout_linux() -> float | None:
    """Best-effort read of the BIOS VRAM carve-out from dmesg.

    The amdgpu driver logs a line like:
        amdgpu 0000:c4:00.0: amdgpu: VRAM: 512M ...
    on init. We need root for dmesg on most distros; if the read fails we
    return None and let the caller report "unknown".
    """
    rc, out, _ = _run(["dmesg"])
    if rc != 0:
        # Some distros restrict dmesg to root. Try journalctl as a fallback.
        rc, out, _ = _run(["journalctl", "-k", "--no-pager"])
        if rc != 0:
            return None
    for line in out.splitlines():
        m = re.search(
            r"amdgpu .*VRAM:\s*(\d+)([KMG])", line, re.IGNORECASE,
        )
        if m:
            value = int(m.group(1))
            unit = m.group(2).upper()
            mult = {"K": 1 / (1024 * 1024), "M": 1 / 1024, "G": 1.0}[unit]
            return round(value * mult, 3)
    return None


def _rocminfo_global_gb() -> float | None:
    """Sum the GLOBAL pool sizes for the GPU agent reported by rocminfo.

    Matches lines like `Size: 65535996 (0x3e7fffc) KB` inside a `Pool 1`
    block whose `Segment:` is `GLOBAL`. Only counts the GPU agent's pool,
    skipping the CPU agent.
    """
    if shutil.which("rocminfo") is None:
        return None
    rc, out, _ = _run(["rocminfo"])
    if rc != 0:
        return None
    in_gpu = False
    in_global_pool = False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Device Type:"):
            in_gpu = "GPU" in s
            in_global_pool = False
        if in_gpu and s.startswith("Segment:"):
            in_global_pool = "GLOBAL" in s
        if in_gpu and in_global_pool and s.startswith("Size:"):
            m = re.match(r"Size:\s+(\d+)\(.*?\)\s+KB", s) or re.match(
                r"Size:\s+(\d+)\s+KB", s
            )
            if m:
                return round(int(m.group(1)) / (1024 * 1024), 2)
    return None


def _query_windows_powershell(snippet: str, timeout: float = 10.0) -> str:
    """Run a PowerShell one-liner and return stdout (stripped)."""
    rc, out, _ = _run(["powershell", "-NoProfile", "-Command", snippet], timeout=timeout)
    return out.strip() if rc == 0 else ""


def _total_ram_gb_windows() -> float | None:
    val = _query_windows_powershell(
        "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"
    )
    if val.isdigit():
        return round(int(val) / (1024 ** 3), 2)
    return None


def _vram_carveout_windows() -> tuple[float | None, str | None]:
    """Return (gb, note). Note explains a WDDM-truncated reading."""
    val = _query_windows_powershell(
        "(Get-CimInstance Win32_VideoController | "
        "Where-Object { $_.Name -like '*AMD*' -or $_.Name -like '*Radeon*' } | "
        "Select-Object -First 1).AdapterRAM"
    )
    if not val.isdigit():
        return None, None
    raw = int(val)
    if raw >= WIN_ADAPTER_RAM_CAP:
        return round(raw / (1024 ** 3), 2), (
            "AdapterRAM is reporting the WDDM 4 GiB cap. The real BIOS "
            "carve-out may be larger; check 'Dedicated GPU memory' in Task "
            "Manager > Performance > GPU for the unclamped value."
        )
    return round(raw / (1024 ** 3), 2), None


def _shared_gpu_gb_windows() -> float | None:
    """Parse `dxdiag /t` for the 'Shared Memory' line.

    dxdiag writes its output to a temp file we have to read back. PowerShell
    is the cleanest way to wait on it without spinlocking.
    """
    snippet = (
        "$tmp = Join-Path $env:TEMP 'apu_dxdiag.txt'; "
        "Start-Process -FilePath dxdiag -ArgumentList '/t', $tmp -Wait; "
        "Get-Content $tmp -Raw"
    )
    body = _query_windows_powershell(snippet, timeout=30.0)
    if not body:
        return None
    m = re.search(r"Shared Memory:\s*(\d+(?:\.\d+)?)\s*MB", body)
    if not m:
        return None
    return round(float(m.group(1)) / 1024, 2)


def collect() -> Config:
    cfg = Config(notes=[])
    sysname = platform.system().lower()
    if sysname == "linux":
        cfg.os_family = "linux"
        cfg.total_ram_gb = _total_ram_gb_linux()
        pages, source = _ttm_pages_limit()
        cfg.ttm_pages_limit = pages
        cfg.ttm_source_file = source
        if pages is not None:
            cfg.shared_gpu_gb = round(pages * PAGE_SIZE_BYTES / (1024 ** 3), 2)
        cfg.dedicated_vram_gb = _vram_carveout_linux()
        if cfg.dedicated_vram_gb is None:
            cfg.notes.append(
                "Could not read the BIOS VRAM carve-out from dmesg/journalctl "
                "(usually requires root). Try `sudo dmesg | grep VRAM`."
            )
        cfg.rocminfo_global_gb = _rocminfo_global_gb()
    elif sysname == "windows":
        cfg.os_family = "windows"
        cfg.total_ram_gb = _total_ram_gb_windows()
        cfg.dedicated_vram_gb, vram_note = _vram_carveout_windows()
        if vram_note:
            cfg.notes.append(vram_note)
        cfg.shared_gpu_gb = _shared_gpu_gb_windows()
        cfg.notes.append(
            "Windows does not expose a user-tunable shared-memory cap; the "
            "value above is the WDDM-managed limit (typically ~50% of RAM)."
        )
    else:
        cfg.os_family = "other"
        cfg.notes.append(f"Unsupported OS family: {sysname}.")
    return cfg


def _fmt_gb(value: float | None) -> str:
    return f"{value:.2f} GB" if isinstance(value, (int, float)) else "unknown"


def _print_human(c: Config) -> None:
    print("APU memory tuner -- current configuration")
    print("-" * 40)
    print(f"OS:                  {c.os_family}")
    print(f"Total RAM:           {_fmt_gb(c.total_ram_gb)}")
    print(f"Dedicated VRAM:      {_fmt_gb(c.dedicated_vram_gb)}  (BIOS carve-out, static)")
    print(f"Shared GPU memory:   {_fmt_gb(c.shared_gpu_gb)}  (GTT/WDDM cap, dynamic)")
    if c.os_family == "linux":
        if c.ttm_pages_limit is not None:
            print(
                f"TTM pages_limit:     {c.ttm_pages_limit} pages "
                f"(source: {c.ttm_source_file})"
            )
        else:
            print("TTM pages_limit:     unknown")
        if c.rocminfo_global_gb is not None:
            print(f"rocminfo GPU pool:   {_fmt_gb(c.rocminfo_global_gb)}  (runtime sanity check)")

    print()
    if c.total_ram_gb and (c.dedicated_vram_gb is not None or c.shared_gpu_gb is not None):
        vram = c.dedicated_vram_gb or 0.0
        shared = c.shared_gpu_gb or 0.0
        cpu_only = max(c.total_ram_gb - vram, 0.0)
        print(
            f"Verdict: {vram:.2f} GB dedicated VRAM, {shared:.2f} GB shared, "
            f"~{cpu_only:.2f} GB available to CPU-only workloads, "
            f"{c.total_ram_gb:.2f} GB total."
        )
    for note in c.notes or []:
        print(f"  note: {note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human summary.",
    )
    args = parser.parse_args(argv)
    cfg = collect()
    if args.json:
        print(json.dumps(asdict(cfg), indent=2))
    else:
        _print_human(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
