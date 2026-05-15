#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Apply an AMD APU memory tuning profile, or print BIOS guidance.

Usage:
    python scripts/apply_profile.py --profile large-models
    python scripts/apply_profile.py --profile balanced
    python scripts/apply_profile.py --profile graphics
    python scripts/apply_profile.py --profile reset
    python scripts/apply_profile.py --profile custom --gtt-gb 100 --vram-gb 0.5
    python scripts/apply_profile.py --profile large-models --dry-run

Profiles map the user's high-level intent to concrete numbers:

  large-models  Maximum shared GPU memory; minimum BIOS carve-out.
                For LLM inference, large image-gen, training.
                GTT  = 75% of total RAM.
                VRAM = 0.5 GB (smallest most BIOSes allow).

  balanced      Default-ish split for mixed dev work.
                GTT  = 50% of total RAM (kernel default).
                VRAM = 1 GB.

  graphics      Reserve more VRAM for predictable framebuffer (gaming).
                GTT  = 50% of total RAM.
                VRAM = max(8, total_ram * 0.25) GB.

  reset         Revert any change this skill made.
                Linux: `amd-ttm --clear`.
                Windows: instruct user to set UMA Frame Buffer Size to Auto.

  custom        Use the explicit --gtt-gb / --vram-gb the user passed.

What this script CAN do automatically:
  Linux: run `amd-ttm --set <N>` (writes /etc/modprobe.d/ttm.conf).
         Reboot is still needed; we never auto-reboot.

What this script will NEVER do:
  - Modify or flash BIOS / firmware.
  - Edit Windows registry keys controlling VRAM (driver-managed and risky).
  - Reboot the machine.
  - Install packages without an explicit confirmation flag.

For BIOS-side changes (the only knob for the dedicated VRAM carve-out on
both OSes), this script prints step-by-step instructions and exits.
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
from dataclasses import dataclass
from pathlib import Path

PAGE_SIZE_BYTES = 4096
MIN_VRAM_GB = 0.5    # Floor most BIOSes allow for the UMA frame buffer.
MIN_GTT_GB = 1.0     # Below this, even routine GPU work fails.

# `amd-ttm` ships in the `amd-debug-tools` PyPI package. We never install
# silently; instead we print this command for the user to run.
AMD_TTM_INSTALL_CMD = "pipx install amd-debug-tools"


@dataclass
class ProfileTargets:
    name: str
    gtt_gb: float | None       # None = leave alone (graphics keeps default)
    vram_gb: float | None      # None = leave at firmware default
    rationale: str


def _run(cmd: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return r.returncode, r.stdout or "", r.stderr or ""
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
        return 127, "", str(e)


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _total_ram_gb_linux() -> float | None:
    txt = _read_text("/proc/meminfo")
    m = re.search(r"^MemTotal:\s+(\d+)\s+kB", txt, re.MULTILINE)
    return round(int(m.group(1)) / (1024 * 1024), 2) if m else None


def _total_ram_gb_windows() -> float | None:
    rc, out, _ = _run([
        "powershell", "-NoProfile", "-Command",
        "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
    ], timeout=8)
    if rc == 0 and out.strip().isdigit():
        return round(int(out.strip()) / (1024 ** 3), 2)
    return None


def total_ram_gb() -> float | None:
    sysname = platform.system().lower()
    if sysname == "linux":
        return _total_ram_gb_linux()
    if sysname == "windows":
        return _total_ram_gb_windows()
    return None


def resolve_profile(
    name: str,
    total_gb: float | None,
    custom_gtt: float | None,
    custom_vram: float | None,
) -> ProfileTargets:
    """Map a profile name + system RAM into concrete GTT/VRAM targets.

    `total_gb` is required for the percentage-based profiles. If we don't
    know it, we degrade to a conservative absolute number (16 GB GTT) so the
    user still gets a useful suggestion, and we annotate the rationale.
    """
    if name == "custom":
        return ProfileTargets(
            name="custom",
            gtt_gb=custom_gtt,
            vram_gb=custom_vram,
            rationale="User-specified values.",
        )

    if name == "reset":
        return ProfileTargets(
            name="reset",
            gtt_gb=None,
            vram_gb=None,
            rationale="Revert to firmware/kernel defaults.",
        )

    if total_gb is None:
        # Fallback when /proc/meminfo or CIM probe failed. Better to give a
        # reasonable absolute number than to crash; the user can override with
        # --profile custom.
        if name == "large-models":
            return ProfileTargets(
                "large-models", 32.0, MIN_VRAM_GB,
                "Total RAM unknown; falling back to 32 GB GTT, 0.5 GB VRAM.",
            )
        if name == "balanced":
            return ProfileTargets(
                "balanced", 16.0, 1.0,
                "Total RAM unknown; falling back to 16 GB GTT, 1 GB VRAM.",
            )
        if name == "graphics":
            return ProfileTargets(
                "graphics", None, 8.0,
                "Total RAM unknown; reserving 8 GB VRAM, leaving GTT at default.",
            )

    if name == "large-models":
        return ProfileTargets(
            "large-models",
            gtt_gb=round(total_gb * 0.75, 1),
            vram_gb=MIN_VRAM_GB,
            rationale=(
                f"75% of {total_gb:.0f} GB RAM as GTT, minimum BIOS carve-out. "
                "Maximizes memory available to LLMs, image-gen, training."
            ),
        )
    if name == "balanced":
        return ProfileTargets(
            "balanced",
            gtt_gb=round(total_gb * 0.50, 1),
            vram_gb=1.0,
            rationale=(
                f"50% of {total_gb:.0f} GB RAM as GTT, 1 GB BIOS carve-out. "
                "Mirrors kernel/driver defaults; good for mixed dev work."
            ),
        )
    if name == "graphics":
        vram = max(8.0, round(total_gb * 0.25, 1))
        return ProfileTargets(
            "graphics",
            gtt_gb=round(total_gb * 0.50, 1),
            vram_gb=vram,
            rationale=(
                f"{vram:.0f} GB BIOS carve-out for predictable framebuffer; "
                "GTT left near default. Tuned for gaming."
            ),
        )
    raise ValueError(f"Unknown profile: {name}")


def _validate_targets(t: ProfileTargets, total_gb: float | None) -> list[str]:
    errs: list[str] = []
    if t.gtt_gb is not None and t.gtt_gb < MIN_GTT_GB:
        errs.append(f"GTT target {t.gtt_gb} GB is below the {MIN_GTT_GB} GB floor.")
    if t.vram_gb is not None and t.vram_gb < MIN_VRAM_GB:
        errs.append(
            f"VRAM target {t.vram_gb} GB is below the {MIN_VRAM_GB} GB floor "
            "most BIOSes allow."
        )
    if total_gb is not None and t.gtt_gb is not None and t.gtt_gb > total_gb * 0.95:
        errs.append(
            f"GTT target {t.gtt_gb} GB is >95% of total RAM ({total_gb} GB); "
            "leaves no headroom for the kernel and CPU processes."
        )
    if total_gb is not None and t.vram_gb is not None and t.vram_gb > total_gb * 0.5:
        errs.append(
            f"VRAM target {t.vram_gb} GB is >50% of total RAM ({total_gb} GB); "
            "permanently reserves more than half the machine for the GPU."
        )
    return errs


def _print_targets(t: ProfileTargets) -> None:
    print(f"Profile:    {t.name}")
    print(f"Rationale:  {t.rationale}")
    print(f"Target GTT: {t.gtt_gb if t.gtt_gb is not None else 'unchanged'}"
          + (" GB" if t.gtt_gb is not None else ""))
    print(f"Target VRAM (BIOS carve-out): "
          + (f"{t.vram_gb} GB" if t.vram_gb is not None else "unchanged"))
    print()


def apply_linux(t: ProfileTargets, dry_run: bool) -> int:
    """Apply the GTT half on Linux via amd-ttm; print VRAM-side guidance."""
    if t.name == "reset":
        if shutil.which("amd-ttm") is None:
            print("amd-ttm not found; nothing to revert. Install with:")
            print(f"  {AMD_TTM_INSTALL_CMD}")
            return 0
        cmd = ["amd-ttm", "--clear"]
        print("Will run:", " ".join(cmd))
        if dry_run:
            print("(dry-run; not executed)")
            return 0
        rc, out, err = _run(cmd, timeout=15)
        sys.stdout.write(out)
        sys.stderr.write(err)
        if rc == 0:
            print("Reverted. Reboot for the kernel to pick up the default.")
        return rc

    if t.gtt_gb is not None:
        if shutil.which("amd-ttm") is None:
            print("ERROR: amd-ttm is required to set the GTT/shared cap on Linux.")
            print("Install it with:")
            print(f"  {AMD_TTM_INSTALL_CMD}")
            print("Then re-run this script.")
            return 4
        # amd-ttm takes integer GB; round down so we never silently overshoot
        # into a value the kernel rejects on the next boot.
        gb_int = int(t.gtt_gb)
        cmd = ["amd-ttm", "--set", str(gb_int)]
        print("Will run:", " ".join(cmd))
        if dry_run:
            print("(dry-run; not executed)")
        else:
            rc, out, err = _run(cmd, timeout=15)
            sys.stdout.write(out)
            sys.stderr.write(err)
            if rc != 0:
                print(f"amd-ttm exited {rc}; the change was NOT persisted.")
                return rc
            print(
                f"GTT/shared cap set to {gb_int} GB. "
                "Reboot for the change to take effect."
            )

    if t.vram_gb is not None:
        print()
        print("BIOS-side change required for the dedicated VRAM carve-out:")
        print(f"  Set 'UMA Frame Buffer Size' (or equivalent) to {t.vram_gb} GB")
        print("  in the BIOS. Reboot, then re-run scripts/show_config.py")
        print("  to verify.")
        print()
        print("Common BIOS paths:")
        print("  Advanced > AMD CBS > NBIO Common Options > GFX Configuration > UMA Frame Buffer Size")
        print("  Advanced > AMD Overclocking > UMA Frame Buffer Size")
        print()
        print("This script will NOT change BIOS for you.")
    return 0


def apply_windows(t: ProfileTargets, dry_run: bool) -> int:
    """Windows is BIOS-only for the meaningful knobs; print guidance."""
    if t.name == "reset":
        print("To revert APU memory settings on Windows:")
        print("  1. Reboot and enter BIOS (Del/F2/F10 depending on OEM).")
        print("  2. Set 'UMA Frame Buffer Size' back to 'Auto' (or your")
        print("     OEM's default).")
        print("  3. Save & exit.")
        print("  4. If you previously raised VRAM via AMD Adrenalin's")
        print("     'Variable Graphics Memory' slider, set it back to default.")
        return 0

    print("Windows does not expose the GTT/shared-memory cap as a user-tunable")
    print("knob; the WDDM driver picks it (typically ~50% of RAM). The only")
    print("lever you have is the BIOS UMA Frame Buffer Size, which this")
    print("script will NOT change for you.")
    print()
    if t.vram_gb is not None:
        print(f"Recommended BIOS UMA Frame Buffer Size: {t.vram_gb} GB")
    elif t.gtt_gb is not None:
        # The user wanted more shared memory but we have nothing to set on
        # Windows. Surface the gap honestly.
        print(
            f"Profile asked for {t.gtt_gb} GB shared GPU memory, but Windows "
            "does not let you raise the WDDM shared cap directly. To get more "
            "GPU-visible memory, raise the BIOS UMA Frame Buffer Size instead "
            f"(suggested: {min(t.gtt_gb, 64.0)} GB)."
        )
    print()
    print("BIOS steps (OEM-agnostic):")
    print("  1. Reboot, press your BIOS key (Del/F2/F10/Esc; varies by OEM).")
    print("  2. Find 'UMA Frame Buffer Size' (sometimes 'Dedicated GPU')")
    print("     under one of:")
    print("       Advanced > AMD CBS > NBIO Common Options > GFX Configuration")
    print("       Advanced > AMD Overclocking")
    print("       Chipset > North Bridge")
    print("  3. Set the value, save & exit.")
    print()
    print("Alternative on supported AMD laptops: AMD Adrenalin Software")
    print("(System > Hardware > Variable Graphics Memory). VGM behaves")
    print("similarly to a BIOS carve-out and is reset on reboot if you change")
    print("your mind.")
    if dry_run:
        print("(dry-run mode -- no commands were going to run on Windows anyway)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["large-models", "balanced", "graphics", "reset", "custom"],
        required=True,
        help="High-level intent to map to concrete VRAM/GTT numbers.",
    )
    parser.add_argument(
        "--gtt-gb", type=float, default=None,
        help="Custom GTT/shared cap in GB (only for --profile custom).",
    )
    parser.add_argument(
        "--vram-gb", type=float, default=None,
        help="Custom BIOS VRAM carve-out in GB (only for --profile custom).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned change without executing it.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON of the resolved profile and exit.",
    )
    args = parser.parse_args(argv)

    if args.profile == "custom" and args.gtt_gb is None and args.vram_gb is None:
        parser.error("--profile custom requires --gtt-gb and/or --vram-gb")

    total_gb = total_ram_gb()
    targets = resolve_profile(args.profile, total_gb, args.gtt_gb, args.vram_gb)
    errs = _validate_targets(targets, total_gb)

    if args.json:
        print(json.dumps({
            "profile": targets.name,
            "rationale": targets.rationale,
            "target_gtt_gb": targets.gtt_gb,
            "target_vram_gb": targets.vram_gb,
            "total_ram_gb": total_gb,
            "validation_errors": errs,
        }, indent=2))
        return 0 if not errs else 5

    _print_targets(targets)
    if errs:
        print("Validation failed:")
        for e in errs:
            print(f"  - {e}")
        print("Refusing to apply. Use --profile custom with safer numbers.")
        return 5

    sysname = platform.system().lower()
    if sysname == "linux":
        rc = apply_linux(targets, args.dry_run)
    elif sysname == "windows":
        rc = apply_windows(targets, args.dry_run)
    else:
        print(f"Unsupported OS: {sysname}. This skill targets Linux and Windows.")
        return 2

    print()
    print("To verify after reboot, run:")
    print("  python scripts/show_config.py")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
