#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Apply a low-risk fix proposed by `diagnose.py`, or print the plan.

This is the ONLY rocm-doctor script that can change the system. Every
diagnosis from `diagnose.py` carries a stable `fix_id`; pass it here:

    python scripts/apply_fix.py --fix-id fix-4-render-group
    python scripts/apply_fix.py --fix-id fix-2-unset-override --dry-run
    python scripts/apply_fix.py --list

`--dry-run` is the default safety hatch: it prints the planned commands
and exits 0 without executing anything. Use it to show the user exactly
what would change.

When a fix has `auto_applicable=False` (most of the structural fixes:
kernel-module blacklist, repo cleanup, multi-GPU IOMMU, amdgpu-install
rebuild), this script prints the commands and exits 0 without running
them, even without `--dry-run`. The user has to copy-paste, because the
risk of a half-applied state is too high for a tool to take.

Each recipe carries an `applies_on` set of os_family values. `main` refuses
with exit 3 when the running OS isn't in that set, replacing the per-runner
platform.system() checks. Linux-only recipes (fix-3, -4, -5, -7, -10, -11,
-12) refuse on Windows; Windows-only recipes (fix-13, -14, -15) refuse on
Linux; the rest are cross-platform.

Exit codes:
  0 = success (or dry-run finished, or fix is advisory-only).
  2 = unknown --fix-id.
  3 = required environment is missing (e.g. fix needs `sudo` and there's no
      sudo, or fix doesn't apply to the running OS).
  4 = the underlying command exited non-zero; nothing was rolled back.
  5 = user declined the change at the interactive prompt.

Design constraints:
  - Never run anything `sudo` without printing the command first.
  - Never modify the Windows registry, BIOS, or kernel cmdline non-interactively.
  - Never restart services or reboot the machine.
  - Never reinstall packages without an explicit --yes flag.
  - Never silently fall through to an unrelated fix because the requested
    one wasn't applicable -- exit 3 and tell the user why.
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
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FixRecipe:
    fix_id: str
    title: str
    rationale: str
    auto_applicable: bool          # True iff we can run the commands ourselves
    commands: list[str] = field(default_factory=list)
    needs_sudo: bool = False
    needs_reboot: bool = False
    needs_relogin: bool = False
    verify: str = ""
    notes: list[str] = field(default_factory=list)
    # OS families this recipe applies on. `main` refuses (exit 3) when the
    # running OS isn't in this set, replacing the per-runner platform.system()
    # checks that used to live in each runner.
    applies_on: frozenset[str] = field(default_factory=lambda: frozenset({"linux"}))
    # When auto_applicable, this callable runs the actual change. It's
    # invoked with (args, recipe) and must return an int exit code. We
    # split this off from `commands` so we can compose multi-step actions
    # (e.g. usermod followed by checking the resulting group list) without
    # shelling out to bash.
    runner: object = None          # Callable[[argparse.Namespace, FixRecipe], int]


def _run(cmd: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return r.returncode, r.stdout or "", r.stderr or ""
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        return 127, "", str(exc)


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        # Non-interactive context (CI, agent harness). Refuse to apply
        # without explicit --yes; printing the plan is enough.
        print("Non-interactive shell and --yes not passed; refusing to apply.")
        return False
    try:
        ans = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def _print_recipe(r: FixRecipe) -> None:
    print(f"Fix:        {r.fix_id}  -- {r.title}")
    print(f"OS scope:   {', '.join(sorted(r.applies_on))}")
    print(f"Rationale:  {r.rationale}")
    if r.commands:
        print("Commands:")
        for c in r.commands:
            print(f"  $ {c}")
    flags = []
    if r.needs_sudo: flags.append("requires sudo")
    if r.needs_reboot: flags.append("requires reboot")
    if r.needs_relogin: flags.append("requires re-login")
    if not r.auto_applicable: flags.append("manual only (apply_fix.py will NOT run it)")
    if flags:
        print(f"Flags:      {', '.join(flags)}")
    for n in r.notes:
        print(f"Note:       {n}")
    if r.verify:
        print(f"Verify:     {r.verify}")


# ---------------------------------------------------------------------------
# Runners. One per auto-applicable fix.
#
# Each runner returns the process exit code. It must:
#   - Refuse to act when the platform isn't right (return 3).
#   - Print every command it runs.
#   - Respect args.dry_run.
#   - Respect args.yes (skip the interactive confirm).
# ---------------------------------------------------------------------------

def run_render_group(args, recipe: FixRecipe) -> int:
    """fix-4: add the current user to the render group (and 'video' for safety)."""
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    if not user:
        print("Could not determine current user from $USER/$LOGNAME.")
        return 3
    if not _have("usermod"):
        print("`usermod` not on PATH; cannot add groups.")
        return 3
    if not _have("sudo") and os.geteuid() != 0:
        print("`sudo` is not on PATH and we are not root; cannot add groups.")
        return 3

    cmd_prefix = [] if os.geteuid() == 0 else ["sudo"]
    cmd = cmd_prefix + ["usermod", "-a", "-G", "render,video", user]
    print("Will run:", " ".join(cmd))
    if args.dry_run:
        print("(dry-run; not executed)")
        return 0
    if not _confirm("Add user to render,video groups?", args.yes):
        return 5
    rc, out, err = _run(cmd, timeout=20)
    if out: sys.stdout.write(out)
    if err: sys.stderr.write(err)
    if rc != 0:
        print(f"usermod exited {rc}; group membership NOT changed.")
        return 4
    print(f"Added {user} to render,video.")
    print(
        "IMPORTANT: log out and back in (or reboot) for the membership to "
        "take effect in new shells and services. `newgrp render` patches "
        "the current shell only."
    )
    return 0


def run_unset_override(args, recipe: FixRecipe) -> int:
    """fix-2: unset HSA_OVERRIDE_GFX_VERSION for future shells.

    We can only affect THIS process. Persisting the unset requires editing
    user dotfiles (Linux) or the per-user environment registry (Windows),
    which we never do unannounced. We instead:
      Linux:
        1. Inspect ~/.bashrc, ~/.zshrc, ~/.profile, ~/.config/fish/config.fish
           for an `export HSA_OVERRIDE_GFX_VERSION=...` line.
        2. Print exact $EDITOR instructions for any hit.
      Windows:
        1. Read the User and Machine env scopes via PowerShell.
        2. Tell the user which scope still holds the value and how to clear
           it (`setx HSA_OVERRIDE_GFX_VERSION ""` or System Properties UI).
    """
    if platform.system().lower() == "windows":
        return _run_unset_override_windows(args, recipe)
    return _run_unset_override_linux(args, recipe)


def _run_unset_override_linux(args, recipe: FixRecipe) -> int:
    current = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
    if not current:
        print("HSA_OVERRIDE_GFX_VERSION is already unset in this shell.")
    else:
        print(f"HSA_OVERRIDE_GFX_VERSION={current} is set in this shell.")
        print("In your current shell, run:")
        print("  unset HSA_OVERRIDE_GFX_VERSION")
        print("(This script can't unset it in your parent shell; it only sees a copy.)")

    candidates = [
        Path.home() / ".bashrc",
        Path.home() / ".bash_profile",
        Path.home() / ".zshrc",
        Path.home() / ".profile",
        Path.home() / ".config" / "fish" / "config.fish",
    ]
    rc_hits: list[Path] = []
    for f in candidates:
        if not f.exists():
            continue
        try:
            body = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if re.search(r"HSA_OVERRIDE_GFX_VERSION", body):
            rc_hits.append(f)

    if not rc_hits:
        print("\nNo persistent HSA_OVERRIDE_GFX_VERSION found in your shell rc files.")
        return 0

    print("\nPersistent HSA_OVERRIDE_GFX_VERSION found in:")
    for f in rc_hits:
        print(f"  - {f}")
    print(
        "\nRemove or comment those lines manually. apply_fix.py does NOT edit "
        "your shell rc files for you; that's your dotfiles. Suggested:"
    )
    for f in rc_hits:
        print(f"  $ $EDITOR {f}   # delete or comment the HSA_OVERRIDE_GFX_VERSION line")
    return 0


def _run_unset_override_windows(args, recipe: FixRecipe) -> int:
    current = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
    if current:
        print(f"HSA_OVERRIDE_GFX_VERSION={current} is set in this shell.")
        print("Note: clearing it in your Windows env scope does NOT affect this")
        print("already-open shell -- close and reopen your terminal afterwards.")
    else:
        print("HSA_OVERRIDE_GFX_VERSION is not set in this shell.")

    user_val = ""
    machine_val = ""
    rc, out, _ = _run([
        "powershell", "-NoProfile", "-Command",
        "[Environment]::GetEnvironmentVariable('HSA_OVERRIDE_GFX_VERSION','User')",
    ], timeout=8)
    if rc == 0:
        user_val = out.strip()
    rc, out, _ = _run([
        "powershell", "-NoProfile", "-Command",
        "[Environment]::GetEnvironmentVariable('HSA_OVERRIDE_GFX_VERSION','Machine')",
    ], timeout=8)
    if rc == 0:
        machine_val = out.strip()

    if not user_val and not machine_val:
        print("\nNo persistent HSA_OVERRIDE_GFX_VERSION found in either the User")
        print("or Machine env scope. You're done after closing/reopening shells.")
        return 0

    print("\nPersistent HSA_OVERRIDE_GFX_VERSION found in:")
    if user_val:
        print(f"  User scope:    {user_val}")
    if machine_val:
        print(f"  Machine scope: {machine_val}")

    if user_val:
        print('\nClear from the User scope (no admin needed):')
        print('  Will run: setx HSA_OVERRIDE_GFX_VERSION ""')
        if args.dry_run:
            print("  (dry-run; not executed)")
        elif _confirm("Clear HSA_OVERRIDE_GFX_VERSION from User scope?", args.yes):
            rc, out, err = _run(["setx", "HSA_OVERRIDE_GFX_VERSION", ""], timeout=15)
            if out: sys.stdout.write(out)
            if err: sys.stderr.write(err)
            if rc != 0:
                print(f"setx exited {rc}; User scope NOT changed.")
                return 4
            print("Cleared from User scope. Reopen your terminal for it to take effect.")

    if machine_val:
        print(
            "\nThe Machine scope value cannot be cleared without an Admin shell. "
            "Either run an elevated PowerShell and execute:"
        )
        print("  [Environment]::SetEnvironmentVariable('HSA_OVERRIDE_GFX_VERSION', $null, 'Machine')")
        print(
            "or remove it through System Properties -> Environment Variables -> "
            "System variables. apply_fix.py does NOT elevate itself."
        )
    return 0


def run_path_export(args, recipe: FixRecipe) -> int:
    """fix-6: persist the ROCm/HIP bin directory on PATH (with consent)."""
    if platform.system().lower() == "windows":
        return _run_path_export_windows(args, recipe)
    return _run_path_export_linux(args, recipe)


def _run_path_export_linux(args, recipe: FixRecipe) -> int:
    """Append `/opt/rocm/bin` to ~/.bashrc (or ~/.zshrc).

    Simplest possible thing: append a single line. We never reorder PATH,
    we never edit /etc/environment. If the line is already there we exit
    0 without re-appending.
    """
    bin_dir = "/opt/rocm/bin"
    if not Path(bin_dir).is_dir():
        print(f"{bin_dir} does not exist; nothing to add to PATH.")
        return 3

    shell = os.environ.get("SHELL", "")
    rc_file = Path.home() / (".zshrc" if "zsh" in shell else ".bashrc")
    if not rc_file.exists() and (Path.home() / ".bashrc").exists():
        rc_file = Path.home() / ".bashrc"

    export_line = f'export PATH="{bin_dir}:$PATH"'
    existing = ""
    if rc_file.exists():
        try:
            existing = rc_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"Could not read {rc_file}: {exc}")
            return 3
        if re.search(rf"PATH=.*{re.escape(bin_dir)}", existing):
            print(f"{rc_file} already adds {bin_dir} to PATH; no change.")
            return 0

    print(f"Plan: append the following line to {rc_file}:")
    print(f"  {export_line}")
    if args.dry_run:
        print("(dry-run; not executed)")
        return 0
    if not _confirm(f"Append to {rc_file}?", args.yes):
        return 5

    try:
        with rc_file.open("a", encoding="utf-8") as fh:
            fh.write(f"\n# Added by rocm-doctor (apply_fix.py fix-6-path)\n")
            fh.write(export_line + "\n")
    except OSError as exc:
        print(f"Failed to write {rc_file}: {exc}")
        return 4

    print(
        f"Appended to {rc_file}. Open a new shell or run `source {rc_file}` "
        "for the change to take effect."
    )
    return 0


def _run_path_export_windows(args, recipe: FixRecipe) -> int:
    """Append the HIP SDK's bin directory to the User PATH via setx.

    `setx` is the only documented way to persist a User env var on
    Windows without elevation. It rewrites the whole variable; here we
    fetch the current User-scope PATH first, append our directory if it
    isn't there yet, and write the result back.
    """
    sdk_path = os.environ.get("HIP_PATH", "")
    if not sdk_path:
        for root in (r"C:\Program Files\AMD\ROCm", r"C:\Program Files (x86)\AMD\ROCm"):
            try:
                base = Path(root)
                if base.is_dir():
                    for child in sorted(base.iterdir(), reverse=True):
                        if child.is_dir() and re.match(r"\d+(\.\d+)+", child.name):
                            sdk_path = str(child)
                            break
            except OSError:
                continue
            if sdk_path:
                break
    if not sdk_path:
        print("No HIP SDK install found. Run fix-13-hip-sdk-missing first.")
        return 3
    bin_dir = str(Path(sdk_path) / "bin")
    if not Path(bin_dir).is_dir():
        print(f"{bin_dir} does not exist on disk; HIP SDK install looks incomplete.")
        return 3

    rc, out, _ = _run([
        "powershell", "-NoProfile", "-Command",
        "[Environment]::GetEnvironmentVariable('PATH','User')",
    ], timeout=8)
    user_path = out.strip() if rc == 0 else ""
    if user_path and bin_dir.lower() in user_path.lower():
        print(f"User PATH already contains {bin_dir}; no change.")
        return 0
    new_path = (user_path + ";" + bin_dir).lstrip(";") if user_path else bin_dir

    print(f"Plan: prepend {bin_dir} to your User PATH:")
    print(f"  setx PATH \"{new_path}\"")
    if args.dry_run:
        print("(dry-run; not executed)")
        return 0
    if not _confirm("Update User PATH?", args.yes):
        return 5

    rc, out, err = _run(["setx", "PATH", new_path], timeout=15)
    if out: sys.stdout.write(out)
    if err: sys.stderr.write(err)
    if rc != 0:
        print(f"setx exited {rc}; User PATH NOT changed.")
        return 4
    print(
        f"Added {bin_dir} to your User PATH. setx only takes effect in NEW "
        "shells -- close this terminal and reopen it before re-running hipInfo."
    )
    return 0


def run_hip_visible_devices(args, recipe: FixRecipe) -> int:
    """fix-9: persist HIP_VISIBLE_DEVICES so the iGPU is hidden.

    We DO NOT pick a device index automatically -- rocminfo / hipInfo
    ordering can surprise even experienced users on dual-GPU laptops.
    Instead, we print a guided query and accept --device-index as the
    explicit input.
    """
    if platform.system().lower() == "windows":
        return _run_hip_visible_devices_windows(args, recipe)
    return _run_hip_visible_devices_linux(args, recipe)


def _run_hip_visible_devices_linux(args, recipe: FixRecipe) -> int:
    idx = args.device_index
    if idx is None:
        print(
            "Run `rocminfo | grep -E 'Agent |Marketing|gfx'` and identify the "
            "row of your DISCRETE GPU (the iGPU is typically Agent 1). Then "
            "re-run apply_fix.py with --device-index N."
        )
        return 3

    shell = os.environ.get("SHELL", "")
    rc_file = Path.home() / (".zshrc" if "zsh" in shell else ".bashrc")
    if not rc_file.exists() and (Path.home() / ".bashrc").exists():
        rc_file = Path.home() / ".bashrc"

    export_line = f'export HIP_VISIBLE_DEVICES={idx}'
    if rc_file.exists():
        try:
            existing = rc_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"Could not read {rc_file}: {exc}")
            return 3
        if re.search(r"HIP_VISIBLE_DEVICES=", existing):
            print(
                f"{rc_file} already sets HIP_VISIBLE_DEVICES; edit by hand "
                "rather than appending a second copy."
            )
            return 0

    print(f"Plan: append the following line to {rc_file}:")
    print(f"  {export_line}")
    if args.dry_run:
        print("(dry-run; not executed)")
        return 0
    if not _confirm(f"Append to {rc_file}?", args.yes):
        return 5
    try:
        with rc_file.open("a", encoding="utf-8") as fh:
            fh.write("\n# Added by rocm-doctor (apply_fix.py fix-9-igpu-dgpu)\n")
            fh.write(export_line + "\n")
    except OSError as exc:
        print(f"Failed to write {rc_file}: {exc}")
        return 4
    print(
        f"Appended to {rc_file}. Open a new shell for the change to take effect, "
        "then re-run your workload."
    )
    return 0


def _run_hip_visible_devices_windows(args, recipe: FixRecipe) -> int:
    idx = args.device_index
    if idx is None:
        print(
            "Run the following to identify the discrete GPU's index:"
        )
        print(
            '  & "$env:HIP_PATH\\bin\\hipInfo.exe" | '
            'Select-String "device#|Name|gcnArchName"'
        )
        print(
            "Then re-run apply_fix.py with --device-index N (the iGPU is "
            "typically device# 0; the dGPU is usually device# 1)."
        )
        return 3

    rc, out, _ = _run([
        "powershell", "-NoProfile", "-Command",
        "[Environment]::GetEnvironmentVariable('HIP_VISIBLE_DEVICES','User')",
    ], timeout=8)
    existing = out.strip() if rc == 0 else ""
    if existing:
        print(
            f"User scope already sets HIP_VISIBLE_DEVICES={existing!r}; "
            "remove or update it manually rather than overwriting from this script."
        )
        return 0

    print("Plan: persist HIP_VISIBLE_DEVICES in the User env scope:")
    print(f"  setx HIP_VISIBLE_DEVICES {idx}")
    if args.dry_run:
        print("(dry-run; not executed)")
        return 0
    if not _confirm("Set HIP_VISIBLE_DEVICES in the User scope?", args.yes):
        return 5

    rc, out, err = _run(["setx", "HIP_VISIBLE_DEVICES", str(idx)], timeout=15)
    if out: sys.stdout.write(out)
    if err: sys.stderr.write(err)
    if rc != 0:
        print(f"setx exited {rc}; HIP_VISIBLE_DEVICES NOT changed.")
        return 4
    print(
        "setx only takes effect in NEW shells -- close this terminal and "
        "reopen it before re-running your workload."
    )
    return 0


# ---------------------------------------------------------------------------
# Recipe registry. Mirrors the diagnosis catalog in `diagnose.py`. Only the
# small, safe, well-bounded fixes are auto-applicable; everything else is
# advisory and prints the plan only.
# ---------------------------------------------------------------------------

LINUX_AND_WINDOWS = frozenset({"linux", "windows"})
LINUX_ONLY = frozenset({"linux"})
WINDOWS_ONLY = frozenset({"windows"})


RECIPES: dict[str, FixRecipe] = {
    "fix-1-arch": FixRecipe(
        fix_id="fix-1-arch",
        title="GPU gfx target not in framework arch list",
        rationale=(
            "Your GPU's gfx target is not in the framework wheel's compiled "
            "kernel list. Re-install the framework from an index that includes "
            "this gfx, OR rebuild llama.cpp with AMDGPU_TARGETS=<gfx>."
        ),
        auto_applicable=False,
        applies_on=LINUX_AND_WINDOWS,
        commands=[
            "# PyTorch (Linux): switch to the ROCm nightly that ships the gfx115x kernels.",
            "pip uninstall -y torch torchvision torchaudio",
            "pip install --pre torch torchvision torchaudio \\",
            "  --index-url https://download.pytorch.org/whl/nightly/rocm6.4",
            "# PyTorch (Windows): use TheRock's per-gfx wheels (https://github.com/ROCm/TheRock).",
            "# llama.cpp:",
            "# cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=<your_gfx_target>",
            "# cmake --build build -j",
        ],
        notes=[
            "TheRock per-gfx wheels are the recommended fallback when the "
            "official pytorch index does not yet cover your gfx (and the only "
            "first-party option on Windows AMD).",
            "HSA_OVERRIDE_GFX_VERSION is NOT the right fix here -- it papers "
            "over the mismatch and risks page faults at runtime.",
        ],
        verify="python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())\"",
    ),
    "fix-2-unset-override": FixRecipe(
        fix_id="fix-2-unset-override",
        title="Unset HSA_OVERRIDE_GFX_VERSION",
        rationale=(
            "HSA_OVERRIDE_GFX_VERSION is set, but your GPU now has a native "
            "wheel. The override hides the real gfx and causes page faults / "
            "OUT_OF_REGISTERS at runtime."
        ),
        auto_applicable=True,
        applies_on=LINUX_AND_WINDOWS,
        commands=[
            "# Linux:",
            "unset HSA_OVERRIDE_GFX_VERSION",
            "# Then remove the line from ~/.bashrc / ~/.zshrc / ~/.profile.",
            "# Windows:",
            'setx HSA_OVERRIDE_GFX_VERSION ""',
            "# Or remove via System Properties -> Environment Variables.",
        ],
        runner=run_unset_override,
        verify="env | grep HSA_OVERRIDE_GFX_VERSION || echo OK_UNSET",
    ),
    "fix-3-rocm-kernel": FixRecipe(
        fix_id="fix-3-rocm-kernel",
        title="ROCm/distro/kernel triple unsupported",
        rationale=(
            "ROCm is installed but your kernel/distro combination is outside "
            "the supported matrix. Match the kernel to the matrix before "
            "reinstalling, or rerun with --no-dkms and accept the risk."
        ),
        auto_applicable=False,
        applies_on=LINUX_ONLY,
        commands=[
            "# Cross-check the live AMD matrix before changing anything:",
            "#   https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html",
            "# Common fix on Ubuntu: install the HWE kernel that matches your ROCm release, then reboot.",
        ],
        needs_reboot=True,
        verify="lsmod | grep amdgpu && rocminfo | head -n 5",
    ),
    "fix-4-render-group": FixRecipe(
        fix_id="fix-4-render-group",
        title="Add user to render/video groups",
        rationale=(
            "The current user can't open /dev/kfd because they aren't in the "
            "render group. Adding the user is the safe, standard fix."
        ),
        auto_applicable=True,
        applies_on=LINUX_ONLY,
        commands=['sudo usermod -a -G render,video "$USER"'],
        needs_sudo=True,
        needs_relogin=True,
        runner=run_render_group,
        verify="groups | tr ' ' '\\n' | grep -E '^(render|video)$' && rocminfo | head -n 5",
    ),
    "fix-5-amdgpu-load": FixRecipe(
        fix_id="fix-5-amdgpu-load",
        title="Load amdgpu (and clear any blacklist)",
        rationale=(
            "The amdgpu kernel module is not loaded. Check /etc/modprobe.d "
            "for a blacklist entry, regenerate the initramfs, and modprobe."
        ),
        auto_applicable=False,
        applies_on=LINUX_ONLY,
        commands=[
            "grep -RIl 'blacklist amdgpu' /etc/modprobe.d /usr/lib/modprobe.d 2>/dev/null || true",
            "sudo $EDITOR <file shown above>     # remove the blacklist line",
            "sudo update-initramfs -u            # Debian/Ubuntu",
            "sudo dracut -f                      # Fedora/RHEL",
            "sudo modprobe amdgpu",
        ],
        needs_sudo=True,
        needs_reboot=True,
        verify="lsmod | grep amdgpu && rocminfo | head -n 5",
        notes=[
            "If Secure Boot is enabled and amdgpu still won't load, the DKMS "
            "module isn't signed. Either sign it with mokutil or disable "
            "Secure Boot in firmware.",
        ],
    ),
    "fix-6-path": FixRecipe(
        fix_id="fix-6-path",
        title="Add the ROCm/HIP bin directory to PATH",
        rationale=(
            "Linux: ROCm is installed at /opt/rocm but its bin directory isn't "
            "on PATH, so `rocminfo` / `hipcc` aren't visible to the shell. "
            "Windows: the HIP SDK is installed but its bin directory isn't on "
            "the User PATH, so `hipInfo.exe` and the runtime DLLs can't be found."
        ),
        auto_applicable=True,
        applies_on=LINUX_AND_WINDOWS,
        commands=[
            "# Linux:",
            'echo \'export PATH="/opt/rocm/bin:$PATH"\' >> ~/.bashrc',
            "# Windows:",
            'setx PATH "%PATH%;C:\\Program Files\\AMD\\ROCm\\<version>\\bin"',
        ],
        runner=run_path_export,
        verify="rocminfo | head -n 5 && hipcc --version",
    ),
    "fix-7-stale-repos": FixRecipe(
        fix_id="fix-7-stale-repos",
        title="Quarantine duplicate AMD repos",
        rationale=(
            "More than one ROCm/AMDGPU repo file exists. The package manager "
            "is mixing versions; quarantine the extras before reinstalling."
        ),
        auto_applicable=False,
        applies_on=LINUX_ONLY,
        commands=[
            "ls /etc/apt/sources.list.d/ | grep -iE 'rocm|amdgpu|radeon'",
            "# For each duplicate file:",
            "sudo mv /etc/apt/sources.list.d/<file>.list /etc/apt/sources.list.d/<file>.list.bak",
            "sudo apt update",
        ],
        needs_sudo=True,
        verify="sudo apt update 2>&1 | tail -n 20",
    ),
    "fix-8-wheel-rocm": FixRecipe(
        fix_id="fix-8-wheel-rocm",
        title="Reinstall the framework against the system ROCm/HIP major",
        rationale=(
            "The framework's bundled HIP version doesn't match the system "
            "ROCm (Linux) or HIP SDK (Windows). libamdhip64.so.X / "
            "amdhip64_X.dll load failures are the usual signal."
        ),
        auto_applicable=False,
        applies_on=LINUX_AND_WINDOWS,
        commands=[
            "pip uninstall -y torch torchvision torchaudio",
            "# Linux: pick the index that matches your system ROCm major:",
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4",
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3",
            "# Windows: use TheRock's wheels matching your HIP SDK major:",
            "#   https://github.com/ROCm/TheRock",
        ],
        verify="python -c \"import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())\"",
    ),
    "fix-9-igpu-dgpu": FixRecipe(
        fix_id="fix-9-igpu-dgpu",
        title="Hide the iGPU with HIP_VISIBLE_DEVICES",
        rationale=(
            "Both an APU iGPU and a discrete AMD GPU are visible. Pin the "
            "runtime to the dGPU so the iGPU doesn't destabilise it."
        ),
        auto_applicable=True,
        applies_on=LINUX_AND_WINDOWS,
        commands=[
            "# Linux:",
            "rocminfo | grep -E 'Agent |Marketing|gfx'   # find the dGPU index",
            "export HIP_VISIBLE_DEVICES=<dGPU-index>",
            "# Windows:",
            '& "$env:HIP_PATH\\bin\\hipInfo.exe" | Select-String "device#|Name"',
            "setx HIP_VISIBLE_DEVICES <dGPU-index>",
        ],
        runner=run_hip_visible_devices,
        verify="python -c \"import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))\"",
        notes=[
            "Pass --device-index N to persist the env var; without it, "
            "this fix only prints the rocminfo / hipInfo query so you can identify N.",
        ],
    ),
    "fix-10-container": FixRecipe(
        fix_id="fix-10-container",
        title="Re-launch the container with AMD devices passed through",
        rationale=(
            "The container can't see /dev/kfd or /dev/dri/renderD*. Pass the "
            "devices and the host's render group via the runtime flags."
        ),
        auto_applicable=False,
        applies_on=LINUX_ONLY,
        commands=[
            "docker run --rm -it \\",
            "  --device=/dev/kfd \\",
            "  --device=/dev/dri \\",
            "  --group-add render \\",
            "  --security-opt seccomp=unconfined \\",
            "  --shm-size=8g \\",
            "  rocm/pytorch:latest",
        ],
        verify="rocminfo | head -n 5",
        notes=[
            "Rootless podman additionally needs `--userns=keep-id` and a "
            "host user that is in the render group; podman maps it through.",
        ],
    ),
    "fix-11-iommu": FixRecipe(
        fix_id="fix-11-iommu",
        title="Add iommu=pt to the kernel command line",
        rationale=(
            "Multi-GPU jobs hang when the IOMMU is in the default 'on' mode "
            "with translation; pass-through mode fixes the hang. This requires "
            "editing GRUB and rebooting; we will not do that for you."
        ),
        auto_applicable=False,
        applies_on=LINUX_ONLY,
        commands=[
            "cat /proc/cmdline",
            "sudo $EDITOR /etc/default/grub        # add iommu=pt to GRUB_CMDLINE_LINUX_DEFAULT",
            "sudo update-grub                       # Debian/Ubuntu",
            "sudo grub2-mkconfig -o /boot/grub2/grub.cfg   # Fedora/RHEL",
            "# Reboot, then retry the multi-GPU workload.",
        ],
        needs_sudo=True,
        needs_reboot=True,
        verify="cat /proc/cmdline | grep -o 'iommu=\\w*'",
    ),
    "fix-12-installer": FixRecipe(
        fix_id="fix-12-installer",
        title="Reset amdgpu-install state and reinstall",
        rationale=(
            "amdgpu-install left a half-configured DKMS / repo state. Run "
            "the documented uninstall, clean up, and reinstall without the "
            "flag that broke things (commonly --accept-eula on newer installers)."
        ),
        auto_applicable=False,
        applies_on=LINUX_ONLY,
        commands=[
            "sudo amdgpu-install --uninstall",
            "sudo apt autoremove --purge -y",
            "sudo apt update",
            "sudo amdgpu-install --usecase=rocm,hip",
        ],
        needs_sudo=True,
        needs_reboot=True,
        verify="dpkg -l | grep -E 'rocm|amdgpu' | head -n 20 && rocminfo | head -n 5",
        notes=[
            "If `apt autoremove --purge` warns it will remove unrelated "
            "packages, stop and resolve those by hand before continuing.",
        ],
    ),
    "fix-13-hip-sdk-missing": FixRecipe(
        fix_id="fix-13-hip-sdk-missing",
        title="Install the AMD HIP SDK for Windows",
        rationale=(
            "Your framework links against HIP but the HIP SDK isn't installed "
            "on this host. The runtime DLLs (amdhip64_X.dll, hipblas.dll, "
            "hsa-runtime64.dll) and hipInfo.exe ship inside the SDK installer."
        ),
        auto_applicable=False,
        applies_on=WINDOWS_ONLY,
        commands=[
            "# Download and install the HIP SDK (matched to your framework's HIP major):",
            "#   https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html",
            "# After install, reopen the shell so HIP_PATH and PATH pick up the new install.",
        ],
        verify=(
            'powershell -NoProfile -Command '
            '"& \\"$env:HIP_PATH\\bin\\hipInfo.exe\\" | Select-Object -First 5"'
        ),
        notes=[
            "If you only need PyTorch on Windows AMD and don't need the C/C++ "
            "HIP toolchain, the TheRock wheels bundle their own HIP runtime "
            "and may not require a system HIP SDK install.",
        ],
    ),
    "fix-14-adrenalin-too-old": FixRecipe(
        fix_id="fix-14-adrenalin-too-old",
        title="Update the Adrenalin / kernel-mode driver",
        rationale=(
            "The HIP SDK is installed but the AMD kernel-mode driver "
            "(Adrenalin / Adrenalin Pro) is older than the SDK release notes "
            "call out. The user-space SDK and the driver have to match."
        ),
        auto_applicable=False,
        applies_on=WINDOWS_ONLY,
        commands=[
            "# Cross-check the HIP SDK release notes for the exact driver pairing:",
            "#   https://rocm.docs.amd.com/projects/install-on-windows/en/latest/install/install.html",
            "# Then download the matching driver from:",
            "#   https://www.amd.com/en/support",
            "# Reboot after the install for the kernel-mode driver to take effect.",
        ],
        needs_reboot=True,
        verify=(
            'powershell -NoProfile -Command '
            '"(Get-CimInstance Win32_VideoController | '
            "Where-Object { $_.Name -like '*AMD*' -or $_.Name -like '*Radeon*' } | "
            'Select-Object -First 1).DriverVersion"'
        ),
    ),
    "fix-15-msvc-redist": FixRecipe(
        fix_id="fix-15-msvc-redist",
        title="Install the MSVC 2015-2022 runtime redistributable",
        rationale=(
            "The HIP SDK's amdhip64_X.dll links against the MSVC 2015-2022 "
            "runtime. When vcruntime140.dll / vcruntime140_1.dll aren't on "
            "PATH, `import torch` fails with a missing-DLL error that points "
            "at vcruntime140_1.dll, not at the HIP runtime itself."
        ),
        auto_applicable=False,
        applies_on=WINDOWS_ONLY,
        commands=[
            "# Download and install (x64):",
            "#   https://aka.ms/vs/17/release/vc_redist.x64.exe",
            "# After the install, reopen the shell and re-run your import / hipInfo check.",
        ],
        verify="where vcruntime140.dll && where vcruntime140_1.dll",
        notes=[
            "If installing the redistributable still leaves a missing-DLL "
            "error, the failing DLL is probably amdhip64_X.dll itself; that "
            "points at fix-13-hip-sdk-missing rather than this fix.",
        ],
    ),
}


def _list_recipes() -> None:
    print("Available fix-ids (mirror diagnose.py):")
    for r in RECIPES.values():
        kind = "AUTO" if r.auto_applicable else "PRINT-ONLY"
        scope = "/".join(sorted(r.applies_on))
        print(f"  [{kind:>10s}] [{scope:>14s}] {r.fix_id}  -- {r.title}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix-id",
        help="Stable fix identifier from diagnose.py (e.g. fix-4-render-group).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List every fix-id and exit.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the plan without changing anything.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive confirmation. Use only when the user has "
             "already approved the plan in chat.",
    )
    parser.add_argument(
        "--device-index", type=int, default=None,
        help="For fix-9-igpu-dgpu: the rocminfo Agent index of the discrete GPU.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the recipe as JSON instead of running it.",
    )
    args = parser.parse_args(argv)

    if args.list:
        _list_recipes()
        return 0

    if not args.fix_id:
        parser.error("--fix-id or --list is required")

    recipe = RECIPES.get(args.fix_id)
    if recipe is None:
        print(f"Unknown fix-id: {args.fix_id}", file=sys.stderr)
        print("Run `python scripts/apply_fix.py --list` for the full list.", file=sys.stderr)
        return 2

    if args.json:
        # Strip the runner callable; it isn't JSON-serialisable. Convert
        # the frozenset for `applies_on` into a sorted list so the JSON
        # output is stable.
        d = {}
        for k, v in recipe.__dict__.items():
            if k == "runner":
                continue
            if k == "applies_on":
                d[k] = sorted(v)
            else:
                d[k] = v
        print(json.dumps(d, indent=2))
        return 0

    _print_recipe(recipe)
    print()

    sysname = platform.system().lower()
    if sysname not in recipe.applies_on:
        print(
            f"This fix only applies on: {', '.join(sorted(recipe.applies_on))}. "
            f"Running OS is: {sysname}."
        )
        return 3

    if not recipe.auto_applicable:
        print("This fix is print-only (manual change required).")
        print("Copy the commands above, run them yourself, then verify with:")
        if recipe.verify:
            print(f"  $ {recipe.verify}")
        return 0

    if recipe.runner is None:
        # Defensive: an auto_applicable recipe with no runner is a bug.
        print("Internal error: auto-applicable recipe has no runner.", file=sys.stderr)
        return 4
    return recipe.runner(args, recipe)  # type: ignore[misc]


if __name__ == "__main__":
    raise SystemExit(main())
