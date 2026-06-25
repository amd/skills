#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Match an `examine.py` snapshot against the rocm-doctor failure-mode list.

This script is the opinionated decision tree the `rocm-doctor` skill is
built around. It takes:

  1. The JSON output of `examine.py` (machine state).
  2. Optionally the user's error text (symptom).

and returns a ranked list of matches against the catalog of known
misconfigurations in `reference.md`. Each match comes with:

  - id       : stable identifier reused by `apply_fix.py` (e.g. "fix-4-render-group").
  - title    : one-line description of the failure mode.
  - score    : 0..100 confidence the user is hitting this case.
  - evidence : the concrete facts the score is based on.
  - fix      : the next action and a `verify` command the agent can re-run.

Usage:
    python scripts/examine.py --json > exam.json
    python scripts/diagnose.py --exam exam.json
    python scripts/diagnose.py --exam exam.json --symptom "HIP error: invalid device function"
    python scripts/diagnose.py --exam exam.json --json
    python scripts/diagnose.py --exam exam.json --top 3

Exit codes:
  0 = at least one diagnosis matched (score >= MIN_SCORE_FOR_MATCH).
  1 = nothing matched; this is the explicit "I don't recognise this failure
      mode" path. The agent should NOT speculate; it should hand the user
      the upstream tracker URL printed by --json.
  2 = exam JSON is missing or malformed.

The closed list is deliberate. New failure modes go through a code change
here; they do not get invented by the agent at runtime.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

# A score above this threshold is treated as "we think this is it".
# Tuned so that a single direct symptom keyword match (worth ~40) plus a
# corroborating state signal (worth ~20+) is enough to surface a diagnosis.
MIN_SCORE_FOR_MATCH = 50

# Above this score we tell the agent to propose the fix immediately; below
# it (but above MIN_SCORE_FOR_MATCH) we surface as "likely" and ask the
# user to confirm one more piece of evidence first.
HIGH_CONFIDENCE = 75

# Upstream router used when nothing matches. Keeping the URL list short so
# the agent has exactly one place to send each kind of report.
UPSTREAM_TRACKERS = {
    "rocm-core":   "https://github.com/ROCm/ROCm/issues",
    "pytorch":     "https://github.com/pytorch/pytorch/issues  (tag with rocm label)",
    "llama-cpp":   "https://github.com/ggml-org/llama.cpp/issues",
    "lemonade":    "https://github.com/lemonade-sdk/lemonade/issues",
    "ollama":      "https://github.com/ollama/ollama/issues",
    "lm-studio":   "https://lmstudio.ai/docs/app  (use in-app support; no public repo)",
    "amdgpu-install": "https://repo.radeon.com  (raise via your AMD support contact)",
}


@dataclass
class Fix:
    summary: str                          # one-line plan
    commands: list[str] = field(default_factory=list)
    needs_sudo: bool = False
    needs_reboot: bool = False
    needs_relogin: bool = False
    fix_id: str = ""                      # passed to apply_fix.py --fix-id
    auto_applicable: bool = False         # True iff apply_fix.py can run it
    notes: list[str] = field(default_factory=list)
    verify: str = ""                      # command the agent should run after


@dataclass
class Diagnosis:
    id: str
    title: str
    score: int
    evidence: list[str] = field(default_factory=list)
    fix: Fix | None = None


# ---------------------------------------------------------------------------
# Symptom keyword tables. Each tuple is (regex, weight, label-for-evidence).
# Weights are tuned so that one specific error message (libamdhip64.so.X,
# HSA_STATUS_ERROR_INVALID_ISA) is enough to dominate the diagnosis on its
# own, while vague matches (the word "hang") only nudge the score.
# ---------------------------------------------------------------------------

KEYWORDS_INVALID_ISA = [
    (r"hiperrornobinaryforgpu", 45, "error mentions hipErrorNoBinaryForGpu"),
    (r"hsa_status_error_invalid_isa", 50, "error mentions HSA_STATUS_ERROR_INVALID_ISA"),
    (r"invalid device function", 40, "error mentions 'invalid device function'"),
    (r"no kernel image is available", 35, "error mentions 'no kernel image is available'"),
    (r"gfx\d{3,4}.* not (?:in|on) .*arch", 35, "error names a missing gfx in arch list"),
]

KEYWORDS_KFD_PERMISSION = [
    (r"unable to open /dev/kfd", 50, "error mentions /dev/kfd open failure"),
    (r"/dev/kfd.*permission denied", 45, "error mentions /dev/kfd permission denied"),
    (r"hsa_status_error_out_of_resources", 25, "HSA out-of-resources (often perms)"),
    (r"failed to open kfd", 35, "error mentions kfd open failure"),
]

KEYWORDS_MODULE_NOT_LOADED = [
    (r"rock module is not loaded", 50, "rocminfo says ROCk module is NOT loaded"),
    (r"no devices? found", 20, "vague 'no devices found'"),
    (r"hsa_status_error", 10, "HSA error (broad)"),
]

KEYWORDS_PATH_MISSING = [
    (r"rocminfo: command not found", 50, "rocminfo not on PATH"),
    (r"command not found.*hipcc", 40, "hipcc not on PATH"),
    (r"/opt/rocm/bin", 15, "user mentions /opt/rocm/bin"),
]

KEYWORDS_LIB_MISMATCH = [
    (r"libamdhip64\.so", 50, "error mentions libamdhip64.so"),
    (r"libhsa-runtime", 45, "error mentions libhsa-runtime"),
    (r"libhipblas", 40, "error mentions libhipblas"),
    (r"amdhip64_\d+\.dll", 50, "error mentions amdhip64_X.dll (Windows)"),
    (r"hipblas\.dll", 40, "error mentions hipblas.dll (Windows)"),
    (r"cannot open shared object file", 25, "ldopen failure"),
    (r"dll load failed", 25, "Windows DLL load failure"),
    (r"version `?glibc", 5, "tangential glibc version error"),
]

KEYWORDS_HIP_SDK_MISSING = [
    (r"amdhip64.*not found", 50, "error names amdhip64 missing"),
    (r"could not find hip", 40, "error mentions HIP not found"),
    (r"hip_path.*not set", 35, "user mentions HIP_PATH unset"),
    (r"hipinfo.*not recognized", 45, "Windows says hipInfo is not a command"),
]

KEYWORDS_MSVC_REDIST = [
    (r"vcruntime140(?:_1)?\.dll", 50, "error mentions vcruntime140 / vcruntime140_1"),
    (r"api-ms-win-crt-.*\.dll", 35, "error mentions api-ms-win-crt-* DLL"),
    (r"the (program|application) can't start because", 25, "Windows missing-DLL dialog text"),
    (r"msvcp140\.dll", 30, "error mentions msvcp140.dll"),
]

KEYWORDS_REPO_BROKEN = [
    (r"404.*repo\.radeon\.com", 50, "404 against repo.radeon.com"),
    (r"release file (is )?not (yet )?valid", 30, "apt 'release file not valid'"),
    (r"the following packages have unmet dependencies", 25, "apt unmet dependencies"),
    (r"unable to locate package rocm", 35, "apt cannot find ROCm package"),
]

KEYWORDS_CONTAINER = [
    (r"hsa_status_error.*permission", 20, "HSA permission error (often container)"),
    (r"/dev/dri.*permission", 30, "/dev/dri permission failure"),
    (r"failed to open device", 25, "device open failure"),
]

KEYWORDS_IOMMU_HANG = [
    (r"hang", 20, "user mentions 'hang'"),
    (r"deadlock", 20, "user mentions deadlock"),
    (r"timed out waiting", 25, "ring/queue timeout"),
    (r"iommu", 30, "user mentions iommu"),
]

KEYWORDS_DPKG_BROKEN = [
    (r"half[- ]configured", 50, "dpkg 'half-configured'"),
    (r"dkms .*failed", 45, "DKMS build failure"),
    (r"dpkg: error", 25, "generic dpkg error"),
    (r"sub-process /usr/bin/dpkg returned", 25, "apt mentions dpkg failure"),
    (r"--accept-eula", 40, "user mentions --accept-eula"),
]

KEYWORDS_PAGE_FAULT = [
    (r"page fault", 40, "user mentions page fault"),
    (r"vm_fault", 35, "kernel vm_fault"),
    (r"hw_fault", 30, "amdgpu HW fault"),
    (r"out_of_registers", 30, "compiler OUT_OF_REGISTERS"),
]


def _keyword_score(symptom: str, table: list[tuple[str, int, str]]) -> tuple[int, list[str]]:
    """Return (score, evidence_lines) for the strongest matches in `table`.

    We DO NOT sum every match: a long error string mentioning the same
    underlying problem in two ways shouldn't double-count. Instead we take
    the top two distinct hits and sum those. That keeps signal strong but
    bounded.
    """
    if not symptom:
        return 0, []
    sym = symptom.lower()
    hits: list[tuple[int, str]] = []
    for pattern, weight, label in table:
        if re.search(pattern, sym):
            hits.append((weight, label))
    if not hits:
        return 0, []
    hits.sort(reverse=True)
    top = hits[:2]
    return sum(h[0] for h in top), [h[1] for h in top]


# ---------------------------------------------------------------------------
# Examination accessors. The script accepts either the dict that
# `examine.py --json` emits OR a Python dict the agent has constructed by
# hand. We avoid pulling in the dataclass module here to keep diagnose.py
# usable standalone.
# ---------------------------------------------------------------------------

def _g(exam: dict, *path: str, default: Any = None) -> Any:
    """Safe nested-key getter."""
    cur: Any = exam
    for p in path:
        if not isinstance(cur, dict):
            return default
        if p not in cur:
            return default
        cur = cur[p]
    return cur if cur is not None else default


def _amd_gpus(exam: dict) -> list[dict]:
    return [g for g in _g(exam, "gpus", default=[]) if isinstance(g, dict) and g.get("is_amd")]


def _amd_gfx_targets(exam: dict) -> list[str]:
    return [g.get("gfx_target", "") for g in _amd_gpus(exam) if g.get("gfx_target")]


# ---------------------------------------------------------------------------
# Per-misconfiguration checkers
#
# Each `check_*` function returns a Diagnosis with score=0 to mean "not a
# match". `run_all_checks` filters those out. The MIN_SCORE_FOR_MATCH
# threshold then promotes the survivors to "we think this is it".
# ---------------------------------------------------------------------------

def check_1_arch_not_in_wheel(exam: dict, symptom: str) -> Diagnosis:
    """GPU gfx target not in the framework's build arch list."""
    score = 0
    evidence: list[str] = []

    kw_score, kw_ev = _keyword_score(symptom, KEYWORDS_INVALID_ISA)
    score += kw_score
    evidence += kw_ev

    framework_arch = _g(exam, "framework_arch_list", default=[]) or []
    gfx_targets = _amd_gfx_targets(exam)
    # Direct check: any AMD gfx target in the system that is NOT in the
    # framework's arch list. This is the strongest possible signal.
    missing = [t for t in gfx_targets if framework_arch and t not in framework_arch]
    if framework_arch and gfx_targets:
        if missing:
            score += 55
            evidence.append(
                f"GPU gfx target(s) {missing} not in framework arch list {framework_arch}"
            )
        else:
            # Strong negative: every GPU is covered. Push score down so a
            # weak symptom keyword alone doesn't surface this diagnosis.
            score -= 30
            evidence.append(
                f"framework arch list {framework_arch} already includes GPU target(s) {gfx_targets}"
            )

    framework = _g(exam, "framework", default="")
    if framework in ("pytorch", "llama-cpp") and not framework_arch and gfx_targets:
        # We at least know there is a framework and a GPU; can't confirm
        # without arch list, but the symptom keywords still apply.
        evidence.append(
            "Framework arch list unknown -- cannot confirm without "
            "`python -c 'import torch; print(torch.cuda.get_arch_list())'`."
        )

    if score <= 0:
        return Diagnosis(id="fix-1-arch", title="GPU gfx not in framework arch list", score=0)

    fix = Fix(
        summary=(
            "Reinstall the framework from a wheel index that includes this GPU's "
            "gfx target. Use HSA_OVERRIDE_GFX_VERSION ONLY as a temporary "
            "workaround when no native wheel exists."
        ),
        commands=[
            "# Recommended: PyTorch ROCm nightly that ships the gfx115x kernels.",
            "pip uninstall -y torch torchvision torchaudio",
            "pip install --pre torch torchvision torchaudio \\\n"
            "  --index-url https://download.pytorch.org/whl/nightly/rocm6.4",
            "# llama.cpp: rebuild with AMDGPU_TARGETS set to this GPU's gfx.",
            "# cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=<gfx_target>",
        ],
        fix_id="fix-1-arch",
        auto_applicable=False,
        verify=(
            "python -c \"import torch; print(torch.cuda.is_available(), "
            "torch.cuda.get_arch_list())\""
        ),
        notes=[
            "TheRock (rocm/TheRock) ships nightly per-gfx wheels and is the "
            "preferred fallback when the official pytorch wheel index does "
            "not yet cover your gfx target.",
        ],
    )
    return Diagnosis(
        id="fix-1-arch",
        title="GPU gfx target not in framework's build arch list",
        score=min(score, 100),
        evidence=evidence,
        fix=fix,
    )


def check_2_hsa_override_unneeded(exam: dict, symptom: str) -> Diagnosis:
    """HSA_OVERRIDE_GFX_VERSION set on a GPU that now has native support."""
    env = _g(exam, "env", default={}) or {}
    override = env.get("HSA_OVERRIDE_GFX_VERSION", "")
    if not override:
        return Diagnosis(id="fix-2-unset-override", title="HSA_OVERRIDE_GFX_VERSION set unnecessarily", score=0)

    score = 30
    evidence = [f"HSA_OVERRIDE_GFX_VERSION={override} is set in the current shell"]

    # Page faults are the classic late-binding symptom of an override that
    # masks the real gfx.
    pf_score, pf_ev = _keyword_score(symptom, KEYWORDS_PAGE_FAULT)
    score += pf_score
    evidence += pf_ev
    dmesg = _g(exam, "dmesg_amdgpu_tail", default=[]) or []
    if any("page fault" in line.lower() for line in dmesg):
        score += 20
        evidence.append("kernel ring shows amdgpu page faults")

    framework_arch = _g(exam, "framework_arch_list", default=[]) or []
    gfx_targets = _amd_gfx_targets(exam)
    if framework_arch and gfx_targets and all(t in framework_arch for t in gfx_targets):
        score += 25
        evidence.append(
            f"every detected GPU target ({gfx_targets}) is in the framework arch "
            f"list ({framework_arch}); the override is hiding the native gfx."
        )

    if _g(exam, "os_family", default="linux") == "windows":
        fix = Fix(
            summary="Clear HSA_OVERRIDE_GFX_VERSION (Windows) and use the native HIP SDK / wheel.",
            commands=[
                "# Inspect the User and Machine env scopes:",
                "[Environment]::GetEnvironmentVariable('HSA_OVERRIDE_GFX_VERSION','User')",
                "[Environment]::GetEnvironmentVariable('HSA_OVERRIDE_GFX_VERSION','Machine')",
                "# Clear from the User scope (does NOT affect already-open shells):",
                'setx HSA_OVERRIDE_GFX_VERSION ""',
                "# Or remove via System Properties -> Environment Variables.",
            ],
            fix_id="fix-2-unset-override",
            auto_applicable=True,
            verify=(
                "powershell -NoProfile -Command "
                "\"[Environment]::GetEnvironmentVariable('HSA_OVERRIDE_GFX_VERSION','User')\""
            ),
        )
    else:
        fix = Fix(
            summary="Unset HSA_OVERRIDE_GFX_VERSION and use the native wheel.",
            commands=[
                "unset HSA_OVERRIDE_GFX_VERSION",
                "# Also remove it from ~/.bashrc / ~/.zshrc / ~/.profile if persisted.",
            ],
            fix_id="fix-2-unset-override",
            auto_applicable=True,
            verify=(
                "env | grep HSA_OVERRIDE_GFX_VERSION || echo OK_UNSET; "
                "python -c \"import torch; print(torch.cuda.is_available())\""
            ),
        )
    return Diagnosis(
        id="fix-2-unset-override",
        title="HSA_OVERRIDE_GFX_VERSION set on a GPU that has a native wheel",
        score=min(score, 100),
        evidence=evidence,
        fix=fix,
    )


def check_3_rocm_kernel_unsupported(exam: dict, symptom: str) -> Diagnosis:
    """ROCm <-> distro/kernel unsupported triple."""
    score = 0
    evidence: list[str] = []

    kernel = _g(exam, "kernel_release", default="")
    distro = _g(exam, "distro_id", default="")
    distro_v = _g(exam, "distro_version", default="")
    rocm_version = _g(exam, "rocm_version", default="")
    amdgpu_loaded = _g(exam, "amdgpu_loaded", default=None)

    if rocm_version and amdgpu_loaded is False:
        score += 30
        evidence.append(
            f"ROCm {rocm_version} is installed but the amdgpu kernel module is not loaded; "
            "this is typical when DKMS failed against an unsupported kernel."
        )

    kw_score, kw_ev = _keyword_score(symptom, KEYWORDS_DPKG_BROKEN)
    if kw_ev and any("dkms" in e.lower() for e in kw_ev):
        score += 30
        evidence += kw_ev

    if kernel and rocm_version:
        # We do NOT hardcode a matrix here -- it's stale within months.
        # The check is purely "you have ROCm + amdgpu didn't load"; the
        # fix points the user at the live AMD matrix page.
        pass

    if score <= 0:
        return Diagnosis(id="fix-3-rocm-kernel", title="ROCm/distro/kernel triple unsupported", score=0)

    fix = Fix(
        summary=(
            "Cross-check your kernel/distro against the live AMD compatibility "
            "matrix before reinstalling."
        ),
        commands=[
            f"# Current: kernel={kernel} distro={distro} {distro_v} rocm={rocm_version}",
            "# Compare to the live AMD matrix:",
            "#   https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html",
            "# If your kernel is above the supported range, install the HWE",
            "# kernel that matches ROCm, or rerun amdgpu-install with --no-dkms.",
        ],
        fix_id="fix-3-rocm-kernel",
        auto_applicable=False,
        needs_reboot=True,
        verify="lsmod | grep amdgpu && rocminfo | head -n 20",
    )
    return Diagnosis(
        id="fix-3-rocm-kernel",
        title="ROCm version + distro/kernel form an unsupported triple",
        score=min(score, 100),
        evidence=evidence,
        fix=fix,
    )


def check_4_render_group(exam: dict, symptom: str) -> Diagnosis:
    """User not in render/video groups, or /dev/kfd group is wrong."""
    score = 0
    evidence: list[str] = []

    in_render = _g(exam, "in_render_group", default=None)
    in_video = _g(exam, "in_video_group", default=None)
    kfd = _g(exam, "kfd", default=None) or {}
    if in_render is False:
        score += 35
        evidence.append("user is NOT in the 'render' group")
    if in_video is False:
        score += 10
        evidence.append("user is NOT in the 'video' group")
    if kfd.get("exists") is True and kfd.get("user_can_write") is False:
        score += 25
        evidence.append(
            f"/dev/kfd exists (mode {kfd.get('mode')}, group {kfd.get('owner_group')}) "
            "but the current user can't write to it"
        )
    kw_score, kw_ev = _keyword_score(symptom, KEYWORDS_KFD_PERMISSION)
    score += kw_score
    evidence += kw_ev

    if score <= 0:
        return Diagnosis(id="fix-4-render-group", title="User missing render/video group", score=0)

    kfd_group = kfd.get("owner_group") or "render"
    fix = Fix(
        summary=f"Add the current user to '{kfd_group}' (and 'video' for safety) and log out/in.",
        commands=[
            f"sudo usermod -a -G {kfd_group},video \"$USER\"",
        ],
        needs_sudo=True,
        needs_relogin=True,
        fix_id="fix-4-render-group",
        auto_applicable=True,
        verify="groups | tr ' ' '\\n' | grep -E '^(render|video)$' && ls -l /dev/kfd && rocminfo | head -n 5",
        notes=[
            "Group membership only takes effect after a full re-login (or "
            "reboot). `newgrp render` will give the current shell access "
            "but not other terminals or services.",
        ],
    )
    return Diagnosis(
        id="fix-4-render-group",
        title="User not in render/video group (or /dev/kfd owned by the other group)",
        score=min(score, 100),
        evidence=evidence,
        fix=fix,
    )


def check_5_amdgpu_blacklisted(exam: dict, symptom: str) -> Diagnosis:
    """amdgpu module not loaded or actively blacklisted."""
    score = 0
    evidence: list[str] = []

    amdgpu_loaded = _g(exam, "amdgpu_loaded", default=None)
    blacklisted = _g(exam, "amdgpu_blacklisted_in", default=[]) or []
    rocm_status = _g(exam, "rocminfo_status", default="")
    secure_boot = _g(exam, "secure_boot", default="unknown")

    if blacklisted:
        score += 55
        evidence.append(f"amdgpu is blacklisted in: {blacklisted}")
    if amdgpu_loaded is False:
        score += 35
        evidence.append("amdgpu module is not loaded")
    if rocm_status == "not-loaded":
        score += 25
        evidence.append("rocminfo says 'ROCk module is NOT loaded'")
    if secure_boot == "enabled" and amdgpu_loaded is False:
        score += 10
        evidence.append(
            "Secure Boot is enabled and amdgpu didn't load -- DKMS modules "
            "are often blocked until you sign them or disable Secure Boot."
        )
    kw_score, kw_ev = _keyword_score(symptom, KEYWORDS_MODULE_NOT_LOADED)
    score += kw_score
    evidence += kw_ev

    if score <= 0:
        return Diagnosis(id="fix-5-amdgpu-load", title="amdgpu not loaded", score=0)

    commands: list[str] = []
    if blacklisted:
        for f in blacklisted:
            commands.append(f"# Inspect & remove the blacklist line: sudo $EDITOR {f}")
        commands.append("sudo update-initramfs -u   # Debian/Ubuntu")
        commands.append("sudo dracut -f             # Fedora/RHEL")
    commands.append("sudo modprobe amdgpu")
    if secure_boot == "enabled":
        commands.append(
            "# Secure Boot is on; if amdgpu still won't load, the DKMS "
            "module isn't signed. Sign it (mokutil) or disable Secure Boot."
        )

    fix = Fix(
        summary="Remove amdgpu from any modprobe blacklist and load it.",
        commands=commands,
        needs_sudo=True,
        needs_reboot=bool(blacklisted),
        fix_id="fix-5-amdgpu-load",
        auto_applicable=False,
        verify="lsmod | grep amdgpu && rocminfo | head -n 5",
    )
    return Diagnosis(
        id="fix-5-amdgpu-load",
        title="amdgpu kernel module not loaded (or blacklisted)",
        score=min(score, 100),
        evidence=evidence,
        fix=fix,
    )


def check_6_path_missing(exam: dict, symptom: str) -> Diagnosis:
    """ROCm/HIP binaries not on PATH after install."""
    score = 0
    evidence: list[str] = []

    os_family = _g(exam, "os_family", default="linux")
    env_path = _g(exam, "env", default={}).get("PATH", "")

    if os_family == "windows":
        sdk_path = _g(exam, "hip_sdk_path", default="")
        hipinfo_present = _g(exam, "hipinfo_present", default=None)
        bin_dir = f"{sdk_path}\\bin" if sdk_path else r"C:\Program Files\AMD\ROCm\<version>\bin"
        if sdk_path and hipinfo_present is False:
            score += 50
            evidence.append(f"{sdk_path} exists but hipInfo.exe wasn't found in its bin directory")
        if sdk_path and env_path and bin_dir.lower() not in env_path.lower():
            score += 20
            evidence.append(f"{bin_dir} is not in PATH")
    else:
        rocm_path = _g(exam, "rocm_path", default="")
        rocminfo_present = _g(exam, "rocminfo_present", default=None)
        bin_dir = f"{rocm_path}/bin" if rocm_path else "/opt/rocm/bin"
        if rocm_path and rocminfo_present is False:
            score += 50
            evidence.append(f"{rocm_path} exists but `rocminfo` is not on PATH")
        if rocm_path and env_path and bin_dir not in env_path:
            score += 20
            evidence.append(f"{bin_dir} is not in $PATH")

    kw_score, kw_ev = _keyword_score(symptom, KEYWORDS_PATH_MISSING)
    score += kw_score
    evidence += kw_ev

    if score <= 0:
        return Diagnosis(id="fix-6-path", title="ROCm not on PATH", score=0)

    if os_family == "windows":
        fix = Fix(
            summary=f"Add {bin_dir} to your User PATH and reopen the shell.",
            commands=[
                f'setx PATH "%PATH%;{bin_dir}"',
                "# Or: System Properties -> Environment Variables -> Path -> Edit -> New.",
                "# `setx` only affects NEW shells; close and reopen this terminal afterwards.",
            ],
            fix_id="fix-6-path",
            auto_applicable=True,
            verify=f'powershell -NoProfile -Command "& \\"{bin_dir}\\hipInfo.exe\\" | Select-Object -First 5"',
        )
    else:
        fix = Fix(
            summary=f"Add {bin_dir} to PATH for this shell and persist in your shell rc.",
            commands=[
                f"export PATH={bin_dir}:$PATH",
                f"echo 'export PATH={bin_dir}:$PATH' >> ~/.bashrc   # or ~/.zshrc",
            ],
            fix_id="fix-6-path",
            auto_applicable=True,
            verify="rocminfo | head -n 5 && hipcc --version",
        )
    return Diagnosis(
        id="fix-6-path",
        title="ROCm/HIP binaries not on PATH after install",
        score=min(score, 100),
        evidence=evidence,
        fix=fix,
    )


def check_7_stale_repos(exam: dict, symptom: str) -> Diagnosis:
    """Stale or conflicting APT/DNF repos from prior installer runs."""
    score = 0
    evidence: list[str] = []
    repos = _g(exam, "rocm_repos_seen", default=[]) or []
    # Two or more ROCm repo files is the usual smoking gun (often one from
    # the old amdgpu-install pin and one from a fresh radeon.com line).
    if len(repos) >= 2:
        score += 40
        evidence.append(
            f"{len(repos)} ROCm/AMDGPU repo files present: {repos}"
        )
    kw_score, kw_ev = _keyword_score(symptom, KEYWORDS_REPO_BROKEN)
    score += kw_score
    evidence += kw_ev

    if score <= 0:
        return Diagnosis(id="fix-7-stale-repos", title="Stale ROCm repos", score=0)

    commands = ["ls /etc/apt/sources.list.d/ | grep -iE 'rocm|amdgpu|radeon' || true"]
    for r in repos:
        commands.append(f"# sudo mv {r} {r}.bak     # quarantine, do not delete yet")
    commands.append("sudo apt update")
    commands.append("# If apt now resolves, reinstall via the correct method only:")
    commands.append("#   amdgpu-install --usecase=rocm,hip --no-dkms   # if you want amdgpu-install")
    commands.append("#   or use the distro packages exclusively")
    fix = Fix(
        summary=(
            "Quarantine duplicate ROCm/AMDGPU repo files and resolve apt before "
            "re-running any installer."
        ),
        commands=commands,
        needs_sudo=True,
        fix_id="fix-7-stale-repos",
        auto_applicable=False,
        verify="sudo apt update 2>&1 | tail -n 20",
    )
    return Diagnosis(
        id="fix-7-stale-repos",
        title="Stale or conflicting APT/DNF repos from prior installer runs",
        score=min(score, 100),
        evidence=evidence,
        fix=fix,
    )


def check_8_wheel_rocm_mismatch(exam: dict, symptom: str) -> Diagnosis:
    """Framework wheel built for a different ROCm major than the system."""
    score = 0
    evidence: list[str] = []
    os_family = _g(exam, "os_family", default="linux")
    fw_rocm = _g(exam, "framework_rocm_version", default="")
    if os_family == "windows":
        sys_rocm = _g(exam, "hip_sdk_version", default="")
    else:
        sys_rocm = _g(exam, "rocm_version", default="")

    def _major(s: str) -> str | None:
        m = re.search(r"(\d+)\.(\d+)", s)
        return f"{m.group(1)}.{m.group(2)}" if m else None

    fw_major = _major(fw_rocm)
    sys_major = _major(sys_rocm)
    if fw_major and sys_major and fw_major != sys_major:
        score += 50
        runtime = "HIP SDK" if os_family == "windows" else "ROCm"
        evidence.append(
            f"Framework links HIP {fw_major} but system {runtime} is {sys_major}"
        )

    kw_score, kw_ev = _keyword_score(symptom, KEYWORDS_LIB_MISMATCH)
    score += kw_score
    evidence += kw_ev

    if score <= 0:
        return Diagnosis(id="fix-8-wheel-rocm", title="Wheel/ROCm mismatch", score=0)

    if os_family == "windows":
        fix = Fix(
            summary=(
                "Reinstall the framework against the HIP SDK major you have "
                "installed (or install the HIP SDK major the wheel needs)."
            ),
            commands=[
                "pip uninstall -y torch torchvision torchaudio",
                "# TheRock publishes Windows ROCm wheels per HIP SDK release:",
                "#   https://github.com/ROCm/TheRock",
                "# Match the wheel index to the HIP SDK major you have on disk.",
                "python -c \"import torch; print(torch.__version__, torch.version.hip)\"",
            ],
            fix_id="fix-8-wheel-rocm",
            auto_applicable=False,
            verify="python -c \"import torch; print(torch.cuda.is_available(), torch.version.hip)\"",
        )
    else:
        fix = Fix(
            summary=(
                "Reinstall the framework from the wheel index that matches the "
                "system ROCm major (or upgrade the system ROCm to match the wheel)."
            ),
            commands=[
                "pip uninstall -y torch torchvision torchaudio",
                "# Pick the index that matches your system ROCm major. Examples:",
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4",
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3",
                "# Then re-check:",
                "python -c \"import torch; print(torch.__version__, torch.version.hip)\"",
            ],
            fix_id="fix-8-wheel-rocm",
            auto_applicable=False,
            verify="python -c \"import torch; print(torch.cuda.is_available(), torch.version.hip)\"",
        )
    return Diagnosis(
        id="fix-8-wheel-rocm",
        title="Framework wheel built for a different ROCm major than the system",
        score=min(score, 100),
        evidence=evidence,
        fix=fix,
    )


def check_9_igpu_dgpu_collision(exam: dict, symptom: str) -> Diagnosis:
    """iGPU enumerated alongside dGPU and crashing the runtime."""
    has_apu = _g(exam, "has_apu", default=False)
    has_discrete = _g(exam, "has_discrete_amd", default=False)
    if not (has_apu and has_discrete):
        return Diagnosis(id="fix-9-igpu-dgpu", title="iGPU+dGPU collision", score=0)

    env = _g(exam, "env", default={}) or {}
    visible = env.get("HIP_VISIBLE_DEVICES") or env.get("ROCR_VISIBLE_DEVICES")
    score = 40
    evidence = ["machine has both an AMD APU and an AMD discrete GPU"]
    if not visible:
        score += 25
        evidence.append("HIP_VISIBLE_DEVICES is unset; runtime sees BOTH GPUs")
    # Crashes are vague but a crash on a dual-GPU box is the classic signal.
    if symptom and re.search(r"(crash|segfault|signal 11)", symptom, re.IGNORECASE):
        score += 15
        evidence.append("user mentions a crash / segfault")

    gfx_targets = _amd_gfx_targets(exam)
    if _g(exam, "os_family", default="linux") == "windows":
        fix = Fix(
            summary=(
                "Pin the HIP runtime to the discrete GPU with HIP_VISIBLE_DEVICES "
                "so the iGPU is hidden."
            ),
            commands=[
                "# Confirm which index is the dGPU (hipInfo.exe output order):",
                '& "$env:HIP_PATH\\bin\\hipInfo.exe" | Select-String "device#|Name|gcnArchName"',
                "# Then persist HIP_VISIBLE_DEVICES in the User environment:",
                "setx HIP_VISIBLE_DEVICES 1",
                "# `setx` only takes effect in NEW shells; reopen the terminal.",
            ],
            fix_id="fix-9-igpu-dgpu",
            auto_applicable=True,
            verify=(
                'powershell -NoProfile -Command "$env:HIP_VISIBLE_DEVICES=1; '
                'python -c \\"import torch; print(torch.cuda.device_count())\\""'
            ),
            notes=[
                f"Detected gfx targets: {gfx_targets}. The dGPU is usually the higher-numbered family (gfx11xx).",
            ],
        )
    else:
        fix = Fix(
            summary=(
                "Pin the runtime to the discrete GPU with HIP_VISIBLE_DEVICES "
                "so the iGPU is hidden."
            ),
            commands=[
                "# Confirm which index is the dGPU (`rocminfo` output order):",
                "rocminfo | grep -E 'Agent |gfx|Marketing'",
                "# Then pin HIP to the dGPU (typically index 1 when an APU is index 0):",
                "export HIP_VISIBLE_DEVICES=1",
                "# Persist in your shell rc or your launch script.",
            ],
            fix_id="fix-9-igpu-dgpu",
            auto_applicable=False,
            verify="HIP_VISIBLE_DEVICES=1 python -c \"import torch; print(torch.cuda.device_count())\"",
            notes=[
                f"Detected gfx targets: {gfx_targets}. The dGPU is usually the higher-numbered family (gfx11xx).",
            ],
        )
    return Diagnosis(
        id="fix-9-igpu-dgpu",
        title="iGPU enumerated alongside dGPU and destabilising the runtime",
        score=min(score, 100),
        evidence=evidence,
        fix=fix,
    )


def check_10_container_devices(exam: dict, symptom: str) -> Diagnosis:
    """Container can't see /dev/kfd or /dev/dri/renderD*."""
    in_container = _g(exam, "in_container", default=False)
    if not in_container:
        return Diagnosis(id="fix-10-container", title="Container missing devices", score=0)

    score = 25
    evidence = [f"running inside a {_g(exam, 'container_kind', default='container')}"]
    kfd = _g(exam, "kfd", default=None) or {}
    if kfd.get("exists") is False:
        score += 40
        evidence.append("/dev/kfd is not present in the container")
    elif kfd.get("user_can_write") is False:
        score += 30
        evidence.append("/dev/kfd is present but not writable by the container user")
    if not _g(exam, "render_devices", default=[]):
        score += 20
        evidence.append("no /dev/dri/renderD* visible in the container")
    kw_score, kw_ev = _keyword_score(symptom, KEYWORDS_CONTAINER)
    score += kw_score
    evidence += kw_ev

    fix = Fix(
        summary=(
            "Re-launch the container with the AMD devices and the render group "
            "passed through."
        ),
        commands=[
            "# Docker / Podman flags AMD-recommends:",
            "docker run --rm -it \\",
            "  --device=/dev/kfd \\",
            "  --device=/dev/dri \\",
            "  --group-add render \\",
            "  --security-opt seccomp=unconfined \\",
            "  --shm-size=8g \\",
            "  rocm/pytorch:latest",
            "# Rootless podman: also pass `--userns=keep-id` and ensure the",
            "# host user is in the render group; podman maps it through.",
        ],
        fix_id="fix-10-container",
        auto_applicable=False,
        verify="rocminfo | head -n 5",
        notes=[
            "Use rocm/pytorch or rocm/dev-ubuntu-22.04 as a known-good image. "
            "Mixing host ROCm + container ROCm versions is a separate footgun.",
        ],
    )
    return Diagnosis(
        id="fix-10-container",
        title="Container can't see /dev/kfd or /dev/dri/renderD*",
        score=min(score, 100),
        evidence=evidence,
        fix=fix,
    )


def check_11_iommu_hang(exam: dict, symptom: str) -> Diagnosis:
    """Multi-GPU hang on systems with IOMMU enabled."""
    amd_count = len(_amd_gpus(exam))
    if amd_count < 2:
        return Diagnosis(id="fix-11-iommu", title="Multi-GPU IOMMU hang", score=0)

    score = 0
    evidence = [f"{amd_count} AMD GPUs detected"]
    iommu = _g(exam, "iommu_kernel_param", default="")
    if iommu and iommu != "pt":
        score += 25
        evidence.append(f"kernel cmdline has iommu={iommu} (not 'pt')")
    if not iommu:
        # IOMMU is on by default on most modern BIOSes even without the
        # kernel cmdline flag. A multi-GPU hang is still the classic signal.
        score += 10
        evidence.append("no iommu= flag on kernel cmdline (default may be 'on')")
    kw_score, kw_ev = _keyword_score(symptom, KEYWORDS_IOMMU_HANG)
    score += kw_score
    evidence += kw_ev

    if score < 25:
        return Diagnosis(id="fix-11-iommu", title="Multi-GPU IOMMU hang", score=0)

    fix = Fix(
        summary=(
            "Add `iommu=pt` to the kernel command line so DMA goes through "
            "pass-through mode. This requires editing GRUB and rebooting."
        ),
        commands=[
            "# Inspect the current cmdline:",
            "cat /proc/cmdline",
            "# Edit /etc/default/grub and add iommu=pt to GRUB_CMDLINE_LINUX_DEFAULT:",
            "sudo $EDITOR /etc/default/grub",
            "sudo update-grub                # Debian/Ubuntu",
            "sudo grub2-mkconfig -o /boot/grub2/grub.cfg   # Fedora/RHEL",
            "# Reboot for the change to take effect, then retry the multi-GPU job.",
        ],
        needs_sudo=True,
        needs_reboot=True,
        fix_id="fix-11-iommu",
        auto_applicable=False,
        verify="cat /proc/cmdline | grep -o 'iommu=\\w*'",
    )
    return Diagnosis(
        id="fix-11-iommu",
        title="Multi-GPU hang on systems with IOMMU enabled",
        score=min(score, 100),
        evidence=evidence,
        fix=fix,
    )


def check_12_amdgpu_install_broken(exam: dict, symptom: str) -> Diagnosis:
    """amdgpu-install left a broken DKMS / repo state."""
    score = 0
    evidence: list[str] = []
    method = _g(exam, "rocm_install_method", default="")
    if method == "amdgpu-install":
        evidence.append("ROCm was installed via amdgpu-install")
    else:
        # Not a hard requirement; users sometimes hit this after the
        # installer fails and they don't realize they did one. Don't add
        # base score, but allow keyword evidence to count.
        pass
    kw_score, kw_ev = _keyword_score(symptom, KEYWORDS_DPKG_BROKEN)
    score += kw_score
    evidence += kw_ev
    if method == "amdgpu-install" and kw_score > 0:
        score += 20

    if score <= 0:
        return Diagnosis(id="fix-12-installer", title="amdgpu-install broken state", score=0)

    fix = Fix(
        summary=(
            "Run amdgpu-install's documented uninstall sequence to clear the "
            "half-configured state, THEN reinstall without the flag that broke it."
        ),
        commands=[
            "sudo amdgpu-install --uninstall",
            "sudo apt autoremove --purge -y",
            "sudo apt update",
            "# Reinstall. Drop --accept-eula if you used it previously; the",
            "# newer installer rejects it and leaves a half-configured repo.",
            "sudo amdgpu-install --usecase=rocm,hip",
        ],
        needs_sudo=True,
        needs_reboot=True,
        fix_id="fix-12-installer",
        auto_applicable=False,
        verify="dpkg -l | grep -E 'rocm|amdgpu' | head -n 20 && rocminfo | head -n 5",
        notes=[
            "If `apt autoremove` warns it will remove unrelated packages, stop "
            "and resolve those by hand before continuing.",
        ],
    )
    return Diagnosis(
        id="fix-12-installer",
        title="amdgpu-install left a broken state (repo regression / partial DKMS)",
        score=min(score, 100),
        evidence=evidence,
        fix=fix,
    )


def check_13_hip_sdk_missing(exam: dict, symptom: str) -> Diagnosis:
    """Windows: framework imports HIP but the HIP SDK isn't installed."""
    if _g(exam, "os_family", default="") != "windows":
        return Diagnosis(id="fix-13-hip-sdk-missing", title="HIP SDK not installed", score=0)

    score = 0
    evidence: list[str] = []
    sdk_path = _g(exam, "hip_sdk_path", default="")
    hipinfo_present = _g(exam, "hipinfo_present", default=None)
    framework = _g(exam, "framework", default="")
    fw_rocm = _g(exam, "framework_rocm_version", default="")
    has_amd = _g(exam, "has_amd_gpu", default=False)

    if not sdk_path:
        score += 35
        evidence.append("No HIP SDK install found under C:\\Program Files\\AMD\\ROCm")
    elif hipinfo_present is False:
        score += 30
        evidence.append(f"HIP SDK at {sdk_path} but hipInfo.exe is missing from its bin directory")

    if has_amd and framework == "pytorch" and fw_rocm.startswith("hip="):
        score += 25
        evidence.append("PyTorch is a HIP build but the HIP SDK is not present on this host")

    kw_score, kw_ev = _keyword_score(symptom, KEYWORDS_HIP_SDK_MISSING)
    score += kw_score
    evidence += kw_ev

    if score <= 0:
        return Diagnosis(id="fix-13-hip-sdk-missing", title="HIP SDK not installed", score=0)

    fix = Fix(
        summary=(
            "Install the AMD HIP SDK for Windows; the HIP runtime DLLs and "
            "hipInfo.exe come from there."
        ),
        commands=[
            "# Download and install the HIP SDK (matched to your framework's HIP major):",
            "#   https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html",
            "# After install, reopen the shell so HIP_PATH and PATH pick up the new install.",
        ],
        fix_id="fix-13-hip-sdk-missing",
        auto_applicable=False,
        verify=(
            'powershell -NoProfile -Command '
            '"& \\"$env:HIP_PATH\\bin\\hipInfo.exe\\" | Select-Object -First 5"'
        ),
        notes=[
            "If you only need PyTorch on Windows AMD and don't need the C/C++ "
            "HIP toolchain, the TheRock wheels bundle their own HIP runtime "
            "and may not require a system HIP SDK install.",
        ],
    )
    return Diagnosis(
        id="fix-13-hip-sdk-missing",
        title="HIP SDK not installed (Windows)",
        score=min(score, 100),
        evidence=evidence,
        fix=fix,
    )


def check_14_adrenalin_too_old(exam: dict, symptom: str) -> Diagnosis:
    """Windows: HIP SDK present but Adrenalin (kernel-mode driver) is too old.

    We deliberately do NOT hardcode a minimum Adrenalin version here: AMD
    bumps the HIP SDK <-> Adrenalin pairing every release, and the live
    table goes stale within months. Instead we trigger on observable
    failure patterns (HIP SDK present + hipInfo unable to enumerate, or
    user pasted a 'Driver version too old' style symptom) and route the
    user to the live release notes.
    """
    if _g(exam, "os_family", default="") != "windows":
        return Diagnosis(id="fix-14-adrenalin-too-old", title="Adrenalin driver too old", score=0)

    score = 0
    evidence: list[str] = []
    sdk_path = _g(exam, "hip_sdk_path", default="")
    hipinfo_present = _g(exam, "hipinfo_present", default=False)
    hipinfo_status = _g(exam, "hipinfo_status", default="")
    adrenalin = _g(exam, "adrenalin_version", default="")

    if sdk_path and hipinfo_present and hipinfo_status not in ("ok", ""):
        score += 35
        evidence.append(
            f"HIP SDK at {sdk_path} is installed but hipInfo.exe reports {hipinfo_status!r}; "
            "this typically means the kernel-mode driver doesn't match the SDK."
        )
    if adrenalin:
        evidence.append(f"Adrenalin / kernel-mode driver version: {adrenalin}")

    if symptom and re.search(r"driver.*(too old|out of date|unsupported)", symptom, re.IGNORECASE):
        score += 35
        evidence.append("error mentions 'driver too old / out of date / unsupported'")
    if symptom and re.search(r"hsa.*invalid agent|no agents (were )?found", symptom, re.IGNORECASE):
        score += 25
        evidence.append("HSA error suggests driver/runtime can't enumerate the GPU")

    if score <= 0:
        return Diagnosis(id="fix-14-adrenalin-too-old", title="Adrenalin driver too old", score=0)

    fix = Fix(
        summary=(
            "Update the AMD Adrenalin (or PRO) graphics driver to the version "
            "the HIP SDK release notes call out as the supported pairing."
        ),
        commands=[
            "# Cross-check the HIP SDK release notes for the exact driver pairing:",
            "#   https://rocm.docs.amd.com/projects/install-on-windows/en/latest/install/install.html",
            "# Then download the matching driver from:",
            "#   https://www.amd.com/en/support",
            "# Reboot after the install for the kernel-mode driver to take effect.",
        ],
        needs_reboot=True,
        fix_id="fix-14-adrenalin-too-old",
        auto_applicable=False,
        verify=(
            'powershell -NoProfile -Command '
            '"(Get-CimInstance Win32_VideoController | '
            "Where-Object { $_.Name -like '*AMD*' -or $_.Name -like '*Radeon*' } | "
            'Select-Object -First 1).DriverVersion"'
        ),
    )
    return Diagnosis(
        id="fix-14-adrenalin-too-old",
        title="Adrenalin / kernel-mode driver too old for the installed HIP SDK",
        score=min(score, 100),
        evidence=evidence,
        fix=fix,
    )


def check_15_msvc_redist(exam: dict, symptom: str) -> Diagnosis:
    """Windows: MSVC runtime DLL missing -- HIP DLLs fail to load."""
    if _g(exam, "os_family", default="") != "windows":
        return Diagnosis(id="fix-15-msvc-redist", title="MSVC runtime missing", score=0)

    score = 0
    evidence: list[str] = []
    redist = _g(exam, "msvc_redist_present", default=None)
    if redist is False:
        score += 45
        evidence.append("vcruntime140.dll / vcruntime140_1.dll not resolvable on PATH")

    kw_score, kw_ev = _keyword_score(symptom, KEYWORDS_MSVC_REDIST)
    score += kw_score
    evidence += kw_ev

    if score <= 0:
        return Diagnosis(id="fix-15-msvc-redist", title="MSVC runtime missing", score=0)

    fix = Fix(
        summary=(
            "Install the Microsoft Visual C++ 2015-2022 redistributable so the "
            "HIP SDK's amdhip64_*.dll can load."
        ),
        commands=[
            "# Download & install (x64):",
            "#   https://aka.ms/vs/17/release/vc_redist.x64.exe",
            "# After the install, reopen the shell and re-run your import / hipInfo check.",
        ],
        fix_id="fix-15-msvc-redist",
        auto_applicable=False,
        verify=(
            "where vcruntime140.dll && where vcruntime140_1.dll"
        ),
        notes=[
            "If installing the redistributable still leaves a missing-DLL error, "
            "the failing DLL is probably amdhip64_X.dll itself; that points at "
            "fix-13-hip-sdk-missing (the HIP SDK install) rather than this fix.",
        ],
    )
    return Diagnosis(
        id="fix-15-msvc-redist",
        title="MSVC runtime missing (HIP DLLs cannot load)",
        score=min(score, 100),
        evidence=evidence,
        fix=fix,
    )


# Each entry: (checker, frozenset of os_family values it applies to).
# `run_all_checks` reads the running OS from the exam JSON and skips
# checkers whose `applicable_on` doesn't include it. This keeps the OS
# branching in one place rather than scattered through the checkers.
CHECKERS: list[tuple[Callable[[dict, str], Diagnosis], frozenset[str]]] = [
    (check_1_arch_not_in_wheel,       frozenset({"linux", "windows"})),
    (check_2_hsa_override_unneeded,   frozenset({"linux", "windows"})),
    (check_3_rocm_kernel_unsupported, frozenset({"linux"})),
    (check_4_render_group,            frozenset({"linux"})),
    (check_5_amdgpu_blacklisted,      frozenset({"linux"})),
    (check_6_path_missing,            frozenset({"linux", "windows"})),
    (check_7_stale_repos,             frozenset({"linux"})),
    (check_8_wheel_rocm_mismatch,     frozenset({"linux", "windows"})),
    (check_9_igpu_dgpu_collision,     frozenset({"linux", "windows"})),
    (check_10_container_devices,      frozenset({"linux"})),
    (check_11_iommu_hang,             frozenset({"linux"})),
    (check_12_amdgpu_install_broken,  frozenset({"linux"})),
    (check_13_hip_sdk_missing,        frozenset({"windows"})),
    (check_14_adrenalin_too_old,      frozenset({"windows"})),
    (check_15_msvc_redist,            frozenset({"windows"})),
]


def run_all_checks(exam: dict, symptom: str) -> list[Diagnosis]:
    """Run every applicable checker, drop zero-score results, sort by score desc.

    Checkers whose `applicable_on` set doesn't include the running OS are
    skipped silently (they were never going to score against this exam).
    """
    os_family = _g(exam, "os_family", default="linux")
    results: list[Diagnosis] = []
    for fn, applicable_on in CHECKERS:
        if os_family not in applicable_on:
            continue
        try:
            d = fn(exam, symptom or "")
        except Exception as exc:  # checker bug should not kill diagnose
            results.append(Diagnosis(
                id=f"checker-error-{fn.__name__}",
                title=f"Internal checker error in {fn.__name__}",
                score=0,
                evidence=[f"{type(exc).__name__}: {exc}"],
            ))
            continue
        if d.score > 0:
            results.append(d)
    results.sort(key=lambda d: d.score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _route_when_no_match(exam: dict) -> dict:
    """Pick the right upstream tracker for the user's framework."""
    fw = _g(exam, "framework", default="unknown")
    target = {
        "pytorch": "pytorch",
        "llama-cpp": "llama-cpp",
        "lemonade": "lemonade",
        "ollama": "ollama",
        "lm-studio": "lm-studio",
    }.get(fw, "rocm-core")
    return {"target": target, "url": UPSTREAM_TRACKERS[target]}


def _print_human(diagnoses: list[Diagnosis], exam: dict, top: int) -> None:
    if not diagnoses:
        route = _route_when_no_match(exam)
        print("rocm-doctor: no known misconfiguration matched.")
        print()
        print(
            "This is the explicit 'I don't recognise this failure mode' case. "
            "Do not speculate; file the symptom + this examination output upstream:"
        )
        print(f"  {route['target']:>12s}: {route['url']}")
        print()
        print("Include the JSON from `python scripts/examine.py --json` in your report.")
        return

    for i, d in enumerate(diagnoses[:top], 1):
        tier = "HIGH" if d.score >= HIGH_CONFIDENCE else (
            "LIKELY" if d.score >= MIN_SCORE_FOR_MATCH else "WEAK"
        )
        print(f"#{i} [{tier} score={d.score}/100] {d.title}")
        print(f"   id: {d.id}")
        for e in d.evidence:
            print(f"   - {e}")
        if d.fix:
            print(f"   plan: {d.fix.summary}")
            for c in d.fix.commands:
                print(f"     $ {c}")
            flags = []
            if d.fix.needs_sudo: flags.append("sudo")
            if d.fix.needs_reboot: flags.append("reboot required")
            if d.fix.needs_relogin: flags.append("re-login required")
            if d.fix.auto_applicable: flags.append("apply_fix.py can run it")
            if flags:
                print(f"   flags: {', '.join(flags)}")
            for n in d.fix.notes:
                print(f"   note: {n}")
            if d.fix.verify:
                print(f"   verify after fix: {d.fix.verify}")
        print()

    high = [d for d in diagnoses if d.score >= HIGH_CONFIDENCE]
    if high:
        print(f"Next step: propose `apply_fix.py --fix-id {high[0].id}` to the user.")
    else:
        print(
            "Highest-scoring match is below the HIGH_CONFIDENCE threshold. "
            "Confirm one more piece of evidence with the user before applying."
        )


def _to_jsonable(diagnoses: list[Diagnosis], exam: dict) -> dict:
    return {
        "matched": [asdict(d) for d in diagnoses],
        "min_score_for_match": MIN_SCORE_FOR_MATCH,
        "high_confidence_threshold": HIGH_CONFIDENCE,
        "route_when_no_match": _route_when_no_match(exam),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exam", type=Path, required=True,
        help="Path to the JSON produced by `examine.py --json`.",
    )
    parser.add_argument(
        "--symptom", default="",
        help="Raw error text from the user; symptom-keyword scoring uses it.",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Show at most this many matching diagnoses (default 5).",
    )
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON instead of the human view.")
    args = parser.parse_args(argv)

    if not args.exam.exists():
        print(f"exam file not found: {args.exam}", file=sys.stderr)
        return 2
    try:
        exam = json.loads(args.exam.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"exam file is not valid JSON: {exc}", file=sys.stderr)
        return 2

    diagnoses = run_all_checks(exam, args.symptom)
    matched = [d for d in diagnoses if d.score >= MIN_SCORE_FOR_MATCH]
    if args.json:
        print(json.dumps(_to_jsonable(diagnoses, exam), indent=2))
    else:
        _print_human(diagnoses, exam, args.top)
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
