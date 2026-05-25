---
name: rocm-doctor
description: >-
  Diagnoses why ROCm, the HIP SDK, PyTorch, or llama.cpp fails on an AMD
  GPU by matching symptoms against a closed catalog of known
  misconfigurations on Linux and Windows, then either applies a low-risk
  fix with consent or hands back the exact next step. Also routes
  Lemonade, LM Studio, and Ollama users to the right upstream channel.
  Use when the user says "ROCm/HIP isn't working", "torch.cuda.is_available()
  is False on Radeon/Ryzen AI", "rocminfo can't find my GPU",
  "hipInfo.exe can't see my Radeon", "amdhip64_6.dll could not be found",
  "vcruntime140_1.dll missing", "HIP SDK installer left things broken",
  "Adrenalin driver too old for the HIP SDK", "hipErrorNoBinaryForGpu",
  "HSA_STATUS_ERROR_INVALID_ISA", "invalid device function",
  "Unable to open /dev/kfd", "ROCk module is NOT loaded",
  "libamdhip64.so cannot open shared object file", "ROCm wheel doesn't
  see my gfx1151/gfx1150/gfx1103 (Strix Halo, Phoenix)", "iGPU/dGPU
  collision", "multi-GPU hang on AMD"; or mentions HSA_OVERRIDE_GFX_VERSION,
  HIP_VISIBLE_DEVICES, HIP_PATH, PYTORCH_ROCM_ARCH, render-group / /dev/kfd
  permissions, amdgpu blacklist, Secure Boot, or asks where to file a
  Lemonade / LM Studio / Ollama issue. Do NOT use for non-AMD GPUs,
  fresh installs, performance tuning, or ROCm-on-WSL2.
---

# ROCm Doctor

Given a "ROCm/PyTorch/llama.cpp isn't working on my AMD GPU" complaint,
identify which **known misconfiguration** is the cause and either fix it
or hand back the exact next step.

This is a diagnose-and-fix skill, not a setup or tuning skill. The
catalog of failure modes is a **closed list** that lives in
`reference.md` and `scripts/diagnose.py`: if the user's symptom doesn't
match one of them, the skill explicitly routes upstream rather than
guessing. New failure modes get added by editing the catalog, not by
the agent inventing them at runtime.

## When to use this skill

Use it when **any** of the following are true:

- The user has an **AMD** GPU and a functional error with **PyTorch**,
  **llama.cpp**, or anything else built directly against the system ROCm
  (`/opt/rocm` or a pip wheel that bundles HIP). The skill examines the
  host and diagnoses against the catalog.
- The user is on **Lemonade**, **LM Studio**, or **Ollama**. These apps
  ship their own ROCm and don't need a host-level examination, but the
  user often doesn't know *where* to report the problem -- the skill
  knows the right upstream channel for each (see
  [Framework routing](#framework-routing)) and hands it over.

Out of scope:

- NVIDIA / Intel / Apple Silicon GPUs. Exit cleanly and tell the user.
- Fresh installs on a clean machine. That's a setup task; point at
  [`amdgpu-install`](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-overview.html)
  (Linux) or the [HIP SDK installer](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)
  (Windows).
- Pure performance complaints. Those belong in `mi-tuner` /
  `omniperf-tune` / `apu-memory-tuner`.
- **WSL2** (running Linux on top of Windows). The ROCm-on-WSL flow needs
  Adrenalin Pro plus the WSL kernel update on the Windows host -- those
  failure modes are not in this catalog. `examine.py` detects WSL via
  `/proc/version` and exits 2 with a route-out message; if the user wants
  WSL specifically, point them at <https://rocm.docs.amd.com/projects/install-on-wsl/en/latest/>.

## Prerequisites

- **OS:** Linux **or** Windows (native). The catalog has 12 Linux entries
  (5 of which are also valid on Windows) and 3 Windows-only entries; the
  scripts pick the right subset for the host they run on.
- **Linux tools the agent will invoke as part of examination** (best-effort;
  the script degrades when one is missing):
  - `lspci` (always present on desktop distros)
  - `rocminfo` (when ROCm is installed)
  - `journalctl` or `dmesg` (for amdgpu kernel-ring evidence)
  - `python` / `python3` to introspect PyTorch
  - `llama-cli` / `llama-server` / `main` to introspect llama.cpp
- **Windows tools the agent will invoke as part of examination**:
  - `powershell` (always present on Windows 10+) for `Get-CimInstance
    Win32_VideoController` / `Win32_Processor` and the env-scope reads.
  - `hipInfo.exe` from `%HIP_PATH%\bin` -- the Windows analog of `rocminfo`.
    Absence is itself a signal (see `fix-13-hip-sdk-missing`).
  - `setx` for env-var persistence and User-PATH edits (analog of editing
    `~/.bashrc` on Linux).
  - `python` to introspect PyTorch.
- **Permissions:** examination is fully read-only and works as a regular
  user on both OSes. Linux fixes that need `sudo` are flagged in their
  recipe metadata; Windows fixes that touch the Machine env scope are
  flagged similarly and `apply_fix.py` does NOT self-elevate -- the user
  has to run an Administrator PowerShell when those are required.

Silent footguns to surface explicitly when relevant:

- `HSA_OVERRIDE_GFX_VERSION` -- forcing an unsupported gfx target works
  for `rocminfo`/`hipInfo` but causes page faults at runtime. Diagnosis
  `fix-2-unset-override` is the response when this is set on a GPU that
  already has a native wheel; on Windows it can be persisted in either
  the User or Machine env scope, so check both.
- `HIP_VISIBLE_DEVICES` -- on dual-GPU systems (APU + dGPU) the iGPU is
  often index 0 and destabilises HIP unless explicitly hidden.
- `HIP_PATH` (Windows) -- if the user has multiple HIP SDK versions
  installed under `C:\Program Files\AMD\ROCm\`, `HIP_PATH` decides which
  one PyTorch / hipInfo actually loads. Pointing it at the wrong major
  produces the same failure mode as `fix-8-wheel-rocm`.
- `PYTORCH_ROCM_ARCH` -- only honored during a *build* of PyTorch. Setting
  it at runtime does nothing for a prebuilt wheel.
- `LD_LIBRARY_PATH` (Linux) -- a wheel-bundled `libamdhip64.so` shadowed
  by a system one (or vice versa) gives confusing `cannot open shared
  object file` errors that look like fix-8 but are really a load-order
  bug. The Windows analog is `PATH` order: a stale HIP SDK bin directory
  earlier on PATH than the one matching `HIP_PATH`.

## The three-step flow

Run these in order. The first two are read-only. The third asks before
changing anything.

```
[ ] 1. Identify the framework, then examine (read-only).
[ ] 2. Diagnose: match examination + symptom against the catalog.
[ ] 3. Propose the fix; only apply with explicit consent; re-verify.
```

### Step 1: identify the framework and examine

If the user hasn't said, ask which framework they are running. Use the
`AskQuestion` tool with PyTorch / llama.cpp / Lemonade / LM Studio /
Ollama / other as the options. The routing in [Framework routing](#framework-routing)
keys off the answer.

If the framework is in the "skip examination" bucket, jump straight to
the upstream link and exit. Otherwise run:

```bash
python scripts/examine.py --framework pytorch --json > exam.json
```

Replace `pytorch` with `llama-cpp`, or pass `--framework auto` to let the
script pick. Exit codes:

| Exit | Meaning | Next action |
|---|---|---|
| 0 | Examined; AMD GPU present | Continue to Step 2. |
| 2 | Wrong platform (WSL, neither Linux nor Windows, no AMD GPU) | Stop. Route the user. |
| 3 | Probes partially failed | Continue but warn the user. |

For a quick read-only summary without piping JSON, drop `--json`:

```bash
python scripts/examine.py --framework pytorch
```

`examine.py` collects exactly the facts the diagnosis catalog needs.
On Linux: OS / kernel, AMD GPUs and gfx targets, `amdgpu` / `amdkfd`
status, `/dev/kfd` ownership and group, user's group membership, system
ROCm version and install method, framework version and arch list, the
silent-footgun env vars, container/IOMMU state, and recent `amdgpu`
kernel log lines. On Windows: AMD adapters and gfx targets via
`Win32_VideoController` + `hipInfo.exe`, the HIP SDK install path and
version, the Adrenalin / kernel-mode driver version, MSVC redistributable
presence, and the same env-var snapshot. It deliberately does NOT spawn
heavy probes (no kernel launches, no model downloads).

### Step 2: diagnose

Hand the JSON snapshot plus the user's error text to `diagnose.py`:

```bash
python scripts/diagnose.py --exam exam.json \
  --symptom "HIP error: invalid device function on gfx1151"
```

The script runs every checker in the catalog, scores each from 0..100,
and prints a ranked list. Each match has a stable `fix-N-...` id used by
`apply_fix.py`.

Score tiers:

- `>= 75` (`HIGH`) -- propose the fix and (if auto-applicable) ask for
  consent to apply it.
- `>= 50` (`LIKELY`) -- describe the match and ask the user to confirm one
  more piece of evidence before applying.
- Below `50` -- print but do **not** act. If nothing scores `>= 50`, the
  script exits 1 with a single-line route to the right upstream tracker.
  Do not speculate.

JSON output (`--json`) is the same data the agent should use programmatically:

```bash
python scripts/diagnose.py --exam exam.json --symptom "..." --json
```

### Step 3: apply the fix (with consent)

Show the user the proposed fix (it's already printed by `diagnose.py`).
If they consent, run:

```bash
python scripts/apply_fix.py --fix-id fix-4-render-group --dry-run
python scripts/apply_fix.py --fix-id fix-4-render-group --yes
```

`--dry-run` prints the exact commands without executing. `--yes` skips
the interactive `[y/N]` prompt (only pass this after the user has agreed
in chat).

A subset of fixes are auto-applicable; the rest are deliberately
print-only because the risk of a half-applied state is too high for an
agent to take. To see which is which without consulting `reference.md`:

```bash
python scripts/apply_fix.py --list
```

That prints every `fix-id` with an `AUTO` or `PRINT-ONLY` tag. Auto
fixes are bounded operations like unsetting an env var, adding the user
to a group, or appending a single line to a shell rc. Print-only fixes
involve reinstalling frameworks, editing GRUB, regenerating the
initramfs, or moving system repo files; those need a human at the
keyboard.

After every fix, re-run the `verify` command the recipe printed. Only
declare success when the user's *original* failing command now succeeds
(e.g. `torch.cuda.is_available()` returns `True`, `rocminfo` lists the
GPU, the llama.cpp build runs).

## Framework routing

The skill's first decision is which framework the user runs. Some
frameworks ship their own ROCm and bypass the system install; for those
the right answer is "you're in the wrong place, here's where to file
it", and the skill delivers that answer directly rather than running
useless probes against the host.

| Framework | Examine the host? | Action |
|---|---|---|
| PyTorch (Linux ROCm wheel) | Yes | `python scripts/examine.py --framework pytorch`, then `diagnose.py`. |
| PyTorch (Windows TheRock wheel) | Yes | Same scripts; on Windows the catalog filters to Linux+Windows + Windows-only entries. |
| llama.cpp (built against system ROCm/HIP SDK) | Yes | `python scripts/examine.py --framework llama-cpp`, then `diagnose.py`. |
| Lemonade | No -- ships its own ROCm | Route to <https://github.com/lemonade-sdk/lemonade/issues> and the Lemonade [Discord](https://discord.gg/5xXzkMu8Zk). |
| LM Studio | No -- ships its own runtime | Route to <https://lmstudio.ai/docs/app> (in-app support; no public repo). |
| Ollama | No -- ships its own runtime | Route to <https://github.com/ollama/ollama/issues> and the Ollama Discord. |
| vLLM / SGLang | Out of scope until phase 1+ | Route to the project's own issue tracker. |

If a Lemonade / LM Studio / Ollama user *does* have a host-level ROCm
problem (rare), it shows up when their app fails AND a standalone
`rocminfo` (Linux) / `hipInfo.exe` (Windows) also fails. Only then
escalate to the full examination.

## Safety rules

- Read-only by default. Examination and diagnosis never change state.
- Always print before applying. `apply_fix.py` shows every command before
  asking for consent, even with `--yes`.
- Never reboot, never touch BIOS, never flash firmware.
- Never reinstall system packages without an interactive prompt or `--yes`.
- Never set `HSA_OVERRIDE_GFX_VERSION` as the *first* fix when a native
  wheel exists. That is `fix-2-unset-override`'s entire reason for being.
- Never silently fall back to a different fix when the requested one
  isn't applicable. Exit 3 and tell the user why.
- When nothing in the catalog matches, **do not speculate**. Hand the
  user the upstream tracker URL from `diagnose.py --json`.

## Verification checklist

Mark this skill complete only when **all** are true:

- [ ] `python scripts/examine.py` exits 0 (or 3 with the user's explicit
      go-ahead to continue despite a partial probe).
- [ ] `python scripts/diagnose.py --exam exam.json --symptom "..."` exits 0
      and surfaced exactly one HIGH-confidence diagnosis, OR it exited 1
      and the user has been routed to the right upstream tracker.
- [ ] If a fix was applied: the recipe's `verify` command exits cleanly.
- [ ] The user's *original* failing command now succeeds end-to-end (run
      it again in their original shell).
- [ ] If any fix needed a re-login or reboot, the user has actually done
      it before declaring success.

If any box is unchecked, the failure isn't resolved -- say so out loud
rather than declaring victory.

## Reference

For the full catalog of known misconfigurations, every fix-id and its
verify command, the silent-footgun env-var reference, and the
upstream-routing table in machine-readable form, see
[reference.md](reference.md).
