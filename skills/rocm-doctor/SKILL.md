---
name: rocm-doctor
description: >-
  Diagnoses why ROCm, the HIP SDK, PyTorch, or llama.cpp is broken on an AMD GPU
  on Linux, Windows, or WSL2, then applies a low-risk fix with consent or hands
  back the exact next step. Also routes Lemonade, LM Studio, and Ollama problems to the
  right upstream channel. Use when the user reports that ROCm or HIP "isn't
  working", torch.cuda.is_available() is False, rocminfo / hipInfo can't see the
  GPU, or hits hipErrorNoBinaryForGpu, HSA_STATUS_ERROR_INVALID_ISA, "invalid
  device function", "no kernel image is available", cannot open /dev/kfd,
  permission denied on /dev/kfd, "ROCk module is NOT loaded", a missing
  libamdhip64.so / amdhip64_6.dll / hipblas.dll / vcruntime140_1.dll, an
  HSA_OVERRIDE_GFX_VERSION page fault, an iGPU+dGPU crash, a container that can't
  see the GPU, or an amdgpu-install / DKMS failure. Backed by the `rocm` CLI
  (`rocm examine` / `rocm diagnose` / `rocm fix`); this skill is a thin driver
  over those commands, not a re-implementation.
---

# ROCm Doctor

Given a "ROCm / PyTorch / llama.cpp isn't working on my AMD GPU" complaint,
identify which **known misconfiguration** is the cause and either fix it (with
consent) or hand back the exact next step.

This skill does **not** probe or reason on its own. The `rocm` CLI owns the
probe, the closed failure-mode catalog, and the fixes; the skill just drives it
and relays the results. The catalog is a **closed list** — if the symptom
doesn't match a known mode, route the user upstream instead of guessing.

## Scope gate — check before anything else

Read the user's symptom and answer one question first: **is this an AMD GPU?**

If it is **not** — an **NVIDIA / Intel / Apple** GPU — then **stop and decline**:

- Say plainly that it is **out of scope** for this skill and why (not an AMD GPU).
- Give **no** troubleshooting for it: no commands to run, no driver or CUDA
  advice, no diagnostic checklist, no "try this first" — not even generic GPU
  suggestions. Point at the vendor's own docs and stop there.
- Do **not** run `rocm examine` / `rocm diagnose` / `rocm fix`.

Being helpful here means being honest about the boundary — confidently-wrong
advice for a stack this skill does not cover is worse than no advice. Only
continue past this gate when the GPU is AMD. See
[Out of scope](#out-of-scope).

**WSL2 is in scope** and does not stop this gate. It is a platform family of its
own, with its own catalog entries covering `/dev/dxg`, the DXCore handoff,
ROCDXG, the distro floor, the Windows host driver and WSL 1. Do not decline a
WSL2 user or route them away untouched — run the workflow as for any other
platform. The CLI scopes entries itself, so a bare-metal Linux fix is never
offered there.

## Prerequisites

- **The `rocm` CLI.** This skill is only a driver over it; Phase 0 below installs
  it with the user's consent if `rocm --version` fails. Nothing else here is
  assumed — the CLI does the probing.
- **Platform:** native Linux (in-tree `amdgpu` module + `/dev/kfd`), Windows
  (HIP SDK), or WSL2 (`/dev/dxg` + the Windows host driver). NVIDIA/Intel/Apple
  GPUs and clean-machine installs are out of scope (see
  [Out of scope](#out-of-scope)).
- **No fixed ROCm version, GPU arch (`gfx…`), or container image is assumed** —
  `rocm examine`/`diagnose` detect the installed ROCm, the GPU's `gfx` target, and
  container context, and match fixes to what they find. Never hand-set
  `HSA_OVERRIDE_GFX_VERSION` (or similar footgun env vars) yourself; let the CLI
  decide.

## Workflow

Only start here once the [Scope gate](#scope-gate--check-before-anything-else)
passes — the GPU is AMD. Linux, Windows and WSL2 all run the same workflow.

0. **Ensure the `rocm` CLI is present.** Everything below shells out to it, so
   check first and install it if missing:

   ```
   rocm --version
   ```

   If that succeeds, skip to step 1. If it's not found, install it **with the
   user's consent** (this fetches and runs an installer that drops the `rocm` and
   `rocmd` binaries into `~/.local/bin`). Both installers default to the
   `release` channel, so no channel argument is needed:

   - **Linux (x86_64 only):**
     ```
     curl -fsSL https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.sh | sh
     ```
   - **Windows (PowerShell):**
     ```
     irm https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.ps1 | iex
     ```

   There is no macOS build. `install.sh` refuses anything but Linux x86_64, so
   on any other host hand the user the install page and stop rather than
   suggesting the command anyway. To pin an unreleased build instead, pass
   `nightly` (`sh -s -- nightly`, or `$env:ROCM_CLI_CHANNEL = "nightly"`).

   After install, confirm `~/.local/bin` is on `PATH` and re-run `rocm --version`.
   If it still isn't available, hand the user the install page
   (https://github.com/ROCm/rocm-cli) and stop.

1. **Diagnose.** Pass the user's error text as the symptom:

   ```
   rocm diagnose --symptom "<paste the exact error>" --json
   ```

   Read the JSON:
   - `has_match` — **the gate.** True means a cause cleared the threshold and
     you may propose a fix; false means nothing was established, so route
     upstream. Do **not** substitute "is `matched` empty?": several checkers
     open with a nonzero base score for a merely *potentially* relevant
     situation (a container, an APU beside a discrete GPU), so a healthy host
     still returns a non-empty `matched` of sub-threshold entries. Gating on
     emptiness proposes a fix for a machine with nothing wrong.
   - `matched[]` — ranked causes, each with `id`, `title`, `score` (0–100),
     `evidence[]`, and a `fix` (with `fix_id`, `summary`, `commands`, `verify`,
     `notes`, and the `needs_sudo` / `needs_reboot` / `needs_relogin` /
     `auto_applicable` flags). `score >= 75` = high confidence; `50–74` = likely
     (confirm one more piece of evidence with the user first).
   - `out_of_scope` — set only when the host's platform family has no catalog
     entries at all. Linux, Windows and WSL2 are all covered, so this does
     **not** fire for WSL2. When it is set, nothing was checked — say so rather
     than implying the machine looks fine. First, if the user's symptom clearly
     names an app that ships its own runtime (Lemonade, Ollama, LM Studio),
     route them to that app's tracker (see
     [Framework routing](#framework-routing)) — those trackers apply regardless
     of platform. Otherwise relay the `out_of_scope` message and stop.
   - `route_when_no_match` — when `has_match` is false, hand the user this
     upstream tracker; **do not speculate**. Note the CLI picks this target from
     the *host-detected* framework, not from the symptom text — so for an app
     named only in the symptom, route it yourself per
     [Framework routing](#framework-routing).

2. **Propose the fix.** Show the top match's `title`, `evidence`, plan, and
   `verify` command. Only propose applying it when the user is on board.

3. **Apply with consent.** For an auto-applicable fix:

   ```
   rocm fix <fix-id>            # auto fixes: prompt before changing anything
   rocm fix <fix-id> --dry-run  # show the exact change, touch nothing
   rocm fix <fix-id> --yes      # required to apply in a non-interactive shell
   ```

   Only the four auto-applicable fixes are ones the CLI runs itself. The other 22
   are **print-only** (bootloader, kernel, reinstall, Windows driver, …): `rocm
   fix <id>` just prints the plan for the user to run themselves — no prompt, and
   the CLI never performs those.

   Of the four, two do not always mutate:

   - **`fix-2-unset-override` mutates on Windows only.** On Linux it reports
     where the override is set and which rc files carry it, then stops — it
     will not edit the user's dotfiles. So there is no prompt to answer and
     nothing for `--dry-run` to preview. Tell the Linux user what to edit; do
     not describe it as a change the CLI made or previewed.
   - **`fix-9-igpu-dgpu` mutates only when `--device-index N` is passed.**
     Without it — on either platform — the runner just prints the
     `rocminfo`/`hipInfo` query that identifies the discrete GPU's index and
     returns 0; there is no prompt, no `--dry-run` preview, and nothing is
     pinned. Once the user knows N, re-run with
     `rocm fix fix-9-igpu-dgpu --device-index N`.

4. **Verify.** Have the user run the `verify` command from the diagnosis.

Use `rocm examine` (or `rocm examine --json`) when you only need the host state
(GPU, driver, ROCm install, groups, framework) without a diagnosis.

## Framework routing

`rocm diagnose` covers frameworks that build against the **system** ROCm/HIP:

- **PyTorch**, **llama.cpp** — in scope; diagnose normally.

Apps that ship their **own** ROCm runtime aren't diagnosed here — route the user
to the right tracker. This routing is **yours, not the CLI's**: the host probe
reports only `pytorch`, `llama-cpp` or `unknown`, so `route_when_no_match` can
never name one of these apps. Use the list below.

- **Lemonade** → https://github.com/lemonade-sdk/lemonade/issues
- **Ollama** → https://github.com/ollama/ollama/issues
- **LM Studio** → in-app support (no public repo)
- Anything else with no catalog match → ROCm core:
  https://github.com/ROCm/ROCm/issues (this is what `route_when_no_match`
  returns by default).

## Out of scope

- **NVIDIA / Intel / Apple Silicon GPUs**, and **fresh installs on a clean
  machine** (a setup task, not a diagnosis). Exit cleanly and say so.

WSL2 is **not** in this list. It is a supported platform family with its own
catalog entries — see [reference.md](reference.md).

## Rules

- Never run the workflow — or offer *any* troubleshooting, generic GPU fixes
  included — for a non-AMD GPU. State it is out of scope and stop. This does
  not apply to WSL2, which is in scope and runs the normal workflow.
- Never invent a fix. If `rocm diagnose` returns no match, route upstream.
- Never run a mutating fix without the user's explicit OK; prefer `--dry-run`
  first. New failure modes are added to the CLI catalog, not improvised here.

See [reference.md](reference.md) for the full closed catalog and the CLI
command/exit-code reference.
