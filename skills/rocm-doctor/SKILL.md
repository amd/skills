---
name: rocm-doctor
description: >-
  Diagnoses why ROCm, the HIP SDK, PyTorch, or llama.cpp is broken on an AMD GPU
  on Linux or Windows, then applies a low-risk fix with consent or hands back the
  exact next step. Also routes Lemonade, LM Studio, and Ollama problems to the
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

## Workflow

0. **Ensure the `rocm` CLI is present.** Everything below shells out to it, so
   check first and install it if missing:

   ```
   rocm --version
   ```

   If that succeeds, skip to step 1. If it's not found, install it **with the
   user's consent** (this fetches and runs an installer that drops the `rocm` and
   `rocmd` binaries into `~/.local/bin`). Only nightly builds are published
   today, so install from the `nightly` channel:

   - **Linux / macOS:**
     ```
     curl -fsSL https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.sh | sh -s -- nightly
     ```
   - **Windows (PowerShell):**
     ```
     $env:ROCM_CLI_CHANNEL = "nightly"
     irm https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.ps1 | iex
     ```

   (Once rocm-cli cuts a stable release, drop the `nightly` channel — `sh` /
   `iex` alone will pull the latest stable build.)

   After install, confirm `~/.local/bin` is on `PATH` and re-run `rocm --version`.
   If it still isn't available, hand the user the install page
   (https://github.com/ROCm/rocm-cli) and stop.

1. **Diagnose.** Pass the user's error text as the symptom:

   ```
   rocm diagnose --symptom "<paste the exact error>" --json
   ```

   Read the JSON:
   - `matched[]` — ranked causes, each with `id`, `title`, `score` (0–100),
     `evidence[]`, and a `fix` (with `fix_id`, `summary`, `commands`, `verify`,
     `notes`, and the `needs_sudo` / `needs_reboot` / `needs_relogin` /
     `auto_applicable` flags). `score >= 75` = high confidence; `50–74` = likely
     (confirm one more piece of evidence with the user first).
   - `out_of_scope` — when set (e.g. WSL2), do **not** diagnose. First, if the
     user's symptom clearly names an app that ships its own runtime (Lemonade,
     Ollama, LM Studio), route them to that app's tracker (see
     [Framework routing](#framework-routing)) — those trackers apply regardless
     of platform. Otherwise relay the `out_of_scope` message and stop (see
     [Out of scope](#out-of-scope)).
   - `route_when_no_match` — when `matched` is empty, hand the user this
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

   Only the four auto-applicable fixes prompt and mutate. The other 11 are
   **print-only** (bootloader, kernel, reinstall, Windows driver, …): `rocm fix
   <id>` just prints the plan for the user to run themselves — no prompt, and the
   CLI never performs those.

4. **Verify.** Have the user run the `verify` command from the diagnosis.

Use `rocm examine` (or `rocm examine --json`) when you only need the host state
(GPU, driver, ROCm install, groups, framework) without a diagnosis.

## Framework routing

`rocm diagnose` covers frameworks that build against the **system** ROCm/HIP:

- **PyTorch**, **llama.cpp** — in scope; diagnose normally.

Apps that ship their **own** ROCm runtime aren't diagnosed here — route the user
to the right tracker. (The CLI's `route_when_no_match` also targets these, but
only when the host probe *detects* that app; when the app is named only in the
symptom, do the routing yourself using the list below.)

- **Lemonade** → https://github.com/lemonade-sdk/lemonade/issues
- **Ollama** → https://github.com/ollama/ollama/issues
- **LM Studio** → in-app support (no public repo)
- Anything else with no catalog match → ROCm core:
  https://github.com/ROCm/ROCm/issues (this is what `route_when_no_match`
  returns by default).

## Out of scope

- **WSL2** — a distinct platform (`/dev/dxg` + the Windows host driver, not the
  in-tree `amdgpu` module or `/dev/kfd`). `rocm examine`/`diagnose` detect it and
  route out; relay that guidance and point at AMD's ROCm-on-WSL guide.
- **NVIDIA / Intel / Apple Silicon GPUs**, and **fresh installs on a clean
  machine** (a setup task, not a diagnosis). Exit cleanly and say so.

## Rules

- Never invent a fix. If `rocm diagnose` returns no match, route upstream.
- Never run a mutating fix without the user's explicit OK; prefer `--dry-run`
  first. New failure modes are added to the CLI catalog, not improvised here.

See `reference.md` for the full closed catalog and the CLI command/exit-code
reference.
