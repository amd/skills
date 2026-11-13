---
name: rocm-doctor
description: >-
  Diagnoses why ROCm, the HIP SDK, PyTorch, or llama.cpp is broken on an AMD GPU
  on Linux, Windows, or WSL2, then applies a low-risk fix with consent or hands
  back the exact next step. Also routes Lemonade, LM Studio, and Ollama problems to the
  right upstream channel. Use when the user reports that ROCm or HIP "isn't
  working", torch.cuda.is_available() is False, rocminfo / hipInfo can't see the
  GPU, a vLLM startup out-of-memory error, a libtorch_cuda.so OSError/ImportError
  at vLLM start, or hits hipErrorNoBinaryForGpu, HSA_STATUS_ERROR_INVALID_ISA, "invalid
  device function", "no kernel image is available", cannot open /dev/kfd,
  permission denied on /dev/kfd, "ROCk module is NOT loaded", a missing
  libamdhip64.so / amdhip64_6.dll / hipblas.dll / vcruntime140_1.dll, an
  HSA_OVERRIDE_GFX_VERSION page fault, an iGPU+dGPU crash, a container that can't
  see the GPU, or a repo-native package-manager install (formerly `amdgpu-install`) / DKMS
  failure. Backed by the `rocm` CLI (`rocm examine` / `diagnose` / `fix`).
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
advice for a stack this skill does not cover is worse than no advice.

A fresh install on a clean machine is out of scope too: that is a setup task,
not a diagnosis. Exit cleanly and say so.

If the user names no GPU vendor, treat ROCm / HIP wording as AMD and continue to
step 1, which checks what the machine has. Continue when the GPU is AMD or no
vendor is named. If the user then says it is NVIDIA, Intel or Apple, decline as
above.

**WSL2 is in scope** and does not stop this gate. It is a platform family of its
own, with its own catalog entries covering `/dev/dxg`, the DXCore handoff,
ROCDXG, the distro floor, the Windows host driver and WSL 1. Do not decline a
WSL2 user or route them away untouched — run the workflow as for any other
platform. The CLI scopes entries itself, so a bare-metal Linux fix is never
offered there.

## Prerequisites

- **The `rocm` CLI.** This skill is only a driver over it; Phase 0 below installs
  it with the user's consent if `rocm --version` fails, and stops if the installed
  CLI is older than the catalog contract this skill requires. Nothing else here is
  assumed — the CLI does the probing.
- **Platform:** native Linux (in-tree `amdgpu` module + `/dev/kfd`), Windows
  (HIP SDK), or WSL2 (`/dev/dxg` + the Windows host driver).
- **No fixed ROCm version, GPU arch (`gfx…`), or container image is assumed** —
  `rocm examine`/`diagnose` detect the installed ROCm, the GPU's `gfx` target, and
  container context, and match fixes to what they find. Never hand-set
  `HSA_OVERRIDE_GFX_VERSION` (or similar footgun env vars) yourself; let the CLI
  decide.

## Workflow

Only start here once the [Scope gate](#scope-gate--check-before-anything-else)
passes — the GPU is AMD or no vendor was named.

0. **Ensure the `rocm` CLI is present.** Everything below shells out to it, so
   check first and install it if missing:

   ```
   rocm --version
   ```

   If that succeeds, go to the contract check below. If it's not found, install
   it **with the user's consent** (this fetches and runs an installer that drops
   the `rocm` and `rocmd` binaries into `~/.local/bin`). Both installers default
   to the `release` channel, so no channel argument is needed:

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

   **Then check the catalog contract.** Run:

   ```
   rocm fix --json
   ```

   It prints the catalog manifest as JSON and applies no fix. Read
   `contract_version` from that JSON and ignore the rest. Stdout holds only the
   JSON. Any `rocm:` notice goes to stderr. This skill requires
   `contract_version` 1 or higher.

   An older CLI rejects `--json` as an unexpected argument and exits with code 2.
   In [reference.md](reference.md), exit code 2 is a usage error such as an unknown fix id; here
   it means the CLI is too old. If the command exits with code 2 and an
   unexpected-argument error, or prints a lower `contract_version`, **stop**. Say
   that the installed `rocm` CLI is older than this skill requires, so the CLI is
   the side to update. Do **not** run `rocm examine`, `rocm diagnose`, or
   `rocm fix <id>`. Offer to update the CLI by re-running the installer above,
   with the user's consent, then repeat this check. If the user declines, the
   installer cannot run, or the check still fails afterwards (an older `rocm`
   earlier on `PATH` can cause that), hand the user the install page
   (https://github.com/ROCm/rocm-cli) and stop.

   Any other failure, such as a different exit code or output with no
   `contract_version`, is an error from the CLI and not proof that it is old.
   Relay the error text and stop, because reinstalling the same version would not
   clear it. Against a CLI without this contract, the entry ids, which fixes have
   a runner, and the exit codes this skill documents are guesses, and a guess
   presented as a fact is the failure this check prevents.

1. **Diagnose.** If the user named no GPU vendor, run `rocm examine --json`
   first. Status `no-amd-gpu` means the CLI found no AMD GPU. On WSL2 the status
   is `wsl` whatever the GPU, so read `has_amd_gpu` there: false means the same.
   Either way, say what the examination reported, naming the finding: on WSL2
   that is status `wsl` with `has_amd_gpu` false. Then ask which GPU the machine
   has, and ask it open. "Which GPU does this machine have?" settles it;
   "is it AMD?" does not, because a user who does not know says yes and the
   question stays open. Propose no fix until the answer is in. That result alone
   does not prove the GPU is not AMD: a container without GPU passthrough, or
   the very failure being diagnosed, can hide it. If the user confirms AMD or
   cannot say, continue with `rocm diagnose` below, which only reads; if the
   answer is NVIDIA, Intel or Apple, the scope gate declines. Any other result:
   go on. If `rocm examine` itself fails (it exits 0 whenever it ran, so a
   non-zero exit is an error from the CLI), relay the error text and stop, as
   the contract check does.

   Pass the user's error text as the symptom:

   ```
   rocm diagnose --symptom "<paste the exact error>" --json
   ```

   If the user pasted no error text, run `rocm diagnose --json` without
   `--symptom`: the CLI probes the host on its own, and a symptom you write
   yourself is matched as though the user had said it.

   Read the JSON:
   - `has_match` — **the gate.** True means a cause cleared the threshold and
     you may propose a fix; false means nothing was established, so route
     upstream. Do **not** substitute "is `matched` empty?": several checkers
     open with a nonzero base score for a merely *potentially* relevant
     situation (a container, an APU beside a discrete GPU), so a healthy host
     still returns a non-empty `matched` of sub-threshold entries. Gating on
     emptiness proposes a fix for a machine with nothing wrong. True is not
     automatically the end of the question, either: a `diagnose-only` top
     match can clear the gate from host state alone, independent of whatever
     symptom was pasted — see the refinement in step 2.
   - An **advice-only** entry in `matched[]` (today `fix-22-display-driver-old`)
     scores below the threshold on purpose and never sets `has_match`. It is
     not a cause of the symptom. Relay the entry's `title` and `evidence` to
     the user as advice, and never apply it for them. Keep routing upstream if
     `has_match` is false. Step 2's rule that a lower cause may be mentioned only
     if it also clears `min_score_for_match` does not apply to this entry.
   - `matched[]` — ranked causes, each with `id`, `title`, `score` (0–100),
     `evidence[]`, and a `fix` (with `fix_id`, `summary`, `commands`, `verify`,
     `notes`, and the `needs_sudo` / `needs_reboot` / `needs_relogin` /
     `auto_applicable` flags). `score >= 75` = high confidence; `50–74` = likely
     (confirm one more piece of evidence with the user first).
     **Read `fix.class` before you propose anything.** `diagnose-only` means
     the cause is real and explained but has no remedy — currently only
     `info-20-gpu-no-rocm-support`, which carries an `info-` id rather than a
     `fix-N` / `fix-wsl-N` one for that reason. `rocm fix` still takes that id
     and answers for it, by naming the problem and changing nothing.
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

2. **Decide what the diagnosis supports.** Take the first case that fits:

   - **`has_match` is false.** Nothing was established. Say so and route
     upstream per `route_when_no_match` (or per
     [Framework routing](#framework-routing) for an app named in the symptom).
     Propose no fix.
   - **The top match is `diagnose-only` and the user pasted a symptom its
     `evidence` does not plausibly explain.** A `diagnose-only` cause like
     `info-20-gpu-no-rocm-support` scores from host state alone (the GPU
     architecture), not from the symptom, so it can be the top match even when
     the pasted text describes something the evidence never mentions — a crash
     in the engine actually running on this host, say. Relaying the evidence
     and stopping would silently drop the user's actual complaint. Say what the
     cause does establish about the host, then route upstream for the pasted
     symptom itself, exactly as if `has_match` were false.
   - **The top match is `diagnose-only` and no symptom was pasted, or its
     `evidence` plausibly accounts for the one that was.** There is no fix to
     propose: relay its `title` and `evidence` as an explanation of the host and
     stop there.
   - **Otherwise.** Show the top match's `title`, `evidence`, plan, and `verify`
     command. Only propose applying it when the user is on board.

   Mention a lower cause only if it *also* clears `min_score_for_match`: below
   that threshold nothing was established, which is the bar the `has_match`
   bullet sets, and a `diagnose-only` top match does not lower it. The advice-only
   `fix-22-display-driver-old` entry described under `has_match` is the one
   exception: relay it as advice, never as a cause.

3. **Apply with consent.** For a fix that has a runner:

   ```
   rocm fix <fix-id>            # prompts before changing anything
   rocm fix <fix-id> --dry-run  # show the exact change, touch nothing
   rocm fix <fix-id> --yes      # required to apply in a non-interactive shell
   ```

   Only four fixes have a runner the CLI executes itself. The other 24 are
   **print-only** (bootloader, kernel, reinstall, Windows driver, …): `rocm fix
   <id>` just prints the plan for the user to run themselves — no prompt, and the
   CLI never performs those. In a diagnosis, `fix.auto_applicable` is true only
   for a fix whose `fix.class` is auto on this host, so it is false for
   `fix-9-igpu-dgpu` and for `fix-2-unset-override` on Linux and WSL2.

   One entry is in neither group. `info-20-gpu-no-rocm-support` is
   **diagnose-only**: `rocm fix` takes the id, names the problem, says there is
   no reliable fix, and exits 0 having changed nothing.

   Of the four, two do not always mutate:

   - **`fix-2-unset-override` mutates on Windows only.** On Linux and WSL2 it
     reports where the override is set and which rc files carry it, then stops
     — it will not edit the user's dotfiles. So there is no prompt to answer and
     nothing for `--dry-run` to preview. Tell the user what to edit; do not
     describe it as a change the CLI made or previewed.
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

## vLLM startup failures

Two vLLM engine failures are in the catalog. Both are **print-only**: `rocm fix`
prints the plan and the user runs it. Say so first, in one short sentence at the
start of the reply, before relaying a multi-step plan.

- **Out of memory at startup** (`fix-16-vllm-oom`, Linux and WSL2). The match is
  keyword-based, and `vllm` or `gpu-memory-utilization` has to be on the same line
  as the error. Pass one `--symptom` value on a single line: the exact error line
  prefixed with `vllm: `, for example `--symptom 'vllm: torch.OutOfMemoryError: HIP
  out of memory. ...'`. If `rocm serve` printed a `rocm diagnose --symptom '...'`
  command, use that one. Do not put the vLLM mention on a separate line: the
  match then fails, and a bare `torch.OutOfMemoryError` does not match either.
  Relay both cases the CLI returns: lowering the reservation helps only on a
  shared or busy GPU, and a model that does not fit needs a smaller or quantized
  model.
- **OSError or ImportError at engine start** (`fix-17-torch-dlpack`, Linux and
  WSL2). Pass the exact error line, for example one naming `libtorch_cuda.so`; no
  `vllm` mention is needed. The match is *likely*, not high confidence, so say that
  and have the user check the engine's torch version first, as the printed plan
  does, before the engine reinstall it ends with. It is not a group membership
  change, so do not suggest `render` or `video` group edits for it.
- **Native Windows** gets no match from either entry, which is correct because
  the vLLM engine does not run there. Do not route that user to an issue tracker.
  Relay what the CLI itself says for vLLM on native Windows (`rocm serve` and
  `rocm engines install vllm`): vLLM is not supported there, so use WSL or Linux,
  or the Lemonade engine.

`rocm diagnose --model <ref>` answers whether a model fits before a download.

## Rules

- Never invent a fix. If `rocm diagnose` returns no match, route upstream.
- Never run a mutating fix without the user's explicit OK; prefer `--dry-run`
  first. New failure modes are added to the CLI catalog, not improvised here.

Read [reference.md](reference.md) when `rocm fix` exits non-zero (its exit codes
are listed there), before telling the user which platforms a fix id applies to,
and for the full closed catalog.
