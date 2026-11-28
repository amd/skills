---
name: rocm-doctor
description: >-
  Diagnoses why ROCm, the HIP SDK, PyTorch, or llama.cpp is broken on an AMD GPU on
  Linux, Windows, or WSL2, then applies a low-risk fix with consent or hands back
  the exact next step. Also routes Lemonade, LM Studio, and Ollama problems
  upstream. Use when ROCm or HIP "isn't working",
  torch.cuda.is_available() is False, rocminfo / hipInfo can't see the GPU,
  hipErrorNoBinaryForGpu, HSA_STATUS_ERROR_INVALID_ISA, "invalid device function",
  "no kernel image is available", cannot open or permission denied on /dev/kfd,
  "ROCk module is NOT loaded", a missing libamdhip64.so / amdhip64_6.dll /
  hipblas.dll / vcruntime140_1.dll, an HSA_OVERRIDE_GFX_VERSION page fault, an
  iGPU+dGPU crash, a container that can't see the GPU, a vLLM startup out-of-memory
  error or libtorch_cuda.so OSError/ImportError, or a failed package-manager /
  amdgpu-install / DKMS install. Do NOT activate for NVIDIA, Intel, or Apple GPUs,
  or for a first-time ROCm setup on a clean machine.
---

# ROCm Doctor

Given a "ROCm / PyTorch / llama.cpp isn't working on my AMD GPU" complaint,
identify which **known misconfiguration** is the cause and either fix it (with
consent) or hand back the exact next step.

The `rocm` CLI owns the probe, the closed failure-mode catalog, the scoring and
the fixes. This skill drives the CLI, reads what it returns, and decides only what
that result supports (step 2). The catalog is a **closed list**: a symptom that
matches no known mode goes upstream, never guessed at.

## Scope gate — check before anything else

Read the user's symptom and answer one question first: **is this an AMD GPU?**

If it is **not** — an **NVIDIA / Intel / Apple** GPU — then **stop and decline**:

- Say plainly that it is **out of scope** for this skill and why (not an AMD GPU).
- Give **no** troubleshooting for it: no commands to run, no driver or CUDA
  advice, no diagnostic checklist, no "try this first". Point at the vendor's own
  documentation and stop there; do not invent a URL.
- Do **not** run `rocm examine` / `rocm diagnose` / `rocm fix`.

Confidently wrong advice for a stack this skill does not cover is worse than no
advice.

A failed or half-finished install is in scope; a request to set ROCm up for the
first time on a clean machine is not, because that is a setup task and not a
diagnosis. For that request, say so and point at AMD's ROCm installation
documentation (for WSL2, AMD's guide at
https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/wsl/howto_wsl.html);
do not invent other URLs.

If the user names no GPU vendor, treat ROCm / HIP wording as AMD and continue to
the Workflow, whose step 1 checks what the machine has. If the user then says it is
NVIDIA, Intel or Apple, decline as above.

**WSL2 is in scope** and does not stop this gate. It is a platform family of its
own, with its own catalog entries, and the CLI scopes entries itself, so a
bare-metal Linux fix is never offered there. Do not decline a WSL2 user or route
them away untouched: run the workflow as for any other platform.

## Prerequisites

- **The `rocm` CLI.** This skill is only a driver over it. Phase 0 below installs
  it with the user's consent when it is missing, and stops when the installed CLI
  is older than the catalog contract this skill requires. Nothing else is
  assumed: the CLI does the probing.
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

### Phase 0 — ensure the `rocm` CLI is present and current

Everything below shells out to the CLI, so check first:

```
rocm --version
```

If that succeeds, go to the contract check. If it is not found, install the CLI
**with the user's consent**. Before asking, say what the installer does: it runs a
script from the project's GitHub repository that installs `rocm` and `rocmd` in
`~/.local/bin` by default (Windows: `%USERPROFILE%\.local\bin`), edits the user's
PATH configuration, and writes its own config under the home directory. It prints
what it changed. To skip the PATH edit, set `ROCM_CLI_UPDATE_SHELL_PATH=0` for
the installer: on Linux and WSL2 put it before `sh` in the command below, on
Windows run `$env:ROCM_CLI_UPDATE_SHELL_PATH = "0"` before the PowerShell
command.

- **Linux and WSL2 (x86_64 only):**
  ```
  curl -fsSL https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.sh | sh
  ```
- **Windows (PowerShell, x86_64 only):**
  ```
  irm https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.ps1 | iex
  ```

There is no macOS build and no ARM build: both installers refuse other hosts, so
on such a host hand the user the install page (https://github.com/ROCm/rocm-cli)
and stop rather than suggesting the command anyway. Both installers default to the
`release` channel, so pass no channel argument. Use `nightly` (`sh -s -- nightly`,
or `$env:ROCM_CLI_CHANNEL = "nightly"`) only when the user asks for an unreleased
build; it is a moving tag, so it names whatever was published last, not a fixed
build.

After any install or update, the shell you are in may not find the new `rocm` first
on the PATH, and an older one earlier on the PATH can still answer. Run the check by
the full path of the directory the installer says it installed to, and use that full
path wherever this skill writes `rocm`, in commands handed to the user too (in
PowerShell as `& "<full path>"`, which a path with a space needs), for the rest of
the session. If the full path fails as well, hand the user the install page
and stop.

**Then check the catalog contract.** Run:

```
rocm fix --json
```

It prints the catalog manifest as JSON and applies no fix. Read `contract_version`
from that JSON, and `exit_codes`, which names the numbers `rocm fix` exits with, and
`exit_code_descriptions` when present; ignore the rest. Stdout holds only the JSON;
any `rocm:` notice goes to stderr. This skill requires `contract_version` 1 or
higher.

- **An older CLI** rejects `--json` as an unexpected argument and exits with code 2
  (a usage error), or prints a lower `contract_version`. **Stop.** Say that the
  installed `rocm` CLI is older than this skill requires, so the CLI is the side to
  update. Do **not** run `rocm examine`, `rocm diagnose`, or `rocm fix <id>`.
  Offer to update the CLI by re-running the installer above, with the user's
  consent, then repeat this check by the full path. If the user declines, the
  installer cannot run, or the check still fails afterwards, hand the user the
  install page and stop.
- **Any other failure**, such as a different exit code or output with no
  `contract_version`, is an error from the CLI and not proof that it is old. Relay
  the error text and stop, because reinstalling the same version would not clear
  it.

Against a CLI without this contract, the entry ids, fix classes and field names
this skill documents are guesses, and a guess presented as a fact is the failure
this check prevents.

### Steps 1–4 — diagnose, decide, apply, verify

1. **Diagnose.** If the user named no GPU vendor, run `rocm examine --json`
   first. Status `no-amd-gpu` means the CLI found no AMD GPU. On WSL2 the status
   is `wsl` whatever the GPU, so read `has_amd_gpu` there: false means the same.
   Either way, say so and ask which GPU the machine has, as an open question:
   "is it AMD?" invites a yes. End the reply on the question, with no
   `rocm diagnose` run and no fix offered in it, because a diagnosis or plan
   shown beside the question is a fix proposed before the question is settled.
   That result alone does not prove the GPU is not AMD: a container without GPU
   passthrough, or the very failure being diagnosed, can hide it. If the user
   confirms AMD or cannot say, continue with `rocm diagnose` below, which only
   reads, in the reply after the one that asked; if the answer is NVIDIA, Intel
   or Apple, the scope gate declines. Any other result: go on in the same reply.
   `rocm examine` exits 0 whenever it ran, so a non-zero exit is an error from
   the CLI: relay the error text and stop, as the contract check does.

   Pass the user's error text as the symptom:

   ```
   rocm diagnose --symptom='<exact error text>' --json
   ```

   In bash or zsh, quote the pasted text as one literal argument in single quotes,
   never double quotes, so that a paste containing `$(…)` or a backtick cannot run
   as a command. A `'` inside the paste ends the quote early and the rest is parsed
   by the shell, so write each `'` as `'\''`. In any other shell, or if you are
   unsure the quoting holds, do not run the command with a pasted symptom, because
   a paste could otherwise run as a command: hand the user the command to run
   themselves, and ask them to paste the JSON back (a run without `--symptom`
   would answer a different question). Write the command for them, in their
   shell's form (never the bash-escaped one above), only when the paste is one
   line with no `"`, no `\` and no `%`:
   - PowerShell: `rocm diagnose '--symptom=<text>' --json`, with every single quote
     in the text doubled by repeating it. The typographic ‘ ’ ‚ ‛ are single quotes
     there too (`it’s` becomes `it’’s`). One left undoubled ends the quote early,
     and a `$(…)` after it runs.
   - cmd.exe: `rocm diagnose "--symptom=<text>" --json`.

   For any other paste, shells split or rewrite the text in ways that differ by
   shell and version, so write no form for it. Ask the user for a single line
   without those characters, in their own words (never trim it yourself); only if
   they cannot give one, have them add `--symptom=` and their text, quoted for
   their shell, to `rocm diagnose --json` themselves.

   Keep the `=` in every form: with a space instead, a paste that begins with `-`
   (an exit code such as `-1073741819`) is read as a flag and the command fails.

   If the user pasted no error text, run `rocm diagnose --json` without
   `--symptom`: the CLI probes the host on its own, except for a vLLM startup
   failure, where the vLLM section says what to ask first. Do not write a
   symptom yourself, because the CLI matches it as though the user had said it; the
   one edit to the user's text this skill allows, and the vLLM rule, are both under
   [vLLM startup failures](#vllm-startup-failures).

   `rocm diagnose` exits 0 whether or not it matched. If it fails or prints
   something that is not JSON, relay the error text and stop.

   Read the JSON:
   - `has_match` — **the gate.** True means a cause cleared
     `min_score_for_match`; false means nothing was established. Do **not**
     substitute "is `matched` empty?": several checkers open with a nonzero base
     score for a merely *potentially* relevant situation (a container, an APU
     beside a discrete GPU), so a healthy host still returns a non-empty
     `matched` of sub-threshold entries. Gating on emptiness proposes a fix for a
     machine with nothing wrong.
   - `matched[]` — ranked causes, each with `id`, `title`, `score`, `evidence[]`,
     `symptom_matched` (absent before v0.3.0; true when the passed text contributed
     to the entry, however weakly; false when only the host's state did, and always
     false with no text passed) and a `fix` whose `fix_id`, `summary`, `commands`, `verify`, `notes`,
     `class`, `needs`, `needs_sudo`, `needs_relogin` and `needs_reboot` are the
     fields this skill acts on (its "plan" is `summary` and `commands`). **Read
     `fix.class` before you propose anything**: step 3 says what each class lets
     the CLI do.
   - An entry that scores below `min_score_for_match` on purpose and never sets
     `has_match` is **advice**, not a cause: `fix-22-display-driver-old` is one, and
     the Failure mode cell of [reference.md](reference.md)'s catalog table says
     which entries are advice only.
   - `out_of_scope` — set only when the host's platform family has no catalog
     entries at all, and then `matched` is empty.
   - `route_when_no_match` — the upstream tracker for when `has_match` is false.
     The CLI picks it from the *host-detected* framework, not from the symptom
     text or from whose bug it is; step 2 says when that makes it the wrong
     tracker.

2. **Decide what the diagnosis supports.** This judges `matched`. A `--model`
   question is answered from its `model` object and is never routed upstream on
   `has_match`. Take the first case that fits:

   - **`out_of_scope` is set.** Nothing was checked: say so rather than implying
     the machine looks fine. If the symptom names an app in
     [Framework routing](#framework-routing), send the user to that app's
     tracker, which applies on any platform. Otherwise relay the `out_of_scope`
     message and stop. `route_when_no_match` does not apply here.
   - **`has_match` is false.** Nothing was established. Say so and propose no fix.
     Then route the symptom:
     - A vLLM startup failure on native Windows has no tracker to go to. Say that
       vLLM is not supported on native Windows and to use WSL or Linux, or the
       Lemonade engine (`--engine lemonade`). State this yourself; running
       `rocm serve` or `rocm engines install vllm` to read the CLI's refusal adds
       nothing.
     - A failure of `rocm serve` itself, with any engine (Lemonade included), goes
       to this project's tracker, https://github.com/ROCm/rocm-cli/issues. It is
       rocm's own run of the engine, so this bullet wins over the next one.
     - Any other symptom that names an app in
       [Framework routing](#framework-routing) goes to that app's tracker, even if
       it also names PyTorch or llama.cpp ("Lemonade's llama.cpp server crashes"
       goes to Lemonade). Only when no app is named does a symptom that names
       PyTorch or llama.cpp go to that project's tracker (listed in
       [reference.md](reference.md)).
     - Anything else goes to `route_when_no_match` as returned. Give the user the
       URL without the CLI's trailing note, and pass the note on as advice (for
       pytorch: tag the issue with the rocm label).
   - **The user described a problem, and no match that cleared
     `min_score_for_match` plausibly explains it.** Many entries score from host
     state alone (a small `/dev/shm`, an old distro, a missing runtime), so a match
     can clear the threshold while nothing in it concerns what the user described:
     an `ImportError` in an engine, say, or an app that will not use the GPU.
     Judge each entry's fit by its `evidence`. `symptom_matched` says only whether
     the passed text contributed, never whether the entry explains the problem: a
     true can be a weak keyword hit, and a false can still be the cause, since some
     checks ignore the text (an unsupported GPU, say). Presenting a match that does not concern the problem as
     the cause silently drops the user's actual complaint. Say what the matches
     establish about the host and offer their fixes (step 3), where they have one,
     as worth doing anyway and never as the remedy for the problem described. Then
     route that problem as in the `has_match` is false case.
   - **Otherwise.** Take the highest-ranked match that cleared the threshold and
     explains the problem the user described (with none described, the top match).
     If its `fix.class` is `diagnose-only`, the cause is real and explained and no
     remedy exists: relay its `title`, `evidence`, `summary` and `notes` as an
     explanation of the host, and offer no fix for it. With any other class, show
     its `title`, `evidence`, plan, `notes` and `verify` command, and propose
     applying it only when the user is on board. `notes` carry warnings the user
     must see before acting. A score at or above `high_confidence_threshold` is
     high confidence; a lower one that cleared `min_score_for_match` is likely, so
     confirm one more piece of evidence with the user first. Any other match that
     cleared the threshold without explaining the problem is offered as the case
     above offers its fixes (worth doing anyway, never the remedy); it adds no
     routing.

   Mention a lower cause only if it *also* clears `min_score_for_match`: below
   that threshold nothing was established. An advice entry (step 1) is the
   exception: relay its `title` and `evidence` to the user as advice, never as a
   cause, and never apply it. It does not change which case above applies, so
   `has_match` false still routes upstream.

3. **Apply with consent.** Never run a mutating fix without the user's explicit
   OK. Pass the full `fix_id` from the diagnosis. What `rocm fix <fix-id>` does is
   the entry's `fix.class` on the machine the diagnosis describes, so run it only
   there. A diagnosis from `rocm diagnose --distro`, taken from a Windows host,
   describes the WSL distribution while `rocm fix` would act on the host, and
   `--distro` needs nothing installed in the distribution: do not run `rocm fix`
   for it, and hand the user the diagnosis's `fix.commands` to run inside the
   distribution.

   - `auto` — carries the change out itself, after confirming.
   - `needs-argument` — carries it out only once the argument `fix.needs` names is
     supplied; without it, it prints what to pass and changes nothing. For
     `fix-9-igpu-dgpu` that is `--device-index N`, and the bare command prints the
     `rocminfo` / `hipInfo` query that identifies the discrete GPU's index. Once
     the user knows N, re-run with it.
   - `print-only` — prints the plan for the user to run and changes nothing. There
     is no prompt, and nothing for `--dry-run` to preview. Such an entry can still
     inspect the machine: `fix-2-unset-override` on Linux and WSL2 reports where
     the override is set and which rc files carry it, and will not edit the user's
     dotfiles. Tell the user what to edit; do not describe it as a change the CLI
     made or previewed.
   - `diagnose-only` — names the problem, says there is no reliable fix, and
     changes nothing.

   [reference.md](reference.md)'s table marks the entries that are not print-only.
   The other 25 are **print-only** on every host (bootloader, kernel, reinstall,
   Windows driver, …): the CLI never performs those.

   To apply an `auto` fix, or a `needs-argument` fix with its argument, go in this
   order. For a `needs-argument` fix, add the argument (`--device-index N` for
   `fix-9-igpu-dgpu`) to both commands: without it each exits 0 having changed
   nothing.

   - (a) `rocm fix <fix-id> --dry-run` shows the exact change and touches nothing.
     It needs no terminal.
   - (b) Show that to the user and get their explicit OK in the conversation. If
     the change reaches only new sessions (`needs_relogin`, `needs_reboot`, or a
     PATH or environment edit), say so here, before the OK: the user must log out
     and back in, reboot, or open a new terminal before step 4's check, which fails
     in the current session until then. If the fix has `needs_sudo`, say too that
     `rocm` runs `sudo` with no input and kills it after one minute. Where sudo
     needs a password, it prompts on the user's terminal and the fix applies if
     they answer in time; cached credentials, a NOPASSWD rule or running as root
     skip the prompt. A shell with no controlling terminal fails at once, and an
     unanswered prompt is stopped after the minute; both are the `failed` exit.
   - (c) `rocm fix <fix-id> --yes` applies it. `--yes` is the OK from (b) passed on
     to the CLI, so add it only after that OK. Without it the CLI refuses in an
     agent's shell, which has no terminal to ask on, and exits with the `declined`
     code. Nobody was asked, so do not tell the user they declined.

   Judge the outcome from the CLI's output and from step 4, not from the exit status
   alone: a run can exit 0 having changed nothing, because the setting was already
   in place. Read a non-zero exit by its `exit_codes` name, the manifest's
   `exit_code_descriptions` entry for it when the manifest has one (what it means)
   and the output; relay the output, and never report it as applied. An exit the
   map does not name gets the same treatment. What to do:
   - `failed` — the output says whether the change was made; take the first case
     that matches. A stderr line starting `Could not read the current` means a read
     before the change failed (a timed-out read included) and nothing was changed,
     under `--dry-run` too, which then shows no plan, so the entry printed before it
     is not a preview: relay the cause and offer the same `rocm fix` once it
     clears. Never hand over the fix's commands instead; for fix-9 that is the
     `setx` overwrite the CLI declined to risk. Otherwise, if the output says
     `may or may not have changed`, a step did not finish: claim neither until the
     check the output names has run, then offer the `rocm fix` it names. Otherwise
     give the user the diagnosis's `fix.commands` to run in their own terminal.
   - `not_applicable` — the entry is real but not for this machine, or a
     precondition is unmet (for `fix-4-render-group`: no `$USER`/`$LOGNAME`, no
     `usermod`, or no `sudo` and not root): nothing was applied.
     Do not offer `--dry-run`, which exits the same way; when the cause is a
     precondition rather than the machine, hand the user `fix.commands` to run
     themselves.
   - `unknown_id` — a stderr line starting `Unknown fix-id:` means the id is wrong:
     take it again from the diagnosis. Exit 2 is also a malformed command, whose
     line starts `error:`: fix the command.
   - `internal` — a fault in the CLI, not in the machine: stop.
   - `declined` — nothing changed; without `--yes` in an agent's shell it means
     nobody was asked (step (c) above).

4. **Verify.** Have the user run the `verify` command from the diagnosis.

Use `rocm examine --json` when you only need the host state (GPU, driver, ROCm
install, groups, framework) without a diagnosis; only the JSON carries the groups
and the framework. `rocm diagnose --model <ref> --json` answers whether a model
fits before a download: relay `model.verdict` and `model.evidence`, and read the
rest of the `model` object as [reference.md](reference.md) describes.

## Framework routing

`rocm diagnose` covers frameworks that build against the **system** ROCm/HIP:
**PyTorch** and **llama.cpp**. Diagnose those normally.

Apps that ship their **own** ROCm runtime get routed by you, not by the CLI:
`route_when_no_match` (step 1) can never name one. Still run `rocm diagnose`
first, because the catalog covers host faults these apps depend on (a missing
MSVC runtime, an old distro), and a match that explains the symptom applies.
Step 2 says when to route.

- **Lemonade** → https://github.com/lemonade-sdk/lemonade/issues
- **Ollama** → https://github.com/ollama/ollama/issues
- **LM Studio** → https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues

## vLLM startup failures

Three vLLM engine failures are in the catalog. All are **print-only**: `rocm fix`
prints the plan and the user runs it. Say so first, in one short sentence at the
start of the reply, before relaying a multi-step plan.

If `rocm serve` printed a `rocm diagnose --symptom '...'` command, take the text
inside its quotes as the symptom and run it the way step 1 shows. The printed form
has no `--json`, so it would give step 2 no `has_match` to read.

If the user describes a vLLM startup failure on Linux or WSL2 in words and pasted no
error line, ask for the exact error line (or that printed command) before
diagnosing. The vLLM entries score only the symptom text, so a description alone
matches none of them, and step 2 would route a failure the catalog covers to a tracker.
If the user cannot give the line, or the platform is unstated, run
`rocm diagnose --json` without `--symptom` in the next reply, and say that without
the error line no vLLM entry can match; step 2 judges what comes back.

- **Out of memory at startup** (`fix-16-vllm-oom`, Linux and WSL2). The match is
  keyword-based, and `vllm` or `gpu-memory-utilization` has to be on the same line
  as the error. Without a printed command, pass one `--symptom` value on a single
  line: the exact error line prefixed with `vllm: `, for example

  ```
  rocm diagnose --symptom='vllm: torch.OutOfMemoryError: HIP out of memory. ...' --json
  ```

  Add the prefix only when vLLM is known to have printed the line (the user said
  so, or the log shows the engine), and tell the user you added it: this is the one
  edit to the user's text this skill allows. When the engine is unknown, pass the
  line unchanged and accept no match, because a prefix on another framework's
  out-of-memory error would blame vLLM for it. Do not put the vLLM mention on a
  separate line: the match then fails, and a bare `torch.OutOfMemoryError` does not
  match either. Relay both cases the CLI returns: lowering the reservation helps
  only on a shared or busy GPU, and a model that does not fit needs a smaller or
  quantized model.
- **OSError or ImportError at engine start** (`fix-17-torch-dlpack`, Linux and
  WSL2). Pass the exact error line, for example one naming `libtorch_cuda.so`; no
  `vllm` mention is needed. Relay the confidence the CLI reports: a paste of only
  the `libtorch_cuda.so` line scores at the minimum, so it is likely and not high
  confidence, and a fuller traceback scores higher. Have the user check the
  engine's torch version first, as the printed plan does, before the engine
  reinstall it ends with. It is not a group membership change, so do not suggest
  `render` or `video` group edits for it.
- **An op missing from `torch.ops` at engine start** (`fix-23-vllm-extension-load`,
  Linux and WSL2): `AttributeError: '_OpNamespace' '_C' object has no attribute
  '...'`. One of vLLM's compiled extensions did not load. Pass the error line with
  its log context, or the command `rocm serve` printed: every PyTorch-ecosystem
  library with a `_C` extension raises the identical line, so the entry matches
  only when the text also identifies vLLM (the word `vllm`, the
  `(EngineCore pid=...)` prefix, or a vLLM extension name such as `_rocm_C`). If a
  bare `AttributeError` line gets no match, ask for the surrounding log lines
  rather than concluding there is nothing to find. Relay the plan, which finds the
  failing extension inside `rocm engines shell vllm`. Do not suggest an engine
  reinstall, which fetches the same wheel, or group edits.

Native Windows gets no match from any of them, because the vLLM engine does not
run there. Do not route that user to an issue tracker, whatever else matched; step
2 says what to tell them.

## Rules

- Never suggest running on the CPU instead of the GPU, whatever the diagnosis: not
  for a model that does not fit, and not for a GPU ROCm does not support. Keeping
  some of a model's layers in system RAM on a GPU that works (llama.cpp's
  `--n-gpu-layers`) is not CPU fallback: relay that suggestion when
  `rocm diagnose --model` makes it.
- Relay a `suggestions[]` entry of a model estimate in the CLI's words, as the
  llama.cpp option it names. `rocm serve` takes no context size, KV cache type or
  GPU layer count (`model.warnings` says so), so never write one of them into a
  `rocm serve` command.
- Run `rocm diagnose --report` or `--send` only when the user asks: they prepare a
  report for the user to send and are no part of diagnosing a problem.
- Never invent a fix: offer only a remedy the CLI returns, a diagnosis `fix` or the
  steps in `model.fix`. New failure modes are added to the CLI catalog, not
  improvised here.

Read [reference.md](reference.md) before telling the user which platforms a fix id
applies to, before reading the `model` object from `rocm diagnose --model`, before
handing over a PyTorch or llama.cpp tracker, and for the full closed catalog.
