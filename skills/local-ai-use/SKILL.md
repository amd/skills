---
name: local-ai-use
description: >-
  Routes image generation, text-to-speech, and speech-to-text through a local
  Lemonade Server so the agent stops paying for cloud calls on those modalities.
  Use when the user wants to save tokens, save cost, or save money using local
  AI; default to local, offline, on-device, or private image generation,
  transcription, or text-to-speech in this workspace; stop using DALL-E,
  Whisper-as-a-service, ElevenLabs, or other paid multimodal APIs; route the
  agent's image, TTS, or STT tool calls to a local model; or mentions Lemonade
  Server, OmniRouter, SD-Turbo, kokoro, Whisper, Ryzen AI, NPU/iGPU/dGPU
  inference, or "use local for images but cloud for chat". Run once per
  workspace; the rule it installs handles every later request.
---

# Local AI Use (route image, TTS, STT through Lemonade)

This is a **meta-skill**. You run it once. After that, every later request that
needs image generation, text-to-speech, or speech-to-text uses the local
[Lemonade Server](https://lemonade-server.ai) instead of a cloud API. The
agent's own LLM keeps handling text; only the expensive multimodal calls move
on-device.

The skill does three things:

1. **Makes sure local Lemonade is installed and running.** If no modern
   `lemonade` CLI is found, the setup script installs the latest version of
   Lemonade on the user's behalf. Modern Lemonade has no `serve` command — the
   Lemonade service (the `lemond` daemon) auto-starts on install and is managed
   by the OS — so the setup script waits for the service and, if it stays down,
   prints the exact OS-specific command to start it (e.g. `sudo systemctl start
   lemond` on Linux).
2. **Verifies that local Lemonade is reachable.**
3. **Drops a `Local AI Use` block into the workspace `AGENTS.md`** so the agent
   reads the routing rule on every later turn, in Cursor, Claude Code, Codex,
   Gemini CLI, and any other agent that respects `AGENTS.md`.

> **Requires modern Lemonade (v10.1.0 or newer).** Modern Lemonade unified
> everything under one `lemonade` CLI (`lemonade status`, `lemonade pull`, ...)
> driving an always-on `lemond` service. This skill targets that and installs
> Lemonade only from the official installers (see below) — never via
> `pip install lemonade-sdk`, which is a separate, older release line. If an
> older/incompatible `lemonade` is already on the PATH (from any channel — an
> old `.msi`/`.deb` or a pip install), it will shadow the modern CLI; uninstall
> it first (see the removal commands in Step 1a) before running this skill.

Models are **not** downloaded during setup. Each default model is pulled
lazily, on first use, by the routing rule (e.g. the first image request pulls
the image model). This keeps setup fast and avoids gigabytes of downloads the
user may never need.

## When to use this skill

Use this skill when **all** of the following are true:

- The user wants local Lemonade. If it is not yet installed, the setup script
  installs the latest version for them automatically.
- The user accepts the default Lemonade endpoint `http://localhost:13305`.
- The user wants the change to be **persistent** across future turns and
  agent restarts (the rule is written to disk).

If the user is instead **embedding** Lemonade as a private subprocess inside
an app installer, do not use this skill; use `local-ai-app-integration`
instead.

## Prerequisites

- **OS:** Windows 11 x64, Ubuntu/Debian x64, or macOS (beta).
- **Lemonade:** the setup script installs it if missing. It downloads and
  silently installs the latest version (Windows `lemonade.msi`, the
  Ubuntu/Debian `ppa:lemonade-team/stable` PPA, or the macOS `.pkg`). The
  `lemond` service auto-starts after install; the script waits for it rather
  than launching it. On Linux/macOS the install needs `sudo`. Pass
  `--no-install` if the user wants to install it themselves instead.
- **Disk:** ~8 GB free for the three default models (SD-Turbo + Whisper-Tiny
  + kokoro-v1), plus ~0.1 GB for the installer itself.
- **Network:** required for the install download and the first `lemonade pull`
  of each model. After that, every modality runs offline.

## The opinionated path

Run this checklist top to bottom. Track progress against it; do not move on
until each step verifies.

```
[ ] 1. Ensure Lemonade Server is installed and running (auto-install if missing)
[ ] 2. Install the routing rule into the workspace AGENTS.md
```

The single command that does both steps in one shot is:

```bash
python scripts/setup_local_ai.py
```

**Always run this script first — even if Lemonade is already installed and the
server is already running, and even before generating a single image.** Writing
the routing rule into `AGENTS.md` is what makes this skill complete; skipping it
because "Lemonade is already up" leaves the workspace unconfigured for future
turns. The script is safe to run in that case: it detects the running service,
skips the install, and just writes the rule.

It auto-installs the latest version of Lemonade if no modern `lemonade` CLI
is found, waits for the auto-started `lemond` service, then writes the rule.
The script is idempotent: re-running it on a fully configured workspace is a
no-op apart from a healthcheck. Read the sections below for what to do when
each step fails.

---

## Step 1: ensure Lemonade Server is installed and running

`scripts/setup_local_ai.py` handles this end to end, but here is what it does
so you can do it by hand or debug it:

**1a. Is a modern `lemonade` CLI installed?** Run `lemonade status`. The check
is by *capability*, not by name: modern Lemonade prints `Server is running...`
or `Server is not running`. If instead you get an "invalid choice" / usage
error, the `lemonade` on `PATH` is an old, incompatible build that predates the
unified CLI (v10.1.0) — do **not** use it. It could have come from any install
channel, so remove it based on how it was installed, then re-run this skill (or
install Lemonade manually):

| Installed via | Uninstall with |
|---|---|
| Windows `.msi` | `winget uninstall -e --id AMD.LemonadeServer`, or Settings > Apps > Installed apps > Lemonade Server > Uninstall |
| Ubuntu/Debian apt/PPA | `sudo apt remove lemonade-server` |
| pip | `pip uninstall lemonade-sdk` |
| macOS `.pkg` | Delete the installed `Lemonade.app` / remove the package receipt |

Never try to drive or auto-remove it for the user.

If no `lemonade` is found at all, install the latest version on the user's
behalf:

| OS | Install |
|---|---|
| Windows | Download `lemonade.msi` from the [latest release](https://github.com/lemonade-sdk/lemonade/releases/latest/download/lemonade.msi) and run `msiexec /i lemonade.msi /qn` (silent, per-user, no elevation). |
| Ubuntu/Debian | `sudo add-apt-repository -y ppa:lemonade-team/stable && sudo apt-get update && sudo apt-get install -y lemonade-server` (the apt package is `lemonade-server`; the CLI you then run is `lemonade`) |
| macOS (beta) | Download the `Lemonade-<ver>-Darwin.pkg` from the latest release and run `sudo installer -pkg Lemonade-<ver>-Darwin.pkg -target /`. |

After a Windows install the CLI lands in `%LOCALAPPDATA%\lemonade_server` and
is added to the *user* PATH (new shells only); the setup script probes that
directory so it works in the same run.

**1b. Is the service running?** Check `lemonade status --json`. The `lemond`
service auto-starts on install — there is **no** `lemonade serve` in modern
Lemonade.

| `lemonade status` says | Action |
|---|---|
| `Server is running on port 13305` | Continue to Step 2. |
| `Server is not running` | Wait a few seconds for the auto-started service (the script polls `/api/v1/health`). If it stays down, start it via the OS service manager: `sudo systemctl start lemond` (Linux system install) or `systemctl --user start lemond` (per-user install); `launchctl load /Library/LaunchDaemons/com.lemonade.server.plist` (macOS); the Lemonade tray app or `Start-Service lemond` (Windows). |

Only if the automatic install genuinely fails (no `apt-get`, no `sudo`,
download blocked) should you stop and point the user at
<https://lemonade-server.ai/docs/guide/install/>.

The rest of this skill assumes the endpoint is `http://localhost:13305/api/v1`
and no API key is required (the system-wide server defaults to no auth on
loopback). If the user has set `LEMONADE_API_KEY`, the routing rule template
in `templates/local-ai-rule.md` shows where to add the `Authorization` header.

### Default modality models (pulled on first use, not during setup)

Setup does **not** download these. The installed rule pulls each one the first
time that modality is requested. They are the **Lite Collection** defaults from
Lemonade OmniRouter, sized to keep token-and-cost savings real on commodity
hardware:

| Modality | Model | Size | Why this default |
|---|---|---|---|
| Image generation | `SD-Turbo` | ~5 GB | Single-step generation, runs on CPU and AMD iGPU/dGPU |
| Text-to-speech | `kokoro-v1` | ~0.3 GB | Only TTS model Lemonade currently supports; CPU-only, low latency |
| Speech-to-text | `Whisper-Tiny` | ~0.1 GB | Smallest Whisper; fast on CPU. Upgrade to `Whisper-Large-v3-Turbo` if accuracy matters more than latency. |

To write a different model ID into the rule, pass it to the setup script. For
example, to make future image requests use SDXL:

```bash
python scripts/setup_local_ai.py --image-model SDXL-Turbo
```

That model ID is written into the installed `AGENTS.md` rule and pulled on its
first use. The same pattern works for `--tts-model` and `--stt-model`. For
larger / higher-quality alternatives (`SDXL-Turbo`, `Flux-2-Klein-4B`,
`Whisper-Large-v3-Turbo`), see the
[model picker in reference.md](reference.md#model-picker).

## Step 2: install the routing rule into AGENTS.md

The rule is a Markdown block stored in [`templates/local-ai-rule.md`](templates/local-ai-rule.md).
Append it to the workspace's `AGENTS.md` (create the file if missing). Both
Cursor and Claude Code load `AGENTS.md` automatically on every turn, so the
agent will see the rule on its next message without any further setup.

`scripts/setup_local_ai.py` does this for you. It bakes the selected endpoint
and model IDs into the rule, surrounded by stable markers so re-running the
script replaces the block in place rather than appending a second copy. The
markers look like:

```
<!-- BEGIN amd-skills:local-ai-use -->
...rule...
<!-- END amd-skills:local-ai-use -->
```

If you write the file by hand, keep those exact markers. The script relies
on them for idempotent updates.

If the user's agent only respects a different convention, mirror the same
block to:

- `CLAUDE.md` (Claude Code, project-scoped) or `~/.claude/CLAUDE.md` (global)
- `.cursor/rules/local-ai-use.mdc` (Cursor user/project rules)
- `GEMINI.md` (Gemini CLI)

The rule's content is identical; only the file location changes.

---

## What changes after this skill runs

From the next turn onward, the agent reads the rule in `AGENTS.md` on every
message. The rule explicitly tells the agent:

- **For image generation:** call `POST /api/v1/images/generations` on the
  local server. Do **not** call any cloud image API and do **not** use the
  built-in `GenerateImage` tool (that path bills tokens to the cloud
  provider).
- **For text-to-speech:** call `POST /api/v1/audio/speech`. Do **not** call
  cloud TTS providers (OpenAI TTS, ElevenLabs, etc.).
- **For speech-to-text:** call `POST /api/v1/audio/transcriptions`. Do
  **not** call cloud transcription providers.
- **Fallback:** only fall back to a cloud API after one local attempt has
  failed *and* the user has been told the local call failed. Never
  silently fall back; the whole point of this skill is to keep cost
  predictable.

The agent's own text reasoning continues to use whatever LLM Cursor / Claude
Code / Codex is configured with. This skill does not redirect chat tokens;
it only redirects the multimodal calls that would otherwise leave the
machine.

## Troubleshooting cheatsheet

| Symptom | Cause | Recovery |
|---|---|---|
| `lemonade: command not found` | CLI not installed | Re-run `python scripts/setup_local_ai.py` (auto-installs the latest version). If it just installed on Windows, open a new shell so the user PATH refreshes, or the script will find it under `%LOCALAPPDATA%\lemonade_server`. |
| `status` gives an "invalid choice" / usage error | An old, incompatible `lemonade` (pre-v10.1.0, from any install channel) is shadowing the modern CLI | Uninstall it the way it was installed (see the Step 1a table: `winget uninstall -e --id AMD.LemonadeServer` / `sudo apt remove lemonade-server` / `pip uninstall lemonade-sdk`), then re-run the setup script or install Lemonade from the docs link. |
| `Server is not running` | `lemond` service stopped | Start it via the OS service manager — `sudo systemctl start lemond` / `systemctl --user start lemond` (Linux), `launchctl load /Library/LaunchDaemons/com.lemonade.server.plist` (macOS), or the tray app / `Start-Service lemond` (Windows). There is no `lemonade serve`. |
| `POST /v1/images/generations` returns 404 model not found | Image model not downloaded | `lemonade pull SD-Turbo` and retry. |
| `lemonade pull` keeps printing `Progress: NN%` but never finishes | Download target is a bad path (out of space, no write permission, quota, read-only mount). The write error may surface only in the server log while the console keeps showing progress | Check the target and free space first: `GET /api/v1/system-info` reports `models_dir` and `model_storage.free_bytes`. If a pull stalls, read the recent lines of the server log (typically `lemonade-server.log` in the OS temp dir) for the real error (e.g. a download/write failure like `CURL code 23`, or an out-of-space message), then point the download at a writable disk with room. |
| Image generation is slow on CPU (~4–5 min) | sd-cpp on CPU backend | Install the GPU backend on supported AMD hardware: `lemonade backends install sd-cpp:rocm`. |
| `POST /v1/audio/transcriptions` returns 400 unsupported format | Input is not 16 kHz mono WAV | Re-encode with `ffmpeg -i in.* -ar 16000 -ac 1 out.wav`. |
| `POST /v1/audio/speech` returns 404 | TTS model not downloaded | `lemonade pull kokoro-v1`. |
| 401 Unauthorized on every request | User has set `LEMONADE_API_KEY` | Add `Authorization: Bearer $LEMONADE_API_KEY` to every request and to the rule block. |

## Verification checklist

Mark this skill complete only when **all** of the following are true:

- [ ] `lemonade status --json` reports the server running on port 13305.
- [ ] The workspace `AGENTS.md` contains the
      `amd-skills:local-ai-use` block. This is required even when Lemonade was
      already installed and running — generating an image alone does not
      complete the skill.
- [ ] On a follow-up turn, asking the agent to "generate an image of X"
      causes it to POST to `http://localhost:13305/api/v1/images/generations`
      (pulling the model on first use) rather than calling a cloud tool.

If any box is unchecked, the user is still paying cloud cost for at least
one modality.

---

## Reference

For the full model picker, alternate-quality options, the complete endpoint
reference, the API-key flow, and the OmniRouter tool definitions you can
hand to an agent's tool-calling loop, see [reference.md](reference.md).
