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

The skill does two things:

1. **Verifies that local Lemonade is reachable.**
2. **Drops a `Local AI Use` block into the workspace `AGENTS.md`** so the agent
   reads the routing rule on every later turn, in Cursor, Claude Code, Codex,
   Gemini CLI, and any other agent that respects `AGENTS.md`.

Models are **not** downloaded during setup. Each default model is pulled
lazily, on first use, by the routing rule (e.g. the first image request pulls
the image model). This keeps setup fast and avoids gigabytes of downloads the
user may never need.

## When to use this skill

Use this skill when **all** of the following are true:

- The user has, or is willing to install, the system-wide Lemonade Server.
- The user accepts the default Lemonade endpoint `http://localhost:13305`.
- The user wants the change to be **persistent** across future turns and
  agent restarts (the rule is written to disk).

If the user is instead **embedding** Lemonade as a private subprocess inside
an app installer, do not use this skill; use `local-ai-app-integration`
instead.

## Prerequisites

- **OS:** Windows 11 x64, Ubuntu/Debian x64, or macOS (beta).
- **Lemonade Server CLI on `PATH`:** verify with `lemonade --version`. If
  missing, install from <https://lemonade-server.ai/install_options.html>
  before continuing. Do not silently install on the user's machine; that is a
  system-wide change and must be the user's call.
- **Disk:** ~8 GB free for the three default models (SD-Turbo + Whisper-Tiny
  + kokoro-v1).
- **Network:** required for the first `lemonade pull` of each model. After
  that, every modality runs offline.

## The opinionated path

Run this checklist top to bottom. Track progress against it; do not move on
until each step verifies.

```
[ ] 1. Confirm Lemonade Server is installed and reachable
[ ] 2. Install the routing rule into the workspace AGENTS.md
```

The single command that does both steps in one shot is:

```bash
python scripts/setup_local_ai.py
```

The script is idempotent: re-running it on a
fully configured workspace is a no-op apart from a healthcheck. Read the
sections below for what to do when each step fails.

---

## Step 1: confirm Lemonade Server is reachable

Run:

```bash
lemonade status --json
```

Two acceptable outcomes:

| `lemonade status` says | Action |
|---|---|
| `Server is running on port 13305` | Continue to Step 2. |
| `Server is not running` | Start it. On Windows, launch the **Lemonade** Start Menu shortcut. On Linux, run `sudo systemctl start lemonade-server`. Re-check `lemonade status`. |

If `lemonade` is not on `PATH` at all, the server is not installed. Stop and
point the user at <https://lemonade-server.ai/install_options.html>. Do not
attempt a silent install.

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
| `lemonade: command not found` | Server CLI not installed | Install from <https://lemonade-server.ai/install_options.html>; restart shell. |
| `Server is not running` | Service stopped after install | Windows: launch the **Lemonade** Start Menu shortcut. Linux: `sudo systemctl start lemonade-server`. |
| `POST /v1/images/generations` returns 404 model not found | Image model not downloaded | `lemonade pull SD-Turbo` and retry. |
| Image generation is slow on CPU (~4–5 min) | sd-cpp on CPU backend | Install the GPU backend on supported AMD hardware: `lemonade backends install sd-cpp:rocm`. |
| `POST /v1/audio/transcriptions` returns 400 unsupported format | Input is not 16 kHz mono WAV | Re-encode with `ffmpeg -i in.* -ar 16000 -ac 1 out.wav`. |
| `POST /v1/audio/speech` returns 404 | TTS model not downloaded | `lemonade pull kokoro-v1`. |
| 401 Unauthorized on every request | User has set `LEMONADE_API_KEY` | Add `Authorization: Bearer $LEMONADE_API_KEY` to every request and to the rule block. |

## Verification checklist

Mark this skill complete only when **all** of the following are true:

- [ ] `lemonade status --json` reports the server running on port 13305.
- [ ] The workspace `AGENTS.md` contains the
      `amd-skills:local-ai-use` block.
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
