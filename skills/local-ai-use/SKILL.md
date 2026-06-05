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

This is a **meta-skill**. After you run it, every later request that needs
image generation, text-to-speech, or speech-to-text uses the local
[Lemonade Server](https://lemonade-server.ai) instead of a cloud API. The
agent's own LLM keeps handling text; only the expensive multimodal calls move
on-device.

The skill covers **two independent modality groups** — set up only the one(s)
the user actually needs:

| Group | Covers | Default model(s) |
|---|---|---|
| **image** | image generation + editing | `SD-Turbo` |
| **speech** | text-to-speech **and** speech-to-text | `kokoro-v1`, `Whisper-Tiny` |

> **Do ONLY the group the user asked for.** If the user wants to generate an
> image, set up `image` only — do **not** pull the speech models. The setup
> command takes the group as an argument (`image`, `speech`, or `all`), and the
> rule installed into `AGENTS.md` contains only the group(s) you set up.

For each group you set up, the skill does two things:

1. **Verifies that local Lemonade is reachable and has the right models.**
2. **Drops a `Local AI Use` block into the workspace `AGENTS.md`** so the agent
   reads the routing rule on every later turn, in Cursor, Claude Code, Codex,
   Gemini CLI, and any other agent that respects `AGENTS.md`.

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
- **Disk:** ~5 GB for `image` (SD-Turbo); ~0.4 GB for `speech`
  (kokoro-v1 + Whisper-Tiny). Only the group(s) you set up are downloaded.
- **Network:** required for the first `lemonade pull` of each model. After
  that, every modality runs offline.

## The opinionated path

Run this checklist top to bottom for the group(s) the user needs. Track progress
against it; do not move on until each step verifies.

```
[ ] 1. Confirm Lemonade Server is installed and reachable
[ ] 2. Pull the selected group's default models
[ ] 3. Install the routing rule into the workspace AGENTS.md
[ ] 4. Smoke-test the selected group's endpoints
```

The single command that does steps 1, 2, and 3 in one shot, scoped to a group:

```bash
python scripts/setup_local_ai.py image     # image only
python scripts/setup_local_ai.py speech    # TTS + STT only
python scripts/setup_local_ai.py all       # both (only if the user wants both)
```

(Run from this skill's folder.) The script pulls only the selected group's
models and writes only that group's rule section. It is idempotent: re-running
with the same group is a no-op apart from a healthcheck. To add a group later,
re-run with the full set you want (e.g. `all`). Read the sections below for what
to do when each step fails.

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

## Step 2: pull the selected group's default models

Pull only the models for the group(s) you are setting up. They are the
**Lite Collection** defaults from Lemonade OmniRouter, sized to keep
token-and-cost savings real on commodity hardware:

| Group | Modality | Model | Size | Why this default |
|---|---|---|---|---|
| `image` | Image generation | `SD-Turbo` | ~5 GB | Single-step generation, runs on CPU and AMD iGPU/dGPU |
| `speech` | Text-to-speech | `kokoro-v1` | ~0.3 GB | Only TTS model Lemonade currently supports; CPU-only, low latency |
| `speech` | Speech-to-text | `Whisper-Tiny` | ~0.1 GB | Smallest Whisper; fast on CPU. Upgrade to `Whisper-Large-v3-Turbo` if accuracy matters more than latency. |

```bash
# image group
lemonade pull SD-Turbo
# speech group
lemonade pull kokoro-v1
lemonade pull Whisper-Tiny
```

To choose a different model while installing the rule, pass it to the setup
script alongside the group. For example, to make future image requests use SDXL:

```bash
python scripts/setup_local_ai.py image --image-model SDXL-Turbo
```

The script will pull the selected model and write that model ID into the
installed `AGENTS.md` rule. The same pattern works for `--tts-model` and
`--stt-model` with the `speech` group.

Each `pull` is idempotent. To verify what is already downloaded:

```bash
lemonade list --downloaded
```

For coverage of larger / higher-quality alternatives (`SDXL-Turbo`,
`Flux-2-Klein-4B`, `Whisper-Large-v3-Turbo`), see the
[model picker in reference.md](reference.md#model-picker).

## Step 3: install the routing rule into AGENTS.md

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

## Step 4: smoke-test the group(s) you set up

Verify each modality you set up against the live server before declaring
success — run only the tests for the group(s) you installed. These mirror the
inline patterns in the installed rule, so a green pass here means the rule will
work. If you installed with a model override such as `--image-model SDXL-Turbo`,
use that model ID in the smoke test and confirm the installed `AGENTS.md` rule
contains it.

**Image generation** — `image` group (writes `out.png`):

```bash
curl -sX POST http://localhost:13305/api/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"SD-Turbo","prompt":"a single red apple on a white table","size":"512x512","steps":4,"response_format":"b64_json"}' \
  | python -c "import sys,json,base64; open('out.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"
```

**Text-to-speech** — `speech` group (writes `out.mp3`):

```bash
curl -sX POST http://localhost:13305/api/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro-v1","input":"Local AI is now active.","response_format":"mp3"}' \
  -o out.mp3
```

**Speech-to-text** — `speech` group (round-trips `out.mp3` → text via a wav re-encode):

```bash
ffmpeg -y -i out.mp3 -ar 16000 -ac 1 out.wav
curl -sX POST http://localhost:13305/api/v1/audio/transcriptions \
  -F "file=@out.wav" -F "model=Whisper-Tiny"
```

If a test for a group you set up returns a non-2xx status, fix it now. The rule
we just installed sends future requests to these same endpoints, so a broken
endpoint becomes a broken user experience.

---

## What changes after this skill runs

From the next turn onward, the agent reads the rule in `AGENTS.md` on every
message. For each group you set up, the rule explicitly tells the agent:

- **image group — image generation:** call `POST /api/v1/images/generations`
  (or `/images/edits`) on the local server. Do **not** call any cloud image API
  and do **not** use the built-in `GenerateImage` tool (that path bills tokens
  to the cloud provider).
- **speech group — text-to-speech:** call `POST /api/v1/audio/speech`. Do
  **not** call cloud TTS providers (OpenAI TTS, ElevenLabs, etc.).
- **speech group — speech-to-text:** call `POST /api/v1/audio/transcriptions`.
  Do **not** call cloud transcription providers.
- **Fallback (any group):** only fall back to a cloud API after one local
  attempt has failed *and* the user has been told the local call failed. Never
  silently fall back; the whole point of this skill is to keep cost predictable.

A group you did **not** set up is untouched — the agent keeps using its
configured providers for that modality.

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

Mark a group complete only when **all** of the following are true for it:

- [ ] `lemonade status --json` reports the server running on port 13305.
- [ ] `lemonade list --downloaded` shows the group's model(s): `SD-Turbo` for
      `image`; `kokoro-v1` and `Whisper-Tiny` for `speech`.
- [ ] The workspace `AGENTS.md` contains the `amd-skills:local-ai-use` block,
      and that block includes the group's section (`### Image` and/or
      `### Speech`).
- [ ] The group's smoke test(s) in Step 4 succeed.
- [ ] On a follow-up turn, a request for that modality causes the agent to POST
      to the local endpoint rather than calling a cloud tool.

You only need the rows for the group(s) you set up. A group you skipped is
expected to still use cloud providers.

---

## Reference

For the full model picker, alternate-quality options, the complete endpoint
reference, the API-key flow, and the OmniRouter tool definitions you can
hand to an agent's tool-calling loop, see [reference.md](reference.md).
