# AMD Skills Walkthroughs: `local-ai-app-integration`

The goal of this skill is to teach your AI agent to add a **local AI mode** to an
existing app that today only talks to cloud AI APIs.

For this walkthrough we use [`danielholanda/dictate`](https://github.com/danielholanda/dictate),
a Windows dictation app that currently sends every recording to cloud
speech-to-text providers (Groq, Deepgram, Cartesia, Gemini, Mistral, etc.).

**What you'll end up with:** one new launcher module, one config change to
the existing HTTP client, and `lemond` vendored under `vendor/lemonade/`.
Transcription moves from cloud to your local device. Expect 1–2 hours.

## Prerequisites

This sample app requires the Rust toolchain (install from https://rustup.rs/).

**Hardware:** Any Windows x64 PC works. You do **not** need an AMD NPU. The
skill detects what your machine has and runs transcription on the fastest option
available, in this priority order:

| Priority | Your hardware | What you get |
|---|---|---|
| 1 (fastest) | Ryzen AI with XDNA2 NPU (Strix, Strix Halo, Kraken, Gorgon Point) | NPU-accelerated transcription |
| 2 | AMD iGPU / dGPU | GPU-accelerated transcription |
| 3 (fallback) | Any other Windows x64 PC | CPU transcription |

If you have an NPU the skill uses it first; if not, it transparently falls back
to your iGPU and then CPU. You don't choose any of this — the steps below are the
same on every machine.

## Step 1 - Get the target app

* Clone the cloud-only app you want to upgrade:

```
git clone https://github.com/danielholanda/dictate.git
cd dictate
```

## Step 2 - Understanding which skills are available

* Run `claude "Which skills can you see?" --model opus`. You should see a list of skills that should *not* include anything related to local AI app integration.

## Step 3 - Enabling claude to see `local-ai-app-integration`

> **Future:** this will be a one-liner: `/plugin install local-ai-app-integration@amd/skills`.
> Until the marketplace ships, install manually:

```bash
# Clone the AMD skills repo (if you haven't already)
git clone https://github.com/amd/skills path/to/amd-skills

# Copy the skill into Claude's skill directory for the dictate repo
# Run this from inside the dictate repo root
mkdir -p .claude/skills
cp -r path/to/amd-skills/skills/local-ai-app-integration .claude/skills/
```

On Windows (PowerShell):

```powershell
New-Item -ItemType Directory -Force .claude\skills
Copy-Item -Recurse path\to\amd-skills\skills\local-ai-app-integration .claude\skills\
```

* Run `claude "Which skills can you see?" --model opus`. You should see a list of skills that includes `local-ai-app-integration`.

## Step 4 - Running the skill

Run `claude --model opus` inside the `dictate` repo with this prompt:

```
This app sends my dictation audio to cloud speech-to-text providers.
Add a local AI mode that runs transcription on my machine instead by default.
Use the best available local backend — NPU if I have one, otherwise iGPU or CPU.
Keep the cloud providers as an option and minimize code changes.
```

Claude should:

1. Survey where the app calls its cloud transcription APIs.
2. Probe hardware (`GET /api/v1/system-info`) and pick the fastest available
   backend for `Whisper-Large-v3-Turbo`, NPU-first:
   - XDNA2 NPU present → whispercpp NPU backend
   - else AMD iGPU/dGPU → whispercpp iGPU/dGPU backend
   - else → whispercpp CPU backend
3. Vendor the Embeddable Lemonade (`lemond`) binary into the app tree.
4. Add a launcher that spawns `lemond` on a free port with retry logic, logging
   each lifecycle stage (spawn → health → backend install → model pull → result).
5. Re-point the app's existing client at the local endpoint and wait for
   `/api/v1/health`. Because local mode talks to your own machine, it needs **no
   cloud API key** — Claude should bypass the app's key-entry gate in local mode.
6. Install the backend, then **pull the model** (`POST /api/v1/pull`) so its
   weights are on disk before the first recording. Skipping this makes the very
   first transcription come back blank with no error.

Please note this may take several minutes as this app has a fairly large codebase.

## Step 5 - Running the modified app

Dictate is a Tauri (Rust + Node) app. From the repo root:

```
npm install
npm run tauri dev
```

**What the first launch looks like.** Watch the terminal (not the browser
console). On a cold first run you should see the staged log lines as setup
progresses — the model download in particular can take a while:

```
[lemond] Starting on port 56748
[lemond] Healthy on port 56748
[lemond] whispercpp:npu already installed
[lemond] Pulling model Whisper-Large-v3-Turbo...
[lemond] Model Whisper-Large-v3-Turbo ready
[local] Using transcription model: Whisper-Large-v3-Turbo backend: npu
[local] Transcription result: " Hi, can you hear me?"
```

The **first transcription can be slow** because it covers the whole setup chain:
server spawn + backend setup + model download + model load. Subsequent
recordings are fast. Once the window opens, press the microphone button to
speak, and confirm transcription runs through your local device instead of a
cloud provider — the text appears where your cursor was last located.

> **Blank result?** If a recording produces no text and the terminal shows no
> error, the model was not pulled — `[local] Transcription result: ""`. The
> model-pull step (item 6 of "Claude should" above) fixes this; it is the most
> common first-run snag.

> **Repeated phrases** like `" How can you hear me now?\n How can you hear me
> now?\n"` on quiet audio are a known Whisper behavior on silence/low-energy
> input, not an integration bug.

## Step 6 - (Optional) Going beyond

`local-ai-app-integration` works for any modality, not just speech-to-text. The
same pattern adds local chat, embeddings, image generation, or text-to-speech to
any app that already calls into the cloud. You can try using this skill to turn other cloud apps into local apps.

## Step 7 - (Optional) Try to get things done without AMD Skills

Remove the added skill from `.claude/skills/` and rerun the experiment above. This should lead to a high variance in execution length and token usage. Some common issues without the skill include:
* Model produces a local implementation that does not use NPU acceleration as instructed.
* Model inventing a brittle local server setup that does not handle health checks, API keys, or shutdown.
* Model touching many files instead of flipping a single base-URL/key config object.
* Model providing a knowledge article instead of actually integrating local AI into the app.
