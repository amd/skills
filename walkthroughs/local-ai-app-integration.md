# AMD Skills Walkthroughs: `local-ai-app-integration`

The goal of this skill is to teach your AI agent to add a **local AI mode** to an
existing app that today only talks to cloud AI APIs.

For this walkthrough we use [`danielholanda/dictate`](https://github.com/danielholanda/dictate),
a Windows dictation app that currently sends every recording to cloud
speech-to-text providers (Groq, Deepgram, Cartesia, Gemini, Mistral, etc.).

## Prerequiresites
This sample app used here requires the Rust toolchain (install from https://rustup.rs/).

## Step 1 - Get the target app

* Clone the cloud-only app you want to upgrade:

```
git clone https://github.com/danielholanda/dictate.git
cd dictate
```

## Step 2 - Understanding which skills are available

* Run `claude "Which skills can you see?" --model sonnet`. You should see a list of skills that should *not* include anything related to local AI app integration.

## Step 3 - Enabling claude to see `local-ai-app-integration`

In the future this will be enabled directly through claude's marketplace. For now, we have to manually add it.

* Clone `https://github.com/amd/skills`
* Move the `local-ai-app-integration` skill from the repo to `.claude/skills/`
* Run `claude "Which skills can you see?" --model sonnet`. You should see a list of skills that includes `local-ai-app-integration`.

## Step 4 - Running the skill

Open Claude inside the `dictate` repo and run the prompt:

```
This app sends my dictation audio to cloud speech-to-text providers.
Add a local AI mode that runs transcription on my machine instead by default.
I want it to run using the NPU. Keep the cloud providers as an option.
```

Claude should:

1. Survey where the app calls its cloud transcription APIs.
2. Pick a local speech-to-text model + backend (e.g. `Whisper-Large-v3-Turbo` on `whispercpp`).
3. Vendor the Embeddable Lemonade (`lemond`) binary into the app tree.
4. Add a launcher that spawns `lemond` with a fresh API key on a free port.
5. Re-point the app's existing client at the local endpoint and wait for `/v1/health`.

## Step 5 - Running the modified app

Dictate is a Tauri (Rust + Node) app. From the repo root:

```
npm install
npm run tauri dev
```
Once the window opens, press `Ctrl+Shift+D`, speak, and confirm transcription now runs through
your local `lemond` instead of a cloud provider.

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
