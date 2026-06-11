---
name: local-ai-app-integration
description: >-
  Integrates local AI capabilities into applications using Embeddable Lemonade.
  Use when the user wants to add local AI, offline AI, private AI, on-device AI,
  a local LLM, local chat, embeddings, image generation, speech-to-text, or
  text-to-speech to an app; replace or supplement OpenAI, Anthropic, Ollama, or
  other cloud AI APIs with a local backend; bundle AI inference into an app
  installer; or mentions Lemonade, `lemond`, embeddable lemonade, Ryzen AI,
  NPU/iGPU/dGPU inference, or auto-optimizing local AI.
---

# Local AI App Integration (Embeddable Lemonade)

Add a local AI mode to an existing app that already talks to a cloud AI API
(OpenAI, Anthropic, or Ollama-compatible). The app launches `lemond`, the
Embeddable Lemonade binary, as a private subprocess and the existing client
talks to it on `http://localhost:PORT/api/v1`. The user gets local, private,
hardware-optimized inference (CPU, AMD iGPU/dGPU, XDNA2 NPU) with no separate
install.

## When this skill is the right tool

Use this skill when **all** of the following are true:

- The app already calls a cloud AI service over HTTP (OpenAI Chat Completions,
  Anthropic Messages, or Ollama).
- The user wants that AI to run on the end-user's PC, with the AI engine
  bundled into the app, not as a separate user install.
- The target platform is Windows x64 or Linux x64 (macOS embeddable is in beta).

If the user instead wants a **system-wide** Lemonade Server (one install,
shared across apps), do not use this skill; point them at
`https://lemonade-server.ai/install_options.html` and the standard OpenAI base
URL `http://localhost:13305/api/v1`.

## The opinionated path

This skill follows one fixed sequence. Do not deviate without a stated reason.

```
[ ] 1. Survey the app's current AI integration
[ ] 2. Pick a model + backend profile
[ ] 3. Place Embeddable Lemonade in the app's tree
[ ] 4. Add a `lemond` launcher (subprocess + API key + port)
[ ] 5. Re-point the existing client at lemond
[ ] 6. Wait for /v1/health and pre-load the default model
[ ] 7. Wire shutdown and error recovery
```

Track progress against this checklist. Move on only when each step verifies.

---

## Step 1: Survey the app

Find every place the app currently calls a cloud AI API. Search the repo for:

- `openai`, `OpenAI(`, `chat.completions`, `responses.create`
- `anthropic`, `Anthropic(`, `messages.create`
- `api.openai.com`, `api.anthropic.com`, `localhost:11434` (Ollama)
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

Record three things before continuing:

1. **Client library and language** (e.g., `openai-python`, `openai-node`,
   `@anthropic-ai/sdk`, `go-openai`, raw `fetch`).
2. **Modalities used:** text chat, tool calling, embeddings, image gen,
   transcription, TTS. This drives the model + backend choice in Step 2.
3. **One single place** where the base URL and API key are constructed. If
   there isn't one, refactor to one before going further. Local-mode toggling
   must flip exactly one config object.

## Step 2: Pick a model + backend profile

Choose **one** default profile based on the app's primary modality. Do not
ship a buffet. Ship one good default and document how the user can override
it.

| App's primary need | Default model | Recipe | Why |
|---|---|---|---|
| General chat / assistant | `Qwen3-4B-GGUF` | `llamacpp` | Small, fast, good tool calling, fits 8GB systems |
| Coding assistant | `Qwen2.5-Coder-7B-Instruct-GGUF` | `llamacpp` | Strong code, runs on iGPU |
| Vision / multimodal chat | `Gemma-4-E2B-it-GGUF` | `llamacpp` | Small multimodal default |
| NPU-first on Ryzen AI | `Llama-3.2-3B-Instruct-Hybrid` | `ryzenai-llm` | XDNA2 NPU on Windows |
| Speech-to-text | `Whisper-Large-v3-Turbo` | `whispercpp` | Best quality/speed |
| NPU speech-to-text (Ryzen AI) | `Whisper-Medium` | `flm` | XDNA2 NPU on Windows; `flm` must also load an LLM alongside ASR |
| Text-to-speech | `kokoro-v1` | `kokoro` | CPU-only, low latency |
| Image generation | `SDXL-Turbo` | `sd-cpp` | Single-step generation |

For the LLM backend, default to `llamacpp` and let `lemond` pick
`rocm` → `vulkan` → `cpu` automatically by leaving `llamacpp_backend`
unset. Override only if the app has hard hardware requirements.

For more options and tradeoffs, see [reference.md](reference.md).

## Step 3: Place Embeddable Lemonade in the app's tree

Get the embeddable artifact from the latest Lemonade release:

- Windows: `lemonade-embeddable-{VERSION}-windows-x64.zip`
- Linux: `lemonade-embeddable-{VERSION}-ubuntu-x64.tar.gz`

Unpack into the app source tree at `vendor/lemonade/` (or whatever the app's
existing convention for vendored binaries is). The expected layout after
customization:

```
vendor/lemonade/
  lemond[.exe]                     # the only binary the app ships
  LICENSE
  config.json                      # generated on first run
  resources/
    server_models.json             # trim to just the models you ship
    backend_versions.json
  bin/                             # backends bundled at packaging time
    llamacpp/vulkan/llama-server[.exe]
  models/                          # pre-bundled model weights (optional)
    models--unsloth--Qwen3-4B-GGUF/
```

**Bundle decisions: pick deliberately**

- **Backends:** Bundle `llamacpp:vulkan` at packaging time (works on every
  GPU). Install `llamacpp:rocm` at first run on supported AMD systems via
  `POST /v1/install` after probing `GET /v1/system-info`. Never ship every
  backend, or the artifact balloons.
- **Models:** Either bundle the default model under `models/` (offline
  install, larger installer) **or** pull on first run with `POST /v1/pull`
  (smaller installer, needs network). Pick one and document it.
- **`models_dir`:** Set to `./models` in `config.json` to keep weights
  private to the app. Leave as `auto` only if the user explicitly wants to
  share weights with other apps.

Strip what you don't ship: delete the `lemonade` CLI and
`resources/defaults.json` from the shipping artifact once `config.json` is
initialized.

## Step 4: Add a `lemond` launcher

The launcher is a thin process supervisor. Its only jobs:

1. Generate a fresh random API key per app launch.
2. Pick a free localhost port.
3. Spawn `lemond <dir> --port <port>` with `LEMONADE_API_KEY` set.
4. Expose the chosen `port` and `key` to the rest of the app.

**Python reference launcher** (adapt to the app's language):

```python
import os, secrets, socket, subprocess, sys, time, urllib.request
from pathlib import Path

LEMOND_DIR = Path(__file__).parent / "vendor" / "lemonade"
LEMOND_BIN = LEMOND_DIR / ("lemond.exe" if sys.platform == "win32" else "lemond")

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def start_lemond() -> tuple[subprocess.Popen, str, int]:
    port = _free_port()
    key = secrets.token_urlsafe(32)
    env = {**os.environ, "LEMONADE_API_KEY": key}
    proc = subprocess.Popen(
        [str(LEMOND_BIN), str(LEMOND_DIR), "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    _wait_for_health(port, key, timeout_s=30)
    return proc, key, port

def _wait_for_health(port: int, key: str, timeout_s: int) -> None:
    url = f"http://127.0.0.1:{port}/api/v1/health"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=1) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("lemond failed to become healthy")
```

**Node.js reference launcher:**

```js
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { createServer } from "node:net";
import path from "node:path";

const LEMOND_DIR = path.join(import.meta.dirname, "vendor", "lemonade");
const LEMOND_BIN = path.join(LEMOND_DIR, process.platform === "win32" ? "lemond.exe" : "lemond");

const freePort = () => new Promise((res) => {
  const s = createServer().listen(0, "127.0.0.1", () => {
    const { port } = s.address(); s.close(() => res(port));
  });
});

export async function startLemond() {
  const port = await freePort();
  const key = randomBytes(32).toString("base64url");
  const proc = spawn(LEMOND_BIN, [LEMOND_DIR, "--port", String(port)], {
    env: { ...process.env, LEMONADE_API_KEY: key },
    stdio: ["ignore", "pipe", "pipe"],
  });
  await waitForHealth(port, key, 30_000);
  return { proc, key, port };
}

async function waitForHealth(port, key, timeoutMs) {
  const url = `http://127.0.0.1:${port}/api/v1/health`;
  const headers = { Authorization: `Bearer ${key}` };
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(url, { headers });
      if (r.ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error("lemond failed to become healthy");
}
```

## Step 5: Re-point the existing client at `lemond`

Change exactly two values in the app's existing client config: the base URL
and the API key. Nothing else.

| Existing client | New `base_url` | New auth |
|---|---|---|
| `openai-python` / `openai-node` | `http://127.0.0.1:{port}/api/v1` | `api_key=key` |
| `@anthropic-ai/sdk` | `http://127.0.0.1:{port}/api/v1` | `apiKey: key` (Lemonade serves the Anthropic API too) |
| Raw `fetch` / `requests` | same as above | `Authorization: Bearer {key}` header |
| Ollama-compatible code | `http://127.0.0.1:{port}/api/v0` | none required, but pass the key anyway |

The model identifier on requests stays a Lemonade model name (e.g.
`Qwen3-4B-GGUF`), not the cloud name.

**Python (openai) example:**

```python
from openai import OpenAI
proc, key, port = start_lemond()
client = OpenAI(
    base_url=f"http://127.0.0.1:{port}/api/v1",
    api_key=key,
)
resp = client.chat.completions.create(
    model="Qwen3-4B-GGUF",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Step 6: Wait for health, then preload the default model

`lemond` lazy-loads models on first inference. To eliminate cold-start
latency on the user's first message, preload right after the health check
passes:

```http
POST /api/v1/load
Authorization: Bearer {key}
Content-Type: application/json

{"model": "Qwen3-4B-GGUF"}
```

If the model isn't downloaded yet, follow the recovery flow in Step 7.

## Step 7: Lifecycle and recovery

These are the only failure modes worth handling. Do not over-engineer.

| Symptom | Cause | Recovery |
|---|---|---|
| `POST /v1/load` returns 404 / model not found | Model not pulled yet | `POST /v1/pull` with `{"model": "..."}` then retry `/v1/load` |
| `/v1/load` returns 500 with backend error | Backend not installed for this hardware | `GET /v1/system-info`, pick a supported backend, `POST /v1/install` with `{"recipe": "...", "backend": "..."}`, retry |
| Subprocess exits immediately | Port already in use by another `lemond` | Pick a new free port and retry once |
| `/v1/health` never returns 200 | First-run backend extraction is slow on cold disk | Extend timeout to 90s on first launch, 30s after |
| HTTP 401 on every request | Forgot the `Authorization: Bearer` header | Audit the client config because Lemonade rejects unauth'd calls when `LEMONADE_API_KEY` is set |

**Shutdown:** On app exit, `proc.terminate()` (Unix) or
`proc.kill()` (Windows). `lemond` flushes config and exits cleanly within a
couple of seconds. Always wait on the process; never orphan it.

**Do not** parse `lemond` stdout to detect readiness; use the HTTP
`/v1/health` probe. Stdout format is not a stable contract.

---

## Verification checklist

The integration is done when **all** of these are true:

- [ ] `lemond` starts as a subprocess with a fresh API key per launch.
- [ ] `GET /api/v1/health` returns 200 within the timeout.
- [ ] The default model loads successfully via `POST /v1/load`.
- [ ] The existing client's chat / image / speech call returns a valid
      response with the base URL and key swapped, with no other code changed.
- [ ] Killing the parent process leaves no `lemond` subprocess behind.
- [ ] On a fresh machine without the optimal backend, the app still works
      via the Vulkan fallback bundled in `bin/`.

If any box is unchecked, do not declare the task complete.

---

## Reference

For detailed model catalog, backend selection matrix, full endpoint reference,
config keys, and per-model `recipe_options.json` tuning, see
[reference.md](reference.md).
