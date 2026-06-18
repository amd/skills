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

**What you'll end up with:** one new launcher module (~30 lines), one config
change to the existing HTTP client (base URL + API key), one vendored binary
under `vendor/lemonade/`. Typical integration: 1–2 hours on a new codebase.

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
[ ] 3. Place Embeddable Lemonade in the app's tree (full package, not just the binary)
[ ] 4. Add a `lemond` launcher (subprocess + API key + port)
[ ] 5. Re-point the existing client at lemond (set HTTP timeout to 120s)
[ ] 6. Wait for /api/v1/health — do not pre-load; surface first-run latency to user
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
4. **Any API-key gating** that blocks the app before a key is entered
   (onboarding walls, validators that reject empty keys, startup checks that
   disable AI until a key exists). Note each one — Step 5 bypasses them in
   local mode.

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
| CPU Speech-to-text | `Whisper-Large-v3-Turbo` | `whispercpp` | Best quality/speed |
| NPU speech-to-text | `whisper-v3-turbo-FLM` | `flm` | XDNA2 NPU on Windows |
| Text-to-speech | `kokoro-v1` | `kokoro` | CPU-only, low latency |
| Image generation | `SDXL-Turbo` | `sd-cpp` | Single-step generation |

For the LLM backend, default to `llamacpp` and let `lemond` pick
`rocm` → `vulkan` → `cpu` automatically by leaving `llamacpp_backend`
unset. Override only if the app has hard hardware requirements.

For more options and tradeoffs, see [reference.md](reference.md).

## Step 3: Place Embeddable Lemonade in the app's tree and install backends

**Get the embeddable artifact** from the latest Lemonade release:

```
https://github.com/lemonade-sdk/lemonade/releases/latest
```

Download the file matching your target OS:

- Windows: `lemonade-embeddable-{VERSION}-windows-x64.zip`
- Linux:   `lemonade-embeddable-{VERSION}-ubuntu-x64.tar.gz`

**First, create the target directory** — it does not exist in a fresh repo:

```powershell
# Windows
New-Item -ItemType Directory -Force vendor\lemonade
```

```bash
# Linux
mkdir -p vendor/lemonade
```

Then download and unpack on Windows (PowerShell):

```powershell
$ver = (Invoke-RestMethod https://api.github.com/repos/lemonade-sdk/lemonade/releases/latest).tag_name
Invoke-WebRequest "https://github.com/lemonade-sdk/lemonade/releases/download/$ver/lemonade-embeddable-$ver-windows-x64.zip" -OutFile lemond.zip
Expand-Archive lemond.zip -DestinationPath "$env:TEMP\lemond-unpack"
Copy-Item -Recurse "$env:TEMP\lemond-unpack\lemonade-embeddable-$ver-windows-x64\*" vendor\lemonade\
```

On Linux (bash):

```bash
VER=$(curl -s https://api.github.com/repos/lemonade-sdk/lemonade/releases/latest | grep tag_name | cut -d'"' -f4)
curl -L "https://github.com/lemonade-sdk/lemonade/releases/download/$VER/lemonade-embeddable-$VER-ubuntu-x64.tar.gz" | tar -xz --strip-components=1 -C vendor/lemonade
```

> **Copy the full package, not just the binary.** The archive contains
> `lemond[.exe]`, `lemonade[.exe]`, `LICENSE`, and `resources/`. The
> `resources/` directory is required — without it lemond starts and passes the
> health check but fails on every model and backend request. Copying only the
> binary produces a server that looks healthy but cannot function.

> **`lemond` vs `lemonade` CLI:** `lemond` is the embedded server binary that
> ships with the app. The `lemonade` CLI is a separate packaging tool used
> only during development/build time to install backends. Install it once on
> the developer machine with `pip install lemonade-sdk`.

The expected layout after unpacking and customization:

```
vendor/lemonade/
  lemond[.exe]                     # the only binary the app ships
  LICENSE
  config.json                      # generated on first run; commit a seed copy
  resources/
    server_models.json             # do not edit; use GET /api/v1/models at runtime
    backend_versions.json
  bin/                             # backends bundled at packaging time
    llamacpp/vulkan/llama-server[.exe]
  models/                          # pre-bundled model weights (optional)
    models--unsloth--Qwen3-4B-GGUF/
```

> **`server_models.json`:** Do not edit or rely on this file. It can be stale.
> The only authoritative model list is `GET /api/v1/models` on a running
> `lemond` instance with the backend already installed.

**Bundle decisions: pick deliberately**

- **Backends:** Bundle `llamacpp:vulkan` at packaging time (works on every
  GPU). Install `llamacpp:rocm` at first run on supported AMD systems via
  `POST /api/v1/install` after probing `GET /api/v1/system-info`. Never ship
  every backend, or the artifact balloons.
- **Models:** Either bundle the default model under `models/` (offline
  install, larger installer) **or** pull on first run with
  `POST /api/v1/pull` (smaller installer, needs network). Pick one and
  document it.
- **`models_dir`:** Set to `./models` in `config.json` to keep weights
  private to the app. Leave as `auto` only if the user explicitly wants to
  share weights with other apps.

**Backend install timing — two distinct paths:**

> **Packaging time** (developer machine, before bundling):
> ```
> lemonade backends install llamacpp:vulkan
> lemonade backends install flm:npu    # Windows NPU path only
> ```
> This bakes the backend binaries into `vendor/lemonade/bin/` before the app
> ships. `lemond` does not need to be running.
>
> **First-run / runtime** (user's machine, after `lemond` is running):
> ```http
> POST /api/v1/install
> {"recipe": "llamacpp", "backend": "rocm"}
> ```
> Use this for hardware-specific backends (e.g. `llamacpp:rocm`) that cannot
> be bundled universally. `lemond` must already be running (Step 4 complete).

## Step 4: Add a `lemond` launcher

The launcher is a thin process supervisor. Its only jobs:

1. Generate a fresh random API key per app launch.
2. Pick a free localhost port.
3. Spawn `lemond <dir> --port <port>` with `LEMONADE_API_KEY` set.
4. Expose the chosen `port` and `key` to the rest of the app.

> **Dev-mode file watchers:** If the app runs with a file watcher (Tauri,
> Electron, Next.js, Vite, etc.) that watches the source tree, ensure
> `vendor/lemonade/` is excluded from the watched paths. Lemond writes config
> and cache files at runtime; a watcher that picks these up will restart the
> app, kill the lemond subprocess, and spawn a new one on a new port —
> silently breaking any in-flight transcription. Add `vendor/` (or the
> equivalent) to the watcher's ignore list before testing.

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

def start_lemond(retries: int = 3) -> tuple[subprocess.Popen, str, int]:
    # _free_port releases the socket before lemond binds — another process
    # can grab the port in that window. Retry with a fresh port on failure.
    last_err: Exception | None = None
    for _ in range(retries):
        port = _free_port()
        key = secrets.token_urlsafe(32)
        env = {**os.environ, "LEMONADE_API_KEY": key}
        proc = subprocess.Popen(
            [str(LEMOND_BIN), str(LEMOND_DIR), "--port", str(port)],
            env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_health(port, key, timeout_s=30)
            return proc, key, port
        except RuntimeError as e:
            proc.kill()
            proc.wait()
            last_err = e
    raise RuntimeError(f"lemond failed to start after {retries} attempts") from last_err

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
    raise RuntimeError(f"lemond on port {port} did not become healthy within {timeout_s}s")
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

// freePort releases the socket before lemond binds — retry with a fresh port on failure.
export async function startLemond(retries = 3) {
  let lastErr;
  for (let i = 0; i < retries; i++) {
    const port = await freePort();
    const key = randomBytes(32).toString("base64url");
    const proc = spawn(LEMOND_BIN, [LEMOND_DIR, "--port", String(port)], {
      env: { ...process.env, LEMONADE_API_KEY: key },
      stdio: ["ignore", "pipe", "pipe"],
    });
    try {
      await waitForHealth(port, key, 30_000);
      return { proc, key, port };
    } catch (e) {
      proc.kill();
      lastErr = e;
    }
  }
  throw new Error(`lemond failed to start after ${retries} attempts: ${lastErr?.message}`);
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
  throw new Error(`lemond on port ${port} did not become healthy within ${timeoutMs}ms`);
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

**Bypass the app's API-key gate in local mode.** A local backend needs no
cloud key, so any onboarding wall, validator, or startup check that demands
one must not block local-mode users. Skip or auto-satisfy the key-entry
screen, treat local mode as already-authorized in validation logic, and
re-enable the gate only for cloud mode. The `lemond` key from Step 4 is set
internally by the launcher, so the user never enters one and any UI
placeholder (e.g. `"local"`) is fine. Flipping into local mode should never
strand the user on a key-entry wall.

**Set the HTTP client timeout to at least 120 seconds.** The default timeout
on most HTTP clients (30s) is shorter than the time lemond takes to load a
model on first use. A silent timeout looks identical to a broken integration
— the request fires, nothing comes back, and the UI shows nothing. 120s
covers first-run model load on any supported hardware.

**Python (openai) example:**

```python
from openai import OpenAI
import httpx

proc, key, port = start_lemond()
client = OpenAI(
    base_url=f"http://127.0.0.1:{port}/api/v1",
    api_key=key,
    http_client=httpx.Client(timeout=120.0),  # covers first-run model load
)
resp = client.chat.completions.create(
    model="Qwen3-4B-GGUF",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Step 6: Wait for health — do not pre-load

Once `GET /api/v1/health` returns 200, the integration is ready. **Do not
call `POST /api/v1/load` at startup.** Lemond lazy-loads models on the first
inference request and handles this correctly on its own. Pre-loading is
unreliable across lemond versions (request body shape has changed between
releases) and a malformed `/load` call can crash or destabilise the server
before the user takes any action.

**First-run latency is expected and must be surfaced to the user.** On the
very first inference after a cold start, lemond loads the model into memory.
This takes 10–30 seconds depending on model size and hardware. An app that
makes no attempt to communicate this will look broken.

Minimum: show a loading indicator or status message ("Starting local AI…")
from the moment the user triggers inference until the first response arrives.
The simplest implementation is a flag that is set when the first request is
sent and cleared when the first response arrives.

## Step 7: Lifecycle and recovery

These are the only failure modes worth handling. Do not over-engineer.

| Symptom | Cause | Recovery |
|---|---|---|
| `POST /api/v1/load` returns 404 / model not found | Model not pulled yet | `POST /api/v1/pull` with `{"model": "..."}` then retry `/api/v1/load` |
| `POST /api/v1/load` returns 500 with backend error | Backend not installed for this hardware | `GET /api/v1/system-info`, pick a supported backend, `POST /api/v1/install` with `{"recipe": "...", "backend": "..."}`, retry |
| Subprocess exits immediately | Port race: another process grabbed the port between `freePort()` and lemond binding | The reference launcher retries with a fresh port automatically (3 attempts) |
| `/api/v1/health` never returns 200 | First-run backend extraction is slow on cold disk | Extend timeout to 90s on first launch, 30s after |
| HTTP 401 on every request | Forgot the `Authorization: Bearer` header | Audit the client config because Lemonade rejects unauth'd calls when `LEMONADE_API_KEY` is set |

**Shutdown:** On app exit, `proc.terminate()` (Unix) or
`proc.kill()` (Windows). `lemond` flushes config and exits cleanly within a
couple of seconds. Always wait on the process; never orphan it.

**Do not** parse `lemond` stdout to detect readiness; use the HTTP
`/v1/health` probe. Stdout format is not a stable contract.

---

## Verification checklist

The integration is done when **all** of these are true:

- [ ] `vendor/lemonade/` contains the full package: `lemond[.exe]`,
      `lemonade[.exe]`, `LICENSE`, and `resources/` — not just the binary.
- [ ] `lemond` starts as a subprocess with a fresh API key per launch.
- [ ] `GET /api/v1/health` returns 200 within the timeout.
- [ ] The existing client's chat / image / speech call returns a valid
      response with the base URL and key swapped, with no other code changed.
- [ ] First-run latency is surfaced: the UI shows a loading state from the
      moment the first inference request is sent until the response arrives.
- [ ] The HTTP client timeout is set to at least 120 seconds.
- [ ] In local mode the app's API-key gate is bypassed: no onboarding wall,
      validator, or startup check blocks the user for lacking a cloud key.
- [ ] If the app uses a dev-mode file watcher, `vendor/lemonade/` is excluded
      from the watched paths so runtime writes by lemond do not trigger restarts.
- [ ] Killing the parent process leaves no `lemond` subprocess behind.
- [ ] On a fresh machine without the optimal backend, the app still works
      via the Vulkan fallback bundled in `bin/`.

If any box is unchecked, do not declare the task complete.

---

## Reference

For detailed model catalog, backend selection matrix, full endpoint reference,
config keys, and per-model `recipe_options.json` tuning, see
[reference.md](reference.md).
