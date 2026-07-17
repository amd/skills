# Local AI App Integration: Reference

Detailed reference material for the `local-ai-app-integration` skill. Read
this only when the main `SKILL.md` flow needs a decision that isn't covered
by the default-path tables.

## Contents

- [Backend selection matrix](#backend-selection-matrix)
- [Model picker by use case](#model-picker-by-use-case)
- [Hardware probing with /v1/system-info](#hardware-probing-with-v1system-info)
- [Endpoint reference](#endpoint-reference)
- [Config keys you may need to set](#config-keys-you-may-need-to-set)
- [Per-model tuning via recipe_options.json](#per-model-tuning-via-recipe_optionsjson)
- [Linux packaging notes](#linux-packaging-notes)

---

## Backend selection matrix

`lemond` supports multiple inference backends per modality. Bundle the
broadest-compatibility one at packaging time and install the
hardware-optimized one at first run after a system probe.

### Text generation (`llamacpp` recipe)

| Backend | Hardware | OS | Bundle strategy |
|---|---|---|---|
| `vulkan` | x86_64 CPU, AMD iGPU/dGPU, most others | Windows, Linux | **Bundle at packaging time.** Universal fallback. |
| `rocm` | gfx1151 (Strix Halo), gfx120X (RDNA4), gfx110X (RDNA3) | Windows, Linux | **Install at first run** if `/api/v1/system-info` shows `state: installable`. Cannot be packaging-time bundled. |
| `cpu` | x86_64 CPU | Windows, Linux | Install only if you need a non-Vulkan CPU path. |
| `metal` | Apple Silicon | macOS (beta) | macOS-only path. |

### Text generation (NPU recipes, Windows only)

| Recipe | Backend | Hardware | Notes |
|---|---|---|---|
| `flm` | `npu` | XDNA2 NPU | Cannot be packaging-time bundled on Linux. |
| `ryzenai-llm` | `npu` | XDNA2 NPU | Windows only. Best for the Hybrid model family. |

### Speech-to-text

| Recipe | Backend | Model | Hardware | OS |
|---|---|---|---|---|
| `whispercpp` | `vulkan` | `Whisper-Large-v3-Turbo` | AMD iGPU / dGPU | Windows, Linux |
| `whispercpp` | `cpu` | `Whisper-Large-v3-Turbo` | x86_64 CPU | Windows, Linux |
| `whispercpp` | `npu` | `Whisper-Large-v3-Turbo` | XDNA2 NPU | Windows |
| `flm` | `npu` | `whisper-v3-turbo-FLM` | XDNA2 NPU | Linux (runtime-install only) |

### Text-to-speech

| Recipe | Backend | Hardware |
|---|---|---|
| `kokoro` | `cpu` | x86_64 CPU |

### Image generation (`sd-cpp`)

| Backend | Hardware | OS |
|---|---|---|
| `rocm` | Supported AMD ROCm iGPU/dGPU | Windows, Linux |
| `cpu` | x86_64 CPU | Windows, Linux |

---

## Model picker by use case

Pick **one** model as the app default. Do not list options to the user;
ship a default and document how to override.

| Use case | Recommended model | Approx size | Recipe |
|---|---|---|---|
| Smallest viable chat | `Qwen3-0.6B-GGUF` | 0.5 GB | `llamacpp` |
| General chat (default) | `Qwen3-4B-GGUF` | 2.5 GB | `llamacpp` |
| Tool calling / agents | `Qwen3-4B-GGUF` or `OmniCoder-9B-GGUF` | 2.5 / 5.7 GB | `llamacpp` |
| Coding | `Qwen2.5-Coder-7B-Instruct-GGUF` | 4.5 GB | `llamacpp` |
| Multimodal (vision) chat | `Gemma-4-E2B-it-GGUF` | 2.0 GB | `llamacpp` |
| Hybrid NPU chat (Ryzen AI) | `Llama-3.2-3B-Instruct-Hybrid` | 2.0 GB | `ryzenai-llm` |
| Speech-to-text | `Whisper-Large-v3-Turbo` | 1.6 GB | `whispercpp` |
| NPU speech-to-text (Ryzen AI) | `whisper-v3-turbo-FLM` | 0.6 GB | `flm` |
| Text-to-speech | `kokoro-v1` | 0.3 GB | `kokoro` |
| Image generation | `SDXL-Turbo` | 6.9 GB | `sd-cpp` |

For a catalog with more models, fetch `GET /api/v1/models` after starting `lemond`.
This is the **only** trusted source of available models. Never read or trust
`vendor/lemonade/resources/server_models.json` (or any other static file) as a
model catalog; it can be stale or incomplete. A model only appears in
`GET /v1/models` once its backend is installed (see Step 3), so install the
backend first or the list will look empty/incomplete.

**Catalogued ≠ downloaded.** A model listed by `GET /v1/models` is *available
to use*, not necessarily present on disk. It must be **pulled**
(`POST /api/v1/pull {"model":"..."}`) before it can serve — until then,
inference returns an empty result with HTTP 200, not an error. The surest
signal that a model is ready is a successful pull, not its presence in the
catalog. See SKILL.md
[Step 6](SKILL.md#step-6-health-backend-then-pull-the-model--before-first-inference).

---

## Hardware probing with /v1/system-info

Call this **once at app first-run**, cache the result, and use it to decide
which optional backend to install.

```http
GET /api/v1/system-info
Authorization: Bearer {key}
```

Response shape (truncated):

```json
{
  "recipes": {
    "llamacpp": {
      "backends": {
        "rocm":   { "devices": ["amd_igpu"], "state": "installable" },
        "vulkan": { "devices": ["amd_igpu", "cpu"], "state": "installed" },
        "cpu":    { "devices": ["cpu"], "state": "installed" }
      }
    },
    "ryzenai-llm": {
      "backends": { "npu": { "devices": ["xdna2"], "state": "installable" } }
    }
  }
}
```

The same pattern applies to **every** recipe: read the per-backend `state`,
install the best one that is `installable`, use it if already `installed`, and
fall back down the priority list otherwise. Apply it to whichever recipe matches
the app's modality.

Decision rules in priority order, for the default `llamacpp` recipe (text gen):

1. If `recipes.llamacpp.backends.rocm.state == "installable"` →
   `POST /api/v1/install {"recipe":"llamacpp","backend":"rocm"}`.
2. Else if `state == "installed"` for `vulkan` → use it as-is.
3. Else fall back to `cpu`.

Decision rules for the `whispercpp` recipe (speech-to-text), NPU-first:

1. If `recipes.whispercpp.backends.npu.state == "installed"` → use NPU as-is.
2. Else if `npu.state == "installable"` →
   `POST /v1/install {"recipe":"whispercpp","backend":"npu"}`, then use NPU.
3. Else if `vulkan` is `installed`/`installable` → use the iGPU/dGPU path.
4. Else fall back to `cpu`.

Probe **once**, cache the chosen backend for the session (the result does not
change while the app runs), and log which backend was selected. This is the
mechanism that lets one build run on an NPU machine and a CPU-only machine
without any user configuration.

For Ryzen AI Hybrid models on Windows, additionally check
`ryzenai-llm.backends.npu.state` and install if `installable`.

---

## Endpoint reference

All endpoints require `Authorization: Bearer {key}` when
`LEMONADE_API_KEY` is set (it always should be in an embedded deployment).

### App-facing (use these from the app's existing client)

| Endpoint | Purpose |
|---|---|
| `GET  /api/v1/health` | Readiness probe and loaded-model list |
| `GET  /api/v1/models` | List available models |
| `POST /api/v1/chat/completions` | OpenAI Chat Completions (text + vision + tool calls) |
| `POST /api/v1/embeddings` | OpenAI Embeddings |
| `POST /api/v1/audio/transcriptions` | OpenAI Whisper-style transcription |
| `POST /api/v1/audio/speech` | OpenAI TTS |
| `POST /api/v1/images/generations` | OpenAI image generation |
| `POST /api/v1/messages` | Anthropic Messages API |

### Lifecycle (use these from the launcher / supervisor)

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/pull` | Download a model |
| `POST /api/v1/load` | Load a model into memory |
| `POST /api/v1/unload` | Free a model |
| `POST /api/v1/delete` | Remove a downloaded model |
| `POST /api/v1/install` | Install a backend (`{"recipe","backend"}`) |
| `POST /api/v1/uninstall` | Remove a backend |
| `GET  /api/v1/system-info` | Probe supported backends and devices |

### Internal config (use sparingly)

| Endpoint | Purpose |
|---|---|
| `GET  /internal/config` | Full runtime config snapshot |
| `POST /internal/set` | Update one or more config keys atomically |

---

## Config keys you may need to set

Set these via the `lemonade` CLI's `config set` at packaging time, by
hand-editing `config.json`, or at runtime via `POST /internal/set`.

### Server-level (immediate effect)

| Key | Type | Notes |
|---|---|---|
| `port` | int | Bind port. Override at launch with `--port` instead. |
| `host` | string | Default `127.0.0.1`. **Do not** expose on `0.0.0.0` from an embedded app. |
| `log_level` | enum | `trace`/`debug`/`info`/`warning`/`error`/`fatal`/`none` |
| `global_timeout` | int seconds | HTTP client timeout for backend installs and pulls |
| `no_broadcast` | bool | **Set `true` for embedded apps**, disables UDP discovery beacon |
| `extra_models_dir` | string | Search path for arbitrary GGUFs (see below) |

### Deferred (apply on next load)

| Key | Type | Notes |
|---|---|---|
| `max_loaded_models` | int (-1 or positive) | Cap concurrent loaded models |
| `ctx_size` | int | LLM context window |
| `llamacpp_backend` | string | Pin to `rocm` / `vulkan` / `cpu` / `metal`; leave unset for auto |
| `llamacpp_args` | string | Raw args appended to `llama-server` |
| `sdcpp_backend` | string | `rocm` / `cpu` |
| `whispercpp_backend` | string | `npu`/`cpu` (Windows), `cpu`/`vulkan` (Linux). For NPU prefer the `flm` recipe instead |
| `whispercpp_args` | string | Raw whisper.cpp args |
| `flm_args` | string | Raw FastFlowLM args |
| `steps` | int | SD step count |
| `cfg_scale` | number | SD CFG scale |
| `width`, `height` | int | SD output size |

### Recommended embedded defaults

```json
{
  "host": "127.0.0.1",
  "no_broadcast": true,
  "log_level": "warning",
  "models_dir": "./models",
  "max_loaded_models": 2,
  "ctx_size": 8192
}
```

---

## Per-model tuning via recipe_options.json

For per-model overrides (custom `llama-server` args, alternate context size
for one model only, alternate prompt template), drop a `recipe_options.json`
next to `config.json`. Example:

```json
{
  "Qwen3-4B-GGUF": {
    "llamacpp_args": "--threads 8 --batch-size 512",
    "ctx_size": 16384
  }
}
```

This file is consulted on every model load. No restart required.
---

## Linux packaging notes

Two backend limitations on Linux as of this writing:

- `flm` (FastFlowLM, NPU) cannot be bundled at packaging time on Linux.
  Install at runtime only.
- `llamacpp:rocm` cannot be bundled at packaging time on **any** OS. Always
  install at runtime via `/api/v1/install`.

When building from source for an unusual Linux distro, see the upstream
`docs/embeddable/building.md` in the lemonade-sdk/lemonade repo.

---

## Reference launchers

Full implementations for Step 4. Adapt to the app's language; the key
constraints are: retry with a fresh port on spawn failure (the socket is
released before lemond binds), poll `/api/v1/health` with the Bearer key,
and kill the process on app exit.

**Python:**

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

**Node.js:**

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
