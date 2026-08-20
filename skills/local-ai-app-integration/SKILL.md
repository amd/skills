---
name: local-ai-app-integration
description: >-
  Integrates local AI capabilities into applications using Embeddable Lemonade.
  Use when the user wants to add local AI, offline AI, private AI, on-device AI,
  a local LLM, local chat, embeddings, image generation, speech-to-text, or
  text-to-speech to an existing app; replace or supplement OpenAI, Anthropic, Ollama, or
  other cloud AI APIs with a local backend; only use to convert user apps. Do not use when
  the user just wants the agent itself to generate images, transcribe, or speak locally in
  the current workspace, even to cut their own API bill.
---

# Local AI App Integration (Embeddable Lemonade)

Add a local AI mode to an existing app that already talks to a cloud AI API
(OpenAI, Anthropic, or Ollama-compatible). The app launches `lemond`, the
Embeddable Lemonade binary, as a private subprocess and the existing client
talks to it on `http://localhost:PORT/api/v1`. The user gets local, private,
hardware-optimized inference (CPU, AMD iGPU/dGPU, XDNA2 NPU) with no separate
install.

**What you'll end up with:** one new launcher module (~30 lines), three mandatory changes to the existing HTTP client (`base_url`, `api_key`, and a 120-second HTTP timeout), one vendored binary under `vendor/lemonade/`.

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
[ ] 4. Add a `lemond` launcher (subprocess + API key + port + per-stage logging)
[ ] 5. Re-point the existing client at lemond (base_url, api_key, 120s timeout — all three required)
[ ] 6. Wait for /api/v1/health, install backend, then PULL the model before first use
[ ] 7. Wire shutdown and error recovery
```

Track progress against this checklist. Move on only when each step verifies.

> **Log every stage.** A local integration has many silent failure points —
> spawn, health, backend install, model download, first inference. Without a
> log line at each transition, "nothing happened" is indistinguishable from
> "broke at stage 3." Emit one clear line per stage as you build (see
> [Step 4](#step-4-add-a-lemond-launcher)); the most common dead-end in this
> integration — a first request that hangs or comes back blank — is invisible
> without them.

---

## Step 1: Survey the app

Find every place the app currently calls a cloud AI API. Search the repo for:

- `openai`, `OpenAI(`, `chat.completions`, `responses.create`
- `anthropic`, `Anthropic(`, `messages.create`
- `api.openai.com`, `api.anthropic.com`, `localhost:11434` (Ollama)
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

Record four things before continuing:

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
| Speech-to-text (Windows) | `Whisper-Large-v3-Turbo` | `whispercpp` | One model; probe picks NPU → iGPU/dGPU → CPU automatically |
| Speech-to-text (Linux) | `Whisper-Large-v3-Turbo` | `whispercpp` | iGPU (`vulkan`) or CPU. **No NPU path** — see below |
| Text-to-speech | `kokoro-v1` | `kokoro` | CPU-only, low latency |
| Image generation | `SDXL-Turbo` | `sd-cpp` | Single-step generation |

For the LLM backend, default to `llamacpp` and let `lemond` pick
`rocm` → `vulkan` → `cpu` automatically by leaving `llamacpp_backend`
unset. Override only if the app has hard hardware requirements.

**Scope: this skill selects a backend once at integration time on the
developer's machine.** Runtime fallback based on the end user's hardware is
out of scope. Bundle `vulkan` as the universal fallback so the app works on
any machine. If the dev machine has an NPU and the chosen recipe supports it,
the skill will use the NPU backend — otherwise it falls back to `vulkan`.

> **Note:** having an NPU does not mean every recipe supports NPU. Confirm
> the recipe/backend pair is `installed` or `installable` via
> `GET /api/v1/system-info` before committing to it. See
> [reference.md](reference.md#hardware-probing-with-v1system-info) for
> per-recipe decision rules.

> **No NPU backend is installable through Lemonade on Linux** — including
> `flm:npu`, and including hosts whose NPU `system-info` reports as available.
> Plan Linux around `vulkan` (iGPU) or `cpu`. Details and the exact refusals
> are in [reference.md](reference.md#speech-to-text).

For more options and tradeoffs, see [reference.md](reference.md).

## Step 3: Place Embeddable Lemonade in the app's tree and install backends

**Get the embeddable artifact** from the latest Lemonade release:

```
https://github.com/lemonade-sdk/lemonade/releases/latest
```

Download the file matching your target OS:

- Windows: `lemonade-embeddable-{VERSION}-windows-x64.zip`
- Linux:   `lemonade-embeddable-{VERSION}-ubuntu-x64.tar.gz`

> **Don't hand-build the download URL from the tag.** The git tag carries a
> leading `v` (e.g. `v10.8.0`) but the asset filename strips it
> (`lemonade-embeddable-10.8.0-...`), so using the tag verbatim 404s. Ask the
> GitHub API for the asset by its stable name pattern and use the URL it
> returns, as below — this stays correct across version and naming changes.

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
$rel = Invoke-RestMethod https://api.github.com/repos/lemonade-sdk/lemonade/releases/latest
$asset = $rel.assets | Where-Object { $_.name -like "lemonade-embeddable-*-windows-x64.zip" } | Select-Object -First 1
Invoke-WebRequest $asset.browser_download_url -OutFile lemond.zip
Expand-Archive lemond.zip -DestinationPath "$env:TEMP\lemond-unpack"
$folder = $asset.name -replace '\.zip$',''   # unpacked dir = asset name without .zip
Copy-Item -Recurse "$env:TEMP\lemond-unpack\$folder\*" vendor\lemonade\
# Sanity check: resources/ must be nested under vendor\lemonade\ (not flattened)
if (-not (Test-Path vendor\lemonade\resources\*.json)) { throw "resources/ missing — re-extract and copy again" }
```

On Linux (bash):

```bash
URL=$(curl -s https://api.github.com/repos/lemonade-sdk/lemonade/releases/latest \
  | grep browser_download_url | grep ubuntu-x64.tar.gz | cut -d'"' -f4)
curl -L "$URL" | tar -xz --strip-components=1 -C vendor/lemonade
```

> **Copy the full package, not just the binary.** The archive contains
> `lemond[.exe]`, `lemonade[.exe]`, `LICENSE`, and `resources/`. The
> `resources/` directory is required — without it lemond starts and passes the
> health check but fails on every model and backend request. Copying only the
> binary produces a server that looks healthy but cannot function.

> **`lemond` vs `lemonade` CLI:** `lemond` is the embedded server binary that
> ships with the app. The `lemonade` CLI is a separate packaging tool used
> only during development/build time to install backends. The same embeddable
> archive unpacked above already contains a matching `lemonade[.exe]` next to
> `lemond[.exe]`, so its version aligns with the bundled `lemond`. Do **not**
> `pip install lemonade-sdk` to get it: the PyPI package is a separate, older
> release line whose ports, model names, and install API do not match the
> `lemond` bundled here, and mixing the two is a known source of silent
> version mismatches. Keep the `lemonade` CLI, `lemond`, and the backends all
> from the one release downloaded in this step so their versions stay aligned.

The expected layout **after setup** (first run + backend install). A freshly
unzipped package contains only `lemond[.exe]`, `lemonade[.exe]`, `LICENSE`, and
`resources/` — the items below are created later, as their comments note:

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
> `lemond` instance with the backend already installed. That call returns
> **downloaded models only** — add `?show_all=true` for the full catalog.

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

> **Packaging time** (developer machine, before bundling). Use the lemonade
> CLI that shipped inside `vendor/lemonade/` so it matches the bundled
> `lemond` version (prefix with `./` or the full path):
> ```
> vendor/lemonade/lemonade backends install llamacpp:vulkan
> vendor/lemonade/lemonade backends install flm:npu    # Windows NPU path only
> ```
> This bakes the backend binaries into `vendor/lemonade/bin/` before the app
> ships. `lemond` does not need to be running. Use a modern `lemonade` CLI
> whose version matches the bundled `lemond` (the copy in the archive you
> unpacked works); do not `pip install lemonade-sdk` for it.
>
> **First-run / runtime** (user's machine, after `lemond` is running):
> ```http
> POST /api/v1/install
> {"recipe": "llamacpp", "backend": "rocm"}
> ```
> Use this for hardware-specific backends (e.g. `llamacpp:rocm`) that cannot
> be bundled universally. `lemond` must already be running (Step 4 complete).

## Step 4: Add a `lemond` launcher

Write the launcher as a new module named **`lemond_launcher.py`** (or
`lemond_launcher.<ext>` for the app's language). It is a thin process
supervisor. Its only jobs:

1. Generate a fresh random API key: `key = secrets.token_urlsafe(32)`
2. Pick a free localhost port: bind a `socket` to port 0, read back the assigned port, close it.
3. Spawn lemond as a `subprocess`: `subprocess.Popen([LEMOND_BIN, LEMOND_DIR, "--port", str(port)], env={**os.environ, "LEMONADE_API_KEY": key})`
4. Poll `GET /api/v1/health` with `Authorization: Bearer {key}` in a loop until HTTP 200 — this is the only correct readiness check.
5. Expose the chosen `port` and `key` to the rest of the app.

> **Log one line per lifecycle stage.** Build the logging in from the start —
> not as an afterthought when something breaks. Each silent transition needs a
> visible marker so a failure points at the exact stage. Aim for:
>
> ```
> [lemond] Starting on port <port>
> [lemond] Healthy on port <port>
> [lemond] <recipe>:<backend> installed        (or: already installed / install failed)
> [lemond] Pulling model <name>...             then: Model <name> ready  (or: pull returned <status>)
> [local]  <modality> result: <value>          (first inference output verbatim — empty or very slow here = unpulled model)
> ```
>
> Logging the **first inference result verbatim**, with its elapsed time, is
> what turns the unpulled-model failure (Step 6) from a multi-hour mystery into
> a one-line diagnosis. Route these through the app's normal logging so they can be quieted
> for release.

> **Dev-mode file watchers:** If the app runs with a file watcher (Tauri,
> Electron, Next.js, Vite, etc.) that watches the source tree, ensure
> `vendor/lemonade/` is excluded from the watched paths. Lemond writes config
> and cache files at runtime; a watcher that picks these up will restart the
> app, kill the lemond subprocess, and spawn a new one on a new port —
> silently breaking any in-flight transcription. Add `vendor/` (or the
> equivalent) to the watcher's ignore list before testing.

**Use the reference implementation from [reference.md § Reference launchers](reference.md#reference-launchers) directly** — copy it verbatim and adapt only the `LEMOND_DIR` path. Do not write a launcher from scratch. The reference Python launcher uses `secrets` (for the API key), `socket` (for the free-port probe), and `subprocess` (to spawn lemond); the Node.js launcher uses the equivalent stdlib modules. Both handle port-race retries and health polling correctly.

Readiness is always determined by polling the exact endpoint
`GET http://127.0.0.1:<port>/api/v1/health` and checking for HTTP 200 — never
by reading `lemond`'s stdout or stderr. Any health-check helper you write must
hit that `/api/v1/health` path.

## Step 5: Re-point the existing client at `lemond`

Make **three** changes to the app's existing client construction — all three
are required, not optional:

1. Set `base_url` so the SDK's own path suffix resolves correctly — for
   OpenAI-compatible clients that is `http://127.0.0.1:{port}/api/v1`; see the
   table below, the prefix is **not** the same for every client
2. Set `api_key` to the launcher key
3. **Set the HTTP timeout to 120 seconds** — this is mandatory, not optional

The 120-second timeout is not a tuning suggestion. The default on most HTTP
clients is 30s, which is shorter than lemond's first-run model load time on
real hardware. Without it the request silently times out and the UI shows
nothing, which is indistinguishable from a broken integration.

**Python (openai) — the exact change to make:**

```python
import httpx
from openai import OpenAI

proc, key, port = start_lemond()
client = OpenAI(
    base_url=f"http://127.0.0.1:{port}/api/v1",
    api_key=key,
    http_client=httpx.Client(timeout=120),  # required: 120s for first-run model load
)
```

For other clients:

| Existing client | New `base_url` | Resolves to | New auth | Timeout |
|---|---|---|---|---|
| `openai-python` | `http://127.0.0.1:{port}/api/v1` | `/api/v1/chat/completions` | `api_key=key` | `httpx.Client(timeout=120)` |
| `openai-node` | `http://127.0.0.1:{port}/api/v1` | `/api/v1/chat/completions` | `apiKey: key` | `timeout: 120000` |
| `@anthropic-ai/sdk` | `http://127.0.0.1:{port}` — **no path suffix** | `/v1/messages` | `apiKey: key` | `timeout: 120000` |
| Raw `fetch` / `requests` | full path, per the route table | — | `Authorization: Bearer {key}` | set per-request |
| Ollama-compatible code | `http://127.0.0.1:{port}/api/v0` | `/api/v0/...` | pass key anyway | 120s |

> **Path prefixes are not uniform — do not assume `/api/v1` for everything.**
> OpenAI-compatible routes live under `/api/v1`, but Anthropic Messages is
> served at **`/v1/messages`, outside that prefix** (`/api/v1/messages` 404s).
> `@anthropic-ai/sdk` appends `/v1/messages` to whatever `base_url` you give
> it, so `.../api/v1` requests `/api/v1/v1/messages` and 404s on the very first
> call — during the cold-start window where an unpulled model or a short
> timeout is the more natural suspect. A 404 whose body carries a `path` field
> is a routing mistake, not a model problem. Verify with one real request
> before wiring up the rest of the app; see
> [reference.md](reference.md#app-facing-use-these-from-the-apps-existing-client).

The model identifier on requests stays a Lemonade model name (e.g.
`Qwen3-4B-GGUF`), not the cloud name.

**Local mode needs no cloud API key — at all.** This is a defining property of
local mode, not an edge case: there is no cloud service to authenticate to, so
nothing should ever ask the user for a key. Any onboarding wall, validator, or
startup check that demands one must not block local-mode users. Concretely:

- Skip or auto-satisfy the key-entry screen in local mode.
- Treat local mode as already-authorized in every validation path — an
  empty-key check must short-circuit to "valid" when the active mode is local,
  never throw "API key not configured".
- Re-enable the gate **only** for cloud mode.

The `lemond` key from Step 4 is generated internally by the launcher and used
only for the local loopback connection, so the user never sees or enters one;
any UI placeholder (e.g. `"local"`) is fine. Flipping into local mode should
never strand the user on a key-entry wall.

## Step 6: Health, backend, then pull the model — *before* first inference

`GET /api/v1/health` returning 200 means the **server** is up. It does **not**
mean inference will work. Before the first real request succeeds, three more
things must be true: the backend for your modality is installed, the model's
weights are **downloaded to disk**, and (on the first call) the model is loaded
into memory. Treating health=200 as "ready" is the single biggest cause of a
broken-looking integration.

**Do not call `POST /api/v1/load` at startup.** Lemond lazy-loads the model
into memory on the first inference request and handles that step on its own.
Pre-loading is unreliable across lemond versions (the `/load` request body
shape has changed between releases) and a malformed call can crash or
destabilise the server before the user takes any action. Loading is the one
step you let lemond do lazily — pulling is not.

### Pull the model so it exists on disk

If the model was never pulled, the first inference request **blocks until the
weights finish downloading**. It does not error and it does not return early —
it just takes as long as the download takes, which on a large model over a slow
link is minutes. That collides directly with the 120-second timeout Step 5
makes mandatory: the client gives up mid-download and the user sees a blank UI
with no error, indistinguishable from a broken integration.

So after health passes and the backend is installed, proactively pull the
model. Pulling up front is what makes first-run latency **predictable** and
gives you somewhere to hang a progress indicator:

```http
POST /api/v1/pull
{"model": "Whisper-Large-v3-Turbo"}
```

This is **idempotent** — a no-op if the weights are already present, a download
if they are not. Run it once during setup (after backend install, before the
first user-triggered inference) and log the result.

- **Default model** (the one you chose in Step 2): pull it by name as above.
- **Custom / user-overridden model:** do not assume it exists. Confirm it is a
  real Lemonade model first via `GET /api/v1/models?show_all=true` (the **only**
  trusted catalog — see [reference.md](reference.md); without `show_all` the
  response lists downloaded models only, so a valid name looks invalid), then
  pull it the same way. A model appearing in the catalog is **not** proof its
  weights are downloaded; a successful pull is.

> **If the app skips the pull, first inference inherits the download.** Log the
> pull result and the first inference result (see Step 4) so the console tells
> you which one you are waiting on. If a first request hangs past the timeout
> and a later identical request is fast, the first one was downloading.
>
> Behaviour here has changed across releases: some versions return an empty
> string with HTTP 200 for an unpulled model instead of downloading it. Both
> failure modes are cured by the same explicit pull, which is why it stays a
> mandatory step rather than something to leave to lemond.

### Surface the *whole* setup, not just model load

First-run cold start is more than a model load. The full sequence is:

```
server spawn  →  health 200  →  backend install  →  model download  →  model load  →  first result
```

On a fresh machine, backend install and model download can each take from tens
of seconds to several **minutes** (multi-GB weights over the network). Model
load alone is 10–30s. An app that shows nothing during this will look frozen.

Minimum: show a loading indicator or status message ("Setting up local AI…")
from the moment setup begins until the first response arrives — covering the
*entire* sequence above, not just the final load. The simplest implementation
is a flag set when setup/first-request starts and cleared when the first
response arrives. Once the model is pulled and loaded once, subsequent runs are
fast; the long wait is first-run only.

## Step 7: Lifecycle and recovery

These are the only failure modes worth handling. Do not over-engineer.

| Symptom | Cause | Recovery |
|---|---|---|
| **First inference hangs past the timeout, but a later identical request is fast** | Model was never pulled, so the first request blocked while lemond downloaded the weights | Add the explicit `POST /api/v1/pull` step — see [Step 6](#step-6-health-backend-then-pull-the-model--before-first-inference). Raise the client timeout only as a stopgap; a pull-first setup makes the wait bounded and coverable by a progress indicator |
| **Inference returns empty / blank with HTTP 200, no error** | Model never pulled: on some lemond versions an unpulled model yields an empty result rather than a download | Same fix — `POST /api/v1/pull` with `{"model":"..."}`, wait for success, retry. Log the pull result and the first inference result |
| A 404 body contains a `path` field | The **route** does not exist — a client base-URL mistake, not a missing model. Most often `@anthropic-ai/sdk` pointed at `/api/v1`, producing `/api/v1/v1/messages` | Fix `base_url` per the Step 5 table. Anthropic Messages is `/v1/messages`; reranking is `/api/v1/reranking`. A model-not-found 404 has no `path` field |
| `POST /api/v1/load` returns 404 / model not found | Model not pulled yet (same root cause as the empty-result row above) | `POST /api/v1/pull` with `{"model": "..."}` then retry `/api/v1/load` |
| `POST /api/v1/load` returns 500 with backend error | Backend not installed for this hardware | `GET /api/v1/system-info`, pick a supported backend, `POST /api/v1/install` with `{"recipe": "...", "backend": "..."}`, retry |
| Subprocess exits immediately | Port race: another process grabbed the port between `freePort()` and lemond binding | The reference launcher retries with a fresh port automatically (3 attempts) |
| `/api/v1/health` never returns 200 | First-run backend extraction is slow on cold disk | Extend timeout to 90s on first launch, 30s after |
| HTTP 401 on every request | Forgot the `Authorization: Bearer` header | Audit the client config because Lemonade rejects unauth'd calls when `LEMONADE_API_KEY` is set |

**Shutdown:** On app exit, call `proc.terminate()`, then `proc.wait(timeout=5)`,
and only fall back to `proc.kill()` if the wait expires. `lemond` flushes config
and exits cleanly within a couple of seconds, so the wait is the part that
matters. (On POSIX `terminate()` is SIGTERM and `kill()` is SIGKILL; on Windows
Python maps both to `TerminateProcess`, so there is no graceful/forceful
distinction to make there.) Always wait on the process; never orphan it.

**Do not** parse `lemond` stdout to detect readiness; use the HTTP
`/api/v1/health` probe. Stdout format is not a stable contract.

---

## Verification checklist

The integration is done when **all** of these are true:

- [ ] `vendor/lemonade/` contains the full package: `lemond[.exe]`,
      `lemonade[.exe]`, `LICENSE`, and `resources/` — not just the binary.
- [ ] `lemond` starts as a subprocess with a fresh API key per launch.
- [ ] `GET /api/v1/health` returns 200 within the timeout.
- [ ] The default model is pulled (or bundled) before the first inference; a
      custom/overridden model is confirmed via `GET /api/v1/models?show_all=true`
      and then pulled. A first request that hangs for minutes, or returns blank
      with no error, means this step was skipped.
- [ ] Each lifecycle stage logs a clear line (spawn, health, backend install,
      model pull, first result) so a failure is diagnosable from the console.
- [ ] The existing client's chat / image / speech call returns a valid
      response with the base URL and key swapped, with no other code changed.
- [ ] First-run latency is surfaced: the interface shows a loading state from the
      moment the first inference request is sent until the response arrives.
- [ ] The HTTP client timeout is set to 120 seconds.
- [ ] In local mode the app requires **no** cloud API key: no onboarding wall,
      validator, or startup check blocks the user, and no code path throws
      "API key not configured" when the active mode is local.
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
