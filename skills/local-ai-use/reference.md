# Local AI Use: Reference

Detailed reference for the `local-ai-use` skill. Read this only when the
default path in `SKILL.md` doesn't cover a decision.

## Contents

- [Model picker](#model-picker)
- [Endpoint reference](#endpoint-reference)
- [API key handling](#api-key-handling)
- [Hardware-accelerated backends](#hardware-accelerated-backends)
- [OmniRouter tool definitions](#omnirouter-tool-definitions)
- [Re-pointing the rule at a remote host](#re-pointing-the-rule-at-a-remote-host)
- [Removing the rule](#removing-the-rule)

---

## Model picker

The default trio (`SD-Turbo`, `kokoro-v1`, `Whisper-Tiny`) is sized for
"keeps cost savings real on a typical laptop". Override only if the user
asks for higher quality or has explicit hardware to spare.

### Image generation (`recipe: sd-cpp`)

| Model | Approx size | When to use | Trade-off |
|---|---|---|---|
| `SD-Turbo` | ~5 GB | **Default.** General-purpose, single-step (4-step) generation. | Lower fidelity than SDXL. |
| `SDXL-Turbo` | ~6.9 GB | When the user notices quality issues with SD-Turbo. | Larger model, slower on CPU. |
| `SD-1.5` | ~4 GB | When the user asks for "Stable Diffusion 1.5" by name. | Needs more steps (~20). |
| `Flux-2-Klein-4B` | ~4 GB | Image **editing** (`/v1/images/edits`). | Editing-capable, slower than SD-Turbo for plain generation. |

To upgrade: re-run setup with the target model, for example:

```bash
python scripts/setup_local_ai.py image --image-model SDXL-Turbo
```

The script pulls the model and rewrites the `AGENTS.md` rule in place.

### Text-to-speech (`recipe: kokoro`)

| Model | Approx size | When to use |
|---|---|---|
| `kokoro-v1` | ~0.3 GB | **Default and only supported model today.** CPU-only, low latency. |

Voices: `shimmer` (default), plus all OpenAI-named voices (`alloy`, `ash`,
`ballad`, `coral`, `echo`, `fable`, `nova`, `onyx`, `sage`, `verse`) and the
kokoro-native voices (`af_sky`, `am_echo`, `bf_emma`, `bm_george`, ...).
Pass `voice` in the request body to override.

### Speech-to-text (`recipe: whispercpp`)

| Model | Approx size | When to use |
|---|---|---|
| `Whisper-Tiny` | ~0.1 GB | **Default.** English-only fast path; great for short clips and meeting notes. |
| `Whisper-Base` | ~0.3 GB | Slightly better accuracy, still tiny. |
| `Whisper-Small` | ~1.0 GB | Multilingual, modest CPU cost. |
| `Whisper-Large-v3-Turbo` | ~1.6 GB | Highest quality / latency mix; recommended when accuracy matters. |

Whisper requires 16 kHz mono PCM WAV input. Convert anything else first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 input.wav
```

For full live coverage, run `lemonade list` after starting the server, or
browse <https://lemonade-server.ai/models.html>.

---

## Endpoint reference

All multimodal endpoints accept the standard OpenAI request shape and
return the standard OpenAI response shape, so any OpenAI-compatible client
works (`openai-python`, `openai-node`, `openai-dotnet`, `go-openai`, ...).

| Method | Path | Purpose | Backend |
|---|---|---|---|
| `POST` | `/api/v1/images/generations` | text → image (b64) | `sd-cpp` |
| `POST` | `/api/v1/images/edits` | image + prompt → image | `sd-cpp` |
| `POST` | `/api/v1/images/variations` | image → varied image | `sd-cpp` |
| `POST` | `/api/v1/images/upscale` | image → upscaled image | ESRGAN |
| `POST` | `/api/v1/audio/speech` | text → audio file | `kokoro` |
| `POST` | `/api/v1/audio/transcriptions` | wav → text | `whispercpp` |
| `WS`   | `/realtime` | streaming microphone → text | `whispercpp` |
| `GET`  | `/api/v1/models` | list models (add `?show_all=true` for catalog) | n/a |
| `GET`  | `/api/v1/health` | readiness probe | n/a |

Notable per-endpoint quirks:

- **`/v1/images/generations`**: only `n=1` and `response_format=b64_json`
  are supported today. `size` defaults to `512x512`. The model's
  `image_defaults` (steps / cfg_scale / width / height) returned by
  `/v1/models/{id}` are the right values to use as your defaults.
- **`/v1/images/edits`**: `multipart/form-data` (not JSON). `mask` is
  optional; without one the entire image is the editable region.
- **`/v1/audio/transcriptions`**: only `wav` input and `json` response are
  supported today. Non-WAV input must be re-encoded with `ffmpeg`.
- **`/v1/audio/speech`**: `mp3`, `wav`, `opus`, and `pcm` outputs supported.
  Streaming requires `stream_format: "audio"`, which only emits `pcm`.

For the full parameter list of any endpoint, see `lemonade/docs/api/openai.md`
upstream.

---

## API key handling

The system-wide Lemonade Server defaults to **no auth** on `localhost`. If
the user has set `LEMONADE_API_KEY` (rare for the system-wide flow,
standard for the embeddable flow), every HTTP request must carry:

```
Authorization: Bearer ${LEMONADE_API_KEY}
```

Update both:

1. The shell environment that the agent runs commands in
   (`export LEMONADE_API_KEY=...`).
2. The rule block in `AGENTS.md`. Add a sentence near the top of the rule
   that says "send `Authorization: Bearer $LEMONADE_API_KEY` on every
   request" — it's already mentioned but worth highlighting.

---

## Hardware-accelerated backends

Default install ships the broad-compatibility backends. To get GPU / NPU
acceleration on supported AMD hardware:

| Modality | Recipe | Faster backend | Install command |
|---|---|---|---|
| Image gen | `sd-cpp` | `rocm` (Strix Halo, RDNA3/4) | `lemonade backends install sd-cpp:rocm` |
| LLM (separate skill) | `llamacpp` | `rocm` or `vulkan` | `lemonade backends install llamacpp:rocm` |
| ASR | `whispercpp` | `npu` (XDNA2) | `lemonade backends install whispercpp:npu` |
| TTS | `kokoro` | `cpu` only | n/a |

After installing a backend, set the corresponding pin in `lemonade config
set`:

```bash
lemonade config set sdcpp_backend rocm
lemonade config set whispercpp_backend npu
```

These are persisted in `config.json` and apply on the next model load.

---

## OmniRouter tool definitions

If the agent uses an OpenAI-style tool-calling loop (Continue, OpenHands,
custom code) instead of plain HTTP, register the same endpoints as named
tools so the LLM can pick them on its own. Lemonade publishes canonical
tool schemas under `OmniRouter`; the minimum useful set for this skill is:

```json
[
  {
    "type": "function",
    "function": {
      "name": "generate_image",
      "description": "Generate an image from a text prompt using local Lemonade Server.",
      "parameters": {
        "type": "object",
        "properties": {
          "prompt": {"type": "string"},
          "size":   {"type": "string", "default": "512x512"},
          "steps":  {"type": "integer", "default": 4}
        },
        "required": ["prompt"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "text_to_speech",
      "description": "Speak the given text aloud using local Lemonade Server.",
      "parameters": {
        "type": "object",
        "properties": {
          "input": {"type": "string"},
          "voice": {"type": "string", "default": "shimmer"}
        },
        "required": ["input"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "transcribe_audio",
      "description": "Transcribe a WAV audio file using local Lemonade Server.",
      "parameters": {
        "type": "object",
        "properties": {
          "file_path": {"type": "string"},
          "language":  {"type": "string"}
        },
        "required": ["file_path"]
      }
    }
  }
]
```

Map each `tool_call` to the corresponding endpoint:

- `generate_image` → `POST /api/v1/images/generations`
- `text_to_speech` → `POST /api/v1/audio/speech`
- `transcribe_audio` → `POST /api/v1/audio/transcriptions`

For the full canonical schema (including `edit_image` and `analyze_image`),
read `examples/lemonade_tools.py` in the upstream lemonade-sdk repo.

---

## Re-pointing the rule at a remote host

Lemonade can run on another machine (a workstation with a Ryzen AI NPU,
say) while the agent runs on the laptop. To point this skill at it:

1. Set `LEMONADE_HOST` and `LEMONADE_PORT` (or pass `--host` / `--port` to
   `setup_local_ai.py`).
2. Re-run `python scripts/setup_local_ai.py` so the rule block is rewritten
   with the new endpoint baked in.
3. Make sure the remote server is bound to a non-loopback interface
   (`lemonade config set host 0.0.0.0`) and that firewall rules allow
   inbound 13305. Setting `host` to `0.0.0.0` exposes the server; pair it
   with `LEMONADE_API_KEY` so it isn't open to the LAN.

---

## Removing the rule

To stop routing locally (e.g., the user wants cloud back), open the
workspace `AGENTS.md` and delete everything between
`<!-- BEGIN amd-skills:local-ai-use -->` and
`<!-- END amd-skills:local-ai-use -->`. The agent picks up the change on
its next turn.

The downloaded models stay on disk; remove them with `lemonade delete
<model>` if you want the space back.
