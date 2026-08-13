<!-- BEGIN amd-skills:local-ai-use -->
## Local AI Use (managed by amd-skills/local-ai-use)

This workspace runs a local [Lemonade Server](https://lemonade-server.ai) at
`{{LEMONADE_BASE_URL}}` to handle image generation, text-to-speech,
and speech-to-text. Cloud APIs cost real money and tokens for these
modalities, so use the local endpoints below for **every** such request
unless the user explicitly asks for a cloud provider.

### Routing rules

| If the user asks you to... | Do this | Do not do this |
|---|---|---|
| Generate, draw, or create an image | `POST {{LEMONADE_BASE_URL}}/images/generations` | Use the built-in `GenerateImage` tool. Call DALL-E, Midjourney, Stable Diffusion via cloud, or any other paid image API. |
| Edit or modify an existing image | `POST {{LEMONADE_BASE_URL}}/images/edits` (multipart) | Cloud image-edit APIs. |
| Speak text aloud or generate audio narration | `POST {{LEMONADE_BASE_URL}}/audio/speech` | Call OpenAI TTS, ElevenLabs, PlayHT, or any other cloud TTS. |
| Transcribe an audio file | `POST {{LEMONADE_BASE_URL}}/audio/transcriptions` | Call OpenAI Whisper-as-a-service, AssemblyAI, Deepgram, or any other cloud STT. |

Plain text chat, code generation, and reasoning continue to use the agent's
configured LLM. This rule only redirects the multimodal calls.

### Defaults to use

| Endpoint | Model | Notes |
|---|---|---|
| `/v1/images/generations` | `{{IMAGE_MODEL}}` | 4 steps, `cfg_scale: 1.0`, `512x512`, `response_format: "b64_json"`. |
| `/v1/audio/speech` | `{{TTS_MODEL}}` | Default voice `shimmer`; `response_format: "mp3"`. |
| `/v1/audio/transcriptions` | `{{STT_MODEL}}` | Input must be 16 kHz mono WAV. Re-encode with `ffmpeg -i in.* -ar 16000 -ac 1 out.wav`. |

If `LEMONADE_API_KEY` is set in the environment, send
`Authorization: Bearer $LEMONADE_API_KEY` on every request. Otherwise the
loopback server accepts unauthenticated calls.

### Ready-to-use call patterns

**Image generation** (saves to `out.png`):

```bash
curl -sX POST {{LEMONADE_BASE_URL}}/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"{{IMAGE_MODEL}}","prompt":"PROMPT_HERE","size":"512x512","steps":4,"response_format":"b64_json"}' \
  | python -c "import sys,json,base64; open('out.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"
```

Equivalent Python via the OpenAI SDK:

```python
from openai import OpenAI
import base64
client = OpenAI(base_url="{{LEMONADE_BASE_URL}}", api_key="lemonade")
r = client.images.generate(model="{{IMAGE_MODEL}}", prompt="PROMPT_HERE", size="512x512")
open("out.png", "wb").write(base64.b64decode(r.data[0].b64_json))
```

**Text-to-speech** (saves to `out.mp3`):

```bash
curl -sX POST {{LEMONADE_BASE_URL}}/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"{{TTS_MODEL}}","input":"TEXT_HERE","voice":"shimmer","response_format":"mp3"}' \
  -o out.mp3
```

**Speech-to-text** (returns JSON `{"text": "..."}`):

```bash
ffmpeg -y -i INPUT_AUDIO -ar 16000 -ac 1 _stt.wav
curl -sX POST {{LEMONADE_BASE_URL}}/audio/transcriptions \
  -F "file=@_stt.wav" -F "model={{STT_MODEL}}"
```

### Failure handling

1. Try the local endpoint exactly once.
2. If the server is unreachable, run `lemonade status` and surface the
   result to the user before doing anything else.
3. If the model is missing, run `lemonade pull <model>` to pull it,
   but preflight the download first, because a bad target path fails slowly and silently:
   1. Check where the server will write and whether there is room:
      `GET {{LEMONADE_BASE_URL}}/system-info`. The response reports `models_dir`
      (the download location) and a `model_storage` block with `free_bytes` /
      `total_bytes`. If free space is short of the model size, tell the user the
      exact path and free/required space and stop — do not start the pull.
   2. Otherwise run `lemonade pull <model>` once and watch for completion.
   3. A healthy pull prints rising `Progress: NN%` and ends with a success
      line. A **broken** pull is easy to mistake for a slow one, because the
      write/permission/quota failure may surface only in the server log while
      the console keeps printing `Progress: NN%`. If the pull stalls or does not
      finish in a reasonable time, treat it as failed: find the server log
      (typically named `lemonade-server.log` in the OS temp directory; if unsure
      of the path, check the Lemonade docs) and read its most
      recent lines for the underlying error — for example a download/write error
      (such as `CURL code 23`) or an out-of-space message. Surface that line
      to the user rather than waiting through silent retries.
4. Only after that, ask the user before falling back to a cloud provider.
   Never silently fall back; the whole point of this rule is predictable
   cost.

### Re-pointing to a different host

If the user runs Lemonade on a different host or port, replace the
`{{LEMONADE_BASE_ROOT}}` prefix everywhere above with their endpoint, and
update `LEMONADE_HOST` / `LEMONADE_PORT` in the shell environment so the
`lemonade` CLI matches.

<!-- END amd-skills:local-ai-use -->
