<!-- BEGIN amd-skills:local-ai-use -->
## Local AI Use (managed by amd-skills/local-ai-use)

This workspace runs a local [Lemonade Server](https://lemonade-server.ai) at
`{{LEMONADE_BASE_URL}}` to handle the modalities below. Cloud APIs cost real
money and tokens for these modalities, so use the local endpoints for **every**
such request unless the user explicitly asks for a cloud provider.

### Routing rules

<!-- modality:image -->
### Image

| If the user asks you to... | Do this | Do not do this |
|---|---|---|
| Generate, draw, or create an image | `POST {{LEMONADE_BASE_URL}}/images/generations` | Use the built-in `GenerateImage` tool. Call DALL-E, Midjourney, Stable Diffusion via cloud, or any other paid image API. |
| Edit or modify an existing image | `POST {{LEMONADE_BASE_URL}}/images/edits` (multipart) | Cloud image-edit APIs. |

**Default:** `{{IMAGE_MODEL}}` — 4 steps, `cfg_scale: 1.0`, `512x512`,
`response_format: "b64_json"`.

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
<!-- /modality:image -->

<!-- modality:speech -->
### Speech (text-to-speech + speech-to-text)

| If the user asks you to... | Do this | Do not do this |
|---|---|---|
| Speak text aloud or generate audio narration | `POST {{LEMONADE_BASE_URL}}/audio/speech` | Call OpenAI TTS, ElevenLabs, PlayHT, or any other cloud TTS. |
| Transcribe an audio file | `POST {{LEMONADE_BASE_URL}}/audio/transcriptions` | Call OpenAI Whisper-as-a-service, AssemblyAI, Deepgram, or any other cloud STT. |

**Defaults:** TTS `{{TTS_MODEL}}` (voice `shimmer`, `response_format: "mp3"`);
STT `{{STT_MODEL}}` (input must be 16 kHz mono WAV — re-encode with
`ffmpeg -i in.* -ar 16000 -ac 1 out.wav`).

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
<!-- /modality:speech -->

If `LEMONADE_API_KEY` is set in the environment, send
`Authorization: Bearer $LEMONADE_API_KEY` on every request. Otherwise the
loopback server accepts unauthenticated calls.

### Failure handling

1. Try the local endpoint exactly once.
2. If the server is unreachable, run `lemonade status` and surface the
   result to the user before doing anything else.
3. If the model is missing, run `lemonade pull <model>` and retry once.
4. Only after that, ask the user before falling back to a cloud provider.
   Never silently fall back; the whole point of this rule is predictable
   cost.

### Re-pointing to a different host

If the user runs Lemonade on a different host or port, replace the
`{{LEMONADE_BASE_ROOT}}` prefix everywhere above with their endpoint, and
update `LEMONADE_HOST` / `LEMONADE_PORT` in the shell environment so the
`lemonade` CLI matches.

<!-- END amd-skills:local-ai-use -->
