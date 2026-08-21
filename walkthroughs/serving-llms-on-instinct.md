# AMD Skills Walkthroughs: `serving-llms-on-instinct`

The goal of this skill is to teach your AI agent to bring up a vLLM OpenAI-compatible
endpoint on an **AMD Instinct™ GPU** host on ROCm: detecting the GPU, validating the
environment, picking the right vLLM recipe for the model, checking the model fits VRAM,
launching the container, and verifying the endpoint responds.

**What you'll end up with:** a running vLLM endpoint on your Instinct box (in a Docker
container built from the ROCm vLLM image), sized to the GPUs you actually have, and
ready to answer OpenAI requests via `/v1/chat/completions`.

## Prerequisites

- A supported **AMD Instinct™ GPU**: **MI300X** / **MI325X** / **MI300A** (`gfx942`) or **MI350X** / **MI355X** (`gfx950`). `detect.py` reports `gfx_version`; anything else is out of scope. This skill explicitly does **not** cover MI250X, MI100, consumer Radeon (RX series), or Ryzen AI / NPU.
- **ROCm driver and `amd-smi`** installed on the host, with `/dev/kfd` and `/dev/dri` present. Check the driver is loaded with `lsmod | grep amdgpu`.
- **Docker** running and accessible — `docker ps` must work without `sudo`. If it errors on `/dev/kfd` permissions, add yourself to the GPU groups: `sudo usermod -aG video,render $USER` (requires re-login).
- Disk space in `~/.cache/huggingface` for the model weights (the default model below is roughly 18 GB at BF16).
- A HuggingFace token in `HF_TOKEN` **only** for gated models (Llama, Gemma). The default model (Qwen3.5) needs none. For gated models the token must belong to an account that has **accepted the license** at `huggingface.co/<model_id>` — a valid token without license acceptance fails with an opaque `Engine core initialization failed`.
- **Node.js ≥ 18**, required by the `skills` CLI used in Step 2 (`npx skills ...`). Check with `node -v`.
- *No Instinct hardware handy?* AMD Developer Cloud offers hosted MI300X instances that satisfy the requirements above. Note this is just a way to **get** a host — the skill itself does not provision or onboard cloud instances; it expects a machine that's already up.

## Step 1 - Understanding which skills are available

* Start in a clean scratch directory, then run `claude "Which skills can you see?" --model sonnet`. You should see a list of skills that does **not** include anything about serving LLMs on Instinct / AMD GPUs.
* Confirm the scratch directory has no `AGENTS.md`, `CLAUDE.md`, `.claude/skills`, or `.agents/skills`. Existing agent instructions or installed skill copies can change discovery and invalidate the before/after comparison. Do not delete instructions from a real project; use a clean scratch directory instead.

## Step 2 - Enabling claude to see `serving-llms-on-instinct`

* Install the skill with the [`skills` CLI](https://github.com/vercel-labs/skills). Run this in your terminal, not inside Claude:

```bash
npx skills add amd/skills --skill serving-llms-on-instinct --agent claude-code
```

* Run `claude "Which skills can you see?" --model sonnet`. You should see a list of skills that now includes `serving-llms-on-instinct`.

## Step 3 - Running the skill

Run `claude --model sonnet` on your Instinct host with this prompt:

```
Serve Qwen/Qwen3.5-9B on this AMD Instinct GPU with vLLM.
```

Claude should:

1. **Detect the GPU**: read `gfx_version`, `vram_gb`, `gpu_count`, and `rocm_version`, and map the gfx target to the hardware (`gfx942` → MI300X/MI325X/MI300A, `gfx950` → MI350X/MI355X). The gfx version drives precision support and the workarounds applied later.
2. **Validate the environment**: check `/dev/kfd`, `/dev/dri`, Docker accessibility, NUMA balancing, hipBLASLt, and `HF_TOKEN`, classifying each as error / warning / advisory. Safe fixes (like disabling NUMA balancing, which causes latency spikes for GPU workloads) are applied with `--auto-fix`. It does not proceed if the environment isn't ready.
3. **Refresh the vLLM recipe cache** if it's older than 24 hours, pulling the latest model recipes from [vllm-project/recipes](https://github.com/vllm-project/recipes) and resolving the current ROCm vLLM Docker tag (~10 seconds). A stale cache still works if the refresh fails.
4. **Check the model is actually servable**: reject blacklisted models — diffusion/image and audio generation, embeddings, rerankers, ASR models needing an audio pipeline, and models that require an unreleased vLLM nightly — and suggest an alternative rather than failing at launch.
5. **Build the config from the model's recipe**: mandatory AMD Docker flags, the HF cache mount, gfx-specific env defaults, the recipe's vLLM args plus tool-calling and reasoning parsers, and the ROCm image (`vllm/vllm-openai-rocm`, never the CUDA-only `vllm/vllm-openai`). It picks a precision variant the GPU supports — on `gfx942` FP8 is FNUZ and MXFP4 compute is emulated; on `gfx950` MXFP4 is native; NVFP4 is rejected on both.
6. **Check it fits VRAM**: estimate weight memory and KV cache from the HuggingFace Hub API (no download), reserving ~4 GB for vLLM runtime overhead, then decide from what's left — cap `--max-model-len` if context is KV-limited, raise tensor parallelism if the weights need more than one GPU, or switch to a quantized checkpoint (recipe variant, same-provider FP8, or an `amd/` Quark model) if they don't fit at all.
7. **Confirm the plan with you**: present a summary (model, precision and why, weight memory, GPU, TP degree, achievable context, port) and wait for you to approve before launching. If it swapped in a quantized alternative, it says so and explains why.
8. **Launch and verify**: check the port is free, run the container, poll `/health` until the server answers (a 503 while loading is normal), then send a warmup request — the first inference compiles HIP kernels and takes ~40-45 seconds on `gfx942` — and hand back a connection table.

Note the one place this skill *does* retry: if the model fits but vLLM OOMs during HIP graph
capture, it relaunches once with `--enforce-eager`, which frees 1-2 GB at the cost of slightly
higher decode latency. That is a bounded, known-cause retry, not a debugging loop.

## Step 4 - Talk to the endpoint

Once Claude reports the endpoint is healthy, use the **base URL, served-model name, and
port from Claude's connection table** (it uses port `8000` by default):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```

Prefer Python? Point the OpenAI SDK at the local server (`base_url` ends in `/v1`;
the SDK needs a non-empty key, so use any placeholder when there is no auth):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
model = client.models.list().data[0].id
r = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=128,
)
print(r.choices[0].message.content)
```

`max_tokens` caps the output and `prompt_tokens + max_tokens` must stay within the served
`--max-model-len`. Set `temperature` (0 for deterministic) and `stream=True` to stream tokens.

To stop the endpoint: `docker rm -f <container name from the connection table>`.

## Step 5 - (Optional) Going beyond

* **A bigger model:** ask for something that needs more than one GPU, e.g. *"Serve Qwen/Qwen3-235B-A22B"*. Claude re-runs the fit check, raises tensor parallelism, and adds `--distributed-executor-backend mp` for MoE models on multi-GPU.
* **A model that doesn't fit:** ask for a large dense model on a single GPU and watch it find a quantized alternative (recipe FP8 variant, a same-provider FP8 checkpoint, or an `amd/` Quark model) instead of launching into an OOM.
* **Gated models:** `export HF_TOKEN=...`, accept the license on HuggingFace, then ask for `meta-llama/Llama-3.2-...`.
* **Longer context:** if Claude reports the context is KV-limited, ask about FP8 KV cache (`--kv-cache-dtype fp8`) to buy more sequence length out of the same VRAM.
* **Share a host:** on a busy multi-GPU node, tell it which GPUs to use (*"only use GPUs 0 and 1"*) — it restricts with `HIP_VISIBLE_DEVICES`.
* **Drive it remotely:** every script accepts `--host user@hostname` (or the `ROCM_SSH_HOST` / `ROCM_SSH_USER` env vars) and runs over SSH, so you can serve on a remote Instinct node from your laptop. Key-based SSH must already work.

## Step 6 - (Optional) Try to get things done without AMD Skills

Remove the added skill and rerun the experiment above. The `skills` CLI installs a copy
under **both** `.claude/skills/serving-llms-on-instinct` **and**
`.agents/skills/serving-llms-on-instinct`, so delete both (otherwise the leftover copy
keeps the skill active and the comparison isn't clean). Without the skill, common issues
include:

* Reaching for `vllm/vllm-openai` — the CUDA-only image — instead of `vllm/vllm-openai-rocm`, so nothing finds a GPU.
* Missing the mandatory AMD Docker flags (`--device /dev/kfd`, `--device /dev/dri`, `--group-add video/render`, `--cap-add SYS_PTRACE`, `--security-opt seccomp=unconfined`, `--ipc=host`), so the container starts but sees no GPU or dies during ROCm JIT.
* No VRAM fit check: launching a model that doesn't fit and looping on `Engine core initialization failed`, or fitting the weights but OOM-ing during HIP graph capture without knowing `--enforce-eager` is the fix.
* Picking an NVFP4 checkpoint, which has no dequant kernel on ROCm and will not load — or assuming MXFP4 is native on MI300X, where it is emulated.
* Missing the gfx-specific workarounds, most notably `VLLM_ROCM_USE_AITER_FP4BMM=0` on `gfx942`, which otherwise segfaults during warmup.
* Missing per-model args like `--block-size 1` for MLA models (DeepSeek-R1/V3, Kimi-K2.5), which silently falls back to a slower attention path.
* Trying to serve a diffusion, embedding, or ASR model as a chat endpoint instead of catching it up front.
* Declaring success at `/health` 200 without a warmup request, so your first real call appears to hang for ~40 seconds of HIP kernel compilation.
* Providing a knowledge article instead of actually bringing up a working endpoint.
