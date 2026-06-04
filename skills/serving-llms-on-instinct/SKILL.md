---
name: serving-llms-on-instinct
description: >-
  Serves AI models on AMD Instinct GPU hardware using vLLM. Use this skill
  whenever the user wants to run, serve, deploy, start, host, or launch a
  language model on an AMD GPU, AMD Instinct, MI300X, MI350X, or MI325X.
  Also use when the user mentions vLLM on ROCm, vLLM on AMD, serving on HBM,
  or asks how to get a model running on AMD data center hardware. Use when the
  user asks "run Qwen3", "serve DeepSeek", "start a vLLM endpoint", "get a
  model running on my AMD machine", or any similar phrasing. Handles the full
  flow: GPU detection, environment validation, vLLM configuration, launch, and
  health verification. Do not use for NVIDIA GPUs, consumer AMD GPUs (RX
  series, Radeon), Ryzen AI, NPU, MI250X, or MI100.
allowed-tools: Bash, Read
---

# Serving LLMs on AMD Instinct

Get a vLLM endpoint running on AMD Instinct GPU hardware.

## Prerequisites

- ROCm driver and `amd-smi` installed on the GPU host
- Docker running and accessible (check with `docker ps`)
- `/dev/kfd` and `/dev/dri` present on the GPU host
- HuggingFace token in `HF_TOKEN` env var (required for gated models; not
  required for Qwen3 or Gemma)
- For remote GPU: SSH key access configured (`ssh <user>@<host>` must work
  without a password prompt)

## Data files

Read these files directly to get model and GPU configuration:

- **`data/recipes_cache.json`** -- model configs synced from
  [vllm-project/recipes](https://github.com/vllm-project/recipes). Each entry
  under `models.<HF_ID>.recipe` contains the full recipe with `model.base_args`,
  `model.base_env`, `features.tool_calling.args`, `features.reasoning.args`,
  `hardware_overrides.amd.extra_args`, `hardware_overrides.amd.extra_env`.
  The top-level `docker_image` field has the latest resolved vLLM ROCm image.

- **`data/gpu_overrides.json`** -- GPU-specific configuration. Contains
  `docker_flags` (mandatory for all AMD Instinct), `gpu_configs` keyed by
  gfx_version with `env_defaults` and `workarounds`, and `legacy_models` for
  models not yet in vLLM recipes.

- **`data/blacklist.json`** -- models in vLLM recipes that cannot be served
  as LLM endpoints. Includes diffusion/image/audio generation models, embedding
  models, rerankers, ASR models needing audio pipelines, and models requiring
  unreleased vLLM nightly builds. Check this before attempting to serve a model.
  If the user requests a blacklisted model, explain why it won't work and
  suggest an alternative.

If the user doesn't specify a model, default to **Qwen/Qwen3.5-9B**: dense
multimodal with MTP, Apache 2.0 license (no HF token needed), fits on a single
GPU, strong reasoning and tool-calling.

## Step 1: Detect the GPU

```bash
python3 scripts/detect.py
# Remote:
python3 scripts/detect.py --host user@hostname
```

Returns JSON with `gfx_version`, `vram_gb`, `gpu_count`, `rocm_version`.

| gfx_version | Hardware | VRAM |
|---|---|---|
| gfx950 | MI350 / MI350X | 192-294 GB |
| gfx942 | MI300X / MI300A / MI325X | 128-288 GB |

If `gfx_version` is `unknown`: `amd-smi` ran but found no GPU. Check
`lsmod | grep amdgpu`.

## Step 2: Validate the environment

```bash
python3 scripts/validate.py --auto-fix
# Remote:
python3 scripts/validate.py --auto-fix --host user@hostname
```

Returns JSON with `ready` (bool), `errors`, `warnings`, `fixes_applied`.
Do not proceed if `ready` is `false`.

## Step 3: Refresh recipes (if stale)

Check `fetched_at` in `data/recipes_cache.json`. If older than 24 hours or
the file is missing, refresh:

```bash
python3 scripts/sync_recipes.py
```

This shallow-clones vllm-project/recipes from GitHub and fetches the latest
Docker tag from Docker Hub. Takes ~10 seconds. If it fails, the existing
cache still works.

## Step 4: Construct the Docker command

Read `data/recipes_cache.json` and `data/gpu_overrides.json` directly.
Build the Docker command by combining:

1. **Docker flags** from `gpu_overrides.json > docker_flags` (mandatory for all AMD GPUs)
2. **HF cache mount**: `-v ~/.cache/huggingface:/root/.cache/huggingface`
   (if a shared cache like `/home/amd/models` exists, mount it to
   `/root/.cache/huggingface/hub` instead)
3. **Port**: `-p <port>:<port>` (default 8000)
4. **Environment variables**: merge `gpu_configs.<gfx_version>.env_defaults`
   with the recipe's `model.base_env` and `hardware_overrides.amd.extra_env`.
   Always add `--env HF_TOKEN=${HF_TOKEN}`.
5. **Docker image**: use `docker_image` from `recipes_cache.json` top level
   (unless the model needs a pinned image, e.g. GLM-4.5 needs `v0.15.1`)
6. **Model ID**: `--model <HF_ID>`
7. **vLLM args**: combine the recipe's `model.base_args` +
   `hardware_overrides.amd.extra_args` + `features.tool_calling.args` +
   `features.reasoning.args`. Add `--enable-auto-tool-choice` if not present.
   For multi-GPU, add `--tensor-parallel-size N`.
8. **Port arg**: `--port <port>`

If the model is not in `recipes_cache.json`, check `legacy_models` in
`gpu_overrides.json`. If not there either, use a generic config with
`--enable-auto-tool-choice --trust-remote-code --tool-call-parser hermes`.

Docker command template:
```
docker run -d --name vllm-<model-slug> \
  <docker_flags> \
  -v <hf_cache_mount> \
  -p <port>:<port> \
  --env <key>=<value> (for each env var) \
  --env HF_TOKEN=${HF_TOKEN} \
  <docker_image> \
  --model <model_id> \
  <vllm_args> \
  --port <port>
```

## Step 5: Launch and verify

Before launching, check for port conflicts:
```bash
ss -tlnp 2>/dev/null | grep ':<port> '
```
If a Docker container is on that port, stop it with `docker rm -f <name>`.

Run the Docker command. Then poll health:
```bash
until curl -sf http://localhost:8000/health; do sleep 10; done && echo "READY"
```

Expected load times after the model is cached locally:
- Small models (< 20B): 2-4 minutes
- Large models (70B+): 8-15 minutes

A 503 during this window is normal. Only conclude failure after 15+ minutes.

After health returns 200, send a warmup request (triggers HIP kernel compilation,
30-90 seconds on gfx942):
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<model_id>","messages":[{"role":"user","content":"say hi"}],"max_tokens":5}'
```

Return to the user:
- `base_url`: `http://<host>:8000/v1`
- `api_key`: none required for local
- `model`: the model ID used

## Remote vs. local

All scripts accept `--host user@hostname`. When given, they SSH to the target.
Set `ROCM_SSH_HOST` and `ROCM_SSH_USER` env vars to avoid passing `--host`
every time.

For remote Docker commands, run them over SSH:
```bash
ssh user@host 'docker run -d ...'
```
Use `localhost` for health/warmup curl URLs (curl runs on the remote host).

## Gotchas

**`CUDA_VISIBLE_DEVICES` set on the host** -- AMD GPUs disappear. The ROCm
runtime treats this NVIDIA variable as "no visible GPUs." Unset it before
launching: `unset CUDA_VISIBLE_DEVICES`. Pass `--env CUDA_VISIBLE_DEVICES=`
in the Docker command to block it inside the container.

**FP4BMM crash on gfx942 (MI300X)** -- If the container exits immediately
with a segfault or illegal instruction: `VLLM_ROCM_USE_AITER_FP4BMM` must be
`0` on gfx942. This is set correctly in `gpu_overrides.json` for gfx942.
See vLLM issue #34641.

**`HIP error: no kernel image`** -- The Docker image has no compiled kernel
for your GPU's gfx version. Use `vllm/vllm-openai-rocm:latest`; it includes
gfx942 and gfx950 kernels.

**MLA models need `--block-size 1`** -- DeepSeek-R1/V3, Kimi-K2.5.
Without it the MLA attention backend silently falls back to a slower path.
This is in the recipe args for these models.

**MoE models on multi-GPU need `--distributed-executor-backend mp`** --
Qwen3-235B, GLM-4.5, MiniMax-M2. The default distributed executor does not
work reliably with MoE on ROCm.

**`/dev/kfd` permission denied** -- User is not in the `video` or `render`
group. Fix: `sudo usermod -aG video,render $USER` (requires re-login).

**SSH key not configured** -- The scripts use `BatchMode=yes` SSH. If SSH
fails with `Permission denied (publickey)`, configure key-based access first.

**Restricting GPUs on shared hosts** -- Use `--env HIP_VISIBLE_DEVICES=0,1`
to target specific GPUs by index. Never set `CUDA_VISIBLE_DEVICES` on AMD.

---

## Reference

Full GPU architecture table, env var reference, flag details, and known quirks:
[reference.md](reference.md)
