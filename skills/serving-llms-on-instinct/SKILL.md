---
name: serving-llms-on-instinct
description: >-
  Serves AI models on AMD Instinct GPU hardware using vLLM. Use this skill
  whenever the user wants to run, serve, deploy, start, host, or launch a
  language model on an AMD GPU, AMD Instinct, MI300X, MI325X, MI350X, or MI355X.
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
  required for Qwen3 or Gemma). For gated models (Llama 3.2, Gemma, etc.),
  the HF token must belong to an account that has accepted the model's license
  at `huggingface.co/<model_id>`. A valid token without license acceptance will
  fail with an opaque "Engine core initialization failed" error.
- For remote GPU: SSH key access configured (`ssh <user>@<host>` must work
  without a password prompt). If only password access is available, set up
  keys first: `ssh-copy-id <user>@<host>`

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
| gfx950 | MI350X / MI355X | 288 GB HBM3E |
| gfx942 | MI300X (192 GB) / MI325X (256 GB) / MI300A (128 GB) | varies |

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
   (if a shared model cache directory exists on the host, check whether
   `models--*` directories are at the cache root or inside a `hub/`
   subdirectory -- mount accordingly to `/root/.cache/huggingface` or
   `/root/.cache/huggingface/hub`)
3. **Port**: `-p <port>:<port>` (default 8000)
4. **Environment variables**: merge `gpu_configs.<gfx_version>.env_defaults`
   with the recipe's `model.base_env` and `hardware_overrides.amd.extra_env`.
   Always add `--env HF_TOKEN=${HF_TOKEN}`.
5. **Docker image**: use `docker_image` from `recipes_cache.json` top level
   (unless the model needs a pinned image, e.g. GLM-4.5 needs `v0.15.1`).
   If the user specifies a Docker image version, check it against the recipe's
   `model.min_vllm_version`. Warn if the image is older -- the model may crash
   on startup with an opaque "Engine core initialization failed" error.
6. **Model ID**: `--model <HF_ID>`
7. **vLLM args**: combine the recipe's `model.base_args` +
   `hardware_overrides.amd.extra_args` + `features.tool_calling.args` +
   `features.reasoning.args`. Add `--enable-auto-tool-choice` if not present.
   For multi-GPU, add `--tensor-parallel-size N` (see VRAM estimation below).
   For MoE models on multi-GPU, also add `--distributed-executor-backend mp`.
8. **Port arg**: `--port <port>`

If the exact model ID is not in `recipes_cache.json`, check for a base model
match by stripping date/version suffixes (e.g., `Kimi-K2-Instruct` matches
`Kimi-K2-Instruct-0905`). Use the base model's recipe if found.

If no recipe match, check `legacy_models` in `gpu_overrides.json`. If not
there either, use a generic config with
`--enable-auto-tool-choice --trust-remote-code --tool-call-parser hermes`.

**Precision variant selection:** Recipes may offer variants (default, fp8,
nvfp4). Check `gpu_configs.<gfx_version>.precision.native` in
`gpu_overrides.json` before selecting a variant. On gfx942 (MI300X), only
`bf16`, `fp16`, `fp8_fnuz`, and `int8` are hardware-native. MXFP4 and NVFP4
compute is emulated (dequant to BF16 during matmul), but weights stay
compressed in VRAM so quantized models still fit in less memory.
On gfx950 (MI350X), MXFP4 is hardware-native.

**VRAM estimation and fit check:** Before constructing the Docker command,
estimate whether the model fits the available hardware:
```bash
python3 scripts/estimate_vram.py --model-id <HF_ID> --vram-gb <per_gpu_vram> --tp <N>
```
This queries the HuggingFace Hub API (no model download) and returns JSON with:
- `weight_memory_gb` -- total weight size
- `kv_cache_bytes_per_token` -- KV cache cost per token at BF16
- `fit.weights_fit` -- whether weights fit at the given TP
- `fit.recommended_max_model_len` -- max context the GPU can serve
- `fit.context_limited` -- true if KV cache limits context below the
  model's native max
- `fit.min_tp_required` -- minimum TP needed (only if weights don't fit)

**Understanding the overhead:** The script reserves ~4 GB for vLLM's runtime
overhead (activation profiling, HIP graph capture, internal buffers). During
startup, vLLM runs a profiling forward pass to measure peak activations, then
captures HIP graphs for optimized decode. This startup peak is higher than
steady-state. The `remaining_for_kv_gb` field reflects what's left after
weights and this overhead.

Use `remaining_for_kv_gb` to decide:

1. **`remaining_for_kv_gb >= 6`**: safe to run. If `context_limited: true`,
   add `--max-model-len <recommended_max_model_len>` to the vLLM args.
   Mention the FP8 KV cache option (`--kv-cache-dtype fp8`) if the user
   needs longer context (`fit.max_seq_len_fp8_kv` shows the gain).
2. **`remaining_for_kv_gb` between 2 and 6**: tight but worth trying. Launch
   normally. If vLLM OOMs during HIP graph capture (check container logs for
   "out of memory" after "capturing CUDA/HIP graphs"), retry with
   `--enforce-eager` added to the vLLM args. This skips graph capture and
   frees 1-2 GB. The only cost is slightly higher decode latency.
3. **`remaining_for_kv_gb < 2`**: too tight. Will likely OOM during the
   activation profiling step. Do not attempt.
4. **`weights_fit: false` with multiple GPUs**: re-run with
   `--tp <min_tp_required>` and check again.
5. **`weights_fit: false`, not enough GPUs**: look for quantized
   alternatives in this order:
   a. **Recipe variants**: the recipe may have `fp8` or `mxfp4` variants
      with a different `model_id` that points to a quantized checkpoint.
   b. **Same provider**: many providers release quantized versions alongside
      the base model (e.g. `Qwen/Qwen3.5-122B-FP8` from Qwen). Search
      HuggingFace for `<provider>/<model-name>` with FP8/GPTQ/AWQ suffixes.
   c. **AMD quantized**: AMD's Quark team publishes quantized models under
      the `amd/` org on HuggingFace (e.g. `amd/Kimi-K2-Instruct-w-mxfp4-a-fp8`).
      Search for `amd/<model-name>` variants.
   Run `estimate_vram.py` on the quantized model ID to verify it fits,
   then use that model ID instead.
6. **Still doesn't fit**: tell the user the model requires more VRAM than
   available and suggest either a smaller model or multi-GPU hardware.
   Do not attempt to launch.

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

## Step 5: Confirm with the user

Before launching, present a summary and ask the user to confirm:
- **Model**: full HuggingFace ID (e.g. `Qwen/Qwen3.5-122B-Instruct`)
- **Precision**: variant being used (e.g. BF16, FP8) and why
- **Weight memory**: from estimate_vram.py
- **GPU**: detected hardware and VRAM
- **TP**: tensor parallelism degree (1, 2, 4, 8)
- **Context**: max achievable context length (and whether it's limited)
- **Port**: which port the endpoint will be on

If a quantized alternative was selected (Step 4 fit check), explain that
the original model doesn't fit and which alternative is being used.

Wait for the user's confirmation before proceeding.

## Step 6: Launch and verify

Before launching, check for port conflicts:
```bash
ss -tlnp 2>/dev/null | grep ':<port> '
```
If a Docker container is on that port, stop it with `docker rm -f <name>`.

Run the Docker command. Then poll health using this loop:

```bash
while docker inspect --format='{{.State.Running}}' <container_name> 2>/dev/null | grep -q true; do
  curl -sf http://localhost:<port>/health && echo "READY" && exit 0
  sleep 60
done
echo "FAILED -- container exited"
```

A 503 during loading is normal. Choose the polling strategy based on
model size (weight memory from hf-mem):

- **Small models (< 100 GB weights)**: run the poll as a blocking command
  with the Bash tool's `timeout` set to 600000 (10 minutes). Most cached
  models are ready within 2-5 minutes.
- **Large models (>= 100 GB weights)**: run the poll with the Bash tool's
  `run_in_background` set to `true`. Then use `TaskOutput` with
  `block: true` and `timeout: 600000` to wait up to 10 minutes per check.
  If the task is still running after that, call `TaskOutput` again with
  the same parameters. This uses only 1 turn per 10-minute wait instead
  of burning a turn every check. The background loop runs until the
  container is healthy or dies.

After health returns 200, send a warmup request (triggers HIP kernel compilation,
~40-45 seconds on gfx942):
```bash
curl -s http://localhost:<port>/v1/chat/completions \
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

**`CUDA_VISIBLE_DEVICES` set to empty string** -- ROCm maps this variable to
`HIP_VISIBLE_DEVICES`. Setting it to an empty string hides all GPUs.
`CUDA_VISIBLE_DEVICES=0,1` works fine for restricting GPUs (same as
`HIP_VISIBLE_DEVICES=0,1`). If the host has it set to empty, unset it:
`unset CUDA_VISIBLE_DEVICES`. Do not pass `--env CUDA_VISIBLE_DEVICES=` (empty)
into Docker -- that also hides all GPUs inside the container.

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

**OOM during HIP graph capture** -- If the container logs show "out of memory"
after "capturing CUDA graphs" or "capturing HIP graphs", the model fits in
VRAM but there isn't enough headroom for graph capture. Retry with
`--enforce-eager` added to the vLLM args. This disables graph capture and
frees 1-2 GB. Trade-off: slightly higher decode latency, but the model runs.

**"Engine core initialization failed"** -- This opaque error means the engine
core subprocess died. Check early container logs: `docker logs <name> 2>&1 |
head -50`. Common causes: gated model access denied (license not accepted on
HF), unsupported architecture on this vLLM version, OOM during weight loading,
missing `--trust-remote-code` for custom architectures, or vLLM version too old
for the model (check `min_vllm_version` in the recipe).

**`/dev/kfd` permission denied** -- User is not in the `video` or `render`
group. Fix: `sudo usermod -aG video,render $USER` (requires re-login).

**SSH key not configured** -- The scripts use `BatchMode=yes` SSH. If SSH
fails with `Permission denied (publickey)`, configure key-based access first.

**Restricting GPUs on shared hosts** -- Use `--env HIP_VISIBLE_DEVICES=0,1`
or `--env CUDA_VISIBLE_DEVICES=0,1` to target specific GPUs by index.
`HIP_VISIBLE_DEVICES` is the canonical AMD variable; `CUDA_VISIBLE_DEVICES`
also works (ROCm maps it). Never set either to an empty string.

---

## Reference

Precision compatibility, VRAM estimation, Docker flags, and known quirks:
[reference.md](reference.md)
