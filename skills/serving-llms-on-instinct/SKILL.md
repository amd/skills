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

Get a vLLM endpoint running on AMD Instinct GPU hardware. `scripts/plan.py`
does all the lookups (GPU, recipe, precision, fit, flags, environment); your
job is to run it, relay what it says, and then launch and verify.

## Prerequisites

- ROCm, `amd-smi`, and Docker on the GPU host; `/dev/kfd` and `/dev/dri` present.
- Remote GPU: key-based SSH (`ssh <user>@<host>` works with no password prompt).
- Gated models (Llama, some Gemma): `HF_TOKEN` set on the GPU host, for an
  account that accepted the model's license on huggingface.co.

## Step 1: Plan

```bash
python3 scripts/plan.py --model <HF_ID> [--host user@hostname]
```

Run it from this skill's directory. Omit `--model` to get the default,
`Qwen/Qwen3.5-9B`. Add `--gpus 0,1` or `--tp N` only if the user asked for
specific GPUs or tensor parallelism, `--port N` for a port other than 8000, and
`--image <tag>` only if the user insists on an image.

Do not write the `docker run` command yourself: `plan.py` reads the recipe,
GPU overrides, and blocklist data and builds it. Its JSON has a `status`:

- **`refuse`**: the model can't be served here (not an LLM, needs vLLM
  nightly, NVIDIA-only FP4, or the id doesn't exist). Tell the user the
  `reason` and the `suggestion`. Do not launch anything.
- **`does_not_fit`**: not enough VRAM even when quantized. Tell the user the
  `reason`, show what was `tried`, and give the `suggestion`. Do not launch.
- **`error`**: `plan.py` couldn't finish (GPU detection failed, or Hugging
  Face was unreachable or rate-limited). Tell the user the `reason` and
  `suggestion`; retry once after a minute if it was a rate limit. Do not
  write a command yourself.
- **`ok`**: continue to Step 2. `command` is the exact command;
  `run_command` is the same command wrapped to run on `--host`.

Always pass every entry in `warnings` on to the user. They cover things like
a substituted quantized checkpoint, a capped context length, a gated model, an
image that is too old, or GPUs the user asked for that don't exist.

## Step 2: Confirm with the user

Show a short summary from the JSON and wait for the user's go-ahead (skip the
wait only if the user already approved launching):

- Model (`model_id`) and precision (`precision`), and why if it differs from
  what was asked
- GPU (`gpu`), tensor parallel (`tp`), max context (`max_model_len`), port (`port`)
- The `command`, in a code block
- Every warning

## Step 3: Launch and verify

1. Run `run_command` exactly as printed (it already includes `ssh` for a
   remote host). If a container with the same name exists from an earlier
   attempt, remove it first with `docker rm -f <container_name>`. Never stop
   or remove containers you didn't create.
2. Poll until healthy (a 503 while loading is normal; on a remote host, run
   this over `ssh`):
   ```bash
   while docker inspect --format='{{.State.Running}}' <container_name> 2>/dev/null | grep -q true; do
     curl -sf http://localhost:<port>/health && echo READY && break
     sleep 30
   done
   ```
   For models under 100 GB this takes 2-5 minutes; use a 10-minute timeout.
   For larger models run the loop in the background and check back.
   If the container exits, read `docker logs <container_name> 2>&1 | tail -50`
   and see Troubleshooting.
3. Warm up (the first request compiles HIP kernels, about 40 seconds):
   ```bash
   curl -s http://localhost:<port>/v1/chat/completions -H "Content-Type: application/json" \
     -d '{"model":"<model_id>","messages":[{"role":"user","content":"say hi"}],"max_tokens":5}'
   ```
4. Report the connection details: model, base URL `http://<host>:<port>/v1`,
   API key (none), tensor parallel, max context, GPU, and a ready-to-run
   `curl` example with those values.

## Troubleshooting

- **OOM while capturing HIP graphs**: add `--enforce-eager` to the vLLM args
  and relaunch.
- **"Engine core initialization failed"**: check the first 50 lines of the
  logs. Usual causes: gated model without an accepted license, an image older
  than the model needs, OOM while loading weights.
- **Segfault on MI300X / MI325X**: `VLLM_ROCM_USE_AITER_FP4BMM` must be `0`
  on gfx942; `plan.py` sets it.
- **No GPUs visible**: `CUDA_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES` set to
  an empty string hides every GPU. Unset it; never pass it empty to Docker.
- **`/dev/kfd` permission denied**: `sudo usermod -aG video,render $USER`, then
  log in again.
- **Environment problems** (Docker, devices, NUMA): run
  `python3 scripts/validate.py --auto-fix [--host user@hostname]`.

## Reference

Precision support, Docker flags, known quirks, and the manual planning
procedure (if `plan.py` itself fails): [reference.md](reference.md)
