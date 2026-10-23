# serving-llms-on-instinct -- Reference

## Table of Contents
1. [Precision Compatibility](#precision-compatibility)
2. [Docker Flags](#docker-flags)
3. [Known Quirks](#known-quirks)
4. [Manual Planning](#manual-planning)

---

## Precision Compatibility

| Format | gfx942 (MI300X) | gfx950 (MI350X) | Notes |
|---|---|---|---|
| BF16 / FP16 | Native | Native | Default for all models |
| FP8 (FNUZ) | Native | Emulated | MI300X uses E4M3FNUZ dialect |
| FP8 (OCP) | Emulated | Native | MI350X uses E4M3FN (OCP standard) |
| INT8 | Native | Native | |
| MXFP4 | Emulated | Native | On gfx942: compute dequants to BF16, weights stay compressed |
| MXFP6 | Emulated | Native | On gfx942: compute dequants to BF16, weights stay compressed |
| NVFP4 | Not supported | Not supported | NVIDIA-specific, no dequant kernel on ROCm |

"Emulated" means compute is handled via dequantization to BF16 during matmul.
Weights stay in their compressed format in VRAM, so quantized models still
benefit from reduced memory. vLLM auto-converts between FP8 dialects
(FNUZ/OCP) transparently. NVFP4 models (e.g. `nvidia/*-NVFP4`) will not
load on AMD GPUs -- use FP8 or MXFP4 alternatives instead.

### VRAM Estimation

Use `scripts/estimate_vram.py` to estimate weight memory and KV cache
requirements from the HuggingFace Hub API (no model download):
```bash
python3 scripts/estimate_vram.py --model-id <HF_ID> --vram-gb <per_gpu_vram>
```
Returns JSON with `weight_memory_gb`, `kv_cache_bytes_per_token`,
achievable context length, and fit status. The script reserves ~4 GB for
vLLM's runtime overhead (activation profiling, HIP graph capture, internal
buffers). Weight memory is derived from safetensors metadata (tested:
GPT-OSS-120B reports 65 GB, vLLM logs show 68.7 GB actual load on MI300X).
KV cache per token is calculated from the model's `config.json` architecture
parameters. MLA models (DeepSeek-R1/V3) are detected and use their compressed
KV dimensions.

---

## Docker Flags

### Mandatory (all AMD Instinct)

| Flag | Why |
|---|---|
| `--group-add=video` | amdgpu exposes GPUs to the `video` group |
| `--group-add=render` | GPU render nodes require the `render` group on many hosts |
| `--cap-add=SYS_PTRACE` | ROCm JIT compilation requires ptrace |
| `--security-opt seccomp=unconfined` | ROCm mmap variants blocked by default seccomp |
| `--device /dev/kfd` | Kernel Fusion Driver -- primary GPU access |
| `--device /dev/dri` | Render nodes for GPU command submission |
| `--ipc=host` | ROCm shared memory needs host IPC namespace |

### Docker image

`vllm/vllm-openai-rocm:<tag>` -- tag is auto-resolved from Docker Hub
during recipe sync. Includes gfx942 and gfx950 kernels.
Do NOT use `vllm/vllm-openai` (CUDA-only).

### GPU visibility

| Variable | Rule |
|---|---|
| `CUDA_VISIBLE_DEVICES` | ROCm maps this to `HIP_VISIBLE_DEVICES`. Works with explicit indices (e.g. `0,1`). **Never set to empty string** -- hides all GPUs. |
| `HIP_VISIBLE_DEVICES` | Canonical AMD variable. Use to restrict visible GPUs by index on multi-GPU hosts. |

---

## Known Quirks

**vLLM #34641 -- FP4BMM crash on gfx942**
Segfault or illegal instruction during model warmup on MI300X/MI325X/MI300A.
Triggered when `VLLM_ROCM_USE_AITER_FP4BMM=1` on gfx942.
Fix: always set `VLLM_ROCM_USE_AITER_FP4BMM=0` on gfx942.
This is set correctly in `data/gpu_overrides.json` for gfx942.

**CUDA_VISIBLE_DEVICES empty string**
ROCm maps `CUDA_VISIBLE_DEVICES` to `HIP_VISIBLE_DEVICES`. Setting it to an
empty string hides all GPUs. Setting it to explicit indices (e.g. `0,1`) works
correctly. If the host has it set to empty, unset it: `unset CUDA_VISIBLE_DEVICES`.
Do not pass `--env CUDA_VISIBLE_DEVICES=` (empty) into Docker.

**NUMA balancing latency spikes**
`/proc/sys/kernel/numa_balancing=1` periodically migrates pages between NUMA
nodes. For GPU workloads this causes latency spikes as GPU DMA must follow
moved pages. Disable: `echo 0 | sudo tee /proc/sys/kernel/numa_balancing`
Non-persistent -- resets on reboot.

**First-token warmup delay**
vLLM compiles and caches HIP kernels on first use per input shape.
First inference after model load: ~40-45 seconds on gfx942.
Send a warmup request immediately after `/health` returns 200.

**"Engine core initialization failed"**
This opaque error covers many root causes. Check early container logs
(`docker logs <name> 2>&1 | head -50`). Common causes:
- Gated model: HF license not accepted (not just missing token)
- Unsupported architecture on this vLLM version
- OOM during weight loading
- Missing `--trust-remote-code` for custom model architectures
- vLLM version too old (check `min_vllm_version` in the recipe)

---

## Manual Planning

Use this only if `scripts/plan.py` fails or can't express what the user needs. It is
the procedure `plan.py` automates.

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
   Then apply `gpu_configs.<gfx_version>.model_overrides`: if the model ID
   contains an entry's `match` string, add its `env_set` vars last, so they
   win over the recipe (e.g. `VLLM_ROCM_USE_AITER_MOE=0` for gpt-oss on gfx950).
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
