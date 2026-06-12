# serving-llms-on-instinct -- Reference

## Table of Contents
1. [Precision Compatibility](#precision-compatibility)
2. [Docker Flags](#docker-flags)
3. [Known Quirks](#known-quirks)

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
