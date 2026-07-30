---
name: picking-amd-gpu-backend
description: >-
  Chooses and installs the right GPU compute backend (vulkan, rocm, cpu) for
  local AI on an AMD GPU, then verifies the workload actually landed on the
  discrete GPU. Use when the user is setting up local image generation, local
  LLM inference, llama.cpp, stable-diffusion.cpp, or Lemonade on AMD hardware;
  when a ROCm or HIP install fails; when they mention TheRock, gfx1100/gfx1102/
  gfx1151, Radeon RX, eGPU, or an external GPU enclosure; when local AI is
  unexpectedly slow, falls back to CPU, or runs on the wrong GPU; or when they
  ask whether they need ROCm at all. Also use when generation fails above a
  certain resolution, or when a machine has both an integrated and a discrete
  GPU. Do not use for AMD Instinct data center parts (see
  serving-llms-on-instinct) or for NVIDIA hardware.
---

# Picking an AMD GPU backend

Consumer AMD GPUs have three backend options for local AI. Picking wrong costs
hours: ROCm currently fails to install on Windows, and several settings that
look like optimizations are large regressions.

**Default to Vulkan.** Reach for ROCm only on Linux, or after Vulkan is proven
working and you have a specific reason.

## Decision

```
Is this AMD Instinct (MI300X/MI325X/MI350X)?
  -> yes: stop, use the serving-llms-on-instinct skill instead.

Is the OS Windows?
  -> yes, and the target is native Windows: use vulkan. The Lemonade/TheRock
     ROCm backend install currently fails there (see below).
  -> yes, and the target is WSL2: rocm is a real option, not just a fallback
     — see the WSL2 note below. Still confirm the specific GPU model, since
     package-name guessing is the main way this goes wrong.
  -> no (bare-metal Linux): try rocm; fall back to vulkan if install or
     runtime fails. [UNVERIFIED ON BARE METAL -- see note]

No GPU detected, or GPU unsupported?
  -> use cpu, and tell the user the expected slowdown (~20x for images).
```

> **Bare-metal Linux is unverified; WSL2 is verified working — for LLM
> inference. Image generation is verified too, and is slower there.** Every
> measurement in the Windows branch is measured on real hardware. WSL2 ROCm
> was also measured directly — a real HIP kernel compiled and ran correctly,
> llama.cpp built with the HIP backend matched Vulkan's throughput on the
> same 8B model (36.4 vs 33-34 tok/s), but stable-diffusion.cpp built with
> the HIP backend (`SD_HIPBLAS`) generated a real, visually-checked image
> **~3-4x slower** than the Vulkan warm-server baseline (10.1s vs 2.47s,
> SD-Turbo 512²) — all on the same Radeon RX 7600M XT (gfx1102). Don't
> assume ROCm/WSL2 is a blanket win just because LLM throughput matched;
> check the specific workload. Full repro steps, and every wrong turn taken
> to get there (including two Windows-host-to-WSL2 scripting footguns that
> have nothing to do with the GPU), are at
> <https://github.com/gucciwong/amd-local-ai-bench/blob/main/docs/rocm-on-wsl2.md>.
>
> Bare-metal Linux (no WSL) follows upstream documentation only — no one has
> run this skill's guidance on real bare-metal Linux. Say so when recommending
> it there, and prefer vulkan as the fallback the moment anything fails rather
> than debugging a ROCm install.
>
> **If a ROCm install on WSL2 looks like it's failing, don't conclude the GPU
> is unsupported before checking these two things** — both cost real time in
> the referenced session:
> - `amdgpu-install --usecase=wsl` errors with "not supported or invalid"
>   because that usecase name does not exist in current amdgpu-install
>   builds, not because WSL itself is unsupported. The actual bridge is a
>   separate package, `amdrocm-wsl`, installed alongside the versioned ROCm
>   package.
> - Per-architecture package names carry a ROCm version suffix
>   (`amdrocm7.14-gfx1102`, not `amdrocm-gfx1102`). Run
>   `apt-cache search amdrocm | grep <gfx-target>` before concluding a
>   package "doesn't exist" for a given GPU.

## Step 1: identify the hardware before installing anything

Never assume which GPU will be used. Machines with an integrated *and* a
discrete GPU are common, and the integrated one is often enumerated first.

```bash
lemonade backends            # which backends are installable for this hardware
```

A backend shown as `installable` means hardware filtering already accepts it.
`unsupported` messages name the reason (`Requires AMD XDNA 2 AMD NPU`,
`Unsupported GPU`, `Requires Linux`).

To see how the compute library enumerates devices — the order matters, because
index 0 is not necessarily the discrete GPU:

```bash
llama-server --list-devices
```

Look for `matrix cores`. On RDNA3/4 the discrete GPU reports `KHR_coopmat`;
integrated GPUs typically report `none`. Prefer the device with matrix cores.

## Step 2: install the backend

```bash
lemonade backends install <recipe>:vulkan
lemonade config set sdcpp.backend=vulkan
lemonade config set llamacpp.backend=vulkan
```

Vulkan installs in seconds because it ships with the AMD display driver.

### If the user insists on ROCm on Windows

Set expectations first: on Windows this currently downloads several GB and then
fails at extraction:

```
Error: TheRock extraction failed: bin directory not found
```

This is an upstream packaging bug, tracked in
[lemonade-sdk/lemonade#2722](https://github.com/lemonade-sdk/lemonade/issues/2722),
reproduced on more than one GPU architecture. `--force` does not help — it
bypasses hardware filtering, which is not the blocker. There is no CLI flag to
pin an alternate ROCm version.

Vulkan reaches the same matrix-core hardware, so the practical cost of skipping
ROCm on Windows is low.

## Step 3: verify the workload landed on the discrete GPU

A backend that "works" may still be running on the integrated GPU. Confirm by
watching dedicated VRAM rise when a model loads and fall when it unloads.

On Windows:

```powershell
Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory |
  Select-Object Name, DedicatedUsage, SharedUsage
```

The discrete GPU is the adapter with non-zero `DedicatedUsage`; integrated GPUs
are UMA and report 0 dedicated with large `SharedUsage`. Load a model, re-run,
and confirm the expected adapter grew.

If dedicated VRAM never moves, the work is on the wrong device or on CPU.

## Settings that look helpful and are not

State these before the user finds them:

| Setting | Reality |
|---|---|
| `enable_dgpu_gtt=true` | Lets the driver allocate from system memory. On a discrete GPU — especially over Thunderbolt — measured 58x slower image generation, and it does **not** raise the resolution ceiling. Keep `false`. |
| Splitting one job across two GPUs | `sd-cpp` accepts `--backend clip=...,vae=...,diffusion=...`. On a discrete + integrated pair this regressed at every resolution measured. Integrated GPUs lack matrix cores; cross-device transfer is pure overhead. |
| `max_loaded_models=2` | No benefit if the two models do not both fit in VRAM — they evict each other anyway. Shrink the models instead. |
| `GGML_VK_FORCE_MAX_ALLOCATION_SIZE` | Bounded near 4 GB by a uint32 limit, so it can only *lower* the cap. It cannot raise a 5 GiB per-allocation limit. |

## When generation fails above a certain resolution

If image generation fails abruptly past some size with:

```
ggml_vulkan: Device memory allocation of size 5368709120 failed
```

that is Vulkan's **per-allocation** cap (5 GiB here), not total VRAM. Adding
memory does not help — quantized models and system-memory borrowing were both
measured and neither moved the ceiling.

Recommend the upscale path instead of fighting it: generate small, then run a
pure upscale. This is dramatically faster than generating large directly, and
uses a fraction of the VRAM.

```bash
sd-cli -M upscale --upscale-model <esrgan.pth> -i small.png -o large.png
```

Note that `--hires` is *not* this: it runs a second full denoising pass, so it
costs about as much as generating at the target size.

## Before reporting any performance number

Two mistakes invalidate measurements on this hardware. Both are easy to repeat.

**Align arguments across execution paths.** A server may launch its worker with
flags the CLI does not default to (e.g. `--vae-tiling --diffusion-fa`). Comparing
the two without matching flags produces conclusions that point the wrong
direction. Capture what actually ran:

```powershell
Get-CimInstance Win32_Process -Filter "Name='sd-server.exe'" | Select-Object CommandLine
```

**Clear VRAM first.** Two processes each wanting 6-7 GB on an 8 GB card
regressed one workload by 34x. Unload resident models before benchmarking, and
restart the server between configurations.

```bash
lemonade unload <model>
```

## Reference

Measured numbers, the hardware they came from, and the negative results behind
each recommendation are in the `local-ai-use` skill's
[hardware-notes.md](../local-ai-use/hardware-notes.md).
