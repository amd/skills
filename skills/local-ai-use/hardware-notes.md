# Hardware notes: consumer AMD dGPUs on Windows

Measured findings from running this skill end-to-end on consumer AMD hardware.
Every number here came from a real run; negative results are included because
they are what save the next person time.

- [Contents](#contents)
- [ROCm backends currently fail to install on Windows](#rocm-backends-currently-fail-to-install-on-windows)
- [Silent footguns](#silent-footguns)
- [The resolution ceiling is an allocation limit, not a VRAM limit](#the-resolution-ceiling-is-an-allocation-limit-not-a-vram-limit)
- [Do not split one diffusion job across two GPUs](#do-not-split-one-diffusion-job-across-two-gpus)
- [Model selection on an 8 GB card](#model-selection-on-an-8-gb-card)
- [Fast path for high-resolution images](#fast-path-for-high-resolution-images)
- [Benchmarking methodology](#benchmarking-methodology)

## Contents

Reference hardware for all numbers below:

| | |
|---|---|
| GPU | Radeon RX 7600M XT 8 GB, gfx1102 (Navi 33, RDNA3) |
| Connection | Thunderbolt 3 eGPU enclosure |
| Host | Intel Core Ultra + Arc iGPU (dual-GPU system) |
| OS | Windows 11 x64, Adrenalin 32.0.21030.2001 |
| Lemonade | 11.5.0 |

## ROCm backends currently fail to install on Windows

`reference.md` recommends `lemonade backends install sd-cpp:rocm` for GPU
acceleration on RDNA3/4. On Windows this currently fails after downloading
5.2 GB (~18 minutes):

```
Error: TheRock extraction failed: bin directory not found
```

This is an upstream packaging bug, not a hardware limitation:

- AMD's Windows support matrix lists gfx1102 as fully supported in both the
  *Runtime* and *HIP SDK* columns.
- [lemonade-sdk/lemonade#2722](https://github.com/lemonade-sdk/lemonade/issues/2722)
  reports the same failure on gfx1151 — different silicon, same extraction step.
- `lemonade backends` reports `sd-cpp:rocm` as `installable`, so hardware
  filtering is not the blocker.
- There is no CLI flag to pin an alternate ROCm/TheRock version.

**Use `vulkan` instead.** It installs in under 10 seconds, ships with the
Adrenalin driver, and reaches the AMD matrix cores:

```
ggml_vulkan: 1 = AMD Radeon RX 7600M XT | bf16: 1 | matrix cores: KHR_coopmat
```

Measured on the reference hardware, image generation at 512x512, warm:

| Backend | Time |
|---|---|
| cpu | 50.58 s |
| **vulkan** | **2.47 s** |

Text inference through `llamacpp:vulkan` reaches 33-34 tok/s on an 8B model and
55.6 tok/s on Gemma-4-E2B.

## Silent footguns

### `enable_dgpu_gtt` — do not enable on a dGPU

This setting lets the driver satisfy allocations from system memory. Over an
external (Thunderbolt) link that is catastrophic, and it does **not** raise the
resolution ceiling it appears to target:

| `enable_dgpu_gtt` | 512x512 image, back to back |
|---|---|
| `false` | **2.73 s** |
| `true` | **158 s** |

Keep it `false`. The 58x regression is silent — nothing in the logs points at
this setting.

### VRAM contention between `sd-cli` and a resident `sd-server`

Running `sd-cli` directly while Lemonade still holds a model resident puts two
processes on the same card, each wanting 6-7 GB:

| State | SDXL 512x512 |
|---|---|
| Lemonade model still resident | 825.9 s |
| **VRAM cleared first** | **24.1 s** |

Unload before invoking the CLI directly:

```
lemonade unload <model>
```

This also corrupts benchmarks — see
[Benchmarking methodology](#benchmarking-methodology).

## The resolution ceiling is an allocation limit, not a VRAM limit

Above 2816x2816 on an 8 GB card, generation fails with:

```
ggml_vulkan: Device memory allocation of size 5368709120 failed
```

`5368709120` is exactly 5 GiB — Vulkan's per-allocation cap. Three approaches
that add total memory were measured and **none** moved the ceiling:

- `enable_dgpu_gtt=true` (borrow system memory) — still fails at 3072
- switching to a quantized model that frees 1.6 GB — still fails at 3072
- `GGML_VK_FORCE_MAX_ALLOCATION_SIZE` — capped near 4 GB by a uint32 bound, so
  it can only lower the limit

Total capacity and per-allocation size are different constraints. The error text
does not say "per-allocation", which makes this easy to misdiagnose.

## Do not split one diffusion job across two GPUs

`sd-cpp` accepts per-component device assignment
(`--backend clip=...,vae=...,diffusion=...`). On a dGPU + iGPU machine this is a
regression at every resolution measured:

| Resolution | All dGPU | dGPU diffusion + iGPU VAE/CLIP |
|---|---|---|
| 512 | 2.47 s | 7.55 s |
| 1024 | 10.93 s | 39.42 s |
| 2048 | 73.22 s | 199.17 s |

The iGPU has no matrix cores (`matrix cores: none`), so any work assigned to it
is slower, and cross-device transfers are pure overhead. Leave the iGPU to drive
the display so the dGPU's VRAM is not consumed by desktop compositing.

## Model selection on an 8 GB card

Text models:

| Model | Size | tok/s | TTFT |
|---|---|---|---|
| Gemma-4-E2B | 4.09 GB | **55.6** | 128 ms |
| DeepSeek-Qwen3-8B | 5.25 GB | 33-34 | 156-455 ms |
| Gemma-4-12B | 7.29 GB | 6.9-12.3 | 935-2850 ms |

12B fits but runs ~3x slower than 8B — treat it as the usable ceiling.

Image models:

| Model | 512 | 1024 | Peak VRAM |
|---|---|---|---|
| SD-Turbo-GGUF | 3.09 s | — | **2.52 GB** |
| SD-Turbo | 3.07 s | 10.93 s | 4.23 GB |
| SDXL-Turbo | 3.90 s | 15.96 s | 7.14 GB |
| SDXL-Base-1.0 | 14.81 s | 60.99 s | 7.14 GB |

The GGUF build of SD-Turbo is the same speed as the full model while using
1.6 GB less VRAM. SDXL-Turbo delivers SDXL-class quality at roughly a quarter of
SDXL-Base's time.

**Pairing matters more than either model alone.** Alternating chat and image
requests:

| Pair | chat | image | Round trip |
|---|---|---|---|
| 8B + SD-Turbo | 14.6-15.5 s | 3.7-4.2 s | ~19 s |
| **Gemma-4-E2B + SD-Turbo-GGUF** | **0.27-0.60 s** | **2.49-2.51 s** | **~3.0 s** |

The fast pair fits in 5.68 GB with 0.52 GB spill; the slow pair needs 9.3 GB and
spills 3.22 GB, so the two models keep evicting each other. On a capacity-limited
card, shrink the models until they coexist rather than splitting work across
devices.

Context length is cheaper than it looks: an 8B model at `ctx_size=32768` uses
6.39 GB versus 5.39 GB at the 4096 default. Quantizing the KV cache
(`-ctk q8_0 -ctv q8_0`, with or without `-fa on`) produced no measurable VRAM
change — llama.cpp fits parameters to a memory budget
(`common_fit_params: fitting params to free device memory`), so freed space is
reallocated rather than returned.

## Fast path for high-resolution images

Producing a 2048x2048 image three ways:

| Approach | Time | Peak VRAM |
|---|---|---|
| Generate 2048 directly | 483.64 s | 6.66 GB |
| `--hires --hires-scale 2.0` from 1024 | 536.04 s | 6.66 GB |
| **Generate 512, then ESRGAN 4x upscale** | **24.25 s** | **0.85 GB** |

`--hires` runs a second full denoising pass, so it costs about as much as
generating at the target size; its value is composition, not speed. A pure
upscale skips denoising entirely:

```
lemonade pull RealESRGAN-x4plus
sd-cli -M upscale --upscale-model <RealESRGAN_x4plus.pth> -i small.png -o large.png
```

For anime or line art, `RealESRGAN-x4plus-anime` is roughly twice as fast
(3.40 s vs 7.74 s) with cleaner edges. Note the weights file is named
`RealESRGAN_x4plus_anime_6B.pth` — underscores, not hyphens.

## Benchmarking methodology

Two mistakes made the first round of measurements on this hardware worthless.
Both are easy to repeat.

**Align arguments across execution paths before comparing them.** Lemonade
launches `sd-server` with `--vae-tiling --diffusion-fa`; invoking `sd-cli`
without them hits an out-of-memory path and produces numbers that point the
wrong direction. Capture what actually runs:

```powershell
Get-CimInstance Win32_Process -Filter "Name='sd-server.exe'" | Select-Object CommandLine
```

**Restart the server between configurations.** Residual resident models and
leftover experimental settings moved the same measurement between 2.5 s and
158 s across runs:

```powershell
Get-Process -Name "LemonadeServer" | Stop-Process -Force
Get-CimInstance Win32_Process -Filter "Name='sd-server.exe' OR Name='llama-server.exe'" |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Process "$env:LOCALAPPDATA/lemonade_server/bin/LemonadeServer.exe" -WindowStyle Hidden
# then poll http://127.0.0.1:13305/api/v1/health
```
