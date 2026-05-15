---
name: apu-memory-tuner
description: >-
  Inspects and tunes the shared-vs-dedicated memory split on AMD Ryzen APUs
  with unified memory (UMA) so larger LLMs and image-gen models fit on the
  iGPU, or so reserved GPU memory is returned to the CPU. Use when the user
  mentions Ryzen AI, Strix Halo / Strix Point / Krackan / Phoenix / Hawk
  Point, Ryzen AI Max, gfx1150 / gfx1151 / gfx1152, integrated Radeon, iGPU
  memory, UMA Frame Buffer Size, AMD Variable Graphics Memory, VGM, GTT,
  GART, TTM, pages_limit, amd-ttm, amd-debug-tools, "shared GPU memory",
  "dedicated GPU memory", carve-out, "not enough VRAM", "out of VRAM",
  "GPU OOM", llama.cpp on iGPU, ROCm on APU; or asks how much memory the
  iGPU can use, how to give the iGPU more memory, how to balance memory
  between CPU and GPU on UMA, or how to change the BIOS UMA reservation.
  Read-only diagnostics work everywhere; tuning runs automatically on Linux
  via `amd-ttm` and prints guided BIOS steps on Windows. Do not use for
  discrete Radeon cards, Intel iGPUs, or Apple Silicon -- it is APU-only.
---

# APU Memory Tuner

Help the user inspect and tune the shared-vs-dedicated memory split on AMD
Ryzen APUs with unified memory architecture (UMA). The user states intent
("I want to run a 70B model", "I just want my games to be smooth"); this
skill picks the numbers, explains the trade-offs, and applies the change
where it can.

## What's actually going on

On a UMA APU (Strix Halo, Strix Point, Krackan, Phoenix, Hawk Point, etc.)
the CPU and integrated Radeon share **one physical pool of DRAM**. There is
no separate VRAM chip. Two logical knobs cut that pool up:

| Knob | Where set | What it does | Reversible? |
|---|---|---|---|
| **Dedicated VRAM** (a.k.a. UMA Frame Buffer / carve-out / GART) | BIOS, requires reboot | Permanently reserves DRAM for the GPU. CPU can't see it. | Only via BIOS. |
| **Shared GPU memory** (a.k.a. GTT) | Kernel/driver-managed | Dynamic, OS-reclaimable cap on how much system RAM the GPU may map at a time. | Yes; Linux: `amd-ttm`. Windows: no user knob. |

Key insight the rest of this skill rests on: **"VRAM" and shared RAM run at
the same speed on UMA**, because they're the same DRAM. So the right default
for AI workloads is small VRAM + large GTT, not the other way around.

## When to use

Use this skill when the user wants to:

- Run a model that doesn't fit ("out of VRAM", "GPU OOM" on an iGPU).
- Inspect the current memory split before changing anything.
- Move the slider toward more shared memory (LLMs, large image gen) or
  toward more dedicated VRAM (gaming, predictable framebuffer).
- Revert any prior changes.

Do not use it for: discrete Radeon cards (no GTT/UMA on those), Apple
Silicon (different architecture entirely), or NVIDIA/Intel GPUs.

## Prerequisites

State these to the user up front so a missing one doesn't surface as a
mystery script error halfway through:

- **GPU architecture**: AMD APU with integrated Radeon. Officially tuned
  for RDNA3.5 (`gfx1150` / `gfx1151` / `gfx1152`); RDNA3 / RDNA2 APUs work
  but with conservative profile numbers.
- **OS**: Linux (kernel + `amd-ttm` path) or Windows (guided BIOS path).
  macOS is not supported.
- **Linux kernel**: see the live AMD matrix linked from `reference.md`.
  The detection script enforces a conservative floor (mainline 6.18.4 /
  Ubuntu HWE 6.17.0 / Ubuntu OEM 6.14.0).
- **Linux extras**: `pipx install amd-debug-tools` for the `amd-ttm` CLI.
  The skill prints this command but never installs it silently. Reading
  the BIOS carve-out from `dmesg` typically requires `sudo`.
- **Windows extras**: AMD Adrenalin Software 25.x or newer if the user
  wants the Variable Graphics Memory slider as an alternative to BIOS.
- **Reboot tolerance**: any tuning change requires a reboot. Don't propose
  this skill mid-workload.

Silent footguns to surface when relevant:

- `HSA_OVERRIDE_GFX_VERSION` — users running ROCm/PyTorch on an APU often
  set this to convince ROCm the iGPU is a supported target. It does not
  affect memory tuning, but if the user reports `HIP error: invalid
  device` after raising GTT, this env var is usually the cause, not the
  GTT change.
- Windows `Win32_VideoController.AdapterRAM` is a 32-bit field capped at
  4 GiB. If the user's reported "dedicated VRAM" is exactly 4096 MB, that's
  the WDDM truncation, not the real BIOS reservation. The real value lives
  in Task Manager > Performance > GPU.

## The four-step flow

Run these in order. Each one is read-only until step 4.

```
[ ] 1. Detect platform and support level
[ ] 2. Show current configuration
[ ] 3. Pick a profile from the user's intent
[ ] 4. Apply (Linux) or print BIOS guidance (Windows); verify after reboot
```

### Step 1: detect platform

```bash
python scripts/detect_platform.py
```

Add `--json` for parseable output. Exit codes:

| Exit | Meaning | Next action |
|---|---|---|
| 0 | Supported AMD APU. | Continue to Step 2. |
| 2 | Wrong hardware (not an AMD APU, or unclassifiable). | Stop. Tell the user this skill can't help them. |
| 3 | AMD APU but a hard prerequisite is missing (Linux kernel too old). | Stop. Tell the user the prereq and stop. Do not attempt to upgrade the kernel from this skill. |

The script reports the OS, CPU, GPU LLVM target (e.g. `gfx1151`), the
generation bucket (RDNA3.5 / RDNA3 / RDNA2 / older), total RAM, and on
Linux the kernel version vs. the minimums in the AMD doc.

### Step 2: show current configuration

```bash
python scripts/show_config.py
```

Reports current dedicated VRAM, current shared-GPU cap, total RAM, and on
Linux the raw `pages_limit` value plus a `rocminfo` sanity-check of what
the runtime actually sees. Note any messages it prints — Linux often needs
`sudo` for the dmesg read of the BIOS carve-out, and Windows' `AdapterRAM`
field is capped at 4 GiB by WDDM (real value lives in Task Manager).

### Step 3: pick a profile

Ask the user what they want, in workload terms, not numbers. Map their
answer to one of these:

| Profile | What it does | Use when the user says... |
|---|---|---|
| `large-models` **(default)** | GTT to ~75% of RAM, BIOS VRAM at the floor (0.5 GB). | "Run a big model", "fit Llama 70B", "I keep getting OOM on the iGPU", "give the GPU as much memory as possible". |
| `balanced` | GTT at the kernel default (~50%), 1 GB BIOS VRAM. | "I just do mixed dev work", "back to defaults", "don't waste RAM on the GPU". |
| `graphics` | GTT at default; BIOS VRAM raised to the larger of 8 GB or 25% of RAM. | "I'm gaming", "I want a predictable framebuffer", "stuttering in games". |
| `reset` | Revert all changes this skill made. | "Undo it", "go back to stock". |
| `custom` | Use the explicit `--gtt-gb` / `--vram-gb` the user passed. | "I want exactly N GB". |

**Default**: if the user is here at all and didn't specify, they almost
always want `large-models` -- that's the only profile that meaningfully
changes the experience for the workload that brought them here (running a
model that didn't fit). Use it unless they explicitly said gaming, said
they want defaults, or said an exact number.

If you still can't tell, use the `AskQuestion` tool with the five options
above labeled in plain English; do not invent a sixth.

### Step 4: apply or guide

```bash
python scripts/apply_profile.py --profile <choice>
```

Add `--dry-run` first if the user wants to see the planned change before
committing.

What happens on each OS:

- **Linux**: writes the new GTT cap via `amd-ttm --set <N>` (which persists
  to `/etc/modprobe.d/ttm.conf`). Reboot is required; the script tells the
  user but never reboots automatically. If `amd-ttm` is missing, the script
  exits with a clear install hint (`pipx install amd-debug-tools`) — do not
  install it yourself without confirming with the user.
- **Windows**: prints step-by-step BIOS instructions for the UMA Frame
  Buffer Size, plus a note about AMD Adrenalin's "Variable Graphics Memory"
  slider as an alternative on supported laptops. Nothing is written to disk
  or registry. The script never modifies BIOS for the user.

Then re-run `python scripts/show_config.py` after reboot to verify.

## OS-specific reality check

| Capability | Linux | Windows |
|---|---|---|
| Inspect dedicated VRAM | Yes (`dmesg`/`journalctl`, may need sudo) | Yes (Task Manager / dxdiag; AdapterRAM is truncated) |
| Inspect shared cap | Yes (`/sys/module/ttm/parameters/pages_limit`) | Yes (dxdiag "Shared Memory") |
| Change shared cap automatically | Yes (`amd-ttm`) | **No** — WDDM-managed, not user-tunable |
| Change dedicated VRAM automatically | No (BIOS only) | No (BIOS only; VGM via Adrenalin is the closest UI) |

Net effect: on Windows, raising the BIOS UMA Frame Buffer Size is the only
real way to give the GPU more memory. On Linux you almost always want to
*lower* BIOS VRAM and raise GTT instead.

## Safety rules

- Never auto-reboot. Always tell the user the reboot is needed and let them
  do it.
- Never touch BIOS programmatically. The script prints instructions; the
  user navigates the firmware menu.
- Never silently install `amd-debug-tools`. Print the `pipx install` line
  and ask before running it.
- Never set a profile whose validation fails. The script refuses GTT
  targets above 95% of RAM or VRAM targets above 50% of RAM.
- Never claim a change took effect before the user has rebooted and
  re-verified with `show_config.py`.

## Verification checklist

Mark this skill complete only when **all** are true:

- [ ] `python scripts/detect_platform.py` exits 0.
- [ ] `python scripts/show_config.py` reports the *new* values after the
      reboot following Step 4.
- [ ] The user has tried the workload that motivated the change (loaded
      the model, launched the game) and confirmed the new headroom helps.
- [ ] On Linux, `cat /sys/module/ttm/parameters/pages_limit` matches what
      `apply_profile.py` reported.

If any box is unchecked the change either didn't take effect or the user
hasn't validated it yet — say so out loud rather than declaring success.

## Reference

For the full glossary, the link to AMD's authoritative kernel-version /
ROCm-compatibility matrix, per-OEM BIOS notes, profile math, and
troubleshooting, see [reference.md](reference.md).
