# APU Memory Tuner — Reference

Detailed background for the `apu-memory-tuner` skill. Read this only when
the four-step flow in `SKILL.md` doesn't cover a decision.

## Contents

- [Glossary](#glossary)
- [Linux kernel version matrix](#linux-kernel-version-matrix)
- [Installing amd-debug-tools](#installing-amd-debug-tools)
- [Windows BIOS hints by OEM](#windows-bios-hints-by-oem)
- [AMD Adrenalin Variable Graphics Memory (VGM)](#amd-adrenalin-variable-graphics-memory-vgm)
- [Profile math](#profile-math)
- [Troubleshooting](#troubleshooting)
- [Source](#source)

---

## Glossary

The same physical pool of DRAM is described by half a dozen overlapping
terms in firmware menus, AMD docs, and forum posts. They are NOT all the
same thing.

| Term | What it actually means |
|---|---|
| **DRAM** | The physical LPDDR5X / DDR5 sticks. There is exactly one pool on a UMA APU. |
| **VRAM** | Marketing-speak for "memory the GPU sees". On UMA this is a *logical* concept — there is no separate VRAM chip. |
| **Carve-out** / **UMA Frame Buffer** / **Dedicated GPU Memory** | The DRAM region the BIOS permanently reserves for the GPU at boot. Static, GPU-only, set in firmware, requires reboot. |
| **GART** (Graphics Address Remapping Table) | Driver-internal mapping of system RAM into the kernel-mode GPU address space. Small, used for driver bookkeeping; not the user-tunable knob. |
| **GTT** (Graphics Translation Table) | The dynamic, OS-reclaimable pool of system RAM that user processes (PyTorch, llama.cpp, games) can map into GPU virtual addresses. **This is the knob you actually want for AI workloads.** |
| **TTM** (Translation Table Manager) | The Linux kernel subsystem that enforces the GTT cap. The cap lives at `/sys/module/ttm/parameters/pages_limit`. |
| **Shared GPU Memory** (Windows Task Manager term) | Windows' name for the same idea as GTT. Managed by WDDM, typically capped at ~50% of RAM, **not** user-tunable. |
| **Variable Graphics Memory (VGM)** | An AMD Adrenalin feature on supported laptops that resizes the carve-out without entering BIOS. Behaves like a soft BIOS reservation. |

When the AMD doc says "VRAM", it means the carve-out. When it says "GTT",
it means the dynamic shared pool. This skill follows the same convention.

---

## Linux kernel version matrix

Strix Halo (gfx1151) requires Linux KFD fixes that update internal queue
limits and memory checks. Without them, GPU compute workloads may fail to
initialize or behave unpredictably regardless of how you set GTT.

Minimum kernel versions (per the source AMD doc):

| Distribution | Minimum kernel |
|---|---|
| Ubuntu 24.04 HWE | `6.17.0-19.19~24.04.2` |
| Ubuntu 24.04 OEM | `6.14.0-1018` |
| Ubuntu 26.04 (generic) | shipped |
| Fedora 43 | shipped |
| Arch Linux 2026.02.01 | shipped |
| All other distributions | mainline `6.18.4` or newer |

ROCm release compatibility (from the AMD doc):

| ROCm release | Ubuntu 24.04 HWE / OEM / Ubuntu 26.04 | Other distros >= 6.18.4 | Other distros < 6.18.4 |
|---|---|---|---|
| 7.11.0 / 7.12.0 | Stable | Stable | Unstable |
| 7.10.0 / 7.9.0 | Unsupported | Unsupported | Unstable |
| 7.2.1 | Stable | Stable | Unstable |
| 7.2.0 | Stable | Stable | Unsupported |
| 7.1.x | Unsupported | Unsupported | Unstable |
| 6.4.x | Unsupported | Unsupported | Unstable |

`detect_platform.py` only enforces the **mainline 6.18.4 / Ubuntu HWE
6.17.0 / Ubuntu OEM 6.14.0** floor. Anything below that is reported as
`kernel_supported: false` and the skill stops.

For RDNA3 / RDNA2 APUs (gfx1103, gfx1036, etc.) the kernel requirements
are looser — any reasonably recent (6.x) kernel works for the GTT knob,
because the new KFD fixes are Strix Halo specific. The detection script
does not enforce a floor for those generations.

---

## Installing amd-debug-tools

`amd-ttm` ships in the [`amd-debug-tools`](https://pypi.org/project/amd-debug-tools/)
PyPI package, maintained by AMD.

```bash
sudo apt install pipx
pipx ensurepath
pipx install amd-debug-tools
```

After install, the helpers `amd-ttm`, `amd-pstate`, etc. are on the user's
PATH. `amd-ttm --set <N>` writes a single-line modprobe override:

```
# /etc/modprobe.d/ttm.conf
options ttm pages_limit=<N_in_pages>
```

The kernel reads this file when the `ttm` module loads, which happens
during early boot before the amdgpu driver initializes. That's why a
reboot is required — the live `pages_limit` cannot be raised on a running
kernel.

`amd-ttm --clear` removes the file. `amd-ttm` (no args) prints the current
limit in pages and GB.

---

## Windows BIOS hints by OEM

The setting is the same on every AMD platform; only the menu path differs.
General navigation tips:

| OEM | BIOS key | Typical menu path |
|---|---|---|
| Asus / ROG | F2 or Del | Advanced > AMD CBS > NBIO Common Options > GFX Configuration > **UMA Frame Buffer Size** |
| HP | F10 | Advanced > Built-in Device Options > **Video Memory** (limited values) |
| Lenovo (consumer) | F1 / F2 / Enter then F1 | Configuration / Devices > **Video Memory** |
| Lenovo ThinkPad | F1 | Config > Display > **Total Graphics Memory** |
| Framework (AMD) | F2 | Advanced > AMD CBS > NBIO Common Options > GFX Configuration > **UMA Frame Buffer Size** (Strix Halo / Ryzen AI Max) |
| MSI | Del | Advanced > AMD CBS > **UMA Frame Buffer Size** |
| Dell (consumer) | F2 | Video > **Integrated Graphics Memory** |
| ASRock | F2 | Advanced > AMD CBS > NBIO > **UMA Frame Buffer Size** |

Notes:

- Some OEM BIOSes hide the slider behind an Advanced Mode toggle (Asus
  ROG: F7 to switch from EZ Mode to Advanced).
- Maximum value depends on installed RAM and OEM caps. Strix Halo systems
  with 64–128 GB usually allow up to half of installed RAM as carve-out.
- A few business laptops (esp. older Dell Latitude / HP EliteBook) hide
  the setting entirely. On those machines, the carve-out is fixed at the
  OEM default and your only option is AMD Adrenalin VGM (if supported).

---

## AMD Adrenalin Variable Graphics Memory (VGM)

On supported AMD APU laptops (most Ryzen AI 300 series, Ryzen AI Max
series), AMD Adrenalin Software exposes a **Variable Graphics Memory**
slider:

```
Adrenalin > System > Hardware > Variable Graphics Memory
```

VGM behaves much like a BIOS carve-out — it permanently reserves DRAM for
the GPU until you change it back — but you don't have to enter firmware.
Trade-offs vs. setting it in BIOS:

- **Pro:** No reboot to change it (some configs); no BIOS dance.
- **Pro:** Survives Windows updates better than fragile BIOS profiles.
- **Con:** Not all AMD APU SKUs expose VGM; OEM has to enable it.
- **Con:** Only goes up to a per-OEM cap, sometimes lower than the BIOS
  cap.

`detect_platform.py` records the Adrenalin driver version on Windows so
you can flag VGM as a likely-available alternative. It cannot tell whether
the OEM enabled the feature; the user has to open Adrenalin to confirm.

---

## Profile math

Every profile resolves to concrete (GTT, VRAM) numbers based on total RAM.
`scripts/apply_profile.py` does this in `resolve_profile()`. The math:

| Profile | GTT | VRAM | Notes |
|---|---|---|---|
| `large-models` | `0.75 * total_ram_gb` | `0.5` GB | Floors at 1 GB GTT, 0.5 GB VRAM. |
| `balanced` | `0.50 * total_ram_gb` | `1.0` GB | Mirrors kernel default. |
| `graphics` | `0.50 * total_ram_gb` | `max(8, 0.25 * total_ram_gb)` GB | Reserves enough for AAA framebuffers. |
| `reset` | `None` | `None` | Clears amd-ttm config (Linux); BIOS instructions (Windows). |
| `custom` | `--gtt-gb` | `--vram-gb` | Whatever the user asked for. |

Worked examples (in GB):

| Total RAM | large-models GTT | balanced GTT | graphics VRAM |
|---|---|---|---|
| 16 | 12 | 8 | 8 |
| 32 | 24 | 16 | 8 |
| 64 | 48 | 32 | 16 |
| 96 | 72 | 48 | 24 |
| 128 | 96 | 64 | 32 |

Validation rules the script enforces (refuses to apply on failure):

- `gtt_gb >= 1`
- `vram_gb >= 0.5`
- `gtt_gb <= 0.95 * total_ram_gb` (leaves 5% for kernel + CPU)
- `vram_gb <= 0.50 * total_ram_gb` (no more than half the machine for the
  GPU carve-out)

To override these, the user must use `--profile custom` and explicitly
pass numbers — there is no `--force` flag.

---

## Troubleshooting

### `amd-ttm: command not found`

`amd-debug-tools` is not installed. Install with:

```bash
pipx install amd-debug-tools
```

If `pipx` itself is missing: `sudo apt install pipx && pipx ensurepath`.

### GTT didn't change after reboot

Most common cause: `amd-ttm --set` succeeded but a *different* modprobe
override is taking precedence. Check:

```bash
cat /etc/modprobe.d/ttm.conf
ls /etc/modprobe.d/
sudo update-initramfs -u   # if you have a custom initramfs
cat /sys/module/ttm/parameters/pages_limit
```

If the live `pages_limit` doesn't match `ttm.conf`, the file isn't being
applied — usually because another conf file under `/etc/modprobe.d/`
overrides it, or the initramfs hasn't been regenerated.

Less common: the kernel silently capped the value at total system RAM.
The TTM subsystem will not let you map more pages than physically exist.

### Windows shared GPU memory looks fixed at 50%

That's WDDM. There is no user-facing knob to raise it. Your only lever is
to raise the BIOS UMA Frame Buffer Size (carve-out) instead. This is the
opposite of the Linux advice and is not a workaround you'd choose for AI
workloads — it permanently steals DRAM from the CPU. But on Windows it's
the only knob that exists.

### `dmesg` returns "Operation not permitted" on Linux

Recent kernels restrict dmesg to root. Either:

```bash
sudo dmesg | grep -i 'amdgpu .*VRAM'
```

or grant the current user access:

```bash
sudo sysctl kernel.dmesg_restrict=0
```

### `rocminfo` says "ROCk module is NOT loaded"

The `amdkfd` kernel module isn't loaded. On most distros it auto-loads
with the `amdgpu` driver; if it doesn't, ROCm itself isn't installed
and the runtime cross-check is unavailable. The TTM/GTT knob still works
without ROCm — it's a kernel-level setting, not a userspace one.

### "Can I tune my discrete RX 7900 XTX with this skill?"

No. Discrete GPUs have their own VRAM and don't use the GART/GTT model.
This skill is APU-only. `detect_platform.py` will report the discrete
card but exit `2` ("not an APU").

### "Can I tune my Intel iGPU or Apple Silicon with this skill?"

No. Intel iGPUs use a different (i915/Xe) memory model, and Apple Silicon
UMA is managed entirely by macOS with no user-facing knobs. This skill
only knows AMD.

---

## Source

The Linux mechanics, kernel version requirements, and the recommendation
to keep BIOS VRAM small + GTT large come from the AMD ROCm documentation:

<https://rocm.docs.amd.com/en/latest/how-to/system-optimization/rdna3-5.html>

The Windows mechanics are documented across AMD Adrenalin release notes
and the WDDM developer guide on docs.microsoft.com. The OEM BIOS paths in
the table above are observed from public BIOS manuals as of 2026.
