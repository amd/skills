# ROCm Doctor -- Reference

Detailed background for the `rocm-doctor` skill. Read this only when the
three-step flow in `SKILL.md` doesn't cover a decision.

## Contents

- [The known-misconfigurations catalog](#the-known-misconfigurations-catalog)
- [Silent-footgun environment variables](#silent-footgun-environment-variables)
- [Windows-specific footguns](#windows-specific-footguns)
- [Framework support matrix](#framework-support-matrix)
- [Device support, phased](#device-support-phased)
- [Live AMD compatibility matrices](#live-amd-compatibility-matrices)
- [Wheel index reference](#wheel-index-reference)
- [Upstream routing](#upstream-routing)
- [Why we do not auto-set HSA_OVERRIDE_GFX_VERSION](#why-we-do-not-auto-set-hsa_override_gfx_version)
- [Why WSL is out of scope](#why-wsl-is-out-of-scope)
- [Adjacent problem: matrices in hand-typed tables](#adjacent-problem-matrices-in-hand-typed-tables)

---

## The known-misconfigurations catalog

The closed list `diagnose.py` checks against. Each row maps to one
`fix-N-...` recipe in `apply_fix.py`. **If a user's symptom doesn't
match any of these, the skill must not speculate** -- it exits 1 and
prints the upstream tracker URL from `_route_when_no_match`.

This catalog grows over time. To add a new failure mode: add a
`check_N_*` function to `scripts/diagnose.py`, a `FixRecipe` with the
matching `fix-id` to `scripts/apply_fix.py`'s `RECIPES`, and a row to
the table below. The decision-tree contract -- score 0..100, emit the
recipe's `verify` command on a hit, exit 1 + route upstream on a miss --
stays the same regardless of catalog size.

| # | fix-id | OS | Failure pattern | Typical signal | Default fix |
|---|---|---|---|---|---|
| 1 | `fix-1-arch` | both | GPU `gfx` target not in framework's compiled arch list | `hipErrorNoBinaryForGpu`, `HIP error: invalid device function`, `HSA_STATUS_ERROR_INVALID_ISA`, `torch.cuda.get_arch_list()` missing the GPU's gfx | Reinstall the framework from a wheel index that ships kernels for the GPU's gfx (TheRock per-gfx wheels are the recommended fallback, and the only first-party option on Windows AMD). |
| 2 | `fix-2-unset-override` | both | `HSA_OVERRIDE_GFX_VERSION` set on a GPU that has a native wheel | Hangs, `amdgpu: page fault` in `dmesg`, `OUT_OF_REGISTERS` from the compiler | Linux: `unset HSA_OVERRIDE_GFX_VERSION` and remove from shell rc. Windows: `setx HSA_OVERRIDE_GFX_VERSION ""`, plus check the Machine env scope. |
| 3 | `fix-3-rocm-kernel` | linux | ROCm <-> distro/kernel forms an unsupported triple | `amdgpu-install` DKMS build fails; `amdgpu` not loaded after install | Cross-check the live AMD compatibility matrix; install matching HWE kernel; consider `--no-dkms`. |
| 4 | `fix-4-render-group` | linux | User not in `render` / `video` groups, or `/dev/kfd` group is wrong | `Unable to open /dev/kfd: Operation not permitted`; `rocminfo` works under `sudo` but not as user | `sudo usermod -a -G render,video "$USER"`; log out/in. |
| 5 | `fix-5-amdgpu-load` | linux | `amdgpu` kernel module not loaded or blacklisted | `rocminfo` says "ROCk module is NOT loaded"; `lsmod \| grep amdgpu` empty; blacklist line in `/etc/modprobe.d/*` | Remove blacklist; `update-initramfs -u`; `modprobe amdgpu`; check Secure Boot. |
| 6 | `fix-6-path` | both | ROCm/HIP binaries not on `PATH` after install | `rocminfo: command not found` (Linux) or `hipInfo.exe` not in `%PATH%` (Windows) immediately after a clean install | Linux: append `/opt/rocm/bin` to `PATH` in the shell rc. Windows: `setx PATH "%PATH%;C:\Program Files\AMD\ROCm\<ver>\bin"` and reopen the shell. |
| 7 | `fix-7-stale-repos` | linux | Stale / conflicting APT or DNF repos from prior installer runs | `404` on `repo.radeon.com`, "Release file not valid", mixed-version packages | Quarantine duplicate repo files in `/etc/apt/sources.list.d/`; re-run `apt update` cleanly. |
| 8 | `fix-8-wheel-rocm` | both | Framework wheel built for a different ROCm/HIP major than the system | Linux: `libamdhip64.so.X: cannot open shared object file`. Windows: `amdhip64_X.dll could not be found` / `DLL load failed`. | Reinstall the framework from the index matching the system ROCm/HIP SDK major (or upgrade the system to match). |
| 9 | `fix-9-igpu-dgpu` | both | iGPU enumerated alongside dGPU and destabilising the runtime | Random crashes / segfaults on systems with both an APU and a dGPU | Linux: `export HIP_VISIBLE_DEVICES=<dGPU-index>`. Windows: `setx HIP_VISIBLE_DEVICES <dGPU-index>` and reopen the shell. |
| 10 | `fix-10-container` | linux | Container can't see `/dev/kfd` or `/dev/dri/renderD*` | `rocminfo` inside container fails with permission denied; host works | Re-launch with `--device=/dev/kfd --device=/dev/dri --group-add render`; on rootless podman also `--userns=keep-id`. |
| 11 | `fix-11-iommu` | linux | Multi-GPU hang when IOMMU is in default 'on' mode | First multi-GPU job hangs indefinitely | Add `iommu=pt` to the kernel cmdline; reboot. |
| 12 | `fix-12-installer` | linux | `amdgpu-install` left a half-configured state | Subsequent `apt update` errors; `dpkg` complains about half-configured packages; `--accept-eula` repo regression | Run the documented uninstall sequence, then reinstall without the offending flag. |
| 13 | `fix-13-hip-sdk-missing` | windows | Framework links HIP but the HIP SDK isn't installed on this host | `amdhip64_X.dll not found`, `Could not find HIP`, `hipInfo` is not a command, `HIP_PATH` unset | Install the AMD HIP SDK matched to the framework's HIP major: <https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html> |
| 14 | `fix-14-adrenalin-too-old` | windows | Adrenalin / kernel-mode driver older than the HIP SDK pairs with | HIP SDK installed but `hipInfo.exe` reports no agents; `driver too old` style errors | Update Adrenalin from <https://www.amd.com/en/support>; cross-check the SDK release notes for the exact pairing; reboot. |
| 15 | `fix-15-msvc-redist` | windows | MSVC 2015-2022 runtime DLL missing -- HIP DLLs cannot load | `vcruntime140.dll` / `vcruntime140_1.dll` missing dialog; `api-ms-win-crt-*.dll` errors | Install the VC++ redistributable: <https://aka.ms/vs/17/release/vc_redist.x64.exe>. |

For the exact heuristics each checker uses (state signals vs. symptom
keyword weights), see the per-function comments in `scripts/diagnose.py`.

## Silent-footgun environment variables

These four change ROCm/HIP behaviour without printing a warning. Each one
gets a named callout in this section because they account for a
disproportionate share of "ROCm doesn't work" reports.

### `HSA_OVERRIDE_GFX_VERSION`

Tells HSA to advertise a different `gfx` target to user-space than the
kernel actually has. Useful in exactly one situation: when no
framework wheel ships kernels for your real gfx and a close-enough gfx
exists. Outside that case it causes page faults at runtime because the
compiler emits ISA for the override target but the hardware executes a
different ISA.

The doctor's default response when this variable is set on a GPU that
*does* have a native wheel is `fix-2-unset-override`, which:

1. Tells the user the variable is set.
2. Suggests `unset HSA_OVERRIDE_GFX_VERSION`.
3. Greps the user's shell rc files for persistent exports and points
   them at the lines to delete.

It deliberately does not edit the user's dotfiles. Editing someone
else's `~/.bashrc` is too easy to get wrong and too easy to forget you
did.

### `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`

The HIP / HSA equivalents of `CUDA_VISIBLE_DEVICES`. They restrict which
agents the runtime enumerates, by integer index in `rocminfo` order.
Setting either to `0,1` does not change anything on a single-GPU box but
matters on dual-GPU boxes (APU + dGPU, or two dGPUs).

The doctor uses `HIP_VISIBLE_DEVICES` (not `ROCR_VISIBLE_DEVICES`)
because both ROCm and PyTorch honour it; PyTorch also honours
`CUDA_VISIBLE_DEVICES` as an alias on HIP builds, which surprises
users who set both to different values. If both are set, the agent
should ask the user to pick one and unset the other.

### `PYTORCH_ROCM_ARCH`

A **build-time** variable, not a runtime one. Used when compiling
PyTorch from source to select which `gfx` targets the wheel will ship
kernels for. Setting it at runtime against a prebuilt wheel does
nothing; the wheel's arch list was baked at build time.

The agent should treat `PYTORCH_ROCM_ARCH` in a user's runtime shell as
a tell that the user has been pasting recipes from the wrong tutorial.
It is not a fix; it is misinformation.

### `LD_LIBRARY_PATH`

Frameworks that bundle their own HIP (most pip wheels) ship a private
`libamdhip64.so.X`. If the user has `LD_LIBRARY_PATH` pointing at a
system `/opt/rocm/lib` that contains a different major version, the
loader may pick the wrong one and the import fails with `cannot open
shared object file` or `version 'X' not found`. This LOOKS like
`fix-8-wheel-rocm` (wheel/ROCm major mismatch) but the underlying cause
is a load-order conflict.

If `examine.py` reports `hip_libs_on_ld_path=true` and the framework
also bundles HIP, suggest unsetting `LD_LIBRARY_PATH` and re-running the
import before reinstalling anything.

## Windows-specific footguns

Windows uses different mechanisms for the same failure modes Linux has;
keep the analogies straight rather than transplanting Linux fixes.

### `HIP_PATH` and multiple HIP SDK installs

The HIP SDK installer drops files under
`C:\Program Files\AMD\ROCm\<version>\` and sets `HIP_PATH` (and a
versioned `HIP_PATH_<ver>`) in the user/machine env. Multiple SDKs can
coexist on disk; whichever `HIP_PATH` points at is the one PyTorch and
`hipInfo.exe` actually load. Pointing it at the wrong major has the same
end result as `fix-8-wheel-rocm` -- `amdhip64_X.dll` from the SDK's `bin`
directory has the wrong major number for the installed framework.

`examine.py` records the `HIP_PATH` env var alongside the discovered SDK
install path. When they disagree (`HIP_PATH` is set but `hip_sdk_path`
points at a different directory), surface both values to the user and let
them decide which one is right before any other fix.

### PATH ordering on Windows

Windows uses PATH for DLL search; there is no `LD_LIBRARY_PATH` analog.
If the user has more than one `...\AMD\ROCm\<ver>\bin` on PATH, the first
one wins for DLL resolution, which can be a different SDK than `HIP_PATH`
points at. The signal is the same as Linux's load-order conflict: a
`cannot find amdhip64_X.dll` error that doesn't go away after reinstalling
the right SDK.

### Adrenalin pairing

The user-space HIP SDK and the kernel-mode driver (Adrenalin / Adrenalin
Pro) have to match. AMD bumps the supported pairing every HIP SDK
release; the live table is in
<https://rocm.docs.amd.com/projects/install-on-windows/en/latest/install/install.html>.
We deliberately do NOT hardcode a minimum Adrenalin version in
`diagnose.py` -- the table goes stale within months. `fix-14-adrenalin-too-old`
triggers on observable failure (HIP SDK present + `hipInfo.exe` cannot
enumerate, or matching keyword in the user's symptom) and routes the user
to the live page.

### MSVC redistributable

The HIP SDK's `amdhip64_*.dll` links against the MSVC 2015-2022 runtime
(`vcruntime140.dll`, `vcruntime140_1.dll`). Without the redistributable,
`import torch` fails with a missing-DLL dialog that points at
`vcruntime140_1.dll`, not at the HIP runtime. `fix-15-msvc-redist` is
specifically the path for this -- do NOT route it to `fix-8-wheel-rocm`
even though the surface error involves a missing DLL.

### `setx` does not affect open shells

Both `apply_fix.py`'s Windows runners and the recipe `commands` use
`setx` to persist env vars. `setx` writes to the User registry but does
NOT update the current process or already-open shells. After running any
`setx`-based fix, instruct the user to close and reopen the terminal
before re-verifying.

## Framework support matrix

The skill's first decision is which framework the user is running. Only
the "yes" rows trigger system examination; the "no" rows route upstream
without running any local probes.

| Framework | Examine the system? | Action |
|---|---|---|
| **PyTorch** (Linux ROCm wheels) | Yes | `python scripts/examine.py --framework pytorch` followed by `scripts/diagnose.py`. |
| **PyTorch** (Windows TheRock wheels) | Yes | Same scripts; on Windows `diagnose.py` filters the catalog to the cross-platform + Windows-only entries. |
| **llama.cpp** (built against system ROCm/HIP SDK) | Yes | `python scripts/examine.py --framework llama-cpp` followed by `scripts/diagnose.py`. |
| **Lemonade** | No -- ships its own ROCm | Route to <https://github.com/lemonade-sdk/lemonade> + [Discord](https://discord.gg/5xXzkMu8Zk). |
| **LM Studio** | No -- ships its own runtime | Route to <https://lmstudio.ai/docs/app> + Discord (in-app support, no public repo). |
| **Ollama** | No -- ships its own runtime | Route to <https://github.com/ollama/ollama> + Discord. |
| **vLLM** | Out of scope until phase 1+ | Route to <https://github.com/vllm-project/vllm/issues>. |
| **SGLang** | Out of scope until phase 1+ | Route to <https://github.com/sgl-project/sglang/issues>. |

If a Lemonade / LM Studio / Ollama user reports a problem AND a
standalone `rocminfo` (Linux) / `hipInfo.exe` (Windows) also fails (i.e.
the issue is the host install, not the bundled runtime), only then
escalate to a full examination. That is rare; the default action is
still to route upstream.

## Device support, phased

The skill ships in three phases. Phase 0 is the only one validated end
to end; later phases reuse the same scripts but loosen heuristics in
`diagnose.py`.

| Phase | GPUs | Status |
|---|---|---|
| 0 | Ryzen AI APUs (Strix Halo, Strix Point, Krackan, Phoenix, Hawk Point) -- gfx1151 / gfx1150 / gfx1103 / gfx1036 | Validated. Default target. |
| 1 | Instinct (MI300X, MI300A, MI250, MI210) -- gfx942 / gfx90a | Scripts work; not validated against the full failure list. |
| 2 | Radeon dGPUs (RDNA3, RDNA4) -- gfx1100, gfx1101, gfx1102, gfx12xx | Scripts work; iGPU/dGPU collision logic specifically targets this case. |

## Live AMD compatibility matrices

Hand-typed kernel/ROCm/distro matrices in skill bodies go stale within
months. Always fetch live from these pages instead of inlining them:

- **ROCm Linux system requirements** (kernel ranges, distro versions,
  Python versions): <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>
- **ROCm release compatibility matrix** (per-release driver / framework
  versions): <https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html>
- **RDNA3.5 system optimization** (APU-specific kernel notes referenced
  by `apu-memory-tuner`): <https://rocm.docs.amd.com/en/latest/how-to/system-optimization/rdna3-5.html>

`diagnose.py`'s `fix-3-rocm-kernel` recipe always links to the first
page rather than asserting a fixed kernel floor. The same goes for
wheel-index URLs in `fix-1-arch` and `fix-8-wheel-rocm`.

## Wheel index reference

For `fix-1-arch` and `fix-8-wheel-rocm`, prefer indexes in this order:

### Linux

1. **Official PyTorch ROCm wheels** -- `https://download.pytorch.org/whl/rocm6.4`
   (stable) and `https://download.pytorch.org/whl/nightly/rocm6.4` (nightly).
   Replace `6.4` with the user's system ROCm major.
2. **TheRock per-gfx wheels** -- <https://github.com/ROCm/TheRock>.
   The recommended fallback when the official index doesn't yet cover
   a gfx (typically true for newly released APUs in the first 2-3 ROCm
   releases after launch).
3. **Build from source** -- last resort. Pin `PYTORCH_ROCM_ARCH=<gfx>`
   at build time, not at runtime. See the PyTorch ROCm build guide.

### Windows

1. **TheRock Windows wheels** -- <https://github.com/ROCm/TheRock>. The
   live source of truth for which gfx targets are supported on Windows
   right now and which HIP SDK major each wheel pairs with. Always pull
   the install command from the project README rather than asserting a
   fixed `--index-url` here.
2. **Build from source** -- last resort. Requires Visual Studio Build
   Tools, the HIP SDK on PATH, and `HIP_PATH` set. See the PyTorch ROCm
   build guide for the Windows-specific environment variables.

For llama.cpp:

```bash
# Linux:
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=<gfx_target>
cmake --build build -j
```

```powershell
# Windows: needs the HIP SDK installed and HIP_PATH set; targets MSVC.
cmake -B build -G "Visual Studio 17 2022" -DGGML_HIP=ON `
  -DAMDGPU_TARGETS=<gfx_target>
cmake --build build --config Release
```

`AMDGPU_TARGETS` accepts a semicolon-separated list. Build a fat binary
for multiple GPUs with `-DAMDGPU_TARGETS=gfx1100;gfx1151`.

## Upstream routing

When `diagnose.py` returns no matches (exit 1), route the user to
exactly one upstream tracker rather than guessing. The mapping
`UPSTREAM_TRACKERS` in `diagnose.py` is the source of truth; the
abbreviated version:

| Framework | Tracker |
|---|---|
| PyTorch | <https://github.com/pytorch/pytorch/issues> (tag with `rocm`) |
| llama.cpp | <https://github.com/ggml-org/llama.cpp/issues> |
| Lemonade | <https://github.com/lemonade-sdk/lemonade/issues> |
| Ollama | <https://github.com/ollama/ollama/issues> |
| LM Studio | <https://lmstudio.ai/docs/app> (in-app support) |
| ROCm core (default) | <https://github.com/ROCm/ROCm/issues> |

Always attach the JSON from `python scripts/examine.py --json` to the
upstream report. It contains the kernel, GPU(s), ROCm version, install
method, framework version, and the env-var snapshot that the upstream
maintainer would otherwise have to ask for.

## Why we do not auto-set `HSA_OVERRIDE_GFX_VERSION`

This deserves its own callout because every other "ROCm not working"
tutorial on the internet suggests it as the first fix. We deliberately
suggest it last.

`HSA_OVERRIDE_GFX_VERSION` works by tricking HSA into reporting the
override gfx string to user space. The compiler then emits ISA for the
*override* target. The hardware still executes the ISA it natively
supports. When the two are close (e.g. gfx1100 → gfx1030) most kernels
run; when they differ in subtle ways (register count, LDS layout, queue
size) you get OUT_OF_REGISTERS, page faults, or silently wrong results.

Per the SCOPE document's success criteria:

> The skill never proposes `HSA_OVERRIDE_GFX_VERSION` as the *first*
> fix when a native wheel exists for the user's `gfx` target.

`diagnose.py`'s `fix-1-arch` recipe lists the override only in the notes
field, marked as a fallback when no native wheel exists. The auto-applied
path (`fix-2-unset-override`) is the OPPOSITE direction: removing the
override when the user already has one set unnecessarily.

## Why WSL is out of scope

`examine.py` detects WSL2 (via `microsoft` in `/proc/version` or
`WSL_DISTRO_NAME` in the environment) and exits 2 with a route-out
message. It does this on purpose: ROCm-on-WSL has its own failure modes
that are NOT in this catalog, and pretending they are Linux-native bugs
just gives users wrong fixes.

What's actually different on WSL:

- The kernel-mode driver lives on the **Windows host**, not in WSL. The
  user needs a recent Adrenalin Pro / Adrenalin install on the host, plus
  the WSL kernel update. None of those touch the WSL distro.
- `/dev/kfd` is replaced by `/dev/dxg` (the DirectX-on-WSL passthrough);
  the `fix-4-render-group` and `fix-5-amdgpu-load` checks are wrong for
  the wrong reasons.
- The HIP runtime libraries are loaded via `/usr/lib/wsl/lib/` rather
  than `/opt/rocm/lib`, so an `LD_LIBRARY_PATH` debug session is
  qualitatively different.

If a WSL user really does need a host-level ROCm fix, the right path is
the WSL install guide:
<https://rocm.docs.amd.com/projects/install-on-wsl/en/latest/>. Once
those WSL-specific prereqs are in place, the user is back to running
either pure Windows (this skill) or pure native Linux (this skill); WSL
itself stays out of scope.

## Adjacent problem: matrices in hand-typed tables

Most of what this skill needs (supported GPUs, kernel ranges, ROCm
releases, wheel arch lists, gfx families) is scattered across hand-typed
tables in docs pages, READMEs, and release notes. Everyone re-parses the
same matrix, and they drift.

The real fix is bigger than this skill: ROCm wants a **single,
agent-friendly source of truth** that feeds both the docs and skills like
`rocm-doctor`. Until that exists, the scripts here scrape
`rocm.docs.amd.com` at run time (`fix-3-rocm-kernel` links to the live
page rather than asserting a version) and the skill body is careful not
to assert a matrix that will be wrong in 90 days.

When ROCm ships that source of truth, `examine.py` and `diagnose.py`
should switch to it. Until then, prefer "here is the live URL" over
"the supported kernels as of this writing are".
