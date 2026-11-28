# ROCm Doctor — reference

The closed failure-mode catalog and the CLI fields the skill reads. The catalog is
authoritative in the `rocm` CLI; where this file and what the CLI prints
disagree, the CLI is right. What to do with a diagnosis is in
[SKILL.md](SKILL.md); this file carries the shapes, and the few directives that
belong to one field.

## Contents

- CLI commands — `rocm examine`, `rocm diagnose` (its fields and the `model`
  object), `rocm fix`
- Closed catalog — every failure mode, the platforms it applies to, and what
  `rocm fix` does with it
- Framework routing — what `route_when_no_match` returns

## CLI commands

### `rocm examine [--json]`

Inspect the host: GPU + gfx target, driver, ROCm install, render/video groups,
`/dev/kfd` + render devices, kernel modules, framework introspection, recent
amdgpu kernel-log evidence. `--json` emits the **Examination** document for
tooling; the groups and the framework are in it only.

- It reports the host and does not judge it, so a finding such as no GPU still
  exits 0. The verdict is the `status` field: `ok` · `no-amd-gpu` · `wsl` · `unsupported-os` · `degraded`.
  A non-zero exit means the command could not examine the host at all.

### `rocm diagnose [--symptom TEXT] [--top N] [--json] [--distro [NAME] | --report [--send]]`

Match the host + symptom against the closed catalog. It exits 0 whether it
matched, found nothing or is out of scope, so read the result from `--json`. A
non-zero exit is an error from the command itself, such as a usage error, or
`--model` together with `--distro`, which is refused because the model verdict is
about the machine running the command and not the one `--distro` names.

`rocm diagnose --model REF [--context-size N] [--kv-cache-type f16|q8_0|q4_0] [--json]`
adds the `model` object below.

`--top N` limits the text report to N matches; the JSON is not limited by it.
`--distro [NAME]` diagnoses a WSL distribution from the Windows host: its result
describes the distribution, and its fixes are not run from the host
([SKILL.md](SKILL.md), step 3). It examines the distribution's WSL facts only:
checks that read its own framework, ROCm version, environment, `/dev/shm` or
library selection do not run, so `has_match` false there is weaker than it reads;
for a full diagnosis run `rocm diagnose` inside the distribution. `--report`
prints the report this machine would contribute and sends nothing, and `--send`
(which needs `--report` and cannot be combined with `--json`) adds a prefilled
mail the user sends from their own account. With `--report` the model is not
assessed at all and nothing is fetched; a note on stderr names the model flags
that were ignored.

- `matched[]` — ranked `{ id, title, score, evidence[], symptom_matched, fix }`; scores run 0 to 100.
  The `fix` fields SKILL.md acts on are named in its step 1. `class`
  is what `rocm fix <id>` does on the machine the diagnosis describes
  ([SKILL.md](SKILL.md), step 3; the classes are listed under Marker in the
  catalog). `needs` is present only when the entry has one: the argument a
  `needs-argument` entry waits for. A `diagnose-only` entry carries an `info-` id
  rather than a `fix-N` / `fix-wsl-N` one because it has no fix to offer;
  `rocm fix` still takes that id and answers for it.
- `has_match` — whether any entry cleared `min_score_for_match`; the gate for acting
  on `matched` ([SKILL.md](SKILL.md), step 1).
- `min_score_for_match` (50), `high_confidence_threshold` (75).
- `out_of_scope` — set when the host's platform family has no catalog entries at all,
  and then `matched` is empty. Linux, Windows and WSL2 each have entries, so it does
  not fire on them.
- `route_when_no_match` — `{ target, url }`, the upstream tracker for when
  `has_match` is false. [Framework routing](#framework-routing) lists what it returns.
- `model` — set only when `--model <ref>` was passed; otherwise not set. Answers
  "will this model run here", without downloading the weights: for a curated
  model from the recipe catalog and this host's GPU/RAM, and for a GGUF outside
  the catalog with an `estimate` from its file header (below). Its fields include
  `verdict`, `evidence[]`, `alternatives[]`, `undetermined_reason`, `warnings[]`,
  `fix` and `route`; `warnings[]` carries, for a GGUF estimate that was
  compared with the GPU, the standing note on which options `rocm serve` does not
  take. `verdict` is one of `ready`, `degraded`, `blocked`,
  `undetermined` — treat `undetermined` as "the CLI could not find out", never as
  evidence the model is incompatible. `alternatives[]` are curated models, each with the `model_ref` to pass to `rocm serve`. `model.fix` has
  the shape of a diagnosis `fix` with an empty `fix_id`: no catalog entry stands
  behind it, so `rocm fix` would reject one, and its steps are relayed rather than
  applied.
  `route` says where to report a model the CLI cannot answer for.
  For a GGUF outside the catalog (a local `.gguf` path, or `owner/repo:QUANT`)
  the verdict stays `undetermined` and an `estimate` object is added, sized from
  the file's header (the weights are never downloaded). Its fields, by group:
  where it came from (`source`, `architecture`, `layers`); the outcome
  (`outcome` is `fits`, `does_not_fit` or `cannot_estimate`, and with
  `cannot_estimate` the `cannot_estimate_reason` and `detail` say why; the
  figures are still given when the reason is about the host (`no_gpu`,
  `accelerator_memory_unknown`, `unified_memory_unreadable`: what the model would
  need, with nothing to compare it to) and are null when it is about the file
  (`header_unreadable`, `not_a_model` for a multimodal projector named in place
  of a model, `split_file`, `latent_attention`, `no_attention_layers`,
  `missing_metadata`), and are given with no verdict for `unmeasured_layout` (a
  state-space, hybrid or mixture-of-experts model, part of whose memory is not
  counted or not measured, so the total can be too low) and for a
  `missing_metadata` header that lacks only the feed-forward or embedding width
  (the compute buffer is then understated)); what was asked for (`context_size`, `context_size_defaulted`,
  `kv_cache_type`); the memory figures (`weights_gib`, `kv_cache_gib`,
  `compute_buffer_gib`, `runtime_reserve_gib`, `total_gib`, the same in bytes as
  `weights_bytes`, `kv_cache_bytes`, `compute_buffer_bytes`,
  `runtime_reserve_bytes` and `total_bytes`, and `kv_cache_bytes_per_token`, the
  cache's cost of one token); the comparison (`available_gib` is the memory the
  estimate was compared against, not what is free: the total dedicated VRAM of
  the GPU with the most of it that ROCm can see, or -- on an integrated GPU --
  the ceiling on the pool it shares with the operating system.
  `available_is_shared_pool` says which of the two it is, and is the only way
  to tell them apart from the number alone; `max_context` is the largest context that fits, capped at
  `training_context`, and is null when GPU memory was not measured, when no
  verdict is given (`unmeasured_layout`, or `missing_metadata` with figures),
  and when not even one 256-token step fits; it can sit below `context_size` for an
  outcome of `fits`); and what the figures leave out
  (`conservative_ceiling` says the ceiling is known to be conservative,
  `suggestions[]` what would make it fit, `notes[]` the caveats).
  It is an estimate, not a tested minimum: report it as one.
  The CPU-fallback rule and what to do with a suggestion are in
  [SKILL.md](SKILL.md), Rules.

### `rocm fix [<id>] [--yes] [--dry-run] [--device-index N] [--json]`

Apply a fix by id (run with no id to list). `rocm fix --json` with no id prints
the catalog manifest instead: `contract_version`, one entry per fix, and the
`exit_codes` map from each exit's name to its number, and, when present,
`exit_code_descriptions`, what each exit means. It applies no fix and exits 0.
The skill reads `contract_version` before it diagnoses (Phase 0 in
[SKILL.md](SKILL.md)). `--json` with an id is refused. What to do with an outcome
is in SKILL.md, step 3.

Three ids are ever `auto`, and which of the three depends on the **host**, because
`rocm fix`'s listing marks each entry for the machine you run it on, never for
every machine the id applies to:

- **linux** — auto-applicable: `fix-4-render-group`, `fix-6-path`.
- **windows** — auto-applicable: `fix-2-unset-override`, `fix-6-path`.
- **wsl** — auto-applicable: `fix-6-path`.

"Auto-applicable" is the `auto` class on that host, the same as the
`fix.auto_applicable` flag in a diagnosis. An entry whose class is `print-only`
prints its plan for the user to run.

## Closed catalog (30 failure modes)

The OS column is the platform family the CLI scopes an entry to, and WSL2 is a
family of its own — not a flavour of `linux`. An entry reaches a WSL host only
by naming `wsl`, so the bare-metal Linux entries (the `amdgpu` module,
`/dev/kfd`, the render group) stop applying there automatically rather than
reporting confident nonsense.

**Marker** is the class `rocm fix`'s listing reports **on bare-metal Linux**, in
the listing's spelling, lower-cased here: `auto`, `needs-arg`, `print-only` or
`diagnose-only`. `fix.class` in a diagnosis carries the same classes, with
`needs-arg` spelled `needs-argument` ([SKILL.md](SKILL.md), step 3, explains
each). A class can differ by platform, which the cell says where it does
(`fix-2-unset-override`).

**Typical signal** is a short handle for what the check looks for, not a rule for
when it fires: the CLI's score and the entry's `evidence` decide, so read
`has_match` and `evidence`.

| id | OS | Failure mode | Typical signal | Marker (bare-metal Linux) |
| --- | --- | --- | --- | --- |
| `fix-1-arch` | linux/windows/wsl | GPU gfx target not in the framework's build arch list | `hipErrorNoBinaryForGpu`, `HSA_STATUS_ERROR_INVALID_ISA`, "invalid device function" | print-only |
| `fix-2-unset-override` | linux/windows/wsl | `HSA_OVERRIDE_GFX_VERSION` set on a GPU that now has a native wheel | page faults / `OUT_OF_REGISTERS`, override set in env | print-only (auto on windows) |
| `fix-3-rocm-kernel` | linux | ROCm + distro/kernel form an unsupported triple | ROCm installed but `amdgpu` not loaded; DKMS build failure | print-only |
| `fix-4-render-group` | linux | User not in `render`/`video` group (or `/dev/kfd` owned by the other group) | cannot open `/dev/kfd`, permission denied | auto |
| `fix-5-amdgpu-load` | linux | `amdgpu` module not loaded (or blacklisted) | "ROCk module is NOT loaded", blacklist entry, Secure Boot | print-only |
| `fix-6-path` | linux/windows/wsl | ROCm/HIP binaries not on PATH after install | `rocminfo: command not found`, `hipInfo` missing from PATH | auto |
| `fix-7-stale-repos` | linux | Stale/conflicting APT/DNF/zypper repos from prior installer runs | apt 404 `repo.radeon.com`, unmet deps, ≥2 ROCm repo files | print-only |
| `fix-8-wheel-rocm` | linux/windows/wsl | Framework wheel built for a different ROCm X.Y release than the system | `libamdhip64.so.X` / `amdhip64_X.dll` load failure | print-only |
| `fix-9-igpu-dgpu` | linux/windows | iGPU enumerated alongside dGPU, destabilising the runtime | APU + discrete AMD present, `HIP_VISIBLE_DEVICES` unset, crash/segfault | needs-arg |
| `fix-10-container` | linux | Container can't see `/dev/kfd` or `/dev/dri/renderD*` | running in docker/podman, kfd/render devices missing | print-only |
| `fix-11-iommu` | linux | Multi-GPU hang with IOMMU enabled | ≥2 AMD GPUs, `iommu=` not `pt`, hang/deadlock/timeout | print-only |
| `fix-12-installer` | linux | Repo-native package-manager install left a broken DKMS / repo state | dpkg half-configured, DKMS build failure, dpkg/apt error | print-only |
| `fix-13-hip-sdk-missing` | windows | HIP SDK not installed | no HIP SDK under Program Files (or `hipInfo.exe` missing from its `bin`) while something on the host calls into HIP; `amdhip64` not found, `HIP_PATH` not set. A Vulkan-backed engine (the Lemonade default) or a managed-runtime wheel needs no system SDK, so its absence is not evidence there | print-only |
| `fix-14-adrenalin-too-old` | windows | Adrenalin / kernel-mode driver too old for the HIP SDK | `hipInfo` can't enumerate, "driver too old", HSA "no agents found" | print-only |
| `fix-15-msvc-redist` | windows | MSVC runtime missing (programs that link it, such as the HIP SDK DLLs and the llama.cpp server, can't load) | `vcruntime140.dll` / `vcruntime140_1.dll` missing | print-only |
| `fix-16-vllm-oom` | linux/wsl | vLLM reserved ~90% of the GPU's TOTAL VRAM for its KV cache and ran out at startup | vLLM engine start fails with an out-of-memory error; the GPU is shared/busy, or the model does not fit | print-only |
| `fix-17-torch-dlpack` | linux/wsl | `torch-c-dlpack-ext` loads its CUDA prebuilt on a ROCm torch, aborting vLLM's engine start at import time | vLLM engine start fails on import (`libtorch_cuda.so`) | print-only |
| `fix-18-comgr-conflict` | linux/wsl | The code object manager library (`libamd_comgr`) that would load belongs to a different installation than the HIP runtime that would load, so device code compilation fails with an error naming neither | a compilation error naming neither library; `comgr_selected` and `hip_selected` in `rocm examine --json` come from different installations | print-only |
| `fix-19-shm-too-small` | linux/wsl | `/dev/shm` too small for a serving workload, which needs gigabytes; fires wherever `/dev/shm` is under 1 GiB, and a container defaults to 64 MB | `/dev/shm` reported small; a data-loader worker killed by a bus error | print-only |
| `info-20-gpu-no-rocm-support` | linux/windows | Every AMD GPU on the host is a part ROCm has never shipped a runtime build for (gfx90c, the Vega iGPU in Renoir / Lucienne / Cezanne / Barcelo, is one). Vulkan-based engines still work | every detected AMD GPU identifies as an unsupported target | diagnose-only |
| `fix-21-msvc-redist-outdated` | windows | MSVC runtime present but older than 14.40, so programs built with a recent toolset (such as the llama.cpp server that Lemonade installs) crash | `msvc_redist_version` below 14.40 in `rocm examine --json`; an access violation (`0xC0000005`, `-1073741819`) in `msvcp140.dll` | print-only |
| `fix-22-display-driver-old` | windows | The AMD display driver is dated before the first HIP SDK for Windows (27 July 2023). Advice only: AMD publishes no minimum driver version for the HIP SDK or the Lemonade Vulkan backend, so the minimum is not established and the entry never claims the driver is broken. The date is only a floor and cannot judge newer GPUs such as gfx120X, gfx1150 and gfx1151, which may need newer drivers | `rocm examine` shows `driver_age: older than the first HIP SDK for Windows` | print-only |
| `fix-23-vllm-extension-load` | linux/wsl | One of vLLM's compiled extensions (`vllm/_*.abi3.so`) failed to load, for example because it was built against a newer glibc than the host has, so an op it registers is missing and the engine fails at startup | `AttributeError: '_OpNamespace' '_C' object has no attribute '...'` — enough on its own, but only when the pasted text also identifies vLLM (the word `vllm`, the `EngineCore` process prefix, or a vLLM extension name such as `_rocm_C`/`_moe_C`/`_C_cache_ops`/`_C_stable_libtorch`), since every PyTorch-ecosystem library with a `_C` extension raises the identical line; `Failed to import from vllm._...`; `GLIBC_2.NN' not found` naming a vllm `.abi3.so` | print-only |
| `fix-wsl-1-gpu-not-exposed` | wsl | `/dev/dxg` absent, so the distro cannot reach the GPU at all | no `/dev/dxg`; in a container, the device was never passed through | print-only |
| `fix-wsl-2-dxcore-missing` | wsl | `/usr/lib/wsl/lib` DXCore shims missing, so the runtime cannot reach the host driver | `/usr/lib/wsl/lib/libdxcore.so` missing (or the directory absent entirely) | print-only |
| `fix-wsl-3-rocdxg-missing` | wsl | ROCDXG, the ROCm-to-DXCore shim the WSL path runs on, is not installed | `librocdxg.so` not found under any ROCm install | print-only |
| `fix-wsl-4-rocdxg-not-linked` | wsl | `librocdxg` installed but not in the linker cache, so it is unloadable | `librocdxg` installed but not found by the loader | print-only |
| `fix-wsl-5-distro-too-old` | wsl | Distro release below the floor the WSL path requires | an Ubuntu release under the supported floor (e.g. 22.04) | print-only |
| `fix-wsl-6-host-driver-too-old` | wsl | Windows host driver too old or absent | the Windows host reports no AMD display adapter; `rocminfo` enumerates no GPU although the distro side is in place | print-only |
| `fix-wsl-7-wsl1` | wsl | Distro running under WSL 1, which exposes no GPU device at all | the running kernel is a WSL 1 kernel | print-only |

One thing the numbering does not tell you: the `fix-wsl-N` entries are a parallel
series, not a continuation of the numeric one, because they answer for a
different platform family. The numeric series itself runs `fix-1` through
`fix-19`, then `fix-21` through `fix-23`. The ids are stable handles rather than positions, so
the number `fix-20` is simply not used here: slot 20 belongs to
`info-20-gpu-no-rocm-support`, which is in the table above but under an `info-`
prefix because it has no fix to offer. A gap is not an error, and a retired
entry would leave a hole rather than renumber the ones after it.

## Framework routing

`rocm diagnose` diagnoses frameworks built against the **system** ROCm/HIP:
PyTorch and llama.cpp. Apps that ship their own runtime are routed upstream **by
the skill** ([SKILL.md](SKILL.md), Framework routing): the route the CLI returns
has no arm for them.

`route_when_no_match` itself returns one of:

- **pytorch** → https://github.com/pytorch/pytorch/issues (tag with the rocm label)
- **llama-cpp** → https://github.com/ggml-org/llama.cpp/issues
- otherwise **rocm-core** → https://github.com/ROCm/ROCm/issues
