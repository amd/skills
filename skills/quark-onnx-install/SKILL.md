---
name: quark-onnx-install
description: >-
  Install or verify the correct ONNX Runtime build (and the matching `onnx` package) for a user's
  accelerator backend before Quark ONNX-flow usage. Trigger for "install onnxruntime", "pip
  install onnxruntime", "set up onnxruntime for ROCm", "set up onnxruntime for CUDA",
  "onnxruntime-gpu vs onnxruntime", "onnx version mismatch", "CPU-only onnxruntime installed",
  "onnxruntime providers list missing CUDAExecutionProvider/ROCMExecutionProvider", failures
  importing `onnxruntime`, or any request to get the correct ONNX Runtime build running. Also
  trigger when quark-install reports that ONNX Runtime is missing or mismatched before proceeding
  with the ONNX-to-ONNX flow.
---

# quark-onnx-install

## Catalog portability

This standalone skill is federated from `amd/Quark` at commit `1b229f781a1974cc742884e42d8eefc1eebb4f0a`. Resolve bundled files relative to this `SKILL.md`. Repository-relative paths such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a); use a local Quark checkout when available, otherwise consult that pinned source.

## Purpose

Install the correct ONNX Runtime build for the user's accelerator backend, plus the matching `onnx`
package and supporting tooling (`onnxslim`, `onnxscript`). ONNX Runtime must be installed before
Quark's ONNX-to-ONNX flow because Quark uses ORT as the calibration / inference engine and registers
custom ops (`BFPQuantizeDequantize`, `MXQuantizeDequantize`, `Extended*`) into it. Getting this wrong —
installing the CPU build on a GPU machine, or installing both `onnxruntime` and `onnxruntime-gpu`
side-by-side — causes EP-not-available errors, silent CPU fallback, or import-time DLL conflicts that
are hard to diagnose later. This skill exists separately from `quark-install` so that ONNX Runtime
setup has a clear, single-responsibility boundary, parallel to `quark-torch-install` for Torch.

## Inputs

- `env_context.json` with detected accelerator info (CPU / CUDA major+minor / ROCm major+minor)

## Outputs: onnx_install_result.json

Records the installed ONNX Runtime build, accelerator backend tag, the `onnx` package version, and
verification status.

```json
{
  "status": "ok",
  "onnxruntime_package": "onnxruntime-gpu",
  "onnxruntime_version": "1.23.2",
  "accelerator_tag": "cuda-12",
  "onnx_version": "1.18.0",
  "onnxslim_version": "0.1.84",
  "onnxscript_version": "0.1.0",
  "verification": {
    "import_onnx_ok": true,
    "import_onnxruntime_ok": true,
    "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "expected_provider_present": true,
    "custom_ops_compile_ok": true
  }
}
```

On failure, set `status: "failed"` and include a `failure_reason` with the exact failing verification
command.

## Python Version Requirements

- **Supported**: Python 3.11, 3.12, 3.13
- **Not supported**: Python 3.14+
- **Recommended for new setups**: Python 3.13 via Miniforge/Miniconda

## Package Version Matrix

**Authoritative sources**:

- `tools/ci/install_onnxruntime.sh` — accelerator → `onnxruntime*` variant + version mapping (CI truth)
- `docs/source/install.rst` — user-facing supported version range
- `requirements.txt` — `onnx`, `onnxscript`, `onnxslim` pin

Before generating install commands, **always read these sources** to get the current verified
combinations. Do not memorize version numbers — the matrix changes with each Quark release.

### Current ranges (verify before use)

| Package | Range (verify against requirements.txt / install.rst) |
|---------|-------------------------------------------------------|
| `onnx` | `>=1.21.0, <=1.22.0` |
| `onnxruntime*` | `>=1.22.2, <=1.25.1` |
| `onnxslim` | `>=0.1.84` |
| `onnxscript` | unpinned |

### How to read the source

1. Open `tools/ci/install_onnxruntime.sh` and locate the `install_onnxruntime` function. It dispatches
   on `accelerator_version` (`cpu`, `cuda-11.*`, `cuda-12.*`, `rocm-*`) and decides:
   - which variant to install (`onnxruntime`, `onnxruntime-gpu`, `onnxruntime_rocm`),
   - whether to use pypi.org or the AMD internal Artifactory wheel.
2. Cross-check the chosen `onnxruntime` version against the range in `docs/source/install.rst`
   (search for "ONNX Runtime version").
3. Read `requirements.txt` for the `onnx` / `onnxslim` / `onnxscript` constraints.
4. Construct the install commands using the patterns below.

### Install command patterns

#### CPU

```bash
pip install "onnxruntime>=1.22.2,<=1.25.1"
pip install "onnx>=1.21.0,<=1.22.0" "onnxslim>=0.1.84" onnxscript
```

#### CUDA 12.x / 13.x

Matches the current `install.rst` recommendation:

```bash
pip install onnxruntime-gpu                # default pypi build targets recent CUDA
pip install "onnx>=1.21.0,<=1.22.0" "onnxslim>=0.1.84" onnxscript
```

#### CUDA 11.x

Per `install_onnxruntime.sh`:

```bash
pip install --no-cache-dir onnxruntime-gpu \
  --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
pip install "onnx>=1.21.0,<=1.22.0" "onnxslim>=0.1.84" onnxscript
```

#### ROCm 6.x

Internal `onnxruntime_rocm` wheel from AMD Artifactory (no pypi build):

```bash
# Resolved via _install_onnxruntime_from_artifactory in tools/ci/install_onnxruntime.sh
# Wheel pattern: onnxruntime_rocm-<ort_ver>-cp<py_ver>-*.whl
# Base URL: https://xcoartifactory.xilinx.com/artifactory/uai-pip-local/onnxruntime/rocm-<ver>
```

If the user does not have access to `xcoartifactory.xilinx.com`, **stop** and surface the gap — do
not silently fall back to a CPU build.

#### ROCm 7.x and above

Per `install_onnxruntime.sh`, build incompatibilities mean the CPU variant is used:

```bash
pip install "onnxruntime>=1.22.2,<=1.25.1"   # CPU variant; ROCm EP not available in this case
pip install "onnx>=1.21.0,<=1.22.0" "onnxslim>=0.1.84" onnxscript
```

Make this trade-off explicit to the user (no `ROCMExecutionProvider`, calibration runs on CPU).

#### Optional: ONNX Runtime GenAI (OGA flow for LLM models)

```bash
pip install onnxruntime-genai
```

#### Optional: ONNX Runtime Extensions

Referenced in `pyproject.toml` mypy config:

```bash
pip install onnxruntime-extensions
```

**Critical**: Never install both `onnxruntime` and `onnxruntime-gpu` (or `onnxruntime_rocm`) into the
same environment — pip allows it but the imports collide and ORT may load the wrong shared library.
If a different variant is already installed, **uninstall it first** (`pip uninstall -y onnxruntime
onnxruntime-gpu onnxruntime_rocm onnxruntime-genai`).

## C++ Compiler Requirement

Quark's ONNX custom-ops library (`quark.onnx.operators.custom_ops`, providing `BFPQuantizeDequantize`,
`MXQuantizeDequantize`, `Extended*`) is **compiled on first import** using the local toolchain. This
must succeed for any BFP / MX / Extended quant scheme to work.

| OS | Required compiler |
|----|-------------------|
| Linux | `g++` (`apt install g++` on Ubuntu) |
| Windows | Visual Studio 2022+ with the *Desktop development with C++* workload (use the Developer Command Prompt) |

For GPU kernels, set the corresponding env var so the compiler can find headers:

- ROCm: `export ROCM_PATH=/opt/rocm`
- CUDA: `export CUDA_HOME=/usr/local/cuda`

Verify the compile by running:

```bash
python -c "import quark.onnx.operators.custom_ops"
```

## Rules

- **Always read `tools/ci/install_onnxruntime.sh` before generating install commands.** The version
  matrix and Artifactory paths change with each Quark release. Never rely on memorized version
  numbers — always verify against the upstream script and `requirements.txt`.
- **Always detect the accelerator before choosing the ORT variant.** Run or reference
  `quark-env-preflight` if hardware facts are missing. The entire install plan depends on getting
  this right (CPU `onnxruntime`, GPU `onnxruntime-gpu`, ROCm 6.x `onnxruntime_rocm`, ROCm 7.x falls
  back to CPU `onnxruntime`).
- **Bind ORT variant and accelerator to the same backend.** Never mix `onnxruntime-gpu` (CUDA) with a
  ROCm environment or vice versa. If multiple `onnxruntime*` variants are detected installed,
  uninstall all of them before installing the correct one.
- **Pin within the supported ranges.** `onnx` must be `>=1.21.0,<=1.22.0` per `requirements.txt`;
  ORT must be in the range stated in `docs/source/install.rst`. Versions outside these ranges
  silently break Quark's QDQ insertion or custom-op registration.
- **Never skip verification.** After installation, always run the verification commands below,
  including the custom-ops compile check.
- **If accelerator or AMD Artifactory access is unclear, stop after the plan.** Present the install
  plan but do not execute. Hand the gap back to `the matching public quark-onnx skill` so it lands
  in `session_context.json`'s `open_questions`, and ask the user to confirm.
- **Show exact commands before execution.** The user should see every `pip uninstall` /
  `pip install` command, every version, and every `--extra-index-url` before anything runs.

## Verification Commands

```bash
# onnx package check
python -c "import onnx; print('onnx:', onnx.__version__)"

# onnxruntime check + EP list
python -c "import onnxruntime as ort; print('ORT:', ort.__version__); print('EPs:', ort.get_available_providers())"

# Expected EPs (assert at least one of these is in the list):
#   CPU:       'CPUExecutionProvider'
#   CUDA:      'CUDAExecutionProvider' (and 'CPUExecutionProvider')
#   ROCm 6.x:  'ROCMExecutionProvider' (and 'CPUExecutionProvider')
#   ROCm 7.x:  'CPUExecutionProvider' only (no ROCm EP — by design, see install_onnxruntime.sh)

# Quark ONNX custom-ops compile (first run triggers compilation)
python -c "import quark.onnx.operators.custom_ops"

# Optional: GenAI for LLM OGA flow
python -c "import onnxruntime_genai; print('GenAI:', onnxruntime_genai.__version__)"
```

## Interaction Flow

1. **Intake**: Determine what the user already has installed and what accelerator they need. Check
   if `quark-env-preflight` has already run. Detect any pre-existing `onnxruntime*` variants.
2. **Plan**: Present the accelerator-specific ONNX Runtime install command, the `onnx` /
   `onnxslim` / `onnxscript` commands, the C++ compiler check, and (if relevant) the GenAI add-on.
   Justify each version against `install_onnxruntime.sh` and `requirements.txt`.
3. **Confirm**: Required before any package change. Show: what will be uninstalled, what will be
   installed, which `--extra-index-url` will be used, and what environment will be modified.
4. **Execute**: Run uninstall (if a conflicting variant is present), then the install commands.
5. **Verify**: Run all verification commands. Report pass/fail for each, especially:
   - expected EP present in `get_available_providers()`,
   - custom-ops compile succeeds.

## Recovery

- **If `get_available_providers()` does not include the expected accelerator EP**: The CPU build of
  `onnxruntime` was installed instead of the GPU build (or both variants are present). Show the
  exact uninstall + reinstall commands.
- **If both `onnxruntime` and `onnxruntime-gpu` are installed**: Uninstall both (`pip uninstall -y
  onnxruntime onnxruntime-gpu onnxruntime_rocm`), then reinstall only the correct variant.
- **If `import quark.onnx.operators.custom_ops` fails to compile**: Check `g++` (Linux) or VS 2022
  (Windows) is installed and on PATH; for GPU builds, check `ROCM_PATH` / `CUDA_HOME` is set.
- **If `onnx` import succeeds but Quark complains about a schema mismatch**: `onnx` version is
  outside `>=1.21.0,<=1.22.0`. Reinstall to a pinned version inside the range.
- **If on ROCm 6.x and the Artifactory wheel is unreachable**: The user is off the AMD internal
  network. Surface the gap — do not silently install the CPU variant. Document the ROCm-EP loss
  before proceeding if the user accepts CPU fallback.
- **If Python version is wrong**: Recommend creating a new conda environment with a supported
  version (3.11, 3.12, or 3.13).

## Windows-Specific Notes

- ROCm is not supported on Windows — only CUDA and CPU variants of ONNX Runtime are available.
- The custom-ops library compile requires Visual Studio 2022+ with the *Desktop development with
  C++* workload. Use the Developer Command Prompt or set the build-tool paths via the developer
  command file.
- If pip fails with long path errors when installing `onnx` / `onnxruntime`: enable Win32 long
  paths via Group Policy Editor (Computer Configuration > Administrative Templates > System >
  Filesystem > Enable Win32 long paths).
- WSL2 with Ubuntu is recommended as an alternative for Windows users who need ROCm.
