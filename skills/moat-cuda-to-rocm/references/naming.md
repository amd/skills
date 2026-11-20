## Naming: ROCm vs HIP

These name two different facets and are not interchangeable. The confusion is structural: NVIDIA uses one brand ("CUDA") for the language, the runtime, and the platform, while AMD split that into two names, so no single word maps cleanly.

- **HIP** is the programming model -- the kernel-language dialect (`__global__`, `__device__`, `<<<>>>`) and the `hipXxx` host runtime API. It is the analogue of *CUDA C++ plus the CUDA Runtime API*. HIP is portable: it compiles to AMD (via ROCm) or to NVIDIA (a thin wrapper over CUDA), so "HIP" alone does not imply AMD hardware.
- **ROCm** is the platform/toolkit -- the amdgpu/KFD driver, the ROCr/HSA runtime, the amdclang/LLVM AMDGPU compiler, HIP itself, and the domain libraries. It is the analogue of the *CUDA Toolkit*, and it is what pins execution to an AMD GPU.
- **roc\* vs hip\* libraries**: `roc*` (rocBLAS, rocFFT, rocSOLVER, rocPRIM, rocThrust) are AMD-native libraries with AMD-shaped APIs; `hip*` (hipBLAS, hipFFT, hipSOLVER, hipCUB) are thin portability wrappers exposing a CUDA-library-compatible API that dispatch to `roc*` on AMD (or `cu*` on NVIDIA). hipBLAS-on-AMD is rocBLAS underneath.

Mapping: CUDA C++ and the CUDA runtime -> **HIP**; nvcc, the CUDA Toolkit, and cuBLAS/cuFFT/cuDNN/CUB -> **ROCm** (hipBLAS+rocBLAS, hipFFT+rocFFT, MIOpen, hipCUB+rocPRIM).

How to say it in prose, commits, and PRs:
- The **code transformation** is a port **to HIP** (hipify, the `cuda_to_hip.h` shim, the `cudaXxx -> hipXxx` runtime layer).
- The **target/platform/build flag/libraries** are **ROCm** ("validated on gfx90a, ROCm 7.2.1", the `USE_HIP`/`USE_ROCM` option, "swap cuFFT for hipFFT").
- A pure language+runtime port is most precisely "a HIP port targeting ROCm." Do not call the platform "HIP" or the kernel dialect "ROCm".
- Commit prefix stays `[ROCm]` as the umbrella tag (it signals "adds AMD support" to upstream maintainers and matches AMD/PyTorch usage); name a specific `roc*`/`hip*` library only when the port actually substitutes it.

PyTorch is the canonical conflation to NOT imitate: its build macro is `USE_ROCM`, `torch.version.hip` exists, yet the runtime device is still literally `"cuda"` -- it brands the platform ROCm while keeping CUDA's names and uses "HIP"/"ROCm" interchangeably in build code. Be precise instead.

Upstream-visible text (commit messages, code comments, PR bodies) carries no in-house vocabulary -- no internal stage, state or tool names, and none of this skill's own labels such as "Strategy A/B" or "fault class". Keep the technical rationale, drop the in-house labels; this text is read by external maintainers.

A term list does not catch a comment that explains a hazard by naming ANOTHER ported project as shorthand for it. arrayfire carried eleven such comments; the six saying "fault class" were caught by a term check and five saying "the amgcl void*-aliasing fix" or "the gsplat target-attribute rule" were not. To an upstream reader both are dangling references to a codebase they have never seen. State the hazard itself -- "hipSPARSE typedefs several descriptor types to the same void*, so a wrapper keyed on the handle type cannot separate them" -- and before pushing, grep new comments for the names of other projects you have ported.
