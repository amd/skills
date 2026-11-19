# When hipify has nothing to translate

Some CUDA code has no HIP analogue to translate INTO. The build succeeds at finding nothing to do, which reads like a small port and is the opposite. This is the most common reason a project gets declared not portable, and it is worth separating two very different verdicts that get written the same way:

- **No AMD-native target exists.** Genuinely not portable. NVIDIA-proprietary runtimes (TensorRT, OptiX), closed binary kernels shipped as wheels, or a codegen backend emitting PTX with no AMDGPU counterpart.
- **A port would be a REIMPLEMENTATION against an AMD-native target.** Tractable, sometimes valuable, but not a translation -- it needs its own scope, its own estimate, and a person's decision. Recording it as not portable throws away the fact that a target exists.

## CUTLASS -> Composable Kernel

CUTLASS does not port; upstream projects say so themselves (NATTEN's own porting guide states it). The AMD-native equivalent is Composable Kernel (`ROCm/composable_kernel`), including `ck_tile` for tile-level programming. A CUTLASS-based GEMM or attention kernel becomes a CK reimplementation: the algorithm and the tiling strategy carry over, the source does not.

## CuTe -> FlyDSL

`ROCm/FlyDSL` is AMD's Flexible Layout Python DSL, an MLIR-based Python front end for expressing tiling, partitioning, data movement and kernel structure. Its core, FLIR (Flexible Layout Intermediate Representation), is a layout algebra **explicitly inspired by CuTe**, expressing tiling, swizzling and vectorization through composable `(Shape, Stride)` abstractions -- the same design vocabulary. AMD built it in part because so many community and customer workloads depend on CUTLASS and CuTe.

That correspondence is the useful part. Nothing translates CuTe to FlyDSL automatically, but the concepts map: a layout is a layout, a tiled copy is a tiled copy. So a CuTe kernel is not a blank-page rewrite the way a CUTLASS-to-CK move can feel -- there is a structural correspondence to follow, which changes the estimate.

Treat FlyDSL as new: nightly wheels, blogs dated 2026, and a moving surface. Check what it currently supports for the specific kernels in question before promising anything, and prefer it where the source is CuTe-shaped rather than as a general-purpose target.

- Repository: https://github.com/ROCm/FlyDSL
- Porting guide: https://rocm.blogs.amd.com/software-tools-optimization/porting-hip-flydsl/README.html
- Worked example (fused MoE, MI300X): https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html

## TensorRT -> MIGraphX

TensorRT is an inference engine, not a kernel library, so "no HIP equivalent" was always the wrong frame: the AMD answer is MIGraphX (`ROCm/AMDMIGraphX`), a graph compiler and inference runtime that ingests ONNX (`parse_onnx`), TensorFlow and its own MXR format, compiles for AMD hardware, and exposes C++ and Python APIs.

What decides whether a project can move is HOW it uses TensorRT, and the two shapes are far apart:

- **ONNX in, tensors out.** The project loads an `.onnx`, builds an engine, and runs inference. This is the common case -- a detector class, a few hundred lines -- and MIGraphX does exactly this shape. Not a drop-in: TensorRT's builder/network/engine/execution-context model does not match MIGraphX's program/compile/eval, so the integration layer is rewritten. But it is bounded, and it is one file more often than not.
- **Deep TensorRT.** Custom plugins for unsupported layers, INT8 calibration flows, engine serialisation tied to a specific TensorRT version. Now the project's substance IS the TensorRT integration, and replacing it is a different project rather than a port.

There is a third option worth checking before either: if the project already goes through ONNX Runtime, the MIGraphX Execution Provider may need no application change at all. Note that the older ROCm Execution Provider was removed in ONNX Runtime 1.23 and MIGraphX EP is the supported path, so anything citing ROCm EP is out of date.

- Repository: https://github.com/ROCm/AMDMIGraphX
- Documentation: https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/
- ONNX Runtime provider: https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html

Related, and easy to miss for the same reason: `cv::cuda::*` is not a dead end either. OpenCV's CUDA modules -- cudaarithm, cudawarping, cudaimgproc, cudaoptflow, cudafilters, cudafeatures2d, cudaobjdetect, cudastereo, cudabgsegm, cudacodec -- have a HIP port validated on four architectures, upstream at opencv/opencv_contrib#4147. A project blocked on "OpenCV CUDA has no ROCm build" should depend on that port rather than be declined.

## What this means when deciding whether to port

"Not portable" should mean "no AMD-native target exists", not "hipify will not do it". When the blocker is CUTLASS or CuTe, say which target a reimplementation would aim at and roughly what it would cost, and let a person decide whether that is work worth taking on. A decline that does not name the alternative is hiding a decision rather than making one. Declines on CUTLASS/CuTe grounds made before FlyDSL was available may deserve revisiting.

Raw PTX tensor-core intrinsics are a separate case and stay blocked: `mma.sync`, `ldmatrix.sync`, Hopper `wgmma`/TMA and `tcgen05` have no HIP-level equivalent, and the AMD analogue is MFMA/WMMA reached through rocWMMA, CK or FlyDSL rather than a direct instruction swap.

The warp-level `mma.sync` m16n8kK family (with `ldmatrix`/`movmatrix` feeding it) is the exception that can be reimplemented directly when a project's API exposes the PTX fragment layout (an accumulator whose elements user code reads, a documented per-lane layout) so that rocWMMA's own fragment types cannot stand in. TornadoVM's HIP backend does it in device helpers (TC `Helpers.java`, the `tc_mma_*` HIP entries; source TornadoVM porting round 1):

- Keep fragments in the PTX layout. Emulate `ldmatrix` (.x4, .x2.trans) and `movmatrix.trans` with plain loads: they are pure data movement, and the emulation is bit-identical.
- In `mma`, move each A/B k-word (2 halves or 4 bytes) from the lane that holds it into the matrix instruction's operand layout with `__shfl`, multiply, and shuffle the result back into the PTX C/D layout. A shuffle's source lane may vary per lane, but the source REGISTER may not: shuffle every candidate register and select at the destination.
- CDNA wave64 holds two 32-lane warps per wavefront, and 16x16 MFMA reads operands from all 64 lanes, so a kernel launched with one warp per block (half the wavefront idle) breaks it. The 4x4 MFMA forms (`__builtin_amdgcn_mfma_f32_4x4x4f16`, `..._4x4x4bf16_1k` on gfx90a+, `..._4x4x2bf16` on gfx908, `__builtin_amdgcn_mfma_i32_4x4x4i8`) are sixteen independent 4 x 4 blocks of four lanes each (block = lane / 4; A_b[lane % 4][k], B_b[k][lane % 4], D_b[i][lane % 4]; measured on gfx950): eight blocks cover a warp's 16 x 8 tile without touching the other warp's lanes.
- Layouts measured on gfx950: `mfma_f32_16x16x16f16` lane l holds A[l%16][4(l/16)+i], B[4(l/16)+i][l%16], D[4(l/16)+i][l%16]. WMMA (from the ISA documentation): gfx11 lane l holds all 16 k of row/column l%16 (rows duplicated in both half-waves), D column l%16 rows 2i + l/16 (not measured); gfx12 lane l holds k = 8(l/16)..+7, D rows 8(l/16)+i (confirmed on gfx1201 for f16, bf16, iu8 and FP8 widened to f16 by TornadoVM's fragment harness, `projects/TornadoVM/probe/mma/`). gfx1250 has its own 16x16x32 WMMA forms.
- Do not test a WMMA/MFMA port for bit-exact fp32 results on integer-valued data, even where every product and partial sum is exactly representable. gfx1151 (RDNA3.5 WMMA, f16 and bf16 inputs) returned 1-ulp-off results (3e-8 to 9e-8 relative to sum|a*b|) for such inputs while lane layout was right (random data within fp32 rounding); gfx950 MFMA and gfx1201 WMMA were exact. Gate exactness per arch, and use a relative bound on gfx11. A single product is not safe either: three negative e5m2 subnormals (bytes 0x81-0x83) widened exactly to f16 and multiplied by 1.0 came back 1-2 fp32 ulps off on gfx1151, so a one-product-per-output "every byte" encoding check needs the gfx11 bound (1e-6 still pins the byte to its value) for exactly those three bytes; every e4m3 byte and every other e5m2 byte was bit-exact on gfx1151 and gfx1100 (TornadoVM probe), so keep the rest of the check exact rather than loosening the whole kind. gfx1250 FP8 WMMA (`v_wmma_f32_16x16x64_fp8_fp8`) sums with about 14-16 bits: random FP8 data reached 6.6e-5 relative, so bound FP8 there at about 1e-4.
- FP8 (E4M3 and E5M2, OCP) widens exactly to half, so one half-precision path serves every target, including gfx942 whose native FP8 is FNUZ.
- Select the path at compile time with the architecture macros (`__gfx942__`, `__GFX11__`; `__GFX12__` also covers gfx1250).
## Runtime-compiled kernels inherit nothing from the host build

Projects that compile device code at runtime -- an OptiX PTX pipeline ported to
HIP RT/hiprtc, or any hiprtc/comgr JIT -- have two compilers, and only one of
them is configured by the build system. Every macro the host build supplies
(`USE_ROCM`, `__HIP_PLATFORM_AMD__`, anything from `extra_compile_args`) is
absent from the runtime compile unless it is passed explicitly in the option
list handed to the JIT.

This bites the moment a header is shared between the ahead-of-time and runtime
paths. Collapsing a duplicated `params.h` back into the CUDA path's copy behind
`#ifdef USE_ROCM` is the right call for maintainability, and it will then take
the CUDA branch under the JIT and fail on `#include <optix.h>` (or on any other
CUDA-only include) with an error that looks nothing like a missing define. Pass
`-DUSE_ROCM=1` alongside the include paths in the JIT option vector.

Two related traps in the same place:

- **Guarding by compiler identity is not enough.** `__HIPCC_RTC__` and
  `__HIP_DEVICE_COMPILE__` are defined by the runtime compiler, but a shared
  header also gets included by the *host* translation unit, where neither is
  set. Guard on the platform macro, and set that macro in both compiles.
- **Scalar typedefs collide.** A CUDA header that writes its own
  `typedef unsigned long int uint64_t;` clashes under hiprtc with HIP's
  `using uint64_t = __hip_internal::uint64_t` (`unsigned long long`), because
  the HIP device headers are already in scope. Guard the typedefs out on the
  ROCm side rather than trying to match the type.

Diagnosing this is easy once you know to look: the runtime compiler's log is
usually printed to stderr and then swallowed, and the visible symptom is a
launch failure (`hipErrorInvalidHandle`) or an outright hang from launching a
module that was never built. Read the whole stderr, not the exception.

**The staged copy, not your working tree, is what runs.** A project that
compiles at runtime has to put the kernel sources somewhere the installed
package can find them, so its installer copies them next to `__init__.py` and
the JIT reads *that* copy. Edit a shared header, run the tests, watch them
pass, and you have tested the copy from the last install. This is not
hypothetical: a whole round of evidence for `diff-surfel-tracing` -- "the
forward works" -- was produced by a header the same commit deleted, and the
port turned out not to compile from its own sources at all. Two habits close
it: reinstall (not just rebuild) after touching anything the installer stages,
and, when a result surprises you, diff the package copy against the source
before believing either. Clearing the JIT cache does not help, because the
cache key hashes the staged source it just read.

Related: the JIT's on-disk cache may not be where you set it. HIP RT's
`hiprtSetCacheDirPath` is applied to a compiler that has already latched its
default, so the binaries land in a `cache/` directory relative to the process
working directory. Before concluding that a cache-clearing test cleared
anything, find the `.bin` files.

Source: `diff-surfel-tracing` (OptiX to HIP RT), gfx942.
