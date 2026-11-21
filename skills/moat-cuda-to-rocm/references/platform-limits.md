# Platform limits and other build classes

Device queries and counters that answer differently on AMD, kernels compiled at run time, and integrated APUs.

<!-- toc:begin -->
Contents:

- Device queries and counters
- Runtime-compiled kernels
- Integrated APUs
<!-- toc:end -->

## Device queries and counters

**AMD compute-capability values COLLIDE with NVIDIA arch numbers**, so CC-keyed dispatch
picks the NVIDIA path at RUNTIME on AMD. `hipDeviceAttributeComputeCapabilityMajor` and `hipDeviceProp_t::major` report 9 on gfx90a --
the same value as Hopper sm90 -- so code selecting a kernel config by "cc>=9 means Hopper
DPX/wgmma" tries to launch intrinsics AMD lacks, and a green compile does not catch it.
Gate arch-specific selection on `__HIP_PLATFORM_AMD__` and force the portable config; never
trust CC major/minor for arch dispatch on AMD. Found independently by both libmarv ports.
(MMseqs2, foldseek: the CUDASW++ short2-DPX Smith-Waterman selector.)

The values also collide AMD with AMD, so per-arch tuning tables cannot be keyed on them
either: gfx900, gfx906, gfx908 and gfx90a all report 9.0, gfx1100-gfx1102 report 11.0, and
gfx1150/gfx1151 11.5. A library that builds one tuned object per `--offload-arch` and picks
the table by major/minor hands the gfx900 kernels to a gfx90a in a fat binary ("device
kernel image is invalid"). Select by `hipDeviceProp_t::gcnArchName` with the feature suffix
stripped (`gfx90a:sramecc+:xnack-` -> `gfx90a`), and give every requested target without
its own table a generic build rather than a hard error. In CMake, `CMAKE_HIP_ARCHITECTURES`
from native detection lists one entry per GPU (`gfx90a;gfx90a;gfx90a;gfx90a` on a four-GPU
host): `list(REMOVE_DUPLICATES)` before creating per-arch targets (strategy-a-cmake.md).
(MMseqs2: libmarv's per-arch AMD aligner tables.)

**A query HIP reports as unsupported leaves a sticky error that the NEXT check reports.**
Reading or setting `cudaLimitPrintfFifoSize` (`cudaDeviceGetLimit`/`cudaDeviceSetLimit`) maps
to a call that returns `hipErrorUnsupportedLimit`. Code that ignores that return value and
later runs the usual
`launch; CHECK(cudaGetLastError());` sees the stale error at the launch check and blames the
kernel. Check or clear the error at the unsupported call, under the HIP branch only, so the
CUDA path is unchanged. The same reading applies to any failure reported by a last-error
check: confirm the blamed call's own return value first. (LichtFeld-Studio: a startup
memory report read the printf FIFO limit, and model initialization then failed with
`hipErrorUnsupportedLimit`.)

**`clock64()` is a 20-bit wrapping counter on RDNA3, so a cycle-bounded spin, delay or
timeout misbehaves.** HIP's `clock64()` is `__builtin_readcyclecounter()`. On gfx11 (RDNA3/3.5)
that lowers to `s_getreg_b32 hwreg(HW_REG_SHADER_CYCLES, 0, 20)`, which wraps about every
million cycles, while gfx9 emits a 64-bit `s_memtime`. Depending on how the comparison is
written, a `while (clock64() - start < cycles)` loop with `cycles` above 2^20 either exits at
the counter's first wrap (after anywhere from 0 to about a million cycles) or never
terminates, and which happens is intermittent because it depends on where the counter stood
at `start`. Nothing fails at the loop itself. What breaks is whatever the delay was holding
open: a test that gates a producer stream to prove a consumer waits for it passes or fails by
luck, a device-side timeout fires immediately, or the kernel hangs. Elapsed-time measurements
are silently wrong the same way. Time such loops with `wall_clock64()`, a 64-bit
constant-rate counter on every AMD GPU, and convert the duration with
`hipDeviceGetAttribute(..., hipDeviceAttributeWallClockRate, dev)` (kHz). Grep a port for
`clock64` and `clock()` in device code and treat any difference compared against more than
2^20 as this bug. (LichtFeld-Studio: tensor hardening `GateStream` delay kernel; FLAMEGPU2,
found in a test probe.)

**On RDNA, a nonzero occupancy does not mean the dynamic shared memory fits.** A common
CUDA idiom picks a shared-memory kernel when `cudaOccupancyMaxActiveBlocksPerMultiprocessor(
..., smem)` returns > 0 and otherwise falls back to an uncached kernel (OneFlow's softmax,
LayerNorm and RMSNorm, and the many projects that copied them). CUDA returns 0 once `smem`
exceeds the per-block limit. On gfx1100 (ROCm 10.1.0) the HIP query counts against the 128 KiB
WGP (`hipDeviceAttributeMaxSharedMemoryPerMultiprocessor`) and returns 1 for dynamic sizes
between the 64 KiB per-workgroup limit and 128 KiB. The launch then either fails with
`invalid argument` (dynamic > 64 KiB) or, when dynamic alone is <= 64 KiB but dynamic plus
static is over it, is ACCEPTED and aborts the queue with `HSA_STATUS_ERROR_INVALID_ALLOCATION`,
which kills the process. `hipFuncSetAttribute(MaxDynamicSharedMemorySize)` succeeds and is
not enforced. The result is a crash window in the problem size, not a threshold: in oneflow
fp32 LayerNorm forward failed at 16350..32568 columns and passed above it, because the query
drops back to 0 at 128 KiB. gfx90a and gfx942 (64 KiB LDS per CU, equal to the per-block limit)
answer 0 correctly, so wave64 runs there never see it. Fix: under `USE_HIP`, also require `smem + attr.sharedSizeBytes <=
hipDeviceAttributeMaxSharedMemoryPerBlock` before taking the shared-memory path. With a
compatibility header, map `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to a small template
that calls the HIP query and then reports 0 when dynamic plus static exceeds the per-block
limit: that restores CUDA's answer for every caller at once (oneflow also had the idiom in its
CTC decoder) and leaves the CUDA sources untouched. Check each candidate block size
separately, because static shared memory (block-reduce scratch) grows with it. Sweep widths
past 64 KiB/element-size on an RDNA card. Sweeping to 8192 only catches the
8-byte-per-column backward kernels. (oneflow, gfx1100)

**`__launch_bounds__(1024)` changes nothing on HIP and something on CUDA.** HIP-clang
already gives every kernel `amdgpu-flat-work-group-size="1,1024"` (`.max_flat_workgroup_size:
1024` in the code-object metadata) with or without the attribute -- check the metadata with
`--offload-device-only -S` before crediting it with fixing a 1024-thread launch failure. On
nvcc the same attribute caps registers per thread for every instantiation, which can move
the occupancy-based block-size choice and with it the reduction order. If a port keeps it,
put it behind a macro that is empty unless `USE_HIP`, so the CUDA kernels stay
token-identical to upstream. (oneflow, ROCm SDK clang 24)

**A kernel launched with fewer than 1024 threads needs `__launch_bounds__` on HIP to get its
registers.** This is the converse of the previous entry. HIP-clang assumes a 1024-thread block when
a kernel has no launch bounds and caps registers to fit it (64 VGPR + 64 AGPR on gfx942), where nvcc
gives a kernel up to 255 registers regardless. A register-heavy kernel (FlashAttention-2 tiles, 128
fp32 accumulators per lane) launched with 128 threads then spills to scratch. Fix:
`__launch_bounds__(<actual block size>)` under the HIP guard; verify with
`-Rpass-analysis=kernel-resource-usage`. A second spill source showed up in the same kernel:
emulating a matrix instruction with shuffles inside fully unrolled fragment loops let the scheduler
keep every shuffle result live (4.4 KB/lane scratch at 256 VGPR + 256 AGPR); `#pragma unroll 1` on
the emulation's inner loop took it to 0 in a one-instantiation probe with the helper inlined. The
shipped module keeps the helper out of line (for compile time) and its prefill kernels report
240-848 B/lane of private segment (848 B for the instantiation the end-to-end run uses; read with
`llvm-readelf --notes` on the gfx942 code object). (Quest, ROCm SDK 10.1)

**A host-side launch heuristic keyed on the register count can break a kernel whose block width is a
compile-time constant.** A launcher that doubles the block when `cudaFuncGetAttributes(...).numRegs
<= 128` is harmless for kernels that read `blockDim.x`, but a kernel that hard-codes `constexpr int
nsq_per_block = 256` then runs every task twice and overlaps its per-thread shared-memory slices.
NVIDIA builds of those kernels sat above 128 registers, so the mismatch was latent; on gfx90a/gfx942
one landed at exactly `.vgpr_count: 128` and the analytic gradient came out wrong by orders of
magnitude, different on every run, while gfx1100 passed. When results are wrong and nondeterministic
on one wave size only, compare each launcher's block geometry with the constants inside the kernel
it launches; `llvm-readelf --notes` on the extracted code object gives the VGPR count. (gpu4pyscf)

**Kernels that opt in to more than 64 KB of shared memory per block cannot launch on most AMD
GPUs.** CUDA lets a kernel raise its dynamic shared memory with
`cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)` up to `sharedMemPerBlockOptin` (99 KB to
227 KB on recent NVIDIA GPUs). AMD gives a block 64 KB of LDS on gfx90a, gfx942 and RDNA (gfx950 has
160 KB), and HIP counts the kernel's static `__shared__` arrays against the same limit, so a launch
whose static plus dynamic size exceeds it fails even when the dynamic part alone fits. gpu4pyscf saw
the failure at `hipFuncSetAttribute`, with invalid argument; oneflow on gfx1100 with ROCm 10.1.0
(the entry "On RDNA, a nonzero occupancy does not mean the dynamic shared memory fits" above) saw
that call succeed unenforced and the launch itself fail or abort the queue, so check both return
values and do not rely on either call to refuse. Find the cases by computing the driver's size
formula over the whole parameter range and adding each kernel's `.group_segment_fixed_size` (from
`llvm-readelf --notes` on the extracted code object). CuPy-ROCm has no `sharedMemPerBlockOptin`, so
a Python guard on it needs a fallback. Fix for the over-limit cases only, chosen at run time from
`hipDeviceAttributeMaxSharedMemoryPerBlock` plus `hipFuncGetAttributes().sharedSizeBytes`: move the
largest scratch arrays that the inner loop does not read to a per-block global-memory workspace,
size the grid to the resident blocks (`hipOccupancyMaxActiveBlocksPerMultiprocessor` times CUs) with
each block looping over tasks so the workspace does not scale with the task count, and allocate it
with `hipMallocAsync`/`hipFreeAsync` on the launch stream. Put the variant in a HIP-only file so the
CUDA translation unit stays identical. In gpu4pyscf's ECP derivative kernels the variant gave
bit-identical results and, at a few hundred tasks or more, ran in 0.6x the time of the LDS-heavy
kernel, because its smaller LDS footprint keeps more blocks resident. (gpu4pyscf)

**AMD's 64 KB LDS breaks "half the per-SM shared memory" tile heuristics, and the dispatcher may
then launch nothing.** FlashInfer-derived dispatchers size tiles from
`cudaDevAttrMaxSharedMemoryPerMultiprocessor / 2` (two blocks per SM). MI300X reports 64 KB per CU
for `hipDeviceAttributeMaxSharedMemoryPerMultiprocessor` (verified), so half leaves room for no tile
at head_dim 128, and FlashInfer's `SWITCH_NUM_FRAGS_Z` then prints `Unsupported max_frags_z` to
stderr and returns success without launching: the output tensor is whatever `torch::empty` held.
Size for one block per CU under the HIP guard (the largest tile is exactly 64 KB). Quest did not
record the RDNA values; the occupancy entry above measured 128 KiB for that attribute on gfx1100, so
query it on each target (gfx1100, gfx1151) before reusing the sizing there. Check every such
`SWITCH_*` default branch in a port: an unlaunched kernel looks like a numerics bug. (Quest)

**gfx12 work-group IDs y and z are 16 bits: a grid y or z above 65536 is rejected or silently
wraps.** On gfx12 (gfx1200/gfx1201) the compiler reads `blockIdx.y` as `ttmp7 & 0xffff` and
`blockIdx.z` as `ttmp7 >> 16`; gfx11 and gfx9 receive 32-bit IDs in SGPRs (check with `hipcc
--cuda-device-only -S` per arch). `hipGetDeviceProperties` reports `maxGridSize = (2^31-1, 65535,
65535)`; 65536 still works. Above it the launch API decides what happens:
`hipLaunchKernel`/`hipExtLaunchKernel` return hipErrorInvalidConfiguration and run nothing
(unchecked, that is silently missing output: MIOpen's RNN bias op,
https://github.com/ROCm/rocm-libraries/issues/10551), while `hipModuleLaunchKernel` and
`hipExtModuleLaunchKernel` return success and run every work-group with the y/z index taken modulo
65536, so work-group 65536 + i redoes work-group i and the tail is never touched. A kernel of your
own that needs more than 65536 work-groups in y or z must fold them into x. A library kernel can hit
it too: rocBLAS complex GEMM does (libraries.md, "On gfx12, rocBLAS complex GEMM with n > 2^21
leaves columns unwritten"). (gpu4pyscf)

## Runtime-compiled kernels

**Runtime PTX plus the CUDA Driver API is a THIRD build class** beyond Strategy A and B.
Some projects never compile `.cu` into the binary -- they emit PTX text, embed it, and load
it at runtime via `cuModuleLoadData`/`cuLaunchKernel`, often through a hand-written FFI
binding. Port to the HIP Driver API: compile each `.cu` to a code object with
`hipcc --genco --offload-arch=<arch>` (not PTX); embed the bytes length-delimited and
binary-safe (base64) keyed by `gcnArchName`, suffix-stripped, NOT by "highest compute
capability"; pass `hipModuleLoadData` a pointer to the byte BUFFER, since a C-string
conversion truncates at the first NUL; select by `gcnArchName` with a trial-load fallback;
and launch via `hipModuleLaunchKernel`. Two cgo-specific traps: `hiprand.h` does
`typedef __half half` and AMD's `__half` is C++-only, breaking the C parse, so alias
`typedef _Float16 __half;` in the cgo preamble (it must sit in the comment block touching
`import "C"`); and `hipCtxSynchronize`/`hipCtxGetApiVersion` return `hipErrorNotSupported` on ROCm 7.2.1, so
synchronize via `hipStreamSynchronize`. (mumax3: Go + cgo + runtime code-object loading.)

**NVRTC (CUDA C++ generated at run time) ports to hipRTC.** It is the same build class with a run-time compiler instead of embedded PTX.
Bind it by absolute path under `ROCM_PATH`/`HIP_PATH` before any bare name (on Windows the hipRTC DLL
carries the version in its name, e.g. `hiprtc0702.dll`, with a digit layout that varies by release, so scan `bin` for `hiprtc0*.dll`). hipRTC declares `half`, `__hip_bfloat16` and
the fp16/bf16 intrinsics itself, but `#include <hip/hip_fp16.h>` fails without `-I<rocm>/include`:
guard any include with `__has_include`. Spellings that differ or are missing in hipRTC:
`__float2bfloat16` (no `_rn`), no `__double2half`, `__dp4a` or `llmin`/`llmax`. Pass
`--offload-arch=<gcnArchName>` explicitly (read `gcnArchName` from `hipDeviceProp_t` through the
versioned `hipGetDevicePropertiesR0600`, whose layout that version fixes), and read device attributes
by their HIP enum values, which differ from CUDA's. HIP's context API is deprecated: a backend built
around primary contexts can stand the device in for the context and make it current with
`hipSetDevice`. clang contracts `a * b + c` by default as NVRTC does; `-ffp-contract=off` is the
`--fmad=false` equivalent (faults-numerics.md has where the two compilers still differ). (TornadoVM: Java FFM bindings, TC-generated HIP C++.)

### Runtime-compiled kernels inherit nothing from the host build

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

hipRTC is not NVRTC in ways the API does not show, so a port that maps `nvrtcXxx` to
`hiprtcXxx` and stops will build, load a 0-byte or wrong code object, and render nothing.
A project whose whole pipeline is runtime-compiled hits all of these at once (CuRast):

- **No `-default-device` equivalent.** NVRTC can make every unmarked free function
  `__device__`; clang cannot. Wrap each block of helper functions in
  `#pragma clang attribute push (__attribute__((device)), apply_to=function)` / `pop`,
  guarded by `#if defined(__HIPCC_RTC__)`. The pragma conflicts with `__global__`, so the
  regions must EXCLUDE the kernel entry points: regions, not whole files.
- **Do not count on fixed-width types being predeclared.** NVRTC supplies
  `uint32_t`/`uint64_t` implicitly. Under hipRTC a header that used them without
  `#include <cstdint>` failed with a cascade of "unknown type name" in CuRast, while the
  typedef collision above shows a hipRTC that had `uint64_t` in scope; which names arrive
  for free evidently depends on the release and on what else the source includes. Add the
  `<cstdint>` include: it is a real omission, it is correct either way, and it helps the
  offline build too.
- **A translated fast-math flag assumes finite math.** `-ffast-math` implies
  `-ffinite-math-only` under clang where NVRTC's `--use_fast_math` does not, and sentinel
  values then vanish (faults-numerics.md).
- **The host runtime headers are not on the include path**, so a shared compat header must
  skip its `hip/hip_runtime.h` include under `__HIPCC_RTC__` and provide the handful of
  intrinsics that came with it (`__float_as_uint` and friends) as macros. Macros, not inline
  functions: one combined TU can reach the same header by two relative paths, where
  `#pragma once` does not dedupe and duplicate definitions are an error.
- **Do not pass `-fgpu-rdc`** (it produces 0-byte code objects). Pass the clang resource
  directory and the ROCm include root as absolute `-isystem`/`-I` (derive them from
  `ROCM_PATH`), and NUL-terminate the `hiprtcGetProgramLog` buffer yourself.

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

(diff-surfel-tracing: OptiX to HIP RT, gfx942.)

## Integrated APUs

### An integrated APU reports the whole shared memory pool as device memory

On an integrated APU `hipMemGetInfo` reports the shared system pool (tens of GB), so a
library that eagerly reserves a fraction of *reported* device memory can ask for most of
the machine. Suspect that shape when an APU run stalls at first allocation, and also when
it is uniformly slow: measured on a gfx1151 APU with 72 GB of unified memory, with a library
whose default pool asked for 90% of free device memory (about 65 GB), the reservation
succeeded and the suite then ran roughly 2x slower on the light tests and 3-4x on the heavy
ones than the same code with no pooling. That comparison is pooling at a 90% default
against no pooling at all, so it does not apportion the cost between the reservation and
the allocator; measure before publishing numbers. The fix belongs to the application, not
the port: pass the library's own pool configuration (a small initial fraction or a fixed
size). Do not change a shared default that is correct for discrete cards, and do not
special-case APUs in shared code; document the sizing where the project documents its
defaults. (HEonGPU)

### An integrated APU can crash the device linker under build parallelism

    clang: error: amdgcn-link command failed due to signal (use -v to see invocation)
    clang: note: diagnostic msg: Error generating preprocessed source(s).

This is resource exhaustion, not a broken toolchain and not bad code. It showed up on a
20-CU APU building 14 template-heavy rasterizer variants at `MAX_JOBS=8`, always in the
device-link step; the same variant alone at `MAX_JOBS=2` built immediately, and the whole
14-variant sweep at `MAX_JOBS=4` passed with no failures. An APU shares one memory pool
between CPU and GPU, so concurrent device links compete with everything else on the box.

Halve the job count before investigating anything else, and do not read a signal-killed
`amdgcn-link` as evidence about the port. An APU can also hard-reset under heavy build
parallelism plus sustained GPU load; a build that takes the machine down is the same lesson
one notch further along.

### A low-CU integrated GPU can outrun a fixed upstream test timeout without any fault

A 20-CU integrated APU (gfx1151) can be genuinely, correctly ~40x slower than datacenter
and desktop cards for the single heaviest kernel sequence in a suite, and a project's own
`TIMEOUT` on `add_test`/`gtest_discover_tests` is usually sized against those faster cards.
Found porting HEonGPU: `ctest` reported `TFHE_Gate_Boots` as `***Timeout` at the project's
hardcoded 30-second per-test limit (`test/CMakeLists.txt`, identical on the CUDA path),
while a gfx942 card finishes the entire 20-test suite in 13-15s. Running the same
executable directly, outside ctest's harness, showed it was not hung or wrong -- it passed
in 112.9s, `[ PASSED ]` from gtest's own assertions.

Diagnose by running the flagged executable directly before concluding anything: `rc=0`
plus a `[ PASSED ]`/correct-value line after the harness's cutoff means "too slow for this
budget," not "broken." A hang or wrong answer looks different -- no completion line ever
appears, or it appears with wrong values.

Getting a clean harness-level pass count (worth doing when the bar is an exact N/N matching
the other platforms) may need a throwaway local bump of the test's own `TIMEOUT` property:
ctest's `--timeout <seconds>` flag does **not** override an explicit per-test `TIMEOUT` set
via `set_tests_properties`/`gtest_discover_tests` -- confirmed, `ctest --timeout 180`
against a `TIMEOUT 30` property still printed `***Timeout 30.07 sec`. Edit the property,
reconfigure (cheap -- CMake only regenerates the test files if no source changed), rebuild
only the one affected test target so `gtest_discover_tests`'s post-build discovery step
re-runs and picks up the new property, run ctest, then `git checkout --` the file and verify
`git status --porcelain` is empty. This is not a port fix and must not ship -- it exists
only to prove the harness-level count. (HEonGPU)
