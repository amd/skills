# Faults: textures and graphics interop

Texture objects, the GPUs that have none, and sharing memory with OpenGL and Vulkan.

<!-- toc:begin -->
Contents:

- Texture objects
- GPUs without texture support (gfx94x and later)
- OpenGL interop
- Vulkan interop
<!-- toc:end -->

## Texture objects

**Texture pitch alignment is 256 bytes on AMD against 32 on NVIDIA**, and it bites in two
distinct ways.

- At the BIND: pitched 2D texture binds need 256-byte rows, so widths that work on CUDA can
  fail. If a kernel only point-samples, a linear (`tex1Dfetch`-style) bind avoids pitch
  entirely. (colmap BindTexture2D.)
- Through the ATTRIBUTE: `cudaDevAttrTexturePitchAlignment` (`hipDeviceAttributeTexturePitchAlignment`) reports 256 on gfx90a,
  so libraries deriving a row pitch from it pad more on AMD -- a tight 640-byte uchar row
  becomes 768. Tests that fill the valid region, run the op, then compare the WHOLE strided
  buffer against a zeroed reference bake in the CUDA pitch and mismatch. If nothing is
  actually bound to a hardware texture object the 256B pitch is unnecessary: clamp the
  attribute query to 32 in a compat shim so AMD pitches match CUDA. (CV-CUDA: Erase, SIFT,
  Gaussian, FindHomography, Normalize, TensorBatchWrap, many Interpolation cases.)

**A hardware linear-filter texture over an element-read float array is ARCH-SPECIFIC, not a
blanket AMD limitation.** On gfx90a (ROCm 7.2.1), `filterMode=cudaFilterModeLinear` with
`readMode=cudaReadModeElementType` over a float array is REJECTED at creation
(`hipCreateTextureObject` -> "operation not supported"). On gfx1100 (ROCm 7.2.1) and
gfx1201 (ROCm 7.14) the same texture is ACCEPTED and genuinely hardware-interpolates, so
RDNA needs no fallback. For the rejecting arch, create it `cudaFilterModePoint` and lerp in
software, matching CUDA's unnormalized -0.5 texel-center convention: coordinate c samples
`floor(c-0.5)` and `floor(c-0.5)+1`, weight `(c-0.5)-floor(c-0.5)`. Put it behind the
project's fetch helper, `__GFX9__`-gated. Empirical, no upstream issue filed. SCOPE: this is
a property of the hardware TEXTURE OBJECT, not of linear filtering as such -- a field
sampled through a software accessor in device code (a NanoVDB `getAccessor().getValue()`)
is unaffected and needs no fallback. (popsift readTex; TIGRE probe; anari-visionaray.)

**Gate the hardware-vs-software texture path on a VERIFIED runtime self-test**, never on
creation success or an `_WIN32`/version proxy -- and probe the EXACT configuration you use.
Detecting at runtime lets one binary use hardware where present, which is real perf: the
interpolated fetch can be ~90% of kernel time. Two traps. Creation success is NOT proof:
some hardware accepts a linear fp32 texture and silently point-samples, so the self-test
must SAMPLE a known ramp and confirm it interpolates (1.5 at the midpoint, not 2.0). And
the probe must mirror the real configuration exactly -- on gfx90a/ROCm 7.2.1 a small 1D
`cudaAddressModeClamp` linear texture is ACCEPTED while the real 3D,
`cudaAddressModeBorder`, `cudaReadModeElementType` one is REJECTED, so a 1D probe gives a false "supported" and the real texture
then fails at creation. Sample an interior texel so border padding does not pollute the
lerp. Drive both the `filterMode` and the fetch helper from one cached verdict so they
cannot disagree. RDC-free mechanism: a header-local `static __constant__` flag plus a
`static inline` host sync helper using `hipMemcpyToSymbol(HIP_SYMBOL(flag), ...)`, so each
TU binds its own copy. The once-per-process init log also makes the port self-report each
arch's verdict during validation. (TIGRE `cuda_to_hip.h` + gpuUtils.cu.)

**A layered `cudaArray` collapses across kernel launches -- use a non-layered 3D array.**
CONFIRMED bug, gfx90a/CDNA2, ROCm 7.2.1, ROCm/clr#275. A
`cudaArrayLayered | cudaArraySurfaceLoadStore` float array written one layer at a time via
`surf2DLayeredwrite` reads back the LAST-written layer for EVERY layer index on a later
launch -- through `tex2DLayered`, `surf2DLayeredread` and host `hipMemcpy3D` alike.
`hipDeviceSynchronize` between writes and recreating the texture do not help. Fix: drop
`cudaArrayLayered` and allocate with `hipMalloc3DArray`, accessed via
`surf3Dwrite`/`surf3Dread`/`tex3D` with the layer as a real z coordinate (a tall 2D array,
W x H*L, also works). Map each `surf2DLayeredwrite(v,s,x,y,layer)` to
`surf3Dwrite(v,s,x,y,layer)` once inside the compat header so call sites are untouched.
CUDA keeps the real layered array byte-for-byte. AMD's candidate fix
(ROCm/rocm-systems#6683) corrects only the `surf2DLayered` builtins -- `tex2DLayered` uses a
different builtin and may collapse independently, so re-verify per ROCm version and keep
the workaround until proven fixed on your stack. NOT confirmed on RDNA; re-run the repro on a
wave32 part to establish arch scope. (popsift: the Gaussian pyramid and DoG were
layered arrays, and without this the DoG was all-zero.)

## GPUs without texture support (gfx94x and later)

**gfx94x+ (CDNA3 and newer: MI300-class) disables the HIP image/texture object API at COMPILE
TIME, not just at runtime.** ROCm's own headers define it out:
`hip/amd_detail/amd_device_functions.h` has
`#if !defined(__HIP_NO_IMAGE_SUPPORT) && defined(__gfx94plus_clr__) #define
__HIP_NO_IMAGE_SUPPORT 1 #endif`, which marks every `tex1Dfetch<T>` overload in
`texture_indirect_functions.h` `__attribute__((unavailable(...)))`. Any call is a hard
compile error (`'tex1Dfetch<T, nullptr>' is unavailable: The image/texture API not supported
on the device`), gated purely on `--offload-arch`, so the identical source compiles clean on
gfx90a/gfx11xx/gfx12xx and fails ONLY on gfx942/gfx950 -- this surfaces as a validation
failure on exactly one arch even though nothing about the build command differs (confirm the
arch pin is correct before assuming a real regression; then check for `__gfx94plus_clr__` in
the toolchain headers). A project doing a bare point-lookup through a texture object has a
portable, unconditional fix, and the qualifying criterion is the ACCESS PATTERN, not the table
size: point sampling only, no hardware filtering, no normalized coordinates, no address/border
modes (arrayfire's FAST LUT is 64 KiB and still qualifies, so do not read "LUT" as "small").
Replace `tex1Dfetch<T>(tex, n)` with a plain indexed read through a `const T*` and drop the
`hipTextureObject_t` create/destroy. Apply it unconditionally rather than under an `#ifdef`:
the results are bit-identical everywhere and one path is what gets tested. What that costs on
the arches where textures do work is a THROUGHPUT question the compile does not answer (the
texture path reads through the read-only cache), so if the tables are hot, time the affected
tests before and after on a texture-capable arch instead of asserting the texture bought
nothing. A project that genuinely needs
hardware linear filtering has no CDNA3 hardware path at all and needs the software-lerp
fallback below unconditionally on gfx94x+. (arrayfire: `fast.cu`/`orb.cu`/`regions.cu` FAST/
ORB/connected-components kernels, all plain LUT point-lookups with no filtering, on MI300X.)
A port that already point-samples and lerps in software for that fallback is one step from
the buffer form: the texture then supplies only two point fetches and the border address
mode, so allocate the table with one trailing zero element standing in for the border texel
and index it directly; the arithmetic, and so every result, is unchanged on every arch.
Check the index range first -- here the caller's window bound kept the low index >= 0, so
only the high side could touch the border. (gpuRIR `image_sample_lut`, gfx942.)

**A project that reads EVERYTHING through texture objects, with address modes, needs more than a pointer swap.** The emulation is still exact when no binding relies on hardware filtering (LEAP: its HIP port already created every texture point-mode and blended corners in software, so every fetch was a point fetch). Keep the handle type and every kernel signature: on an image-less GPU have the texture loader cudaMalloc one buffer holding a small header (extents, address mode) followed by the dense data, and return that buffer as the `hipTextureObject_t` (and as the `hipArray*` the call sites free). The device accessor applies the address mode in float before converting to an index (clamp: `fminf/fmaxf` to `[0, n-1]`; border: any axis outside `[0, n)` returns 0), reads `floor(c)` for a point fetch, and maps a non-finite coordinate to a finite one. Casting the header and data pointers to `__attribute__((address_space(4)))` (read-only constant) lets the compiler hoist the header into one `s_load_dwordx4` per kernel instead of reloading it around stores.

Measured on gfx90a (MI250X) against the point-mode texture build, LEAP: the emulation is exact (every forward, back and FBP output and every point-sampled kernel bit-identical, including point fetches at fractional coordinates, so the AMD sampler does not quantize where `floor` does not; the only differences were FMA-contraction ULPs, rel ~1e-7, from inlining different code), but 2-3x SLOWER where it feeds software trilinear blends (projection 3.0-3.3x, backprojection 1.8-2.5x): eight bounds-checked row-major loads replace eight `image_sample`s against a tiled image. A lone point read costs about what a hardware point fetch does. So unlike the LUT case above, do not apply it unconditionally. Use a dual path:
- Device: select the buffer accessor under `#if __HIP_NO_IMAGE_SUPPORT`, which ROCm's `amd_device_functions.h` defines per target in the device pass (gfx942/gfx950, via `__gfx94plus_clr__`); every other target keeps the texture helpers byte-for-byte. A fat binary then carries the right code for each arch.
- Host: the loader picks the allocation at run time from `hipDeviceAttributeImageSupport` of the current device. Make a mismatch impossible, not just unlikely: define a `__device__ int` in the loader's translation unit whose initializer is chosen by the same `#if`, read it with `hipMemcpyFromSymbol` (that returns the value from the code object loaded for THIS device), and refuse to create a handle if it disagrees with the attribute. Cache the answer per device.
- Free/destroy: keep the call sites by redirecting `cudaFreeArray` and `cudaDestroyTextureObject` (CUDA spelling, so hipify renames them too) as FUNCTION-LIKE macros, so the helper can still call the runtime function as `(cudaFreeArray)(a)`. Dispatch on a host registry of live buffers, and free a buffer only when both halves of its free/destroy pair have been released: call sites typically free the array before destroying the handle, and freeing at the first call would let a new texture object reuse the address before the second.
To test the image-less path on a texture-capable GPU, build a throwaway copy with `__HIP_NO_IMAGE_SUPPORT` forced to 1 and the attribute forced to 0; forcing only the device side must hit the refusal. (LEAP.)

**A global-memory texture replacement must COPY the source, never alias it.** Creating a texture from a device buffer copies it into a `cudaArray`, so callers treat the texture as a snapshot: they free the source right after the load, or reuse it as the output of the kernel that samples the texture. A fallback that stores the caller's pointer to save the copy turns those callers into use-after-free (a memory fault, and a dead context that zeroes every later call in the process) or into a kernel reading its own output (a silent wrong answer). Copy device-to-device into a buffer the handle owns and free it with the handle; the copy cost 0-2.5% of a projection or backprojection on an MI300X. Audit for it by grepping each loader call for a `cudaFree`/`setToConstant`/output alias of the source before the matching free. Because a fault kills the context, a test script that runs many cases can attribute the zeros or unchanged outputs of LATER cases to the wrong cause; check the first failing case. (LEAP: upstream's `__USE_NOTEX` sampler aliased in the device-pointer `loadTexture`; `consistency_cost` returned 0 and attenuated backprojection faulted on gfx942.)

Two things make that replacement cheap to get right. A pre-existing non-texture path for a
type textures cannot carry (CUDA has no `double` texture, so such kernels usually already
have a `double` specialization reading the pointer directly) is proof the algorithm is
correct without the texture, and it is the specialization to fold the generic case into
rather than to keep. And an RAII texture holder normally owns a copy of the source array
purely to hold its refcount for the kernel's lifetime: keep that member, drop only the
handle, so the holder's shape and its call sites are unchanged. Afterwards, grep the
project's compat header -- removing the last texture user typically orphans a whole block of
`cudaTextureObject_t`/`cudaResourceDesc`/channel-format aliases that should go in the same
change. (arrayfire.)

## OpenGL interop

**A graphics-interop device pointer dies at the unmap.**
`cudaGraphicsResourceGetMappedPointer`/`hipGraphicsResourceGetMappedPointer` returns a pointer that is
only valid until the matching `...GraphicsUnmapResources`. CUDA commonly keeps such a pointer usable
afterwards, so the "map once at registration, unmap, keep the pointer" idiom copied out of CUDA GL-interop
samples survives upstream. On ROCm the pointer is not a valid HIP allocation once unmapped: `hipMemcpy`
against it returns `hipErrorInvalidValue` and moves NOTHING. The tell is an UNCHECKED interop copy -- the
error code is discarded, so the failure surfaces arbitrarily far away as a physics, geometry or rendering
bug (in Velvet the seed copy from the mesh VBO left every particle at the origin, the solver exploded into
NaN, and the reverse copy never reached the VBO so the mesh appeared absent). Audit every
`cudaGraphics*`/`hipGraphics*` user for the map scope, wrap each interop copy in the project's error check,
and scope map/unmap tightly around each access. Fix it portably with no `#ifdef`: a permanently mapped
resource is undefined on CUDA too, and leaving it mapped while the graphics API draws from it is a second
bug. (Velvet: `VtBuffer.hpp` `VtRegisteredBuffer::registerBuffer`.)

**Compute-only CDNA has no graphics pipeline**, so OpenGL/Vulkan-interop apps build but
cannot run on gfx90a. Datacenter parts have no display engine: radeonsi refuses a GL context
on a compute chip, only software GL is available, and a software-GL buffer cannot be
registered with HIP. Anything sharing a buffer via `cudaGraphicsGLRegisterBuffer`/`hipGraphicsGLRegisterBuffer` fails at
registration with `hipErrorInvalidValue`, and kernels gated behind it are unreachable. The
build links cleanly, which is the trap. Mark gfx90a non-viable and scope it out of the PR
claim rather than recording a build-only pass; such a port runs on graphics-capable RDNA
only. The exception is a headless-compute refactor decoupling the solver from interop --
a larger port, rarely worth it for an interactive renderer. (Velvet: XPBD cloth solver
writing particle positions into GL VBOs.)

## Vulkan interop

**External memory ports 1:1; external semaphores do not.** `hipImportExternalMemory` with
OPAQUE_FD/OPAQUE_WIN32, `hipExternalMemoryGetMappedMipmappedArray` + surface objects, and
mapped buffers round-trip Vulkan data bit-exactly (RDNA3 Linux and Windows). Semaphores:
`hipImportExternalSemaphore` rejects the timeline handle types (`TimelineSemaphoreFd`,
`TimelineSemaphoreWin32`) in a front-end range check on both OSes, and on Linux nothing
imports at all (binary OPAQUE_FD returns not-supported down to the KFD). On Windows, do not
import a timeline VkSemaphore as `hipExternalSemaphoreHandleTypeOpaqueWin32`: the import
succeeds and a single wait/signal honors values, but with several values in flight it has
been reported to let HIP read a buffer before Vulkan finished writing it (gfx1102, about 1
run in 25, stale data and no error; ROCm/hip#3909). 500 queued bounces on gfx1151 did not
reproduce it, so a short passing test proves nothing. Order on the host on both OSes: import
the exported handle into a second timeline VkSemaphore on the app's device
(`vkImportSemaphoreFdKHR`, or `vkImportSemaphoreWin32HandleKHR` with OPAQUE_WIN32, which
leaves the NT handle with the caller), wait = `vkWaitSemaphores` before enqueueing HIP work,
signal = `hipStreamSynchronize` then `vkSignalSemaphore`. Routing the four
`cuda*ExternalSemaphore*` calls to such an implementation from the compat header leaves every
call site untouched. A blocking host wait deadlocks if the Vulkan submit that signals the
value comes later on the same thread, so check submission order at each wait and put a
timeout on the host wait. If host ordering costs too much on Windows, a GPU-ordered route
has been reported stable but is untested here: create the fence in D3D12 (`CreateFence` +
`CreateSharedHandle`; Vulkan cannot export one) and import it into HIP as
`hipExternalSemaphoreHandleTypeD3D12Fence` and into Vulkan with `D3D12_FENCE_BIT`. HIP has no
`cudaDevAttrTimelineSemaphoreInteropSupported`. (LichtFeld-Studio, from the interop spikes
and the ROCm/hip#3909 report)

**HIP VMM memory exported to Vulkan works on Linux.** `hipMemCreate` with
`hipMemHandleTypePosixFileDescriptor` + `hipMemExportToShareableHandle` gives an fd that RADV
imports as OPAQUE_FD (and DMA_BUF), correct in both directions. Differences from CUDA: HIP
has no handle-type-supported attribute, `hipMemAllocationProp` spells the field
`requestedHandleType` (singular), `hipDeviceptr_t` is `void*` (CUDA's `CUdeviceptr` is an
integer, so VA arithmetic needs adapters), and AMD's minimum granularity is 4 KiB, not 2 MiB.
(LichtFeld-Studio)

**Match HIP and Vulkan devices by PCI location (Linux) or LUID (Windows), never by UUID.**
HIP's UUID is the board serial, RADV's encodes the PCI address, so a CUDA-style
`deviceUUID` comparison never matches and a multi-GPU host picks the wrong Vulkan device.
(LichtFeld-Studio)
