# Fault classes: AMD is strict where CUDA is lenient

The real semantic differences. Most porting bugs are here, not in symbol names, and they
surface at runtime on a build that compiled cleanly.

Sections match the index in SKILL.md, so a line that looked relevant there leads straight
to its entry here. Each entry names the project it was learned on.

## Wavefront and warp semantics

**Never hardcode 32.** NVIDIA warps are always 32 lanes. AMD wavefronts are 64 on CDNA
(gfx90a, gfx94x) and 32 on RDNA (gfx10xx, gfx11xx, gfx12xx).

- Host code (launch geometry, host-side shared-memory sizing) queries at runtime:
  `hipGetDeviceProperties(&prop, dev); prop.warpSize`, or `hipDeviceAttributeWarpSize`.
  PyTorch exposes `at::cuda::warp_size()`.
- Device code uses a compile-time per-arch constant. `__GFX9__` is defined only during
  device compilation; there is no `__AMDGCN_WAVEFRONT_SIZE__` in ROCm 7.2.x, so the
  `__GFX*__` guards are the supported selector:

        #if defined(__HIP_PLATFORM_AMD__)
        #if defined(__GFX9__)
        static constexpr int kWarpSize = 64;   // CDNA: gfx90a, gfx94x
        #else
        static constexpr int kWarpSize = 32;   // RDNA: gfx10xx, gfx11xx
        #endif
        #else
        static constexpr int kWarpSize = 32;   // CUDA
        #endif

- **Never route both host and device through one shared constant.** A CMake-injected
  `-DWARP_SIZE=64` forces single-arch and mis-sizes host buffers in a fat binary. Device
  width comes from the `__GFX*__` guards, host width from the runtime query.

**A static array sized from the warp width needs the bound that matches the DIRECTION the
width enters the formula, and getting it backwards writes out of bounds.** A quantity
scaling WITH the width (`n * warpSize`, a per-lane array) needs the UPPER bound, 64
(PyTorch's `C10_WARP_SIZE_UPPER_BOUND`). A quantity scaling INVERSELY (`blockDim /
warpSize`, warps-per-block, anything sized per-warp) needs the LOWER bound, 32: a smaller
wavefront means MORE warps, so a 64 bound UNDER-allocates on wave32 and the device writes
past the shared-memory region. Ask which way the divide goes before picking the constant.
(SCAMP: `cov_handoff`, `2 * warps_per_block` scalars sized with 64, under-allocated on
every wave32 device.)

**Lane masks must be 64-bit where the API takes one** -- `__shfl*`, `__ballot`,
`__activemask`. A `uint32` mask is wrong on a 64-wide wavefront.

The `*_sync` warp functions go further and refuse it at compile time. HIP's
`__shfl*_sync`, `__ballot_sync`, `__any_sync`, `__all_sync`, `__match_*_sync` and
`__reduce_*_sync` `static_assert` that the mask type is 64 bits wide, so the CUDA idiom
`__shfl_down_sync(0xffffffff, v, o)` -- an `unsigned int` literal -- fails on every AMD
target, wave32 included, with "The mask must be a 64-bit integer. Implicitly promoting a
smaller integer is almost always an error." hipify leaves the literal alone, so a
"hipify ran clean" tree still does not build. CUDA's mask parameter is 32-bit, so one
literal cannot serve both: define the full mask once in the compat header
(`0xffffffffffffffffull` under `__HIP_PLATFORM_AMD__`, `0xffffffffu` otherwise) and use
that name at every call site. On wave32 HIP ignores the upper half. Defining
`HIP_DISABLE_WARP_SYNC_BUILTINS` also silences the error, but it removes the `*_sync`
builtins (and on some ROCm versions `__syncwarp` with them), so it is not a fix. (Verified
with HIP 7.13 for gfx90a and gfx1151: the `unsigned` mask fails, the 64-bit one compiles.)

**And the mask VALUE must be built in 64 bits, which the destination type does not do for
you.** In `const uint64_t mask = ((1 << K) - 1) << (K * subgroup);` the shift result type is
the promoted LEFT operand, i.e. `int`, so the whole expression is evaluated in 32 bits and
the widening to `uint64_t` happens afterwards, too late. Shifting an `int` by 32 or more is
undefined, and AMD does not trap or saturate: the 32-bit shift instruction uses only the low
5 bits of the count, so the count wraps and lanes 32..63 silently get the mask belonging to
lanes 0..31. Any mask literal that can be shifted past bit 31 needs `1ull`. The class is
nasty because it compiles clean, is invisible on wave32 (a 32-lane wavefront never reaches a
shift of 32, which is why upstream RDNA testing does not catch it), and yields a wrong-lane
collective rather than a fault. Grep every `1 <<` and `1u <<` whose shift count is derived
from a lane index. HIP RT 3.1.0.cb09c56 has exactly this in three places
(`hiprt/impl/BvhBuilderKernels.h`, `subwarpMask`): the upper half of the wavefront selects
its widest child from the wrong subgroup, one subtree gets opened twice, the collapse
therefore emits more leaf references than there are primitives, and the persistent builder
kernel -- whose only exit is `emittedCount == primitiveCount` -- steps over its own exit
value and spins forever, hanging the GPU inside a BVH build over a few dozen triangles.
(diff-surfel-tracing, in its HIP RT dependency.)

**A 32-bit literal COMBINED with a 64-bit mask is the same class in a second shape, and it
does not need a shift count over 31 to be wrong.** `mask &= ~( 1u << lane )` computes the
complement in 32 bits, so the operand zero-extends and the AND clears bits 32..63 of the
mask for EVERY lane, not just lanes over 31. `mask ^= 1 << lane` is worse at exactly one
lane: `1 << 31` is `INT_MIN`, which sign-extends to `0xffffffff80000000` and SETS bits
32..63 that a `while ( mask )` loop can then never clear. The first shape ends a
lane-at-a-time loop halfway through, so the upper half of every wavefront is skipped and
whatever it was supposed to produce is silently absent; the second hangs. Both are the same
grep as above -- every `1 <<`, `1u <<` and `~( 1u << ... )` in an expression whose other
operand is a 64-bit mask -- and the grep must be run over the whole dependency, not stopped
at the first site that explains the symptom in front of you. In HIP RT 3.1.0.cb09c56 the
three `subwarpMask` sites are joined by `PairTriangles` (`activeMask &= ~( 1u << ... )`,
which drops the triangles of lanes 32..63 from the BVH: the build succeeds, the image looks
plausible, and half the geometry is simply not there -- 31 of 64 primitives traced, versus
62 of 64 once fixed) and `PackLeavesWarp` (`packetMask ^= 1 << broadcastLane`, on the
RTIP 3.1 leaf path, so RDNA4 rather than CDNA). A validation harness that only checks
finiteness, determinism and plausible ranges passes the first one; check that the geometry
you handed the accelerator is the geometry that gets traced. (diff-surfel-tracing.)

**Sweep for THREE patterns, not one, and sweep the vendored subtrees the port ships.** A
`1 <<` grep finds the shapes above and misses the one that faults. The full sweep is: (a)
every `1 <<`, `1u <<`, `~( 1u << ... )` whose other operand is a 64-bit mask; (b) every
32-bit DESTINATION that receives a wave-wide collective -- `u32 x = __ballot( ... )`, a
`uint32_t` struct field, an explicit `(uint32_t)` cast -- since `__ballot` returns all 64
lanes and the narrowing keeps lanes 0..31 whether or not a literal is involved anywhere;
(c) the IDIOM that makes (b) wrong rather than merely truncated: code that carves a block
into fixed 32-lane logical warps (`warp = threadIdx.x / 32`, `lane = threadIdx.x % 32`,
often a `constexpr WARP_SIZE = 32`) and then calls a wave-wide collective, so on wave64 the
second logical warp in each wavefront silently reads the FIRST one's answer -- its own bits
are sitting in the half that was discarded. Run all three over every subtree the port
compiles, including a dependency's vendored `contrib/` copies of other projects: they are
staged and runtime-compiled exactly like its own headers, and a defect there is reported to
a different repository than the one you are patching.

Orochi's radix sort (`ParallelPrimitives/RadixSortKernels.h`, `OnesweepReorder`, vendored
byte-identically into HIP RT 3.1.0.cb09c56) is (b)+(c): a 256-thread block as eight 32-lane
logical warps, `u32 broThreads = __ballot( itemIndex < numberOfInputs )`, so every odd
logical warp ranks its keys against the even warp's attendee set and the corrupted prefix
sums index a GLOBAL write. Measured on gfx90a it is a hard `Memory access fault ... kernel:
OnesweepReorderKeyPair64`, not a wrong answer. Fix it with the same arch-unified shift as
the cuSZ entry below, `(u32)( __ballot( PRED ) >> ( ( threadIdx.x % warpSize ) / 32 * 32 ) )`,
which is a no-op on wave32 and on the CUDA `__ballot_sync` path. Note the reachability
trap: the sort dispatches a single-pass kernel below 3072 elements and only reaches this one
above it, so a small harness scene never executes the defect -- two review rounds passed
44/44 over a back end that could not build an acceleration structure for any real scene. If
a dependency picks kernels by problem size, one test case has to be big enough to cross
every such threshold. (diff-surfel-tracing, in the Orochi copy vendored by its HIP RT
dependency.)

**Do not hardcode wave64 GEOMETRY either** -- the inverse failure. Packing two logical
32-thread rows into one wavefront (`wflane = threadIdx.x + ((threadIdx.y&1)<<5)`, a
`1ull<<wflane` prefix mask, `__shfl(...,0,64)`, `__popcll` over a 64-bit ballot, a
`wflane==0` leader test) COMPILES on wave32 and silently miscomputes: each 32-thread row is
its own wavefront there, so odd-row lanes land at 32..63, the leader test never fires
(lost atomics), and the mask reads phantom bits. Make wave-collective geometry
`warpSize`-generic, or give it an explicit `__GFX9__` branch AND a wave32 `#else`. A
wave64-only fix that passes CDNA is not done until it is correct on RDNA. (popsift
s_extrema.)

Width-32 LOGICAL warp ops (`__shfl*(...,32)`, `cub::WarpReduce<,32>`,
`cg::tiled_partition<32>`) are arch-agnostic and fine -- they work within a 32-lane
subgroup whatever the physical width.

**A compat `__ballot_sync` that casts `__ballot()` to `uint32_t` takes the LOW 32 lanes**,
which are the wrong lanes for any logical warp not at the wavefront base -- with
`blockDim (32,32)`, odd `threadIdx.y` rows sit in the high lanes. Arch-unified form:
`(uint32_t)(__ballot(PRED) >> (__lane_id() & ~31u))`, where the shift is 0 on wave32. (cuSZ)

**An over-wide `__shfl*` width does not error, it silently degenerates.**
`__shfl_up(v, delta, 64)` lowers to `__builtin_amdgcn_ds_bpermute` with a clamp of
`self & ~(width-1)`; on wave32 that clamp is 0, so the permute domain becomes the
32-lane physical wavefront while the loop still treats lanes as a 64-wide group. In a
lossy or statistical pipeline the wrong collective still round-trips, so it surfaces as a
quality-metric shift (compression ratio, PSNR) rather than a crash -- the cross-arch gate
has to compare the METRIC, not just that the output decodes. (cuSZ)

**A full 64-bit `*_sync` mask in 32-lane group code traps in a HIP debug build on wave64.**
HIP's `__shfl*_sync` / `__ballot_sync` run `__hip_check_mask`, which asserts that the mask
equals the lanes that reached the call (`__ballot(true)`), and `__hip_assert` is a
`__builtin_trap()` unless `NDEBUG` is defined. Code written for 32-lane CUDA warps and kept
correct on wave64 by group-local folds (offsets below 32, width-32 shuffles, the
`>> (__lane_id() & 32)` ballot) is still wrong to pass `0xffff...ffffull` wherever the two
halves of a wavefront can reach the intrinsic separately: a per-group `continue`, a
per-group trip count, or a cross-warp step that runs only `threadIdx.x < 32` (or
`warp_id == 0` with `warp_id = threadIdx.x / 32`). Release builds are correct and wave32
cannot see it, because a group is the whole wave, so only a wave64 debug run traps. Pass the
caller's group mask, `0xffffffffull << (__lane_id() & 32u)`: it is the whole wave on wave32
and one half on wave64. The check runs a waterfall over distinct mask values, so each half
is validated against itself, whether the halves arrive together or apart. HIP evaluates the
shuffle in the caller's context, so the mask changes only that check and the ballot's
`& mask`, never the data movement. Mask-free `__shfl(v, s, 32)` / `__ballot` also work but
drop the check entirely. Folds over the native wavefront (`warp_id = threadIdx.x /
warpSize`) keep the full mask. (LichtFeld-Studio: fastgs/gsplat 32-lane tile macros, nn
softmax/norm block reductions, `block_reduce_min4`.)

**Intra-wave barrier divergence.** A per-row early `return` before `__syncthreads()` is
benign on CUDA when a row is a whole 32-lane warp -- the warp exits and leaves the barrier
wait set. On wave64 two rows share a wavefront, so one row returning is a half-wave
divergence and the wave stays live with lanes missing. Audit every block barrier whose
early-return granularity is under 64, and fold the return into an out-of-bounds predicate
so all lanes reach the barrier. (lc0)

Detector for it: route `__syncthreads()` through a wrapper that, in HIP debug builds only,
computes the expected active-lane mask from the block layout and asserts
`__ballot(1) == expected` first. It does not false-positive on a legitimately partial final
wave. If device code is compiled by explicit hipcc command lines, mirror the release
`-DNDEBUG` yourself or the asserts stay on in release. (lc0)

**`cub`/`hipCUB` block collectives race on a 64-thread block without an explicit
`__syncthreads()`.** At wave64 a 64-thread block is a single wavefront, so a `BlockReduce`
or `BlockRadixSort` lowers to a single-wavefront op with no syncing epilogue; reusing one
`TempStorage` union across back-to-back calls lets the second call's writes overlap the
first's reads. CUDA's 32-wide warp split the block in two and masked it. (CV-CUDA:
OpPairwiseMatcher crossCheck.)

**`__smid()` can EXCEED `multiProcessorCount` on AMD.** NVIDIA's `%smid` is dense in
`[0, multiProcessorCount)`, so host code routinely sizes a per-SM scratch pool to that
count and indexes it `pool + smid()*POOL_SIZE`. AMD packs `(SE,CU)` sparsely -- gfx9 is
`(se_id << 4) | cu_id`, so values reach 127 with ~104 CUs active (measured: max 125 on a
gfx90a GCD with `multiProcessorCount` 104). A pool sized to the CU count is written past
its end by any block on a high-id CU: a SILENT out-of-bounds device-heap write that
corrupts a later read, so a green test run does not prove safety. Size such pools by a
one-time runtime probe -- a saturating kernel that `atomicMax`es `get_smid()`, take
`max+1`, round up, cache it -- rather than hardcoding bit layouts. Do NOT `% workers` the
index: that reintroduces the contention the private-slot pool exists to prevent. Audit
every allocation paired with a `smid()*STRIDE` kernel; pools indexed by `blockIdx.x` with a
`gridDim == multiProcessorCount` launch are self-consistent and fine. Keep the CUDA path at
`multiProcessorCount`. Prove the fix with a sentinel region after the pool. (gpu4pyscf:
int3c2e fill pool, ~4MB silent OOB.)

**HIP's device `warpSize` is not a plain `int`** and device `printf` rejects it as a
vararg: cast it, `printf("%d", (int)warpSize)`. The raw diagnostic is misleading -- it
surfaced as an undeclared-identifier cascade in a generated dispatch header. No-op on
CUDA. (LC-framework)

**Validate wave-size work on BOTH widths, and where there is no host reference use a
CROSS-ARCH CONSISTENCY gate.** "Deterministic, non-zero and plausible" passes a wrong
wave32 result -- popsift's gfx1100 reported deterministic non-zero features from a
miscounted extrema path, a false pass. Where the algorithm is deterministic, diff the
wave32 output against the wave64 output for the same input. (popsift)

**Special case: a warp-width-dependent SERIALIZED FORMAT.** If the warp width determines an
on-disk or in-memory layout rather than just launch geometry, it cannot follow the per-arch
device constant -- pin it to a fixed maximum width, or carry the producing device's width
at runtime. dietgpu's rANS archive serializes one coder state per lane
(`ANSWarpState::warpState[kWarpSize]`), making its header geometry width-coupled; pinning
`warpState[64]` keeps every `sizeof()` and header offset arch-independent, leaves the
wave64 archive byte-identical to its old format, and lets a wave32 device use the first 32
slots. (dietgpu)

## Memory and lifetime

**Out-of-bounds reads.** CUDA often tolerates a read one element past an allocation; AMD
faults. Kernels reading index +/-1 or +/-width at edges (stencils, neighbour gathers) must
clamp. (colmap ComputeDOG.)

**A latent OOB that stopped reproducing is still there, and index math on `size_t` operands
overshoots UPWARD, not past-the-end.** Two traps, usually together. First, whether a wild
read faults depends on what the driver happens to have mapped, so the same kernel can abort
on one tree and complete on the next with no relevant change; absence of a fault is not
evidence the bug is gone, and "it no longer crashes" must never close an OOB finding.
Instrument instead: a `printf` in the trailing threads of the last block, printing the
decoded indices and the sizes they are checked against, decides it in one run. Second, when
padded threads decode an index through arithmetic whose operands are `size_t` -- closed-form
inverses of triangular pair indices are the classic shape -- the intended "negative"
excursion is computed in UNSIGNED 64-bit and wraps to nearly 2^64 instead. The out-of-range
indices are then astronomical rather than slightly too large, and a `sqrt` of the wrapped
value returns a real number where a signed evaluation would have produced a NaN, so the
symptom does not look like the sign error it is. (gRASPA `TotalVDWRealCoulomb`: the trailing
thread decoded `AtomA 2147484974` and `AtomB 2305843010287440402` against a size of 1327 --
an 18-exabyte offset from a `size_t*` base -- and the fix was moving the bounds test that
already existed above the two loads it followed.)

**Rule-of-five on resource handles.** CUDA tolerates a default-constructed or
double-destroyed texture/stream/event handle; AMD faults. Give RAII wrappers explicit
default init (`handle = 0`), move-only semantics, and a guarded destructor. (colmap
CuTexObj.)

**Querying an event after its recording stream was destroyed is a use-after-free on HIP.**
CUDA keeps an event valid after `cudaStreamDestroy` of the stream it was recorded on, and
caches built on that (pinned host allocators that keep per-block "last use" events, event
pools) query those events much later. HIP's `hipEventRecord` stores the stream in the event,
and `hipEventQuery`/`hipEventSynchronize` read that stream's capture state -- and, if the
freed memory happens to say "capturing", lock its mutex and hash into its event set --
without checking the stream still exists (`checkEventCaptureRestrictions`, clr
`hip_event.cpp`; `hipEventRecord`, `hipEventDestroy` and `hipStreamWaitEvent` do check).
The query returns the right answer until the heap reuses the freed `hip::Stream`, then it
hangs in `pthread_mutex_lock`, segfaults, or raises SIGFPE (a zero bucket count), so it
looks like a flaky test that fails after tens to hundreds of repeats. Fingerprint: a gdb
stack of `pthread_mutex_lock` <- two unexported libamdhip64 frames <- `hipEventQuery`, no
other thread inside HIP. Prove it with an `LD_PRELOAD` shim over
`hipEventRecord`/`hipStreamDestroy`/`hipEventQuery` that flags queries on events whose last
recording stream is dead; a plain create/record/destroy/query loop will NOT reproduce it,
because the next `hipStreamCreate` takes the chunk back with a clean status; fill freshly
`malloc`'d 16 B-64 KiB chunks with 32-bit 1s after the destroy and the query hangs every time.
Do not fix it holder by holder: the holders are not only caches but events in transit outside
the holder's lock (blocks an `empty_cache` took and waits on after unlocking) and app-owned
rings and frame fences no allocator hook sees, and a per-site sweep misses some. Fix the class
where the compatibility header already maps the stream calls: route `cudaStreamCreate*` and
`cudaStreamDestroy` to a small pool whose destroy synchronizes and parks the stream (real
destroy only if it is capturing or fails to sync) and whose create reuses a parked stream with
the same device, flags and default priority. No stream an event ever saw is freed, and the
pool is bounded by the peak number of live streams. Do not park without reuse: an idle used
stream holds about 8 MiB of host memory on TheRock 10.1. (LichtFeld-Studio:
`core/cuda/hip_stream_reuse`, which replaced a `PinnedMemoryAllocator::release_stream` sweep;
project save hung or crashed on gfx1100, and the review found three more holders; deferral
`hip-event-query-destroyed-stream-uaf`.)

**Fresh device allocations are NOT zero on ROCm**, where CUDA's allocator often hands back
zeroed pages. A kernel writing only a sub-region and implicitly relying on a zero start
passes on CUDA and reads garbage on ROCm. Fingerprint: a zero-tolerance test that PASSES in
isolation and FAILS in-suite, after earlier tests have dirtied and freed similar buffers.
The reliance is a latent upstream bug, not a ROCm defect: zero explicitly
(`hipMemsetAsync`) before the partial-write kernel, and audit any kernel whose write set is
smaller than its output allocation. (opencv_contrib: cudastereo census_transform border.)

The commonest instance is a counter or accumulator: `cudaMalloc(&count, sizeof(unsigned))`
followed directly by a kernel that `atomicAdd`s into it. On CUDA the page is usually zero,
so the count is right by luck; on ROCm it starts from whatever the last owner of that memory
left, and the result is wrong by a varying offset that depends on what ran before -- easy
to mistake for a race. Audit rule, for porting and for review alike: for each fresh
allocation, find its first access. If that access is a read-modify-write (an atomic, `+=`,
`|=`, a histogram bin increment) rather than a plain store, a `cudaMemset`/`hipMemsetAsync`
or an initializing kernel must precede it on the same stream. This is easy to miss in a
review that is focused on warp-size or mask bugs, because nothing about the kernel itself
looks wrong.

**Bitwise and min/max atomics on `hipMallocManaged` memory are silently dropped on a
discrete xnack- card.** There, managed memory is host-resident fine-grained memory, and
the device reaches it over PCIe, whose AtomicOps cover only add, swap and CAS. A native
`global_atomic_or`/`xor`/`smin`/`umax` into it completes with no error and changes nothing;
`atomicAdd`, `atomicCAS`, plain stores and `hipMalloc` memory are all fine. Whether the
port is hit depends on CODEGEN, not on the source: HIP's `atomicOr` marks the operation
`!amdgpu.no.fine.grained.memory`, which permits the native instruction, but the inliner can
drop that metadata when the atomic sits in a helper inlined into the kernel, and without it
the backend emits a CAS retry loop that works. So one program can pass at `-O3` and come out
entirely empty at device `-O0`, and a standalone probe can contradict the application.
Device `-O0` is not exotic: a CMake HIP build configured with no `CMAKE_BUILD_TYPE` (what
many READMEs give) has empty `CMAKE_HIP_FLAGS` and compiles device code at `-O0`, so
validate the README recipe as written, not with `-DCMAKE_BUILD_TYPE=Release` added.
Disassemble the APPLICATION's code object (`llvm-objdump --offloading <exe>`, then
`llvm-objdump -d` on the gfx bundle): a native `global_atomic_or` or `flat_atomic_or` means
it fails; only `global_atomic_cmpswap` means it is passing by accident. A passing
CPU-vs-GPU differential does not clear this risk. Fix it in the source, never by optimization
level: put the atomically written buffer in `hipMalloc` memory and copy it back, or use the
`_system` atomic variants. (cuda_voxelizer on gfx90a/MI250X: `atomicOr`/`atomicXor` voxel
table, exact at default `-O3` and Debug, 0 of 13,802 / 52,258 voxels at device `-O0`;
gfx942 and the gfx1151 APU do not drop them. cudaKDTree: int `atomicMin`/`atomicMax`.)

**`cudaMemcpy*Async` from a soon-freed pageable host buffer.** CUDA stages pageable async
copies synchronously, so a test that copies from a local `std::vector` and lets it go out of
scope still works. ROCm's is genuinely asynchronous, so the freed and reused buffer is read
after the copy is enqueued. The corruption appears only in bytes the kernel does not
overwrite, and the failing set varies run to run. This is a use-after-free that is latent UB
on CUDA, not an allocator defect -- confirm by keeping the source alive and watching the
failures vanish. (CV-CUDA: InterpolationVarShapeWrap.)

The same staging difference hides a **cross-stream race** with no free involved: an upload
by `cudaMemcpy*Async` from pageable memory on one stream, then a consumer kernel on ANOTHER
stream with no event or stream sync between them. On CUDA the copy has effectively landed
before the launch; on ROCm the first kernel batch can read a partly written array.
Fingerprint: an intermittent wrong result confined to the first launch's outputs, while a
sibling code path that ends its setup with a device sync stays bitwise stable. Detect it by
running the same call 20 times and comparing bitwise with the first; one
`hipStreamSynchronize` on the upload stream makes it stable. (TIGRE: parallel-beam Siddon
`CreateTextureParallel`, wrong pixels up to 22% of max in projections 0 to 8, 2 to 7 runs
in 20 on gfx90a.)

**A functor returning a reference to a by-value or forwarded parameter** binds to a
temporary that expires at the return. nvcc and CUB happen to keep the value live in
registers; rocPRIM block-loads the dangling reference and reads garbage. Return BY VALUE --
it is UB on CUDA too. Localise it by computing the same reduction three ways (library
wrapper, raw hipCUB call, `thrust::reduce` over the same iterator plus the bare operator):
if only the wrapper is wrong, the bug is in its transform, not the iterator or rocPRIM. A
standalone microbench often will NOT reproduce it, because the dangle manifests only
through the library's nested iterator instantiation (here a `thrust::transform_iterator`
feeding `hipCUB DeviceReduce::Reduce`). (cudf: `cast_functor.cuh` same-type
`operator()(T&&)`, selected only when reduction output type equals input element type --
min/max returned the seeded identity, float sum returned nan, int32->int64 sum was correct.)

**`cudaLaunchHostFunc`/`hipLaunchHostFunc` callback threads must not call ANY runtime API** -- an RAII deleter
that frees inside one DEADLOCKS ROCm. HIP forbids it as CUDA does, but CUDA more often
tolerates it, so only AMD hangs. The trap is indirect: a smart pointer whose deleter IS a
runtime call (`hipFree`, `hipStreamDestroy`, `hipEventDestroy`, `hipHostUnregister`) drops its last reference
inside the callback. The symptom masquerades as a slow kernel -- the GPU pegs at 100% while
a runtime worker spins, and serialising the API (`AMD_LOG_LEVEL=3`) slips past the deadlock
window, so the same test alternates between hanging and returning a fast raced result. Do
not dismiss it as a slow test; an op orders of magnitude slower than the CPU engine is a bug
to root-cause, never one to exclude. Detect it with a gdb backtrace of the live hang: the
HIP worker sits in your deleter -> `hipFree` -> `sched_yield` inside `libamdhip64`. Fix by
keeping the callback API-free -- move the handles into a deferred-free list (a container
move, no API call) and drain it from a user thread that already issues runtime calls, under
the existing queue mutex, draining on teardown too. Only the op that hands per-call
allocations into a callback-retired queue trips this, which is why one gate hangs while
everything else passes. (qrack: `_PopQueue` under `UniformlyControlledSingleBit`.)

**An init-time kernel launched on the legacy null stream is NOT ordered against streams
created `cudaStreamNonBlocking`/`hipStreamNonBlocking`.** A weight/constant upload helper
that stages through a SHARED scratch buffer, launches an async conversion kernel on stream
0, and returns without synchronizing is racing the first real evaluation, which overwrites
the same scratch from a non-blocking compute stream. Only the LAST upload before the first
evaluation is exposed, because each earlier one is flushed by the next helper call's
synchronous H2D `hipMemcpy` into the same scratch -- so the corruption lands on whatever
tensor the last-constructed layer uploaded last, and looks deceptively like "that one layer
is numerically broken". Fingerprint: exactly one CONSTANT tensor wrong while every COMPUTED
tensor around it matches a CPU reference to ~1e-7, the wrong values being
plausible-magnitude foreign floats, and the bug VANISHING the moment you add a debug
readback (whose `hipDeviceSynchronize` supplies the missing ordering). Windows loses this
race deterministically -- its KMD batches/defers submission -- where Linux HIP and CUDA
land the tiny copy immediately and win it, so it presents as an OS-specific defect and
invites a toolchain hunt. It is latent UB on CUDA too, so fix it unconditionally, never behind a
platform guard -- but fix it at the BOUNDARY, not in the helper you caught it in. Such a
codebase usually has several upload paths sharing the shape (lc0 has six: `allocAndUpload`
plus five `LoadWeights`/positional-encoding variants), and whichever one happens to run
last depends on the network shape and the precision, so synchronizing the single helper
your reproducer exercised leaves the same bug live for another input. Construction is
single-threaded and every upload is null-stream ordered against every other, so ONE
device-wide synchronize at the end of construction covers all of them in the same one
line. Put it after the LAST null-stream operation, not merely after the uploads: an async
`hipMemset` that zeroes the evaluation's own buffers is the same race, so a sync placed
before that allocation loop leaves it open. (lc0: `allocAndUpload` in
`neural/backends/cuda/layers.cc` corrupting the value head's `ip2_val_b_`; the same 4.4e-02
value error on gfx1101, gfx1151 and gfx1201 under Windows and absent on linux-gfx90a and
linux-gfx1100, which had been misread as a rocBLAS/Tensile problem for two months. The
first fix synchronized `allocAndUpload` alone; on a net WITH a moves-left head the last
fp16 upload is `FCLayer<half>::LoadWeights` instead, which still corrupted the moves-left
head while the value/policy gate passed 34/34. Upstream's own fix, lc0 #2454, is the
reference shape: one `cudaDeviceSynchronize()` after the `tensor_mem_` malloc/memset loop.)

A plain `hipMemcpyDtoD`/`cuMemcpyDtoD` (or `hipMemcpy` with `hipMemcpyDeviceToDevice`) is
the same race in a less suspicious costume: a device-to-device copy is asynchronous with
respect to the host and runs on the legacy stream, so the "synchronous-looking" call returns
before the copy lands and the next kernel on a non-blocking stream can read the destination
first. Fingerprint: the destination reads exactly as it was before the copy (zeros, or the
previous test's values), the copy's own arguments and return code are right, and a
standalone C probe of the same copy passes because it reads back with a synchronizing D2H.
Bisect with a sync on each side: only the one AFTER the copy fixes it. (TornadoVM:
`NativeCommandQueue.mapOnDeviceMemoryNDRegion`, deterministic on Windows gfx1151, passing
on Linux gfx950/gfx1201.)

**`hipFree` is synchronizing** (`hipFreeAsync` is not), so an explicit
`hipDeviceSynchronize()` before it is redundant. (anari-visionaray)

## Textures

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

A project that reads EVERYTHING through texture objects, with address modes, needs more than a pointer swap, but the emulation is still exact when no binding relies on hardware filtering (LEAP: its HIP port already created every texture point-mode and blended corners in software, so every fetch was a point fetch). Keep the handle type and every kernel signature: on an image-less GPU have the texture loader cudaMalloc one buffer holding a small header (extents, address mode) followed by the dense data, and return that buffer as the `hipTextureObject_t` (and as the `hipArray*` the call sites free). The device accessor applies the address mode in float before converting to an index (clamp: `fminf/fmaxf` to `[0, n-1]`; border: any axis outside `[0, n)` returns 0), reads `floor(c)` for a point fetch, and maps a non-finite coordinate to a finite one. Casting the header and data pointers to `__attribute__((address_space(4)))` (read-only constant) lets the compiler hoist the header into one `s_load_dwordx4` per kernel instead of reloading it around stores.

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

## Floating point

**`__fsqrt_rn` is not always correctly rounded on gfx90a** (1 ULP high) where CUDA's
`sqrt.rn.f32` is IEEE correctly-rounded, so bit-exact tests pass on NVIDIA and fail on
gfx90a. Route the f32 device sqrt through the correctly-rounded f64 `__dsqrt_rn` and cast
back; CDNA has fast f64 sqrt so the cost is negligible. (CV-CUDA: OpNormalize, L2
PairwiseMatcher.)

**clang for HIP defaults to `-ffp-contract=fast`** and forms FMAs ACROSS statements, where
nvcc with `--fmad=true` contracts only within one expression. The extra contraction drifts
results ~1 ULP from the CUDA build and CPU gold. MATCH UPSTREAM's setting rather than
pinning one unconditionally: CV-CUDA needed `-ffp-contract=on` to match nvcc, while
LC-framework's nvcc recipe uses `-fmad=false -mno-fma -ffp-contract=off`, so pinning `on`
there would diverge. Integer-only components are unaffected. (CV-CUDA OpWarpPerspective;
LC-framework.)

**An exact float-equality branch fed by approximate division can HANG, not just drift.**
HIP's `__fdividef` differs ~1 ULP from CUDA, so a downstream exact-equality test on the
quotient -- selecting a loop start or end index -- can flip on AMD. When that index feeds an
UNSIGNED loop bound via subtraction, the out-of-range value underflows to a huge count and
the loop never terminates: a data-dependent hang, not a wrong-by-1-ULP number. Do not gate
control flow on exact equality of approximate fp; use a tolerance, or cap the derived bound
at its provable geometric maximum. This is the class where numeric drift becomes a
CONTROL-FLOW failure. (TIGRE Siddon_projection: `Np` derived from `__fdividef` comparisons,
a 0.2s kernel ran >10 min at some angles; capped at `Nx+Ny+Nz+3`, a no-op for valid rays and
identical on CUDA.)

**Device `cuda::min`/`cuda::max` NaN-selection can differ from host `std::min`/`std::max`.**
Host `std::min(a,b)` is `b<a?b:a` and `std::max(a,b)` is `a<b?b:a`; a device version written
`a<b?a:b` picks the OPPOSITE operand on a NaN compare, so tests filling buffers with random
bytes and comparing host gold against device output differ bit-for-bit. Spell the device
ternaries to match the host forms exactly. (CV-CUDA: OpMorphology CLOSE on RGBAf32.)

**Host `min`/`max` on floating-point values truncate to int under hipcc.** CUDA declares
host `min`/`max` for int, unsigned, long, long long, float and double. HIP's
`__clang_hip_math.h` declares only `min`/`max(int, int)` on the host. So host code in a
`.cu` file such as `max(0.25, min(100.0, fwhm))` or `max(atan(a), atan(b))` compiles
cleanly, converts both arguments to int and returns an int. Device code is unaffected,
because the device overloads cover float and double. Expect parameters silently rounded
down, a value below 1 clamped to 0, and geometry decisions taken on a zero angle. The only
build symptom is `-Wliteral-conversion` on fractional literals ("changes value from 0.25
to 0"), and only when a literal is involved. Detection: add
`-D__HIP_NO_HOST_MIN_MAX_IN_GLOBAL_NAMESPACE__ -include poison.h` to the real hipcc
command with `--cuda-host-only -fsyntax-only`. `poison.h` redefines the default pair with
`__attribute__((diagnose_if(true, "HOSTMINMAX", "warning")))`, so every host call that
lands on `(int, int)` is reported with its location. Define CUDA's full overload set the
same way, each overload naming its own signature, to see which calls really take floating
point. Use `diagnose_if`, not `deprecated`: a `deprecated` version missed LEAP's two
`ramp_filter.cu` sites (cause not traced). Fix: change the call sites to what CUDA's
overload calls. `crt/math_functions.hpp` implements `min`/`max(double, double)` and the
mixed float/double forms with `fmin`/`fmax`, and `(float, float)` with `fminf`/`fmaxf`.
Cast a float argument to double where CUDA took a mixed overload, e.g.
`fmax(0.25, fmin(100.0, double(fwhm)))`. That keeps CUDA's results, NaN handling included,
on every ROCm and on CUDA, without an `#ifdef`. Leave int calls alone. Do NOT rely on
`-D__HIP_DEFINE_EXTENDED_HOST_MIN_MAX__=1`. That macro adds the floating-point host
overloads only from ROCm 7.2.0 (LLVM 0248d277ca), defaults to 0, and does not exist in
7.1.x and earlier, where the define silently does nothing and the truncation stays. Rerun
the detection pass with no define: no host call may still land on `(int, int)` with
non-int arguments. Then confirm that the device code objects did not change (`.text`/`.rodata`
of the unbundled `.hip_fatbin`; the `__hip_cuid_*` names change with path and flags).
(LEAP: bilateral FWHMs truncated, intensity FWHM < 1 gave the identity; the modular-beam
Joseph backprojector chose its parallel kernel for cone data, failing adjointness and FBP;
the FBPlowpass window zeroed all but DC.) `std::min`/`std::max` are NOT safe either once a
hipify pass has run: hipify_torch (and torch's hipify, same table) rewrites them to
`::min`/`::max`, which land on the same `(int, int)` host pair, so a correct
`std::min(float, float)` in CUDA source truncates in the hipified copy. Run the detection
on the hipified sources, not the originals. When a build compiles host `.cpp` files as
HIP too, those files are in scope, but unqualified calls there usually bind to `std::min`
through `using namespace std` (a template exact match beats the int pair); to tell a real
truncation from that, report only calls that land on `(int, int)` with ROCm's default
pair AND on a non-int overload with CUDA's full set. (LEAP version two: 27 sites, 10 of
them new, including `1.0 / max(fabs(v_min), fabs(v_max))` dividing by zero in the cone
geometry setup.)

**A HIP workaround in shared host code must not change CUDA's numbers.** HIP declares the
scalar math builtins (`rsqrtf`, `min`/`max`) for the device only, and HIP's `__half`
converts to every floating type through a template, so a port that rewrites a host call for
HIP (`rsqrtf(x)` to `1.0f / std::sqrt(x)`, `std::pow(a, b)` to a cast through `double`,
`__half` to an integer through `float`) silently changes the CUDA build too: `1.0f / sqrtf`
rounds twice and moved 9 of 51 coefficients by 1 ULP, `pow` through double differed in the
last bit, and the float route made NaN/Inf half-to-int undefined where CUDA's
`__half2int_rz` saturates. Gate the rewrite on `__HIPCC__` and keep upstream's
expression in the `#else`. Do not narrow the gate to the toolchain where the error first
showed: `std::pow(__half, __half)` was ambiguous in the HIP host pass on Linux too (glibc
`double` vs libstdc++ `float`) once a `__half` kernel instantiated it. A `_MSC_VER` gate
was chosen because an older Linux build had compiled upstream's call, but that build had no
gate and predated the kernels that instantiate it, and the first Linux build with the gate
failed; key the gate to the cause (HIP's `__half`), not to the first OS that
failed; where HIP needs a substitute, write the formula CUDA's host math
uses (`(float)(1.0 / sqrt((double)x))` for host `rsqrtf`), and give conversions CUDA's
saturation and NaN rules. A compile-only CUDA build with a per-TU preprocessed diff against
upstream is what found these; HIP runs cannot. (LichtFeld-Studio)

## Headers, includes and build

**A shared compat header must be host-includable.** Host `.cpp` TUs reach the shim through
ordinary headers, so an unconditional `#include <cub/...>` or `<hipcub/...>` leaks device
headers into g++ and fails there. Put the shim in the lowest common layer, keep host-safe
includes (`hip_runtime.h`, `hipfft.h`) unconditional, and gate device-only ones behind
`__CUDACC__` or `__HIPCC__ || __HIP_DEVICE_COMPILE__`. (SCAMP, stdgpu)

**`__HIP_PLATFORM_AMD__` is undefined until `hip/hip_runtime.h` has been included in that
TU.** A wave-width gate in a header included BEFORE the runtime header silently takes the
CUDA branch and picks width 32. hipify-perl prepends the runtime include at line 1, which
masks the bug -- so the gate is correct only by accident and breaks for anyone building
without hipify. Verify include order rather than trusting the prepend. (LC-framework)

**Missing-include errors are usually pre-existing upstream omissions**, not port breakage:
CUDA's headers supplied them transitively and the narrower HIP include graph unmasks them.
Fix additively -- it helps the CUDA build too. (LC-framework)

**`CUDART_VERSION` is undefined under HIP, so every `#if CUDART_VERSION >= N` guard silently
takes its fallback.** Audit them: some fallbacks change behaviour, not just speed. A
`cudaMallocAsync` path that falls back to `cudaMalloc` makes the allocation uncapturable, and
`hipMalloc` inside a graph capture fails with `hipErrorStreamCaptureUnsupported`; when the
caller then bails out without ending the capture, every later allocation in the process
fails the same way and a whole test binary goes red behind one root cause. Extend the guard
(`|| defined(USE_HIP)`) where the HIP API exists, rather than defining `CUDART_VERSION`
globally, which flips untested pool paths at once. A pool whose fallback frees with
`hipFree` also synchronizes the device on every free, so a free while another stream waits on
a host callback blocks forever (a gated-producer test deadlocks; the async free returns at
once). To prove the CUDA view unchanged, note that `unifdef -UUSE_HIP` leaves
`#if X || defined(USE_HIP)` as written (it does not reduce `X || 0`); compare with
`-UUSE_HIP -DCUDART_VERSION=<N>` on both the upstream and the port file, for an N on each
side of every guard. Once the HIP default pool is in use, do not count on
`hipMemPoolAttrReleaseThreshold`. On ROCm 10.1 (Linux) it reaches the pool's VM heap only if
it is set after the pool's first allocation; set before it (as most code does, in setup), it
is lost and freed blocks are unmapped at the next synchronize whatever its value. Even when
set late it keeps memory only in 128 MiB chunks, so a threshold below one chunk (such as
64 MiB) keeps nothing, and a free can trim below the threshold. Correctness is unaffected;
measure before working around it (deferral `hip-vmheap-unmap-threshold-lost-or-overtrimmed`,
formerly `hip-mempool-release-threshold-ignored`). (LichtFeld-Studio)

**The vcpkg toolchain already overrides `add_library`/`add_executable`.** A second
top-scope override (the MPPI retag trick) makes `_add_library` point at vcpkg's override,
which recurses. In a vcpkg project retag `.cu` sources with a pass over all targets after
the subdirectories are added: walk `BUILDSYSTEM_TARGETS` and `SUBDIRECTORIES`, and use
`set_source_files_properties(... TARGET_DIRECTORY <tgt> PROPERTIES LANGUAGE HIP)` with paths
made absolute against the target's `SOURCE_DIR`. (LichtFeld-Studio)

**LibTorch's CMake package adds `-D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1`
to every target that links it** (`torch_hip` INTERFACE_COMPILE_OPTIONS). Project `.cu` test
files that use `__half` arithmetic then fail with "invalid operands ... '__half'". If no HIP TU
includes LibTorch, `#undef` both in the force-included compat header under `__HIP__`.
(LichtFeld-Studio)

**For plain C++ TUs the HIP headers `#define __noinline__ __attribute__((noinline))`**
(nvcc defines it only when compiling CUDA). glib's `__has_attribute(__noinline__)` then fails
("too few arguments provided to function-like macro"), so any TU that includes GTK after the
HIP runtime breaks. `#undef __noinline__` in the compat header when `!defined(__HIP__)`.
(LichtFeld-Studio)

**Use the CUDA build's host compiler for host C++ in a large app.** HIP needs clang only for
the HIP language; `CMAKE_C/CXX_COMPILER` can stay gcc. Upstream code tested only with gcc can
trip clang-only C++ rejections unrelated to HIP (`__attribute__((visibility)) [[nodiscard]]`
ordering; `std::expected<std::any, E>` constraint recursion with libstdc++ 14).
(LichtFeld-Studio)

**A force-included compat header creates no build dependency edge.** After editing a header
injected with `-include`, object files are NOT rebuilt: wipe them manually or you validate
stale code and get a silent false pass. (lc0)

**`#if __CUDA_ARCH__ >= NNN` and `#if CUDART_VERSION < NNNN` silently pick the wrong branch on
HIP**, because neither macro is defined there (it reads as 0). An arch-gated kernel body
compiles to an empty stub or its `#else` printf with no error, so a feature can look "built"
yet never run; grep for both macros before calling a port complete. Widen the arch guard with
`|| defined(__HIP_DEVICE_COMPILE__)` (never define `__CUDA_ARCH__` in a shim; HIP headers and
libraries test it), and prefix version guards with `!defined(USE_HIP) &&` so HIP takes the
modern CUDA spelling -- the legacy branch may name intrinsics HIP lacks (`h2div` vs
`__h2div`). Confirm with the disassembly: the kernel should contain real instructions
(`v_pk_*_f16` for half2 code). (gpuRIR)

**Project code that splits on `__CUDACC__` means "compiled by the device compiler", and HIP
does not define it.** Upstreams increasingly hide host-only C++23 (`std::expected` error
types, `std::source_location` capture) from nvcc with `#if !defined(__CUDACC__)` and show
device helpers only under `#if defined(__CUDACC__)`. A HIP source defines `__HIPCC__`
instead, so it parses the host-only branch (a C++23 `#error` or missing `std::expected`
under the HIP C++20 standard) and silently skips device fast paths. Treat `__CUDACC__` in
project code as `defined(__CUDACC__) || defined(__HIPCC__)`; `__HIPCC__` is set only for
HIP sources, so plain C++ TUs keep the host branch. Do not define `__CUDACC__` globally
instead: rocThrust takes its CUDA-system path under it. Re-grep after every upstream sync,
since new splits arrive with new code. (LichtFeld-Studio)

**Extend a CUDA guard for HIP; never re-key it.** Upstream's `#ifndef __CUDACC__` around
host fallbacks for `fminf`/`min`/`max`/`rsqrtf` (CUDA-Samples `helper_math.h` and its
copies) also admits the HIP device pass, because HIP never defines `__CUDACC__`. Re-keying
it to `!defined(__CUDA_ARCH__)` exposes the fallbacks to nvcc's HOST pass, where `int
max(int, int)` redefines CUDA's and every TU including the header fails -- invisible on
ROCm. Add the HIP macro to the original test instead (`!defined(__CUDACC__) &&
!defined(__HIP__)`), which leaves the CUDA and plain C++ expansions identical. HIP's host
pass does not need the fallbacks when no host-emitted code calls them; compile the host
objects without them to check. (LichtFeld-Studio)

**clang checks kernel bodies in the HIP host pass; nvcc does not.** A macro that splits
device and host forms on `#if defined(__CUDA_ARCH__)` takes the host form when clang
compiles the host side of a HIP TU, and clang still type-checks the instantiated
`__global__` bodies there. A host-only call in that branch (a string formatter, a throw)
inside a kernel is then `call to __host__ function from __global__ function ... when
compiling for host`, while nvcc builds the same code. It shows only where the macro is live,
typically a debug assert that is `((void)0)` under NDEBUG, so Release passes and the first
Debug build fails. Build Debug once after every upstream sync, and give HIP sources the
kernel-safe form in both passes (`defined(__CUDA_ARCH__) || defined(__HIPCC__)`) or make
the callee `__host__ __device__`. Only calls that go through overload resolution (templates,
overload sets) fail at once; a call to a single non-template `__host__` function in a kernel
is a deferred diagnostic that never fires, because the host pass does not emit kernel bodies.
That gives a narrower fix that keeps the host form for host code in HIP TUs: under
`#elif defined(__HIPCC__)` evaluate the message argument in `[&]() { return (message); }()`.
clang makes an unattributed lambda `__host__ __device__`, so the template call inside it is
legal in a kernel's host-pass body, while host functions still get the formatted message.
(LichtFeld-Studio)

**CUDA error codes HIP lacks still appear in switch labels.** Error classifiers name codes
such as `cudaErrorMisalignedAddress`, `cudaErrorInvalidPc` or `cudaErrorSystemNotReady`
that have no `hipError_t`. Aliasing them to the nearest HIP code duplicates a case label
when that code is also listed. Define each as `static_cast<hipError_t>(<CUDA value>)` after
checking no `hipError_t` enumerator uses the value: the labels stay distinct, and since HIP
never returns them they are unreachable. (LichtFeld-Studio)

**MSVC-only upstreams accept code that clang and gcc reject**, so the HIP build (and the
CUDA build under nvcc) is a stricter compiler than the project has ever seen. Velvet carried
a member template whose parameter pack shadowed the class pack -- accepted by MSVC, rejected
by Clang and GCC. Expect missing `typename`/`template` disambiguators, two-phase lookup
failures and narrowing conversions. Fixes are additive and arch-independent, so they help
the CUDA build too and are not HIP-specific hacks. (Velvet)

**CMake `LANGUAGE HIP` on host `.cpp` files drops OpenMP.** `OpenMP::OpenMP_CXX` adds
`-fopenmp` through `$<$<COMPILE_LANGUAGE:CXX>:...>` and links the host compiler's runtime
(libgomp for g++). A build that sets every source to `LANGUAGE HIP` therefore compiles its
CPU code without OpenMP: `#pragma omp` loops run serially with no warning, and code that
mixes OpenMP runtime calls with the pragmas can misbehave (LEAP's `downSample_cpu` aborted
with `free(): invalid pointer`; the same code built by g++ ran). Passing `-fopenmp` for HIP
alone while still linking `OpenMP::OpenMP_CXX` fails at load time (`undefined symbol:
__kmpc_fork_call`: clang emits libomp calls, the link brought libgomp).
`find_package(OpenMP COMPONENTS HIP)` failed to find flags with the TheRock SDK and CMake
3.31. What worked: `target_compile_options(t PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fopenmp>)`,
`target_link_options(t PRIVATE -fopenmp)`, and no `OpenMP::OpenMP_CXX` on the HIP target;
clang links its own libomp and adds a runpath to it. Check with `grep -c fopenmp
compile_commands.json` and `ldd` on the library. (LEAP version two.)

**An all-clang Windows HIP build hides three CMake/export faults that MSVC builds never
hit.** (1) `WIN32`-guarded export macros (`#ifdef WIN32` + `#ifdef PROJECTOR_EXPORTS` ->
`__declspec(dllexport)`) export NOTHING: clang predefines `_WIN32`, not `WIN32` (`WIN32` is
CMake's `/DWIN32` in `Platform/Windows-MSVC.cmake` `CMAKE_<LANG>_FLAGS_INIT`, the MSVC-syntax
flag `clang++ -x hip` rejects, so a build that works around that gap loses it), and where a
header does pick `dllimport`, clang drops it on the definition instead of exporting as MSVC
does. The DLL links fine and `ctypes` fails at the first symbol; check `dumpbin /exports` (and
a ~1 KB import lib). Fix: `if (WIN32 AND <hip build>) target_compile_definitions(t PRIVATE
WIN32 <X>_EXPORTS)`; defining `WIN32` also switches every other `#ifdef WIN32` path in the
sources, so run the tests on that build. (2) The vendored `hipify_torch` `cmake/Hipify.cmake`
(still so upstream in ROCm/hipify_torch at 1ea3231) runs `hipify_cli.py` and
`replace_cuda_with_hip_files.py` as the `execute_process` program, relying on the shebang:
on Windows that fails ("Failed to hipify files!"); prefix `${Python3_EXECUTABLE}` from
`find_package(Python3 COMPONENTS Interpreter)`. (3) `get_hipified_list` returns native
backslash paths; fed to `file(GLOB_RECURSE)` they glob from the drive root and configure
runs for many minutes -- normalise each element with `file(TO_CMAKE_PATH)`. (LEAP, windows)

**CMake's HIP language on Windows only works when every C/C++ compiler is ROCm's GNU-style
`clang++`; make the project choose it, do not document a workaround.** `Platform/Windows-Clang.cmake`
is include-guarded and picks MSVC-style or GNU-style rules for ALL languages from whichever
language includes it first (MSVC-style if any language's frontend variant is MSVC). So with
cl.exe as CXX, `enable_language(HIP)` dies with "mixes Clang and MSVC" (`check_language(HIP)`
cannot report this: it probes in a separate HIP-only project, so a "no HIP compiler" message
after it means the probe found no `clang++`); with clang-cl in `project(... CXX HIP)` it dies
with "mixed frontend variants"; and with clang-cl in `project(... CXX)` followed by a later
`enable_language(HIP)` the check is skipped and HIP gets cl-style flags that `clang++` rejects
in CMakeTestHIPCompiler (4.4.4: `/DWIN32 /D_WINDOWS /Ob0 /Od /RTC1` on the compile; 3.31.0:
`/machine:x64` on the link). Fix inside the project, before `project()`, guarded by
`CMAKE_HOST_WIN32` and the HIP option: find the HIP `clang++` (`HIPCXX`, else `hipconfig
--hipclangpath` with `HIP_PATH`/`ROCM_PATH`/the Python interpreter's directory as hints -- a
pip-installed ROCm SDK puts `hipconfig.exe` in the venv `Scripts`) and set `CMAKE_CXX_COMPILER`
to it unless the user gave one. Three more Windows gaps travel with it: CMake picks the
default `CMAKE_HIP_ARCHITECTURES` with `rocm_agent_enumerator`, which ROCm does not ship on
Windows, and silently falls back to gfx906 (run `amdgpu-arch` from the clang directory
instead); `find_package(hip)` needs `CMAKE_HIP_COMPILER_ROCM_ROOT` on `CMAKE_PREFIX_PATH`
(there is no `/opt/rocm/bin` on PATH to derive it from); and CMake before 3.29 links a Windows
HIP DLL without `-shared` (`lld-link: error: subsystem must be defined`; 3.28.4 fails, 3.29.2
works). Outside a Visual Studio developer prompt three more fail, and the documented command
should still work there: CMake enables RC for Windows `clang++` and finds `rc.exe` only on
PATH (take the newest from the Windows SDK, `KitsRoot10` in the registry, `bin/10.*/x64`);
FindOpenMP finds `libomp.lib` for Windows `clang++` only through the prompt's `LIB` (preset
`OpenMP_CXX_FLAGS=-fopenmp`, `OpenMP_CXX_LIB_NAMES=""` when every source is HIP and linked with
`-fopenmp`, which adds libomp itself); and pip's isolated build sets `PYTHONPATH` to a
sitecustomize that hides the environment's own site-packages, so a pip-installed ROCm SDK's
`hipconfig.exe` in that same environment prints nothing (`unset(ENV{PYTHONPATH})` in the
Windows HIP configure; CMake and hipify call hipconfig too, and hipify then misreads the ROCm
version). scikit-build-core picks a Visual Studio generator on Windows, which has no HIP
support. Do not use a `[[tool.scikit-build.overrides]]` with `if.failed = true` to retry with
Ninja: any `if.failed` override makes scikit-build-core withdraw its
`prepare_metadata_for_build_*` hooks on every platform (pip then compiles during "Preparing
metadata" and metadata-only queries build the library), and a failing Windows CUDA or CPU build
is retried with Ninja. Instead use a small in-tree PEP 517 backend (`build-backend` +
`backend-path`) that re-exports `scikit_build_core.build` and, only on Windows with the HIP
option set and no generator chosen, appends `-GNinja` to `CMAKE_ARGS` in the `get_requires_*`
and `build_*` hooks; scikit-build-core (1.0 through 1.1.2) then also lists `ninja` as a build
requirement when Ninja is missing, and every other build is untouched. (LEAP, windows-gfx1151)

**rocRAND's device headers need C++14 from ROCm 10.1.** `rocrand_threefry_common.h` uses
`std::enable_if_t`, so a TU that includes `hiprand_kernel.h` under `-std=c++11` fails with
`no template named 'enable_if_t' in namespace 'std'`. ROCm 7.2 still accepted C++11, so a
port validated there breaks only on a newer SDK. Look for a build script that pins
`-std=c++11` for nvcc and reuses it for hipcc; raise it on the HIP path only. (TIGRE)

**nvcc accepts a spaced-out launch, `kernel << <grid, block >> > (...)`; clang does not.**
Visual Studio's auto-formatter produces it, and clang reports only "expected expression" at
the launch line. Rewrite it as `kernel<<<grid, block>>>(...)`, valid for both. (cuda_voxelizer)

**The CUDA Samples `helper_math.h` makes vector arithmetic ambiguous under HIP.** HIP's
`float3`/`int3`/... are `HIP_vector_type`, which already defines element-wise operators: the
vector-vector compound assignments as non-template MEMBERS and vector-vector `*`, `/` as
non-template FRIENDS, the rest as templates. helper_math's free non-template
`operator+=(float3&, float3)`, `operator*(float3, float3)`, ... tie with those, so every such
expression fails as ambiguous, on host and device, including inside helper_math itself
(`smoothstep`). Do not guard the colliding functions one by one (about 40 of them): wrap the
whole negate/add/subtract/multiply/divide block in `#if !defined(__HIP_PLATFORM_AMD__)` and
let HIP's operators, which compute the same per-component results, take every case. Its
host `min`/`max`/`fminf`/`fmaxf`/`rsqrtf` block is guarded on `#ifndef __CUDACC__`; extend it
to `&& !defined(__HIPCC__)`, since clang's HIP math header defines host+device `min`/`max`
and the copies become redefinitions in `.hip`/`LANGUAGE HIP` TUs. Host `.cpp` files still need
the block. (cuda_voxelizer)

## Types, dispatch and platform limits

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

**Library swaps.** cuBLAS -> hipBLAS, cuFFT -> hipFFT, cuRAND -> hipRAND, cuSPARSE ->
hipSPARSE, cuDNN -> MIOpen, Thrust/CUB -> rocThrust/hipCUB. Mostly 1:1; watch handle types
and a few signature differences such as the hipBLAS v2 enums.

**Negative entries for that table.** CUPTI has no ROCm analogue -- stub the optional
per-kernel profiling path inert and let the timer fall back to `hipEvent` wall-clock rather
than substituting something. ck_tile's fused MHA ships headers but no prebuilt instance
library in `/opt/rocm/lib`, so `fmha_fwd` is declared-only and using it means vendoring CK's
codegen (how, below). (cuSZ, lc0)

**Fused FP16 attention (cuDNN SDPA, flash attention): MIOpen has no path, CK's FMHA does.**
MIOpen's find-2.0 MHA problem (`miopenCreateMhaProblem`, ROCm 10.1 SDK) offers only the
`MhaForward` solver on gfx950: FP32, packed BHSD only (no strided tensors), and the causal
mask (`miopenTensorMhaMask`) is not applied -- the output equals the unmasked attention. For
FP16 generate ck_tile FMHA instances: fetch CK's `include` and `example/ck_tile/01_fmha`
(sparse checkout of a pinned commit), run `generate.py --api fwd --optdim 32,64,128,256
--filter <fnmatch>` and compile the instances with the FMHA example's flags; call
`fmha_fwd(traits, args, stream_config{stream})` from a small C ABI. Gotchas: the filter
matches the kernel name WITHOUT the `_gfx950` suffix (end the pattern at `_nsink`, or it
matches nothing); release tag rocm-7.2.4 generates no gfx11 forward kernels (develop does);
cuDNN's `set_causal_mask` is CK `mask_top_left` (window left -1, right 0); keep
`-fgpu-flush-denormals-to-zero` on the FMHA sources only; for one build spanning gfx9 and gfx11
define `USE_NEW_UNIFIED_FRAMEWORK=0` (CK develop's unified framework needs a single host-side
target, and `CK_CMAKE_GPU_TARGET_IDS` takes the first for every pass, failing the gfx11
device pass). 139 FP16 inference instances for
nine architectures build in about 2.5 minutes at 128 jobs. (TornadoVM)

**Generic GEMM on CK device operations.** For CUTLASS-style row-major GEMMs with fused
epilogues: `DeviceGemmMultipleD_Xdl_CShuffle` on CDNA (FP16/BF16/FP32),
`DeviceGemmMultipleD_Wmma_CShuffleV3` on RDNA3/4 (FP16/BF16; its `MakeArgumentPointer` takes
an extra split-K argument), `DeviceGemmMultipleD_Dl` for FP32 on RDNA. Try each and run the
first whose `IsSupportedArgument` passes on the current device. Epilogues are elementwise
functors `op(e, c, d...)`; a bias is a D operand with row stride 0. Operand vectors must
respect the real pointer alignment (8 bytes for data after a 24-byte header: 4-element FP16
vectors). The DL operation has no K padding (needs M, N, K tile multiples) and its D-operand
epilogue gave wrong results on gfx950: run it on zero-padded copies and combine `beta * C` in
a small store kernel. CK's `ck/config.h` is generated: `configure_file` its `config.h.in` with
`CK_ENABLE_FP16/BF16/FP32`, `CK_ENABLE_DL_KERNELS`, `CK_USE_XDL`, `CK_USE_WMMA`. (TornadoVM)

**RAPIDS libraries: hipDF, hipVS and hipMM as C++ wheels.** AMD's package index
(`https://pypi.amd.com/rocm-7.2.3/simple`) carries `amd-libhipdf` (libcudf headers, libcudf,
hipcomp, CMake packages), `amd-libhipmm` (librmm; it requires `amd-rocmds-logger`, which
holds rapids_logger), `amd-libhipvs` (libcuvs, libcuvs_c) and `amd-libraft`, laid out like the
RAPIDS wheels; `amd-libhipdf` and `amd-libhipvs` declare no dependencies, so add
`amd-libhipmm` yourself (pip then brings `amd-rocmds-logger`; unpacking wheels by hand, take
all three). hipDF keeps cuDF's API and ships CUDA-named
compat headers (`cudf/cuda_runtime.h`), so a cuDF client compiles with ROCm clang, C++20 and
`find_package(cudf)` (CMake >= 3.30.4; put each wheel's `lib64/cmake` on the prefix path, and
the wheels before ROCm's, whose hipccl lacks `CCCL::CCCL`). Differences from current RAPIDS:
hipMM 4.0 keeps memory resources under `rmm/mr/device/` and `set_current_device_resource`
takes a pointer; hipDF 3.0's Parquet writer has no Zstandard compressor ("nvCOMP 2.4 or
newer is required") and `LIBCUDF_HOST_COMPRESSION` does not route around it; hipVS takes
DLPack device tensors as `kDLCUDA` and refuses `kDLROCM`, builds a host dataset's
all-neighbors graph into device outputs only, and needs `libomp.so` from ROCm's
`lib/llvm/lib`. The libraries link `libamdhip64.so.7` and resolved against a ROCm 10.1 SDK.
(TornadoVM)

**Choosing the NVIDIA or AMD build of a library at run time.** A runtime with both a CUDA and a
HIP backend must pick per device, from the device's own backend, and never by enumerating or
starting every installed backend: creating a backend that cannot start (HIP on an NVIDIA host)
throws, and a static initializer that does so breaks the library on the device that works. Where
the two builds have different sonames (NCCL `libnccl.so.2` and RCCL `librccl.so.1`), bind both,
each loaded on first use. Where the HIP port keeps the soname (hipVS `libcuvs_c.so`, hipDF
`libcudf.so`), a process can hold only one, so detect which loaded: a `dlsym` through the
library's handle also searches its dependencies, so finding `hipGetDevice` marks the HIP build.
Then serve only devices of that backend. (TornadoVM)

**Library calls inside HIP graph capture.** hipSPARSE's generic API breaks capture in three
ways cuSPARSE does not: destroying a CSR `SpMat` descriptor (`hipsparseDestroySpMat`) calls
`hipDeviceSynchronize`, which invalidates the capture (`hipStreamEndCapture` returns 901, in
every capture mode; destroying a `DnVec` does not, ROCm 10.1 on gfx950), the default CSR SpMV
algorithm analyses on first use
(`HIPSPARSE_STATUS_INTERNAL_ERROR` while capturing), and a non-null external buffer whose
queried size is 0 returns `INVALID_VALUE` once descriptors are reused. Create descriptors
before capture and cache them per operand set, use `HIPSPARSE_SPMV_CSR_ALG2`, and pass a null
buffer when the size is 0. `AMD_LOG_LEVEL=3` shows the call that broke the capture. (TornadoVM)

**GPU power (the NVML replacement) is AMD SMI.** `amdsmi_init(AMDSMI_INIT_AMD_GPUS)`, find
the GPU by PCI address (`hipDeviceAttributePciDomainId`/`PciBusId`/`PciDeviceId` into the
64-bit `amdsmi_bdf_t`; a HIP device index counts only the `HIP_VISIBLE_DEVICES` GPUs), then
`amdsmi_get_power_info`: watts, `current_socket_power` on MI300 and later, unsupported
fields read `UINT32_MAX`. (TornadoVM)

**A ROCm-only branch inside a shared template still COMPILES on the CUDA path.** Gating the
divergence on a `static constexpr bool` predicate that is false on NVIDIA (`if(needsCast())`)
keeps the branch from RUNNING there, but a plain `if` instantiates both arms in every
instantiation, so an expression that is ill-formed for the CUDA element type breaks the
NVIDIA build -- and nothing on ROCm notices, because there it is the branch that runs. Write
such branches `if constexpr`: the discarded arm of a condition that is not value-dependent
after instantiation is never instantiated, which is exactly the guarantee a
platform-conditional block needs. The trap that found it: `(ScalarType)0` with
`ScalarType = __half`. In a HOST translation unit, CUDA headers up to 12.1 keep `__half`'s
integer converts behind `#if defined(__CUDACC__)`, so `int` converts to `float` and to
`double` at equal rank and the cast is ambiguous ("call of overloaded `__half(int)` is
ambiguous"). CUDA 12.2 relaxed that gate to `#if !defined(__CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__) || defined(__CUDACC__)`,
so every release up to 12.1 is affected, plus any 12.2-or-newer build that defines the
disable macro; a recent toolkit hides the bug unless the project opts out. Prefer a float
literal (`0.f`) in half comparisons anywhere, and compile-check the CUDA path against 12.1
headers or older, not only the newest toolkit installed -- bisected against the real
headers, 12.1 fails and 12.2.53 onward compiles. (marian-dev)

**A scalar's WIDTH follows the compute/scale type, not the matrix type.** When a ROCm
library refuses the narrow compute type its CUDA counterpart accepted and you widen it to
32F, the `alpha`/`beta` passed through a `const void*` must widen too. The library reads
four bytes through that pointer; aiming it at a two-byte `half` scales by whatever adjacent
bytes happen to be there -- large enough to give infinities in one call, near enough to zero
to erase the product in the next, with no error status anywhere. Grep every call whose
compute or scale type you changed for the scalars that go with it. (marian-dev: hipBLASLt
rejects `HIPBLAS_COMPUTE_16F` and hipSPARSE rejects `HIP_R_16F` as an SpMM compute type;
both call sites kept passing `half` scalars, and the hipBLASLt one survived review and three
platform validations because the assertion that would have caught it was never reached.)

**A zero workspace size does not mean "skip the operation".** Upstream code shaped like
`if(bufferSize > 0) { alloc; call; free; }` is correct only where the vendor always wants
scratch. cuSPARSE always does for `CUSPARSE_SPMM_CSR_ALG2`, so the shortcut is invisible on
NVIDIA; hipSPARSE wants none at these shapes, reports zero, and the multiplication is never
issued -- the output keeps whatever it held, which reads as wrong numbers rather than as an
error. Make the call unconditional and only the allocation conditional. Suspect the pattern
wherever a `*_bufferSize` query gates its own operation. (marian-dev)

**A "find the best algorithm" call WRITES to the tensor named as its output.** MIOpen's
`miopenFindConvolution{Forward,BackwardData,BackwardWeights}Algorithm` benchmark by running
each candidate, so the destination argument is clobbered. cuDNN's fixed-algorithm enums need
no such step, so a port that introduces one has nothing to inherit this from. Give each find
a buffer that is about to be overwritten anyway: the forward find may take the real output,
but the backward finds must NOT be handed the live input and weight tensors merely because
the signature accepts them. Getting it wrong corrupts the model in place and presents as
training diverging to NaN, not as a wrong kernel. Cache the choice per input shape -- a graph
replays the same convolution once per batch. (marian-dev)

**MIOpen honours only a unit alpha and a zero beta for 2D convolution, pooling and bias.**
cuDNN's accumulate form (`beta = 1`, used by every backward pass that adds into an existing
gradient) has to be rebuilt as an operation into a staging buffer followed by
`miopenOpTensor(miopenTensorOpAdd, ...)`. MIOpen's pooling backward also reads the indices
its forward pass recorded, so that workspace must outlive the forward call rather than being
a local. (marian-dev)

**hipSPARSE has no uniform half-precision SpMM.** Half operands are accepted only in the
mixed-precision mode, which pairs half sparse and dense inputs with a FLOAT result matrix and
float scalars; asking for a half result is `HIPSPARSE_STATUS_NOT_SUPPORTED` (and an all-half
compute type is `INVALID_VALUE`). Compute into a float scratch and cast back. The supported
combinations are tabulated in `rocsparse_spmm.h`, which is faster to read than to probe.
(marian-dev)

**Never name the pointee struct of an opaque CUDA handle.** `CUstream_st`/`CUevent_st` do
not exist under HIP. Spell it `std::remove_pointer_t<cudaStream_t>` so it survives the
typedef swap. (cuSZ)

**`char` vs `signed char` vector base types differ.** HIP defines
`using char4 = HIP_vector_type<char, N>` with plain `char` members; NVIDIA's `char4` has
`signed char` members, and the two are distinct C++ types even when identically
represented. A type-machinery layer must report `char` as the base type of the canonical
`charN` on HIP, so a reference accessor binds. A test asserting
`is_same_v<signed char, BaseType<MakeType<signed char,N>>>` then fails on HIP. This is
dictated by the upstream HIP header, not the porting layer: forcing it would break the
reference accessor or introduce a non-canonical vector family into shared code. Numeric
behaviour is identical -- document it as an upstream type-identity deferral, not a bug.
(CV-CUDA: TypeTraitsMakeTypeVectorTest/3.)

**AMD compute-capability values COLLIDE with NVIDIA arch numbers**, so CC-keyed dispatch
picks the NVIDIA path at RUNTIME on AMD. `hipDeviceAttributeComputeCapabilityMajor` and `hipDeviceProp_t::major` report 9 on gfx90a --
the same value as Hopper sm90 -- so code selecting a kernel config by "cc>=9 means Hopper
DPX/wgmma" tries to launch intrinsics AMD lacks, and a green compile does not catch it.
Gate arch-specific selection on `__HIP_PLATFORM_AMD__` and force the portable config; never
trust CC major/minor for arch dispatch on AMD. Found independently by both libmarv ports.
(MMseqs2, foldseek: the CUDASW++ short2-DPX Smith-Waterman selector.)

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

The same class with NVRTC instead of embedded PTX (CUDA C++ generated at run time) ports to hipRTC.
Bind it by absolute path under `ROCM_PATH`/`HIP_PATH` before any bare name (on Windows the hipRTC DLL
carries the version, `hiprtc0702.dll`, so scan `bin`). hipRTC declares `half`, `__hip_bfloat16` and
the fp16/bf16 intrinsics itself, but `#include <hip/hip_fp16.h>` fails without `-I<rocm>/include`:
guard any include with `__has_include`. Spellings that differ or are missing in hipRTC:
`__float2bfloat16` (no `_rn`), no `__double2half`, `__dp4a` or `llmin`/`llmax`. Pass
`--offload-arch=<gcnArchName>` explicitly (read `gcnArchName` from `hipDeviceProp_t` through the
versioned `hipGetDevicePropertiesR0600`, whose layout that version fixes), and read device attributes
by their HIP enum values, which differ from CUDA's. HIP's context API is deprecated: a backend built
around primary contexts can stand the device in for the context and make it current with
`hipSetDevice`. clang contracts `a * b + c` by default as NVRTC does; `-ffp-contract=off` is the
`--fmad=false` equivalent. (TornadoVM: Java FFM bindings, TC-generated HIP C++.)

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

**An arch-specific fix keyed on the OS or the ROCm version can BREAK an already-validated
arch**, because those are only PROXIES for the real dependency. A Windows setup on TheRock
nightly wheels can pair a newer ROCm with an OLDER torch while a Linux setup pairs an older
ROCm with a newer torch, so OS, ROCm version and torch hipify generation are tangled and can
ANTI-correlate. A guard written `#ifdef _WIN32` when the true dependency is the
hipify generation works only by that accidental correlation. Guard on the real axis, and
there are three:

- **OS and toolchain** -> `_WIN32`: MSVC-vs-MinGW linkage, DLL import libs, path
  separators. (The `/ALTERNATENAME` workaround for missing inherited-constructor exports
  belongs here.)
- **HIP runtime version** -> `HIP_VERSION_MAJOR`/`MINOR`: header behaviour, added or
  removed symbols. (MMseqs2: a Windows commit defined `HIP_DISABLE_WARP_SYNC_BUILTINS`
  unconditionally because `<amd_hip_bf16.h>` defines `__shfl_*_sync` on ROCm 7.14+, but on
  7.2.x `__syncwarp` lives inside the SAME guard, so the define stripped `__syncwarp` and
  broke Linux.)
- **torch hipify generation** -> `torch.utils.hipify.__version__`. v2 STOPS renaming the
  `c10`/`at` `cuda` classes to `hip`: the CUDA spellings stay public and the hip-spelled
  symbols become `USE_ROCM`-only, where v1 renamed the other way. In a torch CUDAExtension
  the TU IS hipified, so just write the CUDA spelling. In a NON-hipified TU -- a
  CMake/`USE_HIP` port, or a host `.cpp` routed around hipify -- detect the version at build
  time, pass a NEUTRAL define (`-DTORCH_HIPIFY_V2`, never one named after your team or tooling), and branch:
  v2 -> `c10::cuda::getCurrentCUDAStream`, v1 -> `c10::hip::getCurrentHIPStream`. (aihwkit: a Windows commit hard-coded
  `c10::hip::getCurrentHIPStream`, needed on TheRock's older torch but absent on Linux's
  newer one, breaking that build.)

Corollary for validation: a Windows-LABELED change is NOT automatically inert elsewhere --
it can touch a shared header that compiles differently against another platform's ROCm or
torch. When one lands after other platforms have passed, REBUILD on those platforms rather
than skipping them on "it is Windows-gated" reasoning. Rebuilding is how both regressions
above were caught; a reasoning-only skip would have shipped a broken build. (aihwkit,
MMseqs2)

## Vulkan interop and image codecs

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

**rocJPEG writes past the row pitch.** Its output conversion writes each row out to a
multiple of 8 pixels regardless of `RocJpegImage::pitch`, so with a tight pitch the first
pixels of every following row are overwritten -- nondeterministically, up to full-scale
errors. Decode into staging with rows padded to 16 pixels and copy the visible region out
(`hipMemcpy2DAsync`), which also orders rocJPEG's own stream before the caller's.
(LichtFeld-Studio)

**Fall back to the CPU only for JPEGs rocJPEG cannot decode, and say so.** rocJPEG 1.10 on
VCN decodes only 8-bit baseline (SOF0) JPEG, 1 or 3 components, 4:4:4/4:4:0/4:2:2/4:2:0,
at least 64x64. Extended sequential (SOF1, which nvJPEG decodes), progressive,
arithmetic-coded, CMYK and 4:1:1 streams fail `rocJpegStreamParse` with `BAD_JPEG` -- the
same status a corrupt file gets -- and small images fail `rocJpegDecode` with
`JPEG_NOT_SUPPORTED`. So classify from the JPEG frame header
(SOFn marker, precision, component count, sampling factors) yourself, decode only those cases
on the CPU with a warning naming the reason, and let every other failure (OOM, decoder
creation, corrupt data) surface as an error. A catch-all fallback hides real failures behind
a silently slower path. (LichtFeld-Studio review)

**A JPEG that ends before its scan hangs the VCN JPEG engine.** rocJPEG 1.10 parses such a
stream as SUCCESS (dimensions and all) and submits it; the engine hangs until the driver
resets the `jpeg_dec` ring, and the process's GPU context is lost ("context is guilty of a
hard recovery", then abort). One corrupt file in a dataset kills the app, where nvJPEG
returns an error. Only the missing scan was measured to hang: a JPEG cut inside its scan
(no end-of-image marker) decodes, with no reset, and one with no scan hangs. A scan header
with no data was only parse-checked (SUCCESS). Walk the markers on the host before calling
rocJPEG and require at least a scan with data; the port also requires a frame header,
defined quantization tables, a well-formed scan header and an end-of-image marker, as a
precaution against corrupt streams, and treats any failure as a decode error. Such a
stream may appear only in a test of that host-side check, which is then the one barrier
in front of the hardware: a regression in the check costs a ring reset, so run that test
on a GPU no other job is using. (LichtFeld-Studio; deferral `rocjpeg-no-scan-jpeg-hangs-vcn`)

**NVDEC through FFmpeg becomes FFmpeg demux plus rocDecode.** FFmpeg has no rocDecode
hwaccel, and the CUDA device context behind `*_cuvid` fails on AMD, so the code silently
falls back to software decode. TheRock's SDK ships rocDecode's library, headers and CMake
package but not the samples' `VideoDemuxer`/`RocVideoDecoder` helpers, so keep FFmpeg for
demux and drive `rocDecCreateVideoParser`/`rocDecParseVideoData` yourself behind an
`avcodec_send_packet`/`receive_frame`/`flush_buffers`-shaped class: H.264/HEVC packets
through `h264_mp4toannexb`/`hevc_mp4toannexb`, the AV1 `av1C` config OBUs (extradata after
4 bytes) prepended to the first packet after open or flush, a flush that destroys and
recreates the parser (keep the decoder when the next sequence header matches), and frames
of `AV_PKT_FLAG_DISCARD` packets (outside an MP4 edit list) dropped by pts as libavcodec
does, or the frame count is one higher than the software decoder's. Copy each displayed
picture (`rocDecGetVideoFrame`, offset by the display area) into an `AVBufferPool` of
`hipMalloc`'d NV12/P010 buffers so frames stay on the GPU. Measured on gfx1100 against
libavcodec/libdav1d: H.264 High, HEVC Main/Main 10, VP9 profiles 0/2 and AV1 8/10-bit
bit-exact, timestamps preserved. (LichtFeld-Studio)

**rocDecode's capability query is not a support check by itself.** `rocDecGetDecoderCaps`
takes codec, chroma format and bit depth but no profile, and rocDecode 1.10 decodes an
H.264 High 4:4:4 Predictive stream with 4:2:0 chroma (x264 lossless) with
`rocDecodeStatus_Success` and wrong pixels (its AVC picture parameters have no field for
the SPS lossless flag `qpprime_y_zero_transform_bypass_flag`, so the engine is never told).
Gate the hardware path on the stream's profile too (H.264 Baseline/Main/High; HEVC Main/Main
10/Main Still) and on the caps size limits, and decode the rest on the CPU with a logged
reason. The caps limits are coded sizes: before the minimum check, round the display size
up to the codec's block size (16 for H.264, 8, the smallest coding block, for HEVC; VP9 and
AV1 code the frame size itself). Comparing the display size refuses valid streams: a 60x60
H.264 or HEVC stream is coded 64x64, inside VCN 4's 64x64 minimum, and decodes bit-exact.
The maxima (4096, 8192, 4352) are multiples of 16, so the display size gives the same answer
there. Also do not trust the parser's `video_signal_description` for VP9 and AV1:
it reports limited range and matrix 0 for full-range streams; take those from FFmpeg's
stream parameters (H.264/HEVC VUI values are right). Validate a port's decode against the
software decoder frame by frame, not by eye. (LichtFeld-Studio; deferrals
`rocdecode-h264-high444-420-silent-miscode`, `rocdecode-vp9-av1-signal-description-missing`)

**A rocDecode success status does not mean right pixels: check a reference clip once per
process before trusting a codec.** On gfx90a (MI250X, amdgpu-dkms 6.16.6), rocDecode 1.10
from TheRock SDK 10.1 loads the SDK's bundled Mesa VA-API driver
(`rocm_sysdeps/lib/radeonsi_drv_video.so`), and through it every H.264, HEVC and VP9 frame,
8- and 10-bit, is wrong in nearly every byte while `rocDecGetDecodeStatus` says success;
Ubuntu's Mesa driver on the same card (`LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri`)
and the bundled driver on gfx1100 are bit-exact. Caps, profile and size gates cannot see
this, so the port decodes an embedded two-frame 64x64 clip per codec and bit depth the first
time a process asks about it, through the same demux, bitstream filter and copy code as real
files, and compares each frame's MD5 with the digest of FFmpeg's software decode
(`ffmpeg -i clip -pix_fmt nv12|p010le -f framemd5 -`, which hashes the frame packed with no
row padding, the layout `av_image_copy_to_buffer(..., 1)` gives a host copy). A mismatch
refuses that codec for the process with a named CPU reason. Key the result by device,
codec and bit depth under a mutex, run it inside the capability query (lazily, not at
start-up), and do not cache a decode error (that is an error, not a verdict). Cost on
MI250X: 40-46 ms for the first codec (includes loading the VA driver), 7-18 ms per further
codec. Even the intra frame was wrong, and 10-bit failed with 8-bit, but check each codec
and depth: one codec's result is not evidence for another. Test it without a broken driver
by injecting a wrong digest, and add a test that the verdict agrees with a full decode of
the fixtures, so the same suite proves the check on a good driver and a bad one. Do not
switch the VA driver from inside the application. (LichtFeld-Studio; deferral
`rocdecode-sdk-va-driver-gfx90a-wrong-frames`)

**The software fallback for AV1 needs libdav1d.** FFmpeg's built-in `av1` decoder only
drives hardware accelerators (without one it fails with "Get current frame error"), and
vcpkg's `ffmpeg` port builds no software AV1 decoder unless the `dav1d` feature is on. A
CUDA project that relies on NVDEC for AV1 often never enables it, so a ROCm port's CPU
path has nothing to decode AV1 with on GPUs without AV1 decode hardware (gfx90a), or on a
build without rocDecode. Add `dav1d` for the ROCm build only, as a manifest feature (a
`rocm` entry under `features` in `vcpkg.json` that lists `ffmpeg` with `dav1d`) appended
to `VCPKG_MANIFEST_FEATURES` under `if(USE_HIP)` before `project()`, the way projects
already select a `tests` feature; a CUDA configure never selects it, so its dependency set
is unchanged. Do not generate a modified copy of the manifest in the build tree instead: it
is a script rewriting a dependency choice (overlay paths, registries, user overrides and
pkg-config caches all need handling) where the manifest has a declarative mechanism. The
vcpkg toolchain caches `VCPKG_MANIFEST_DIR` with FORCE, so a build tree that ever pointed it
elsewhere keeps doing so until reconfigured with `-UVCPKG_MANIFEST_DIR`. (LichtFeld-Studio)

