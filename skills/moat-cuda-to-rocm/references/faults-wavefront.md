# Faults: wavefront and warp width

NVIDIA warps are always 32 lanes; AMD wavefronts are 64 on CDNA and 32 on RDNA. Everything here follows from code that assumed 32.

<!-- toc:begin -->
Contents:

- Choosing the width
- Lane masks and ballots
- Shuffles and logical warps
- Barriers and block collectives
- Other per-wavefront assumptions
<!-- toc:end -->

## Choosing the width

**Never hardcode 32.** NVIDIA warps are always 32 lanes. AMD wavefronts are 64 on CDNA
(gfx90a, gfx94x) and 32 on RDNA (gfx10xx, gfx11xx, gfx12xx).

- Host code (launch geometry, host-side shared-memory sizing) queries at runtime:
  `hipGetDeviceProperties(&prop, dev); prop.warpSize`, or `hipDeviceAttributeWarpSize`.
  PyTorch exposes `at::cuda::warp_size()`.
- Device code uses a compile-time per-arch constant. `__GFX9__` is defined only during
  device compilation; there is no `__AMDGCN_WAVEFRONT_SIZE__` in ROCm 7.2.x or 7.14, and
  ROCm 10.1's clang no longer defines `__AMDGCN_WAVEFRONT_SIZE` either, so code that picks a
  width from it at preprocessing time falls to its non-AMD default (check with `amdclang++
  -x hip --offload-arch=gfx90a --cuda-device-only -nogpulib -dM -E`). The `__GFX*__` guards
  are the supported selector:

        #if defined(__GFX8__) || defined(__GFX9__)
        static constexpr int kWarpSize = 64;   // CDNA: gfx90a, gfx94x
        #else
        static constexpr int kWarpSize = 32;   // RDNA (gfx10xx, gfx11xx, gfx12xx), CUDA
        #endif

  Do not wrap this in `__HIP_PLATFORM_AMD__`: above the includes that macro is undefined
  and the gate silently picks 32 on wave64 hardware (build-headers-macros.md). The
  `__GFX*__` predefines need no header and nvcc never defines them.

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

**A block NARROWER than one wavefront makes `blockDim.x / warpSize` zero, and a block
reduction built on that count silently returns zero.** The idiom
`val = (tid < blockDim.x / warpSize) ? warpSums[lane] : 0` is exact on NVIDIA for any block
that is a multiple of 32, so a launch that sizes its block from the work
(`block.x = work <= 32 ? 32 : ...`) is safe there and broken on wave64: the count is 0, every
lane takes the empty branch, and the first stage shuffles lanes 32..63 that are not in the
block. Nothing errors; the reduction just yields 0. Ceil-dividing the count is not enough,
because it leaves the out-of-block shuffles. Floor the BLOCK at the wavefront width instead,
from a host-side runtime query (`hipDeviceAttributeWarpSize`), so a wave32 device keeps the
launch it had and a wave64 device gets a whole wavefront. Any launch whose block size is
computed from a data size, rather than being a fixed 128 or 256, is a candidate. (CV-CUDA:
OpFindHomography refinement, reachable at 16 or fewer point pairs.)

**The shuffles themselves read lanes that hold no thread, whatever the count says.** On
NVIDIA a 32-thread block fills its warp exactly, so a reduction such as
`for (off = warpSize / 2; off > 0; off /= 2) v += __shfl_down(v, off);` only ever reads live
lanes. On wave64 the same block is half a wavefront: `warpSize / 2` is 32, lane 0 reads lane
32, and lanes 32-63 hold no thread, so the value returned is undefined and is added in. A
minimal kernel (each thread contributes 1, 4096 blocks) gives the right sum for 64-thread
blocks and a wrong sum in every block at 32 threads (gfx942). Where the launch can change, the
block floor above fixes this too. Where the block width is fixed by the algorithm, add only
lanes that hold a thread
(`if (lane + off < warpSize && (mask >> (lane + off)) & 1) v += other;`, with the mask built
from the block size), or start the loop at the block width when the block is smaller than a
wavefront. Sweep every block size the code can launch, including 32 and below.
(cuda-samples `reduction` `reduce7`/`warpReduceSum`.)

**Warp-tiled dispatch tables written for 32 lanes break when `kWarpSize` becomes 64.**
Making the warp-size constant arch-correct is not the end of it; audit what is built FROM
it. Two shapes recur. (1) A square `kWarpSize x kWarpSize` block (one warp of columns times
one warp of rows) is 1024 threads on CUDA and 4096 on wave64, which fails to launch with
`invalid configuration argument`; cap it at 1024 and let the row count shrink. (2) A width
ladder that picks a thread-group width of 4, 8, 16, 32 for narrow rows and then switches to
"one full warp per row, `min_cols * kWarpSize` columns read unconditionally" rungs: on
wave64 the first full-warp rung receives rows of 33..63 columns and reads past them, and
pack-2 rows of 66..128 match no rung at all (`invalid argument`). Add a 64-wide rung
compiled only when the wavefront is 64. Sweep every width 1..N on wave64; spot widths like
200 and 1024 pass and hide it. (oneflow: `MatrixColReduceByWarpBlock`, and the LayerNorm
and RMSNorm warp dispatch in `layer_norm.cuh`/`rms_norm.cuh`.)

**Special case: a warp-width-dependent SERIALIZED FORMAT.** If the warp width determines an
on-disk or in-memory layout rather than just launch geometry, it cannot follow the per-arch
device constant -- pin it to a fixed maximum width, or carry the producing device's width
at runtime. dietgpu's rANS archive serializes one coder state per lane
(`ANSWarpState::warpState[kWarpSize]`), making its header geometry width-coupled; pinning
`warpState[64]` keeps every `sizeof()` and header offset arch-independent, leaves the
wave64 archive byte-identical to its old format, and lets a wave32 device use the first 32
slots. (dietgpu)

**HIP's device `warpSize` is not a plain `int`** and device `printf` rejects it as a
vararg: cast it, `printf("%d", (int)warpSize)`. The raw diagnostic is misleading -- it
surfaced as an undeclared-identifier cascade in a generated dispatch header. No-op on
CUDA. (LC-framework)

## Lane masks and ballots

**Lane masks are 64-bit on HIP: `__ballot` and `__activemask` RETURN `unsigned long long`,
and the `*_sync` functions TAKE a 64-bit mask.** Store a ballot or an active mask in a 64-bit
integer; a `uint32` keeps lanes 0..31 only and is wrong on a 64-wide wavefront. (The non-sync
`__shfl*` forms take a width, not a mask.)

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
OnesweepReorderKeyPair64`, not a wrong answer. Fix it with the arch-unified shift of
the compat `__ballot_sync` entry below, which is a no-op on wave32 and on the CUDA
`__ballot_sync` path. Note the reachability
trap: the sort dispatches a single-pass kernel below 3072 elements and only reaches this one
above it, so a small harness scene never executes the defect -- two review rounds passed
44/44 over a back end that could not build an acceleration structure for any real scene. If
a dependency picks kernels by problem size, one test case has to be big enough to cross
every such threshold. (diff-surfel-tracing, in the Orochi copy vendored by its HIP RT
dependency.)

**A compat `__ballot_sync` that casts `__ballot()` to `uint32_t` takes the LOW 32 lanes**,
which are the wrong lanes for any logical warp not at the wavefront base -- with
`blockDim (32,32)`, odd `threadIdx.y` rows sit in the high lanes. Arch-unified form:
`(uint32_t)(__ballot(PRED) >> (__lane_id() & ~31u))`, where the shift is 0 on wave32.
(`__lane_id() & 32u`, and `(threadIdx.x % warpSize) / 32 * 32` in a one-dimensional block,
are the same shift; use one spelling per project.) (cuSZ, GPU_IPC)

**A full 64-bit `*_sync` mask in 32-lane group code traps in a HIP debug build on wave64.**
HIP's `__shfl*_sync` / `__ballot_sync` run `__hip_check_mask`, which asserts that the mask
equals the lanes that reached the call (`__ballot(true)`), and `__hip_assert` is a
`__builtin_trap()` unless `NDEBUG` is defined. Code written for 32-lane CUDA warps and kept
correct on wave64 by group-local folds (offsets below 32, width-32 shuffles, the
shifted ballot of the entry above) is still wrong to pass `0xffff...ffffull` wherever the two
halves of a wavefront can reach the intrinsic separately: a per-group `continue`, a
per-group trip count, or a cross-warp step that runs only `threadIdx.x < 32` (or
`warp_id == 0` with `warp_id = threadIdx.x / 32`). Release builds are correct and wave32
cannot see it, because a group is the whole wave, so only a wave64 debug run traps. Pass the
caller's group mask, `0xffffffffull << (__lane_id() & ~31u)`: it is the whole wave on wave32
and one half on wave64. The check runs a waterfall over distinct mask values, so each half
is validated against itself, whether the halves arrive together or apart. HIP evaluates the
shuffle in the caller's context, so the mask changes only that check and the ballot's
`& mask`, never the data movement. Mask-free `__shfl(v, s, 32)` / `__ballot` also work but
drop the check entirely. Folds over the native wavefront (`warp_id = threadIdx.x /
warpSize`) keep the full mask. A PARTIAL mask needs the same move. Kernels that tile a block
into 32-lane warps (`warp = threadIdx.x >> 5`) and pass a sub-group mask
(`atom_mask = 0xf << 4*k`) to a masked `__syncwarp`/`__shfl*_sync` do not compile with the
32-bit mask, and a mask that is only widened still names lanes of the first tile, so it
differs from `__ballot(true)` of the lanes passing it and can trap. Move it to the lanes the
tile occupies, `(unsigned long long)mask << (__lane_id() & 32)` (shift 0 on wave32), behind
a compat-header macro that is the identity for CUDA. Also pass the tile width explicitly:
`__shfl_sync(m, v, 0)` defaults to `warpSize`, so on wave64 lane 0 of the first tile is
broadcast to both. (LichtFeld-Studio: fastgs/gsplat 32-lane tile macros, nn
softmax/norm block reductions, `block_reduce_min4`; GPUMD: NEP-spin.)
Code written for the native wavefront traps the same way wherever part of the wavefront is
inactive: a block narrower than 64 threads on a wave64 GPU, the last wavefront of a block that
is not a whole number of wavefronts, or a branch on a 32-thread group. Pass `__activemask()` on
HIP, keeping a constant only where code compares a mask against "all lanes" to choose a fast
path; release machine code does not change. (warp-1.14)

## Shuffles and logical warps

**Width-32 LOGICAL warp ops are arch-agnostic and fine.** `__shfl*(...,32)`,
`cub::WarpReduce<,32>` and `cg::tiled_partition<32>` work within a 32-lane
subgroup whatever the physical width. That includes the tile's `ballot()`: HIP's
`thread_block_tile<N>::ballot` returns `unsigned long long`, but compacts the tile's lanes
to bits `0..N-1` (`internal::helper::adjust_mask`), so storing it in a 32-bit integer and
indexing it with `__ffs`/`__popc`/`__fns` is correct on wave64. Only a RAW `__ballot`
truncated to 32 bits (the next entry) is the fault. (faster-gaussian-splatting)

**A compat layer for CUDA's `*_sync` warp functions must pin every member to 32 lanes, and
ROCm's native `*_sync` functions are not that layer.** A CUDA warp is always 32 lanes, so
`__shfl_down_sync(0xffffffff, v, d)` is BY DEFINITION a 32-lane operation. Mapping it to the
unqualified `__shfl_down(v, d)` silently retargets it to the WAVEFRONT width, which is right
on RDNA and wrong on CDNA. Write the whole family with the width in it -- `__shfl(v, lane,
32)`, `__shfl_down(v, d, 32)`, `__shfl_up`, `__shfl_xor` -- beside the shifted ballot of the
compat `__ballot_sync` entry above, and the header is correct on both wavefront sizes with
no arch branch. Check the shuffles even when the ballot looks handled: GPU_IPC's header had
a plausible ballot and unqualified shuffles, and the shuffles were the half that would have
broken CDNA, since its PCG and BVH reductions loop `delta = 1,2,4,8,16` over logical
32-element groups. Do not later "simplify" such a layer into ROCm's own `*_sync` functions
called the CUDA way: ROCm's `__ballot_sync` returns a 64-bit WAVEFRONT-wide mask, and its
`__shfl_*_sync` default `width` to `warpSize`, which is 64 on CDNA. That substitution looks
like a cleanup, compiles everywhere, passes on RDNA and silently changes behaviour on
gfx90a/gfx94x. (The native functions are usable with an explicit width of 32 and the group
mask of the debug-trap entry above; what breaks is relying on their defaults.) Write the
layer as functions under project-owned names, not as macros named after the intrinsics
(build-headers-macros.md, "Do not name compat macros after the warp functions ROCm
declares"). (GPU_IPC)
Which calls break shows where to look first: scans and xor/down reductions guarded by `lane %
32` mostly survive the full-wavefront default, because their reads stay inside a 32-lane half,
while a call that names an ABSOLUTE source lane (`__shfl_sync(~0, v, lane / 2)`, a lane-0
broadcast of per-warp data) silently reads the lower half on wave64. A layer whose ballot
already returns the caller's 32-bit half can default its shuffles to 32 through wrappers with
a default argument (`T shfl(T v, int src, int w = 32)`), so explicit widths still pass
through. (cuSZ: LC-framework output outside its error bound on gfx90a only.)

**A fixed-cluster warp reduction is wavefront-safe exactly when the cluster size equals the
pinned shuffle width AND the clusters are aligned to it: check that, do not restructure the
branch.** Code that partitions work into fixed 32-element clusters (GPU_IPC's `BANKSIZE 32`
preconditioner banks) commonly guards a reduction on a cluster-uniform predicate:
`if (__popc(connectMsk) == BANKSIZE) { for (d = 1; d < 32; d <<= 1) v += shfl_down(v, d,
32); }`. At wave64 one cluster can take that branch while its sibling in the same wavefront
does not, which invites the "fix" of hoisting the reduction above the branch so the whole
wavefront participates. Resist it. Once the compat layer pins the width to 32 (previous
entry), `ds_bpermute` clamps every source index with `self & ~(width-1)`, so a lane only
ever reads its own 32-lane group; with `laneId = threadIdx.x % BANKSIZE` that group IS the
cluster, and the reduction is correct at wave32, at wave64 and on CUDA whatever the sibling
cluster did. The invariant to verify is: cluster size == pinned width, cluster base aligned
to it, predicate uniform over the cluster -- the same invariant that makes the reduction
meaningful on CUDA in the first place. Hoisting is not merely unnecessary, it can ADD a
fault: look for an earlier early-return that leaves a PARTIAL group. GPU_IPC launches this
kernel over `totalNodes = vertNum`, which is not padded to BANKSIZE (38386 vertices, so 18
live lanes in the last bank and 14 that already returned). The guarded form never shuffles
there, because a partial bank cannot reach `popc == BANKSIZE`; the hoisted form does, with a
full `0xffffffff` member mask naming returned lanes -- an undefined source-lane read on both
back ends, bought for nothing. (GPU_IPC `__buildMultiLevelR_optimized`, where the hoist was
written, measured clean, and then reverted for exactly this reason.)

**Pass that logical width on BOTH paths, and prefer the project's own pre-CUDA-9 branch over
remapping `__shfl_*_sync` in the compat header.** An `#ifdef USE_HIP` pair that spells the
same reduction twice, width 32 on HIP and bare on CUDA, reads as an AMD-specific hack and
draws review churn. The 32 is algorithmic (block.x, one descriptor row, a bitonic network),
so writing it unconditionally is both correct and clearer, and NVIDIA generates the same
code. Better still, many projects already select between `__shfl_*_sync` and the mask-free
`__shfl*` on a configure-time flag for CUDA before 9.0. HIP's spelling IS the mask-free one,
with a width up to 64, so setting that existing flag to 0 on the HIP path removes the
compat-header macros entirely, and with them the `<hip/hip_bf16.h>` include-ordering
workaround, since newer ROCm declares real `__shfl_*_sync` templates that function-like
macros of the same name would rewrite. (popsift, from the maintainer's review of its pull
request.)

**An over-wide `__shfl*` width does not error, it silently degenerates.**
`__shfl_up(v, delta, 64)` lowers to `__builtin_amdgcn_ds_bpermute` with a clamp of
`self & ~(width-1)`; on wave32 that clamp is 0, so the permute domain becomes the
32-lane physical wavefront while the loop still treats lanes as a 64-wide group. In a
lossy or statistical pipeline the wrong collective still round-trips, so it surfaces as a
quality-metric shift (compression ratio, PSNR) rather than a crash -- the cross-arch gate
has to compare the METRIC, not just that the output decodes. (cuSZ)

**Do not hardcode wave64 GEOMETRY either** -- the inverse failure. Packing two logical
32-thread rows into one wavefront (`wflane = threadIdx.x + ((threadIdx.y&1)<<5)`, a
`1ull<<wflane` prefix mask, `__shfl(...,0,64)`, `__popcll` over a 64-bit ballot, a
`wflane==0` leader test) COMPILES on wave32 and silently miscomputes: each 32-thread row is
its own wavefront there, so odd-row lanes land at 32..63, the leader test never fires
(lost atomics), and the mask reads phantom bits. Make wave-collective geometry
`warpSize`-generic, or give it an explicit `__GFX9__` branch AND a wave32 `#else`. A
wave64-only fix that passes CDNA is not done until it is correct on RDNA. (popsift
s_extrema.)

## Barriers and block collectives

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

**A lock-retry loop deadlocks when the lock holder is in the same wavefront.** A hash-table or
lock-based kernel that passes on CUDA hangs on AMD with the GPU at 100%, often only under
contention (a full bucket, duplicate keys in a batch). A lane that leaves a loop waits at the
loop's exit until every lane of the wavefront has left; CUDA's independent thread scheduling
(Volta and later) lets a lock holder run ahead, AMD does not. Any retry loop that spins while a
slot is locked deadlocks when the holder is another lane or tile of the same wavefront that
already left the loop and does its copy and unlock after it. Three shapes were seen: a helper
that spins in place (`while (load == LOCKED) {}`, or a ballot-and-`continue` loop) with the
critical section after the call; a caller retrying in `do { ... } while (result == CONTINUE)`
with the critical section after the loop; and a critical section inside the do-while whose loop
condition is statically false once the lock is taken, so the compiler may move it to the loop
exit, where it runs after the other lanes (putting the unlock "inside the loop" is not enough).
Retry through a loop whose latch every lane reaches after finishing its critical section, with
a condition the compiler cannot decide on that path: in a grid-stride kernel re-run the same
index from the top (`t -= stride; continue;`), or exit uniformly (`while (__any(!done))`). Or
never wait on a held slot: when choosing an eviction victim, skip locked slots and retry if none
is left. `AMD_LOG_LEVEL=3` names the last kernel launched and `AMD_SERIALIZE_KERNEL=3` pins down
which of several concurrent kernels never returns. Gate the change so the CUDA path stays
identical, and check that with a PTX diff. (HierarchicalKV, ROCm 10.2, gfx942)

**`cub`/`hipCUB` block collectives race on a 64-thread block without an explicit
`__syncthreads()`.** At wave64 a 64-thread block is a single wavefront, so a `BlockReduce`
or `BlockRadixSort` lowers to a single-wavefront op with no syncing epilogue; reusing one
`TempStorage` union across back-to-back calls lets the second call's writes overlap the
first's reads. CUDA's 32-wide warp split the block in two and masked it. (CV-CUDA:
OpPairwiseMatcher crossCheck.)

**A block collective called in a loop without a barrier races only under concurrent GPU load.** The
same missing barrier on a multi-wave block is timing-dependent, so it can pass every serial run. A
`BlockScan` called once per loop iteration on the same storage, with no `__syncthreads()` between
iterations, gave some threads wrong offsets only when other processes shared the GPU (pytest-xdist
workers), on wave32 gfx1100. The symptom was `hipErrorIllegalAddress` in a LATER kernel that read
the unwritten `-1` slots as indices. Before blaming the kernel that faults, check the index arrays
it was given for unfilled entries, and run the producer under concurrent load. Fix it
unconditionally: the barrier is CUB's documented requirement on CUDA too. (gpu4pyscf:
`put_pairs_on_blocks_kernel`.)

## Other per-wavefront assumptions

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

**A leader-only write broadcast by shuffle must be initialised on every lane.** The
warp-aggregated atomic idiom, `int base; if (lane == 0) base = atomicAdd(ctr, n);
base = __shfl(base, 0);`, reads `base` uninitialised on every other lane. That is undefined
behaviour, and "the shuffle only reads lane 0, so it is harmless" is wrong. clang marks
by-value parameters `noundef`, so passing the indeterminate value into a by-value shuffle
wrapper makes the `lane != 0` path undefined; the optimizer replaces the branch with
`llvm.assume(threadIdx.x == 0)`, EVERY lane performs the atomic (the counter advances by
`warpSize * n`), and any `(1u << lane) - 1` prefix offset folds to 0. The symptom is a count
exactly `warpSize` times the real number of writes, with the surplus slots never written
(zeroed or stale entries downstream). Declare `int base = 0;` or use a ternary; both give
identical ISA, so it costs nothing. A volatile counter does not help. To check a suspect
site, emit device IR (`--cuda-device-only -S -emit-llvm`) and grep for `llvm.assume`;
`-Xclang -no-enable-noundef-analysis` making it vanish confirms the cause. Measured by an
upstream user on gfx1201, ROCm 7.2.1, against popsift's pre-port `extrema_count()`. The same
construct left in a CUDA `#else` branch is the same undefined behaviour waiting for a
compiler, so grep the port for the idiom and initialise it on both paths. The padding-lane
form of this compiler behaviour is in faults-memory-streams.md ("An uninitialised local
passed by value"). (popsift, upstream issue 193)

**An uninitialized value RETURNED from a device function lets clang assume the thread index.** The
return-value form of the previous entry. Block-reduction helpers often assign the result only on
thread 0 (`T sum; if (threadIdx.x == 0) sum = ...; return sum;`) and every thread returns it. nvcc
tolerates this. clang marks return values `noundef` as well, so returning an uninitialized value on
any path is immediate undefined behaviour, and the optimizer may conclude the other paths never run:
on AMD GPUs it emitted `llvm.assume(threadIdx.x == 0)`, then folded `partials[threadIdx.x /
warpSize]` to `partials[0]`, so a two-wavefront block sum returned one wavefront's partial.
Symptoms: correct at -O0, wrong at -O1 and above, and the bug disappears when a printf is added.
Find it by compiling to LLVM IR (hipRTC `-fgpu-rdc` plus `llvm-dis`, or `-emit-llvm`) and grepping
for `llvm.assume` on `workitem.id`. Fix: value-initialize (`T sum {};`) every local that is returned
or copied while unassigned on some threads. (warp-1.14)

**Validate wave-size work on BOTH widths, and where there is no host reference use a
CROSS-ARCH CONSISTENCY gate.** "Deterministic, non-zero and plausible" passes a wrong
wave32 result -- popsift's gfx1100 reported deterministic non-zero features from a
miscounted extrema path, a false pass. Where the algorithm is deterministic, diff the
wave32 output against the wave64 output for the same input. (popsift)
`-mwavefrontsize64` on an RDNA card is not a substitute for the wave64 run; what it can and
cannot show is in validation.md ("`-mwavefrontsize64` on an RDNA card checks lane logic").
