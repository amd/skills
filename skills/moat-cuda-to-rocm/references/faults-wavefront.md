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
  device compilation; there is no `__AMDGCN_WAVEFRONT_SIZE__` in ROCm 7.2.x or 7.14, so the
  `__GFX*__` guards are the supported selector:

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
are the same shift; use one spelling per project.) (cuSZ)

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
warpSize`) keep the full mask. (LichtFeld-Studio: fastgs/gsplat 32-lane tile macros, nn
softmax/norm block reductions, `block_reduce_min4`.)

## Shuffles and logical warps

**Width-32 LOGICAL warp ops are arch-agnostic and fine.** `__shfl*(...,32)`,
`cub::WarpReduce<,32>` and `cg::tiled_partition<32>` work within a 32-lane
subgroup whatever the physical width. That includes the tile's `ballot()`: HIP's
`thread_block_tile<N>::ballot` returns `unsigned long long`, but compacts the tile's lanes
to bits `0..N-1` (`internal::helper::adjust_mask`), so storing it in a 32-bit integer and
indexing it with `__ffs`/`__popc`/`__fns` is correct on wave64. Only a RAW `__ballot`
truncated to 32 bits (the next entry) is the fault. (faster-gaussian-splatting)

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

**`cub`/`hipCUB` block collectives race on a 64-thread block without an explicit
`__syncthreads()`.** At wave64 a 64-thread block is a single wavefront, so a `BlockReduce`
or `BlockRadixSort` lowers to a single-wavefront op with no syncing epilogue; reusing one
`TempStorage` union across back-to-back calls lets the second call's writes overlap the
first's reads. CUDA's 32-wide warp split the block in two and masked it. (CV-CUDA:
OpPairwiseMatcher crossCheck.)

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

**Validate wave-size work on BOTH widths, and where there is no host reference use a
CROSS-ARCH CONSISTENCY gate.** "Deterministic, non-zero and plausible" passes a wrong
wave32 result -- popsift's gfx1100 reported deterministic non-zero features from a
miscounted extrema path, a false pass. Where the algorithm is deterministic, diff the
wave32 output against the wave64 output for the same input. (popsift)
