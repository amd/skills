# Faults: memory, lifetime and streams

Where CUDA tolerates undefined behaviour in memory use, object lifetime and stream ordering, and HIP does not.

<!-- toc:begin -->
Contents:

- Bounds and initial contents
- Handle and object lifetime
- Copies and stream ordering
- Host callbacks and fork
- Graph capture
- Managed memory
- Multi-GPU
<!-- toc:end -->

## Bounds and initial contents

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

**An allocation sized by a different expression than the launch indexes it by.** A
`cudaMalloc` whose count comes from one member and a kernel whose global thread id ranges
over another is an overflow of whatever the ratio is, and CUDA can absorb a large one
silently -- HEonGPU's TFHE encryptor allocated 512 random states and initialized and used
512 * 32 of them, a 32x overrun that only AMD turned into wrong answers. Grep every raw
`cudaMalloc` for the expression it is sized by and check it against the grid the consuming
kernel is launched with; when the two disagree, decide which one is the intent (in HEonGPU
it was the launch geometry). This is a memory-safety bug on both platforms, so fix it for
both rather than guarding it. (HEonGPU: `HEEncryptor<Scheme::TFHE>` cuda_rng.)

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

**An uninitialised local passed by value lets clang delete a padding-lane guard, and the
result is NaN that nvcc never produces.** The shape: a kernel declares locals without
initialisers, assigns them only inside `if (idx < n)`, then passes one BY VALUE to a helper
outside the guard (a masked block reduction, `Reduce(valid, value)`, is the usual case).
Padding lanes read an indeterminate value, which is undefined behaviour. clang marks
by-value parameters `noundef`, so LLVM concludes the padding-lane path cannot happen,
inserts `llvm.assume(idx < n)`, and drops both the guard and the helper's
`valid ? value : 0` mask. The padding lanes then compute on out-of-range inputs, store past
the end of outputs, and feed inf/NaN into the reduction. nvcc does not exploit this, so the
CUDA build passes and the HIP build (or CUDA built with clang) gives NaN on any problem
size that leaves padding lanes. It was observed on wave64; the cause is in the compiler and
not the wave width, so expect the same on wave32. To confirm, look for `llvm.assume` on the
thread-index compare in the device IR (`--cuda-device-only -S -emit-llvm`);
`-Xclang -no-enable-noundef-analysis` makes the symptom disappear, as a diagnostic and not
a fix. Fix the source: initialise the locals (zero is the identity of a sum). When the
kernels are generated, a vendored copy of the generator's output can predate the
generator's own fix, so check that the vendored code really carries a fix its changelog
claims. (symforce, colmap)

## Handle and object lifetime

**Rule-of-five on resource handles.** CUDA tolerates a default-constructed or
double-destroyed texture/stream/event handle; AMD faults. Give RAII wrappers explicit
default init (`handle = 0`), move-only semantics, and a guarded destructor. (colmap
CuTexObj.)

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
project save hung or crashed on gfx1100, and a review found three more holders.)

### Using a stream after `hipDeviceReset` corrupts the heap (Windows shows it)

**A process that finishes its work and then dies at exit with `0xC0000374` (heap
corruption), `0xC0000409` or `0xC0000005` -- or `exit 127` under MSYS bash -- may be
destroying a stream that `hipDeviceReset()` already tore down.** Projects that call a
reset in a cleanup function while a C++ object owning streams is still in scope (its
destructor runs after the reset) hit it. Both `cudaDeviceReset` and `hipDeviceReset`
document passing a destroyed resource to a later call as undefined behavior, so this is
the application's misuse; CUDA happens to tolerate it, HIP frees the stream object in the
reset and `hipStreamDestroy` then uses it again (valgrind: use-after-free on Linux too,
where glibc aborts only some runs). The port-side fix is to destroy the owning object
before the reset; whether to change the project is the upstream maintainer's call when
a test exercises the order on purpose. Stale `hipFree` and `hipHostFree` fail cleanly
with `hipErrorInvalidValue` in a single-threaded process (in a multithreaded one, see the
next entry); stale `hipEventDestroy` returns `hipSuccess` and was not
checked under valgrind. Tell it apart from a missing DLL (which dies before `main`
prints anything) by the work having completed first. A 12-line repro and a parity
request to HIP are on FLAMEGPU2.

### A duplicate `hipFree` in one thread can crash a different thread

When a library frees the same device pointer twice and another thread allocates between
the two frees, the second free can release that thread's buffer. The failure then shows
up somewhere unrelated, and only in some runs: for example an OpenCV `cv::cuda` call
aborting with `(-217) invalid argument` from `cudaGetLastError()`, or a jump to address 0.
plvs hit this once its libelas GPU path was enabled. The upstream code freed two buffers
twice, and on gfx1100 4-5 of 13 dense-stereo SLAM runs died that way, against 0 of 26
runs of three controls (the double free removed, the CPU fallback, no GPU code). On
gfx90a the same head never showed it, so a clean run on one architecture does not clear
the code. Read the code for duplicate frees when a crash kind appears only where a newly
enabled GPU path runs. Separate it from upstream races with the counting method in
diagnosis.md (CPU-only control, many runs, one stack per crash kind). Without core dumps (`ulimit -c 0`), and when gdb slows the run too
much to finish, a 15-line `LD_PRELOAD` handler that calls `backtrace_symbols_fd` on
SIGSEGV/SIGABRT still gives the stack. (plvs)

## Copies and stream ordering

**`cudaMemcpy*Async` from a soon-freed pageable host buffer.** CUDA stages pageable async
copies synchronously, so a test that copies from a local `std::vector` and lets it go out of
scope still works. ROCm's is genuinely asynchronous, so the freed and reused buffer is read
after the copy is enqueued. The corruption appears only in bytes the kernel does not
overwrite, and the failing set varies run to run. This is a use-after-free that is latent UB
on CUDA, not an allocator defect -- confirm by keeping the source alive and watching the
failures vanish, or without touching the source at all by re-running under
`AMD_SERIALIZE_COPY=3`, which makes a copy-lifetime bug pass and leaves a genuine numeric
residual untouched. That one switch is also the cheapest way to tell the two apart when a
suite mixes them: run it per failure cluster before admitting anything into a documented
residual family. Expect more than one instance per project: the idiom appears wherever a
test builds a staging buffer per loop iteration, and the giveaway is that only the side fed
from the short-lived buffer is wrong, often carrying values the computation could not produce
at all. (CV-CUDA: InterpolationVarShapeWrap, and independently TestOpColorTwist's planar
upload, whose per-iteration deinterleave result feeds one async copy per channel.)

**The same staging difference hides a cross-stream race with no free involved.** An upload
by `cudaMemcpy*Async` from pageable memory on one stream, then a consumer kernel on ANOTHER
stream with no event or stream sync between them. On CUDA the copy has effectively landed
before the launch; on ROCm the first kernel batch can read a partly written array.
Fingerprint: an intermittent wrong result confined to the first launch's outputs, while a
sibling code path that ends its setup with a device sync stays bitwise stable. Detect it by
running the same call 20 times and comparing bitwise with the first; one
`hipStreamSynchronize` on the upload stream makes it stable. (TIGRE: parallel-beam Siddon
`CreateTextureParallel`, wrong pixels up to 22% of max in projections 0 to 8, 2 to 7 runs
in 20 on gfx90a.)

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
`neural/backends/cuda/layers.cc` corrupted the value head's last-uploaded bias; the same
4.4e-02 value error appeared on three RDNA parts under Windows and on no Linux part, and was
misread as a rocBLAS problem for two months. A first fix that synchronized that one helper
left the moves-left head corrupt on nets that have one. Upstream's own fix, lc0 #2454, is the
reference shape: one `cudaDeviceSynchronize()` after the `tensor_mem_` malloc/memset loop.)

**A zero-filling allocator helper races a non-blocking upload the same way.** The variant
is easy to introduce while fixing the "fresh allocations are not zeroed" class. A helper that zero-fills with
`hipMemset`/`cudaMemset` runs the fill on the null stream, so a caller that allocates and then
uploads with `hipMemcpyAsync` on a NON-BLOCKING stream can have the fill land AFTER its upload
and wipe it. Fingerprint: inputs that arrive all-zero at the kernel while their device
pointers are valid, only for callers on a non-blocking stream, and the failure disappears
under `AMD_SERIALIZE_COPY=3 AMD_SERIALIZE_KERNEL=3` (which is the confirmation, not a
workaround). An operator that checks for degenerate input then reports its own sentinel
(NaNs, an identity model), so the symptom points at the math and not at the allocator. If a
helper promises memory is zero by the time it returns, make that true: synchronize the null
stream after the fill. (CV-CUDA: DefaultAllocator zero-fill against FindHomography varshape.)

**A plain device-to-device copy is the same race in a less suspicious costume.**
`hipMemcpyDtoD`/`cuMemcpyDtoD` (or `hipMemcpy` with `hipMemcpyDeviceToDevice`) looks
synchronous, but a device-to-device copy is asynchronous with
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

## Host callbacks and fork

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

**A host-function callback must not block either.** The runtime runs
`hipLaunchHostFunc` callbacks on its asynchronous-events thread, so a callback that waits on
a host-side gate (a condition variable, a promise another stream fulfils) stalls every later
event on that device, not just its own stream. CUDA runs such a test to completion. Keep
callbacks to a flag store or a notify, and do the waiting on a user thread.
(LichtFeld-Studio: a multi-stream test that parks a callback on a host-side gate hangs on
gfx1100.)

**HIP after `fork()` crashes in the child.** A process that has initialized the HIP runtime
and then forks without exec cannot use the GPU in the child: the child segfaults, typically
in `hipEventQuery`. CUDA documents the same restriction but more code gets away with it. A
test that forks and then touches the GPU fails on ROCm; so does a CPU-only test body run by a
fork-based harness after an earlier test started the runtime (gtest death tests in their
default style: diagnosis.md, "Failures that appear only when the whole suite runs in one
process"). Re-exec instead of forking, or fork before the first HIP call.
(LichtFeld-Studio: an autosave test forks a child that then uses the GPU, and the child dies
with SIGSEGV where the test expects its own SIGKILL; a CPU-only gtest death test fails the
same way in the default fork style and passes with `--gtest_death_test_style=threadsafe`.)

## Graph capture

**Creating a hipBLASLt or MIOpen handle while ANY thread captures a graph exits the process.**
`hipblasLtCreate` zero-fills its buffers with a synchronous `hipMemset` on the null stream. That
is refused with `hipErrorStreamCaptureImplicit` (906, "operation would make the legacy stream
depend on a capturing blocking stream") while any blocking stream on the device is capturing:
in another thread, in thread-local capture mode, and even when the capturing stream was created
non-blocking. The process then exits with status 1 inside the call, so there is no status to
check. `miopenCreate`
creates a hipBLASLt handle and fails the same way, and so do `hipblasCreate` and
`hipsolverDnCreate` whenever rocBLAS uses hipBLASLt (`ROCBLAS_USE_HIPBLASLT=1`, or by default on
GPUs where rocBLAS routes GEMMs to it): rocBLAS creates its hipBLASLt handle eagerly at handle
creation, not at the first GEMM. Test with that variable set even where it is off by default, or
the rocBLAS and hipSOLVER creation sites look safe. `hipblasLtDestroy` and `hipFree` were fine
(ROCm 10.1, hipBLASLt 1.4.1, MIOpen 3.6.1). cuBLASLt
and cuDNN never touch the null stream at creation, so a CUDA framework that creates handles
lazily (a new eager stream, an op's kernel state) while a graph runner captures on another
thread breaks only on HIP. Switching the creating thread to relaxed capture mode does not help.
Tracked as ROCm/rocm-libraries#11838; do not file it again. Pick the workaround by who creates
the handle. If it is the capturing thread or a thread that issues work into the same capture
(PyTorch's autograd worker), create the handles before capture begins and stock a shared pool
those threads draw from; PyTorch does this (`ensureCublasLtHandlesAvailable`, called from
`CUDAGraph::capture_begin`, pytorch #196399) and refuses creation under capture with a clear
error. If unrelated threads create handles while another thread captures (an eager worker next
to a graph runner in thread-local mode), nothing at capture begin can know how many handles they
will need, so serialize instead: a process-wide shared mutex that each capture holds shared from
begin to end, while every creation of a hipBLASLt, MIOpen, rocBLAS or hipSOLVER handle holds it
exclusively. Skip the
wait on a thread that is itself capturing, or it deadlocks. The mutex is deadlock-free only while
no thread blocks on another thread inside its capture window; check that for every capture site. To find who creates the handle, break on `hipMemset` in gdb
and read the backtrace: a log ordered by time alone suggests a lazy allocation inside the
captured call, but here the call came from another thread. (oneflow, 021f4ad34: an eager fused
MLP backward run right after an nn.Graph training step under `ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1`;
and, with `ROCBLAS_USE_HIPBLASLT=1`, the first eager device-to-host copy after one, which
constructs a stream with its rocBLAS and hipSOLVER handles.)

**hipFFT plan creation fails the same way, but quietly.** `hipfftMakePlan*` /
`hipfftXtMakePlanMany` (not `hipfftCreate`) runs null-stream work too. Under another thread's
capture it does not exit: it returns `HIPFFT_PARSE_ERROR` (12), which reads like bad arguments,
and the capturing thread's `hipStreamEndCapture` then fails with 901 (or its next library call
fails). It does this with auto-allocation off and when an identical plan already exists. Put plan
creation under the same exclusive lock as handle creation, which matters most where the framework
makes a new plan on every FFT call. To reproduce a race like this in a test, issue the eager ops
continuously from a second Python thread while the main thread runs several NEW graphs (each
captures again on its first step). A loop of eager ops placed after the graph call mostly runs
after the capture ends on an idle machine, and passes on the unfixed build. (oneflow: 8 graphs
with an FFT thread failed 6 of 6 runs on gfx942 before the fix, and 200 FFTs after each graph call
passed 2 of 2.)

## Managed memory

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

## Multi-GPU

**Cross-GPU flags in ordinary device memory are never seen by the peer on MI250X.** A barrier
where each GPU writes a flag into its own buffer and spins on the peer's buffer through an IPC
mapping (oneflow's P2P embedding shuffle) works on NVIDIA, where the owner's L2 is the point of
coherence for peer accesses. With `hipMalloc` (coarse-grained) memory the write stays in the
writer's L2 and the peer's read over xGMI goes to memory, so both GPUs spin forever -- even with
`__hip_atomic_store`/`__hip_atomic_load` at `__HIP_MEMORY_SCOPE_SYSTEM`. A two-process
reproducer hangs on the first barrier with `hipMalloc` and runs 20000 barriers with
`hipExtMallocWithFlags(..., hipDeviceMallocUncached)` or `hipDeviceMallocFinegrained`; IPC
handles work on both. Allocate any buffer that another GPU polls as uncached or fine-grained.
(oneflow)

**The DATA a peer reads, not only the flags, needs uncached memory -- and on gfx1100 fine-grained
is not enough.** Fixing the flags alone left oneflow's P2P shuffle reading stale rows on two
Radeon W7800 (gfx1100, PCIe): first step right, later steps wrong, run-to-run varying. A
two-process reproducer (producer kernel writes iteration-dependent data, the system-scope
barrier, peer kernel reads it over IPC) reads stale data in every element of every iteration
for `hipMalloc` AND for `hipDeviceMallocFinegrained` buffers that fit in L2, and correctly for
`hipDeviceMallocUncached`. On gfx11 neither the end of the producing kernel nor a
`__ATOMIC_RELEASE`/`__HIP_MEMORY_SCOPE_SYSTEM` store writes the producer's L2 back (on gfx90a and
gfx942 the system release does, which is why the same code passed there). A
`hipEventRecord` of a `hipEventReleaseToSystem` event before the barrier fixes eager runs but is
lost under hipGraph capture/replay. Robust fix: the kernel owns an uncached staging buffer,
shares that over IPC, and copies or gathers the data into it before the barrier. Two traps for
the regression test: buffers larger than L2 get evicted and pass anyway (oneflow's test needed
a small batch to fail), and checking only the last iteration of a loop hides it. (oneflow)

**Streams share hardware queues on HIP, so a spinning kernel can block another stream's work and
deadlock two GPUs.** CUDA streams make progress independently. HIP runs ordinary streams on a pool
of `GPU_MAX_HW_QUEUES` (default 4) hardware queues per device per priority, and with the pool full
ROCclr picks the least loaded queue per submission (`AMD_LOG_LEVEL=3` shows `Selected queue
(mode=1): ... refCount: 2`). A packet that must wait for earlier work in its own stream then waits
for everything ahead of it in the shared queue, including a spinning kernel from another stream. A
cross-GPU spin barrier (oneflow's P2P one_embedding shuffles) issued on one stream and an NCCL/RCCL
collective issued on another, in no fixed order between ranks, deadlocks when the two GPUs queue
them in opposite orders: random step, 5-30 percent of runs, both GPUs at 100 percent. Diagnose
without a debugger by giving each spin loop a `wall_clock64()` timeout that `printf`s rank, phase,
target and seen value (device printf flushes while the kernel still spins): oneflow showed one rank
in the gradient barrier and the peer in no barrier. `GPU_MAX_HW_QUEUES=32` making the hang go away
confirms it (oneflow: 6/20 hung at the default, 0/20 at 32). Fix: create the project's streams with
`hipExtStreamCreateWithCUMask` over all CUs; ROCclr gives a CU-masked stream a hardware queue of
its own outside the pool (a probe deadlocking 6/6 stream pairs at `GPU_MAX_HW_QUEUES=1` passes 30/30
with CU-masked streams). Each priority (low, normal, high) has a separate pool of the same size,
capped by the same `GPU_MAX_HW_QUEUES` (`Device::acquireQueue` in `rocdevice.cpp`), so streams from
`hipStreamCreateWithPriority` deadlock the same way. The mask counts `multiProcessorCount` units: on
RDNA in WGP mode that count and the mask bits are WGPs, and ROCclr expands each bit to the WGP's two
CUs, so `multiProcessorCount` set bits cover every CU on CDNA and RDNA alike. CU-masked queues never
enter the pool; `hipStreamDestroy` destroys the queue (`Device::releaseQueue`), and each costs
device memory for its ring buffer (4 MiB on gfx942), so create them per long-lived stream, not per
operation. Every stream that can carry a cross-GPU spinner or a collective needs this, including
streams the project creates outside its stream wrapper, or an argument that no spinner can share
their pool: oneflow leaves its collective-boxing executor and fused-MLP gradient streams on the
pool, safe only because every spin kernel now runs on a CU-masked stream. A spinning kernel alone
does not block a later kernel of another stream on the same queue; the later kernel has to wait on
prior work in its own stream first. (oneflow)
