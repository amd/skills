# Diagnosing a suspected AMD fault

How to tell a port fault from an environment fault, a test artifact or a ROCm defect before escalating.

<!-- toc:begin -->
Contents:

- Start here
- Method
- Hangs and crashes
- Memory measurements
- Multi-GPU hosts
<!-- toc:end -->

## Start here

### One architecture gets wrong numbers while the others pass

A clean build that produces wrong results on exactly one architecture -- an iterative
solver, an LM/Newton fit, an FP regression head -- has two very different causes: a real
bug that only that architecture's timing or width exposes, or floating-point
accumulation divergence. The checks for a real bug are cheap, so they come first:

- An unsynchronized upload: a `cudaMemcpy*Async` on one stream consumed by a kernel on
  another, or an init-time kernel on the null stream racing other streams (both in
  faults-memory-streams.md under "Copies and stream ordering"). Run the same call 20 times and compare
  bitwise with the first; add one stream synchronize after the last upload and see
  whether the error disappears.
- A device-to-device copy racing its producer, and an allocation the code reads before
  writing (CUDA often hands back zeros; HIP does not).
- A wavefront-width assumption: compare against a second architecture of the same width.
- Single against double precision: if every GPU variant diverges from the
  double-precision result identically at the same positions, and the double path matches
  the CPU, it is reassociation at a comparison boundary (see "Triangulate single- against
  double-precision" under "Method" below).

Only when those are clean is accumulation-order divergence the likely answer; RDNA3.5
(gfx1151) is where it has shown up in iterative solvers. Then record the error magnitude
beside the other architectures' and stop: the comparison that matters is against them,
not against a fix. Stopping at the first step is how one value-head error stayed
recorded as a BLAS-library problem for two months before it turned out to be a
null-stream upload race that was latent on CUDA too (lc0).

Stopping at a recorded magnitude is for small, deterministic deviations. An error that is large
(orders of magnitude) or changes from run to run on one wave size is a defect until shown otherwise:
gpu4pyscf's analytic gradients failed only on gfx90a/gfx942 for months and turned out to be a
launch-geometry mismatch (platform-limits.md, "A host-side launch heuristic keyed on the register
count"). Reduce it to the smallest input that still fails (H2/STO-3G reached a single kernel) before
calling it numerical. (gpu4pyscf)

### Rule out a stale tree and the toolchain version before accepting "arch-specific"

GooFit's HIP backend was reported to drive an unbinned maximum-likelihood fit to a
parameter's upper bound on gfx90a (`alpha` at the +10 bound instead of the true -1,
`FunctionMinimum is invalid: Edm is above max`, 21/25 ctest failures) while the identical
commit passed on gfx1100 (wave32) and on the CPU/OpenMP backend, both giving
`alpha = -1.001102381`. It was recorded as wave64-specific, and that was wrong: the same
commit on the same card under a later ROCm gives the correct answer. Two cheaper
explanations come before "genuinely arch-specific", and both have produced a false report on
that one project:

1. **A stale or uncommitted tree.** Reproduce from a *fresh clone* of the exact commit, not
   a working tree that may carry local edits from earlier debugging. A clean
   `git status --porcelain` before the build is necessary but not sufficient.
2. **Toolchain or driver version.** Compare `hipcc --version` against the version the
   passing architecture ran. A divergence that appears only on a newer or older ROCm than
   the passing run's is a toolchain difference to report, not an arch fault. Matching the
   version of an earlier FAILING run does not rule this out: it only shows the failure
   reproduces on that version, which is the wrong direction. The discriminating run is the
   same arch on a DIFFERENT, current ROCm, because where there is one machine per
   architecture the arch and its ROCm install are confounded: whatever is installed on the
   only gfx90a box is a property of that box, not of gfx90a. Do that run before writing any
   code. GooFit's divergence was accepted as gfx90a-specific on exactly the wrong-direction
   reasoning (the failure reproduced again on the ROCm 7.2.1 series an earlier attempt had
   used, which said nothing about the architecture); a later session on the same
   hardware with ROCm 7.14 built the same commit from a fresh clone and got 25/25 with the
   fitted parameter digit-identical to every passing arch, after someone had gone hunting
   for a lifetime bug in code that had none.

What the failing session did establish -- a fresh clone, and kernels demonstrably
dispatching (`AMD_LOG_LEVEL=3` shows 112 `hipLaunchKernel` calls, `roc-obj-ls` shows one
gfx90a code object) -- is only that real GPU execution produced the wrong answer, i.e. that
it was not a stale tree or an "it never ran on the GPU" artifact. Use those checks to
eliminate the build and the environment, then change the ROCm. Perturbing the allocator and
serialising the runtime (`HSA_DISABLE_FRAGMENT_ALLOCATOR=1`,
`AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3`) are the cheap follow-ups on the passing
version that argue against latent undefined behaviour a newer runtime merely hides; they
are evidence, not proof, so say so in the record. A divergence that survives a current,
different ROCm on the same card is the one the entry above is about. (GooFit)

The same class also shows up as a threshold-boundary detection COUNT, not only as a fit
residual. PopSift's cross-architecture check is exact feature and descriptor counts on six
reference images (the Oxford boat set). On gfx90a, the only wave64 architecture in the
comparison, one image came back one feature and one descriptor short (7945/9451 against the
7946/9452 that gfx1100, gfx1101, gfx1151 and gfx1201 all reported at the same commit), fully
deterministic across 5 repeat runs, with the other five images exact. A single detection
sitting right at a threshold flips between one wavefront width's warp reduction and the
other's without being a correctness bug in either. Treat a single-digit count divergence on
one input out of many, with everything else exact and deterministic, the same way: record it
and stop. (popsift)

## Method

**A "data-dependent, later-data-corrupts-earlier, per-tile" corruption signature is the fingerprint of a REPRODUCER bug, not a codegen fault.** cuSZ chased a suspected miscompile to a BLOCKED state and an IR bisect; the actual cause was the test input -- `np.arange(..., dtype=float32) * (python float)` promotes to float64, so `.tofile()` wrote 8 bytes per element and the tool read the stream as f32. Validate the byte width and dtype of any binary test input before escalating to an ISA bisect or a ROCm bug report. (cuSZ)

**Triangulate single- against double-precision before blaming the wavefront.** When a warp-collective rewrite shows SP divergence, run a second GPU variant and compare both to the DP oracle. If both GPU variants diverge from DP identically at the same positions, and the DP path is bit-identical to the CPU oracle, it is floating-point reassociation at a comparison boundary -- not a wave-size fault. SCAMP used this to clear a ~0.5 divergence at 10/8093 positions as a threshold-boundary artifact. (SCAMP)

**Test "it reads uninitialized device memory" with a fill-pattern differential, not an audit.** The hypothesis is common for an unattributed hang, crash or wrong result in a project that over-allocates worst-case buffers (a collision-pair or candidate-list buffer sized for the theoretical maximum and governed by a live counter is the classic shape), because ROCm does not hand back zeroed memory the way CUDA often does (faults-memory-streams.md, "Fresh device allocations are NOT zero on ROCm"). Auditing every `hipMalloc` for a matching `hipMemset` is slow, easy to get wrong, and proves nothing about whether an unaudited site matters at run time. Test it instead: temporarily wrap the allocator so every fresh allocation gets a controlled fill, behind a compile define so the unmodified build stays a pure passthrough (`#if defined(DEBUG_MALLOC_FILL)`: a template `debug_hipMalloc(T** ptr, size_t size)` that calls `hipMalloc` and, on `hipSuccess`, `hipMemset(*ptr, DEBUG_MALLOC_FILL, size)`, plus `#define cudaMalloc(ptr, size) debug_hipMalloc(ptr, size)`), and run the SAME source three ways: unmodified, zero-fill and 0xFF-fill. 0xFF is the decisive pattern, not 0x00: as an IEEE-754 float or double every all-0xFF value is NaN, and as an index it is -1 or UINT_MAX. A genuine live-uninitialized read on the hot path then shows as an immediate NaN cascade, a crash on an out-of-range index, or a wildly different run length, none of which a healthy program produces by chance. On GPU_IPC (gfx1100) the unmodified, zero-fill and 0xFF-fill builds all stalled in the same narrow frame window (31-37, inside the pre-existing 25-36 range) with zero NaN and zero HIP errors across hundreds of recorded lines each, and every reported index in all three was small and in range: that is what a buffer correctly gated behind a live counter looks like under adversarial fill, and it rules the hypothesis OUT instead of leaving it unaudited. Revert the wrapper afterwards; it is a throwaway diagnostic, not something to land. (GPU_IPC)

**To close an "it might silently fall back" risk, trace it; then bound it from the code, not from the trace.** A dispatch count (validation.md, "A green result does not prove a GPU test ran") proves which backend ran on the run you measured. When the worry is that some ORDER of events could flip a global backend flag ("if the fallback initializes first, the compute flag is cleared and the suite still goes green"), a Release build with no debug info is enough for a trace: statics like `Foo::_UseBackend` and ordinary member functions are in the symbol table, so gdb can breakpoint the decision function (`break *0x<addr>` with `commands` / `silent` / `printf "decide(arg=%d) flag=%d\n", (int)$rdi, *(int*)0x<addr of flag>` / `continue` / `end`), breakpoint each backend's constructor, and `watch *(int*)0x<addr of flag>`. The counts answer outright: how often the decision function was entered and with what argument, how often each backend was constructed, and every value the flag took. In colmap that turned "did not materialise on the machine we happened to use" into "the clearing line did not run once in this binary, and the flag was written exactly once, `0 -> 1`". Then be careful what you claim, because this is where a trace gets oversold: it covers the paths the binary took with the options it ran with, so it is evidence for "not reached in this configuration" and never for "unreachable". colmap's first write-up made that jump ("the clearing line is unreachable, because the only caller always requests the compute backend") and review disproved it with one probe: a public option (`darkness_adaptivity`, with the default negative GPU index) makes the caller omit the flag that selects the compute backend, and the supposedly dead line runs. Upgrading "not reached here" to "cannot happen" takes an argument from the CODE; the one that held was that the API re-asserts the flag on every freshly constructed object that requests the compute backend, so no earlier user can leave it cleared for a later one. When such an argument holds, ask what ELSE the failing path wrote: the initialization that clears flag A usually sets siblings in the same function. In colmap the failed GLSL init also zeroes a second global meaning "this process has no usable GL"; it is sticky (the init returns immediately once it is 0 and never retries), and every later extractor, the compute ones included, early-returns on it and is constructed as a null: a hard failure with nothing extracted, not the silent degrade the risk was written about, and still a live defect. Enumerate every write on the failure path, and for each ask who resets it and who reads it later; sticky-on-failure, process-wide and consulted by a user that would otherwise have worked is the shape that bites. Two traps each produce a confidently wrong answer. Reading a global AFTER the inferior exits gives the ELF initial value, not the last live value: gdb falls back to the executable's `.data`/`.bss` image and prints it without complaint (in colmap, exactly the cleared-flag state the reviewer had predicted, from a run where the flag was never cleared), so read the flag at a breakpoint inside the live process, or watch it. And `LD_PRELOAD` on a GL or driver entry point does not intercept a Qt (or any `GetProcAddress`-style) caller: Qt resolves GL through `QOpenGLFunctions`/`eglGetProcAddress`, so the preloaded symbol is bypassed while a directly-linked caller in the same process IS intercepted; prove a preload shim fires with a positive control first. (colmap)

**`torch.cuda.device_count()` returns 0 while `torch.cuda.is_available()` is True: a
mismatched `amdsmi`.** `torch._C._cuda_getDeviceCount()` says 1, `torch.cuda.device_count()` 0,
then `AssertionError: Invalid device id` on the first device query, or device-count checks in
tests skip while kernels run. On ROCm torch counts devices through `amdsmi` when it can import
it, and that count wins over HIP's; an `amdsmi` Python binding from a different ROCm release
than the one torch loaded returns no processor handles in torch's process. Make the importable
`amdsmi` the one from the same ROCm (with TheRock's packages, `$(rocm-sdk path
--root)/share/amd_smi` of the matching SDK, put first through a `.pth` file), or remove the
stray one from the path. Seen with a ROCm 10.1 `amdsmi` beside a 7.12 torch, and with newer
TheRock wheels installed into an environment that inherited an older SDK's `amdsmi`. (wenet,
ColossalAI)

**A once-per-suite host crash that stress reruns never reproduce: widen the suspected race window instead of rerunning.** If the stack sits in runtime code the port does not touch (`git diff` empty there), read it for a publish-before-register ordering, insert a temporary sleep between the two steps, rebuild, and run a tiny loop. A crash that becomes deterministic, and also happens with CPU-only work, is an upstream race that a fast GPU merely exposes; revert, rebuild and record it. oneflow's `LaunchLazyJob` pushed a graph's SourceTick before `EnqueueNNGraph`, so a tiny graph's finish callback could pop an empty queue: 1 crash in ~550 suite files on two arches, 0 in 70 stress reruns, 5 of 5 with a 20 ms sleep (CPU graphs included). (oneflow)

**ROCm 10.1 `AMD_LOG_MASK` takes decimal only, and graph replays are invisible to it.** A hex value such as `0x60000` is silently read as nothing (level 4 then prints one banner line), so check a mask on a tiny HIP program before trusting a quiet log. Bits on that runtime: 1 = API calls, 131072 = memory, 2097152 = `ShaderName` per dispatch. A graph's kernel nodes log `ShaderName` once, at `hipGraphInstantiate`, never on replay, and an illegal access logs `Memory Fault Error` with no address at any level (and `dmesg` is usually not readable in a container), so neither logging nor `AMD_SERIALIZE_KERNEL=3` names the kernel that faults inside a graph replay. Varying the replay order changes how often such a fault shows (oneflow: 0 of 12 to 7 of 8 at one commit), so an apparent regression between two commits can be the order hiding it.

**Before blaming the code under test for a wrong result, save the output, the inputs as read back and the reference, and recompute the reference on the CPU.** A test that checks a GPU result against an in-process GPU reference (torch, or the project's own float64 path) fails the same way when the REFERENCE is the wrong side. oneflow's "corrupt first-graph replay" on two gfx1100 GPUs survived two rounds of graph-capture hypotheses; dumping the arrays showed the graph output matching a CPU product of the read-back inputs to 9e-7 while the float64 reference GEMM on the other GPU came back as garbage, just before the illegal access. Also turn the suspected feature off first: the same fault with HIP graph capture disabled (`ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=0`, 2 of 2) cleared graphs in two runs. (oneflow)

**When a wrong result survives a `hipDeviceSynchronize` on every device, probe the HIP copy path itself before suspecting kernels.** Between the steps of the failing script, copy a known pattern host-to-device and back with plain `hipMemcpy` (ctypes is enough) into a host buffer pre-filled with -1, and report the wrong byte ranges. In oneflow's two-GPU nn.Graph run on gfx1100 (ROCm 10.1 SDK), such a probe on device 0 came back with whole 48 KiB pieces never delivered (still -1), with rc 0, after the larger graphs ran on device 1. That probe and torch's reference uploads run no oneflow code at all, but the state that makes them fail can still come from the application's two-device activity, so a clean probe on one device does not clear the application. **Do not attribute a timing-dependent fault from one single-variable condition on one build.** `HSA_ENABLE_SDMA=0` (copies by blit kernels instead of the SDMA engines) first looked decisive there (0 of 10 against 23 of 23 at the default), yet interleaved runs on two builds of the same source, differing only by inert env-gated debug hooks, showed it moving the rate in either direction: 8 of 8 to 0 of 8 on one build, 0 of 8 to 3 of 8 on the other, with the illegal access itself in a blit-kernel copy. The two builds also gave 8 of 8 against 0 of 8 at the default, and the clr queue toggles (`DEBUG_CLR_AQL_BARRIER_OPT`, `DEBUG_CLR_AQL_DEV_QUEUE`, `DEBUG_CLR_ENABLE_KDQ`, `DEBUG_CLR_DOORBELL_SKIP`, `DEBUG_CLR_USE_MOVDIR64B`, 3 of 3 each) and `DEBUG_HIP_DYNAMIC_QUEUES=0` (6 of 8) did not remove it. Before a condition attributes anything, interleave it with its control, at least 8 runs per cell, on more than one build. A newer runtime under the same build is the second cheap test: preloading a later nightly's `libamdhip64.so.7` and `libhsa-runtime64.so.1` (`LD_PRELOAD`; ABI-compatible within the same soname) ran 0 of 13 there. That is consistent with a runtime defect but does not prove one, because a runtime swap also changes timing and an application race (a missing cross-device dependency, a buffer freed while another device still reads it) would react the same way; it takes a reproducer without the application, or the one runtime commit that makes the difference, to place the bug in the runtime. `AMD_LOG_LEVEL=3 AMD_LOG_MASK=256` logs the per-stream SDMA engine assignment (one INFO line per assignment, little overhead). A probe on the legacy null stream returns 906 while another thread's stream is capturing, so place probes outside a capture or run with capture off. (oneflow)

**A kernel that hangs only in a Debug build: rebuild Debug at `-O2` before blaming an assert or the port.** A Debug HIP build turns two things on at once, `-O0` device codegen and every `assert` (including HIP's own `__hip_check_mask` in the `*_sync` intrinsics). Configure a second Debug tree with `-DCMAKE_HIP_FLAGS_DEBUG="-O2 -g0"` (NDEBUG still unset). If the test passes there, the asserts are not what fires and the hang is `-O0` codegen; find the kernel with `AMD_SERIALIZE_KERNEL=3 AMD_LOG_LEVEL=3` (last launch that never completes) and the PC with the SDK's `rocgdb` after a SIGINT. Run a Debug suite one test per process with a timeout so one hang does not hide the rest. The asserts-on `-O2` tree is also the better Debug evidence for mask and wave64 checks. (LichtFeld-Studio)

**An intermittent crash seen only in the GPU build needs a CPU-only build control, run as often and under the same load.** A multi-threaded application (SLAM, pipelines with worker threads) can carry an upstream race that the GPU path only changes the timing of. plvs stereo segfaulted in 4 of 9 HIP runs and in 0 of 6 CPU-only runs, which looks like a port fault, but the crash was in CPU line-mapping code (a null `KeyFrame*` in `MapLine::UpdateNormalAndDepth`). Six more of each, run side by side, gave HIP 0/6 and CPU 1/6. Build the control from the same tree with the GPU option off, into a copy of the source tree when the project writes its outputs into the source tree, get a backtrace (`gdb -batch -ex run -ex bt`), and count runs before you decide. A handful of runs per side cannot tell a rate of 25% from a rate of 8%. (plvs)

**A probe that contradicts a passing test, or that newly fails on a second arch, is wrong until proven otherwise: suspect the probe's own inputs first.** Two cases from one project. (1) A bisection concluded that HEonGPU's inverse NTT miscomputed on gfx90a while the encode/decode tests, which round-trip through exactly that inverse NTT, kept passing. The contradiction was written down and not reconciled, and the next person to pick up the work inherited the wrong suspect. Reconcile first: either the passing test does not use the path you think it does, or the probe set up its inputs differently from the library (mismatched tables, wrong API overload, wrong ordering convention). Write the probe against the library's OWN generated parameters and public entry points, never against parameters you regenerate by hand. (2) The Barrett-shift fix (61/62-bit moduli) verified clean on gfx90a; the identical probe strategy on gfx1100 showed 19998/20000 bad at those widths and looked like a wave32 regression the first arch had missed. The probe generated its test moduli as `2^bits - small_offset`, pathologically close to an exact power of two; the library's HOST-side `bit_generator()` computes `bit = (T1)(log2(value) + 1)` in plain `double`, and at 61-62 bits a value within roughly 1024 of a power of two rounds `log2` up to the boundary in double precision, over-counting the bit width by one and mis-sizing the unrelated Barrett `mu` constant -- identical behavior on any host, any back end, not a HIP or wavefront effect. Redrawing the probe's moduli uniformly across each target bit width (not clustered at the boundary) reproduced 0/20000 bad on both arches. Before concluding a fix regressed on a new arch, check whether the SECOND probe run's inputs, not just its outputs, differ from the first. (HEonGPU)

**A cascading wall of SegFault/OOM/"no kernel image available" partway through a long serial suite can be one poisoned physical card, not a port defect -- check `rocm-smi --showpids` against `--showmeminfo vram` before concluding anything.** arrayfire on a host with eight MI300X cards: a full `ctest` run started clean, crashed (SegFault) on an early binary, and every binary after it degraded -- more SegFaults, a "no kernel image is available for execution on the device" (hipErrorNoBinaryForGpu) on a JIT op, then explicit "Device out of memory". `rocm-smi --showmeminfo vram` showed that GPU pinned near its full capacity; `rocm-smi --showpids` showed *no process attached to it at all*, including after `kill -9` of every test binary -- the VRAM was orphaned at the KFD/driver level, not held by a live allocator, and a process crash should not leak GPU memory past its own exit. Re-running the identical suite (fresh JIT disk cache) on a different, verified-idle card on the same host, same build, same commit, came back 131/132 clean with no crash, no OOM, and no "no kernel image" anywhere. The disagreement between the two `rocm-smi` views is the tell: if `--showmeminfo` says a card is nearly full but `--showpids` shows nothing on it, that card's state cannot be trusted for this session -- switch cards to disambiguate before spending time on the port. (arrayfire)

**A "fixed seed" run that relies on C `rand()` is not reproducible on some ROCm builds: the HIP runtime's compiler library reseeds it.** Projects often get reproducible runs in a debug build by simply not calling `srand()` and seeding device RNGs from `rand()`. Some ROCm builds break that. On the first kernel launch the HIP runtime compiles its internal copy kernels through comgr, and each comgr action makes a temp directory via LLVM `sys::fs::createUniquePath` -> `sys::Process::GetRandomNumber`, which calls `srand(<random seed>)` and then `rand()` (38 times on HIP 7.16) whenever LLVM was built without `arc4random` (glibc < 2.36 at LLVM build time, as in TheRock's manylinux SDK). It is not the application's code objects loading: `rand()` after `hipMalloc` is still the default stream. Any `srand()` the program made before its first launch is overridden too. Two runs of the same input then draw different random streams, and a self-comparison suite (same binary as baseline and candidate) fails exactly the stochastic cases. Check a host with `nm -D --undefined-only libLLVM*.so* libamd_comgr.so* | grep -w -e srand -e arc4random`: `arc4random` present is what decides, since a libLLVM can import both (Ubuntu's `libLLVM.so.20.1` does). Ubuntu 24.04's distro comgr is unaffected: it imports only `rand` itself, and the `libLLVM-17.so.1` it loads imports `arc4random`. To classify, rerun the failing cases with an `LD_PRELOAD` shim whose `srand()` is a no-op: if they pass, it is this, not the port. Before trusting a single-directory rerun, check that the outputs are not appended to (GPUMD's `dump_xyz` appends). (GPUMD)

**Seeded `cupy.random` inputs differ between CUDA and ROCm.** CuPy-ROCm draws from hipRAND/rocRAND, so `cp.random.seed(n)` gives different numbers than CUDA's cuRAND. A test that hard-codes a fingerprint of results computed from such inputs fails on ROCm even when the computation is right. Do not count it as a GPU failure or edit it away: rerun the same inputs through the CPU reference (or the test's own CPU cross-check with only the fingerprint assertion disabled, in a scratch copy) and record that comparison. `np.random`-seeded tests are unaffected. (gpu4pyscf)

## Hangs and crashes

### A hang that depends on inlining: long branches clobber the return address

**A kernel that never finishes, or a device function that returns to the wrong place, only at the default inlining level can be llvm/llvm-project#224205.** In a large device function that is not a kernel (the callee of `s_swappc_b64 s[30:31]`, for example a library reduction body with a big functor inlined into it), AMDGPU long-branch relaxation can use `s[30:31]`, the return address, as its scratch pair. Confirm it in the asm: compile the file with its real compile line plus `--cuda-device-only -S` for the target, and in each non-kernel function count `s_getpc_b64 s[30:31]` ... `s_setpc_b64 s[30:31]` long-branch sequences with no save or restore of `s30`/`s31` anywhere in that function. Any count above zero is the defect pattern. It appears on RDNA (gfx11, gfx12) targets in Linux builds as well as Windows ones, and it does not always hang, so a passing run does not clear a build that shows the pattern. Workaround: limit inlining for that one translation unit, through `COMPILE_OPTIONS` on the source, `/Ob1` for clang-cl and `-finline-hint-functions` for clang or hipcc on Linux. The function stays out of line, small enough that no long branch is left; recount in the asm to confirm zero. (h2o4gpu: the `FuncEvalF` reduction in `h2o4gpuglm.cu`, float and double; the double-precision fit hung on gfx1151 on Windows, and gfx1100/gfx1201 Linux builds carry the same pattern without an observed hang.)

Limiting inlining does not help when the out-of-line device function is large by itself: QUICK's `ERint_vertical_spd` (189 KB of VRR code) kept all 6 long branches through `s[30:31]` with `-finline-hint-functions`, at -O1, -O2, -O3 and -Os. A workaround that does not depend on inlining is `-mllvm -amdgpu-long-branch-factor=0` (defined in `GCNPreRALongBranchReg.cpp` since ROCm 6.0's LLVM): it turns off the pre-RA reservation, so branch relaxation takes a dead register pair from the scavenger; recount to confirm zero. CDNA is affected in real code, not only RDNA: gfx90a, gfx908 and gfx942 all emitted the pattern, and on gfx90a it hung the kernel. The same defect can also show as an aperture violation instead of a hang, depending on which block the clobbered return lands in; under rocgdb with precise memory (next entry), compare `s[30:31]` against the long-branch targets. And a "100x slower" report for a kernel that never returns from one launch is a hang: check with `AMD_SERIALIZE_KERNEL=3 AMD_LOG_LEVEL=3` whether a single launch finishes before measuring anything. (QUICK)

### Proving a real miscompile: catch the faulting instruction, then flip one backend pass

An `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` is reported imprecisely. Run the process under `rocgdb -batch -ex 'set amdgpu precise-memory on' -ex run -ex 'x/12i $pc-40' -ex 'p/x $v20'` (rocgdb ships in TheRock's `bin/`) to stop on the exact instruction and read its registers per lane. In gpu4pyscf the faulting op was `global_atomic_add v1, v20, v1, s[20:21]` (the persistent-kernel `atomicAdd(head, 1)`). Its zero offset register v20 held garbage in lane 0 only, and `llvm-objdump -d` of the extracted code object showed v20 set to 0 once and then reused as the destination of a predicated `global_load_dwordx2 v[20:21]` inside the loop. Rebuilding with `-mllvm -disable-machine-licm` in `CMAKE_HIP_FLAGS` made results match the CPU reference to 1e-13. `-O1` did not help. The defect was TheRock ROCm 10.1 (AMD clang 24.0.0git) MachineLICM, and the same pattern appears in gfx90a code from that compiler, so it is a toolchain fault, not an architecture fault. It does not reproduce on every target, though: the same compiler and committed flags ran correctly on gfx1100, so before adding the flag, test whether each arch shows the fault with one run of the failing case (a d-shell calculation there). Under `-fgpu-rdc` device codegen runs at link time, and CMake's HIP link rule reuses `CMAKE_HIP_FLAGS`, so that one setting reaches both; a copy in `CMAKE_SHARED_LINKER_FLAGS` can be silently overwritten by the project (gpu4pyscf's own `set(CMAKE_SHARED_LINKER_FLAGS ...)` did) and protects nothing, so confirm with a verbose build that every compile and link line carries the flag. Whether a port carries a compiler workaround is for the people who own the port to decide; once decided, put it under the HIP build only with a comment naming the compiler version and the arches affected, and the condition for removing it. (gpu4pyscf)

### Before removing a compiler workaround, rerun the whole suite on the new release

**A newer ROCm that contains the upstream fix for a miscompile does not prove the workaround can go.** Build without the flag on the new release and run the full GPU test selection, not only the reproducer or the case that first crashed. In gpu4pyscf, ROCm 10.2's compiler contained the LLVM fix for 10.1's MachineLICM register clobber (previous entry): the original aperture violation was gone and a static check of the kernel was clean, yet `-mllvm -disable-machine-licm` was still needed. 10.2 had a different miscompile in the same kernels, which gave wrong nuclear gradients with no error on gfx942 (only the gradient tests saw it) and hung the GPU in the same J/K gradient path on gfx90a. To tell compiler from runtime, run the new compiler's build on the old runtime and the old compiler's build on the new runtime, and rebuild with the workaround flag on the new compiler: if the flag alone restores the results, the fault is still in codegen. (gpu4pyscf)

### An instruction-selection abort: change how the value is produced

`LLVM ERROR: Cannot select: v4bf16 = scalar_to_vector` during the device LTO link for gfx942
(AMD clang 24.0.0git, ROCm 10.2.0a20261022) came from bf16 values produced through
`__float2bfloat16(__half2float(x))` and stored as bf16x2 pairs in a large kernel. Producing the
value without the rounding `__float2bfloat16` (the conversion was exact there, so the bits could
be built directly) avoided it. When a ROCm build aborts in instruction selection, find the
function named in the crash dump, change how that value is produced, and keep the translation
unit and command line for a report. (ColossalAI)

### A fault inside a vendored JIT entry point: rebuild-with-a-forced-throw beats a debugger

When a crash's stack bottoms out inside a vendored dependency's own runtime-JIT call (for
example `hiprtBuildTraceKernels`, which drives HIP RT's own hipRTC/comgr compile of its
BVH-builder and trace kernels) rather than in the port's device code, there is a cheaper
first test than a debugger or a minidump. Read the JIT call's error handling for a place
where a compile failure could be silently swallowed. The classic shape is a
`result != SUCCESS` check that only acts when the accompanying log string is non-empty; an
empty log on a genuine failure is common on Windows, where comgr can fail silently for
reasons unrelated to the compile. Patch that ONE spot locally to `throw` unconditionally with
the raw result code, rebuild just that translation unit (seconds, not the whole dependency),
and rerun. Both outcomes are informative:

- Your forced exception fires with a real code and message: you have found the actual
  failure.
- The crash is identical (same fault address relative to the containing function, for
  example `hiprtBuildTraceKernels() + 0x894` unchanged across rebuilds even though ASLR moves
  the DLL's base) and your `throw` never fires: you have FALSIFIED that code path as the
  cause. The crash is earlier, or in a different function, and no more time should go into
  that call.

A second hypothesis can be tested the same way, without a debugger: does the crash change
when a plausible environment variable is toggled? Revert the diagnostic edit afterwards (it
is a throwaway), and record the falsification, not just "still crashes": a rebuild that
proves a specific mechanism is NOT the cause narrows the search for whoever picks it up
next. If an earlier diagnosis of the same dependency's JIT or loader exists, verify it
against the CURRENT patched source before assuming it applies: a defect reported earlier in
the same call chain may already be fixed in the project's own patch to the vendored
dependency, which rules it out rather than explaining the new crash. Before any of this, read
the top frame: a null frame above the vendored DLL is a library that never loaded
(windows.md, "A null top frame above a vendored DLL"). (diff-surfel-tracing)

### scikit-learn's KMeans can crash a Python test on a machine with very many cores

A Python test that scores an embedding or a clustering with `sklearn.cluster.KMeans` can
exit with SIGSEGV or SIGABRT on a 224-thread machine after every assertion passed, about
half the time. The fault is in scipy's bundled OpenBLAS (`sgemm_incopy_*`) called from a
libgomp worker hundreds of threads deep, and it reproduces with KMeans alone, without the
port loaded (4 of 6 runs; 0 of 6 with `OMP_NUM_THREADS=16`). It is not the GPU and not the
port. Cap the threads around the call, `with threadpoolctl.threadpool_limits(8):`
(threadpoolctl is a scikit-learn dependency), rather than exporting an environment variable
the next person to run the test will not know to set. (tsne-cuda)

### Failures that appear only when the whole suite runs in one process

Two kinds show up on Linux and not on a host that runs one suite per process (as the
Windows runners do), and neither is a port defect:

- gtest death tests (`EXPECT_EXIT`, `EXPECT_DEATH`) under the default `fast` style
  `fork()` without exec. Once an earlier test has initialized the HIP runtime, the child
  can die with SIGSEGV before running the test body, even when the code under test is
  CPU-only; gtest's "Death tests use fork() ... detected N threads" warning is the
  sign. Rerun the test with `--gtest_death_test_style=threadsafe` (re-exec). If it
  passes, the failure was the harness.
- Tests that assume the state of a process-global allocator or cache: they pass alone
  and fail after some unrelated earlier test. Run the test alone, then bisect the
  suites that precede it with `--gtest_filter` to find the one that leaves the state
  behind, and check that the port does not touch either file before you record it.

### A sync-bound GPU test that passes alone can time out under `ctest -jN`

Processes that submit to one GPU at the same time slow each other down more than linearly, so a test that synchronizes every step can pass alone and hit its timeout in a parallel run with no hang and no port fault. On a Radeon gfx1201 (ROCm 10.1), a loop of small kernels plus a small synchronous device-to-host copy took 30 us per iteration in one process and 135-139 us per iteration in each of two or four concurrent processes; an idle second context cost nothing, only concurrent submission did. ESPResSo's `lb_pressure_tensor` (10000 sample steps, each synchronized) passed in 77-147 s alone and timed out at 1200 s under `ctest -L gpu -j 8`.

Before treating such a timeout as a hang, rerun the test alone, then run two copies at once. A pass alone plus a superlinear slowdown with two copies is GPU sharing. The fix is to run the GPU tests serially (`ctest -j1` for the GPU label, or a `RESOURCE_LOCK` test property shared by the GPU tests), not a longer timeout or a code change. A test that is stuck rather than slow under `ctest -jN` is a different case (linux-host-setup.md, "A GPU test that hangs only under `ctest -jN` is usually the software GL stack"), and so is one that is too slow even alone (platform-limits.md, "A low-CU integrated GPU can outrun a fixed upstream test timeout without any fault"). (ESPResSo)

## Memory measurements

### A free-memory leak check measures the whole GPU, not your process

`hipMemGetInfo` (like `cudaMemGetInfo`) reports free memory for the device, so a leak
harness that compares free memory across repeated calls also counts every other process
allocating on that GPU. plvs: a 21-call libelas leak check reported 8.00 MB lost on one
image while a SLAM run shared the card, and 0.00 MB once it ran alone, which is the
recorded figure. Run memory-delta checks with nothing else on the device (check
`rocm-smi --showpids`), and repeat any non-zero result alone before calling it a leak.
(plvs)

### A free-memory-delta test can fail once per process on gfx90a

HIP keeps device memory from some first uses for the life of the process. On gfx90a a
plain program's first pageable 100 MiB `hipMemcpy` round trip took 150 MiB of
`hipMemGetInfo` free memory for good, and later round trips took none. A test that reads
free memory before an allocation and asserts it comes back afterwards can then fail by a
fixed amount on its first run in a process and pass on repeats. Check with
`--gtest_repeat=3` in one process before calling it a leak: a leak grows each
repetition, a first-use cost appears once. (LichtFeld-Studio)

### A fixed "leak" after the first launch of a spilling kernel is the scratch allocation growing

A "no leak over N calls" test can fail on one AMD architecture by a fixed amount (25.75 MB in
one case) that does not change with problem size, while each operation warmed up on its own
shows zero drift. The first launch of a kernel that needs more private (scratch) memory per
lane than any kernel before it makes the HIP runtime grow the queue's scratch allocation,
sized per lane times the maximum number of resident waves, and keep it. Kernels tuned for
NVIDIA's 255 registers per thread (a fully unrolled column of 8 accumulators) can exceed the
128 VGPRs clang allows on gfx90a for a kernel without `__launch_bounds__` and spill hundreds
of registers (platform-limits.md, "A kernel launched with fewer than 1024 threads needs
`__launch_bounds__` on HIP to get its registers"). Separate the first call from the steady
state, and read `.private_segment_fixed_size`, `.vgpr_count` and `.vgpr_spill_count` for each
kernel from `hipcc --cuda-device-only -S`. Remove the spill rather than the test: a smaller
unroll (4 voxels per thread instead of 8 was also about 2x faster on MI250X) or
`__launch_bounds__` with the real block size (no spill, but slower there). Wave32 RDNA did
not spill on the same kernel. (TIGRE cone backprojectors)

### O_DIRECT I/O fails with EINVAL on a container's overlay filesystem

A storage test that opens its files with `O_DIRECT` in a directory under the test's working
directory can fail with `pwrite ... errno=22 (Invalid argument)` when that directory is on a
container's overlay root, and pass on the same host from tmpfs or a real ext4/xfs mount. It
looks like a GPU or pinned-buffer fault because the write source is a staged device buffer.
Rerun the failing cases from a tmpfs working directory (`cd /dev/shm/<dir> && <test binary>
--gtest_filter=...`), and check whether the project's CPU-only (simulated-device) build fails
the same cases on the same filesystem. If both point at the filesystem, it is the host
environment, not the port. (unified-cache-management)

## Multi-GPU hosts

### Two GPUs visible to one process can crash the runtime

On ROCm 7.14 a process that could see a mixed RDNA3 + RDNA4 pair crashed in the HIP
runtime before any kernel ran. It presents as a launch-time failure of a port that is
correct, so it is easy to spend an afternoon on the wrong suspect. If a machine has
more than one GPU and they are not the same architecture, pin `HIP_VISIBLE_DEVICES`
to exactly one per process and see whether the fault survives.

Do not carry an index in a script. Which index a card holds depends on what is
installed in that machine, so a pinned `HIP_VISIBLE_DEVICES=1` copied from older
notes silently selects a different card, or none. Read the device list at the time
you use it (`rocminfo`, `hipInfo`).

`amd-smi` numbers GPUs in its own order, which need not be HIP's. On a host with four identical cards
`HIP_VISIBLE_DEVICES=2` was amd-smi's GPU 0, the card another job was using, while amd-smi
GPU 2 was HIP device 0. To pick an idle card, match by PCI address: print
`hipDeviceGetPCIBusId` for each `HIP_VISIBLE_DEVICES=<i>`, compare with `amd-smi list`
(BDF) and `amd-smi process`, and read that card's `/sys/class/drm/card*/device/`
`gpu_busy_percent` / `mem_info_vram_used` (the sysfs card number is a third ordering).
(LichtFeld-Studio)
