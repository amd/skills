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

## Method

**A "data-dependent, later-data-corrupts-earlier, per-tile" corruption signature is the fingerprint of a REPRODUCER bug, not a codegen fault.** cuSZ chased a suspected miscompile to a BLOCKED state and an IR bisect; the actual cause was the test input -- `np.arange(..., dtype=float32) * (python float)` promotes to float64, so `.tofile()` wrote 8 bytes per element and the tool read the stream as f32. Validate the byte width and dtype of any binary test input before escalating to an ISA bisect or a ROCm bug report. (cuSZ)

**Triangulate single- against double-precision before blaming the wavefront.** When a warp-collective rewrite shows SP divergence, run a second GPU variant and compare both to the DP oracle. If both GPU variants diverge from DP identically at the same positions, and the DP path is bit-identical to the CPU oracle, it is floating-point reassociation at a comparison boundary -- not a wave-size fault. SCAMP used this to clear a ~0.5 divergence at 10/8093 positions as a threshold-boundary artifact. (SCAMP)

**A once-per-suite host crash that stress reruns never reproduce: widen the suspected race window instead of rerunning.** If the stack sits in runtime code the port does not touch (`git diff` empty there), read it for a publish-before-register ordering, insert a temporary sleep between the two steps, rebuild, and run a tiny loop. A crash that becomes deterministic, and also happens with CPU-only work, is an upstream race that a fast GPU merely exposes; revert, rebuild and record it. oneflow's `LaunchLazyJob` pushed a graph's SourceTick before `EnqueueNNGraph`, so a tiny graph's finish callback could pop an empty queue: 1 crash in ~550 suite files on two arches, 0 in 70 stress reruns, 5 of 5 with a 20 ms sleep (CPU graphs included). (oneflow)

**ROCm 10.1 `AMD_LOG_MASK` takes decimal only, and graph replays are invisible to it.** A hex value such as `0x60000` is silently read as nothing (level 4 then prints one banner line), so check a mask on a tiny HIP program before trusting a quiet log. Bits on that runtime: 1 = API calls, 131072 = memory, 2097152 = `ShaderName` per dispatch. A graph's kernel nodes log `ShaderName` once, at `hipGraphInstantiate`, never on replay, and an illegal access logs `Memory Fault Error` with no address at any level (and `dmesg` is usually not readable in a container), so neither logging nor `AMD_SERIALIZE_KERNEL=3` names the kernel that faults inside a graph replay. Varying the replay order changes how often such a fault shows (oneflow: 0 of 12 to 7 of 8 at one commit), so an apparent regression between two commits can be the order hiding it.

**Before blaming the code under test for a wrong result, save the output, the inputs as read back and the reference, and recompute the reference on the CPU.** A test that checks a GPU result against an in-process GPU reference (torch, or the project's own float64 path) fails the same way when the REFERENCE is the wrong side. oneflow's "corrupt first-graph replay" on two gfx1100 GPUs survived two rounds of graph-capture hypotheses; dumping the arrays showed the graph output matching a CPU product of the read-back inputs to 9e-7 while the float64 reference GEMM on the other GPU came back as garbage, just before the illegal access. Also turn the suspected feature off first: the same fault with HIP graph capture disabled (`ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=0`, 2 of 2) cleared graphs in two runs. (oneflow)

**When a wrong result survives a `hipDeviceSynchronize` on every device, probe the HIP copy path itself before suspecting kernels.** Between the steps of the failing script, copy a known pattern host-to-device and back with plain `hipMemcpy` (ctypes is enough) into a host buffer pre-filled with -1, and report the wrong byte ranges. In oneflow's two-GPU nn.Graph run on gfx1100 (ROCm 10.1 SDK), such a probe on device 0 came back with whole 48 KiB pieces never delivered (still -1), with rc 0, after the larger graphs ran on device 1. That probe and torch's reference uploads run no oneflow code at all, but the state that makes them fail can still come from the application's two-device activity, so a clean probe on one device does not clear the application. **Do not attribute a timing-dependent fault from one single-variable condition on one build.** `HSA_ENABLE_SDMA=0` (copies by blit kernels instead of the SDMA engines) first looked decisive there (0 of 10 against 23 of 23 at the default), yet interleaved runs on two builds of the same source, differing only by inert env-gated debug hooks, showed it moving the rate in either direction: 8 of 8 to 0 of 8 on one build, 0 of 8 to 3 of 8 on the other, with the illegal access itself in a blit-kernel copy. The two builds also gave 8 of 8 against 0 of 8 at the default, and the clr queue toggles (`DEBUG_CLR_AQL_BARRIER_OPT`, `DEBUG_CLR_AQL_DEV_QUEUE`, `DEBUG_CLR_ENABLE_KDQ`, `DEBUG_CLR_DOORBELL_SKIP`, `DEBUG_CLR_USE_MOVDIR64B`, 3 of 3 each) and `DEBUG_HIP_DYNAMIC_QUEUES=0` (6 of 8) did not remove it. Before a condition attributes anything, interleave it with its control, at least 8 runs per cell, on more than one build. A newer runtime under the same build is the second cheap test: preloading a later nightly's `libamdhip64.so.7` and `libhsa-runtime64.so.1` (`LD_PRELOAD`; ABI-compatible within the same soname) ran 0 of 13 there. That is consistent with a runtime defect but does not prove one, because a runtime swap also changes timing and an application race (a missing cross-device dependency, a buffer freed while another device still reads it) would react the same way; it takes a reproducer without the application, or the one runtime commit that makes the difference, to place the bug in the runtime. `AMD_LOG_LEVEL=3 AMD_LOG_MASK=256` logs the per-stream SDMA engine assignment (one INFO line per assignment, little overhead). A probe on the legacy null stream returns 906 while another thread's stream is capturing, so place probes outside a capture or run with capture off. (oneflow)

**A kernel that hangs only in a Debug build: rebuild Debug at `-O2` before blaming an assert or the port.** A Debug HIP build turns two things on at once, `-O0` device codegen and every `assert` (including HIP's own `__hip_check_mask` in the `*_sync` intrinsics). Configure a second Debug tree with `-DCMAKE_HIP_FLAGS_DEBUG="-O2 -g0"` (NDEBUG still unset). If the test passes there, the asserts are not what fires and the hang is `-O0` codegen; find the kernel with `AMD_SERIALIZE_KERNEL=3 AMD_LOG_LEVEL=3` (last launch that never completes) and the PC with the SDK's `rocgdb` after a SIGINT. Run a Debug suite one test per process with a timeout so one hang does not hide the rest. The asserts-on `-O2` tree is also the better Debug evidence for mask and wave64 checks. (LichtFeld-Studio)

**An intermittent crash seen only in the GPU build needs a CPU-only build control, run as often and under the same load.** A multi-threaded application (SLAM, pipelines with worker threads) can carry an upstream race that the GPU path only changes the timing of. plvs stereo segfaulted in 4 of 9 HIP runs and in 0 of 6 CPU-only runs, which looks like a port fault, but the crash was in CPU line-mapping code (a null `KeyFrame*` in `MapLine::UpdateNormalAndDepth`). Six more of each, run side by side, gave HIP 0/6 and CPU 1/6. Build the control from the same tree with the GPU option off, into a copy of the source tree when the project writes its outputs into the source tree, get a backtrace (`gdb -batch -ex run -ex bt`), and count runs before you decide. A handful of runs per side cannot tell a rate of 25% from a rate of 8%. (plvs)

**A probe that contradicts a passing test, or that newly fails on a second arch, is wrong until proven otherwise: suspect the probe's own inputs first.** Two cases from one project. (1) A bisection concluded that HEonGPU's inverse NTT miscomputed on gfx90a while the encode/decode tests, which round-trip through exactly that inverse NTT, kept passing. The contradiction was written down and not reconciled, and the next person to pick up the work inherited the wrong suspect. Reconcile first: either the passing test does not use the path you think it does, or the probe set up its inputs differently from the library (mismatched tables, wrong API overload, wrong ordering convention). Write the probe against the library's OWN generated parameters and public entry points, never against parameters you regenerate by hand. (2) The Barrett-shift fix (61/62-bit moduli) verified clean on gfx90a; the identical probe strategy on gfx1100 showed 19998/20000 bad at those widths and looked like a wave32 regression the first arch had missed. The probe generated its test moduli as `2^bits - small_offset`, pathologically close to an exact power of two; the library's HOST-side `bit_generator()` computes `bit = (T1)(log2(value) + 1)` in plain `double`, and at 61-62 bits a value within roughly 1024 of a power of two rounds `log2` up to the boundary in double precision, over-counting the bit width by one and mis-sizing the unrelated Barrett `mu` constant -- identical behavior on any host, any back end, not a HIP or wavefront effect. Redrawing the probe's moduli uniformly across each target bit width (not clustered at the boundary) reproduced 0/20000 bad on both arches. Before concluding a fix regressed on a new arch, check whether the SECOND probe run's inputs, not just its outputs, differ from the first. (HEonGPU)

**A cascading wall of SegFault/OOM/"no kernel image available" partway through a long serial suite can be one poisoned physical card, not a port defect -- check `rocm-smi --showpids` against `--showmeminfo vram` before concluding anything.** arrayfire on a host with eight MI300X cards: a full `ctest` run started clean, crashed (SegFault) on an early binary, and every binary after it degraded -- more SegFaults, a "no kernel image is available for execution on the device" (hipErrorNoBinaryForGpu) on a JIT op, then explicit "Device out of memory". `rocm-smi --showmeminfo vram` showed that GPU pinned near its full capacity; `rocm-smi --showpids` showed *no process attached to it at all*, including after `kill -9` of every test binary -- the VRAM was orphaned at the KFD/driver level, not held by a live allocator, and a process crash should not leak GPU memory past its own exit. Re-running the identical suite (fresh JIT disk cache) on a different, verified-idle card on the same host, same build, same commit, came back 131/132 clean with no crash, no OOM, and no "no kernel image" anywhere. The disagreement between the two `rocm-smi` views is the tell: if `--showmeminfo` says a card is nearly full but `--showpids` shows nothing on it, that card's state cannot be trusted for this session -- switch cards to disambiguate before spending time on the port. (arrayfire)

## Hangs and crashes

### A hang that depends on inlining: long branches clobber the return address

**A kernel that never finishes, or a device function that returns to the wrong place, only at the default inlining level can be llvm/llvm-project#224205.** In a large device function that is not a kernel (the callee of `s_swappc_b64 s[30:31]`, for example a library reduction body with a big functor inlined into it), AMDGPU long-branch relaxation can use `s[30:31]`, the return address, as its scratch pair. Confirm it in the asm: compile the file with its real compile line plus `--cuda-device-only -S` for the target, and in each non-kernel function count `s_getpc_b64 s[30:31]` ... `s_setpc_b64 s[30:31]` long-branch sequences with no save or restore of `s30`/`s31` anywhere in that function. Any count above zero is the defect pattern. It appears on RDNA (gfx11, gfx12) targets in Linux builds as well as Windows ones, and it does not always hang, so a passing run does not clear a build that shows the pattern. Workaround: limit inlining for that one translation unit, through `COMPILE_OPTIONS` on the source, `/Ob1` for clang-cl and `-finline-hint-functions` for clang or hipcc on Linux. The function stays out of line, small enough that no long branch is left; recount in the asm to confirm zero. (h2o4gpu: the `FuncEvalF` reduction in `h2o4gpuglm.cu`, float and double; the double-precision fit hung on gfx1151 on Windows, and gfx1100/gfx1201 Linux builds carry the same pattern without an observed hang.)

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
