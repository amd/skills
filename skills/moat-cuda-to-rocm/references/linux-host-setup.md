# Linux host setup for graphics and interop tests

Getting a GL or Vulkan context on a headless Linux GPU host, for ports that need one to run their tests.

<!-- toc:begin -->
Contents:

- Getting a real hardware GL context on a headless RDNA host
- Headless Vulkan on a Linux GPU host: Xvfb plus Mesa's software WSI
- Driving a compute demo whose only entry point is an interactive GL window
- A GPU test that hangs only under `ctest -jN` is usually the software GL stack
- HIP and Vulkan in one process can oops the kernel on an old amdgpu-dkms
<!-- toc:end -->

### Getting a real hardware GL context on a headless RDNA host

A headless Linux box with no display manager can still give a GUI/interop port a REAL,
hardware-accelerated OpenGL context, if the GPU has a graphics pipeline (RDNA gfx11xx/gfx12xx;
not CDNA gfx90a/gfx942, which the driver refuses outright: faults-textures-interop.md,
"Compute-only CDNA has no graphics pipeline").
Install `xserver-xorg-core` and `xserver-xorg-video-amdgpu`, then start `Xorg` on a spare
display number against exactly one card. The naive config, a `Device` section with a legacy
PCI `BusID`, fails with `(EE) No devices detected` even though the bus ID is correct: the
amdgpu DDX is udev/platform-probed, not PCI-bus-probed, on a modern kernel, and a manual
`BusID` fights the automatic `OutputClass` match rather than selecting it. Xorg then silently
falls back to whatever other driver DOES probe successfully (on a server board this is
frequently the BMC's own display chip, e.g. Aspeed AST, which is real hardware but has no 3D
acceleration -- `glxinfo` then reports `swrast`/llvmpipe, not the discrete GPU, and looks like
a headless-GL failure when it is actually a wrong-device failure). Drop `BusID` and target the
render node directly instead: `Option "kmsdev" "/dev/dri/cardN"` in the `Device` section
(find N via `cat /sys/class/drm/card*/device/uevent | grep -B5 amdgpu`, matching the PCI slot
of the card you want), plus `Option "AutoAddGPU" "false"` so the other cards are not also
attached. A correct start logs `AIGLX: Loaded and initialized radeonsi` and
`GLX: Initialized DRI2 GL provider`; `DISPLAY=:N glxinfo -B` should then name the real card
(`OpenGL renderer string: AMD Radeon ... (radeonsi, ...)`), not llvmpipe. No input devices, no
monitor, and no physical display cable are needed for this to work.

On a multi-GPU host, that Xorg `kmsdev` card and a `HIP_VISIBLE_DEVICES` index are
independent numbering schemes with no guaranteed correspondence, and a GL-interop port needs
both to name the *same physical GPU*. A mismatch fails the very first
`cudaGraphicsGLRegisterBuffer`/`hipGraphicsGLRegisterBuffer` call with "invalid argument",
which reads like a port defect (the kind of interop bug a HIP port would introduce) but is
actually a host device-selection error external to the code. Decode `rocminfo`'s per-agent
`BDFID` (`bus = (BDFID >> 8) & 0xff`, e.g. `BDFID=17152=0x4300` -> bus `0x43`) and compare it
against the target card's PCI bus (`udevadm info --name=/dev/dri/cardN | grep DEVPATH`, the bus
segment right before `/drm/cardN`) to find the `HIP_VISIBLE_DEVICES` index that matches the
card Xorg is bound to, rather than assuming index 0 (or whatever index was used for a prior,
single-purpose HIP-only check) is also the GL-backed one.

### Headless Vulkan on a Linux GPU host: Xvfb plus Mesa's software WSI

A Vulkan app or test that needs a window can run on a headless RDNA host under `xvfb-run`,
but Xvfb has no DRI3, so RADV reports no presentation support and the app silently picks
llvmpipe (which lacks external memory/semaphore interop and skips the interop tests).
`export MESA_VK_WSI_DEBUG=sw VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json` lets
RADV present through Mesa's software path and hides llvmpipe; the GPU still does all the
rendering and interop. Screenshots come from `xwd -root -display :N`. (LichtFeld-Studio)

### Driving a compute demo whose only entry point is an interactive GL window

A research code with no test suite often puts the whole computation inside a GLUT display
callback, gated on a keypress. It still runs on a headless host: start it under
`Xvfb :N -screen 0 1024x1024x24` with `LIBGL_ALWAYS_SOFTWARE=1`, so the rendering falls to
llvmpipe while the compute stays on the AMD GPU, then drive the keypress with
`xdotool search --name <window title>` plus `xdotool key --window <id> <key>`. Read the
source for the flag the key toggles before assuming the program hung: GPU_IPC starts with
`bool stop = true` and does nothing at all until it sees a space, which for two runs looked
exactly like a wedged GPU. Also run it through `stdbuf -o0`: `printf` to a redirected file
is block-buffered, so a killed run reports zero progress however far it got. (This suits a
demo whose window only displays a computation; an application whose rendering or interop is
part of what was ported needs a real GL context, first entry above.)

Such a demo usually has no convergence criterion and no iteration cap either, so the
evidence to record is frames completed, the per-frame wall time, and whether a stall is
REPRODUCIBLE at the same frame. Codes that accumulate with `atomicAdd` on floats reorder run
to run, so the same binary reaching a different frame on two runs is expected and is NOT by
itself evidence of uninitialized memory or a race. (GPU_IPC)

### A GPU test that hangs only under `ctest -jN` is usually the software GL stack

A test that passes standalone and hangs in the suite is not a GPU fault and rarely a port
fault. Get the stack before theorising (`sudo gdb -p <pid> -batch -ex "bt"`, since
`ptrace_scope` normally blocks a plain attach) and read which library the top frames are in.
colmap's `feature/sift_test` hung under `xvfb-run -a ctest -jN` while passing 5 out of 5
standalone; the stack was `RunGpuTest -> ~QApplicationPrivate -> ~QXcbIntegration ->
XCloseDisplay -> libGLX_mesa -> libgallium -> pthread_clockjoin`, i.e. Mesa llvmpipe
deadlocking on its own worker threads while closing the X display with other GL clients
live on it. Nothing colmap or ROCm wrote appears in the trace, and the Qt frames name the
trigger: a `QApplication` built and destroyed per test.

It is a race, so one green run proves nothing, and a string of green ones does not establish
a rate either. Counts from one host at `-j16`, in the order taken: 2 hangs in 2 initial
`-j8`/`-j16` attempts; then 4 runs giving 1 full pass (159/159, 8.76 s) and 3 timeouts;
then, in a later session holding constant every variable those runs could name (same
machine and 64 cores, same Mesa build, same build directory, same command), 8 runs giving 7
passes (8.7-9.5 s) and 1 timeout, with the hang between clean runs. Those sessions do not
fit a single frequency (P(<= 1 hang in 8) is about 4e-4 if three in four hang), so the
frequency is unstable across conditions nobody has isolated; the only differences on record
are GPU visibility (`HIP_VISIBLE_DEVICES` pinned to one card in the last session) and
background load, and the stack implicates neither. Report the counts you observed and the
conditions you held; do not publish a rate, and do not let a green run retire the hazard.

The shape generalizes to any suite whose GPU tests build a windowing-toolkit context per
test: many short-lived GL clients on one Xvfb, and the teardown races. Isolate the two
variables separately before blaming either: running the one test alone against a
PRE-EXISTING shared display distinguishes "shared display" from "concurrent clients" (in
colmap it was the latter). The reliable workaround is lowering `-j`: colmap ran the whole
159-test suite in 11.6-12.3 s at `-j4` with no hang observed, which is also the shape of the
evidence -- the suite is not slow, one test is stuck. ctest reports the hang as `Timeout`
against its default 1500 s cap, which reads like a slow test rather than a deadlock: a suite
wall time sitting exactly at the cap with one test blamed is the tell. (colmap)

### HIP and Vulkan in one process can oops the kernel on an old amdgpu-dkms

A process that runs both HIP and RADV shares one amdgpu VM (libdrm hands both the same device
file), and once KFD owns that VM it is a compute VM. amdgpu-dkms 6.16.13 dereferences a NULL
`bo_va` in `amdgpu_gem_va_ioctl` when the Vulkan driver sends `AMDGPU_VA_OP_CLEAR` in a compute
VM: `BUG: kernel NULL pointer dereference, address: 0000000000000038`, `RIP:
amdgpu_gem_va_ioctl+0x731`. From user space it looks like a GPU hang: an iteration counter
stops, the process sits in D state on kill, and every later `open("/dev/kfd")` on every GPU in
the host blocks in `kfd_create_process -> __flush_workqueue`. `amd-smi reset` does not clear
it; only a reboot does. amdgpu-dkms 6.19.14 has the guard (`if (bo_va &&
bo_va->last_pt_update)`). A HIP-only or Vulkan-only process never reaches it, so a headless
run passing proves nothing about the interop path. Check `cat /sys/module/amdgpu/version`
before running any HIP+Vulkan interop app, and do not debug the port for this symptom on an
older driver. (LichtFeld-Studio: GUI training crashed twice at 6.16.13; on 6.19.14 the same
binary trained 5000 iterations cleanly.)
