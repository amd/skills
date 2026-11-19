# Linux host setup for graphics and interop tests

Getting a GL or Vulkan context on a headless Linux GPU host, for ports that need one to run their tests.

<!-- toc:begin -->
Contents:

- Getting a real hardware GL context on a headless RDNA host
- Headless Vulkan on a Linux GPU host: Xvfb plus Mesa's software WSI
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
