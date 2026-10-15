---
name: flydsl-kernel-authoring
description: >-
  Authors and reviews FlyDSL GPU kernels using explicit layout algebra, copy
  atoms, tiled MMA, LDS allocation, and loop-carried state. Applies when writing
  or understanding FlyDSL kernel code, mapping tensors to threads, choosing
  MFMA or WMMA atoms, or fixing FlyDSL frontend and compilation errors.
  Runtime trace interpretation belongs to kernel trace analysis.
---

# FlyDSL Kernel Authoring

Use FlyDSL's [existing kernel authoring skill](https://github.com/ROCm/FlyDSL/blob/fa7925e69e0abc313aa483084c2c8a3a58adba0d/.claude/skills/flydsl-kernel-authoring/SKILL.md)
to write or review kernels. This catalog entry locates that skill and loads it;
the API reference, examples, and compiler guidance are maintained in FlyDSL.

## Prerequisites

- Reading and reviewing the source needs no GPU or FlyDSL installation. Git and
  network access are needed only if a source checkout must be fetched.
- GPU execution follows the upstream skill's requirements: Linux, Python 3.12,
  ROCm 7.2, compatible PyTorch, and the intended target (`gfx942` or `gfx950`;
  architecture-specific examples state their own target).
- Use the existing compatible host or container; no container image is pinned
  here. Check `ARCH`, `HIP_VISIBLE_DEVICES`, and `ROCR_VISIBLE_DEVICES` before
  execution. Do not use `HSA_OVERRIDE_GFX_VERSION` to disguise a different GPU.

## Locate the source skill

1. Prefer the FlyDSL checkout supplied by the user or containing the workload.
   Check `FLYDSL_ROOT` if it is already set. Resolve it to an absolute path.
2. If no checkout is available, reuse or create a source checkout in the user's
   cache. On Linux:

   ```bash
   FLYDSL_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/flydsl-skills/FlyDSL"
   # Run clone only if this directory does not already exist.
   git clone --depth 1 https://github.com/ROCm/FlyDSL.git "$FLYDSL_ROOT"
   ```

   On Windows, use a directory under `%LOCALAPPDATA%/flydsl-skills` instead.
   Fetching the repository does not install FlyDSL. Preserve existing checkouts
   and their revision; use a user-requested revision when supplied.
3. Confirm that `$FLYDSL_ROOT/.claude/skills/flydsl-kernel-authoring/SKILL.md`
   exists. If it is missing or fetching fails, report the missing source; do not
   substitute a guessed API reference.

## Load and use the existing skill

Read that upstream `SKILL.md` as a file and follow it for the user's task. Open
the file explicitly rather than invoking this catalog entry again by name.
Resolve repository paths and companion skills against the same FlyDSL checkout.
Keep edits in the user's workload checkout.

Use the upstream reference for layouts, copy/MMA atoms, LDS, control flow, and
compiler behavior. Retain its distinction between API lookup, guided kernel
implementation, and debugging. When the task is code review, do not install a GPU
runtime or run a benchmark merely to read the reference.

Record the source checkout and revision used (`git -C "$FLYDSL_ROOT" rev-parse HEAD`).
Report the code or review result and which correctness or performance checks
actually ran. When no compatible GPU was used, leave GPU validation unclaimed.
