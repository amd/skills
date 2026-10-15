# FlyDSL kernel authoring

Install the catalog entry and use the existing FlyDSL reference to review a
kernel fragment. This walkthrough needs no GPU or installed FlyDSL package.

## Install

From your project directory:

```bash
npx skills add amd/skills --skill flydsl-kernel-authoring --agent claude-code
```

Select another supported agent with `--agent` if needed. The installed
[entry point](../skills/flydsl-kernel-authoring/SKILL.md) finds your FlyDSL
checkout and reads its native `.claude/skills/flydsl-kernel-authoring/SKILL.md`.
If you have no checkout, it fetches the official repository into your cache.
Git and network access are needed only for that fetch.

## Review a runtime condition

Tell the agent where your FlyDSL checkout is if you already have one, then ask:

```text
Use flydsl-kernel-authoring to review this fragment without running a GPU:

lane = gpu.thread_id("x") % fx.Int64(64)
if const_expr(lane == fx.Int64(0)):
    store_result()

The kernel uses known_block_size=(256, 1, 1).
Write review.md with the problem and the corrected condition.
```

The agent should read the native FlyDSL skill, explain why `lane` is a runtime
value, and replace the `const_expr` condition with a runtime `if`. A known block
size constrains thread IDs without making the current lane a Python constant.
The report should identify the source revision used and state that no GPU check
was performed.

## Continue with your kernel

Supply the kernel file, tensor shapes and dtypes, intended operation, and target
GPU. The existing FlyDSL skill owns the layout, copy/MMA, LDS, and compiler
guidance; the catalog entry does not carry a second implementation of it.

For GPU execution, follow the native skill's environment requirements and
validate correctness before comparing performance. Updating the native
[FlyDSL reference](https://github.com/ROCm/FlyDSL/blob/main/.claude/skills/flydsl-kernel-authoring/SKILL.md)
is the way to improve that guidance.
