---
name: optimizing-models-with-quark
description: >-
  Quantizes and optimizes a trained model with AMD Quark by fetching Quark's own
  quantization skills and routing the agent to the right one. Use when the user
  wants to quantize, compress, or shrink a PyTorch, HuggingFace, safetensors, or
  `.onnx` model, or mentions PTQ, post-training quantization, FP8, INT4, INT8,
  UINT4, MX, MXFP4, BFP16, XINT8, A8W8, A16W8, KV-cache quantization, AWQ, GPTQ,
  SmoothQuant, AdaRound, calibration data, or a quantized checkpoint for vLLM or
  Ryzen AI. Also use for "install AMD Quark", "pip install amd-quark", "set up
  Quark", or a ModuleNotFoundError for `quark`. Do not use to serve, deploy, or
  benchmark an already-quantized model on AMD hardware — see
  `serving-llms-on-instinct` or `serving-llms-on-epyc` — and do not use for GPU
  kernel profiling, which belongs to `magpie-kernel-evaluator`.
---

# Optimizing Models with AMD Quark

[AMD Quark](https://github.com/amd/Quark) turns a trained PyTorch or ONNX model
into a smaller, faster one through quantization. Quark ships detailed skills for
that work, but they live in the Quark repository and are **not** part of the
`amd-quark` pip package, so an agent cannot reach them by installing the
package.

This skill is the bridge. It puts Quark's skill tree on disk and names the file
to start reading from. That is all it does: it quantizes nothing itself, it does
not install the pip package, and it registers nothing as a skill.

**State this consequence to the user once, up front.** The tree is read as
ordinary files. Nothing is registered with the agent harness, so a later
quantization question will not route to a Quark skill on its own — every Quark
task comes back through this entry point. What that buys: nothing is written
into the user's workspace or any skills directory, and no restart is needed.

Throughout this skill, `<QUARK_TREE>` means:

- Linux / macOS: `~/.cache/amd-quark-skills`
- Windows: `%LOCALAPPDATA%/amd-quark-skills`

A cache directory, deliberately outside the user's workspace: it keeps their
repository clean, needs no `.gitignore` entry, and is reused across projects.
The commands below spell out the Linux and macOS path; on Windows substitute the
`%LOCALAPPDATA%` one.

## Step 1: Check whether the tree is already there

```bash
test -f "$HOME/.cache/amd-quark-skills/.claude/skills-impl/README.md" \
  && echo present || echo missing
```

If it prints `present`, go straight to Step 4 — do not re-fetch, and do not ask
the user about fetching.

## Step 2: Fetch it, with consent

Tell the user what is about to happen and wait for an explicit yes:

- a shallow, blobless clone of `https://github.com/amd/Quark` into
  `<QUARK_TREE>`, a couple of megabytes on disk
- nothing written to their workspace, their repository, or any skills directory
- no packages installed and no environment change

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/amd/Quark.git "$HOME/.cache/amd-quark-skills"
git -C "$HOME/.cache/amd-quark-skills" sparse-checkout set \
  .claude examples/torch/language_modeling/llm_ptq
```

Two paths, because two things are needed. `.claude` holds the skills. The
`llm_ptq` example holds `quantize_quark.py`, the script the Torch PTQ workflow
runs — the pip package does not ship it, and without it that workflow stalls at
its first step.

Clone the default branch. Do not pin a branch or tag: the default branch tracks
the current Quark release, which is what the installed `amd-quark` package will
match.

To update an existing tree later: `git -C "$HOME/.cache/amd-quark-skills" pull`.

## Step 3: Verify before routing

```bash
cd "$HOME/.cache/amd-quark-skills" && ls \
  .claude/skills-impl/l2-workflows/torch/quark-torch-llm-ptq-workflow/SKILL.md \
  .claude/skills-impl/l2-workflows/onnx/quark-onnx-ptq-workflow/SKILL.md \
  .claude/skills-impl/l1-atomic/shared/quark-install/SKILL.md \
  examples/torch/language_modeling/llm_ptq/quantize_quark.py
```

All four must be listed. If any is missing, **stop and report** — do not fall
back to guessing quantization flags from memory, and do not fetch individual
files one at a time. The files reference each other by relative path, so a
partial tree produces broken instructions. Common causes are a proxy blocking
GitHub and a git older than 2.25 (no `sparse-checkout`); see
[reference.md](reference.md).

## Step 4: Pick the starting point and read it

| The user wants to | Read this file, relative to `<QUARK_TREE>` |
| --- | --- |
| Quantize a PyTorch / HuggingFace / safetensors model | `.claude/skills-impl/l2-workflows/torch/quark-torch-llm-ptq-workflow/SKILL.md` |
| Quantize an `.onnx` model | `.claude/skills-impl/l2-workflows/onnx/quark-onnx-ptq-workflow/SKILL.md` |
| Install or verify the Quark environment | `.claude/skills-impl/l1-atomic/shared/quark-install/SKILL.md` |

Read the file and follow it as written, including its checkpoints — both PTQ
workflows stop for user confirmation after model intake, after the quantization
plan, and before execution. Honor those stops.

If the request is ambiguous, ask which model the user has rather than guessing:
a HuggingFace ID or a directory of safetensors goes to the Torch path, a file
ending in `.onnx` goes to the ONNX path.

## Following references inside the tree

The starting points are orchestrators. They name other skills by bare name
(`quark-torch-model-intake`, `quark-onnx-quant-plan`, and so on) and expect JSON
artifacts produced by skills further upstream. When a file names a skill you
have not read, **find it in the same tree and read it** — every skill lives at
`.claude/skills-impl/<layer>/<backend>/<skill-name>/SKILL.md`, so
`quark-torch-model-intake` is at
`.claude/skills-impl/l1-atomic/torch/quark-torch-model-intake/SKILL.md`.
[reference.md](reference.md) lists all of them with their paths.

Two things follow from that:

- The workflows say *"if any upstream artifact is missing, stop and name the
  missing producer skill."* When that happens, the fix is to read the producer
  skill in the tree, not to invent the artifact.
- The Torch workflow's first step tells you to locate `quantize_quark.py`, with
  a `find /` fallback. Skip the search: it is already at
  `<QUARK_TREE>/examples/torch/language_modeling/llm_ptq/quantize_quark.py`.

## Environment gaps worth pre-empting

Both PTQ paths assume a working environment and will bounce to a setup skill the
user has not heard of. Say so before it happens, so an unfamiliar skill name
does not read as a failure:

- **`quark-install` expects PyTorch to already be installed and verified.** If
  it is not, that skill hands off to `quark-torch-install`, which is the right
  order — Quark builds against the installed torch.
- **The ONNX path needs ONNX Runtime**, recorded in `onnx_install_result.json`
  and produced by `quark-onnx-install`. In `quark-install`, ONNX Runtime is only
  an optional section, so a user who ran just `quark-install` will still be sent
  to `quark-onnx-install`.

## Rules

- **Treat `<QUARK_TREE>` as read-only.** Quark's own skills forbid modifying
  upstream code, examples, and tutorials. Generated quantization scripts, output
  models, and artifacts go in the user's working directory, never in the cache.
- **Never fetch skill files one at a time.** The tree is the unit; relative
  references between files are why.
- **Never substitute your own quantization advice for the tree's.** If the tree
  is unavailable, say so and stop, rather than improvising flags for
  `quantize_quark.py` or a `QConfig`.
- **Do not copy the tree into the user's workspace or a skills directory**, and
  do not ask them to restart the agent. Reading files in place is the whole
  design.

For skill-by-skill paths, the artifact contracts, cache maintenance, and
troubleshooting, see [reference.md](reference.md).
