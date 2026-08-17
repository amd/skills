# Reference

Detail behind [SKILL.md](SKILL.md): what the fetched tree contains, which file
answers which question, and what to do when the fetch or a read fails.

- [The cache directory](#the-cache-directory)
- [How the tree is organized](#how-the-tree-is-organized)
- [The three starting points](#the-three-starting-points)
- [Skills reachable from the starting points](#skills-reachable-from-the-starting-points)
- [Artifacts passed between skills](#artifacts-passed-between-skills)
- [Environment dependency chains](#environment-dependency-chains)
- [Worked examples inside the tree](#worked-examples-inside-the-tree)
- [Maintaining the cache](#maintaining-the-cache)
- [Troubleshooting](#troubleshooting)

## The cache directory

`<QUARK_TREE>` is `~/.cache/amd-quark-skills` on Linux and macOS, or
`%LOCALAPPDATA%/amd-quark-skills` on Windows. It holds a shallow, blobless,
sparse clone of `https://github.com/amd/Quark` containing two paths:

| Path | Why it is there |
| --- | --- |
| `.claude/` | Quark's skills |
| `examples/torch/language_modeling/llm_ptq/` | `quantize_quark.py`, the script the Torch PTQ workflow runs, plus its `requirements.txt` and reference docs |

Together with the git metadata that is a couple of megabytes on disk.

The location is deliberate. A cache directory keeps the user's repository clean,
survives across projects, and is a single `rm -rf` to undo. Nothing is copied
into the workspace and nothing is registered as a skill.

The clone tracks Quark's default branch, which is the current release branch, so
the tree stays aligned with the `amd-quark` build that `pip install amd-quark`
produces. Do not pin a branch or tag.

## How the tree is organized

Quark's skills sit in two layers. `.claude/skills/` holds thin stubs whose only
content is a pointer; `.claude/skills-impl/` holds the real bodies. This skill
routes straight to `skills-impl` and ignores the stubs.

Inside `skills-impl`, every skill is at
`<layer>/<backend>/<skill-name>/SKILL.md`, where backend is `shared`, `torch`,
or `onnx`. The layers, from the ground up:

| Layer | Role |
| --- | --- |
| `l0-foundation` | Environment and workspace facts. No quantization logic. |
| `l1-atomic` | Single-responsibility steps with explicit inputs, outputs, and recovery behavior. |
| `l2-workflows` | Orchestrators that chain atomic skills, manage checkpoints, and hand off artifacts. |
| `l3-recipes` | Named compositions that configure and combine an L2 workflow for a specific goal. |

`skills-impl/shared/` is not a skill layer: it holds the JSON schemas and
templates the skills refer to. The tree also carries a maintenance layer used by
Quark's own developers to keep their skills in sync with the codebase; it plays
no part in quantizing a model, so leave it alone.

## The three starting points

Paths are relative to `<QUARK_TREE>`.

**PyTorch / HuggingFace / safetensors PTQ** —
`.claude/skills-impl/l2-workflows/torch/quark-torch-llm-ptq-workflow/SKILL.md`

Four steps with a user checkpoint after each: model intake, quantization plan,
manifest generation, then execution of `quantize_quark.py`. Stops at the
quantized model. Covers FP8, INT4/UINT4 weight-only, INT8, and AMD FP4, with
optional AWQ, GPTQ, or SmoothQuant, and FP8 KV-cache quantization.

**ONNX PTQ** —
`.claude/skills-impl/l2-workflows/onnx/quark-onnx-ptq-workflow/SKILL.md`

Same four-step shape, but there is no shipped script for ONNX. Step 3 generates
a standalone Python script in the user's working directory that imports from
`quark.onnx`, and step 4 runs it. Covers the XINT8, A8W8, A16W8, BF16, BFP16,
and MX presets, with CLE, AdaRound, and AdaQuant as accuracy algorithms.

**Environment setup** —
`.claude/skills-impl/l1-atomic/shared/quark-install/SKILL.md`

Installs and verifies `amd-quark`: the universal wheel by default, the AMD index
pre-built wheels as the alternative when a C++ compiler is unwanted, plus
Python version limits, ONNX Runtime bounds, and LLM PTQ extras.

## Skills reachable from the starting points

Every path below is relative to `<QUARK_TREE>/.claude/skills-impl/` and ends in
`SKILL.md`. Read one when a starting point names it, not before.

**`l0-foundation/shared/`**

| Skill | What it does |
| --- | --- |
| `quark-env-preflight` | Detects OS, Python, accelerator, and GPU. Produces `env_context.json`. |
| `quark-workspace-validate` | Validates the paths a run will read and write. Produces `workspace_context.json`. |

**`l1-atomic/shared/`**

| Skill | What it does |
| --- | --- |
| `quark-install` | Installs and verifies the `amd-quark` package and its dependencies. Produces `quark_install_result.json`. |

**`l1-atomic/torch/`**

| Skill | What it does |
| --- | --- |
| `quark-torch-install` | Installs PyTorch matched to the accelerator. Produces `pytorch_install_result.json`. |
| `quark-torch-router` | Routes a Torch-side request to the right skill. Produces `session_context.json`. |
| `quark-torch-model-intake` | Parses the model config without loading weights, matches supported templates, flags risks such as MoE, very large models, and transformers version constraints. Produces `model_analysis.json`. |
| `quark-torch-quant-plan` | Chooses scheme, KV-cache scheme, exclude list, algorithm, and calibration settings. Produces `quant_plan.json`. |
| `quark-torch-result-validator` | Verifies the quantized output and the exported weights. |
| `quark-torch-debug` | Diagnoses Torch tracebacks, failed PTQ runs, and CUDA / ROCm out-of-memory errors. |
| `quark-torch-export` | Exports a quantized model to ONNX, JSON-safetensors, or GGUF. |
| `quark-torch-llm-eval` | Evaluates a model, for example perplexity or GSM8K / MMLU. |

**`l1-atomic/onnx/`**

| Skill | What it does |
| --- | --- |
| `quark-onnx-install` | Installs ONNX Runtime within Quark's supported range, CPU or GPU variant. Produces `onnx_install_result.json`. |
| `quark-onnx-router` | Routes an ONNX-side request and marks the backend in `session_context.json`. |
| `quark-onnx-model-intake` | Reads the graph: opset and IR version, input and output shapes, op histogram, whether QDQ nodes already exist, external-data status, deployment-target compatibility. Produces `model_analysis.json`. |
| `quark-onnx-quant-plan` | Chooses the preset, calibration method, and `algo_config`. Produces `quant_plan.json`. |
| `quark-onnx-result-validator` | Checks the quantized `.onnx`: QDQ insertion and initializers. |
| `quark-onnx-debug` | Diagnoses `quantize_static` failures, unavailable execution providers, and custom-op library load errors. |

**`l2-workflows/`** — `torch/quark-torch-llm-ptq-workflow` and
`onnx/quark-onnx-ptq-workflow`, the two PTQ starting points above.

**`l3-recipes/`**

| Skill | What it does |
| --- | --- |
| `torch/quark-torch-llm-ptq-eval` | Torch PTQ plus validation plus evaluation in one pass. Use when the user wants accuracy numbers, not just a quantized model. |
| `onnx/quark-onnx-autosearch-pro` | Searches for a quantization configuration instead of fixing one up front. Use for "auto search" or "find the best config". |

## Artifacts passed between skills

Skills communicate through files, and a workflow stops when one it needs is
absent. Their JSON schemas live in
`.claude/skills-impl/shared/contracts/`.

| Artifact | Produced by |
| --- | --- |
| `env_context.json` | `quark-env-preflight` |
| `workspace_context.json` | `quark-workspace-validate` |
| `session_context.json` | `quark-torch-router` / `quark-onnx-router` |
| `pytorch_install_result.json` | `quark-torch-install` |
| `quark_install_result.json` | `quark-install` |
| `onnx_install_result.json` | `quark-onnx-install` |
| `model_analysis.json` | `quark-torch-model-intake` / `quark-onnx-model-intake` |
| `quant_plan.json` | `quark-torch-quant-plan` / `quark-onnx-quant-plan` |
| `run_manifest.yaml` | either PTQ workflow, at its manifest checkpoint |
| `validation_report.md` | the result validators |

When a workflow reports a missing artifact, it names the skill that produces it.
Read that skill in the tree and run it. Do not hand-write the artifact: the
workflows explicitly forbid improvising a partial one, because downstream steps
trust its contents.

## Environment dependency chains

**Torch PTQ** needs, in order: `quark-env-preflight` →
`quark-torch-install` → `quark-install`. Quark builds against the installed
PyTorch, so torch comes first; `quark-install` refuses to proceed and hands off
to `quark-torch-install` if PyTorch is missing or mismatched.

**ONNX PTQ** needs the same chain plus `quark-onnx-install`. ONNX Runtime is
only an optional section of `quark-install`, so a user who ran just
`quark-install` still lands in `quark-onnx-install` when the ONNX workflow looks
for `onnx_install_result.json`.

Two footguns Quark's skills call out and worth repeating here: the universal
PyPI wheel compiles its kernels on first import and therefore needs a C++
compiler, which the AMD index pre-built wheels avoid; and on ROCm, pin the GPU
with `HIP_VISIBLE_DEVICES` rather than `CUDA_VISIBLE_DEVICES`, even though
`--device cuda` remains the correct flag on ROCm torch.

## Worked examples inside the tree

Both PTQ workflows ship a full walkthrough beside their `SKILL.md`, with real
commands, real numbers, and the expected output layout. They are the fastest way
to see what a run produces:

- `.claude/skills-impl/l2-workflows/torch/quark-torch-llm-ptq-workflow/example-fp8-qwen3-8b.md`
- `.claude/skills-impl/l2-workflows/onnx/quark-onnx-ptq-workflow/example-xint8-yolov8n.md`

## Maintaining the cache

```bash
# Update to the current Quark release
git -C "$HOME/.cache/amd-quark-skills" pull

# Check what was fetched
git -C "$HOME/.cache/amd-quark-skills" sparse-checkout list

# Remove it entirely
rm -rf "$HOME/.cache/amd-quark-skills"
```

Refresh when the user upgrades `amd-quark`, or when a skill in the tree
describes a flag the installed package rejects. Removing the directory is always
safe: the next run re-fetches it, and nothing else depends on it.

## Troubleshooting

**`git: 'sparse-checkout' is not a git command`** — git is older than 2.25.
Upgrade git. As a fallback, a full `git clone --depth 1` of the repository works
and costs a few MB more; the file paths are unchanged.

**Clone fails or hangs** — the host cannot reach `github.com`. Check the proxy
environment (`https_proxy`, `HTTPS_PROXY`) and try
`git ls-remote https://github.com/amd/Quark`. Report the failure and stop; do
not fall back to quantization advice from memory, and do not fetch raw file URLs
one at a time — the skills reference each other by relative path, so isolated
files are not usable.

**Clone succeeded but Step 3 lists no files** — the sparse-checkout patterns did
not apply. Re-run the `sparse-checkout set` command from
[SKILL.md](SKILL.md), then verify with `sparse-checkout list`.

**A skill names a file that does not exist** — Quark's skills cite their sources
with repo-root-relative paths such as `docs/source/install.rst` or
`quark/onnx/quantization/config/custom_config.py`. Those are provenance notes,
and they are outside the two paths fetched here. Nothing needs to be read from
them: the skill body already inlines what it needs. If a step genuinely requires
another part of the Quark repository, add it with
`git -C "$HOME/.cache/amd-quark-skills" sparse-checkout add <path>` rather than
cloning the whole thing again.

**A quantization run fails** — that is the tree's job, not this skill's. Read
`quark-torch-debug` or `quark-onnx-debug` from the paths listed above and follow
it.
