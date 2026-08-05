---
name: quark-workspace-validate
description: >
  Validate workspace shape, model paths, output directories, and repo structure before downstream Quark skills proceed.
  Use this skill whenever a skill needs confirmed file paths, when the user provides a model path or output directory,
  when you need to distinguish a local model from a HuggingFace ID, or when any path assumption is unverified.
  Also trigger for "where should I put outputs", "is my model path right", or before model-intake/export steps.
layer: l0-foundation
primary_artifact: workspace_context.json
source_knowledge:
  - examples/torch/language_modeling/llm_ptq/README.md
  - examples/torch/language_modeling/llm_ptq/quantize_quark.py
---

# quark-workspace-validate

## Purpose

Validate that file paths, model references, and output locations actually exist and are accessible before a downstream skill depends on them. This prevents silent failures where a PTQ run starts, spends time loading a model, and then crashes because the output directory does not exist or the model path has a typo.

## Inputs

- Model path and output directory from the user (typically forwarded by `the matching public quark-torch skill`)

## Outputs: workspace_context.json

Records validated model paths, output directory, and repo locations.

Schema: [`workspace_context.schema.json`](https://github.com/amd/Quark/blob/1b229f781a1974cc742884e42d8eefc1eebb4f0a/.claude/skills-impl/shared/contracts/workspace_context.schema.json)

```json
{
  "model_path": "/models/Qwen3-8B",
  "model_source": "local",
  "trust_remote_code_required": false,
  "output_dir": "./output/qwen3-8b-fp8",
  "output_dir_exists": false,
  "output_dir_parent_writable": true,
  "disk_space_gb_free": 250,
  "quark_script_path": null
}
```

`workspace_context.json` is for path facts only. Unresolved questions (e.g., ambiguous local-vs-HuggingFace reference) belong in `session_context.json` (owned by `the matching public quark-torch skill`).

## What to Validate

### Model Path

A model reference can be one of three things:

1. **Local directory** — contains `config.json`, `*.safetensors` or `*.bin` files (e.g., `/models/Llama-2-7b-hf/`)
2. **HuggingFace ID** — format `org/model-name` (e.g., `Qwen/Qwen3-8B`)
3. **Ambiguous** — could be either (e.g., `./qwen3-8b` might be a local dir or a typo)

For local paths, check:

- Directory exists
- `config.json` is present (minimum requirement for HuggingFace-compatible model)
- At least one weight file exists (`.safetensors`, `.bin`, or `.pth`)
- Read permissions are adequate

For HuggingFace IDs, note:

- Cannot fully validate without network access
- Check format: should contain `/` separator
- Record whether `trust_remote_code` will be needed (some models like DeepSeek VL v2 require it)

### Output Directory

- Check that the parent directory exists and is writable
- If the output directory itself does not exist, note that it will be created (not an error)
- Warn if the directory already contains files (risk of overwriting previous results)

### Quark Repository (if applicable)

- If the user is running from source, check for `quantize_quark.py` at the expected location: `examples/torch/language_modeling/llm_ptq/quantize_quark.py`
- Check that `requirements.txt` dependencies are likely installed

## Validation Commands

```bash
# Check model directory
ls -la /path/to/model/config.json 2>/dev/null
ls /path/to/model/*.safetensors 2>/dev/null

# Check output directory parent
test -d /path/to/output/.. && echo "parent exists"
test -w /path/to/output/.. && echo "parent writable"

# Check disk space (rough)
df -h /path/to/output/..
```

## Rules

- **Only validate existence, accessibility, and shape.** Do not analyze model internals (that is `quark-torch-model-intake`) or choose install paths (that is `quark-install`).
- **Never invent replacement paths.** If `/data/models/llama` does not exist, say so — do not suggest `/data/models/llama-2` as an alternative unless the user asks.
- **Preserve ambiguity explicitly.** When a reference like `meta-llama/Llama-2-7b` could be local or remote, say: "This looks like a HuggingFace model ID. If you meant a local directory, the path does not exist at `./meta-llama/Llama-2-7b`."
- **Check disk space** when the downstream task is PTQ or export — quantized models can be large.

## Interaction Flow

1. **Collect**: Gather the paths the downstream skill needs — model path, output directory, any referenced config files.
2. **Validate**: Run existence and permission checks. Present results as a checklist with pass/fail for each item.
3. **Clarify**: For ambiguous references, ask the user. For missing paths, report the exact issue.
4. **Emit**: Write confirmed facts to `workspace_context.json`. Hand any unresolved items back to `the matching public quark-torch skill` so they land in `session_context.json`'s `open_questions`.

## Recovery

- If a required path is missing, report the exact path tested and what was expected. Suggest the smallest fix: "Directory `/data/models/llama` does not exist. Did you mean `/data/models/Llama-2-7b-hf`?"
- If a model reference is ambiguous between local and HuggingFace, keep it unresolved and ask — do not guess.
- If disk space is low for the intended output, warn with the estimated size needed.
