---
name: quark-torch-llm-ptq-eval
description: >-
  End-to-end Torch LLM PTQ recipe for AMD Quark — for PyTorch / HuggingFace transformers models
  (safetensors input): quantize, then validate, then evaluate accuracy, in one flow. Delegates the
  PTQ path (intake, planning, manifest, execution) to the quark-torch-ptq workflow, runs mandatory
  structural validation, and runs opt-in accuracy evaluation (perplexity / lm_eval /
  vLLM-accelerated). Trigger for "quantize and validate", "quantize and evaluate", "run PTQ end to
  end with accuracy check", "full PTQ pipeline with validation and eval", or "quantize
  Llama/Qwen/Mistral with FP8/INT4 and measure accuracy". For PTQ only (stop at the quantized
  output, no validation/eval) use quark-torch-ptq. Not for .onnx input models — use quark-onnx-ptq
  instead.
---

# quark-torch-llm-ptq-eval

## Catalog portability

This standalone skill is federated from `amd/Quark` at commit `1b229f781a1974cc742884e42d8eefc1eebb4f0a`. Resolve bundled files relative to this `SKILL.md`. Repository-relative paths such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a); use a local Quark checkout when available, otherwise consult that pinned source.

## Purpose

Run the complete PTQ lifecycle for a Torch LLM in one recipe: **quantize → validate → evaluate**.
This recipe composes existing skills rather than re-implementing them — Phase 1 hands off to the
`quark-torch-ptq` workflow (L2) for the PTQ steps, then Phases 2 and 3 chain the atomic validation
and evaluation skills (L1). The end result is identical to running the PTQ workflow and then the
validator and eval skills by hand; this recipe just drives the whole chain so the user does not have
to invoke them separately.

Use this recipe when the user wants quantization **plus** a correctness and/or accuracy check in a
single flow. If they only want the quantized model (no validation, no eval), route to `quark-torch-ptq`
instead — do not run this recipe and then skip its later phases.

## Inputs

- Model path (HuggingFace ID or local directory)
- User goal (target precision, hardware, accuracy)
- Output directory for the quantized model
- `session_context.json` for user goal and constraints
- `env_context.json` for hardware facts
- `workspace_context.json` for validated paths
- `pytorch_install_result.json` and `quark_install_result.json` to confirm runtime is ready

## Outputs

- `run_manifest.yaml` — executed PTQ command and config (from Phase 1)
- quantized model files in the user's output directory (from Phase 1)
- `validation_report.md` — structural validation result (from Phase 2, mandatory)
- `eval_report.md` — accuracy evaluation result (from Phase 3, optional)

## CRITICAL RULES

1. **Compose, do not re-implement.** Phase 1 is the `quark-torch-ptq` workflow run verbatim, with all
   its checkpoints. Do not inline or re-derive the intake/plan/manifest/execute steps here.
2. **STOP at every checkpoint** surfaced by the delegated workflow and at this recipe's eval checkpoint.
3. **Validation (Phase 2) is mandatory and automatic.** Evaluation (Phase 3) requires explicit user opt-in.
4. **NEVER modify Quark's own source code or examples.** The Quark repo (`quark/`, `examples/`, `tools/`,
   `docs/`, `tests/`) is read-only. See the same rule in the `quark-torch-ptq` workflow.

## Required Artifact Flow

```text
Phase 1 (PTQ via quark-torch-ptq)
  Step 1 (Intake)   ──► model_analysis.json
  Step 2 (Plan)     ──► quant_plan.json
  Step 3 (Manifest) ──► run_manifest.yaml      (contains the exact command)
  Step 4 (Execute)  ──► quantized model output (only after user says yes)
Phase 2 (Validate)  ──► validation_report.md   (auto, mandatory)
Phase 3 (Eval)      ──► eval_report.md         (optional, requires user opt-in + ROCm for Tier 3)
```

---

## Phase 1: Quantize (delegate to `quark-torch-ptq`)

**Goal**: Produce the quantized model plus `model_analysis.json`, `quant_plan.json`, and `run_manifest.yaml`.

Run the `quark-torch-ptq` workflow end-to-end (its Steps 1–4, including CHECKPOINT 1/2/3 and the
Step 3 "shall I run this?" confirmation). Carry forward, for the later phases:

- `<MODEL_PATH>` — source model, from intake
- `<OUTPUT_DIR>` — quantized model directory, from the manifest
- `quant_plan.json` — exclude rules and model paths, for the validator

Do not proceed to Phase 2 until Phase 1 has produced a quantized model in `<OUTPUT_DIR>`. If the PTQ
workflow stops at a checkpoint or fails, stop here too — there is nothing to validate or evaluate.

---

## Phase 2: Validate Output

**Goal**: Verify the quantized model is structurally sound. Runs automatically after Phase 1.

### Actions

Call `quark-torch-result-validator` with:

- `source_model_dir`: from intake (`<MODEL_PATH>`)
- `quantized_model_dir`: from manifest (`<OUTPUT_DIR>`)
- `quant_config`: from `quant_plan.json` (exclude rules for the MD5 check)

The validator runs four checks (fuzzy header → aux files → config.json → MD5) and emits `validation_report.md`.

### Result handling

- **All four checks pass** → one-line summary, proceed to Phase 3.
- **fuzzy layout or MD5 fails** → STOP. These are correctness checks; report mismatches and suggest fixes via `quark-torch-debug`. Do not proceed.
- **only config.json or aux files fail** → if the diff is benign metadata drift from a newer transformers re-serializing the model (renamed config keys, repackaged tokenizer files), mark it benign and continue. Otherwise treat as a real failure and STOP.
- **a real failure with an identifiable cause** → diagnose the root cause: is it a Quark issue (export/quantization bug) or an environment issue (transformers / accelerate / torch version, model loading, paths)? Report the cause briefly. If it is fixable, ask the user whether they want help resolving it, and on approval apply the fix and re-run validation. Fix the environment or regenerate the artifact — never edit upstream Quark code (Critical Rule 4). If the cause is unclear or not safely fixable, STOP and hand off to `quark-torch-debug`.

No checkpoint — validation is mandatory and automatic.

---

## Phase 3: Optional Accuracy Evaluation

**Goal**: Measure post-quantization accuracy and judge whether quantization hurt the model. Opt-in only.

### Actions

**Ask the user which evaluation tier to run** (or skip):

| Tier | Method | Speed | Notes |
|------|--------|-------|-------|
| 1. Quick — PPL | perplexity (e.g. wikitext) | fast, ~minutes | sanity check; runs on the quant device, no serving |
| 2. Medium — lm_eval | Python `lm-eval` library, direct generation | slow — warn the user it can take a very long time (no high-throughput serving) | task benchmark, e.g. gsm8k |
| 3. Full — accelerated | hand off to `quark-torch-llm-eval`: vLLM in docker + `lm_eval` over the OpenAI endpoint | fast + comprehensive | ROCm only |
| Skip | "no" | — | finalize the recipe |

### >>> CHECKPOINT: User must explicitly pick a tier (or skip) before eval runs

1. **Estimate the runtime before launching**, based on tier, model size, benchmark size, and hardware. Confirm before a long run. Rough guide:
   - Tier 1 PPL: ~1-3 min
   - Tier 2 lm_eval (direct generation): can be **hours** for a full benchmark — warn explicitly
   - Tier 3 vLLM gsm8k: ~10-20 min on MI300X-class

   Tier 3 requires ROCm. If the host is not ROCm, only tiers 1-2 are available — note this and offer the deferred command:

   ```text
   Full accelerated eval requires AMD ROCm. To run later on a ROCm host:
     /quark-torch-llm-eval model_path=<OUTPUT_DIR> benchmark=gsm8k
   ```

2. **Run the chosen tier:**
   - Tier 1 → run a PPL check on `<OUTPUT_DIR>`.
   - Tier 2 → run host `lm-eval` directly on the model (warn: slow).
   - Tier 3 → first check whether the current environment is **already inside a suitable docker container** (ROCm + vLLM available):
     - already inside a suitable container → run the eval there; do not create a new one.
     - not in a suitable container → **ask the user whether to create a new docker container** for the eval. Proceed only on approval; if declined, fall back to Tier 1/2 or skip.

     Then hand off to `quark-torch-llm-eval` with `model_path=<OUTPUT_DIR>`, `benchmark=<choice>`, `backend=vLLM`. It runs its own flow and produces `eval_report.md`.

   Known eval gotchas (apply while driving the eval skill / lm_eval):
   - `vllm/vllm-openai-rocm` image ENTRYPOINT is `vllm` → start a holder container with `--entrypoint sleep`, then `docker exec` the `vllm serve`.
   - Host `lm_eval` needs the `[api]` extra for OpenAI endpoints (a missing `tenacity` errors out immediately).
   - Reasoning models (e.g. Qwen3 `<think>`) need a large `max_gen_toks` on gsm8k; read the flexible-extract score.

3. **Report scores, then assess impact (impact assessment is opt-in).**
   - Always: report the quantized scores. If a published/known reference exists (model card, leaderboard), compare against it for free — no extra compute.
   - For an exact apples-to-apples delta you must run the BF16 baseline under the same config, which roughly **doubles eval time**. **Ask the user before running a baseline** — do not run it automatically.
   - Verdict (using whatever reference is available):
     - small delta (within ~1-2%) → quantization is safe, model looks healthy
     - large drop (> a few %) → quantization hurt the model; flag it and suggest remedies (try `smoothquant` / `awq`, exclude more sensitive layers, or raise `num_calib_data`)
   - Record the verdict in `eval_report.md`. If no reference exists and the user declines the baseline, report scores only and state the delta is unknown.

4. **If skip** → finalize, report all artifacts, recipe complete.

---

## Complete Example

For an end-to-end walkthrough (FP8 quantization of Qwen3-8B, then validate + eval), see
[`example-fp8-qwen3-8b.md`](example-fp8-qwen3-8b.md) alongside this file.

## Recovery

- If any upstream artifact is missing, stop and name the missing producer skill. Do not improvise a partial artifact.
- If Phase 1 (`quark-torch-ptq`) stops or fails, do not start Phase 2/3 — there is nothing to validate or evaluate.
- If the recipe hits a blocker (model cannot be loaded, scheme incompatible, execution fails), report it with diagnostic context rather than attempting ad-hoc fixes.
- If the user wants to change a decision mid-recipe (e.g., switch from FP8 to INT4 after seeing model analysis), go back to the relevant step in Phase 1 — do not restart from scratch.
