# Hyperloom Workload Optimizer Reference

Iron Rules, CLI flags, and launcher contracts for
[SKILL.md](SKILL.md). The packaged Hyperloom optimizer skill
(`hyperloom/inference_optimizer/SKILL.md` after wheel install) is the
authoritative source for edge cases.

## Table of contents

1. [Iron Rules](#iron-rules)
2. [CLI workload flags](#cli-workload-flags)
3. [Critic and robustness backends](#critic-and-robustness-backends)
4. [Framework selection](#framework-selection)
5. [Failure signals](#failure-signals)
6. [Report fields](#report-fields)

## Iron Rules

Launcher gates that must hold before `python -m hyperloom.inference_optimizer.cli optimize`.

### IR-1 — GPU unoccupied before every launch

Before every `optimize` (fresh or `--resume`), verify every visible GPU has
**zero foreign serving PIDs and ≲ 500 MiB VRAM in use**. Leftover
`sglang.launch_server` / `vllm.entrypoints` / `Magpie` processes silently
degrade the next baseline.

### IR-2 — install.sh before every launch

Run `bash "$INSTALL_SH"` and source
`${KERNEL_AGENT_ENV:-${USER_DATA_PATH}/runtime/kernel-agent.env.sh}` in the
**same shell** that spawns `optimize`. Skipping install fails after baseline:
missing TraceLens/GEAK, hung Ray tasks, or `401` on kernel-opt gateway calls.

**Resume carve-out:** `--resume` may skip install only when all hold:

1. `install.sh` exited 0 earlier in the *same shell*
2. `kernel-agent.env.sh` is still sourced
3. The resumed session's `manifest.json` exists

Any failure → treat as a fresh launch and re-run `install.sh`.

### IR-3 — KB + PR Monitor (soft degrade)

`_preflight()` runs `preflight_kb.sh`. Exit `1` auto-enables `--degraded-kb` /
`--degraded-pr`; launch continues. IR-3 never aborts.

### IR-4 / IR-6 — EXPLORE contracts (Coordinator-internal)

- **IR-4:** EXPLORE is specialist-informed; GPU specialists lease cards via
  `gpu_research_lane` and must not touch production serving on port 8888.
- **IR-6:** EXPLORE force-exits when wall-clock remaining <
  `--explore-force-exit-hours-remaining` (default 3 h) or phase budget <
  `--explore-force-exit-budget-pct` (default 20%).
- Plateau signals are advisory; IR-6 and per-phase budgets are hard gates.

### IR-8 — `--framework atom` is single-node only

`--framework atom` rejects `--nodes >= 2` with exit code 2.

## CLI workload flags

Pass workload values as CLI flags — they are the source of truth for the
Coordinator.

| Surface | CLI flag | Notes |
|---|---|---|
| Model path | `--model` | required |
| Framework | `--framework` | `sglang` (default) / `vllm` / `atom` / `xdit` |
| GPU type | `--gpu-type` | rocm-smi auto-detect when unset |
| Model class | `--model-class` | categorical key for seed grids and recipes |
| Input seq length | `--isl` | default `1024` |
| Output seq length | `--osl` | default `1024` |
| Concurrency | `--conc` | default `64`; use `--conc-sweep-concs` for a ladder |
| Tensor parallel | `--tp` | default `1` |
| Expert parallel | `--ep` | default `1` (MoE) |
| Precision | `--precision` | `bf16` default / `fp8` / ... |
| Budget | `--max-hours` | CLI parser default `2.0`; this skill's Phase 2 workflow recommends `8` (aligns with hyperloom-custom-advanced) — launch always passes it explicitly |
| Max model len | `--max-model-len` | auto-derived from ISL+OSL when omitted |
| Reference GPU | `--compare-against-gpu` | optional external baseline |
| Quantization prelude | `--quantize` | runs quantization-agent once before the loop |

### Quantization prelude

When the user asks to quantize then optimize:

```bash
python3 -m hyperloom.inference_optimizer.cli optimize \
  --model "$MODEL_PATH" \
  --framework vllm \
  --quantize "fp8 global scheme, fp8 kv_cache, exclude lm_head" \
  --max-hours 4
```

Ignored on `--resume`.

## Critic and robustness backends

| Mode | Flag | When |
|---|---|---|
| Live critic | `--critic-agent` (default) | production runs |
| Mock critic | `--critic-mock` | offline / smoke |
| Live robustness | `--robustness-agent` (default) | single-node production |
| Mock robustness | `--robustness-mock` | multi-node auto-downgrade or smoke |

Multi-node (`--nodes >= 2`): CLI auto-downgrades robustness to mock (heartbeat
only) because local probes false-positive across pods.

## Framework selection

| Framework | Serving | Multi-node | Notes |
|---|---|---|---|
| `sglang` | yes | yes | default |
| `vllm` | yes | yes | |
| `atom` | yes | **no** | Magpie atom entrypoint |
| `xdit` | no | varies | diffusion `img/s` |

## Failure signals

| Symptom | Action |
|---|---|
| `stop_reason=no_more_leverage` | stop and report; resume only if user changes strategy |
| `stop_reason=time_exhausted` | `--resume` same session |
| `stop_reason=policy_loop` | inspect `policy_denial_history`; clear stale prunes before retry |
| `correctness_passed=false` | do not integrate kernel patch |
| `No accelerator` (Magpie) | fix `PATH` / `ROCR_VISIBLE_DEVICES` |
| Optimizer died before launch-info JSON | inspect `RUN_LOG`; never guess `session_dir` by timestamp |

## Report fields

Report back:

- session id (`manifest.json`) and log path
- `cumulative_gain`, `current_best`, `baseline_tput`
- explore accepted/rejected summary
- last kernel opt: correctness, micro speedup, E2E gain, KEEP/REVERT
- process alive vs `stop_reason`

Never print API keys or tokens.
