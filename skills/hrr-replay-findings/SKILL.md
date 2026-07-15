---
name: hrr-replay-findings
description: >-
  Run HIP Runtime Replay (HRR) against a capture archive and analyze the result.
  Classifies replay divergence or GPU faults (OOB/read-only write, illegal access,
  NaN/Inf D2H divergence, hang, OOM), extracts fault address, failing event/call
  index, and implicated kernel/allocation. Summarizes what an HRR capture holds.
  Use when given an HRR archive (capture.hrr/pid-*), when asked to replay a
  recording, or when triaging replay PASS vs MAF across workloads (vLLM, PyTorch,
  hipBLASLt StreamK).
---

# HRR Replay Findings

**Run replay** on an HRR archive, then produce a **structured finding** from the output.
Analysis is read-only (no patches); replay needs a GPU and a built `hrr-playback`.

## When to use

- User provides an HRR archive (`capture.hrr/pid-*`) and wants replay + triage
- Compare replay across GPUs or playback builds
- First step before deeper crash root-cause work

## Quick workflow

```
1. Resolve archive (largest events.bin under capture.hrr/pid-*)
2. Run replay → save log (run_hrr_replay.sh)
3. Parse log → structured finding (analyze_replay_finding.py)
4. Interpret + recommend next steps
```

## Step 0 — Prerequisites

| Requirement | How to verify |
|-------------|---------------|
| `hrr-playback` built | `HRR_PLAYBACK` points to executable, or on `PATH` |
| GPU free (~full VRAM for large captures) | `rocm-smi --showmeminfo vram` |
| Docker replay (recommended) | `HIP_SO` + `HSA_SO` from CLR/rocr build; `IMAGE` env |

**This workspace** (if `scripts/maf-hrr-docker-playback.sh` exists): set `HRR_REPO_ROOT` to the repo root and the skill runner delegates to that script.

## Step 1 — Archive metadata (no GPU)

```bash
SKILL=skills/hrr-replay-findings   # or .cursor/skills/hrr-replay-findings in a project

export HRR_PLAYBACK=/path/to/hrr-playback
./$SKILL/scripts/run_hrr_replay.sh \
  --archive /path/to/capture.hrr/pid-NNN \
  --info
```

## Step 2 — Run replay + save log

```bash
export HRR_REPO_ROOT=/path/to/hrr-repo          # optional: use repo docker helper
export HRR_PLAYBACK=/path/to/hrr-playback
export HIP_SO=/path/to/libamdhip64.so.7.*
export HSA_SO=/path/to/libhsa-runtime64.so.1
export GPU=1

./$SKILL/scripts/run_hrr_replay.sh \
  --archive /path/to/capture.hrr/pid-NNN \
  --log /path/to/replay.log \
  --analyze
```

`--analyze` runs the parser and writes `replay.finding.md` next to the log.

**Native replay** (no Docker): `HRR_REPLAY_MODE=native` or omit `HIP_SO` with GPU device visible.

**Extra hrr-playback flags** pass after `--`:

```bash
./$SKILL/scripts/run_hrr_replay.sh --archive ... --log replay.log -- --single-thread
```

## Step 3 — Analyze an existing log only

If replay already ran:

```bash
python3 $SKILL/scripts/analyze_replay_finding.py \
  --log replay.log \
  --archive capture.hrr/pid-NNN \
  --hrr-playback "$HRR_PLAYBACK" \
  --format markdown \
  -o finding.md
```

## Step 4 — Report template

```markdown
## Interpretation

- Fault class `read_only_page_fault` on `Cijk_*_SK3_*` → likely hipBLASLt StreamK edge/OOB write.
- `d2h_fail=0` at MAF → on-GPU divergence before host numerical mismatch.
- Archive incomplete (no EOF trailer) → common for crash captures with event recovery.
```

## Fault taxonomy

| `fault_class` | Meaning | Typical log signals |
|---------------|---------|---------------------|
| `replay_pass` | Replay finished cleanly | `[HRR] PASS`, `d2h_fail=0` |
| `read_only_page_fault` | Write to non-writable page | `Reason: Write access to a read-only page` |
| `illegal_memory_access` | Other GPU memory fault | `Memory access fault`, `MEMORY_FAULT` |
| `nan_inf_divergence` | Host saw bad numerics | `d2h_fail>0`, `[HRR] FAIL` |
| `hang` | Queue/device hang | `HSA_STATUS_ERROR_*`, no PASS/MAF |
| `replay_oom` | Replay ran out of VRAM | `out of memory`, `hipMalloc` + `Fatal:` |
| `replay_fatal_api` | HIP API error aborted replay | `[HRR] Fatal: T* Event *` |
| `replay_aborted` | Aborted without classified fault | `aborting replay` |
| `unknown` | Insufficient signals | — |

## Capture explainer (short)

- **events.bin** — HIP API stream (alloc, memcpy, launches, sync), kernargs, optional D2H snapshots
- **blobs/** — code objects, graphs, memcpy payloads
- **Trailer** — present on clean exit; absent on crash (reader recovers complete events)

Replay remaps recorded device pointers to live allocations. Divergence = replay state ≠ capture at some event index.

See [reference.md](reference.md).

## Agent instructions

When the user gives an **archive only**:

1. Run `--info` if `hrr-playback` is available
2. Check GPU VRAM; free device or pick another `GPU=` if OOM risk
3. Run `run_hrr_replay.sh --archive ... --log ... --analyze`
4. Present finding + interpretation; do not guess root cause without evidence

When the user gives a **log only**: skip replay; run `analyze_replay_finding.py`.

## Do not

- Guess ROCm commit or root cause without evidence
- Modify archives unless user requests `--repair`
- Assume one PASS archive means all archives pass

## Additional resources

- [reference.md](reference.md) — log patterns, kernel name decode
- [examples.md](examples.md) — sample commands and findings
