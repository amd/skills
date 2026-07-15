---
name: hrr-replay-findings
description: >-
  Run HIP Runtime Replay (HRR) against a customer capture archive and analyze
  the result. Classifies replay faults (OOB/read-only write, illegal access,
  NaN/Inf divergence, hang, OOM) and extracts fault address, failing event index,
  and implicated kernel. Requires only the archive and a shipped hrr-playback
  binary — no source checkout or GPU configuration knowledge. Use when given an
  HRR archive (capture.hrr/pid-*), a replay request, or Memory access fault output.
---

# HRR Replay Findings

Run replay on an HRR **archive**, then emit a **structured finding**. The user
supplies the recording; the agent handles playback discovery, GPU selection, and
analysis.

## What the user provides

| Input | Required? |
|-------|-----------|
| HRR archive (`capture.hrr/pid-*` with `events.bin`) | **Yes** |
| `hrr-playback` binary (on `PATH` or path via `HRR_PLAYBACK`) | **Yes** for replay |
| Source code, ROCm build tree, GPU index | **No** |

Optional: container image (`IMAGE`) if not using the default ROCm/vLLM image.

## Quick workflow (agent)

```
1. Find archive — largest events.bin under capture.hrr/pid-*
2. Find hrr-playback — PATH, HRR_PLAYBACK, or ask user once if missing
3. run_hrr_replay.sh --archive ... --analyze  (GPU auto-selected)
4. Present finding + plain-language interpretation
```

## Run replay + analyze

```bash
SKILL=skills/hrr-replay-findings

# Only set HRR_PLAYBACK if hrr-playback is not already on PATH
./$SKILL/scripts/run_hrr_replay.sh \
  --archive /path/to/capture.hrr/pid-NNN \
  --analyze
```

Writes `hrr-replay-pid-NNN-<timestamp>.log` and `.finding.md` in the current directory.

Archive metadata only (no GPU):

```bash
./$SKILL/scripts/run_hrr_replay.sh --archive /path/to/capture.hrr/pid-NNN --info
```

## Analyze an existing log

If replay already ran elsewhere:

```bash
python3 $SKILL/scripts/analyze_replay_finding.py \
  --log replay.log \
  --archive /path/to/capture.hrr/pid-NNN \
  --format markdown -o finding.md
```

## Agent instructions

**Do not** ask the user for GPU index, source paths, or HIP library paths unless replay fails and support provided a custom library bundle.

1. Locate archive (`pid-*` with largest `events.bin`).
2. Resolve `hrr-playback` (`command -v hrr-playback` or `HRR_PLAYBACK`).
3. Run `run_hrr_replay.sh --archive ... --analyze`.
4. If `replay_oom`: auto-retry on another GPU (script picks most free VRAM) or report insufficient VRAM.
5. Present finding; do not invent root cause without log evidence.

**Do not** require checkout of TheRock, CLR, or internal lab scripts.

## Fault taxonomy

| `fault_class` | Meaning |
|---------------|---------|
| `replay_pass` | Clean replay |
| `read_only_page_fault` | Write to read-only page |
| `illegal_memory_access` | Other GPU memory fault |
| `nan_inf_divergence` | D2H numerical mismatch |
| `hang` | Device/queue hang |
| `replay_oom` | Out of VRAM during replay |
| `replay_fatal_api` | HIP API error stopped replay |
| `unknown` | Insufficient log data |

## Capture explainer (short)

- **events.bin** — recorded HIP API sequence (alloc, memcpy, kernel launches, sync)
- **blobs/** — code objects and sidecar payloads
- **Trailer** — missing when the original run crashed; reader still recovers complete events

See [reference.md](reference.md) and [examples.md](examples.md).
