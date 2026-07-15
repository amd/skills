---
name: hrr-replay-findings
description: >-
  Replay and analyze HIP Runtime Replay (HRR) capture archives. Classifies GPU
  faults and replay divergence (read-only/OOB write, illegal access, NaN/Inf,
  hang, OOM) and extracts fault address, event index, and kernel. Use when the
  user provides an HRR archive, asks to replay a recording, triage a GPU crash,
  or mentions hrr-playback, capture.hrr, or Memory access fault during replay.
---

# HRR Replay Findings

Replay an HRR archive on the host GPU, then produce a structured finding.

## What the user should say

The user only needs to point at the recording. Examples:

- *"Replay and analyze this HRR archive: `/data/capture.hrr/pid-1842`"*
- *"What's in this capture? `capture.hrr/pid-1842`"*
- *"Analyze this replay log from an HRR run"* (log-only path)

The user should **not** need to name scripts, set env vars, pick a GPU, or know where ROCm is installed.

## What to ask the user (only if missing)

| Missing | Ask once |
|---------|----------|
| Archive path | *"Which `capture.hrr/pid-*` directory should I use?"* |
| `hrr-playback` not found after discovery | *"Where is `hrr-playback` installed on this machine?"* |

Do **not** ask for: GPU index, Docker, source trees, HIP library paths, ROCm install path (assume `/opt/rocm`).

## Agent workflow

```
1. Resolve archive — user path, or largest events.bin under capture.hrr/pid-*
2. Discover hrr-playback (see below); ask user only if not found
3. Run skill scripts/run_hrr_replay.sh --archive <dir> --analyze
4. Read .finding.md and explain in plain language
```

**Execute in the same turn** — do not narrate planning steps.

### Discover `hrr-playback` (in order)

1. `command -v hrr-playback`
2. `$ROCM_PATH/bin/hrr-playback` (default `ROCM_PATH=/opt/rocm`)
3. `/opt/rocm/bin/hrr-playback`
4. User-provided path → set `HRR_PLAYBACK` for that run only

`run_hrr_replay.sh` adds `/opt/rocm/lib` and a sibling `lib/` next to the playback binary to `LD_LIBRARY_PATH` automatically.

### Discover archive

If the user gives `capture.hrr/` without a `pid-*` child, pick the `pid-*` directory whose `events.bin` is largest.

### Commands (agent runs these — not the user)

```bash
SKILL=<path-to>/hrr-replay-findings   # from .cursor/skills or installed skills dir

# Metadata only (~seconds, no full replay):
"$SKILL/scripts/run_hrr_replay.sh" --archive <pid-dir> --info

# Full replay + structured finding:
"$SKILL/scripts/run_hrr_replay.sh" --archive <pid-dir> --analyze
```

Log-only (no replay):

```bash
python3 "$SKILL/scripts/analyze_replay_finding.py" \
  --log <replay.log> --archive <pid-dir> --format markdown -o finding.md
```

## System assumptions

| Assumption | Default |
|------------|---------|
| ROCm install | `/opt/rocm` (`$ROCM_PATH` override) |
| GPU | Auto-pick device with most free VRAM |
| Replay mode | Native host (`/dev/kfd`); no Docker |

## Fault taxonomy

| `fault_class` | Meaning |
|---------------|---------|
| `replay_pass` | Clean replay |
| `read_only_page_fault` | Write to read-only page |
| `illegal_memory_access` | Other GPU memory fault |
| `nan_inf_divergence` | D2H numerical mismatch |
| `hang` | Device/queue hang |
| `replay_oom` | Out of VRAM — report environment issue |
| `replay_fatal_api` | HIP API error stopped replay |

## Capture explainer (short)

- **events.bin** — recorded HIP API sequence
- **blobs/** — code objects and payloads
- **Trailer** — absent when the original run crashed; reader recovers complete events

See [reference.md](reference.md) and [examples.md](examples.md).
