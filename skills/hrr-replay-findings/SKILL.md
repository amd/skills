---
name: hrr-replay-findings
description: >-
  Read-only analysis of HIP Runtime Replay (HRR) recordings and replay logs.
  Classifies replay divergence or GPU faults (OOB/read-only write, illegal access,
  NaN/Inf D2H divergence, hang, OOM), extracts fault address, failing event/call
  index, and implicated kernel/allocation. Summarizes what an HRR capture holds.
  Use when given an HRR archive (capture.hrr/pid-*), hrr-playback log, capture log,
  multi-replay sweep TSV, Memory access fault output, or when triaging replay PASS
  vs MAF across workloads (vLLM, PyTorch finetune, hipBLASLt StreamK).
---

# HRR Replay Findings (read-only)

Turn **replay output + optional archive metadata** into a **structured finding**.
This skill is **read-only** — it does not patch, rebuild, or re-run capture unless
the user explicitly asks in a follow-up.

## When to use

- Replay or capture log shows `[HRR] PASS`, `Memory access fault`, `Fatal:`, or `d2h_fail`
- Comparing replay outcomes across GPUs, runs, or playback builds
- First triage step **before** end-to-end crash analysis (patch/build/validate workflow)

## Quick workflow

```
1. Identify inputs (archive, replay log, capture log, sweep TSV)
2. Run analyze_replay_finding.py (below)
3. Present the structured finding + capture explainer (reference.md)
4. Only if user asks: reproduce, bisect, or patch
```

## Step 1 — Gather inputs

| Input | How to find |
|-------|-------------|
| **Archive** | Largest `events.bin` under `capture.hrr/pid-*` |
| **Replay log** | `hrr-playback` stdout/stderr (saved to a file) |
| **Capture log** | Application log from the captured run; optional `[capture] HIP_SO=` line |
| **Sweep TSV** | Tab-separated summary from multiple replay runs (`*.summary.tsv`) |

## Step 2 — Run the parser (default)

```bash
SCRIPT=skills/hrr-replay-findings/scripts/analyze_replay_finding.py

python3 "$SCRIPT" \
  --log /path/to/replay.log \
  --log /path/to/capture.log \
  --archive /path/to/capture.hrr/pid-NNN \
  --hrr-playback /path/to/hrr-playback \
  --format markdown \
  -o finding.md
```

Optional multi-run table:

```bash
python3 "$SCRIPT" \
  --log replay-gpu0.log --log replay-gpu1.log \
  --sweep-tsv replay-sweep.summary.tsv \
  --format json
```

## Step 3 — Report template

Use the parser output and add **one paragraph** of interpretation:

```markdown
## Interpretation

- Fault class `read_only_page_fault` on `Cijk_*_SK3_*` → likely hipBLASLt StreamK edge/OOB write.
- `d2h_fail=0` at MAF → replay diverged on-GPU before any host numerical check failed.
- Archive incomplete (no EOF trailer) → common for crash captures with event recovery enabled.
```

## Fault taxonomy

| `fault_class` | Meaning | Typical log signals |
|---------------|---------|---------------------|
| `replay_pass` | Replay finished cleanly | `[HRR] PASS`, `d2h_fail=0` |
| `read_only_page_fault` | Write to non-writable page | `Reason: Write access to a read-only page` |
| `illegal_memory_access` | Other GPU memory fault | `Memory access fault`, `MEMORY_FAULT` |
| `nan_inf_divergence` | Host saw bad numerics | `d2h_fail>0`, `[HRR] FAIL` |
| `hang` | Queue/device hang without clean summary | `HSA_STATUS_ERROR_*`, no PASS/MAF line |
| `replay_oom` | Replay ran out of VRAM | `out of memory`, `hipMalloc` + `Fatal:` |
| `replay_fatal_api` | HIP API error aborted replay | `[HRR] Fatal: T* Event *` |
| `replay_aborted` | Aborted without classified fault | `aborting replay` |
| `unknown` | Insufficient signals | — |

## Capture explainer (short)

An HRR **capture** is a time-ordered **event stream** plus **sidecar blobs**:

- **events.bin** — HIP API calls (alloc, memcpy, kernel launch, sync), kernel names, kernarg payloads, optional D2H snapshots
- **blobs/** — code objects, graph capture data, memcpy payloads
- **Trailer** — clean shutdown marker when capture exits normally; **absent** when the app crashed (reader recovers complete events)

Replay **re-executes** API sequence on a live GPU, remapping recorded device pointers to new allocations. Divergence means replay state ≠ capture state at some event index — not necessarily that capture missed data.

Full layout: [reference.md](reference.md).

## Cross-workload notes

- **hipBLASLt / Tensile `Cijk_*`**: decode `MT<M>x<N>x<K>`, `_SK3_` = StreamK; see reference.md
- **PyTorch `_*` kernels**: last `[HRR progress] last=` demangled name is the active kernel before fault
- **Suballoc OOB lines**: `SUBALLOC OOB: kernel arg[N]` — pointer in segment but outside active tensor block
- **D2H checks**: `d2h_pass` / `d2h_fail` — host numerical validation; independent of on-GPU MAF

## Do not

- Guess ROCm commit, hipBLASLt tag, or root cause without evidence
- Treat `PASS` on one archive as proof all archives pass on the same build
- Modify archives (use `hrr-playback --repair` only when user requests)

## Additional resources

- Capture layout and log patterns: [reference.md](reference.md)
- Worked examples: [examples.md](examples.md)
