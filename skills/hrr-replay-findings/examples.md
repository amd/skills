# Examples

## Example 1 — Archive only (typical customer handoff)

Customer sends `capture.hrr/pid-1842/`. Support ships matching `hrr-playback`.

```bash
export HRR_PLAYBACK=/opt/rocm/bin/hrr-playback   # only if not on PATH

skills/hrr-replay-findings/scripts/run_hrr_replay.sh \
  --archive capture.hrr/pid-1842 \
  --analyze
```

No GPU index, no source tree, no HIP paths.

---

## Example 2 — Analyze log only

```bash
python3 skills/hrr-replay-findings/scripts/analyze_replay_finding.py \
  --log replay.log \
  --archive capture.hrr/pid-1842 \
  --format markdown
```

---

## Example 3 — Expected MAF finding (abridged)

| Field | Value |
|-------|-------|
| outcome | MAF |
| fault_class | read_only_page_fault |
| kernel_name | Cijk_..._SK3_... |
| d2h_fail | 0 |

`d2h_fail=0` → fault happened on GPU before host numerical checks failed.

---

## Example 4 — OOM (environment, not capture bug)

| Field | Value |
|-------|-------|
| fault_class | replay_oom |
| failing_api | hipMalloc |

Retry after freeing GPU memory; script auto-picks the GPU with most free VRAM.
