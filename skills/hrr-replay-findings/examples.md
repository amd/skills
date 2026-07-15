# Examples

## Example 1 — GPU memory fault during replay (StreamK GEMM)

**Input:** `replay.log` + archive `capture.hrr/pid-12345`

**Command:**

```bash
python3 skills/hrr-replay-findings/scripts/analyze_replay_finding.py \
  --log replay.log \
  --archive capture.hrr/pid-12345 \
  --format markdown
```

**Expected finding (abridged):**

| Field | Value |
|-------|-------|
| outcome | MAF |
| fault_class | read_only_page_fault |
| kernel_name | Cijk_..._MT128x192x128_..._SK3_... |
| fault_address | 0x7b9224c09000 |
| failing_event_seq | (from last `[HRR progress]` line) |
| d2h_fail | 0 |

**Interpretation:** On-GPU fault at a hipBLASLt StreamK GEMM; `d2h_fail=0` means host numerical checks had not failed yet — not a NaN/Inf divergence.

---

## Example 2 — clean replay pass

**Input:** `replay-pass.log`

| Field | Value |
|-------|-------|
| outcome | PASS |
| fault_class | replay_pass |
| kernels_launched | (from replay summary) |
| d2h_fail | 0 |

**Interpretation:** Replay completed without GPU fault or D2H mismatch.

---

## Example 3 — replay OOM (insufficient VRAM)

**Input:** `replay-oom.log`

| Field | Value |
|-------|-------|
| outcome | ABORT |
| fault_class | replay_oom |
| failing_call_index | (from `Fatal: T* Event *`) |
| failing_api | hipMalloc |

**Interpretation:** Replay aborted for lack of device memory — free VRAM or reduce conflicting workloads before attributing to capture fidelity.

---

## Example 4 — multi-run sweep summary

```bash
python3 skills/hrr-replay-findings/scripts/analyze_replay_finding.py \
  --log replay-gpu0.log \
  --sweep-tsv replay-sweep.summary.tsv \
  --format markdown
```

Use when several replays of the same archive were run (e.g. across GPUs). Consistent `fault_class` and kernel across runs suggests deterministic replay divergence, not GPU-specific hardware variation.
