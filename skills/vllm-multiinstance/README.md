# vllm-multiinstance

Benchmark a vLLM CPU image on an AMD EPYC box: run **N vLLM instances behind
NGINX**, drive load with **guidellm**, and report **peak memory** + **end-to-end
throughput/latency**. The benchmark harness is vendored here — you only supply a
container image and a model.

This README sets expectations and gives copy-paste commands. For the *why* behind
each step see [`SKILL.md`](SKILL.md); for a full replay log see
[`reference.md`](reference.md).

---

## What you configure

Four things; everything else has sane defaults:

| Knob | Meaning | Default |
|------|---------|---------|
| `VLLM_IMAGE` | container image to benchmark | the Docker Hub image below |
| `MODEL` | `"repo-or-path \| tag"` | — (required) |
| `GUIDELLM_RATES` | concurrency rate list | `[32,64]` |
| `NUM_INSTANCES` | vLLM instances behind NGINX | `3` |

Default image:
```
amdih/zendnn_zentorch:vllm_v0.22.0_zentorch_v2.11.0.1_ubuntu22.04_2026_ww23
```

The `tag` half of `MODEL` may contain any characters — the harness sanitizes it to
`[A-Za-z0-9-]` for the run name (e.g. `qwen3-0.6b` → `qwen3-0-6b`). You don't have
to pre-mangle dots or slashes.

---

## Expectations (read before you start)

- **Time:** each run takes ~8-15 min (model load + `rate × 300s` + teardown). A
  2×2 matrix is ~40-60 min. Run sweeps in the background and wait on a sentinel —
  don't poll.
- **Cores:** instances pin physical cores starting at 32 (`3×32` → cores
  32-63 / 64-95 / 96-127). Only **one stack per machine** at a time — a second
  stack would fight for the same cores. Check `podman ps | grep vllm` first.
- **RAM/disk:** you need room for `NUM_INSTANCES` model copies. If root (`/`) is
  tight, set `BENCH_ROOT` to a roomy filesystem (temp + results land there).
- **Scores:** always read throughput from `guidellm.log`, **not**
  `benchmarks.json` (the JSON numbers are per-request medians and understate
  server throughput).
- **The guidellm load generator runs rootful.** A rootless `podman ps` won't list
  it; it self-exits when the endpoint is torn down. See *Aborting a run* below.

---

## Prerequisites

- `podman` + `podman-compose`
  - On a **podman 3.x** host (e.g. 3.4.4), pin `podman-compose==1.0.6`.
    Newer podman-compose (1.6.0) emits podman-4.x `--network net:ip=` syntax
    that podman 3.x silently ignores, so containers lose their static IPs.
- `ansible-playbook` and collections `containers.podman`, `ansible.posix`,
  `community.general`
- a Python env with `hf` / `huggingface-cli` (for the one-time model pre-warm)

---

## Quick start

```bash
cd skills/vllm-multiinstance

# 1. Size the sweep to your hardware.
python3 scripts/detect.py
#   NUM_INSTANCES = floor((physical_cores - 16) / 32)   # 128 cores -> 3

# 2. One-time: clone + patch the ansible/guidellm automation into harness/.
bash scripts/setup-harness.sh           # idempotent

# 3. One-time: pre-warm the model into a shared HF cache (offline runs need it).
HF_HOME=$HOME/.cache/hf-shared/huggingface hf download Qwen/Qwen3-0.6B

# 4. (Optional) Dry-run — validates preflight + ansible wiring, starts nothing.
VLLM_IMAGE=amdih/zendnn_zentorch:vllm_v0.22.0_zentorch_v2.11.0.1_ubuntu22.04_2026_ww23 \
  HF_TOKEN=offline HF_CACHE_DIR="$HOME/.cache/hf-shared" \
  harness/run_sweep.sh --dry-run -m "Qwen/Qwen3-0.6B | qwen3-0.6b"
```

---

## Run a benchmark

`scripts/run_combo.sh` is env-driven — one run is `LABEL` + `VLLM_IMAGE` + `MODEL`.

```bash
cd skills/vllm-multiinstance
mkdir -p results            # must exist before any nohup/redirect

IMAGE=amdih/zendnn_zentorch:vllm_v0.22.0_zentorch_v2.11.0.1_ubuntu22.04_2026_ww23
MODEL="Qwen/Qwen3-0.6B | qwen3-0.6b"

# Single run (zentorch):
LABEL=run1 VLLM_IMAGE="$IMAGE" MODEL="$MODEL" \
  bash scripts/run_combo.sh > results/run_run1.out 2>&1
```

### A/B: zentorch vs native (same image)

`NATIVE=1` bypasses zentorch to compare against vanilla CPU vLLM — no separate
build needed.

```bash
for row in "zentorch:" "native:NATIVE=1"; do
  label="${row%%:*}"; extra="${row#*:}"
  env $extra LABEL="$label" VLLM_IMAGE="$IMAGE" MODEL="$MODEL" \
    bash scripts/run_combo.sh > "results/run_${label}.out" 2>&1
done
```

### Sweep instance counts (e.g. 3-instance vs single-instance)

```bash
for n in 3 1; do
  LABEL="i${n}" NUM_INSTANCES=$n VLLM_IMAGE="$IMAGE" MODEL="$MODEL" \
    bash scripts/run_combo.sh > "results/run_i${n}.out" 2>&1
done
```

### Sweep concurrency rates (keep outputs separate with `RUN_TAG`)

```bash
for rate in 32 64 96; do
  LABEL=run1 VLLM_IMAGE="$IMAGE" MODEL="$MODEL" \
    GUIDELLM_RATES="[$rate]" RUN_TAG="_c$rate" \
    bash scripts/run_combo.sh > "results/run_run1_c$rate.out" 2>&1
done
```

### Background a long sweep and wait on a sentinel

```bash
nohup bash -c '
  for n in 3 1; do
    LABEL="i${n}" NUM_INSTANCES=$n VLLM_IMAGE="'"$IMAGE"'" MODEL="'"$MODEL"'" \
      bash scripts/run_combo.sh > "results/run_i${n}.out" 2>&1
  done
  echo ALL_DONE
' > results/sweep.out 2>&1 &
while ! grep -q ALL_DONE results/sweep.out; do sleep 60; done   # don't tight-poll
```

---

## Collect scores

```bash
R=harness/vllm-cpu-perf-eval/results/llm/Qwen__Qwen3-0.6B

ls -1dt "$R"/chat-*                         # newest-first; disambiguate by timestamp

# Server-aggregate throughput + median latency (authoritative):
python3 scripts/parse_guidellm_log.py "$R/chat-<ts>-<test_name>/external-endpoint/guidellm.log"
# conc  req/s  in_tok/s  out_tok/s  tot_tok/s  lat_s  TTFT_ms  ITL_ms  TPOT_ms

# Peak aggregate memory for a run (<label> = LABEL, + RUN_TAG if set):
grep "^PEAK" results/mem_<label>.csv

# Sanity: every run must be Failed : 0
grep -E "Failed +:" results/run_*.out
```

---

## Aborting a run / cleaning up

The driver chain is `run_combo.sh → run_sweep.sh → ansible-playbook` plus a
background `mem_poll.sh`.

```bash
pkill -9 -f run_sweep.sh; pkill -9 -f run_combo.sh; pkill -9 -f mem_poll.sh
pkill -9 -f ansible-playbook
bash harness/stop.sh --clean        # stops the vLLM stack AND removes the network
```

If a **rootful** guidellm container is stuck (it normally self-exits):

```bash
sudo podman ps -a | grep guidellm
sudo podman rm -f <guidellm-container>
```

> The harness names its stack `bench-vllm-*` when launched via `run_combo.sh`.
> Before killing anything, confirm you're not stopping **someone else's** stack
> (e.g. plain `vllm-instance-*`) sharing the host.
