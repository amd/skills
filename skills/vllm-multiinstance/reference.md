# vllm-multiinstance — command reference (replay log)

Concrete, copy-pasteable commands for an end-to-end run: benchmarking a vLLM CPU
image across instance counts and concurrency rates, measuring memory footprint +
end-to-end performance.

This is a history, not a tutorial — see `SKILL.md` for the why. The benchmark
harness is **vendored in this skill** (`harness/`); the only thing you supply is a
container image and a model. Commands below assume you `cd` into the skill dir
first:
```bash
cd <repo>/skills/vllm-multiinstance      # all paths below are relative to here
```

Default image used throughout:
```bash
IMAGE=amdih/zendnn_zentorch:vllm_v0.22.0_zentorch_v2.11.0.1_ubuntu22.04_2026_ww23
```
To benchmark a custom image, build it however you like and point `VLLM_IMAGE` at
it — the harness benchmarks whatever image you give it. For a native (non-zentorch)
A/B on the same image, pass `NATIVE=1`; no separate build is needed.

---

## 1. Check the hardware

```bash
python3 scripts/detect.py
# -> physical_cores, sockets, numa_nodes, memory_gb, ...
```
Size the sweep: `CORES_PER_INSTANCE=32` fixed,
`NUM_INSTANCES = floor((physical_cores - 16) / 32)`, single socket. (128 cores → 3
instances.) Also sanity-check `df -h /` — set `BENCH_ROOT` to a roomy fs if root is
tight.

---

## 2. One-time harness setup

```bash
# Clone + patch the external ansible/guidellm automation into harness/.
bash scripts/setup-harness.sh         # idempotent

# Pre-warm the model into a shared HF cache (offline runs need it on disk).
# Use whatever Python env has huggingface-cli / hf (e.g. conda activate base).
HF_HOME=$HOME/.cache/hf-shared/huggingface hf download Qwen/Qwen3-0.6B
```

The patch applied by setup-harness.sh adds the rootless guidellm `user: "0:0"` fix
and the `/tmp → BENCH_TMPDIR` redirect to the ansible automation.

---

## 3. Pre-flight

```bash
nproc; lscpu | grep -E "Socket|Core|NUMA|Model name"   # enough physical cores? 1 socket?
podman ps --format '{{.Names}} {{.Status}}' | grep -i vllm   # any stack pinning your cores?
df -h /                                                  # root full? BENCH_ROOT on a roomy fs sidesteps it

# Stop a stale stack if present (by name):
for c in bench-vllm-instance-1 bench-vllm-instance-2 bench-vllm-instance-3 bench-vllm-nginx-lb; do
  podman rm -f "$c" 2>/dev/null; done

# Dry-run validates preflight + ansible path + env wiring (no containers):
VLLM_IMAGE="$IMAGE" NUM_INSTANCES=3 CORES_PER_INSTANCE=32 HF_TOKEN=offline \
  HF_CACHE_DIR="$HOME/.cache/hf-shared" \
  harness/run_sweep.sh --dry-run -m "Qwen/Qwen3-0.6B | qwen3-0.6b"
```

---

## 4. Run the sweep

`scripts/run_combo.sh` is env-driven: set `LABEL`, `VLLM_IMAGE`, `MODEL` per run.
Define the matrix as a data table and loop — no script edits.

```bash
mkdir -p results          # MUST exist before any nohup redirect or the job dies

IMAGE=amdih/zendnn_zentorch:vllm_v0.22.0_zentorch_v2.11.0.1_ubuntu22.04_2026_ww23
MODEL="Qwen/Qwen3-0.6B | qwen3-0.6b"
# Each row: "label | image | extra-env"  (extra-env e.g. NATIVE=1)
MATRIX=(
  "zentorch | $IMAGE | "
  "native   | $IMAGE | NATIVE=1"
)

cat > run_sweep_all.sh <<EOF
#!/bin/bash
set -uo pipefail
cd "$PWD"
MODEL="$MODEL"
MATRIX=( $(printf '"%s" ' "${MATRIX[@]}") )
for row in "\${MATRIX[@]}"; do
    IFS='|' read -r label image extra <<<"\$row"
    label="\${label// /}"; image="\${image// /}"
    echo "############ STARTING \$label ############"
    env \$extra LABEL="\$label" VLLM_IMAGE="\$image" MODEL="\$MODEL" \\
        bash scripts/run_combo.sh > "results/run_\${label}.out" 2>&1
    echo "############ FINISHED \$label rc=\$? ############"
    grep "^PEAK" "results/mem_\${label}.csv" 2>/dev/null || echo "no peak for \$label"
done
echo "ALL_DONE"
EOF
nohup bash run_sweep_all.sh > results/run_sweep_all.out 2>&1 &
while ! grep -q ALL_DONE results/run_sweep_all.out; do sleep 60; done   # wait on sentinel, don't poll

# One combo at a different concurrency — RUN_TAG keeps outputs separate:
LABEL=zentorch VLLM_IMAGE="$IMAGE" MODEL="$MODEL" \
  GUIDELLM_RATES="[96]" RUN_TAG="_c96" \
  bash scripts/run_combo.sh > results/run_zentorch_c96.out 2>&1
```

---

## 5. Collect scores — from guidellm.log (authoritative)

```bash
R=harness/vllm-cpu-perf-eval/results/llm/Qwen__Qwen3-0.6B

# Result dirs newest-first; runs may share a test_name → disambiguate by timestamp.
ls -1dt "$R"/chat-*

# Perf per run (server-aggregate throughput + median latency):
python3 scripts/parse_guidellm_log.py "$R/chat-<ts>-<test_name>/external-endpoint/guidellm.log"
# conc  req/s  in_tok/s  out_tok/s  tot_tok/s  lat_s  TTFT_ms  ITL_ms  TPOT_ms

# Peak memory per run (<label> = the LABEL you passed, + RUN_TAG if any):
grep "^PEAK" results/mem_<label>.csv
# PEAK label=<label> instances=3 agg_mem_bytes=... agg_mem_human=...

# Sanity: every run must be Failed : 0, no ENOSPC/fatal in guidellm.log
grep -E "Failed +:" results/run_*.out
grep -rliE "no space|fatal|traceback" "$R"/*/external-endpoint/guidellm.log || echo "clean"

# Cross-check each run's image (which image actually ran):
grep -hE "VLLM_IMAGE=" results/run_<label>.out | head
```

Do NOT report throughput from `benchmarks.json` — its
`requests_per_second`/`output_tokens_per_second` are per-request medians and
understate server throughput. `extract_perf.py` parses the JSON as a
fallback/cross-check only.
