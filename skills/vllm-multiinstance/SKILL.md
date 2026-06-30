---
name: vllm-multiinstance
description: Multi-instance vLLM benchmark on AMD EPYC CPU — runs N vLLM instances behind NGINX, drives load with guidellm, and reports peak memory (podman stats) + end-to-end throughput/latency. Use to benchmark a vLLM CPU image across models, concurrency rates, and instance counts. The harness is vendored here; nothing external is required beyond podman + ansible.
user-invocable: true
---

## Overview
A multi-instance vLLM + NGINX load-balancer benchmark for AMD EPYC CPU inference.
It runs N vLLM instances (each pinned to a range of physical cores) behind NGINX,
drives load with guidellm via ansible, and tears the stack down per run. You point
it at a container image and a model; it reports **peak memory** and **end-to-end
performance**.

You configure four things: **image**, **model**, **concurrency rate**, and
**instance count**. Everything else has sensible defaults.

**Self-contained.** The benchmark harness is vendored under `harness/` — there is
no dependency on any other repo. The only external piece is the guidellm/ansible
automation (`redhat-et/vllm-cpu-perf-eval`), which `scripts/setup-harness.sh`
clones + patches into `harness/` on first use.

## Layout
```
skills/vllm-multiinstance/
  SKILL.md            this file
  reference.md        copy-pasteable command-replay log from a real run
  harness/            vendored benchmark stack (no external repo needed)
    run_sweep.sh        sweep driver: stop → start → ansible guidellm → stop
    start.sh            generate compose, pre-warm HF cache, start N+nginx, wait
    generate-config.sh  emit docker-compose + nginx.conf
    stop.sh             podman-compose down (--clean removes volumes)
    vllm-cpu-perf-eval.patch   patch for the external ansible automation
    vllm-cpu-perf-eval/        cloned by setup-harness.sh (gitignored, ~62M)
  scripts/
    detect.py           print local CPU info as JSON (size the sweep from it)
    setup-harness.sh    one-time: clone + patch the ansible automation
    run_combo.sh        env-driven single-run driver (image + model + rate)
    mem_poll.sh         peak aggregate memory via podman stats
    parse_guidellm_log.py   scores from guidellm.log (authoritative)
    extract_perf.py     scores from benchmarks.json (fallback/cross-check)
```

## Harness flow
`run_sweep.sh` → `start.sh` (generate-config → HF pre-warm → `podman-compose up`
→ health wait) → ansible `llm-benchmark-concurrent-load.yml` (guidellm) →
`stop.sh --clean`. The stack is `NUM_INSTANCES` × `CORES_PER_INSTANCE` physical
cores pinned from `VLLM_START_CORE` (default 32), e.g. 3×32 → cores 32-63 / 64-95
/ 96-127. NGINX routes to instances by static IP. Results land in
`harness/vllm-cpu-perf-eval/results/llm/<model>/chat-<ts>-<test_name>/external-endpoint/`
(`benchmarks.json`, `benchmarks.csv`, `guidellm.log`).

Default image:
`amdih/zendnn_zentorch:vllm_v0.22.0_zentorch_v2.11.0.1_ubuntu22.04_2026_ww23`
(Docker Hub). Any vLLM CPU image works — just set `VLLM_IMAGE`.

## Step 1: Check your hardware
```bash
cd skills/vllm-multiinstance
python3 scripts/detect.py
```
This prints `physical_cores`, `sockets`, `numa_nodes`, `memory_gb`, etc. Size the
sweep from `physical_cores`, keeping all instances on **one socket**:

- `CORES_PER_INSTANCE` is fixed at **32** (the sweet spot).
- `NUM_INSTANCES = floor((physical_cores − 16) / 32)` — the 16 leaves headroom for
  NGINX and the OS. On a 128-core part that's 3 instances; on 64 cores, 1.

You also need enough RAM for `NUM_INSTANCES` model copies. `df -h /` — if root is
tight, set `BENCH_ROOT` to a roomy filesystem (temp + results land there).

## Step 2: One-time setup
```bash
# 1. Clone + patch the external ansible/guidellm automation into harness/.
bash scripts/setup-harness.sh          # idempotent

# 2. Pre-warm the model into a shared HF cache (offline runs need it on disk).
#    Use whatever Python env has huggingface-cli / hf (active venv/conda).
HF_HOME=$HOME/.cache/hf-shared/huggingface hf download <model>
```
Needs `podman`, `ansible-playbook`, and ansible collections `containers.podman`,
`ansible.posix`, `community.general` (setup-harness.sh reports which are missing).

## Step 3: Run a benchmark
`scripts/run_combo.sh` is env-driven. A run is defined by `LABEL`, `VLLM_IMAGE`,
and `MODEL`; the script handles the memory poller, stack naming, temp redirection,
HF cache, and teardown.

Required: `LABEL` (output name), `VLLM_IMAGE`, `MODEL` (`"repo-or-path | tag"`).
The `tag` may contain any characters — the harness sanitizes it to `[A-Za-z0-9-]`
for the ansible `test_name` (e.g. `qwen3-0.6b` → `qwen3-0-6b`), so you don't have
to pre-mangle dots/slashes.
Optional: `NUM_INSTANCES` (3), `CORES_PER_INSTANCE` (32), `GUIDELLM_RATES`
(`[32,64]`), `RUN_TAG` (output suffix so rate sweeps don't clobber), `MODELS_DIR`,
`BENCH_ROOT` (where `results/` lands; default `$PWD`). `NATIVE=1` bypasses zentorch
to A/B the same image with vanilla CPU vLLM.

```bash
cd skills/vllm-multiinstance
mkdir -p results          # must exist before any nohup redirect

IMAGE=amdih/zendnn_zentorch:vllm_v0.22.0_zentorch_v2.11.0.1_ubuntu22.04_2026_ww23
MODEL="Qwen/Qwen3-0.6B | qwen3-0.6b"

# One run:
LABEL=run1 VLLM_IMAGE="$IMAGE" MODEL="$MODEL" \
  bash scripts/run_combo.sh > results/run_run1.out 2>&1
```

### Sweeping a matrix
Define the matrix as a data table and loop — no script edits. Sweep **images**
and/or **concurrency rates**:
```bash
IMAGE=amdih/zendnn_zentorch:vllm_v0.22.0_zentorch_v2.11.0.1_ubuntu22.04_2026_ww23
MODEL="Qwen/Qwen3-0.6B | qwen3-0.6b"
# Each row: "label | image | extra-env"
MATRIX=(
  "zentorch | $IMAGE | "
  "native   | $IMAGE | NATIVE=1"
)
for row in "${MATRIX[@]}"; do
  IFS='|' read -r label image extra <<<"$row"
  label="${label// /}"; image="${image// /}"
  env $extra LABEL="$label" VLLM_IMAGE="$image" MODEL="$MODEL" \
    bash scripts/run_combo.sh > "results/run_${label}.out" 2>&1
done

# Sweep concurrency rates for one combo — RUN_TAG keeps outputs separate:
for rate in 32 64 96; do
  LABEL=run1 VLLM_IMAGE="$IMAGE" MODEL="$MODEL" \
    GUIDELLM_RATES="[$rate]" RUN_TAG="_c$rate" \
    bash scripts/run_combo.sh > "results/run_run1_c$rate.out" 2>&1
done
```
Each run takes ~8-15 min (load + rate×300s + teardown). Run the loop in the
background and wait on a sentinel — don't poll.

## Step 4: Pre-flight (optional)
- `podman ps | grep vllm` — kill/await any stack pinning your cores.
- `harness/run_sweep.sh --dry-run ...` — validates preflight, ansible path, env
  without starting containers.

### Host prerequisites & the host preflight
The harness assumes a **podman 4.x + netavark** host. On older/leaner hosts
(rootless podman 3.4.4, CNI 0.9.1) a first-timer hits several host-level
blockers that used to surface only as a cryptic deep failure or a 20-minute
health-wait hang. `harness/check-host.sh` now runs automatically inside both
`run_sweep.sh` (preflight) and `start.sh`, and **exits early with the fix**
instead of hanging:

| Check | Blocker it catches | What you'll see / the fix |
|-------|--------------------|---------------------------|
| **image short-name** | default `amdih/...` won't resolve without unqualified-search registries | `[BLOCK]` → set `VLLM_IMAGE=docker.io/amdih/...` (or `podman pull` it). Bypass: `ALLOW_SHORT_NAME=1`. |
| **rootless cpuset** | cgroup v2 cpuset not delegated → `cpuset.cpus: no such file or directory` | `[BLOCK]` with the one-time `Delegate=cpu cpuset io memory pids` systemd fix (needs root). Or run `--no-limits`. Bypass: `ALLOW_NO_CPUSET=1`. |
| **CNI cniVersion** | podman writes `cniVersion 1.0.0`, host plugins only support ≤0.4.0 → containers drop their static IPs → every LB request 504s | `[WARN]`; `start.sh` auto-downgrades the conflist to `0.4.0`. If it still fails, a **static-IP guard** in `start.sh` aborts in seconds (not 20 min) telling you to upgrade `containernetworking-plugins`. |

> **podman 3.x field paths (handled automatically).** Three places assumed
> podman-4.x inspect/info paths that render empty on podman 3.4.4, so the
> intended logic silently no-op'd. All are now version-agnostic:
> - **CNI backend detection** (`check-host.sh` + `start.sh downgrade_cni_version`)
>   keyed on `podman info .Host.NetworkBackend`, which doesn't exist on podman
>   3.x → the auto-downgrade never fired. Now bails only on an *explicit*
>   `netavark` backend; for `cni`/empty/unknown it proceeds (the conflist-file
>   check keeps it a safe no-op elsewhere).
> - **Health wait** read `.State.Health.Status` (podman 4.x); on podman 3.x the
>   field is `.State.Healthcheck.Status`, so healthy instances looked perpetually
>   unready → 20-min timeout. A `container_health()` helper now tries both paths.

> **podman-compose version on podman 3.x.** podman-compose **1.6.0** emits the
> podman-4.x `--network net:ip=` syntax, which podman 3.4.4 *silently ignores* —
> containers fall back to the default net and lose their static IPs. On a podman
> 3.x host, pin **podman-compose 1.0.6** (it uses `--net <name> --ip=`, which
> assigns the static IPs correctly): `pip install 'podman-compose==1.0.6'`.

`start.sh` also **fast-fails** the health wait the moment any required container
exits/dies (prints its last log lines) rather than polling for the full
timeout. Skip all host checks with `SKIP_HOST_CHECK=1`.

### Aborting a run / cleaning up
The driver chain is `run_combo.sh → run_sweep.sh → ansible-playbook` plus a
background `mem_poll.sh`. To stop a run cleanly, kill the drivers then tear the
stack down (`stop.sh` now also removes the compose network, so the next start is
clean):
```bash
pkill -9 -f run_sweep.sh; pkill -9 -f run_combo.sh; pkill -9 -f mem_poll.sh
pkill -9 -f ansible-playbook
bash harness/stop.sh --clean        # stops vLLM stack + removes network
```
**By default the guidellm load generator runs ROOTFUL** (under
`/var/lib/containers`, owned by root) via ansible `become`, which needs
**passwordless sudo**. A rootless `podman ps` won't even list it, and your
user-level `podman rm` / `kill -9` can't touch it. It normally self-exits the
moment the vLLM endpoint is torn down; if one is stuck, remove it with sudo:
```bash
sudo podman ps -a | grep guidellm
sudo podman rm -f <guidellm-container>
```

**Hosts without passwordless sudo:** guidellm doesn't actually need root (its
container is `network:host` + `user 0:0`), so the whole playbook can run
rootless as the invoking user. `run_sweep.sh` **auto-detects** this — if
`sudo -n true` fails it injects `-e ansible_become=false` and prints an `[INFO]`
line. Force it either way with `--no-become` / `--become` (or
`ANSIBLE_NO_BECOME=1`). When run rootless, the guidellm container *is* listed by
your normal `podman ps` and you can `podman rm -f` it without sudo. This also
means a sudo-less host **fails fast in preflight** instead of ~10 min into the
ansible health-check retries.

## Step 5: Collect scores
**Always read scores from `guidellm.log`, not `benchmarks.json`.** The JSON's
`requests_per_second`/`output_tokens_per_second` are *per-request medians* and
understate server throughput; the log's "Server Throughput Statistics" table is
the correct server-aggregate number.

```bash
python3 scripts/parse_guidellm_log.py <...>/external-endpoint/guidellm.log
# conc  req/s  in_tok/s  out_tok/s  tot_tok/s  lat_s  TTFT_ms  ITL_ms  TPOT_ms
grep "^PEAK" results/mem_<label>.csv      # peak aggregate memory (<label> = LABEL[+RUN_TAG])
```
Disambiguate same-named result dirs by timestamp order (`ls -1dt .../chat-*`),
cross-check each run's image in `results/run_<label>.out` (`VLLM_IMAGE=`), and
verify each run's `Failed : 0`. `extract_perf.py` parses `benchmarks.json` as a
fallback/cross-check only.
