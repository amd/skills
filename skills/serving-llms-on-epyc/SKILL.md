---
name: serving-llms-on-epyc
description: >-
  Serves a language model on an AMD EPYC CPU host using vLLM with the zentorch
  backend, in a container (Docker or Podman) or a conda env. Use whenever the
  user wants to run, serve, deploy, start, host, or launch an LLM on AMD EPYC,
  Zen CPU, "vLLM on CPU", "zentorch serving", or "serve a model without a GPU".
  Use for "serve Qwen on EPYC", "start a CPU vLLM endpoint", "run an OpenAI
  server on my EPYC box", or similar. Handles the full single-instance flow:
  detect the CPU (incl. EPYC generation), validate the runtime/env, check vLLM
  supports the model (via vLLM's registry, not a modality blocklist), check it
  fits host RAM, size CPU threads/KV/NUMA from the hardware, confirm the plan with
  the user, launch, and poll until the endpoint is responsive. Single instance
  only. Does NOT debug failures
  and does NOT retry -- it reports and stops. Do not use for GPU/Instinct (use
  serving-llms-on-instinct) or multi-node.
allowed-tools: Bash, Read
---

# Serving LLMs on AMD EPYC (vLLM + zentorch, CPU)

Bring up a single vLLM OpenAI endpoint on an AMD EPYC host with the zentorch CPU
backend, sized to the hardware. Container-first (Docker or Podman); conda/host
is the fallback.

Hard rule for this skill: **on any failure, report the cause + logs and STOP.
Do not retry, do not debug.** (Debugging is a separate workflow.)

**The agent does the serve flow itself** -- pull, configure, launch, poll --
using the runtime `validate.py` reports. Never hand the user per-serve commands.
Like serving-llms-on-instinct, an accessible container runtime is a one-time
**prerequisite**: if `validate.py` finds none, report its one-time fix (make
docker accessible / install podman / provide a conda env) and stop. Do not
attempt `sudo` or privilege escalation.

## Data file

Read `data/epyc.json` directly. It holds the container image, mandatory CPU run
flags, supported precision, the model-support policy, the default model, and the
verified throughput-flag gotcha. Do not hardcode the image tag from memory -- read it.

## Step 1: Detect the CPU

```bash
python3 scripts/detect.py            # add --host user@box for a remote host
```

Returns `cpu_model`, `is_amd_epyc`, `epyc_generation`
(Naples/Rome/Milan/Genoa/Bergamo/Siena/Turin), `zen_arch`, `avx512`,
`logical_cores`, `physical_cores`, `sockets`, `numa_nodes`, `memory_gb`. If
`is_amd_epyc` is `false`, stop: this skill targets AMD EPYC. (Other x86 may work
but is unsupported here.) Carry `epyc_generation` / `avx512` through the later
phases -- e.g. AVX-512 + bf16 land on Zen4+ (Genoa/Turin), and Turin packs up to
128 cores/socket, which the thread-binding in Step 5 sizes from.

## Step 2: Validate the runtime and environment

```bash
python3 scripts/validate.py --image <image from data/epyc.json>
```

Returns `ready`, `runtime` (`docker`, `podman`, or null), `runtime_detail`,
`conda_path_available`, `ram_gb`, and `errors/warnings/advisories`. Pick the path:
- `runtime` is `docker` or `podman` -> container path (Step 6), used verbatim.
- `runtime` null but `conda_path_available: true` -> conda/host path.
- `runtime` null and no conda -> `ready` is false. Report the one-time
  onboarding `fix` (make docker accessible / install podman / conda env) and stop.

Do not proceed if `ready` is `false`.

## Step 3: Resolve and validate the model

If the user named no model, use `default_model` from `data/epyc.json`
(`Qwen/Qwen3-0.6B` -- ungated, tiny, fast first success). Otherwise use theirs.

Check that vLLM actually supports the model (do **not** blanket-block multimodal):

```bash
python3 scripts/check_model.py --model-id <model> --vllm-version <vllm_version from data/epyc.json>
```

- Exit 0 = vLLM serves it as a generation endpoint (`kind` `text` or `multimodal`),
  or support is undeterminable (gated/offline) -- proceed; launch confirms.
- Exit 1 = positively unsupported: the architecture is not in vLLM's registry, or
  it is a `pooling`/embedding/reranker (not a chat/completion endpoint). Report the
  printed `message` and stop.
- A `multimodal` model is allowed; a vLLM-supported multimodal arch may still hit a
  GPU-only kernel on CPU, which surfaces at load (the no-retry rule then applies).

**Precision/dtype**: native CPU dtypes are `bf16` (default), `fp16`, `fp32`. Use
`bfloat16` unless the user asks otherwise.

For gated models (Llama, Gemma) `HF_TOKEN` must be set and the license accepted on
HuggingFace; if not, stop and say so.

## Step 4: Check it fits host RAM

RAM is the ceiling on CPU (weights + KV cache both live in RAM). Run on ONE line:

```bash
python3 scripts/estimate_memory.py --model-id <model> --ram-gb <memory_gb from detect> --max-model-len <4096 or user value> --num-prompts <1 or desired concurrency>
```

Exit 0 = fits, exit 1 = does not fit. If `fit.fits` is false: **do not launch.**
Tell the user `required_gb` vs `ram_gb` and the printed `fit.action` -- reduce
`--max-model-len` to `fit.suggested_max_model_len` and retry, or use a smaller
model. `--max-model-len` and `--num-prompts` are the two knobs that move KV.
Extra flag: `--weight-gb N` overrides weights if a model has no HF metadata
(rare). KV cache is bf16-only on zentorch CPU (no fp8 KV).

## Step 5: Size the CPU runtime from the hardware

```bash
eval "$(python3 scripts/cpu_tune.py)"      # or --format json to inspect
```

Exports `VLLM_CPU_OMP_THREADS_BIND` (physical cores of **socket 0**) and
`VLLM_CPU_KVCACHE_SPACE` (GB). It does **not** set `OMP_NUM_THREADS` (vLLM derives
it from the bind list) or `VLLM_CPU_NUM_OF_RESERVED_CPU` (vLLM has its own default
when unset). Default policy, the same for NPS1/NPS2/NPS4: a single instance uses
**socket 0's whole CPU with no memory binding**. On a multi-socket host the JSON
gives `container_cpuset` (`--cpuset-cpus` only -- no `--cpuset-mems`) for the
container path; the conda path needs nothing extra (the bind env var binds the
threads). If socket 0 spans multiple NUMA nodes (NPS2/NPS4), `perf_note` flags that
optimal per-node binding could give more performance -- surface it, but proceed.

## Step 6: Confirm the plan, then launch (container-first)

Before launching, present this summary and **wait for the user to confirm** -- do
not launch unprompted. This is the human gate before anything runs:

| Field | Value |
|---|---|
| Model / kind | `<model>` -- `text` or `multimodal` (from `check_model.py`) |
| Path | container (`<runtime>`, image from `data/epyc.json`) or conda/host |
| Precision | `bfloat16` (or the user's choice) |
| Fit | required `<required_gb>` GB vs `<ram_gb>` GB RAM |
| CPU sizing | thread bind `<VLLM_CPU_OMP_THREADS_BIND>` (socket 0), KV `<VLLM_CPU_KVCACHE_SPACE>` GB, no memory binding |
| Hardware | EPYC `<epyc_generation>` (`<zen_arch>`), `<physical_cores>` cores, AVX-512 `<avx512>` |
| Port | `<port>` |

Proceed only on a clear "go". If the user declines or wants changes (model,
`--max-model-len`, port), stop and adjust -- do not launch.

Build the launch from `data/epyc.json`. The CLI is `vllm serve <model>`.
**Do not pass `--device cpu`** on vLLM >= 0.20 -- the zentorch plugin
auto-selects the CPU platform and `vllm serve` rejects the flag. Only add it if
`vllm serve --help` lists it (older vLLM).

**Container path** (`runtime` from validate.py). The agent runs these itself,
including the pull. `RT` is the resolved runtime verbatim:
```bash
RT="<runtime from validate.py: docker | podman>"
$RT pull <image from data/epyc.json>          # agent pulls; do not ask the user to
$RT run -d --name vllm-epyc \
  <run_flags from data/epyc.json>            # --ipc=host --shm-size=16g --network=host
  <hf_cache_mount> \
  <container_cpuset from cpu_tune, on multi-socket>   # --cpuset-cpus=... (no --cpuset-mems)
  --env VLLM_CPU_OMP_THREADS_BIND="$VLLM_CPU_OMP_THREADS_BIND" \
  --env VLLM_CPU_KVCACHE_SPACE=$VLLM_CPU_KVCACHE_SPACE \
  --env HF_TOKEN=${HF_TOKEN} \
  <image from data/epyc.json> \
  vllm serve <model> --dtype bfloat16 --port <port> --max-model-len <len>
```

**Conda/host path** (no container runtime, `conda_path_available` true). `eval`-ing
cpu_tune already exported the env vars; just launch -- `VLLM_CPU_OMP_THREADS_BIND`
binds the threads to socket 0, and there is no memory binding by default:
```bash
vllm serve <model> --dtype bfloat16 --port <port> --max-model-len <len> &
```

Optional throughput flags are **opt-in and must move together** (see Gotchas):
`TORCHINDUCTOR_FREEZING=1` + `VLLM_USE_AOT_COMPILE=0` (+ `ZENTORCH_WEIGHT_PREPACK=1`).
The base launch sets none of them.

## Step 7: Poll until up and responsive

A 503 while loading is normal. Poll until the server answers, then prove the
chat endpoint works. CPU first-token compile can take a minute or two.

```bash
# container alive (or process alive for conda) + /health
for i in $(seq 1 120); do
  # container path:
  $RT inspect -f '{{.State.Running}}' vllm-epyc 2>/dev/null | grep -q true || { echo "FAILED: container exited"; $RT logs --tail 50 vllm-epyc; break; }
  curl -sf http://localhost:<port>/health >/dev/null 2>&1 && { echo "HEALTHY"; break; }
  sleep 3
done
```

Then validate the OpenAI endpoint is actually accessible:
```bash
curl -sf http://localhost:<port>/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<model>","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
```

Resource sanity (your validation list): `$RT stats --no-stream vllm-epyc`.

**If the server never becomes healthy or the endpoint does not respond: print
the container/process logs, state the failure, and STOP. Do not retry. Do not
start a debugging loop.**

## Step 8: On success, hand over the endpoint

Print a connection table (model, runtime, port, OMP threads, KV GB, max-model-len,
NUMA pinning) and a ready-to-run example:
```bash
curl -s http://localhost:<port>/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<model>","messages":[{"role":"user","content":"Hello"}]}'
```
To stop: `$RT rm -f vllm-epyc` (container) or `kill <pid>` (conda).

## Offline (single-instance batch)

For a one-shot offline run instead of a server, replace Step 6-8 with a single
`vllm bench throughput` (or an offline `LLM.generate`) using the same sized env,
wait for completion, and report the metrics. Same no-retry / no-debug rule.

## Gotchas

See [reference.md](reference.md) for the full list. The load-bearing ones:

- **`--device cpu` was removed** from `vllm serve` in vLLM >= 0.20. The zentorch
  plugin auto-selects CPU. Passing it makes `vllm serve` error with
  "unrecognized arguments: --device cpu".
- **`TORCHINDUCTOR_FREEZING=1` alone crashes engine-core init** on vLLM 0.23 /
  zentorch 2.11 (`AssertionError: expected OutputCode, got function`). It only
  works with `VLLM_USE_AOT_COMPILE=0` set alongside it. Never set one without
  the other.
- **`--shm-size`**: vLLM needs a large `/dev/shm`; the container default (64MB)
  is too small. Use `--shm-size=16g` (in `data/epyc.json`).
- **NUMA**: the default is simple -- one instance on **socket 0's CPUs, no memory
  binding** (`--cpuset-cpus` from `cpu_tune.py` for the container; the bind env var
  for conda). If socket 0 spans multiple NUMA nodes (NPS2/NPS4), `cpu_tune.py` notes
  that optimal per-node binding could add performance; the base recipe doesn't do it.
