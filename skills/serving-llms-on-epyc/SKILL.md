---
name: serving-llms-on-epyc
description: >-
  Serves an LLM on a supported AMD EPYC server CPU using vLLM with zentorch, in
  Docker, Podman, or conda. Use for "vLLM on CPU", "zentorch serving", or an
  EPYC CPU endpoint, including on a host that also has AMD Instinct GPUs.
  Detects the EPYC generation, validates the runtime, checks model support and
  RAM fit, sizes threads/KV/NUMA, confirms the plan, launches, and verifies the
  endpoint. Runs one instance on one socket and its memory. Reports and stops
  on failure; does not retry or debug. Use serving-llms-on-instinct when the
  endpoint should run on a GPU. Excludes multi-node, EPYC 4000, and pre-Zen4
  EPYC without AVX-512.
allowed-tools: Bash, Read
---

# Serving LLMs on AMD EPYC™ (vLLM + zentorch, CPU)

Bring up a single vLLM OpenAI endpoint on an AMD EPYC™ host with the zentorch CPU
backend, sized to the hardware. Container-first (Docker or Podman); conda/host
is the fallback. An installed AMD Instinct GPU does not disqualify the host:
select this skill when the endpoint itself should run on the EPYC CPU.

**This is single-socket serving:** one instance pinned to one socket and its memory
(vLLM scales poorly across sockets, so we do not span them). On a dual-socket host it
runs on a single socket; the multi-socket answer is **multiple instances (one per
socket)**, which is out of scope for this single-instance recipe.

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
verified throughput-flag gotcha. Its `vllm_version` and image tag are one
validated default stack; keep them aligned and do not hardcode either from memory.

## Step 1: Detect the CPU

```bash
python3 scripts/detect.py            # add --host user@box for a remote host
```

Returns `cpu_model`, `is_amd_epyc`, `epyc_generation`
(Naples/Rome/Milan/Genoa/Bergamo/Siena/Turin/Venice or EPYC 4004/4005),
`zen_arch`, `is_supported_epyc`, `avx512`, `logical_cores`, `physical_cores`,
`sockets`, `numa_nodes`, `memory_gb`.

Three hard gates -- stop if any fails:
- `is_amd_epyc` is `false` -> stop: this skill targets AMD EPYC. (Other x86 may work
  but is unsupported here.)
- `is_supported_epyc` is `false` -> stop: this recipe supports only the **AMD EPYC
  9000 series** for now -- Genoa (9004), Turin (9005), and Venice (9006). Other EPYC
  (Bergamo, Siena, EPYC 4004/4005, pre-Zen4) may even expose AVX-512, but ISA
  compatibility alone does not make them supported targets for this skill; stop.
- `avx512` is `false` -> stop: the zentorch CPU path **requires AVX-512**, i.e. Zen4+
  on the supported 9000-series parts above. Pre-Zen4 EPYC (Naples / Rome / Milan) is
  not supported -- say so and stop rather than launching into a load-time failure.

Carry `epyc_generation` / `avx512` through the later phases -- e.g. Venice packs up
to 256 cores/socket, which the thread-binding in Step 5 sizes from.

## Step 2: Validate the runtime and environment

```bash
python3 scripts/validate.py --image <image from data/epyc.json> --generation <epyc_generation from detect>
```

Returns `ready`, `requires_confirmation`, `runtime` (`docker`, `podman`, or null),
`runtime_detail`, `conda_path_available`, `stack`, `compatibility`, `ram_gb`, and
`errors/warnings/advisories`. Pick the path:
- `runtime` is `docker` or `podman` -> container path (Step 6), used verbatim.
- `runtime` null but `conda_path_available: true` -> conda/host path.
- `runtime` null and no conda -> `ready` is false. Report the one-time
  onboarding `fix` (make docker accessible / install podman / conda env) and stop.

Do not proceed if `ready` is `false`.

**Stack-compatibility gate.** `validate.py` probes the *selected* runtime for its
exact `vllm`/`zentorch`/`torch` versions and the active vLLM platform, then sets
`compatibility.status`:
- `proceed` -> the stack is the validated default (or a validated family on a Zen
  platform); continue.
- `blocked` -> a stock CPU platform is active, so zentorch acceleration is **not**
  on (error). Report `compatibility.message` and stop.
- `confirmation_required` (`requires_confirmation: true`) -> **Venice on a vLLM
  other than the pinned default**. This recipe has not been validated on Venice
  with that version. Surface `compatibility.message`, recommend the pinned
  `vllm_version` image from `data/epyc.json`, and **stop for an explicit user
  go/no-go** before launching. On the pinned default vLLM, Venice proceeds with no
  warning.

The gate only runs once the image is local. If `validate.py` reports the image is
not pulled, pull it (or let Step 6 pull it) and **re-run `validate.py`** so the
gate probes the real stack rather than only the tag.

## Step 3: Resolve and validate the model

If the user named no model, use `default_model` from `data/epyc.json`
(`Qwen/Qwen3-0.6B` -- ungated, tiny, fast first success). Otherwise use theirs.

Check that vLLM actually supports the model (do **not** blanket-block multimodal).
Pass the vLLM version the model will actually run on: use `stack.vllm` from
`validate.py` when it was probed (the conda env may differ from the pin), else the
`vllm_version` from `data/epyc.json`.

```bash
python3 scripts/check_model.py --model-id <model> --revision <rev or main> --vllm-version <stack.vllm from validate, else vllm_version from data/epyc.json>
```

- Exit 0 = vLLM serves it as a generation endpoint, or support is undeterminable
  (gated/offline) -- proceed; launch confirms.
- Exit 1 = stop: the architecture is not in vLLM's registry, it is a
  `pooling`/embedding/reranker (not a chat/completion endpoint), or it is a
  multimodal model with no usable chat template (`launchable: false`). Report the
  printed `message` and stop.

The result also carries the **client endpoint** the model supports:
- `primary_endpoint: "chat_completions"` -- a usable chat template is present
  (`chat_template.status: present`); serve and hand off `/v1/chat/completions`.
- `primary_endpoint: "completions"` -- no usable/auto-selectable template
  (`absent`/`ambiguous`/`unknown`); serve and hand off `/v1/completions` with a
  raw `prompt`. Chat can still be enabled by passing `--chat-template <file>` (or,
  for `ambiguous`, choosing one of `chat_template.names`); never invent one.
- Carry `primary_endpoint`, `supported_endpoints`, and `chat_template` through to
  verification (Step 7) and the handoff (Step 8).
- A `multimodal` model is allowed; a vLLM-supported multimodal arch may still hit a
  GPU-only kernel on CPU, which surfaces at load (the no-retry rule then applies).

**Precision/dtype**: native CPU dtypes are `bf16` (default), `fp16`, `fp32`. Use
`bfloat16` unless the user asks otherwise.

For gated models (Llama, Gemma) `HF_TOKEN` must be set and the license accepted on
HuggingFace; if not, stop and say so.

## Step 4: Check it fits host RAM

RAM is the ceiling on CPU (weights + KV cache both live in RAM). Run on ONE line:

```bash
python3 scripts/estimate_memory.py --model-id <model> --revision <rev or main> --ram-gb <memory_gb from detect> --max-model-len <4096 or user value> --num-prompts <1 or desired concurrency>
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

A single instance runs on **one socket, with its memory** (vLLM scales poorly across
sockets). `cpu_tune.py` exports `VLLM_CPU_OMP_THREADS_BIND` (the chosen socket's
physical cores) and `VLLM_CPU_KVCACHE_SPACE` (sized from that **socket's local RAM**,
not whole-system, so the KV pool stays on-socket). It does **not** set
`OMP_NUM_THREADS` (vLLM derives it) or `VLLM_CPU_NUM_OF_RESERVED_CPU` (vLLM's own default).

Socket choice on a dual-socket host (load-aware): it samples per-socket CPU busy%
(~0.5s) and prefers a free socket -- both free → socket 0; one free → that socket;
**both busy (≥ `--busy-threshold`, default 15%) → it `warning`s and proceeds on the
least-busy socket**. `--socket N` forces a choice. Single-socket hosts use socket 0.

For the chosen socket it also emits the memory-bound pin: `container_cpuset`
(`--cpuset-cpus=<cores> --cpuset-mems=<nodes>`) for the container path, and
`conda_launch_prefix` (`numactl --cpunodebind/--membind`, falling back to `taskset`
CPU-only, or empty-with-note if neither tool exists) for conda. **Surface `warning`
to the user** if set. On NPS2/NPS4 a socket spans multiple NUMA nodes; memory is
bound across them and `nps_note` flags that finer binding could add performance.

## Step 6: Confirm the plan, then launch (container-first)

Before launching, present this summary and **wait for the user to confirm** -- do
not launch unprompted. This is the human gate before anything runs:

| Field | Value |
|---|---|
| Model / kind | `<model>` -- `text` or `multimodal` (from `check_model.py`) |
| Path | container (`<runtime>`, image from `data/epyc.json`) or conda/host |
| Precision | `bfloat16` (or the user's choice) |
| Fit | required `<required_gb>` GB vs `<ram_gb>` GB RAM |
| CPU sizing | socket `<chosen_socket>` (`<socket_choice_reason>`), bind `<VLLM_CPU_OMP_THREADS_BIND>`, KV `<VLLM_CPU_KVCACHE_SPACE>` GB (socket-local), mem bound to nodes `<numa_nodes_on_socket>` |
| Hardware | EPYC `<epyc_generation>` (`<zen_arch>`), `<physical_cores>` cores, AVX-512 `<avx512>` |
| Port | `<port>` |

If `cpu_tune.py` returned a `warning` (e.g. all sockets busy), include it here so the user sees it before confirming.

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
$RT rm -f vllm-epyc 2>/dev/null               # clear any leftover container from a prior run (name collision otherwise)
$RT pull <image from data/epyc.json>          # agent pulls; do not ask the user to
$RT run -d --name vllm-epyc \
  <run_flags from data/epyc.json>            # --ipc=host --network=host (NO --shm-size: it conflicts with --ipc=host on podman)
  <hf_cache_mount> \
  <container_cpuset from cpu_tune>             # --cpuset-cpus=<cores> --cpuset-mems=<nodes>
  --env VLLM_CPU_OMP_THREADS_BIND="$VLLM_CPU_OMP_THREADS_BIND" \
  --env VLLM_CPU_KVCACHE_SPACE=$VLLM_CPU_KVCACHE_SPACE \
  --env HF_TOKEN=${HF_TOKEN} \
  <image from data/epyc.json> \
  vllm serve <model> --dtype bfloat16 --port <port> --max-model-len <len>
```

**Conda/host path** (no container runtime, `conda_path_available` true). `eval`-ing
cpu_tune already exported the env vars; prefix the launch with `conda_launch_prefix`
from cpu_tune so memory is bound to the chosen socket (empty → unpinned, with a note):
```bash
<conda_launch_prefix from cpu_tune> vllm serve <model> --dtype bfloat16 --port <port> --max-model-len <len> &
# e.g. numactl --cpunodebind=0 --membind=0 vllm serve ...
```

Optional throughput flags are **opt-in and must move together** (see Gotchas):
`TORCHINDUCTOR_FREEZING=1` + `VLLM_USE_AOT_COMPILE=0` (+ `ZENTORCH_WEIGHT_PREPACK=1`).
The base launch sets none of them.

## Step 7: Poll until up and responsive

A 503 while loading is normal. Poll `/health` until the server answers, confirm
the served model is listed, then prove the **selected endpoint** works (from
`primary_endpoint` in Step 3). CPU first-token compile can take a minute or two.
Track a `healthy` flag so a timeout is a failure, not a fall-through.

```bash
# 1. container alive (conda: process alive) + /health, with a real timeout
healthy=""
for i in $(seq 1 120); do
  $RT inspect -f '{{.State.Running}}' vllm-epyc 2>/dev/null | grep -q true || { echo "FAILED: container exited"; $RT logs --tail 50 vllm-epyc; break; }
  curl -sf http://localhost:<port>/health >/dev/null 2>&1 && { healthy=1; echo "HEALTHY"; break; }
  sleep 3
done
[ -n "$healthy" ] || { echo "FAILED: not healthy before timeout"; $RT logs --tail 50 vllm-epyc; }

# 2. the served model is registered
curl -sf --max-time 30 http://localhost:<port>/v1/models
```

Then exercise the endpoint the model actually supports. Use deterministic
sampling and a small output cap for the smoke check:

```bash
# primary_endpoint == chat_completions
curl -sf --max-time 180 http://localhost:<port>/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<served-model>","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"temperature":0}'

# primary_endpoint == completions  (no chat template)
curl -sf --max-time 180 http://localhost:<port>/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"<served-model>","prompt":"Hello, world","max_tokens":16,"temperature":0}'
```

Confirm the response is JSON with a non-error `choices[0]` (chat: `message.content`;
completion: `text`). An HTTP 200 that carries an `error` payload is **not** success.
Resource sanity (your validation list): `$RT stats --no-stream vllm-epyc`.

**If the server never becomes healthy, `/v1/models` omits the model, or the
endpoint returns an error/empty `choices`: print the container/process logs,
state the failing phase, and STOP. Do not retry. Do not start a debugging loop.**

## Step 8: On success, hand over the endpoint

Give the user everything needed to call the server. Print a connection table:

| Field | Value |
|---|---|
| Base URL | `http://localhost:<port>/v1` (the trailing `/v1` matters) |
| Served model | `<served-model>` (the id from `/v1/models`) |
| Endpoint | `/v1/chat/completions` or `/v1/completions` (from `primary_endpoint`) |
| Why | chat = a chat template is present; completions = no template (raw prompts) |
| Runtime / port | `<runtime>` / `<port>` |
| Sizing | OMP threads, KV GB, `--max-model-len`, socket / NUMA pinning |
| Stop | `$RT rm -f vllm-epyc` (container) or `kill <pid>` (conda) |

Then a ready-to-run example **for the selected endpoint**.

Chat model (`primary_endpoint: chat_completions`):
```bash
curl -s http://localhost:<port>/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<served-model>","messages":[{"role":"user","content":"Hello"}],"max_tokens":128,"temperature":0.7}'
```

Base/prompt model (`primary_endpoint: completions`):
```bash
curl -s http://localhost:<port>/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"<served-model>","prompt":"Hello, world","max_tokens":128,"temperature":0.7}'
```

OpenAI Python client (point `base_url` at the local server; the SDK requires a
non-empty key, so any placeholder works when the server has no auth):
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:<port>/v1", api_key="EMPTY")
model = client.models.list().data[0].id

# chat model:
r = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=128, temperature=0.7,
)
print(r.choices[0].message.content)

# base/prompt model:
r = client.completions.create(model=model, prompt="Hello, world", max_tokens=128)
print(r.choices[0].text)
```

Argument guidance to pass along (see [reference.md](reference.md) for the full list):
- `max_tokens` caps the **output**; `prompt_tokens + max_tokens` must be `<= --max-model-len`.
- `temperature` (0 = deterministic/greedy, higher = more random); tune `top_p` *or*
  `temperature`, not both.
- `stream: true` streams tokens (SSE) instead of one blocking response.
- The model's `generation_config.json` can set sampling defaults; pass explicit
  values to be sure.

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
- **`/dev/shm` — use `--ipc=host`, not `--shm-size`.** vLLM needs a large
  `/dev/shm` (the 64MB container default is too small). The base recipe uses
  `--ipc=host`, which shares the host's large shared memory. **Do not also pass
  `--shm-size`**: podman errors with *"cannot set shmsize when running in the host
  IPC Namespace"*, and it is redundant on docker. If you instead isolate IPC (drop
  `--ipc=host`), then add `--shm-size=16g` — one or the other, never both.
- **NUMA / socket**: one instance is pinned to **one socket plus its memory** --
  CPU bind + `--cpuset-mems` (container) / `numactl --membind` (conda), with KV sized
  from that socket's local RAM. On a dual-socket host `cpu_tune.py` picks a free socket
  by load and `warning`s if both are busy. NPS2/NPS4 (multi-node socket) gets an
  `nps_note` that finer per-node binding could add more.
- **Rootless podman + `--cpuset-cpus`/`--cpuset-mems`**: these are cgroup limits and
  may be **ignored or rejected** on rootless podman without cpuset cgroup delegation
  (cgroup v1, or v2 without the controller delegated). This is **not fatal**: CPU
  thread binding still applies via `VLLM_CPU_OMP_THREADS_BIND` inside the container;
  only the container-level memory pin is lost (reduced NUMA locality). If the run
  errors specifically on the cpuset flags, drop them and proceed -- do not treat it
  as a launch failure.
- **HF cache mount**: the default mounts `~/.cache/huggingface`. If `HF_HOME` points
  elsewhere (common on shared hosts, e.g. `/proj/.../vllm`), mount **that** path to
  `/root/.cache/huggingface` instead, or the model re-downloads inside the container.
- **Container name reuse**: a leftover `vllm-epyc` from a prior run makes `run` fail
  with "name already in use" -- Step 6 clears it first with `$RT rm -f vllm-epyc`.
