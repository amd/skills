# serving-llms-on-epyc -- Reference

## Table of Contents
1. [Hardware support](#hardware-support)
2. [Runtime and stack compatibility](#runtime-and-stack-compatibility)
3. [Runtime selection](#runtime-selection)
4. [Container run flags (CPU)](#container-run-flags-cpu)
5. [Precision and modality](#precision-and-modality)
6. [Client endpoints and parameters](#client-endpoints-and-parameters)
7. [CPU sizing](#cpu-sizing)
8. [Known quirks](#known-quirks)

---

## Hardware support

This recipe supports the **AMD EPYC 9000 server series** for now: Genoa (9004),
Turin (9005), and 6th Gen [Venice (9006)](https://ir.amd.com/news-events/press-releases/detail/1294/aai-2026-amd-delivers-full-stack-compute-for-the-agentic-ai-era)
(launched at Advancing AI 2026). `scripts/detect.py` reports only these three
generations as `is_supported_epyc: true`.

AVX-512 is necessary but not sufficient for this support policy. Other EPYC parts
-- Bergamo and Siena, and the AM5 EPYC 4004/4005 -- expose the required ISA but are
outside this skill's current 9000-series scope; the detector still names them but
reports `is_supported_epyc: false`. Do not infer support from AVX-512 alone.

The presence of AMD Instinct GPUs does not change CPU support. Use this skill
when the requested endpoint should execute on EPYC; use
`serving-llms-on-instinct` when it should execute on a GPU. Both serving engines
may coexist on the same host.

## Runtime and stack compatibility

Detecting a supported CPU is not the same as running a validated software stack.
`scripts/validate.py --generation <gen>` probes the **selected** runtime (the
container image when present, else the conda/host env) for its exact
`vllm`/`zentorch`/`torch` versions and the **active vLLM platform**, then reports
`compatibility.status`:

- `proceed` -- a Zen platform is active (zentorch acceleration on) and the stack is
  the validated default or a validated family.
- `blocked` (error) -- the stock `CpuPlatform` is active, so serving would run
  **without** zentorch. Two vLLM paths select a Zen platform: the in-tree
  `ZenCpuPlatform` (vLLM detects an AMD AVX-512 CPU with `zentorch` importable) and
  the out-of-tree `zentorch` plugin. If neither is active, fix the environment or
  use the pinned image; do not serve an unaccelerated CPU stack.
- `confirmation_required` (`requires_confirmation: true`) -- **Venice on a vLLM
  other than the pinned default**. AMD documents 6th Gen EPYC as a zentorch target,
  but this recipe has not validated Venice end-to-end on an off-default version.
  Venice on the pinned `vllm_version` proceeds with no warning; on any other
  version, stop, recommend the pinned image, and get an explicit user go/no-go.

The probe only runs once the image is local, so after a first `pull` re-run
`validate.py` to gate on the real stack rather than the tag. The container tag
pins the AMD-published integration stack; a conda env may differ, so
`check_model.py` should use the probed `stack.vllm` for that path.

## Runtime selection

`scripts/validate.py` resolves a runtime the **agent can drive
non-interactively** and reports it as `runtime` (the exact command prefix the
agent uses for `pull`/`run`/`stats`/`logs`). Preference order maximizes
agent-drivability with no human in the loop:

1. **docker** (direct) -- if `docker ps` exits 0 (user in the `docker` group /
   daemon reachable). No sudo. Best.
2. **podman** (rootless) -- no daemon, no sudo. Note: rootless podman needs a
   storage backend that supports its overlay; some networked/`/proj`
   filesystems reject the overlay `pivot_root` (the run fails even though
   `podman info` succeeds). On those hosts use docker or the conda path.
3. **sudo docker** -- only if `sudo -n docker ps` works (passwordless sudo). The
   agent can still drive it unattended; `runtime` comes back as `"sudo docker"`.
4. **conda/host** -- requires `import vllm, zentorch` in the active env.

If docker is installed but **none** of the above is agent-drivable (no docker
group, no passwordless sudo), `validate.py` returns `runtime: null`,
`runtime_agent_drivable: false`, and a **one-time** setup `fix`:
`sudo usermod -aG docker $USER && newgrp docker` (or a NOPASSWD sudoers entry).
This is one-time onboarding, not a per-serve command. After it, every serve is
fully agent-driven. The skill must not degrade into asking the user to paste
docker commands for each serve.

## Container run flags (CPU)

From `data/epyc.json`. Unlike the Instinct (GPU) skill there are **no**
`/dev/kfd`, `/dev/dri`, `--group-add`, or ROCm flags -- this is pure CPU.

| Flag | Why |
|---|---|
| `--ipc=host` | vLLM workers need a large `/dev/shm`; sharing the host IPC namespace provides it. **Do not also pass `--shm-size`** -- podman rejects the combination, and it is redundant on docker |
| `--shm-size=16g` | **only if you drop `--ipc=host`** (isolated IPC). The 64MB container default is too small for vLLM. Use one or the other, never both |
| `--network=host` | expose the served port directly (or use `-p <port>:<port>`) |
| `--cpuset-cpus` / `--cpuset-mems` | pin the container to the chosen socket's physical cores and its NUMA node(s); from `cpu_tune.py` |
| `-v ~/.cache/huggingface:/root/.cache/huggingface` | reuse the host model cache |

Image: `amdih/zendnn_zentorch:<tag>` -- the public vLLM + zentorch CPU image on
Docker Hub (no internal-registry access needed). The exact tag lives in
`data/epyc.json`; read it, never hardcode it. The image and `vllm_version` are
pinned together so `check_model.py` reads the registry for the runtime that will
actually serve the model. This reproducibility pin applies to the default
container recipe; it does not replace or modify an existing conda environment.

## Precision and modality

| Dtype | Supported EPYC server target | Notes |
|---|---|---|
| BF16 | Native (default) | throughput default |
| FP16 | Native | |
| FP32 | Native | slower; debugging only |
| WOQ int8/int4 | Supported by zentorch | per-channel / per-group; out of scope for the base recipe |

Modality: not gated by a static blocklist. `scripts/check_model.py` checks the
model's architecture against vLLM's model registry (pinned to `vllm_version`):
text **and** multimodal generation endpoints are allowed; pooling/embedding/
reranker and non-LLM architectures are rejected (not chat/completion endpoints).
A vLLM-supported multimodal arch may still hit a GPU-only kernel on CPU -- that
surfaces at load, where the no-retry rule applies.

## Client endpoints and parameters

`check_model.py` reports the endpoint the model actually supports so the handoff
matches reality instead of always assuming chat:

| Model | `chat_template.status` | `primary_endpoint` | Client call |
|---|---|---|---|
| Instruct/chat (ships a template) | `present` | `chat_completions` | `POST /v1/chat/completions` with `messages` |
| Base text (no template) | `absent` | `completions` | `POST /v1/completions` with `prompt` |
| Multiple named templates, no `default` | `ambiguous` | `completions` | completions now; chat needs `--chat-template`/a chosen name |
| Template unreadable (gated/offline) | `unknown` | `completions` | completions; chat also works if a template exists |
| Multimodal, no usable template | `absent`/`ambiguous` | none (`launchable: false`) | stop -- supply `--chat-template` or another model |

`/v1/chat/completions` applies the model's chat template to structured `messages`
and returns `choices[0].message.content`. `/v1/completions` takes a raw `prompt`
(no template) and returns `choices[0].text`. Never invent a chat template; only
enable chat when a real one is present or the user supplies `--chat-template`.

Request parameters worth surfacing to users:

| Parameter | Meaning |
|---|---|
| `max_tokens` | Output-token cap. `prompt_tokens + max_tokens` must be `<= --max-model-len`. |
| `temperature` | Randomness; `0` is deterministic/greedy. |
| `top_p` | Nucleus sampling; tune this **or** `temperature`, not both. |
| `stream` | `true` streams tokens over SSE instead of one blocking reply. |
| `stop` | String(s) that end generation early. |

The base URL always ends in `/v1`. The OpenAI Python SDK requires a non-empty
`api_key`, so pass a placeholder (e.g. `"EMPTY"`) when the server has no auth.
The model repo's `generation_config.json` can set sampling defaults, so pass
explicit values when determinism matters.

## CPU sizing

Policy: a single instance is pinned to **one socket plus its memory** (vLLM scales
poorly across sockets). `scripts/cpu_tune.py` derives:
- **Socket choice** (dual-socket): samples per-socket CPU busy% (~0.5s) and prefers a
  free socket -- both free → socket 0; one free → that one; both at/above
  `--busy-threshold` (default 15%) → `warning` and proceed on the least-busy. `--socket N`
  forces it. Single-socket → socket 0.
- `VLLM_CPU_OMP_THREADS_BIND` = the chosen socket's physical cores (SMT dropped). vLLM
  sets `OMP_NUM_THREADS` from this, so we don't.
- `VLLM_CPU_KVCACHE_SPACE` (GB) = `min(socket_ram*kv_frac, socket_ram-16)` -- sized from
  the **chosen socket's local RAM** so the KV pool stays on-socket (≤32GB → `*0.5`).
- Memory-bound pin: `container_cpuset` = `--cpuset-cpus=<cores> --cpuset-mems=<nodes>`;
  `conda_launch_prefix` = `numactl --cpunodebind=<nodes> --membind=<nodes>` (falls back to
  `taskset` CPU-only, or empty-with-note if neither tool exists).

Not set: `OMP_NUM_THREADS` (vLLM derives it from the bind) and
`VLLM_CPU_NUM_OF_RESERVED_CPU` (vLLM has its own default when unset).

When the chosen socket spans multiple NUMA nodes (NPS2/NPS4), `cpu_tune.py` emits an
`nps_note`: memory is bound across the socket's nodes, and finer per-node binding
(one instance per node) could add more. That tuning is out of
scope for the base recipe.

## Known quirks

**`--device cpu` removed (vLLM >= 0.20)**
`vllm serve` no longer accepts `--device cpu`; the zentorch plugin auto-selects
the CPU platform. Passing it -> `vllm: error: unrecognized arguments: --device cpu`.
Only pass it if `vllm serve --help` advertises it (older vLLM).

**`TORCHINDUCTOR_FREEZING=1` + `VLLM_USE_AOT_COMPILE` (VERIFIED)**
On vLLM 0.23.0 / zentorch 2.11.0.2 (EPYC 9454, facebook/opt-125m, 2026-06-23):
`TORCHINDUCTOR_FREEZING=1` alone crashes engine-core init with
`AssertionError: expected OutputCode, got function` (inductor codecache). Adding
`VLLM_USE_AOT_COMPILE=0` fixes it (healthy in ~99s). The only changed variable
between the failing and passing runs was `VLLM_USE_AOT_COMPILE`. Never set
`FREEZING=1` without `VLLM_USE_AOT_COMPILE=0`. The base recipe leaves both unset.

**`/dev/shm` too small**
vLLM workers need a large `/dev/shm` or they fail to allocate shared memory at
startup. The base recipe uses `--ipc=host` (shares the host's large shared memory).
**Do not combine `--ipc=host` with `--shm-size`** -- podman errors *"cannot set
shmsize when running in the host IPC Namespace"*, and it is redundant on docker. If
you drop `--ipc=host`, use `--shm-size=16g` instead -- one or the other, never both.

**RAM is the ceiling, not VRAM**
CPU serving keeps weights + KV cache in system RAM. `estimate_memory.py` checks
`weights + KV(max_model_len x num_prompts) + reserve <= RAM` (reserve default
16 GB, `--reserve-gb`). It exits 1 when it does not fit and prints
`suggested_max_model_len` + an `action` to reduce and retry. Weights come from
HF file sizes (`.safetensors` or legacy `.bin`); `--weight-gb` overrides when a
model has no metadata. KV cache is bf16-only on zentorch CPU (no fp8 KV), so the estimate always uses 2 bytes/element.

**NUMA cross-node traffic**
On a 2-socket EPYC, an unpinned instance spreads threads + memory across both sockets
and pays cross-socket latency. `cpu_tune.py` keeps one instance on **one socket plus
its memory**: CPU bind (`VLLM_CPU_OMP_THREADS_BIND` + `--cpuset-cpus`), memory bind
(`--cpuset-mems` / `numactl --membind`), and KV sized from that socket's local RAM so
the KV pool never lands on the other socket. The socket is chosen by load (free socket
preferred; warns if both busy). True multi-socket throughput = **multiple instances**
(one per socket) -- out of scope for this single-instance recipe.
