# serving-llms-on-epyc -- Reference

## Table of Contents
1. [Runtime selection](#runtime-selection)
2. [Container run flags (CPU)](#container-run-flags-cpu)
3. [Precision and modality](#precision-and-modality)
4. [CPU sizing](#cpu-sizing)
5. [Known quirks](#known-quirks)

---

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
`data/epyc.json`; read it, never hardcode it.

## Precision and modality

| Dtype | EPYC (Zen) | Notes |
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
