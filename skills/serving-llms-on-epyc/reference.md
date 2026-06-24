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
| `--ipc=host` | vLLM workers use host IPC / shared memory |
| `--shm-size=16g` | vLLM needs a large `/dev/shm`; the 64MB default is too small |
| `--network=host` | expose the served port directly (or use `-p <port>:<port>`) |
| `--cpuset-cpus` | (multi-socket) restrict the container to socket 0's CPUs; from `cpu_tune.py`. No `--cpuset-mems` -- no memory binding by default |
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

Default policy (the same for NPS1/NPS2/NPS4): a single instance uses **socket 0's
whole CPU with no memory binding**. `scripts/cpu_tune.py` derives:
- `VLLM_CPU_OMP_THREADS_BIND` = the physical cores of socket 0 (one thread per
  physical core; SMT siblings do not help vLLM CPU). vLLM sets `OMP_NUM_THREADS`
  itself from this list, so we don't.
- `VLLM_CPU_KVCACHE_SPACE` (GB) = `min(mem*kv_frac, mem-16)`; on <=32GB hosts, `mem*0.5`.
- `container_cpuset` = `--cpuset-cpus=<socket 0 cpus>` (no `--cpuset-mems`) for the
  container path on a multi-socket host. The conda path needs nothing extra -- the
  bind env var binds the threads.

Not set: `OMP_NUM_THREADS` (vLLM derives it from the bind) and
`VLLM_CPU_NUM_OF_RESERVED_CPU` (vLLM has its own default when unset).

When socket 0 spans multiple NUMA nodes (NPS2/NPS4), `cpu_tune.py` emits a
`perf_note`: the simple default leaves some performance on the table versus optimal
per-NUMA-node binding (one instance per node, memory bound). That tuning is out of
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
Without `--shm-size=16g` (or `--ipc=host`), vLLM workers fail to allocate shared
memory at startup.

**RAM is the ceiling, not VRAM**
CPU serving keeps weights + KV cache in system RAM. `estimate_memory.py` checks
`weights + KV(max_model_len x num_prompts) + reserve <= RAM` (reserve default
16 GB, `--reserve-gb`). It exits 1 when it does not fit and prints
`suggested_max_model_len` + an `action` to reduce and retry. Weights come from
HF file sizes (`.safetensors` or legacy `.bin`); `--weight-gb` overrides when a
model has no metadata. KV cache is bf16-only on zentorch CPU (no fp8 KV), so the estimate always uses 2 bytes/element.

**NUMA cross-node traffic**
On a 2-socket EPYC, an unpinned instance spreads threads across both sockets and
pays cross-socket latency. The default keeps one instance on **socket 0's CPUs**
(`cpu_tune.py` -> `VLLM_CPU_OMP_THREADS_BIND`, plus `--cpuset-cpus` for the
container), with **no memory binding**. On NPS2/NPS4, `cpu_tune.py` notes that
optimal per-NUMA-node binding could add performance; the base recipe doesn't do it.
