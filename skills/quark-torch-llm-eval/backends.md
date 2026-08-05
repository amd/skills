# Backends: Docker Setup + Serving Commands

Docker container setup and serving commands for all three backends (vLLM, ATOM, SGLang) on AMD ROCm.

## Docker Setup (Common ROCm Flags)

**Flags required by ROCm (do not change):**

- `--device=/dev/kfd --device=/dev/dri --device=/dev/mem`: expose GPUs
- `--cap-add=SYS_PTRACE --security-opt seccomp=unconfined`: permission set required by the ROCm kernel driver

**Flags you may need to change:**

- `--name`: container name. Recommend `eval-<hf_id_safe>-<username>-<date>` (replace slashes with dashes, username from `whoami`). Example: `eval-amd-MiniMax-M2.7-mxfp4-jiaxwang-0427`
- `-v <host>:<container>`: mount the top-level directory of the user-provided `${MODEL_PATH}` at the same path inside the container (e.g. `MODEL_PATH=/scratch/.../foo` → `-v /scratch:/scratch`). Default to `-v /shareddata:/shareddata` only when no model path was given.
- `--shm-size`: vLLM multi-process communication needs shared memory. Default `64g`; MoE models may need `128g`

**Container lifecycle:**

```bash
docker pull <image>     # pull first to avoid blocking during run
docker run [...]        # see backend-specific templates below
docker exec -it <container-name> bash    # enter container
```

**Private image authentication:** if `docker pull` fails with `unauthorized` / `authentication required` / `denied`, the image requires login. Detect the registry from the image name and guide the user:

| Image name pattern | Registry | Login command |
|---|---|---|
| `<name>/<repo>` (no dots) | Docker Hub | `docker login` |
| `<host>/<repo>` (host has dots) | Private registry | `docker login <host>` |
| `<account>.dkr.ecr.<region>.amazonaws.com/...` | AWS ECR | `aws ecr get-login-password --region <region> \| docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com` |

After the user confirms they have logged in, retry `docker pull`. Do not proceed to `docker run` until the pull succeeds.

Record `container_name` and `image` in `$EVAL_STATE_DIR/current-eval.yaml`.

**Cleanup (after evaluation, on request only):**

```bash
docker stop <container-name>
docker rm <container-name>
```

Do not clean up by default — the user may want to keep the container for reproduction or comparison.

---

## vLLM

**Default image:** `vllm/vllm-openai-rocm:latest` (see SKILL.md Phase 1 for full image priority chain)

### Docker run template

```bash
docker run -it -d \
  --entrypoint /bin/bash \
  --ipc=host \
  --shm-size=64g \
  --network=host \
  --name=<container-name> \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mem \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /shareddata:/shareddata \
  -v ${EVAL_STATE_DIR}:${EVAL_STATE_DIR} \
  <image>
```

The `-v ${EVAL_STATE_DIR}:${EVAL_STATE_DIR}` bind mount lets the in-container server write its log directly to the canonical state path (SKILL.md L134-140), so docker and host runtimes share the same log location with no symlink trick.

### Serving command

```bash
HIP_VISIBLE_DEVICES=${GPUS} vllm serve ${MODEL_PATH} --tensor-parallel-size ${TP} --port ${PORT:-8080}
```

---

## Cross-Backend Launch Adaptation Rules

These apply to **every** backend (vLLM, ATOM, SGLang). They cover only adaptations YOU make on top of the source command (model card / Recipes / Cookbook). Preserve all other flags from the source command verbatim.

| Change | Rule |
|--------|------|
| Replace base/quantized model HF id → local `${MODEL_PATH}` | **Always** — run the user's local copy, never re-download |
| Adjust tensor-parallel size (`--tensor-parallel-size` / `-tp` / `--tp`) | Use the source command's value if it fits; otherwise the smallest TP that satisfies `num_kv_heads % TP == 0` and `aggregate_free_vram >= model_size * 1.3` |
| Set `HIP_VISIBLE_DEVICES=${GPUS}` | **Always** — `${GPUS}` is the `hip_visible_devices` field from `current-eval.yaml` (already HIP indices, comma-joined). **Never** read `chosen_gpu_rocm` — those are rocm-smi indices and map to the wrong physical GPUs (`HIP_VISIBLE_DEVICES` expects HIP indices, derived from KFD node order). See `model-inspection.md` §4 for the rocm-smi → HIP mapping. |
| Set `--port ${PORT}` | If the source command omits a port, pick one (default 8080); if 8080 is busy, fall through 8081…8090 |

**Reactive-only flags** — do not add preemptively, only when an error in `troubleshooting.md` instructs you to:

- `--enforce-eager`, `--max-model-len`, `VLLM_ROCM_USE_AITER=1`, `--trust-remote-code`, `chat_template_kwargs`, `--reasoning-parser`, etc.

The error-signature table in `troubleshooting.md` is the authoritative trigger for each of those. Do not stack flags speculatively.

---

## ATOM

ATOM (AiTer Optimized Model) is AMD's lightweight LLM inference engine built on [AITER](https://github.com/ROCm/aiter) kernels. Repository: [github.com/ROCm/ATOM](https://github.com/ROCm/ATOM).

**Default image:** `rocm/atom-dev:latest` (nightly, recommended). Stable: `rocm/atom:latest`. (See SKILL.md Phase 1 for full image priority chain.)

### Docker run template

```bash
docker run -it -d \
  --entrypoint /bin/bash \
  --network=host \
  --name=<container-name> \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=16G \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /shareddata:/shareddata \
  -v ${EVAL_STATE_DIR}:${EVAL_STATE_DIR} \
  rocm/atom-dev:latest
```

Note: ATOM's docker template differs from vLLM — uses `--group-add video`, `--ulimit memlock=-1`, `--ulimit stack=67108864` instead of `--privileged`.

### Serving commands

**Basic:**

```bash
python -m atom.entrypoints.openai_server --model ${MODEL_PATH} --kv_cache_dtype fp8 -tp ${TP} --port ${PORT:-8080}
```

**With MTP speculative decoding (~60% throughput improvement):**

```bash
python -m atom.entrypoints.openai_server --model ${MODEL_PATH} --kv_cache_dtype fp8 -tp ${TP} --port ${PORT:-8080} \
  --method mtp --num-speculative-tokens 3
```

### Key parameters

- `--kv_cache_dtype fp8`: always recommended for memory efficiency
- `-tp <N>`: tensor parallelism
- `--method mtp --num-speculative-tokens 3`: MTP speculative decoding (best throughput/latency tradeoff)
- `--num-speculative-tokens 1`: more conservative MTP with lower overhead
- `--trust-remote-code`: required for some models (e.g., Kimi-K2)

### ATOM as vLLM plugin

ATOM can run as a vLLM out-of-tree plugin. When `atom` is installed alongside `vllm`, `vllm serve` automatically loads ATOM's optimized kernels.

```bash
pip install amd-aiter
git clone https://github.com/ROCm/ATOM.git && pip install ./ATOM
# Then just use vllm serve as normal — ATOM hooks are auto-activated
vllm serve ${MODEL_PATH} --tensor-parallel-size ${TP}
```

To disable the plugin: `export ATOM_DISABLE_VLLM_PLUGIN=1`

### ATOM Recipes

Per-model deployment guides live in the `recipes/` directory of the [ROCm/ATOM](https://github.com/ROCm/ATOM) repo. **List the current set at runtime — do not rely on a hardcoded list (it goes stale within weeks).**

```bash
# List available recipe names
curl -fsSL https://api.github.com/repos/ROCm/ATOM/contents/recipes \
  | python3 -c "import sys,json; [print(i['name']) for i in json.load(sys.stdin) if i['name'].endswith('.md')]"

# Fetch one
curl -fsSL https://raw.githubusercontent.com/ROCm/ATOM/main/recipes/${RECIPE_NAME}
```

If a recipe matching the model family is found (e.g., `DeepSeek-R1.md` for a DeepSeek-V3 quant), use it as the Phase 1 launch-command source.

### Tips

- Set `AITER_LOG_LEVEL=WARNING` to suppress aiter kernel log noise
- Clear compile cache before restarting: `rm -rf /root/.cache/atom/*`
- First-time execution takes ~10 minutes for model compilation
- Performance dashboard: [rocm.github.io/ATOM/benchmark-dashboard](https://rocm.github.io/ATOM/benchmark-dashboard/)

### Quantization support

ATOM auto-detects quantization configs from HuggingFace model configs: FP8, MXFP4, INT8, INT4.

---

## SGLang

**Default image:** `lmsysorg/sglang:v0.5.11-rocm720-mi35x` (See SKILL.md Phase 1 for full image priority chain.)

> **Status — partial support.** Launch templates and the readiness/smoke flow apply. `troubleshooting.md` currently covers vLLM-only error signatures; SGLang-specific failures fall through to "search GitHub Issues". SGLang-specific troubleshooting will be added in a follow-up.

### Serving command

```bash
python -m sglang.launch_server --model-path ${MODEL_PATH} --tp ${TP} --port ${PORT:-8080}
```

### Docker run template

Use the common ROCm docker template (same shape as the vLLM section; replace the `<image>` line with the SGLang default above). SGLang Cookbook: `https://lmsysorg.mintlify.app/cookbook/`

---

## Host (No-Docker) Launch Commands

Use when `runtime=host` is selected in Phase 0.5. ATOM is docker-only and has no host variant.

**vLLM (host):**

```bash
HIP_VISIBLE_DEVICES=${GPUS} vllm serve ${MODEL_PATH} --tensor-parallel-size ${TP} --port ${PORT:-8080}
```

**SGLang (host):**

```bash
HIP_VISIBLE_DEVICES=${GPUS} python -m sglang.launch_server --model-path ${MODEL_PATH} --tp ${TP} --port ${PORT:-8080}
```

Wrap either of those in the host-mode launch template from `SKILL.md` Phase 3 — the `( ... > "${EVAL_STATE_DIR}/vllm-server.log" 2>&1 ) &` form that captures the PID into `${EVAL_STATE_DIR}/server.pid`. The Cross-Backend Launch Adaptation Rules and reactive-only flag policy above apply identically — only the `docker run` / `docker exec` plumbing drops out.

## Readiness Poll Template

Poll health endpoint every 5s with process liveness check. If the server process exits early (bad arguments, import error), break immediately instead of waiting 300s.

**Docker runtime:**

```bash
# Required vars: CONTAINER, PORT (export before running)
: "${CONTAINER:?must export CONTAINER}"
: "${PORT:?must export PORT}"
for i in $(seq 1 60); do
  if ! docker exec "${CONTAINER}" pgrep -f "vllm serve|sglang.launch_server|atom.entrypoints" >/dev/null 2>&1; then
    echo "PROCESS DEAD after $((i*5)) seconds"
    tail -30 "${EVAL_STATE_DIR}/vllm-server.log"
    break
  fi
  if docker exec "${CONTAINER}" curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "READY after $((i*5)) seconds"; break
  fi
  sleep 5
done
```

**Host runtime variant:**

```bash
# Required vars: PORT, EVAL_STATE_DIR (export before running)
: "${PORT:?must export PORT}"
: "${EVAL_STATE_DIR:?must export EVAL_STATE_DIR}"
SERVER_PID=$(cat "${EVAL_STATE_DIR}/server.pid")
for i in $(seq 1 60); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "PROCESS DEAD after $((i*5)) seconds"
    tail -30 "${EVAL_STATE_DIR}/vllm-server.log"
    break
  fi
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "READY after $((i*5)) seconds"; break
  fi
  sleep 5
done
```

Process dead → read log tail, enter Phase 5. 300s timeout → read log tail, enter Phase 5.

## Smoke Test Templates

Required vars: `PORT`, `MODEL_PATH` (export before running).

**Step 0 — verify model name:** `curl -s "http://localhost:${PORT}/v1/models"` — confirm served model name matches what you will use in eval commands.

**Chat models:**

```bash
curl "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_PATH}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of China?\"}],\"max_tokens\":100}"
```

**Base models:**

```bash
curl "http://localhost:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_PATH}\",\"prompt\":\"The capital of China is\",\"max_tokens\":50}"
```

**Both gates must pass:**

1. HTTP 200 + JSON has `choices[0].message.content` (chat) or `choices[0].text` (completions)
2. Content is reasonable: not empty, not single-character repetition, not gibberish, topically relevant

Either gate fails → enter Phase 5.
