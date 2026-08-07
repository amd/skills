# Model Inspection

Reusable references for reading model metadata, probing GPU state, computing TP candidates, and writing the evaluation state file.

## 1. Input Validation + hf_id Inference

**model_path is provided by the user — never search the local filesystem for model files.** Using `find` / `ls` / `locate` to search for models is strictly forbidden.

After the user provides a local path, infer hf_id from the path:

| Local path | Inferred hf_id |
|---|---|
| `/shareddata/amd/MiniMax-M2.7-mxfp4` | `amd/MiniMax-M2.7-mxfp4` |
| `/data/zai-org/GLM-5` | `zai-org/GLM-5` |
| `/models/Qwen/Qwen2.5-7B` | `Qwen/Qwen2.5-7B` |

Path matching `/<root>/<org>/<repo>` → infer hf_id as `<org>/<repo>`, build `https://huggingface.co/<hf_id>` for model card fetch.

Path does not match this pattern → mark hf_id as `N/A`, try to get it from the `_name_or_path` field in config.json.

User gave only hf_id without local path → confirm existence on HuggingFace, then fetch config remotely (see below). Do not block on asking for a local path — collect all available info first, include download as part of the unified confirmation.

**Remote config fetch (hf_id-only scenario):**
When `model_path` is not yet available, read config files directly from HuggingFace API:

```bash
curl -sL https://huggingface.co/<hf_id>/raw/main/config.json
curl -sL https://huggingface.co/<hf_id>/raw/main/tokenizer_config.json
```

Parse these the same way as local files. This enables full Phase 0 info collection without requiring a download first.

## 2. HuggingFace Fetch Rules

Applies to every HuggingFace HTTP request in this skill (Phase 0 config fetch, Phase 1 model card / sibling variant search, Phase 6 paper extraction).

1. **`curl -fsSL`** (fail on HTTP errors). `curl -sL` swallows 401 HTML and the downstream JSON parser then crashes with a misleading error — never use `-sL` against HF endpoints.
2. **`${HF_ENDPOINT:-https://huggingface.co}`** as the base URL — supports internal mirrors and firewall bypasses.
3. **Token** read from `~/.hf_token` first, then `$HF_TOKEN` env var.

```bash
HF_BASE="${HF_ENDPOINT:-https://huggingface.co}"
HF_TOKEN="${HF_TOKEN:-$(cat ~/.hf_token 2>/dev/null)}"
curl -fsSL "${HF_BASE}/${HF_PATH}" \
  ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"}
```

**On HTTP 401 / 403**: the model is gated or private. Prompt the user to write a token to `~/.hf_token` (takes effect immediately, no restart):

```bash
echo 'hf_xxx...' > ~/.hf_token
```

Then retry the same `curl -fsSL` invocation. Do not silently fall back to a different model — gated access is a user-decision boundary, not a recoverable error.

**On HTTP 404**: treat as "not found", not a fatal error. Walk to the next level of the Phase 1 priority chain (e.g., quantized card 404 → try sibling variant → try base card).

## 3. Reading config.json

Read `<model_path>/config.json` and record the following fields. Do not `cat` the entire file, as `quantization_config` in quantized models can be thousands of lines.

**Execution rule**: write the Python code to a temp file (e.g., `/tmp/parse_config.py`) and run `python3 /tmp/parse_config.py`. Do not use `python3 -c "..."` with multiline code — inline `#` comments trigger Claude Code's security heuristic and cause unnecessary permission prompts.

The full bash invocation (substitute `${MODEL_PATH}` before running):

```bash
cat > /tmp/parse_config.py <<'PYEOF'
import json, os, sys
mp = os.environ['MODEL_PATH']
c = json.load(open(f'{mp}/config.json'))
tc = c.get('text_config', {})  # VLMs store params inside text_config
keep = ['architectures', 'model_type', 'num_hidden_layers',
        'num_attention_heads', 'num_key_value_heads',
        'hidden_size', 'intermediate_size',
        'num_local_experts', 'num_experts_per_tok',
        'n_routed_experts', 'n_shared_experts',
        'max_position_embeddings', 'torch_dtype']
for k in keep:
    val = c.get(k) or tc.get(k, 'N/A')
    print(f'{k}: {val}')
if 'auto_map' in c:
    print('auto_map: yes (requires --trust-remote-code)')
qc = c.get('quantization_config')
if qc:
    print(f'quant_method: {qc.get("quant_method", "?")}')
    gc = qc.get('global_quant_config', {})
    w = gc.get('weight', {})
    print(f'weight_dtype: {w.get("dtype", "?")}, group_size: {w.get("group_size", "?")}')
PYEOF
MODEL_PATH=${MODEL_PATH} python3 /tmp/parse_config.py
```

**VLM note**: Vision-Language Models (Kimi-K2.5, InternVL, Qwen-VL, etc.) store text backbone params inside a `text_config` sub-dict. The snippet above falls back to `text_config` automatically.

These facts are used to:

- Filter TP candidates (must evenly divide `num_key_value_heads`)
- Detect MoE models (`num_local_experts` or `n_routed_experts` present — DeepSeek-family uses `n_routed_experts`)
- Detect `--trust-remote-code` requirement (`auto_map` present in config.json)
- Identify attention type (MLA typically in DeepSeek-family models, check `architectures` / `attn_type_list`)

## 4. Model Type Detection (base vs chat)

Determines whether evaluation uses `local-completions` or `local-chat-completions`.

Wrap into a temp script per the §3 execution rule:

```bash
cat > /tmp/check_chat_template.py <<'PYEOF'
import json, os
tc_path = os.path.join(os.environ['MODEL_PATH'], 'tokenizer_config.json')
if os.path.exists(tc_path):
    tc = json.load(open(tc_path))
    has_chat = 'chat_template' in tc and tc['chat_template']
    print(f'has_chat_template: {bool(has_chat)}')
else:
    print('has_chat_template: N/A (no tokenizer_config.json)')
PYEOF
MODEL_PATH=${MODEL_PATH} python3 /tmp/check_chat_template.py
```

Decision logic (check in order, first match wins):

1. `tokenizer_config.json` has `chat_template` → **chat**
2. Model name contains `-Instruct` / `-Chat` / `-it` → **chat**
3. `architectures` contains `ForConditionalGeneration` (e.g., `KimiK25ForConditionalGeneration`, `InternVLForConditionalGeneration`) → **chat** (VLMs are inherently chat/instruct models)
4. `auto_map` exists in config.json (custom code model) → check model card for guidance on chat vs base
5. None of the above → **base**

Record as `model_type_for_eval: chat | base`.

**Important**: model type determines the *default* eval template, but the model card's eval command takes priority. Some chat models achieve better benchmark accuracy with `local-completions` + few-shot (e.g., Kimi-K2.5 gsm8k uses `local-completions` with `--num_fewshot 5`). See eval-frameworks.md for the priority rule.

## 5. GPU State from rocm-smi

**Timing**: GPU state is checked at execution time (Phase 3 pre-launch), not during info collection (Phase 0), because GPU availability changes between planning and execution.

### Step 1: Collect raw data

```bash
rocm-smi --showuse                # GPU utilization %
rocm-smi --showmeminfo vram       # exact VRAM usage in bytes (Total / Used per GPU)
rocm-smi --showpidgpus            # per-PID GPU device assignment
rocm-smi --showproductname | grep "Card Series" | head -1   # GPU model
```

**Do NOT use `--showpids` for per-GPU process mapping.** Its `GPU(s)` column shows the *number* of GPUs a process uses, not the GPU index. Use `--showpidgpus` instead — it lists the exact device index each PID occupies.

**Why `--showmeminfo vram` instead of `--showmemuse`**: `--showmemuse` reports VRAM% as a rounded integer. On a 288 GB card, rounding can hide tens of GB of actual usage (e.g., VRAM% shows 0 while 120+ GB is consumed). `--showmeminfo vram` returns exact byte counts — compute free memory as `Total - Used` and compare directly against the model's size requirement.

### Step 2: Understand the two index spaces

AMD ROCm has **two** GPU index spaces that may differ:

| Index space | Source | Used for |
|---|---|---|
| rocm-smi device index (GPU[N]) | KFD node order | `rocm-smi` commands (`--showmeminfo vram`, `--showuse`, `--showpidgpus`) |
| HIP device index | KFD GPU node ascending sort | `HIP_VISIBLE_DEVICES` env var |

**`--showpidgpus` device index = rocm-smi device index.** The device numbers reported by `--showpidgpus` use the same numbering as `GPU[N]` in all other rocm-smi commands. No intermediate DRM renderD mapping is needed.

**`HIP_VISIBLE_DEVICES` uses HIP indices**, which are assigned by sorting GPU KFD node numbers in ascending order. This is **NOT** the same as PCI bus address order — do not sort by PCI bus. The KFD node number for each GPU is shown in the `Node` column of `rocm-smi` concise output (run `rocm-smi` with no flags).

### Step 3: Identify occupied GPUs from --showpidgpus

`rocm-smi --showpidgpus` output looks like:

```text
PID 948985 is using 1 DRM device(s):
0
PID 4140639 is using 1 DRM device(s):
5
```

The number after "DRM device(s):" is the **rocm-smi device index** (same as GPU[N]). Parse the output and build the set of occupied device indices. Any device index NOT in this set has no active process.

Example: occupied = {0, 1, 2, 3, 5, 6, 7} → device 4 is free.

### Step 4: rocm-smi → HIP index mapping

HIP enumerates GPUs by **KFD node number ascending**, not by PCI bus address. The `Node` column is visible in `rocm-smi` concise output (no flags).

Use `rocm-smi --json` (stable schema across ROCm versions) and wrap per the §3 execution rule. Fail loud if the mapping is empty — `rocm-smi` text format drift would silently produce `{}` and downstream `HIP_VISIBLE_DEVICES` would be wrong.

```bash
cat > /tmp/parse_gpu_mapping.py <<'PYEOF'
import json, subprocess, sys

# rocm-smi --json --showuniqueid emits {"card0": {"Node ID": "3", ...}, ...}
out = subprocess.run(['rocm-smi', '--json', '--showuniqueid'],
                     capture_output=True, text=True, check=True).stdout
data = json.loads(out)

device_node = {}
for key, fields in data.items():
    if not key.startswith('card'):
        continue
    dev = int(key.removeprefix('card'))
    node = fields.get('Node ID') or fields.get('node_id')
    if node is None:
        continue
    device_node[dev] = int(node)

if not device_node:
    sys.exit("ERROR: rocm-smi --json returned no card/Node ID entries — "
             "ROCm-SMI schema may have drifted. Check `rocm-smi --json` output manually.")

# HIP index = position when sorting GPU KFD nodes ascending
nodes_sorted = sorted(device_node.items(), key=lambda x: x[1])
print("rocm_smi_index -> hip_index (sorted by KFD node):")
for hip_idx, (dev, node) in enumerate(nodes_sorted):
    print(f"  rocm-smi GPU[{dev}] (node {node}) -> HIP device {hip_idx}")
PYEOF
python3 /tmp/parse_gpu_mapping.py
```

If your ROCm build does not include `--json`, fall back to the legacy regex parser **and** add an explicit check that the resulting mapping is non-empty before proceeding.

After identifying idle GPUs by rocm-smi index, convert to HIP indices using this mapping before setting `HIP_VISIBLE_DEVICES`.

### Step 5: Apply idle criteria

A GPU is idle only when **both** conditions hold:

1. **No process on this GPU**: `--showpidgpus` shows no PID using this device index
2. **VRAM free ≥ 95% of VRAM total**: computed from `--showmeminfo vram` bytes

**Why both checks are needed:**

- A process may hold a GPU with minimal VRAM (e.g., just initialized, not yet loaded model) — VRAM looks free but GPU is occupied
- A zombie GPU context may hold VRAM after the process has exited — no PID visible but VRAM is not free

**Additional check — GPU utilization**: `GPU% ≤ 5%` from `--showuse`. A GPU with high utilization but low VRAM may be doing compute (e.g., CPU-offloaded inference). This is a supplementary signal, not a primary filter.

**Zombie detection**: if `--showmeminfo vram` shows high VRAM usage but `--showpidgpus` shows no process → likely a driver-side zombie context. Report this to the user rather than treating the GPU as idle. The GPU may need `sudo rocm-smi --gpureset -d <gpu>` to reclaim VRAM.

**Per-GPU free memory**: for each idle GPU, record the free VRAM in GB (`(Total - Used) / 1e9`). This is used in §6 to verify the TP candidate has enough aggregate free memory for the model.

Record the idle GPU list with free memory (e.g., `free_gpus: [{rocm: 4, hip: 5, free_gb: 287.6}]`) in the state file. The `hip_visible_devices` field uses the **HIP indices**, not the rocm-smi indices.

## 6. TP Candidate Filtering

Candidate TP values must satisfy all constraints:

| Constraint | Source |
|---|---|
| Evenly divides `num_key_value_heads` | GQA splits KV heads across cards |
| Evenly divides `num_attention_heads` | Q heads must also split |
| ≤ number of free GPUs | Physical limit |
| Sum of free VRAM across chosen GPUs ≥ model size × 1.3 | Capacity floor (uses actual free bytes from §5, not total card size) |

Model size estimate: `du -sh` on the model directory for weight bytes; MXFP4 is ~0.5 byte/param already reflected in file size, no conversion needed.

**Prefer the smallest TP satisfying all constraints**: less communication overhead, simpler configuration. Example: 125GB model + 288GB single card → TP=1. If a recipe provides a TP value, prefer it over the auto-inferred minimum — the recipe TP is author-verified.

## 7. State File Template

Write all inferred results to `$EVAL_STATE_DIR/current-eval.yaml` (path is set in SKILL.md Phase 0.5 and persisted to `/tmp/eval-state/.last-session-path-<user>`). The file is YAML — every later phase reads it with `python3 -c "import yaml; yaml.safe_load(open(path))"` rather than ad-hoc string splitting.

```yaml
eval_state_dir: /tmp/eval-state/<user>/<timestamp>     # first field — pinned by Phase 0.5
model_path: /shareddata/amd/MiniMax-M2.7-mxfp4
hf_id: amd/MiniMax-M2.7-mxfp4          # quantized model HF id — primary model card source
base_hf_id: TBD                        # base model HF id — fallback card + Recipes lookup key
arch: MiniMaxM2ForCausalLM
model_type: minimax_m2
is_moe: true                           # check both num_local_experts and n_routed_experts
num_experts: 256
num_kv_heads: 8
max_position_embeddings: 204800
quant: {method: quark, dtype: mxfp4, group_size: 32}
model_size_gb: 125
model_type_for_eval: chat              # chat | base — determines lm-eval template
requires_trust_remote_code: false      # true if auto_map in config.json or custom_code tag on HF
is_vlm: false                          # true if architectures contains ForConditionalGeneration

free_gpus:
  - {rocm: 4, hip: 5, free_gb: 287.6}
gpu_mapping: {rocm0: hip1, rocm1: hip3, rocm2: hip2, rocm3: hip0, rocm4: hip5}
gpu_model: AMD Instinct MI355X (288GB)
chosen_gpu_rocm: 4                     # rocm-smi index (for monitoring via rocm-smi, --showpidgpus)
chosen_gpu_hip: 5                      # HIP index (for HIP_VISIBLE_DEVICES)
chosen_tp: 1                           # auto-inferred; may be overridden by recipe
hip_visible_devices: "5"               # uses HIP index, NOT rocm-smi index

backend: vLLM
ref_url: TBD
benchmark: gsm8k
image: TBD
launch_cmd: TBD
eval_cmd: TBD
launch_cmd_source: TBD                 # quant_model_card | similar_quant_card | base_model_card | recipes | template
eval_cmd_source: TBD                   # quant_model_card | similar_quant_card | base_model_card | recipes | lm-eval_repo | inferencex | template
reference_model: N/A                   # sibling quant variant used as reference (e.g., amd/Model-fp8), if any

source_edits: []                       # files modified during Phase 5; each entry: {path, backup, original_value}

reference_scores:
  gsm8k:
    value: 89.84
    source: "arxiv:2505.09388 Table 4"
    setting: "base model, 4-shot CoT"
    model_variant: "Qwen3-8B Base"

env_preflight:
  docker_run: ok                       # ok | denied
  state_dir_writable: ok               # ok | fallback_to_cwd
  curl: ok
  python3: ok
```

**Why `base_hf_id` exists**: the local model is a quantized variant (e.g., `amd/MiniMax-M2.7-mxfp4`), but the vLLM Recipes index uses the base model (e.g., `MiniMaxAI/MiniMax-M2.7`). The recipe is fetched using base_hf_id; the launch command **replaces** base_hf_id with the local model_path (since the local copy is quantized). Recording both IDs explicitly prevents missed replacements.

Subsequent phases update this file whenever they change a value.

## 8. Unified Confirmation Template

See `templates.md` §1 for the full template, required fields, and display rules.
