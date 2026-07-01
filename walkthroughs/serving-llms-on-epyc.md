# AMD Skills Walkthroughs: `serving-llms-on-epyc`

The goal of this skill is to teach your AI agent to bring up a vLLM OpenAI-compatible
endpoint on an **AMD EPYC CPU** host using the zentorch backend — detecting the CPU,
validating the environment, checking the model fits, sizing the runtime to the
hardware, launching, and verifying the endpoint responds.

**What you'll end up with:** a running `vllm serve` endpoint on your EPYC box (in a
Docker/Podman container, or a conda env), sized to a single socket and ready to answer
OpenAI `/v1/chat/completions` requests.

## Prerequisites

- An **AMD EPYC CPU with AVX-512 support** — i.e. **Zen4+ (Genoa / Bergamo / Siena / Turin) or newer**. This is CPU serving (no GPU required); AVX-512 is required for the zentorch CPU path, and `detect.py` reports it (`avx512`).
- A container runtime — **Docker** or **Podman** — *or* a conda env with `vllm` + `zentorch` installed.
- Enough host RAM for the model (weights + KV cache both live in RAM on CPU).
- A HuggingFace token in `HF_TOKEN` **only** for gated models (Llama, Gemma). The default model (Qwen3) needs none.
- **Node.js ≥ 18** — required by the `skills` CLI used in Step 2 (`npx skills ...`). Check with `node -v`; on older hosts install a newer Node (e.g. `conda create -n node20 -c conda-forge 'nodejs>=20'`).

## Step 1 - Understanding which skills are available

* Run `claude "Which skills can you see?" --model sonnet`. You should see a list of skills that does **not** include anything about serving LLMs on EPYC / CPU.
* Make sure there is no `AGENTS.md` file in your local folder.

## Step 2 - Enabling claude to see `serving-llms-on-epyc`

* Install the skill with the [`skills` CLI](https://github.com/vercel-labs/skills):

```bash
npx skills add amd/skills --skill serving-llms-on-epyc --agent claude-code
```

* Run `claude "Which skills can you see?" --model sonnet`. You should see a list of skills that now includes `serving-llms-on-epyc`.

## Step 3 - Running the skill

Run `claude --model sonnet` on your EPYC host with this prompt:

```
Serve Qwen/Qwen3-0.6B on this AMD EPYC box with vLLM and zentorch.
```

Claude should:

1. **Detect the CPU** — confirm it is AMD EPYC and read the generation (Genoa/Turin/…), AVX-512, physical cores, NUMA layout, and RAM.
2. **Validate the environment** — find an accessible runtime (Docker or Podman, else the conda path), check the image, `HF_TOKEN`, and RAM; report any perf-library advisories.
3. **Check vLLM supports the model** — verify the architecture against vLLM's model registry (it does not blanket-block multimodal; it rejects non-chat models like embeddings/rerankers).
4. **Check it fits host RAM** — weights + KV cache + headroom vs available RAM.
5. **Size the runtime to the hardware** — bind to one socket's physical cores, size the KV cache from that socket's local RAM, and bind memory to that socket (this is **single-socket serving**; vLLM scales poorly across sockets).
6. **Confirm the plan with you** — present a sized summary (model, path, precision, fit, CPU sizing, port) and wait for you to approve before launching.
7. **Launch and verify** — pull the public `amdih/zendnn_zentorch` image, run `vllm serve`, poll `/health`, and prove `/v1/chat/completions` works.

On any failure it reports the cause + logs and **stops** — it does not retry or start a debugging loop.

## Step 4 - Talk to the endpoint

Once Claude reports the endpoint is healthy, call it — use the **port from Claude's
connection table** (it uses `8000` by default):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}]}'
```

## Step 5 - (Optional) Going beyond

* **A real workload:** ask for a larger model once the flow is proven, e.g. *"Serve Qwen/Qwen3-8B ..."*. Claude re-checks the RAM fit and re-sizes.
* **Gated models:** `export HF_TOKEN=...` (and accept the model license on HuggingFace), then ask for `meta-llama/Llama-3.1-8B-Instruct`.
* **Pick a socket:** on a dual-socket box Claude picks a free socket by load; you can steer it (*"serve it on socket 1"*).

## Step 6 - (Optional) Try to get things done without AMD Skills

Remove the added skill and rerun the experiment above. The `skills` CLI installs a
copy under **both** `.claude/skills/serving-llms-on-epyc` **and**
`.agents/skills/serving-llms-on-epyc`, so delete both (otherwise the leftover copy
keeps the skill active and the comparison isn't clean). Without the skill, common
issues include:

* Passing `--device cpu` to `vllm serve` (removed in vLLM ≥ 0.20 with the zentorch plugin) — the server errors out on launch.
* Guessing at a container image or using a GPU/CUDA image instead of the public CPU `amdih/zendnn_zentorch` one.
* No hardware-aware sizing — spreading threads across both sockets and sizing the KV cache from whole-system RAM, so the KV pool spills cross-socket and throughput tanks.
* Launching a model that does not fit host RAM (or an embedding/reranker model that has no chat endpoint) and then looping on the failure.
* Providing a knowledge article instead of actually bringing up a working endpoint.
