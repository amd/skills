#!/usr/bin/env python3
"""
Plan a vLLM deployment on AMD Instinct: one command that does every lookup.

Detects the GPU, checks the blocklist, finds the model's recipe (exact,
base-name, legacy, or generic), picks a precision variant that fits, sizes
tensor parallelism and context, merges the environment, and prints the exact
`docker run` command with the decisions behind it.

Usage:
    python3 scripts/plan.py --model Qwen/Qwen3-8B
    python3 scripts/plan.py --model Qwen/Qwen3-8B --host user@hostname
    python3 scripts/plan.py --model openai/gpt-oss-120b --gpus 0,1 --port 8001
    python3 scripts/plan.py                      # default model

Output: JSON to stdout. "status" is one of
    ok          -- "command" is ready to run ("run_command" wraps it for --host)
    refuse      -- the model can't be served as an LLM endpoint here ("reason")
    does_not_fit -- not enough VRAM on this host even when quantized ("reason")
    error       -- couldn't finish: GPU detection failed or the HF Hub was unreachable ("reason")
Always relay "warnings" to the user. Exit code: 0 ok, 2 refuse, 3 does_not_fit, 1 error.

Env vars: ROCM_SSH_HOST / ROCM_SSH_USER / ROCM_SSH_PORT (as detect.py), HF_TOKEN.
"""

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
sys.path.insert(0, str(HERE))
import detect  # noqa: E402
import estimate_vram as ev  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
GENERIC_ARGS = ["--enable-auto-tool-choice", "--trust-remote-code", "--tool-call-parser", "hermes"]
# Highest precision first: prefer the least-quantized variant that fits.
PRECISION_RANK = {"bf16": 0, "fp16": 0, "fp8": 1, "int8": 2, "mxfp6": 3, "mxfp4": 4, "int4": 4}
UNSUPPORTED_PRECISIONS = {"nvfp4", "fp4"}  # NVIDIA-only formats; no ROCm kernels
_NVFP4_ID = re.compile(r"(nvfp4|[-_]fp4(\b|$))", re.I)
_TP_ARGS = {"--tensor-parallel-size", "-tp"}


def _fetch_with_retry(url, token=None, _fetch=ev._fetch):
    """estimate_vram's fetch, retried on rate limits and server errors (shared IPs hit HF's 429s)."""
    import time
    for delay in (2, 5, 10, 20):
        data, err = _fetch(url, token)
        if data is not None or not err or not re.match(r"HTTP (429|5\d\d)|timed out|URLError", err):
            return data, err
        time.sleep(delay)
    return _fetch(url, token)


ev._fetch = _fetch_with_retry  # every Hub call below goes through the retrying fetch


def transient(err):
    return bool(re.search(r"HTTP (429|5\d\d)|timed out|URLError|Temporary failure|Connection", err or ""))


def load(name):
    return json.loads((DATA / name).read_text())


def find_recipe(model_id, recipes):
    """(recipe, variant_name, how) for model_id, or (None, None, None)."""
    models = recipes["models"]
    lower = {k.lower(): k for k in models}
    key = lower.get(model_id.lower())
    if key:
        return models[key]["recipe"], None, "recipe"
    # The id may be one of a recipe's variant checkpoints (e.g. ...-FP8).
    for k, m in models.items():
        for vname, v in (m["recipe"].get("variants") or {}).items():
            if (v.get("model_id") or "").lower() == model_id.lower():
                return m["recipe"], vname, "recipe variant"
    # Base-model match: strip date/version suffixes (Kimi-K2-Instruct-0905 -> Kimi-K2-Instruct).
    stripped = model_id
    for _ in range(3):
        stripped = re.sub(r"[-_](\d{4,8}|v\d+(\.\d+)*)$", "", stripped, flags=re.I)
        key = lower.get(stripped.lower())
        if key and stripped != model_id:
            return models[key]["recipe"], None, f"base-model recipe ({key})"
    return None, None, None


def blocklist_reason(model_id, blocklist):
    labels = {
        "not_an_llm": "is not a chat/completion LLM (diffusion, image/audio generation, embedding, or reranker)",
        "needs_audio_pipeline": "needs a streaming audio pipeline, not a chat endpoint",
        "needs_vllm_nightly": "requires an unreleased vLLM nightly; it crashes on every stable image",
        "translation_only": "is translation-only, with no chat or tool calling",
        "incompatible_weights": "has weights the current stable vLLM can't load",
    }
    for cat, entry in blocklist.items():
        if isinstance(entry, dict) and model_id.lower() in (m.lower() for m in entry.get("models", [])):
            return f"{model_id} {labels.get(cat, cat)}"
    return None


def split_args(items):
    out = []
    for item in items or []:
        out += shlex.split(item) if " " in item and not item.startswith("{") else [item]
    return out


def drop_flag(args, flags):
    """Remove flags (and their values) we compute ourselves, e.g. TP from a variant built for 8 GPUs."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        name = a.split("=", 1)[0]
        if name in flags:
            i += 1 if "=" in a else 2
            continue
        out.append(a)
        i += 1
    return out


def version_tuple(tag):
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", tag or "")
    return tuple(int(x or 0) for x in m.groups()) if m else None


def estimate(model_id, vram_gb, tp, util):
    token = os.environ.get("HF_TOKEN", "")
    w_bytes, _, err = ev._weight_memory(model_id, "main", token)
    if not w_bytes:
        return {"error": err or "no safetensors found"}
    config, cfg_err = ev._model_config(model_id, "main", token)
    kv, _ = ev._kv_per_token(config)
    w_gb = round(w_bytes / 1024**3, 1)
    usable = vram_gb * util
    remaining = round(max(0.0, usable - w_gb / tp - 4.0), 1)  # same 4 GB overhead as estimate_vram.py
    out = {"weight_gb": w_gb, "remaining_for_kv_gb": remaining, "fits": w_gb / tp < usable and remaining >= 2,
           "max_seq": (config or {}).get("max_position_embeddings"), "config_error": cfg_err,
           "moe": any(k in (config or {}) for k in ("num_experts", "num_local_experts", "n_routed_experts"))}
    if kv:
        out["max_ctx_bf16_kv"] = int(remaining * 1024**3 / (kv / tp)) if remaining else 0
    return out


def min_tp(weight_gb, vram_gb, util, gpu_count):
    tp = 1
    while tp <= gpu_count:
        if weight_gb / tp < vram_gb * util - 6:  # leave room for overhead and some KV cache
            return tp
        tp *= 2
    return None


def remote(cmd, host, user, port, timeout=20):
    return detect._run(cmd, host, user, port, timeout=timeout)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="", help=f"HuggingFace model id (default: {DEFAULT_MODEL})")
    p.add_argument("--host", default="", help="[user@]host of the GPU machine (default: local or ROCM_SSH_HOST)")
    p.add_argument("--gpus", default="", help="comma-separated GPU indices to use, e.g. 0,1")
    p.add_argument("--tp", type=int, default=0, help="tensor parallelism (default: smallest that fits)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--image", default="", help="override the Docker image")
    p.add_argument("--variant", default="", help="force a recipe variant, e.g. fp8")
    args = p.parse_args()

    host, user = args.host, os.environ.get("ROCM_SSH_USER", "")
    if "@" in host:
        user, host = host.split("@", 1)
    host = host or os.environ.get("ROCM_SSH_HOST", "")
    ssh_port = int(os.environ.get("ROCM_SSH_PORT") or 0)  # 0: let ~/.ssh/config decide
    requested = args.model or DEFAULT_MODEL
    warnings, why = [], []
    if not args.model:
        why.append(f"No model given: using the default, {DEFAULT_MODEL} (Apache-2.0, single GPU, tool calling).")

    def finish(status, code, **fields):
        print(json.dumps({"status": status, "requested_model": requested, **fields,
                          "warnings": warnings, "decisions": why}, indent=2))
        sys.exit(code)

    recipes, overrides, blocklist = load("recipes_cache.json"), load("gpu_overrides.json"), load("blocklist.json")

    reason = blocklist_reason(requested, blocklist)
    if reason:
        finish("refuse", 2, reason=reason + ".",
               suggestion="Pick a chat LLM instead, e.g. Qwen/Qwen3.5-9B (small) or openai/gpt-oss-120b (large).")
    if _NVFP4_ID.search(requested):
        base = re.sub(r"[-_](NV)?FP4.*$", "", requested.split("/", 1)[1], flags=re.I)
        alts = [k for k in recipes["models"] if k.split("/", 1)[1].lower() == base.lower()]
        finish("refuse", 2, reason=f"{requested} is an NVFP4/FP4 checkpoint; NVIDIA FP4 has no ROCm kernels and "
               "won't load on AMD GPUs.",
               suggestion=f"Serve {alts[0]} (BF16) or its FP8 variant instead." if alts else
               "Use the BF16 or FP8 checkpoint of the same model instead.")

    # 1. GPU
    det = subprocess.run([sys.executable, str(HERE / "detect.py")] + (["--host", f"{user}@{host}" if user else host]
                         if host else []), capture_output=True, text=True, timeout=90)
    try:
        gpu = json.loads(det.stdout)
    except json.JSONDecodeError:
        gpu = {"error": det.stderr.strip() or det.stdout.strip()}
    if "error" in gpu:
        finish("error", 1, reason=f"GPU detection failed: {gpu['error']}", detail=gpu)
    gfx, count = gpu["gfx_version"], gpu["gpu_count"]
    vram = min(g["vram_gb"] for g in gpu["gpus"] if g.get("vram_gb")) if gpu["gpus"] else 0
    gcfg = overrides["gpu_configs"].get(gfx)
    if not gcfg:
        finish("refuse", 2, reason=f"GPU {gfx} is not an AMD Instinct MI300/MI350-series GPU this skill supports.")
    why.append(f"GPU: {count}x {gpu['gpus'][0]['market_name']} ({gfx}, {vram} GB each).")

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()] if args.gpus else []
    missing = [g for g in gpus if g >= count]
    if missing:
        warnings.append(f"Requested GPU(s) {missing} don't exist: this host has {count} GPU(s) "
                        f"(indices 0-{count - 1}). Using the GPUs that do exist.")
        gpus = [g for g in gpus if g < count]
    usable_gpus = len(gpus) if gpus else count
    if args.tp and args.tp > usable_gpus:
        warnings.append(f"Requested tensor parallel {args.tp}, but only {usable_gpus} GPU(s) are available, so "
                        f"TP={args.tp} can't run here.")

    # 2. Config source
    recipe, variant, how = find_recipe(requested, recipes)
    legacy = None
    if not recipe:
        legacy = next((v for k, v in overrides["legacy_models"].items() if k.lower() == requested.lower()), None)
        how = "legacy_models entry" if legacy else "no recipe: generic config"
    why.append(f"Config source: {how}.")
    if recipe and recipe["model"].get("min_vllm_version") == "nightly":
        finish("refuse", 2, reason=f"{requested} requires an unreleased vLLM nightly; it crashes on stable images.")

    # 3. Candidates in precision order; keep the first that fits.
    util = 0.9
    candidates = []
    if recipe:
        for vname, v in (recipe.get("variants") or {}).items():
            if args.variant and vname != args.variant:
                continue
            prec = (v.get("precision") or "bf16").lower()
            if prec in UNSUPPORTED_PRECISIONS:
                continue
            mid = v.get("model_id") or recipe["model"]["model_id"]
            candidates.append((PRECISION_RANK.get(prec, 5), vname, prec, mid, v))
        candidates.sort(key=lambda c: c[0])
        if variant:  # the user named a variant checkpoint directly: try it first
            candidates.sort(key=lambda c: c[1] != variant)
        # A model id that isn't the recipe's base (suffixed, or a variant) is served as asked.
        if how.startswith("base-model"):
            candidates = [(0, "default", "as published", requested, {})] + candidates
    if not candidates:
        candidates = [(0, "default", "as published", requested, {})]
        token = os.environ.get("HF_TOKEN", "")
        info, err = ev._fetch(f"https://huggingface.co/api/models/{requested}", token)
        if info is None and err and (err.startswith("Not found") or (err.startswith("Authentication") and not token)):
            # The Hub answers 401 (not 404) for ids that don't exist when no token is sent.
            import difflib
            known = list(recipes["models"]) + list(overrides["legacy_models"])
            # An extra suffix is the usual mistake (Qwen3.5-9B-Instruct), so prefixes rank first.
            prefixes = sorted((k for k in known if requested.lower().startswith(k.lower())), key=len, reverse=True)
            close = list(dict.fromkeys(prefixes + difflib.get_close_matches(requested, known, n=3, cutoff=0.6)))[:3]
            finish("refuse", 2, reason=f"{requested} was not found on Hugging Face (check the spelling and "
                   "organization; private or gated repos also need HF_TOKEN).",
                   suggestion=f"Did you mean: {', '.join(close)}?" if close else
                   "Find the exact id at huggingface.co and run this again.")

    chosen, tried, unsized = None, [], []
    for _, vname, prec, mid, v in candidates:
        est = estimate(mid, vram, 1, util)
        if "error" in est:
            tried.append(f"{mid}: can't size it ({est['error']})")
            if transient(est["error"]):
                unsized.append(mid)
            continue
        tp = args.tp if args.tp and args.tp <= usable_gpus else min_tp(est["weight_gb"], vram, util, usable_gpus)
        if tp is None:
            tried.append(f"{mid} ({prec}, {est['weight_gb']} GB): needs more than {usable_gpus} GPU(s)")
            continue
        if tp > 1:
            est = estimate(mid, vram, tp, util)
        if not est["fits"]:
            tried.append(f"{mid} ({prec}, {est['weight_gb']} GB): too tight at TP={tp}")
            continue
        chosen = (vname, prec, mid, v, tp, est)
        break

    if not chosen:  # 5c: AMD's own quantized checkpoints
        base = requested.split("/", 1)[1]
        try:
            import urllib.request
            q = f"https://huggingface.co/api/models?author=amd&search={base}&limit=10"
            with urllib.request.urlopen(q, timeout=20) as r:
                amd_ids = [m["id"] for m in json.load(r)]
        except Exception:
            amd_ids = []
        for mid in amd_ids:
            est = estimate(mid, vram, 1, util)
            if "error" in est:
                continue
            tp = min_tp(est["weight_gb"], vram, util, usable_gpus)
            if tp and estimate(mid, vram, tp, util)["fits"]:
                chosen = ("amd-quantized", "quantized (AMD Quark)", mid, {}, tp, estimate(mid, vram, tp, util))
                warnings.append(f"{requested} doesn't fit on {usable_gpus}x {vram} GB in any recipe variant; "
                                f"using AMD's quantized checkpoint {mid} instead.")
                break
    if not chosen and unsized:
        # Sizing failed for a transient reason (rate limit, network): that is not "doesn't fit".
        finish("error", 1, reason=f"Couldn't reach Hugging Face to size {', '.join(unsized)} "
               f"({tried[-1].split('(', 1)[-1].rstrip(')')}). This says nothing about whether the model fits.",
               tried=tried, suggestion="Wait a minute and run plan.py again, or set HF_TOKEN for higher Hub "
               "rate limits.")
    if not chosen:
        finish("does_not_fit", 3, reason=f"{requested} doesn't fit on this host ({usable_gpus}x {vram} GB) "
               "at any supported precision.", tried=tried,
               suggestion="Use a smaller model, or a host with more GPUs (the 'tried' list shows what each "
               "option needs).")

    vname, prec, mid, v, tp, est = chosen
    for t in tried:
        why.append(f"Skipped {t}.")
    why.append(f"Serving {mid} ({prec}), {est['weight_gb']} GB of weights, TP={tp}, "
               f"{est['remaining_for_kv_gb']} GB left for KV cache per GPU.")
    if mid != requested and vname != "amd-quantized":
        warnings.append(f"{requested} doesn't fit at full precision here; serving the {prec} variant {mid}.")
    if prec in gcfg["precision"].get("emulated", []):
        why.append(f"{prec} is emulated on {gfx}: weights stay compressed, compute dequantizes to BF16.")

    # 4. Args and env
    env = dict(gcfg["env_defaults"])
    vargs = []
    if recipe:
        amd = (recipe.get("hardware_overrides") or {}).get("amd") or {}
        env.update(recipe["model"].get("base_env") or {})
        env.update(amd.get("extra_env") or {})
        env.update(v.get("extra_env") or {})
        feats = recipe.get("features") or {}
        vargs = (split_args(recipe["model"].get("base_args")) + split_args(amd.get("extra_args"))
                 + split_args((feats.get("tool_calling") or {}).get("args"))
                 + split_args((feats.get("reasoning") or {}).get("args")) + split_args(v.get("extra_args")))
        if amd.get("install_note"):
            warnings.append("Recipe note for AMD: " + amd["install_note"].strip().splitlines()[0])
    elif legacy:
        env.update(legacy.get("env_vars") or {})
        vargs = ["--enable-auto-tool-choice", "--tool-call-parser", legacy.get("tool_call_parser", "hermes")]
        if legacy.get("reasoning_parser"):
            vargs += ["--reasoning-parser", legacy["reasoning_parser"]]
        vargs += split_args(legacy.get("vllm_args"))
    else:
        vargs = list(GENERIC_ARGS)
    # GPU-specific model overrides go last so a recipe's env can't undo them
    # (e.g. VLLM_ROCM_USE_AITER_MOE=0 for gpt-oss on gfx950).
    for o in gcfg.get("model_overrides", []):
        if o["match"].lower() in mid.lower():
            env.update(o["env_set"])
            why.append(f"{gfx} override for {o['match']}: {o['env_set']} ({o['reason'][:80]}...)")
    if "--enable-auto-tool-choice" not in vargs:
        vargs.append("--enable-auto-tool-choice")
    vargs = drop_flag(vargs, _TP_ARGS | {"--max-model-len", "--port"})
    if tp > 1:
        vargs += ["--tensor-parallel-size", str(tp)]
        if (recipe and recipe["model"].get("architecture") == "moe") or est.get("moe"):
            if "--distributed-executor-backend" not in vargs:
                vargs += ["--distributed-executor-backend", "mp"]
    ctx_note = None
    if est.get("max_ctx_bf16_kv") is not None and est.get("max_seq"):
        if est["max_ctx_bf16_kv"] < est["max_seq"]:
            ctx = max(1024, est["max_ctx_bf16_kv"] // 1024 * 1024)
            vargs += ["--max-model-len", str(ctx)]
            ctx_note = f"KV cache limits context to {ctx} tokens (native {est['max_seq']}); add " \
                       "--kv-cache-dtype fp8 to roughly double it."
        else:
            ctx = est["max_seq"]
    else:
        # Couldn't read config.json (gated model without HF_TOKEN): vLLM would default to the
        # native context and may refuse to start if the KV cache can't hold it. Cap it.
        ctx = 32768
        vargs += ["--max-model-len", str(ctx)]
        ctx_note = (f"Couldn't read {mid}'s config ({est.get('config_error') or 'unknown'}), so the KV-cache "
                    f"size is unknown; capped context at {ctx} tokens to be safe.")
    if ctx_note:
        warnings.append(ctx_note)
    if est["remaining_for_kv_gb"] < 6:
        warnings.append(f"Only {est['remaining_for_kv_gb']} GB left for KV cache: if the container OOMs "
                        "while capturing HIP graphs, retry with --enforce-eager.")
    if est.get("config_error") and re.search(r"auth|denied|401|403", est["config_error"], re.I):
        warnings.append(f"{mid} is gated: HF_TOKEN must be set on the GPU host for an account that has "
                        f"accepted the license at huggingface.co/{mid}, or startup fails with "
                        "'Engine core initialization failed'.")

    image = args.image or recipes["docker_image"]
    need = recipe["model"].get("min_vllm_version") if recipe else None
    if need and version_tuple(image.rsplit(":", 1)[-1]) and version_tuple(need) and \
            version_tuple(image.rsplit(":", 1)[-1]) < version_tuple(need):
        warnings.append(f"Image {image} is older than this model's minimum vLLM {need}: expect "
                        "'Engine core initialization failed'. Use a newer vllm/vllm-openai-rocm tag.")
    if gpus:
        env["HIP_VISIBLE_DEVICES"] = ",".join(map(str, gpus[:tp] if len(gpus) >= tp else gpus))

    # 5. Host details: HF cache layout and port
    rc, out, _ = remote("ls -d ~/.cache/huggingface/hub ~/.cache/huggingface/models--* 2>/dev/null | head -3",
                        host, user, ssh_port)
    if "/hub" in out or not out.strip():
        mount = "~/.cache/huggingface:/root/.cache/huggingface"
    else:
        mount = "~/.cache/huggingface:/root/.cache/huggingface/hub"
        why.append("Host cache keeps models-- at its root, so it's mounted as the container's hub/ directory.")
    port = args.port
    rc, out, _ = remote("ss -tln 2>/dev/null | awk '{print $4}'", host, user, ssh_port)
    busy = {int(m) for m in re.findall(r":(\d+)$", out, re.M)}
    while port in busy:
        port += 1
    if port != args.port:
        warnings.append(f"Port {args.port} is in use on the host; using {port}.")

    name = "vllm-" + re.sub(r"[^a-z0-9.-]+", "-", mid.split("/", 1)[1].lower())
    lines = [f"docker run -d --name {name}"]
    lines += [f"  {f}" for f in overrides["docker_flags"]]
    lines += [f"  -v {mount}", f"  -p {port}:{port}"]
    lines += [f"  --env {shlex.quote(f'{k}={v}')}" for k, v in env.items()]
    lines += ["  --env HF_TOKEN", f"  {image}", f"  --model {mid}"]
    i = 0
    while i < len(vargs):  # one flag (with its value) per line
        if i + 1 < len(vargs) and not vargs[i + 1].startswith("-"):
            lines.append(f"  {vargs[i]} {shlex.quote(vargs[i + 1])}")
            i += 2
        else:
            lines.append(f"  {vargs[i]}")
            i += 1
    lines.append(f"  --port {port}")
    command = " \\\n".join(lines)
    target = f"{user}@{host}" if user else host
    run_command = command if not host or detect._is_local(host) else \
        f"ssh {target} bash -s <<'EOF'\n{command}\nEOF"

    finish("ok", 0, model_id=mid, variant=vname, precision=prec, tp=tp, max_model_len=ctx, port=port,
           image=image, container_name=name, gpu=f"{count}x {gpu['gpus'][0]['market_name']} ({gfx})",
           weight_gb=est["weight_gb"], host=target or "local", command=command, run_command=run_command)


if __name__ == "__main__":
    main()
