#!/usr/bin/env python3
"""
Does a HuggingFace model fit in host RAM for CPU serving at a given context?

No download -- reads HF metadata over HTTP. Answers one question:
    weights + KV(max_model_len x num_prompts) + reserve  <=  RAM ?
If not, prints the largest max_model_len that would fit, so you reduce it and
retry. Exit 0 = fits, 1 = does not fit (or error).

    estimate_memory.py --model-id Qwen/Qwen3-8B --ram-gb 755 --max-model-len 4096 --num-prompts 8

Three sub-problems, one function each: weight_gb(), kv_bytes_per_token(), fit().
Env: HF_TOKEN for gated models. --weight-gb overrides weights if metadata is missing.
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error

HF = "https://huggingface.co"
KV_BYTES_PER_ELEM = 2  # zentorch CPU KV cache is bf16-only (2 bytes); no fp8 KV support


def _get(url, token):
    """GET JSON from HF. Returns (data, error_message)."""
    headers = {"User-Agent": "estimate-memory/2"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        return None, {401: "not found, or gated (set HF_TOKEN if it is gated)",
                      403: "access denied -- accept the model license on HuggingFace",
                      404: "model not found"}.get(e.code, f"HTTP {e.code}")
    except Exception as e:
        return None, str(e)


def weight_gb(model, rev, token):
    """(1) Weight RAM = sum of uncompressed weight-file sizes. Works for
    .safetensors and legacy .bin; file size is ground truth even for quantized
    checkpoints. Returns (gb, error)."""
    tree, err = _get(f"{HF}/api/models/{model}/tree/{rev}", token)
    if not isinstance(tree, list):
        return None, err or "no file tree"
    total = sum(
        f.get("size", 0) for f in tree
        if f.get("type") == "file" and (
            f.get("path", "").endswith(".safetensors")
            or (f.get("path", "").endswith(".bin") and "model" in f.get("path", "").lower())
        )
    )
    if total == 0:
        return None, "no weight files (.safetensors/.bin) found -- pass --weight-gb"
    return round(total / 2**30, 2), None


def get_config(model, rev, token):
    """Model config.json, unwrapping the LLM sub-config of multimodal models."""
    cfg, _ = _get(f"{HF}/{model}/resolve/{rev}/config.json", token)
    if cfg and "num_hidden_layers" not in cfg:
        for k in ("text_config", "language_config", "llm_config"):
            if isinstance(cfg.get(k), dict) and cfg[k].get("num_hidden_layers"):
                sub = dict(cfg[k])
                sub.setdefault("max_position_embeddings", cfg.get("max_position_embeddings"))
                return sub
    return cfg


def kv_bytes_per_token(cfg):
    """(2) KV-cache bytes per token = 2(K,V) x layers x kv_heads x head_dim x 2 (bf16).
    zentorch CPU caches KV in bf16 only. MLA models (DeepSeek) cache a compressed latent."""
    if not cfg or not cfg.get("num_hidden_layers"):
        return 0
    nbytes = KV_BYTES_PER_ELEM
    layers = cfg["num_hidden_layers"]
    if "kv_lora_rank" in cfg:  # MLA: latent KV
        return 2 * layers * (cfg["kv_lora_rank"] + cfg.get("qk_rope_head_dim", 0)) * nbytes
    kv_heads = cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 0))
    head_dim = cfg.get("head_dim") or (cfg.get("hidden_size", 0) // max(1, cfg.get("num_attention_heads", 1)))
    return 2 * layers * kv_heads * head_dim * nbytes


def fit(weight, kv_per_tok, ctx, prompts, ram, reserve):
    """(3) Verdict + the largest max_model_len that would fit if it doesn't."""
    kv_gb = kv_per_tok * ctx * prompts / 2**30
    required = round(weight + kv_gb + reserve, 2)
    out = {"max_model_len": ctx, "num_prompts": prompts, "weight_gb": weight,
           "kv_cache_gb": round(kv_gb, 2), "reserve_gb": reserve,
           "required_gb": required, "ram_gb": ram, "fits": required <= ram}
    if not out["fits"]:
        budget = (ram - weight - reserve) * 2**30
        best = int(budget / (kv_per_tok * prompts)) // 256 * 256 if kv_per_tok and budget > 0 else 0
        out["suggested_max_model_len"] = max(0, best)
        out["action"] = (f"reduce --max-model-len to {best} or less and retry"
                         if best >= 256 else "weights alone exceed RAM -- use a smaller model")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--ram-gb", type=float, default=0, help="host RAM (enables the fit verdict)")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--num-prompts", type=int, default=1, help="concurrent sequences")
    p.add_argument("--reserve-gb", type=float, default=16, help="RAM held back for OS + vLLM runtime")
    p.add_argument("--weight-gb", type=float, default=0, help="override weight RAM if metadata is unavailable")
    a = p.parse_args()
    token = os.environ.get("HF_TOKEN", "")

    w = a.weight_gb if a.weight_gb > 0 else None
    if w is None:
        w, err = weight_gb(a.model_id, a.revision, token)
        if w is None:
            print(json.dumps({"error": err, "model_id": a.model_id}))
            sys.exit(1)

    cfg = get_config(a.model_id, a.revision, token)
    kv_per_tok = kv_bytes_per_token(cfg)
    max_seq = cfg.get("max_position_embeddings") if cfg else None
    ctx = min(a.max_model_len, max_seq) if max_seq else a.max_model_len

    out = {"model_id": a.model_id, "weight_gb": w, "kv_dtype": "bf16",
           "kv_bytes_per_token": kv_per_tok, "model_max_seq_len": max_seq}
    if a.ram_gb > 0:
        out["fit"] = fit(w, kv_per_tok, ctx, a.num_prompts, a.ram_gb, a.reserve_gb)

    print(json.dumps(out, indent=2))
    sys.exit(0 if out.get("fit", {"fits": True})["fits"] else 1)


if __name__ == "__main__":
    main()
