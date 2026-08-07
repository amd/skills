#!/usr/bin/env python3
"""
Check model-specific overrides in gpu_overrides.json for a given model + GPU.

MUST be run before constructing any Docker command. Prints the env vars and
vLLM arg removals that must be applied (highest precedence, overrides recipe
and GPU defaults). Exits non-zero if the data file is missing.

Usage:
    python3 scripts/check_overrides.py --model openai/gpt-oss-120b --gfx gfx950
    python3 scripts/check_overrides.py --model Qwen/Qwen3-8B --gfx gfx942

Output: JSON with:
    {
      "gfx_version": "gfx950",
      "model_id": "openai/gpt-oss-120b",
      "matched_override": { "match": "...", "env_set": {...}, "reason": "..." },
      "env_set": { "VLLM_ROCM_USE_AITER_MOE": "0" },
      "args_remove": [],
      "summary": "Apply 1 override(s): VLLM_ROCM_USE_AITER_MOE=0 ..."
    }

If no override matches:
    {
      "gfx_version": "gfx950",
      "model_id": "Qwen/Qwen3-8B",
      "matched_override": null,
      "env_set": {},
      "args_remove": [],
      "summary": "No model-specific overrides for Qwen/Qwen3-8B on gfx950."
    }
"""

import argparse
import json
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GPU_OVERRIDES_FILE = DATA_DIR / "gpu_overrides.json"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HuggingFace model ID, e.g. openai/gpt-oss-120b")
    p.add_argument("--gfx", required=True, help="gfx version from detect.py, e.g. gfx950")
    args = p.parse_args()

    if not GPU_OVERRIDES_FILE.exists():
        print(json.dumps({"error": f"Missing {GPU_OVERRIDES_FILE}"}), file=sys.stderr)
        sys.exit(1)

    with open(GPU_OVERRIDES_FILE) as f:
        overrides_data = json.load(f)

    gpu_cfg = overrides_data.get("gpu_configs", {}).get(args.gfx, {})
    model_overrides = gpu_cfg.get("model_overrides", [])

    matched = None
    for entry in model_overrides:
        match_prefix = entry.get("match", "")
        if args.model.startswith(match_prefix):
            matched = entry
            break

    env_set = dict(matched.get("env_set", {})) if matched else {}
    args_remove = list(matched.get("args_remove", [])) if matched else []

    if matched:
        env_parts = ", ".join(f"{k}={v}" for k, v in env_set.items())
        rm_parts = (", ".join(f"remove arg '{a}'" for a in args_remove)) if args_remove else ""
        parts = [p for p in [env_parts, rm_parts] if p]
        summary = (
            f"Apply {len(env_set) + len(args_remove)} override(s) for "
            f"{args.model!r} on {args.gfx}: {'; '.join(parts)}. "
            f"Reason: {matched.get('reason', '')}"
        )
    else:
        summary = f"No model-specific overrides for {args.model!r} on {args.gfx}."

    result = {
        "gfx_version": args.gfx,
        "model_id": args.model,
        "matched_override": matched,
        "env_set": env_set,
        "args_remove": args_remove,
        "summary": summary,
    }
    print(json.dumps(result, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
