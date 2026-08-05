#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""CLI for the vendored ``quant_validation`` module in this skill directory.

Resolves imports without ``PYTHONPATH`` by prepending this file's directory to ``sys.path``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent
if str(_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(_SKILL_DIR))

import quant_validation as qv  # noqa: E402


def _print_json(data: object) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _load_json_arg(raw: str) -> dict:
    path = Path(raw).expanduser()
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quark quant output checks (skill vendored quant_validation).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_self = sub.add_parser("self-test", help="Run embedded mock-data checks for quant_validation.__all__.")
    p_self.set_defaults(func=_cmd_self_test)

    p_fuzzy = sub.add_parser("fuzzy", help="List canonical tensor patterns with dtype_counts (header-only).")
    p_fuzzy.add_argument("--model-path", required=True, help="Model dir or single .safetensors shard.")
    p_fuzzy.set_defaults(func=_cmd_fuzzy)

    p_aux = sub.add_parser("auxiliary", help="Compare non-weight auxiliary files between two model dirs.")
    p_aux.add_argument("--source-model-dir", required=True)
    p_aux.add_argument("--quantized-model-dir", required=True)
    p_aux.add_argument("--ignore", nargs="*", default=[])
    p_aux.add_argument("--max-examples", type=int, default=50)
    p_aux.add_argument("--max-hash-bytes", type=int, default=256 * 1024 * 1024)
    p_aux.set_defaults(func=_cmd_auxiliary)

    p_cfg = sub.add_parser("config", help="Compare two config.json trees after stripping quantization keys.")
    p_cfg.add_argument("--original-config", required=True)
    p_cfg.add_argument("--quantized-config", required=True)
    p_cfg.add_argument("--ignore-key-name", action="append", default=None)
    p_cfg.add_argument("--replace-default-ignores", action="store_true")
    p_cfg.set_defaults(func=_cmd_config)

    p_md5 = sub.add_parser("md5", help="MD5 spot-check exclude-listed tensors (reads raw safetensor bytes).")
    p_md5.add_argument("--source-model-dir", required=True)
    p_md5.add_argument("--output-model-dir", required=True)
    p_md5.add_argument(
        "--quant-config",
        required=True,
        help="JSON object string or path to JSON (exclude, max_samples, seed, etc.).",
    )
    p_md5.set_defaults(func=_cmd_md5)

    args = parser.parse_args()
    args.func(args)


def _cmd_self_test(_args: argparse.Namespace) -> None:
    raise SystemExit(qv.run_selftest())


def _cmd_fuzzy(args: argparse.Namespace) -> None:
    out = qv.get_fuzzy_tensor_names(args.model_path)
    _print_json(out)


def _cmd_auxiliary(args: argparse.Namespace) -> None:
    out = qv.check_auxiliary_files_copied(
        args.source_model_dir,
        args.quantized_model_dir,
        ignore=args.ignore or None,
        max_examples=args.max_examples,
        max_hash_bytes=args.max_hash_bytes,
    )
    _print_json(out)


def _cmd_config(args: argparse.Namespace) -> None:
    out = qv.check_config_json_equal_except_quantization(
        args.original_config,
        args.quantized_config,
        ignore_key_names=args.ignore_key_name,
        replace_default_ignores=args.replace_default_ignores,
    )
    _print_json(out)


def _cmd_md5(args: argparse.Namespace) -> None:
    qconf = _load_json_arg(args.quant_config)
    if not isinstance(qconf, dict):
        print("quant-config must be a JSON object", file=sys.stderr)
        raise SystemExit(2)
    out = qv.check_non_quantized_tensors_md5_unchanged(
        args.source_model_dir,
        args.output_model_dir,
        qconf,
    )
    _print_json(out)


if __name__ == "__main__":
    main()
