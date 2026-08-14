#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""CLI for the vendored ``quant_validation_onnx`` module in this skill directory.

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

import quant_validation_onnx as qv  # noqa: E402


def _print_json(data: object) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _load_json_arg(raw: str) -> dict:
    path = Path(raw).expanduser()
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quark ONNX quant output checks (skill vendored quant_validation_onnx)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_self = sub.add_parser("self-test", help="Run embedded mock-data checks for quant_validation_onnx.__all__.")
    p_self.set_defaults(func=_cmd_self_test)

    p_fuzzy = sub.add_parser(
        "fuzzy",
        help="Summarize op-type histogram, canonical node patterns, initializer dtype counts, "
        "and QDQ / com.amd.quark presence (header-only).",
    )
    p_fuzzy.add_argument("--model-path", required=True, help="Path to a single .onnx file.")
    p_fuzzy.set_defaults(func=_cmd_fuzzy)

    p_aux = sub.add_parser("auxiliary", help="Compare non-weight auxiliary files between two model directories.")
    p_aux.add_argument("--source-model-dir", required=True)
    p_aux.add_argument("--quantized-model-dir", required=True)
    p_aux.add_argument("--ignore", nargs="*", default=[])
    p_aux.add_argument("--max-examples", type=int, default=50)
    p_aux.add_argument("--max-hash-bytes", type=int, default=256 * 1024 * 1024)
    p_aux.set_defaults(func=_cmd_auxiliary)

    p_meta = sub.add_parser(
        "metadata",
        help="Compare two .onnx models' IR / producer / opset / graph I/O signature after "
        "stripping Quark-injected opset domains.",
    )
    p_meta.add_argument("--source-model-path", required=True)
    p_meta.add_argument("--quantized-model-path", required=True)
    p_meta.add_argument(
        "--extra-excluded-domain",
        action="append",
        default=None,
        help="Extra opset domain to strip from the quantized side before equality (may be repeated).",
    )
    p_meta.set_defaults(func=_cmd_metadata)

    p_md5 = sub.add_parser(
        "md5",
        help="MD5 spot-check exclude-listed initializers (reads raw_data + external-data byte ranges).",
    )
    p_md5.add_argument("--source-model-path", required=True)
    p_md5.add_argument("--output-model-path", required=True)
    p_md5.add_argument(
        "--quant-config",
        required=True,
        help="JSON object string or path to JSON (exclude / exclude_initializers / nodes_to_exclude, "
        "max_samples, random_seed, ...).",
    )
    p_md5.set_defaults(func=_cmd_md5)

    args = parser.parse_args()
    args.func(args)


def _cmd_self_test(_args: argparse.Namespace) -> None:
    raise SystemExit(qv.run_selftest())


def _cmd_fuzzy(args: argparse.Namespace) -> None:
    out = qv.get_fuzzy_node_op_summary(args.model_path)
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


def _cmd_metadata(args: argparse.Namespace) -> None:
    out = qv.check_model_metadata_equal_except_quantization(
        args.source_model_path,
        args.quantized_model_path,
        extra_excluded_domains=args.extra_excluded_domain,
    )
    _print_json(out)


def _cmd_md5(args: argparse.Namespace) -> None:
    qconf = _load_json_arg(args.quant_config)
    if not isinstance(qconf, dict):
        print("quant-config must be a JSON object", file=sys.stderr)
        raise SystemExit(2)
    out = qv.check_non_quantized_initializers_md5_unchanged(
        args.source_model_path,
        args.output_model_path,
        qconf,
    )
    _print_json(out)


if __name__ == "__main__":
    main()
