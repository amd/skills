#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Source file for Claude skill ``quark-onnx-result-validator``.
# Lives at
# ``.claude/skills-impl/l1-atomic/onnx/quark-onnx-result-validator/quant_validation_onnx.py``.

"""Header-only quantization result validation helpers for Quark ONNX outputs.

This module performs lightweight checks for ONNX quantization output. Most
logic parses only the ONNX model graph (``onnx.load(..., load_external_data=False)``)
and regular file metadata without decoding full initializer payloads.

**Agent diagnostics**: the four ``__all__`` entry points emit line-oriented
progress and failure clues to **stderr** using ``[quant-validation-onnx][tag]``.
Machine-consumable fields such as ``ok`` and ``errors`` remain in return values
so diagnostics are not mixed into JSON output.

Exception: ``check_non_quantized_initializers_md5_unchanged`` reads raw bytes
for initializers matched by exclude rules and calculates MD5 values to confirm
that non-quantized weights are byte-identical. Use ``quant_config["max_samples"]``
for large models.

Also, ``check_model_metadata_equal_except_quantization`` reads two ``.onnx``
models and compares IR version, producer, default-domain opset, and graph
input / output signatures after stripping Quark-injected custom-op domains.
"""

from __future__ import annotations

import fnmatch
import hashlib
import random
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, external_data_helper

INDEX_SEGMENT = re.compile(r"(?:(?<=^)|(?<=[._/]))\d+(?=[._/]|$)")

# Files that are weights / payload artifacts. Skip them in auxiliary asset
# alignment, since quantization rewrites them and exact-match content checks are
# not meaningful.
AUXILIARY_IGNORE_PATTERNS = [
    "*.onnx",
    "*.onnx_data",
    "*.onnx.data",
    "*_data",  # generic external-data shard naming
    "*.pb",
    "*.bin",
    "*.ckpt",
    "*.gguf",
    "*.safetensors",
    "*.msgpack",
    "*.pt",
    "*.pth",
    "model.onnx.data",
]
AUXILIARY_IGNORE_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".cache", ".venv", "node_modules"}

# Opset domains that Quark may inject as part of quantization. When comparing
# ``opset_import`` across source / quantized, these are stripped from the
# quantized side before equality.
QUARK_INJECTED_DOMAINS: frozenset[str] = frozenset(
    {
        "com.amd.quark",
        "com.microsoft",
        "com.microsoft.experimental",
    }
)

# Quark custom-op type names (registered under ``com.amd.quark``). Their
# presence in the quantized graph is a positive signal that quantization ran.
QUARK_CUSTOM_OP_TYPES: frozenset[str] = frozenset(
    {
        "BFPQuantizeDequantize",
        "MXQuantizeDequantize",
        "ExtendedQuantizeLinear",
        "ExtendedDequantizeLinear",
        "ExtendedInstanceNormalization",
        "ExtendedLSTM",
    }
)

# Standard ONNX QDQ op types. Their presence is the most common positive signal
# that quantization ran.
QDQ_OP_TYPES: frozenset[str] = frozenset({"QuantizeLinear", "DequantizeLinear", "MatMulNBits"})

__all__ = [
    # Auxiliary assets: after ignoring weights / payloads, compare non-weight
    # file sets and content on both sides.
    "check_auxiliary_files_copied",
    # Expected non-quantized initializers: use excludes from ``quant_config``
    # and spot-check raw payload bytes with MD5.
    "check_non_quantized_initializers_md5_unchanged",
    # Two .onnx models: compare IR / producer / opset (after stripping Quark
    # domains) plus graph input/output signature.
    "check_model_metadata_equal_except_quantization",
    # Header-only summary: op-type histogram, canonical node-name patterns,
    # initializer dtype counts, QDQ / custom-op presence.
    "get_fuzzy_node_op_summary",
]


# --------------------------------------------------------------------------- #
# Common helpers
# --------------------------------------------------------------------------- #


def canonicalize_name(name: str) -> str:
    """
    Replace pure numeric path segments with ``*``.

    Many ONNX nodes / initializers differ only by numeric indexes
    (``Conv_12``, ``model.layer.3.Conv``). Collapsing those into ``*``
    deduplicates them into a compact structural view.

    :param str name: Dotted, underscored, or slashed node / initializer name.

    :return: Canonicalized name.
    :rtype: str
    """
    return INDEX_SEGMENT.sub("*", name)


def _path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _agent_log(tag: str, message: str) -> None:
    """Write agent-readable progress and exception clues to stderr."""
    print(f"[quant-validation-onnx][{tag}] {message}", file=sys.stderr, flush=True)


def _sha256_file(path: Path, max_hash_bytes: int) -> str | None:
    if path.stat().st_size > max_hash_bytes:
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(
        fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name.rsplit(".", 1)[0], pattern) for pattern in patterns
    )


def _matches_path_pattern(relative_path: Path, patterns: list[str]) -> bool:
    path = relative_path.as_posix()
    name = relative_path.name
    return any(fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _is_auxiliary_file(relative_path: Path, ignore_patterns: list[str]) -> bool:
    if any(part in AUXILIARY_IGNORE_DIRS for part in relative_path.parts):
        return False
    return not _matches_path_pattern(relative_path, ignore_patterns)


def _collect_auxiliary_files(model_dir: Path, ignore_patterns: list[str]) -> list[Path]:
    files = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(model_dir)
        if _is_auxiliary_file(rel, ignore_patterns):
            files.append(rel)
    return files


def _dtype_name(dtype_enum: int) -> str:
    """Map ``TensorProto.DataType`` enum int to its string name."""
    name_map = {v: k for k, v in TensorProto.DataType.items()}
    return name_map.get(dtype_enum, f"UNKNOWN({dtype_enum})")


def load_model_header(model_path: str | Path) -> onnx.ModelProto:
    """
    Load an ONNX model **without** following external-data references.

    This keeps the parse cheap even for multi-GB models — the graph proto is
    parsed in full but tensor payloads stored externally are left on disk.

    :param str | pathlib.Path model_path: Path to the ``.onnx`` file.

    :return: Parsed model proto (initializer payloads may be stubs).
    :rtype: onnx.ModelProto
    """
    return onnx.load(str(_path(model_path)), load_external_data=False)


def collect_initializer_metadata(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    """
    Collect metadata for every initializer in the model graph.

    :param onnx.ModelProto model: Parsed model (with ``load_external_data=False``).

    :return: Initializer metadata keyed by tensor name.
    :rtype: dict[str, dict[str, Any]]
    """
    out: dict[str, dict[str, Any]] = {}
    for init in model.graph.initializer:
        info = {
            "dtype": _dtype_name(init.data_type),
            "shape": list(init.dims),
            "uses_external_data": external_data_helper.uses_external_data(init),
            "external_data": {},
            "inline_nbytes": 0,
        }
        if info["uses_external_data"]:
            ed = {entry.key: entry.value for entry in init.external_data}
            info["external_data"] = {
                "location": ed.get("location"),
                "offset": int(ed["offset"]) if "offset" in ed else 0,
                "length": int(ed["length"]) if "length" in ed else 0,
            }
        else:
            info["inline_nbytes"] = len(init.raw_data) if init.raw_data else 0
        out[init.name] = info
    return out


# --------------------------------------------------------------------------- #
# Step 1: auxiliary file alignment
# --------------------------------------------------------------------------- #


def check_auxiliary_files_copied(
    source_model_dir: str | Path,
    quantized_model_dir: str | Path,
    ignore: list[str] | None = None,
    max_examples: int = 50,
    max_hash_bytes: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    """
    Check whether the quantized output copied auxiliary assets from the source model directory.

    Quantization rewrites ``.onnx`` / ``.onnx_data`` files. Non-weight auxiliary files
    (``README``, ``LICENSE``, tokenizer / preprocessing JSON, calibration scripts, ``assets/``)
    should appear unchanged in the output directory. After applying the default ignore list,
    this function compares the remaining path sets and file content (SHA256 for small files,
    size only for large ones).

    :param str | pathlib.Path source_model_dir: Source model root directory.
    :param str | pathlib.Path quantized_model_dir: Quantized model output root directory.
    :param list[str] | None ignore: Additional relative path or file-name glob patterns to
        ignore. Merged with ``AUXILIARY_IGNORE_PATTERNS``.
    :param int max_examples: Maximum examples returned for list-like fields.
    :param int max_hash_bytes: Calculate SHA256 only for files at or below this size.

    :return: Auxiliary file alignment report.
    :rtype: dict[str, Any]
    """
    _agent_log("aux", "start: check_auxiliary_files_copied (ignoring .onnx/.onnx_data/payloads)")
    source_path = _path(source_model_dir)
    output_path = _path(quantized_model_dir)
    _agent_log("aux", f"source directory: {source_path}")
    _agent_log("aux", f"quantized output directory: {output_path}")
    ignore_patterns = AUXILIARY_IGNORE_PATTERNS + (ignore or [])
    _agent_log("aux", f"merged ignore pattern count: {len(ignore_patterns)}")
    errors: list[str] = []
    if not source_path.is_dir():
        errors.append(f"Source model directory not found: {source_path}")
    if not output_path.is_dir():
        errors.append(f"Quantized model directory not found: {output_path}")
    if errors:
        for err in errors:
            _agent_log("aux", f"failure: {err}")
        _agent_log("aux", "end: check_auxiliary_files_copied (directory unavailable)")
        return {
            "ok": False,
            "source_model_dir": str(source_path),
            "quantized_model_dir": str(output_path),
            "ignored_patterns": ignore_patterns,
            "source_auxiliary_file_count": 0,
            "output_auxiliary_file_count": 0,
            "missing_file_count": 0,
            "mismatched_file_count": 0,
            "extra_file_count": 0,
            "source_auxiliary_dirs": [],
            "missing_dirs": [],
            "missing_files": [],
            "mismatched_files": [],
            "extra_files": [],
            "hash_skipped_files": [],
            "errors": errors,
        }

    source_files = _collect_auxiliary_files(source_path, ignore_patterns)
    output_files = _collect_auxiliary_files(output_path, ignore_patterns)
    _agent_log(
        "aux",
        f"source auxiliary file count: {len(source_files)}, output auxiliary file count: {len(output_files)}",
    )
    source_set = set(source_files)
    output_set = set(output_files)

    missing_files = sorted(source_set - output_set, key=lambda item: item.as_posix())
    extra_files = sorted(output_set - source_set, key=lambda item: item.as_posix())
    _agent_log(
        "aux",
        f"path set diff: output missing {len(missing_files)}, output extra {len(extra_files)}",
    )

    mismatched_files: list[dict[str, Any]] = []
    hash_skipped_files: list[str] = []

    for rel in sorted(source_set & output_set, key=lambda item: item.as_posix()):
        source_file = source_path / rel
        output_file = output_path / rel
        source_size = source_file.stat().st_size
        output_size = output_file.stat().st_size
        if source_size != output_size:
            mismatched_files.append(
                {
                    "path": rel.as_posix(),
                    "source_size": source_size,
                    "output_size": output_size,
                    "reason": "size_mismatch",
                }
            )
            continue
        source_hash = _sha256_file(source_file, max_hash_bytes)
        output_hash = _sha256_file(output_file, max_hash_bytes)
        if source_hash is None or output_hash is None:
            hash_skipped_files.append(rel.as_posix())
            continue
        if source_hash != output_hash:
            mismatched_files.append(
                {
                    "path": rel.as_posix(),
                    "source_sha256": source_hash,
                    "output_sha256": output_hash,
                    "reason": "hash_mismatch",
                }
            )

    source_dirs = sorted({p.parent.as_posix() for p in source_files if p.parent.as_posix() != "."})
    missing_dirs = sorted({p.parent.as_posix() for p in missing_files if p.parent.as_posix() != "."})

    ok = not missing_files and not mismatched_files
    _agent_log(
        "aux",
        f"result: ok={ok} missing={len(missing_files)} mismatched={len(mismatched_files)} "
        f"extra={len(extra_files)} hash_skipped={len(hash_skipped_files)}",
    )
    _agent_log("aux", "end: check_auxiliary_files_copied")
    return {
        "ok": ok,
        "source_model_dir": str(source_path),
        "quantized_model_dir": str(output_path),
        "ignored_patterns": ignore_patterns,
        "source_auxiliary_file_count": len(source_files),
        "output_auxiliary_file_count": len(output_files),
        "missing_file_count": len(missing_files),
        "mismatched_file_count": len(mismatched_files),
        "extra_file_count": len(extra_files),
        "source_auxiliary_dirs": source_dirs[:max_examples],
        "missing_dirs": missing_dirs[:max_examples],
        "missing_files": [p.as_posix() for p in missing_files[:max_examples]],
        "mismatched_files": mismatched_files[:max_examples],
        "extra_files": [p.as_posix() for p in extra_files[:max_examples]],
        "hash_skipped_files": hash_skipped_files[:max_examples],
        "errors": [],
    }


# --------------------------------------------------------------------------- #
# Step 2: non-quantized initializer MD5 spot-check
# --------------------------------------------------------------------------- #


def _normalize_exclude_patterns(quant_config: dict[str, Any]) -> list[str]:
    """Parse exclude / exclude_initializers / nodes_to_exclude glob lists."""

    def _to_list(raw: Any) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, (list, tuple)):  # noqa: UP038
            return [str(item) for item in raw]
        return [str(raw)]

    parts = (
        _to_list(quant_config.get("exclude"))
        + _to_list(quant_config.get("exclude_initializers"))
        + _to_list(quant_config.get("nodes_to_exclude"))
    )
    seen: set[str] = set()
    merged: list[str] = []
    for item in parts:
        if item not in seen:
            seen.add(item)
            merged.append(item)
    return merged


def _read_initializer_bytes(
    model_path: Path,
    init_meta: dict[str, Any],
    init_name: str,
    raw_inline_lookup: dict[str, bytes],
) -> bytes:
    """
    Read the raw payload bytes for a single initializer.

    Inline initializers come from ``raw_inline_lookup`` (a ``{name: raw_data}``
    map built once per model). External-data initializers are seeked + read
    from the file declared in ``init_meta['external_data']['location']``,
    resolved relative to ``model_path.parent``.
    """
    if init_meta.get("uses_external_data"):
        ed = init_meta["external_data"]
        location = ed.get("location")
        if not location:
            raise ValueError(f"External-data tensor {init_name!r} has no 'location' entry")
        ed_path = (model_path.parent / location).resolve()
        if not ed_path.is_file():
            raise FileNotFoundError(f"External-data file not found: {ed_path}")
        offset = int(ed["offset"])
        length = int(ed["length"])
        with ed_path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
            if len(data) != length:
                raise ValueError(
                    f"Unexpected EOF reading external data for {init_name!r}: got {len(data)}, expected {length}"
                )
            return data
    inline = raw_inline_lookup.get(init_name)
    if inline is None:
        raise KeyError(f"Inline raw_data not found for initializer {init_name!r}")
    return inline


def _build_inline_lookup(model: onnx.ModelProto) -> dict[str, bytes]:
    return {
        init.name: bytes(init.raw_data)
        for init in model.graph.initializer
        if not external_data_helper.uses_external_data(init) and init.raw_data
    }


def check_non_quantized_initializers_md5_unchanged(
    source_model_path: str | Path,
    output_model_path: str | Path,
    quant_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Check initializers expected to remain non-quantized for byte-identical MD5 values.

    Only initializers present on both sides and matched by ``exclude`` /
    ``exclude_initializers`` / ``nodes_to_exclude`` glob rules are checked.
    The function compares dtype / shape first, then MD5s raw payload bytes
    (inline ``raw_data`` or external-data byte ranges). When candidates exceed
    ``quant_config['max_samples']`` (default 200), the set is randomly sampled
    using ``quant_config['random_seed']`` / ``['seed']`` when provided.

    :param str | pathlib.Path source_model_path: Original pre-quantization ``.onnx``.
    :param str | pathlib.Path output_model_path: Quantized output ``.onnx``.
    :param dict[str, Any] quant_config: Configuration with at least one of
        ``exclude`` / ``exclude_initializers`` / ``nodes_to_exclude`` provided.

    :return: Report with ``ok``, candidate count, checked count, mismatches.
    :rtype: dict[str, Any]
    """
    _agent_log(
        "md5",
        "start: check_non_quantized_initializers_md5_unchanged (will read payload bytes for matched initializers)",
    )
    source_path = _path(source_model_path)
    output_path = _path(output_model_path)
    _agent_log("md5", f"source model: {source_path}")
    _agent_log("md5", f"output model: {output_path}")
    patterns = _normalize_exclude_patterns(quant_config)
    _agent_log("md5", f"exclude rules: {patterns if patterns else '(none)'}")
    max_samples_raw = quant_config.get("max_samples", 200)
    max_samples = int(max_samples_raw) if max_samples_raw is not None else 200
    if max_samples < 1:
        _agent_log("md5", "failure: max_samples < 1")
        return {
            "ok": False,
            "errors": ["max_samples must be >= 1 to run MD5 spot checks."],
            "candidate_count": 0,
            "checked_count": 0,
            "sampled": False,
            "exclude_patterns": patterns,
            "matched_samples": [],
            "mismatches": [],
            "external_data_missing": [],
        }

    seed = quant_config.get("random_seed", quant_config.get("seed"))

    errors: list[str] = []
    if not patterns:
        errors.append(
            "quant_config did not provide exclude / exclude_initializers / nodes_to_exclude; "
            "no expected non-quantized initializers can be matched."
        )
    if not source_path.is_file():
        errors.append(f"Source model file not found: {source_path}")
    if not output_path.is_file():
        errors.append(f"Output model file not found: {output_path}")
    if errors:
        for err in errors:
            _agent_log("md5", f"failure: {err}")
        return {
            "ok": False,
            "errors": errors,
            "candidate_count": 0,
            "checked_count": 0,
            "sampled": False,
            "max_samples": max_samples,
            "exclude_patterns": patterns,
            "matched_samples": [],
            "mismatches": [],
            "external_data_missing": [],
        }

    _agent_log("md5", "loading model headers (load_external_data=False)...")
    source_model = load_model_header(source_path)
    output_model = load_model_header(output_path)
    source_meta = collect_initializer_metadata(source_model)
    output_meta = collect_initializer_metadata(output_model)
    source_inline = _build_inline_lookup(source_model)
    output_inline = _build_inline_lookup(output_model)
    _agent_log("md5", f"source initializers: {len(source_meta)}, output: {len(output_meta)}")

    candidates = sorted(name for name in source_meta if _matches_any(name, patterns))
    _agent_log("md5", f"candidates matched by excludes: {len(candidates)}")
    if not candidates:
        return {
            "ok": False,
            "errors": [],
            "candidate_count": 0,
            "checked_count": 0,
            "sampled": False,
            "max_samples": max_samples,
            "exclude_patterns": patterns,
            "matched_samples": [],
            "mismatches": [],
            "external_data_missing": [],
            "note": "No initializers matched exclude rules; verify globs against the source model's initializer names.",
        }

    sampled = len(candidates) > max_samples
    if sampled:
        rng = random.Random(seed)
        checked = sorted(rng.sample(candidates, max_samples))
        _agent_log("md5", f"sampled {len(checked)}/{len(candidates)} candidates")
    else:
        checked = list(candidates)

    matched_samples: list[str] = []
    mismatches: list[dict[str, Any]] = []
    external_data_missing: set[str] = set()

    for step, name in enumerate(checked, start=1):
        if name not in output_meta:
            mismatches.append({"name": name, "reason": "missing_in_output"})
            _agent_log("md5", f"[{step}/{len(checked)}] FAIL {name}: key missing in output")
            continue
        s_meta = source_meta[name]
        o_meta = output_meta[name]
        if s_meta["dtype"] != o_meta["dtype"] or s_meta["shape"] != o_meta["shape"]:
            mismatches.append(
                {
                    "name": name,
                    "reason": "header_mismatch",
                    "source_dtype": s_meta["dtype"],
                    "output_dtype": o_meta["dtype"],
                    "source_shape": s_meta["shape"],
                    "output_shape": o_meta["shape"],
                }
            )
            _agent_log(
                "md5",
                f"[{step}/{len(checked)}] FAIL {name}: header mismatch "
                f"{s_meta['dtype']}/{o_meta['dtype']} {s_meta['shape']}/{o_meta['shape']}",
            )
            continue
        try:
            s_bytes = _read_initializer_bytes(source_path, s_meta, name, source_inline)
            o_bytes = _read_initializer_bytes(output_path, o_meta, name, output_inline)
        except FileNotFoundError as exc:
            external_data_missing.add(str(exc).split(": ", 1)[-1])
            mismatches.append({"name": name, "reason": "read_error", "detail": str(exc)})
            _agent_log("md5", f"[{step}/{len(checked)}] FAIL {name}: external-data file missing")
            continue
        except (OSError, ValueError, KeyError) as exc:
            mismatches.append({"name": name, "reason": "read_error", "detail": str(exc)})
            _agent_log("md5", f"[{step}/{len(checked)}] FAIL {name}: payload read error {exc!r}")
            continue
        s_md5 = hashlib.md5(s_bytes).hexdigest()
        o_md5 = hashlib.md5(o_bytes).hexdigest()
        if s_md5 != o_md5:
            mismatches.append({"name": name, "reason": "md5_mismatch", "source_md5": s_md5, "output_md5": o_md5})
            _agent_log("md5", f"[{step}/{len(checked)}] FAIL {name}: MD5 mismatch")
        else:
            matched_samples.append(name)
            _agent_log("md5", f"[{step}/{len(checked)}] OK {name}: MD5 matched")

    ok = not mismatches
    _agent_log(
        "md5",
        f"result: ok={ok} checked={len(checked)} matched={len(matched_samples)} "
        f"mismatches={len(mismatches)} external_data_missing={len(external_data_missing)}",
    )
    _agent_log("md5", "end: check_non_quantized_initializers_md5_unchanged")
    return {
        "ok": ok,
        "errors": [],
        "candidate_count": len(candidates),
        "checked_count": len(checked),
        "sampled": sampled,
        "max_samples": max_samples,
        "random_seed": seed,
        "exclude_patterns": patterns,
        "matched_samples": matched_samples[:100],
        "mismatches": mismatches[:200],
        "external_data_missing": sorted(external_data_missing)[:50],
    }


# --------------------------------------------------------------------------- #
# Step 3: model metadata equality (modulo Quark-injected domains)
# --------------------------------------------------------------------------- #


def _summarize_io(value_infos: list[Any]) -> list[dict[str, Any]]:
    """Convert a graph input/output proto list into a compact dict list."""
    out = []
    for vi in value_infos:
        t = vi.type.tensor_type
        shape = []
        for d in t.shape.dim:
            if d.HasField("dim_value"):
                shape.append(d.dim_value)
            elif d.HasField("dim_param"):
                shape.append(d.dim_param)
            else:
                shape.append("?")
        out.append({"name": vi.name, "dtype": _dtype_name(t.elem_type), "shape": shape})
    return out


def _opsets_excluding(model: onnx.ModelProto, excluded_domains: frozenset[str]) -> list[dict[str, Any]]:
    result = []
    for op in model.opset_import:
        domain = op.domain or "ai.onnx"
        if domain in excluded_domains:
            continue
        result.append({"domain": domain, "version": op.version})
    return sorted(result, key=lambda x: (x["domain"], x["version"]))


def check_model_metadata_equal_except_quantization(
    source_model_path: str | Path,
    quantized_model_path: str | Path,
    extra_excluded_domains: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compare model-level metadata after stripping Quark-injected opset domains.

    Checks that ``ir_version``, ``producer_name``, ``producer_version``,
    ``model_version``, ``domain``, the default-domain opset version, and the
    graph input / output signatures are identical between source and quantized
    models. Opset entries for domains in ``QUARK_INJECTED_DOMAINS`` (and any
    additional domains supplied via ``extra_excluded_domains``) are stripped
    from the quantized side before comparison — quantization is allowed to
    *add* a ``com.amd.quark`` opset entry but not to change the default domain.

    :param str | pathlib.Path source_model_path: Original ``.onnx``.
    :param str | pathlib.Path quantized_model_path: Quantized ``.onnx``.
    :param list[str] | None extra_excluded_domains: Additional domains to strip
        from the quantized side before opset equality.

    :return: ``ok`` and a list of ``mismatches`` with field-level detail.
    :rtype: dict[str, Any]
    """
    _agent_log("metadata", "start: check_model_metadata_equal_except_quantization")
    source_path = _path(source_model_path)
    quant_path = _path(quantized_model_path)
    _agent_log("metadata", f"source model: {source_path}")
    _agent_log("metadata", f"quantized model: {quant_path}")
    errors: list[str] = []
    if not source_path.is_file():
        errors.append(f"Source model not found: {source_path}")
    if not quant_path.is_file():
        errors.append(f"Quantized model not found: {quant_path}")
    if errors:
        for err in errors:
            _agent_log("metadata", f"failure: {err}")
        return {
            "ok": False,
            "errors": errors,
            "mismatches": [],
            "mismatch_count": 0,
            "source_model_path": str(source_path),
            "quantized_model_path": str(quant_path),
            "quark_injected_domains": [],
        }

    excluded = frozenset(QUARK_INJECTED_DOMAINS) | frozenset(extra_excluded_domains or [])
    source_model = load_model_header(source_path)
    quant_model = load_model_header(quant_path)

    mismatches: list[str] = []

    for field in ("ir_version", "producer_name", "producer_version", "model_version", "domain"):
        s_val = getattr(source_model, field)
        q_val = getattr(quant_model, field)
        if s_val != q_val:
            mismatches.append(f"{field}: {s_val!r} / {q_val!r}")

    s_opsets = _opsets_excluding(source_model, frozenset())
    q_opsets_full = _opsets_excluding(quant_model, frozenset())
    q_opsets_stripped = _opsets_excluding(quant_model, excluded)
    quark_added = sorted(
        {(op["domain"], op["version"]) for op in q_opsets_full} - {(op["domain"], op["version"]) for op in s_opsets}
    )

    if s_opsets != q_opsets_stripped:
        mismatches.append(
            f"opset_import (after stripping Quark domains from quantized): {s_opsets} / {q_opsets_stripped}"
        )

    s_inputs = _summarize_io(list(source_model.graph.input))
    q_inputs = _summarize_io(list(quant_model.graph.input))
    if s_inputs != q_inputs:
        mismatches.append(f"graph.input: {s_inputs} / {q_inputs}")

    s_outputs = _summarize_io(list(source_model.graph.output))
    q_outputs = _summarize_io(list(quant_model.graph.output))
    if s_outputs != q_outputs:
        mismatches.append(f"graph.output: {s_outputs} / {q_outputs}")

    ok = not mismatches
    _agent_log(
        "metadata",
        f"result: ok={ok} mismatch_count={len(mismatches)} quark_added_opsets={quark_added}",
    )
    for line in mismatches[:25]:
        _agent_log("metadata", f"  diff: {line}")
    _agent_log("metadata", "end: check_model_metadata_equal_except_quantization")
    return {
        "ok": ok,
        "errors": [],
        "mismatches": mismatches[:200],
        "mismatch_count": len(mismatches),
        "source_model_path": str(source_path),
        "quantized_model_path": str(quant_path),
        "quark_injected_domains": [{"domain": d, "version": v} for d, v in quark_added],
        "stripped_domains": sorted(excluded),
    }


# --------------------------------------------------------------------------- #
# Step 4: fuzzy node / op / initializer summary
# --------------------------------------------------------------------------- #


def get_fuzzy_node_op_summary(model_path: str | Path) -> dict[str, Any]:
    """
    Collect a header-only structural summary of a quantized ONNX model.

    Surfaces:

    - **op_type_counts**: full op-type histogram for the graph.
    - **node_patterns**: canonical node-name patterns (numeric path segments
      collapsed to ``*``) with per-pattern ``op_type_counts``. A pattern that
      maps to multiple op types signals partial quantization.
    - **initializer_dtype_patterns**: canonical initializer-name patterns with
      per-pattern ``dtype_counts``. Mixed dtypes within one pattern usually
      indicates partial quantization (e.g. some layers quantized, others not).
    - **qdq_op_count**: number of ``QuantizeLinear`` / ``DequantizeLinear`` /
      ``MatMulNBits`` nodes.
    - **quark_custom_op_count**: number of nodes whose op type is in
      ``QUARK_CUSTOM_OP_TYPES`` (BFP / MX / Extended).
    - **quark_opset_domains**: subset of ``opset_import`` that overlaps with
      ``QUARK_INJECTED_DOMAINS``.
    - **quantization_signal**: ``"present"`` / ``"missing"`` / ``"partial"``
      depending on QDQ and custom-op counts.

    :param str | pathlib.Path model_path: Path to an ``.onnx`` file.

    :return: Summary dict (see fields above).
    :rtype: dict[str, Any]
    """
    _agent_log("fuzzy", "start: get_fuzzy_node_op_summary (header only)")
    resolved = _path(model_path)
    _agent_log("fuzzy", f"resolved model path -> {resolved}")
    if not resolved.is_file():
        _agent_log("fuzzy", f"failure: model not found: {resolved}")
        return {
            "ok": False,
            "errors": [f"Model file not found: {resolved}"],
            "model_path": str(resolved),
        }
    model = load_model_header(resolved)
    graph = model.graph

    op_counts: Counter[str] = Counter()
    node_pattern_order: list[str] = []
    node_pattern_op_counts: dict[str, Counter[str]] = {}
    qdq_count = 0
    quark_custom_count = 0

    for node in graph.node:
        op_counts[node.op_type] += 1
        if node.op_type in QDQ_OP_TYPES:
            qdq_count += 1
        if node.op_type in QUARK_CUSTOM_OP_TYPES or node.domain in QUARK_INJECTED_DOMAINS:
            quark_custom_count += 1
        node_name = node.name or f"<anonymous_{node.op_type}>"
        pattern = canonicalize_name(node_name)
        if pattern not in node_pattern_op_counts:
            node_pattern_op_counts[pattern] = Counter()
            node_pattern_order.append(pattern)
        node_pattern_op_counts[pattern][node.op_type] += 1

    init_pattern_order: list[str] = []
    init_pattern_dtype_counts: dict[str, Counter[str]] = {}
    for init in graph.initializer:
        pattern = canonicalize_name(init.name)
        if pattern not in init_pattern_dtype_counts:
            init_pattern_dtype_counts[pattern] = Counter()
            init_pattern_order.append(pattern)
        init_pattern_dtype_counts[pattern][_dtype_name(init.data_type)] += 1

    quark_opset_domains = sorted({op.domain for op in model.opset_import if op.domain in QUARK_INJECTED_DOMAINS})

    if qdq_count + quark_custom_count == 0:
        signal = "missing"
    elif qdq_count > 0 and quark_custom_count > 0:
        signal = "present"
    else:
        signal = "present"

    node_patterns = []
    for pattern in node_pattern_order:
        counts = node_pattern_op_counts[pattern]
        sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        node_patterns.append({"pattern": pattern, "op_type_counts": dict(sorted_items)})

    init_patterns = []
    for pattern in init_pattern_order:
        counts = init_pattern_dtype_counts[pattern]
        sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        init_patterns.append({"pattern": pattern, "dtype_counts": dict(sorted_items)})

    mixed_node_patterns = sum(1 for item in node_patterns if len(item["op_type_counts"]) > 1)
    mixed_init_patterns = sum(1 for item in init_patterns if len(item["dtype_counts"]) > 1)

    _agent_log(
        "fuzzy",
        f"op types: {len(op_counts)}, node patterns: {len(node_patterns)} (mixed: {mixed_node_patterns}), "
        f"init patterns: {len(init_patterns)} (mixed: {mixed_init_patterns})",
    )
    _agent_log(
        "fuzzy",
        f"qdq_op_count={qdq_count}, quark_custom_op_count={quark_custom_count}, "
        f"quark_opset_domains={quark_opset_domains}, signal={signal}",
    )
    _agent_log("fuzzy", "end: get_fuzzy_node_op_summary")
    return {
        "ok": True,
        "errors": [],
        "model_path": str(resolved),
        "op_type_counts": dict(sorted(op_counts.items(), key=lambda x: (-x[1], x[0]))),
        "qdq_op_count": qdq_count,
        "quark_custom_op_count": quark_custom_count,
        "quark_opset_domains": quark_opset_domains,
        "quantization_signal": signal,
        "node_patterns_count": len(node_patterns),
        "node_patterns_mixed_op": mixed_node_patterns,
        "node_patterns": node_patterns[:300],
        "initializer_patterns_count": len(init_patterns),
        "initializer_patterns_mixed_dtype": mixed_init_patterns,
        "initializer_patterns": init_patterns[:300],
    }


# --------------------------------------------------------------------------- #
# Mock model builders for the embedded self-test
# --------------------------------------------------------------------------- #


def _make_tensor_inline(name: str, dtype: int, dims: list[int], raw: bytes) -> Any:
    t = TensorProto()
    t.name = name
    t.data_type = dtype
    t.dims.extend(dims)
    t.raw_data = raw
    return t


def _make_minimal_model(
    opset_version: int,
    initializers: list[Any],
    extra_opset_domain: str | None = None,
    extra_opset_version: int = 1,
    extra_nodes: list[Any] | None = None,
) -> onnx.ModelProto:
    """Build a tiny model with one Identity node so the graph is valid."""
    graph = onnx.GraphProto()
    graph.name = "g"

    # one float input/output so the graph is valid
    inp = onnx.ValueInfoProto()
    inp.name = "in"
    inp.type.tensor_type.elem_type = TensorProto.FLOAT
    d = inp.type.tensor_type.shape.dim.add()
    d.dim_value = 1
    outp = onnx.ValueInfoProto()
    outp.name = "out"
    outp.type.tensor_type.elem_type = TensorProto.FLOAT
    d = outp.type.tensor_type.shape.dim.add()
    d.dim_value = 1
    graph.input.append(inp)
    graph.output.append(outp)

    n = onnx.NodeProto()
    n.op_type = "Identity"
    n.name = "Identity_0"
    n.input.append("in")
    n.output.append("out")
    graph.node.append(n)

    for ex in extra_nodes or []:
        graph.node.append(ex)
    for init in initializers:
        graph.initializer.append(init)

    model = onnx.ModelProto()
    model.ir_version = 9
    model.producer_name = "quark-validator-selftest"
    model.producer_version = "0.1"
    model.graph.CopyFrom(graph)
    op = model.opset_import.add()
    op.domain = ""
    op.version = opset_version
    if extra_opset_domain is not None:
        op2 = model.opset_import.add()
        op2.domain = extra_opset_domain
        op2.version = extra_opset_version
    return model


def _selftest_exported_symbols() -> int:
    """Run mock-data checks for symbols listed in ``__all__``."""
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # -------- get_fuzzy_node_op_summary ----------------------------------
        fuzzy_dir = base / "fuzzy"
        fuzzy_dir.mkdir()
        fuzzy_inits = [
            _make_tensor_inline("model.layers.0.weight", TensorProto.INT8, [2], b"\x01\x02"),
            _make_tensor_inline("model.layers.1.weight", TensorProto.INT8, [2], b"\x03\x04"),
            _make_tensor_inline("model.head.weight", TensorProto.FLOAT, [1], b"\x00\x00\x00\x00"),
        ]
        # Add a QuantizeLinear + a com.amd.quark BFPQuantizeDequantize node.
        qdq_node = onnx.NodeProto()
        qdq_node.op_type = "QuantizeLinear"
        qdq_node.name = "QuantizeLinear_0"
        qdq_node.input.extend(["in", "model.layers.0.weight"])
        qdq_node.output.append("q0")
        bfp_node = onnx.NodeProto()
        bfp_node.op_type = "BFPQuantizeDequantize"
        bfp_node.name = "BFPQuantizeDequantize_0"
        bfp_node.domain = "com.amd.quark"
        bfp_node.input.append("in")
        bfp_node.output.append("bfp0")
        fuzzy_model = _make_minimal_model(
            opset_version=17,
            initializers=fuzzy_inits,
            extra_opset_domain="com.amd.quark",
            extra_opset_version=1,
            extra_nodes=[qdq_node, bfp_node],
        )
        fuzzy_path = fuzzy_dir / "model.onnx"
        onnx.save(fuzzy_model, str(fuzzy_path))
        fuzzy_summary = get_fuzzy_node_op_summary(fuzzy_path)
        if not fuzzy_summary.get("ok"):
            failures.append(f"get_fuzzy_node_op_summary: not ok: {fuzzy_summary}")
        if fuzzy_summary.get("qdq_op_count") != 1:
            failures.append(f"get_fuzzy_node_op_summary: expected 1 qdq, got {fuzzy_summary.get('qdq_op_count')}")
        if fuzzy_summary.get("quark_custom_op_count") != 1:
            failures.append(
                f"get_fuzzy_node_op_summary: expected 1 quark custom op, got {fuzzy_summary.get('quark_custom_op_count')}"
            )
        if "com.amd.quark" not in fuzzy_summary.get("quark_opset_domains", []):
            failures.append("get_fuzzy_node_op_summary: missing com.amd.quark opset domain")
        # initializer pattern canonicalization
        init_patterns = {
            item["pattern"]: item["dtype_counts"] for item in fuzzy_summary.get("initializer_patterns", [])
        }
        if "model.layers.*.weight" not in init_patterns:
            failures.append(f"get_fuzzy_node_op_summary: expected canonical pattern, got {list(init_patterns)}")

        # -------- check_auxiliary_files_copied -------------------------------
        src_aux = base / "src_aux"
        out_aux = base / "out_aux"
        src_aux.mkdir()
        out_aux.mkdir()
        (src_aux / "README.md").write_text("readme", encoding="utf-8")
        (out_aux / "README.md").write_text("readme", encoding="utf-8")
        # Write differing .onnx payloads — should be IGNORED.
        onnx.save(_make_minimal_model(17, []), str(src_aux / "model.onnx"))
        onnx.save(
            _make_minimal_model(17, [_make_tensor_inline("w", TensorProto.INT8, [1], b"\x01")]),
            str(out_aux / "model.onnx"),
        )
        aux_rep = check_auxiliary_files_copied(src_aux, out_aux, max_examples=20)
        if not aux_rep.get("ok"):
            failures.append(f"check_auxiliary_files_copied: expected ok, got {aux_rep}")

        # missing file case
        (src_aux / "only_in_source.txt").write_text("only", encoding="utf-8")
        aux_rep2 = check_auxiliary_files_copied(src_aux, out_aux, max_examples=20)
        if aux_rep2.get("ok"):
            failures.append("check_auxiliary_files_copied: expected FAIL when source has extra non-weight file")

        # -------- check_model_metadata_equal_except_quantization -------------
        meta_dir = base / "meta"
        meta_dir.mkdir()
        src_meta_model = _make_minimal_model(17, [])
        # Quantized version adds com.amd.quark to opset (should still be ok).
        quant_meta_model = _make_minimal_model(17, [], extra_opset_domain="com.amd.quark")
        src_meta_path = meta_dir / "src.onnx"
        quant_meta_path = meta_dir / "quant.onnx"
        onnx.save(src_meta_model, str(src_meta_path))
        onnx.save(quant_meta_model, str(quant_meta_path))
        meta_rep = check_model_metadata_equal_except_quantization(src_meta_path, quant_meta_path)
        if not meta_rep.get("ok"):
            failures.append(f"check_model_metadata_equal_except_quantization: expected ok, got {meta_rep}")
        if not any(d["domain"] == "com.amd.quark" for d in meta_rep.get("quark_injected_domains", [])):
            failures.append(
                "check_model_metadata_equal_except_quantization: expected com.amd.quark in quark_injected_domains"
            )

        # Now bump opset on the quantized side — should FAIL.
        bad_meta_model = _make_minimal_model(18, [], extra_opset_domain="com.amd.quark")
        bad_meta_path = meta_dir / "bad.onnx"
        onnx.save(bad_meta_model, str(bad_meta_path))
        meta_bad = check_model_metadata_equal_except_quantization(src_meta_path, bad_meta_path)
        if meta_bad.get("ok"):
            failures.append("check_model_metadata_equal_except_quantization: expected FAIL on opset bump")

        # -------- check_non_quantized_initializers_md5_unchanged -------------
        md5_dir = base / "md5"
        md5_dir.mkdir()
        payload = b"keep_payload_bytes_12345678"
        src_md5_model = _make_minimal_model(
            17, [_make_tensor_inline("model.layers.0.keep.weight", TensorProto.UINT8, [len(payload)], payload)]
        )
        ok_md5_model = _make_minimal_model(
            17, [_make_tensor_inline("model.layers.0.keep.weight", TensorProto.UINT8, [len(payload)], payload)]
        )
        bad_md5_model = _make_minimal_model(
            17, [_make_tensor_inline("model.layers.0.keep.weight", TensorProto.UINT8, [len(payload)], payload[::-1])]
        )
        src_md5_path = md5_dir / "src.onnx"
        ok_md5_path = md5_dir / "ok.onnx"
        bad_md5_path = md5_dir / "bad.onnx"
        onnx.save(src_md5_model, str(src_md5_path))
        onnx.save(ok_md5_model, str(ok_md5_path))
        onnx.save(bad_md5_model, str(bad_md5_path))
        qconf: dict[str, Any] = {"exclude": ["*.keep.weight"], "max_samples": 20, "random_seed": 0}
        md5_ok = check_non_quantized_initializers_md5_unchanged(src_md5_path, ok_md5_path, qconf)
        if not md5_ok.get("ok") or "model.layers.0.keep.weight" not in md5_ok.get("matched_samples", []):
            failures.append(f"check_non_quantized_initializers_md5_unchanged success case: {md5_ok}")
        md5_bad = check_non_quantized_initializers_md5_unchanged(src_md5_path, bad_md5_path, qconf)
        if md5_bad.get("ok"):
            failures.append("check_non_quantized_initializers_md5_unchanged: expected FAIL on byte change")
        reasons = {m.get("reason") for m in md5_bad.get("mismatches", [])}
        if "md5_mismatch" not in reasons:
            failures.append(f"check_non_quantized_initializers_md5_unchanged: expected md5_mismatch in {md5_bad}")

        # No-exclude case
        md5_none = check_non_quantized_initializers_md5_unchanged(src_md5_path, ok_md5_path, {})
        if md5_none.get("ok"):
            failures.append("check_non_quantized_initializers_md5_unchanged: expected FAIL when no excludes given")

        # Sampling branch
        samp_inits_src = [
            _make_tensor_inline(f"model.layers.{i}.keep.weight", TensorProto.UINT8, [2], b"\xaa\xbb") for i in range(8)
        ]
        samp_inits_out = [
            _make_tensor_inline(f"model.layers.{i}.keep.weight", TensorProto.UINT8, [2], b"\xaa\xbb") for i in range(8)
        ]
        src_samp_path = md5_dir / "src_samp.onnx"
        out_samp_path = md5_dir / "out_samp.onnx"
        onnx.save(_make_minimal_model(17, samp_inits_src), str(src_samp_path))
        onnx.save(_make_minimal_model(17, samp_inits_out), str(out_samp_path))
        qconf_samp = {"exclude": ["*.keep.weight"], "max_samples": 3, "random_seed": 42}
        md5_samp = check_non_quantized_initializers_md5_unchanged(src_samp_path, out_samp_path, qconf_samp)
        if not md5_samp.get("ok"):
            failures.append(f"check_non_quantized_initializers_md5_unchanged sampling: {md5_samp}")
        if not md5_samp.get("sampled") or md5_samp.get("checked_count") != 3:
            failures.append(
                f"check_non_quantized_initializers_md5_unchanged: expected sampled=True checked=3, got {md5_samp}"
            )

    if failures:
        for line in failures:
            print(line, file=sys.stderr)
        return 1

    print("exported symbols self-test (__all__): ok")
    return 0


def run_selftest() -> int:
    """Public entry point for the embedded self-test. Returns 0 on success, 1 on failure."""
    return _selftest_exported_symbols()


if __name__ == "__main__":
    raise SystemExit(_selftest_exported_symbols())
