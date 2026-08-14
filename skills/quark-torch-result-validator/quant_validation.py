#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Source file for Claude skill ``quark-torch-result-validator``.
# Lives at ``.claude/skills/quark-torch-result-validator/quant_validation.py``.

"""Header-only quantization result validation helpers for Quark outputs.

This module performs lightweight checks for large model export outputs. Most
logic parses only safetensors headers, index files, and regular file metadata
without decoding full tensor payloads.

**Agent diagnostics**: the four ``__all__`` entry points emit line-oriented
progress and failure clues to **stderr** using ``[quant-validation][tag]``.
Machine-consumable fields such as ``ok`` and ``errors`` remain in return values
so diagnostics are not mixed into JSON output.

Exception: `check_non_quantized_tensors_md5_unchanged` reads raw bytes for
tensors matched by exclude rules and calculates MD5 values to confirm that
non-quantized weights are byte-identical. Use
``quant_config["max_samples"]`` for large models.

Also, `check_config_json_equal_except_quantization` reads two ``config.json``
files and deeply compares them after recursively removing quantization-related
keys. It does not depend on safetensors files.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import random
import re
import struct
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

INDEX_SEGMENT = re.compile(r"(?:(?<=^)|(?<=\.))\d+(?=\.|$)")

# These files are weights or configs rewritten by quantization/export, so they
# are excluded from auxiliary asset copy checks.
AUXILIARY_IGNORE_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.ckpt",
    "*.gguf",
    "*.msgpack",
    "*.onnx",
    "*.pb",
    "*.pt",
    "*.pth",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
]
AUXILIARY_IGNORE_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".cache", ".venv", "node_modules"}

# Key names recursively removed from ``config.json`` trees at any level. By
# default, only the common HF ``quantization_config`` key is removed. Callers may
# append names or fully replace the set through function arguments.
DEFAULT_CONFIG_QUANT_IGNORE_KEY_NAMES: frozenset[str] = frozenset({"quantization_config"})

# ``__all__`` limits symbols exposed by ``from quant_validation import *``.
# Export only the stable common entry points; import other helpers by full name.
__all__ = [
    # Auxiliary assets: after ignoring weights and known config/index files,
    # compare non-weight file sets and content on both sides.
    "check_auxiliary_files_copied",
    # Expected non-quantized tensors: use excludes from ``quant_config`` and
    # spot-check raw payload bytes with MD5.
    "check_non_quantized_tensors_md5_unchanged",
    # Two config.json files: recursively strip quantization keys, then compare
    # the remaining structure.
    "check_config_json_equal_except_quantization",
    # Safetensors headers: canonical tensor name patterns and dtype counts per
    # pattern.
    "get_fuzzy_tensor_names",
]


def canonicalize_name(name: str) -> str:
    """
    Replace pure numeric path segments with ``*``.

    Many LLM layers or blocks differ only by numeric indexes. Collapsing pure
    numeric path segments deduplicates them into a compact structural view and
    reduces context usage.

    :param str name: Dotted module or tensor name.

    :return: Canonicalized name.
    :rtype: str
    """
    return INDEX_SEGMENT.sub("*", name)


def _path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _agent_log(tag: str, message: str) -> None:
    """
    Write agent-readable progress and exception clues to stderr.

    This avoids polluting stdout when it is captured for machine-readable output
    such as piped JSON.

    :param str tag: Short tag, such as ``fuzzy``, ``aux``, ``md5``, or ``config``.
    :param str message: Single-line message.
    :rtype: None
    """
    print(f"[quant-validation][{tag}] {message}", file=sys.stderr, flush=True)


def _numel(shape: list[int] | tuple[int, ...]) -> int:
    return math.prod(int(dim) for dim in shape)


def _payload_nbytes(info: dict[str, Any]) -> int:
    # ``data_offsets`` in the safetensors header directly gives payload byte
    # counts without reading actual tensor content.
    offsets = info.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        return 0
    return int(offsets[1]) - int(offsets[0])


def _sha256_file(path: Path, max_hash_bytes: int) -> str | None:
    # Large files use size checks only, avoiding excessive reads for auxiliary
    # asset validation.
    if path.stat().st_size > max_hash_bytes:
        return None

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _matches_any(name: str, patterns: list[str]) -> bool:
    # Match both the full tensor name and the module name with the final segment
    # removed so exclude rules can be reused across weight/module names.
    return any(
        fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name.rsplit(".", 1)[0], pattern) for pattern in patterns
    )


def _matches_path_pattern(relative_path: Path, patterns: list[str]) -> bool:
    path = relative_path.as_posix()
    name = relative_path.name
    return any(fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _is_auxiliary_file(relative_path: Path, ignore_patterns: list[str]) -> bool:
    # Auxiliary assets are non-weight files such as README, tokenizer files,
    # assets, and inference helpers. Weights, indexes, and export-rewritten
    # configs are handled by other checks.
    if any(part in AUXILIARY_IGNORE_DIRS for part in relative_path.parts):
        return False
    return not _matches_path_pattern(relative_path, ignore_patterns)


def _collect_auxiliary_files(model_dir: Path, ignore_patterns: list[str]) -> list[Path]:
    files = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(model_dir)
        if _is_auxiliary_file(relative_path, ignore_patterns):
            files.append(relative_path)
    return files


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    """
    Read only the JSON header from a safetensors shard.

    The first 8 bytes of a safetensors file store the header length. Reading the
    header gives tensor names, dtypes, shapes, and data offsets without loading
    large weights.

    :param str | pathlib.Path path: Safetensors shard path.

    :return: Parsed header.
    :rtype: dict[str, Any]
    """
    shard_path = _path(path)
    with shard_path.open("rb") as handle:
        header_len_bytes = handle.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"Invalid safetensors file header: {shard_path}")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_bytes = handle.read(header_len)
    return json.loads(header_bytes)


def tensor_items_from_header(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Return tensor metadata entries from a safetensors header.

    Filter out ``__metadata__`` and keep only real tensor entries.

    :param dict[str, Any] header: Parsed safetensors header.

    :return: Tensor metadata by name.
    :rtype: dict[str, dict[str, Any]]
    """
    return {name: info for name, info in header.items() if name != "__metadata__" and isinstance(info, dict)}


def load_index(model_dir: str | Path) -> dict[str, Any] | None:
    """
    Load ``model.safetensors.index.json`` when present.

    The index maps tensor keys to shard files and is the main source for
    determining whether sharded outputs are complete and locatable.

    :param str | pathlib.Path model_dir: Model directory.

    :return: Parsed index or ``None``.
    :rtype: dict[str, Any] | None
    """
    index_path = _path(model_dir) / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    with index_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_safetensors_shards(model_path: str | Path) -> list[Path]:
    """
    Resolve a model path to safetensors shard files.

    When a directory has an index, resolve shards from that index first so the
    check order matches HuggingFace sharding. Without an index, fall back to
    scanning ``*.safetensors`` files.

    :param str | pathlib.Path model_path: Model directory containing
        ``*.safetensors`` weights and optionally ``model.safetensors.index.json``,
        or a single ``*.safetensors`` shard path.

    :return: Ordered shard paths.
    :rtype: list[pathlib.Path]
    """
    resolved = _path(model_path)
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"Not a file or directory: {resolved}")

    index = load_index(resolved)
    if index is not None:
        weight_map = index.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        return [resolved / shard_name for shard_name in shard_names]

    shards = sorted(resolved.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors shards found under {resolved}")
    return shards


def collect_tensor_metadata(model_path: str | Path) -> dict[str, dict[str, Any]]:
    """
    Collect tensor metadata from safetensors headers only.

    Collect tensor names, shard names, dtypes, shapes, and payload byte counts in
    one lightweight metadata view used by dtype summaries, scale pairing, and
    aggregate validation.

    :param str | pathlib.Path model_path: Model directory containing
        ``*.safetensors`` files, or a single ``*.safetensors`` shard path.

    :return: Tensor metadata keyed by tensor name.
    :rtype: dict[str, dict[str, Any]]
    """
    tensors: dict[str, dict[str, Any]] = {}
    for shard_path in resolve_safetensors_shards(model_path):
        if not shard_path.is_file():
            continue
        header = read_safetensors_header(shard_path)
        for tensor_name, info in tensor_items_from_header(header).items():
            tensors[tensor_name] = {
                "shard": shard_path.name,
                "dtype": str(info.get("dtype", "UNKNOWN")),
                "shape": list(info.get("shape", [])),
                "payload_nbytes": _payload_nbytes(info),
            }
    return tensors


def get_fuzzy_tensor_names(model_path: str | Path) -> list[dict[str, Any]]:
    """
    Collect tensors from safetensors **JSON headers**, collapse pure numeric path
    segments through ``canonicalize_name`` into compact patterns, and summarize
    **dtype counts** for all tensors under each pattern.

    If a single pattern contains multiple dtypes, for example some FP8 layers
    and some BF16 layers, ``dtype_counts`` shows whether quantization is
    consistent. This fills the gap left by pattern names alone.

    Result order follows each pattern's **first appearance**. Each pattern count
    covers **all** tensors mapped to that pattern, not only the first tensor.

    :param str | pathlib.Path model_path: Model path with **no default**. Common
        usages are:

        - **Model directory**: a folder with ``*.safetensors`` shards. If
          ``model.safetensors.index.json`` exists, only shards listed in the
          index are traversed, matching HF sharding order. Otherwise,
          ``*.safetensors`` files in the directory are traversed by sorted name.
        - **Single shard file**: pass a specific ``xxx.safetensors`` path to read
          only that file's header.

        This function **does not load tensor data**. It reads only the header at
        the front of each shard. Missing or unreadable ``*.safetensors`` files
        are handled by ``resolve_safetensors_shards`` or skipped in the loop.

    :return: Items of the form
        ``{"pattern": str, "dtype_counts": {"DTYPE": count, ...}}``.
        ``dtype_counts`` is sorted by descending count, then dtype name.
    :rtype: list[dict[str, Any]]
    """
    _agent_log("fuzzy", "start: get_fuzzy_tensor_names (safetensors header only)")
    resolved = _path(model_path)
    _agent_log("fuzzy", f"resolved path -> {resolved}")
    shards = resolve_safetensors_shards(model_path)
    _agent_log("fuzzy", f"shard count: {len(shards)}")
    for idx, sp in enumerate(shards[:30]):
        _agent_log("fuzzy", f"  shard[{idx}]: {sp.name}")
    if len(shards) > 30:
        _agent_log("fuzzy", f"  ... {len(shards) - 30} more shards not listed individually")

    pattern_order: list[str] = []
    counts_by_pattern: dict[str, Counter[str]] = {}

    for shard_path in shards:
        if not shard_path.is_file():
            _agent_log("fuzzy", f"skip non-file path: {shard_path}")
            continue
        header = read_safetensors_header(shard_path)
        items = tensor_items_from_header(header)
        _agent_log("fuzzy", f"read header: {shard_path.name}, tensor entries={len(items)}")
        for tensor_name, info in items.items():
            pattern = canonicalize_name(tensor_name)
            dtype = str(info.get("dtype", "UNKNOWN"))
            if pattern not in counts_by_pattern:
                counts_by_pattern[pattern] = Counter()
                pattern_order.append(pattern)
            counts_by_pattern[pattern][dtype] += 1

    result: list[dict[str, Any]] = []
    for pattern in pattern_order:
        counter = counts_by_pattern[pattern]
        sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        result.append({"pattern": pattern, "dtype_counts": {dt: c for dt, c in sorted_items}})  # noqa: C416

    multi_dtype = sum(1 for item in result if len(item["dtype_counts"]) > 1)
    _agent_log(
        "fuzzy",
        f"summary: {len(result)} canonical patterns, {multi_dtype} with multiple dtypes "
        "(possible mixed precision or partially unquantized weights)",
    )
    for i, item in enumerate(result[:15]):
        _agent_log("fuzzy", f"  pattern[{i}]: {item['pattern']!r} -> {item['dtype_counts']}")
    if len(result) > 15:
        _agent_log("fuzzy", f"  ... {len(result) - 15} more patterns are in the return value")
    _agent_log("fuzzy", "end: get_fuzzy_tensor_names")
    return result


def check_auxiliary_files_copied(
    source_model_dir: str | Path,
    quantized_model_dir: str | Path,
    ignore: list[str] | None = None,
    max_examples: int = 50,
    max_hash_bytes: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    """
    Check whether the quantized output copied auxiliary assets from the source model.

    Quantization rewrites ``*.safetensors`` files and some ``config``/index
    files. ``README``, ``LICENSE``, tokenizer files, ``assets/``, ``inference/``,
    and similar non-weight files should appear unchanged in the output
    directory. After applying the default ignore list, this function compares
    the remaining path sets and file content. Small files use SHA256; very large
    files use size checks only.

    :param str | pathlib.Path source_model_dir: Source model root directory. No
        default; the caller must provide it.
    :param str | pathlib.Path quantized_model_dir: Quantized model output root
        directory. No default; the caller must provide it.
    :param list[str] | None ignore: Additional relative path or file-name glob
        patterns to ignore. These are merged with ``AUXILIARY_IGNORE_PATTERNS``.
        Defaults to ``None``, which uses only the module defaults.
    :param int max_examples: Maximum examples returned for list-like fields to
        control report length. Total counts are stored in ``*_count`` fields.
        Defaults to ``50``.
    :param int max_hash_bytes: Calculate SHA256 only for files at or below this
        size. Larger files rely on size comparison and are recorded in
        ``hash_skipped_files``. Defaults to ``268435456`` (256 MiB), or
        ``256 * 1024 * 1024``.

    :return: Auxiliary file alignment report, including missing, extra,
        mismatched, and directory hint fields.
    :rtype: dict[str, Any]
    """
    _agent_log("aux", "start: check_auxiliary_files_copied (ignoring weights and known config/index files)")
    source_path = _path(source_model_dir)
    output_path = _path(quantized_model_dir)
    _agent_log("aux", f"source directory: {source_path}")
    _agent_log("aux", f"quantized output directory: {output_path}")
    ignore_patterns = AUXILIARY_IGNORE_PATTERNS + (ignore or [])
    _agent_log("aux", f"merged ignore pattern count: {len(ignore_patterns)} (including module defaults)")
    errors = []
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
        "aux", f"source auxiliary file count: {len(source_files)}, output auxiliary file count: {len(output_files)}"
    )
    source_set = set(source_files)
    output_set = set(output_files)

    missing_files = sorted(source_set - output_set, key=lambda item: item.as_posix())
    extra_files = sorted(output_set - source_set, key=lambda item: item.as_posix())
    _agent_log(
        "aux",
        f"path set diff: output missing {len(missing_files)}, output extra {len(extra_files)}; "
        "will compare content/size for the intersection",
    )
    if missing_files[:10]:
        for rel in missing_files[:10]:
            _agent_log("aux", f"  missing: {rel.as_posix()}")
    if extra_files[:10]:
        for rel in extra_files[:10]:
            _agent_log("aux", f"  extra: {rel.as_posix()}")

    mismatched_files = []
    hash_skipped_files = []

    for relative_path in sorted(source_set & output_set, key=lambda item: item.as_posix()):
        source_file = source_path / relative_path
        output_file = output_path / relative_path
        source_size = source_file.stat().st_size
        output_size = output_file.stat().st_size
        if source_size != output_size:
            mismatched_files.append(
                {
                    "path": relative_path.as_posix(),
                    "source_size": source_size,
                    "output_size": output_size,
                    "reason": "size_mismatch",
                }
            )
            _agent_log(
                "aux",
                f"mismatch(size): {relative_path.as_posix()} source={source_size} output={output_size}",
            )
            continue

        source_hash = _sha256_file(source_file, max_hash_bytes)
        output_hash = _sha256_file(output_file, max_hash_bytes)
        if source_hash is None or output_hash is None:
            # Auxiliary files above the threshold only record a skipped hash;
            # size was already compared above.
            hash_skipped_files.append(relative_path.as_posix())
            _agent_log(
                "aux",
                f"skip SHA256 (file larger than max_hash_bytes={max_hash_bytes}): "
                f"{relative_path.as_posix()} (size already compared)",
            )
            continue
        if source_hash != output_hash:
            mismatched_files.append(
                {
                    "path": relative_path.as_posix(),
                    "source_sha256": source_hash,
                    "output_sha256": output_hash,
                    "reason": "hash_mismatch",
                }
            )
            _agent_log(
                "aux",
                f"mismatch(hash): {relative_path.as_posix()} source SHA256={source_hash[:16]}... "
                f"output={output_hash[:16]}...",
            )

    source_dirs = sorted({path.parent.as_posix() for path in source_files if path.parent.as_posix() != "."})
    missing_dirs = sorted({path.parent.as_posix() for path in missing_files if path.parent.as_posix() != "."})

    ok = not missing_files and not mismatched_files
    _agent_log(
        "aux",
        f"result: ok={ok} missing={len(missing_files)} content/size_mismatch={len(mismatched_files)} "
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
        "missing_files": [path.as_posix() for path in missing_files[:max_examples]],
        "mismatched_files": mismatched_files[:max_examples],
        "extra_files": [path.as_posix() for path in extra_files[:max_examples]],
        "hash_skipped_files": hash_skipped_files[:max_examples],
        "errors": [],
    }


def _normalize_exclude_patterns(quant_config: dict[str, Any]) -> list[str]:
    """Parse ``exclude`` / ``exclude_layers_name`` glob lists from a quantization config.

    Both keys are merged when both are present. A warning is emitted to stderr when
    both are set so callers are aware that neither is silently dropped.
    """

    def _to_list(raw: Any) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, (list, tuple)):  # noqa: UP038
            return [str(item) for item in raw]
        return [str(raw)]

    exclude = _to_list(quant_config.get("exclude"))
    exclude_layers_name = _to_list(quant_config.get("exclude_layers_name"))

    if exclude and exclude_layers_name:
        _agent_log("md5", "warning: both 'exclude' and 'exclude_layers_name' are set; merging both lists")

    seen: set[str] = set()
    merged: list[str] = []
    for item in exclude + exclude_layers_name:
        if item not in seen:
            seen.add(item)
            merged.append(item)
    return merged


def _md5_safetensors_tensor_payload(shard_path: Path, tensor_name: str) -> str:
    """
    Read a tensor's raw byte range according to safetensors layout and MD5 it.

    MD5 is used for integrity equality only (detecting byte-level changes between
    source and quantized shards); it is not a security primitive.

    ``data_offsets`` is relative to the tensor data region after the header, so
    the absolute offset is ``8 + header_len + data_offsets[0]``.
    """
    shard = _path(shard_path)
    hasher = hashlib.md5()
    with shard.open("rb") as handle:
        header_len_raw = handle.read(8)
        if len(header_len_raw) != 8:
            raise ValueError(f"Invalid safetensors header length: {shard}")
        header_len = struct.unpack("<Q", header_len_raw)[0]
        header_bytes = handle.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"Incomplete safetensors header: {shard}")
        header = json.loads(header_bytes)
        if tensor_name == "__metadata__" or tensor_name not in header:
            raise KeyError(f"Tensor not found in shard header: {tensor_name!r} @ {shard.name}")
        info = header[tensor_name]
        if not isinstance(info, dict):
            raise KeyError(tensor_name)
        offsets = info.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"Invalid data_offsets: {tensor_name!r} @ {shard.name}")
        begin, end = int(offsets[0]), int(offsets[1])
        length = end - begin
        if length < 0:
            raise ValueError(f"Invalid tensor length: {tensor_name!r} @ {shard.name}")
        data_start = 8 + header_len
        handle.seek(data_start + begin)
        remaining = length
        while remaining > 0:
            chunk = handle.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"Unexpected EOF while reading tensor: {tensor_name!r} @ {shard.name}")
            hasher.update(chunk)
            remaining -= len(chunk)
    return hasher.hexdigest()


# Byte-level MD5 comparison for tensors declared non-quantized by the
# quantization config. This reads safetensors payload bytes from disk, so large
# models should always use ``quant_config["max_samples"]`` to bound I/O.
def check_non_quantized_tensors_md5_unchanged(
    source_model_dir: str | Path,
    output_model_dir: str | Path,
    quant_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Check tensors expected to remain non-quantized for byte-identical MD5 values.

    Only tensors present on both sides and matched by ``exclude`` /
    ``exclude_layers_name`` glob rules are checked. The function compares
    dtype/shape in headers first, then calculates MD5 over raw payload bytes.
    When there are too many tensors, it samples by
    ``quant_config["max_samples"]`` (default ``200``). Sampling uses
    ``quant_config["random_seed"]`` or ``quant_config["seed"]`` when provided.

    :param str | pathlib.Path source_model_dir: Original pre-quantization model
        directory. No default.
    :param str | pathlib.Path output_model_dir: Quantized output model
        directory. No default.
    :param dict[str, Any] quant_config: Configuration describing which tensors
        should remain non-quantized. Common keys:

        - ``exclude`` or ``exclude_layers_name``: ``list[str]`` or a single
          ``str``. Uses the same semantics as ``_matches_any``, matching either
          a full parameter name or the module prefix after removing the final
          ``.*`` segment. **At least one is required**; otherwise this function
          returns a failure report early.
        - ``max_samples``: random spot-check limit when candidate tensor count
          exceeds this value. **Defaults to ``200``**.
        - ``random_seed`` or ``seed``: **optional** deterministic sampling seed.

    :return: Report containing ``ok``, candidate count, checked count, matched
        samples, and mismatches.
    :rtype: dict[str, Any]
    """
    _agent_log(
        "md5",
        "start: check_non_quantized_tensors_md5_unchanged (will read payload bytes for tensors matched by excludes)",
    )
    # --- 1. Resolve directories plus max_samples and exclude globs from quant_config. ---
    source_path = _path(source_model_dir)
    output_path = _path(output_model_dir)
    _agent_log("md5", f"source directory: {source_path}")
    _agent_log("md5", f"output directory: {output_path}")
    patterns = _normalize_exclude_patterns(quant_config)
    _agent_log("md5", f"exclude rules: {patterns if patterns else '(none)'}")
    max_samples_raw = quant_config.get("max_samples", 200)
    max_samples = int(max_samples_raw) if max_samples_raw is not None else 200
    _agent_log(
        "md5",
        f"max_samples={max_samples}, other quant_config keys: "
        f"{sorted(k for k in quant_config if k not in {'exclude', 'exclude_layers_name', 'max_samples', 'random_seed', 'seed'})}",
    )
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
        }

    seed = quant_config.get("random_seed", quant_config.get("seed"))
    _agent_log("md5", f"sampling random seed: {seed!r}")
    errors: list[str] = []
    if not patterns:
        errors.append(
            "quant_config did not provide exclude / exclude_layers_name, so no expected non-quantized tensors matched."
        )

    # Directories must exist before shard bytes can be read from both sides.
    if not source_path.is_dir():
        errors.append(f"Source model directory not found: {source_path}")
    if not output_path.is_dir():
        errors.append(f"Output model directory not found: {output_path}")

    if errors:
        for err in errors:
            _agent_log("md5", f"failure: {err}")
        _agent_log("md5", "end: check_non_quantized_tensors_md5_unchanged (preflight failed)")
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
        }

    # --- 2. Enumerate tensors from headers only, then filter source names matched by excludes. ---
    _agent_log("md5", "scan safetensors headers on both sides and collect tensor metadata...")
    source_tensors = collect_tensor_metadata(source_path)
    output_tensors = collect_tensor_metadata(output_path)
    _agent_log("md5", f"source tensor key count: {len(source_tensors)}, output: {len(output_tensors)}")
    candidates = sorted(name for name in source_tensors if _matches_any(name, patterns))
    _agent_log("md5", f"candidate tensors matched by excludes: {len(candidates)}")

    if not candidates:
        _agent_log("md5", "warning: no candidates matched exclude rules; check globs against checkpoint keys")
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
            "note": "No tensors matched exclude rules; check whether globs match checkpoint keys.",
        }

    # --- 3. Randomly sample when there are too many candidates to reduce large-model MD5 cost. ---
    sampled = len(candidates) > max_samples
    if sampled:
        rng = random.Random(seed)
        checked = sorted(rng.sample(candidates, max_samples))
        _agent_log("md5", f"candidates exceed max_samples; randomly sampled {len(checked)}/{len(candidates)}")
    else:
        checked = list(candidates)
        _agent_log("md5", f"will include all {len(checked)} candidates in MD5 validation")

    matched_samples: list[str] = []
    mismatches: list[dict[str, Any]] = []

    # --- 4. Per tensor: require output presence and matching dtype/shape, then compare payload MD5. ---
    for step, name in enumerate(checked, start=1):
        if name not in output_tensors:
            mismatches.append({"name": name, "reason": "missing_in_output"})
            _agent_log("md5", f"[{step}/{len(checked)}] FAIL {name}: key missing in output")
            continue
        s_meta = source_tensors[name]
        o_meta = output_tensors[name]
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
                f"dtype {s_meta['dtype']} vs {o_meta['dtype']} shape {s_meta['shape']} vs {o_meta['shape']}",
            )
            continue
        s_shard = source_path / s_meta["shard"]
        o_shard = output_path / o_meta["shard"]
        try:
            s_md5 = _md5_safetensors_tensor_payload(s_shard, name)
            o_md5 = _md5_safetensors_tensor_payload(o_shard, name)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            mismatches.append({"name": name, "reason": "read_error", "detail": str(exc)})
            _agent_log("md5", f"[{step}/{len(checked)}] FAIL {name}: payload read error {exc!r}")
            continue
        if s_md5 != o_md5:
            mismatches.append({"name": name, "reason": "md5_mismatch", "source_md5": s_md5, "output_md5": o_md5})
            _agent_log(
                "md5",
                f"[{step}/{len(checked)}] FAIL {name}: MD5 mismatch source={s_md5} output={o_md5}",
            )
        else:
            matched_samples.append(name)
            _agent_log(
                "md5",
                f"[{step}/{len(checked)}] OK {name}: MD5 matched  "
                f"source shard={s_meta['shard']} output shard={o_meta['shard']}",
            )

    ok = not mismatches
    _agent_log(
        "md5",
        f"result: ok={ok} checked={len(checked)} matched={len(matched_samples)} mismatches={len(mismatches)}",
    )
    _agent_log("md5", "end: check_non_quantized_tensors_md5_unchanged")
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
    }


def _drop_keys_recursive(obj: Any, ignore_key_names: frozenset[str]) -> Any:
    """Recursively remove keys listed in ``ignore_key_names`` from dict/list structures."""
    if isinstance(obj, dict):
        return {
            key: _drop_keys_recursive(value, ignore_key_names)
            for key, value in obj.items()
            if key not in ignore_key_names
        }
    if isinstance(obj, list):
        return [_drop_keys_recursive(item, ignore_key_names) for item in obj]
    return obj


def _deep_compare_json_values(left: Any, right: Any, path: str) -> list[str]:
    """Deeply compare two JSON subtrees and return human-readable path-based diffs."""
    mismatches: list[str] = []
    if type(left) is not type(right):
        mismatches.append(f"{path or '<root>'}: type mismatch {type(left).__name__} / {type(right).__name__}")
        return mismatches
    if isinstance(left, dict):
        keys_l = set(left)
        keys_r = set(right)
        base = path or "<root>"
        only_l = sorted(keys_l - keys_r)
        only_r = sorted(keys_r - keys_l)
        if only_l:
            mismatches.append(f"{base}: keys only in original config {only_l}")
        if only_r:
            mismatches.append(f"{base}: keys only in quantized config {only_r}")
        for key in sorted(keys_l & keys_r):
            sub = f"{base}.{key}" if path else key
            mismatches.extend(_deep_compare_json_values(left[key], right[key], sub))
        return mismatches
    if isinstance(left, list):
        base = path or "<root>"
        if len(left) != len(right):
            mismatches.append(f"{base}: list length mismatch {len(left)} / {len(right)}")
            return mismatches
        for index, (item_l, item_r) in enumerate(zip(left, right, strict=False)):
            mismatches.extend(_deep_compare_json_values(item_l, item_r, f"{base}[{index}]"))
        return mismatches
    if left != right:
        mismatches.append(f"{path or '<root>'}: value mismatch {left!r} / {right!r}")
    return mismatches


def check_config_json_equal_except_quantization(
    original_config_path: str | Path,
    quantized_config_path: str | Path,
    ignore_key_names: Iterable[str] | None = None,
    replace_default_ignores: bool = False,
) -> dict[str, Any]:
    """
    Read two ``config.json`` files and compare them after removing quantization keys.

    By default, subtrees keyed by ``quantization_config`` are removed at **any
    nested level**, matching common HuggingFace config conventions.
    ``ignore_key_names`` can append more names, or fully replace the set when
    ``replace_default_ignores`` is true. The remaining JSON is compared
    recursively.

    :param str | pathlib.Path original_config_path: Original model
        ``config.json`` path. No default.
    :param str | pathlib.Path quantized_config_path: Quantized model
        ``config.json`` path. No default.
    :param Iterable[str] | None ignore_key_names: Additional key names to remove.
        **Defaults to** ``None``, which uses only
        ``DEFAULT_CONFIG_QUANT_IGNORE_KEY_NAMES`` (currently
        ``quantization_config``). A non-empty iterable is unioned with defaults
        unless ``replace_default_ignores`` is ``True``.
    :param bool replace_default_ignores: If ``True``, use only
        ``ignore_key_names`` as the removal set. In that case
        ``ignore_key_names`` should not be ``None`` because no quantization
        config would be removed. **Defaults to ``False``**.

    :return: ``ok``, removed key names, diff list, and file read errors.
    :rtype: dict[str, Any]
    """
    _agent_log("config", "start: check_config_json_equal_except_quantization")
    orig_path = _path(original_config_path)
    quant_path = _path(quantized_config_path)
    _agent_log("config", f"original config: {orig_path}")
    _agent_log("config", f"quantized config: {quant_path}")
    errors: list[str] = []

    if replace_default_ignores and ignore_key_names is None:
        _agent_log("config", "failure: replace_default_ignores=True but ignore_key_names was not provided")
        _agent_log("config", "end: check_config_json_equal_except_quantization")
        return {
            "ok": False,
            "errors": ["ignore_key_names is required when replace_default_ignores=True."],
            "ignored_key_names": [],
            "mismatches": [],
            "mismatch_count": 0,
            "original_config_path": str(orig_path),
            "quantized_config_path": str(quant_path),
        }

    if replace_default_ignores:
        key_set = frozenset(str(x) for x in ignore_key_names or ())
    elif ignore_key_names is None:
        key_set = DEFAULT_CONFIG_QUANT_IGNORE_KEY_NAMES
    else:
        key_set = frozenset(DEFAULT_CONFIG_QUANT_IGNORE_KEY_NAMES) | frozenset(str(x) for x in ignore_key_names)
    _agent_log(
        "config",
        f"key names removed from JSON tree: {sorted(key_set)} replace_default_ignores={replace_default_ignores}",
    )

    if not orig_path.is_file():
        errors.append(f"Original config file not found: {orig_path}")
    if not quant_path.is_file():
        errors.append(f"Quantized config file not found: {quant_path}")

    if not orig_path.is_file() or not quant_path.is_file():
        for err in errors:
            _agent_log("config", f"failure: {err}")
        _agent_log("config", "end: check_config_json_equal_except_quantization")
        return {
            "ok": False,
            "errors": errors,
            "ignored_key_names": sorted(key_set),
            "mismatches": [],
            "mismatch_count": 0,
            "original_config_path": str(orig_path),
            "quantized_config_path": str(quant_path),
        }

    try:
        with orig_path.open("r", encoding="utf-8") as handle:
            raw_orig = json.load(handle)
        with quant_path.open("r", encoding="utf-8") as handle:
            raw_quant = json.load(handle)
        _agent_log(
            "config",
            f"parsed both JSON files: original top-level keys {list(raw_orig)[:20]}"
            f"{'...' if len(raw_orig) > 20 else ''}, quantized {list(raw_quant)[:20]}"
            f"{'...' if len(raw_quant) > 20 else ''}",
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _agent_log("config", f"read or parse failure: {exc!r}")
        _agent_log("config", "end: check_config_json_equal_except_quantization")
        return {
            "ok": False,
            "errors": [f"Failed to read or parse JSON: {exc}"],
            "ignored_key_names": sorted(key_set),
            "mismatches": [],
            "mismatch_count": 0,
            "original_config_path": str(orig_path),
            "quantized_config_path": str(quant_path),
        }

    stripped_orig = _drop_keys_recursive(raw_orig, key_set)
    stripped_quant = _drop_keys_recursive(raw_quant, key_set)
    mismatches = _deep_compare_json_values(stripped_orig, stripped_quant, "")
    ok = not mismatches
    _agent_log("config", f"deep comparison after key removal: mismatch_count={len(mismatches)} ok={ok}")
    for line in mismatches[:25]:
        _agent_log("config", f"  diff: {line}")
    if len(mismatches) > 25:
        _agent_log("config", f"  ... {len(mismatches) - 25} more diffs are in the mismatches return value")
    _agent_log("config", "end: check_config_json_equal_except_quantization")
    return {
        "ok": ok,
        "errors": [],
        "ignored_key_names": sorted(key_set),
        "mismatches": mismatches[:500],
        "mismatch_count": len(mismatches),
        "original_config_path": str(orig_path),
        "quantized_config_path": str(quant_path),
    }


def _mock_tensor_dtype_nbytes(dtype: str) -> int:
    return {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "F16": 2,
        "BF16": 2,
        "F32": 4,
        "F64": 8,
        "I32": 4,
        "I64": 8,
    }.get(dtype, 1)


def _write_mock_safetensors_shard(path: Path, specs: list[tuple[str, str, list[int], bytes]]) -> None:
    """
    Write a minimal valid safetensors shard for self-tests.

    Each spec is ``(tensor_name, dtype, shape, payload)``; payload length must equal
    ``prod(shape) * _mock_tensor_dtype_nbytes(dtype)``.
    """
    offset = 0
    header: dict[str, Any] = {}
    blobs: list[bytes] = []
    for name, dtype, shape, payload in specs:
        n = _numel(shape)
        el = _mock_tensor_dtype_nbytes(dtype)
        need = n * el
        if len(payload) != need:
            raise ValueError(f"mock shard {path.name}: tensor {name!r} payload {len(payload)} != {need}")
        start, end = offset, offset + len(payload)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, end]}
        blobs.append(payload)
        offset = end
    data_blob = b"".join(blobs)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(data_blob)


def _selftest_exported_symbols() -> int:
    """
    Run mock-data checks for symbols listed in ``__all__``.

    :return: ``0`` if all checks pass, ``1`` otherwise.
    :rtype: int
    """
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # --- get_fuzzy_tensor_names ---
        fuzzy_dir = base / "fuzzy_model"
        fuzzy_dir.mkdir()
        _write_mock_safetensors_shard(
            fuzzy_dir / "model.safetensors",
            [
                ("mock.layers.0.cell.weight", "U8", [2], b"\x01\x02"),
                ("mock.layers.1.cell.weight", "U8", [2], b"\x03\x04"),
                ("other.bias", "F32", [1], b"\x00\x00\x00\x00"),
            ],
        )
        fuzzy = get_fuzzy_tensor_names(fuzzy_dir)
        patterns = {item["pattern"]: item["dtype_counts"] for item in fuzzy}
        if "mock.layers.*.cell.weight" not in patterns:
            failures.append(f"get_fuzzy_tensor_names: expected pattern mock.layers.*.cell.weight, got {list(patterns)}")
        elif patterns.get("mock.layers.*.cell.weight") != {"U8": 2}:
            failures.append(
                f"get_fuzzy_tensor_names: dtype_counts for layers pattern want {{'U8': 2}}, got {patterns.get('mock.layers.*.cell.weight')}"
            )
        if "other.bias" not in patterns or patterns["other.bias"] != {"F32": 1}:
            failures.append(f"get_fuzzy_tensor_names: other.bias unexpected: {patterns.get('other.bias')}")

        # --- check_auxiliary_files_copied ---
        src_aux = base / "src_m"
        out_aux = base / "out_m"
        src_aux.mkdir()
        out_aux.mkdir()
        (src_aux / "README.md").write_text("mock readme\n", encoding="utf-8")
        (out_aux / "README.md").write_text("mock readme\n", encoding="utf-8")
        _write_mock_safetensors_shard(src_aux / "model.safetensors", [("w", "U8", [1], b"\xab")])
        _write_mock_safetensors_shard(out_aux / "model.safetensors", [("w", "U8", [1], b"\xcd")])
        aux_rep = check_auxiliary_files_copied(src_aux, out_aux, max_examples=20)
        if not aux_rep.get("ok"):
            failures.append(f"check_auxiliary_files_copied: expected ok, got {aux_rep}")

        # --- check_config_json_equal_except_quantization ---
        orig_cfg = base / "orig_config.json"
        quant_cfg = base / "quant_config.json"
        orig_cfg.write_text(
            json.dumps({"model_type": "m", "hidden_size": 8, "quantization_config": {"bits": 4}}, ensure_ascii=False),
            encoding="utf-8",
        )
        quant_cfg.write_text(
            json.dumps({"model_type": "m", "hidden_size": 8, "quantization_config": {"bits": 8}}, ensure_ascii=False),
            encoding="utf-8",
        )
        cfg_rep = check_config_json_equal_except_quantization(orig_cfg, quant_cfg)
        if not cfg_rep.get("ok"):
            failures.append(f"check_config_json_equal_except_quantization: expected ok, got {cfg_rep}")

        # --- check_non_quantized_tensors_md5_unchanged ---
        src_nm = base / "src_nm"
        out_ok = base / "out_ok"
        out_bad = base / "out_bad"
        src_nm.mkdir()
        out_ok.mkdir()
        out_bad.mkdir()
        keep_payload = b"quant_keep_payload_bytes_12345678"
        _write_mock_safetensors_shard(
            src_nm / "model.safetensors",
            [("model.layers.0.keep.weight", "U8", [len(keep_payload)], keep_payload)],
        )
        _write_mock_safetensors_shard(
            out_ok / "model.safetensors",
            [("model.layers.0.keep.weight", "U8", [len(keep_payload)], keep_payload)],
        )
        _write_mock_safetensors_shard(
            out_bad / "model.safetensors",
            [("model.layers.0.keep.weight", "U8", [len(keep_payload)], keep_payload[::-1])],
        )
        qconf: dict[str, Any] = {"exclude": ["*.keep.weight"], "max_samples": 20, "random_seed": 0}
        nm_ok = check_non_quantized_tensors_md5_unchanged(src_nm, out_ok, qconf)
        if not nm_ok.get("ok") or "model.layers.0.keep.weight" not in nm_ok.get("matched_samples", []):
            failures.append(f"check_non_quantized_tensors_md5_unchanged success case: {nm_ok}")
        nm_bad = check_non_quantized_tensors_md5_unchanged(src_nm, out_bad, qconf)
        if nm_bad.get("ok"):
            failures.append("check_non_quantized_tensors_md5_unchanged: expected failure on MD5 mismatch")
        reasons = {m.get("reason") for m in nm_bad.get("mismatches", [])}
        if "md5_mismatch" not in reasons:
            failures.append(f"check_non_quantized_tensors_md5_unchanged: expected md5_mismatch in {nm_bad}")

        # --- check_non_quantized_tensors_md5_unchanged: sampling branch ---
        src_samp = base / "src_samp"
        out_samp = base / "out_samp"
        src_samp.mkdir()
        out_samp.mkdir()
        samp_payload = b"\xaa\xbb"
        samp_specs = [(f"model.layers.{i}.keep.weight", "U8", [len(samp_payload)], samp_payload) for i in range(10)]
        _write_mock_safetensors_shard(src_samp / "model.safetensors", samp_specs)
        _write_mock_safetensors_shard(out_samp / "model.safetensors", samp_specs)
        # max_samples=3 forces sampling (10 candidates > 3)
        qconf_samp: dict[str, Any] = {"exclude": ["*.keep.weight"], "max_samples": 3, "random_seed": 42}
        nm_samp = check_non_quantized_tensors_md5_unchanged(src_samp, out_samp, qconf_samp)
        if not nm_samp.get("ok"):
            failures.append(f"check_non_quantized_tensors_md5_unchanged sampling branch: {nm_samp}")
        if not nm_samp.get("sampled"):
            failures.append("check_non_quantized_tensors_md5_unchanged: expected sampled=True")
        if nm_samp.get("checked_count") != 3:
            failures.append(
                f"check_non_quantized_tensors_md5_unchanged: expected checked_count=3, got {nm_samp.get('checked_count')}"
            )

        # --- check_config_json_equal_except_quantization: replace_default_ignores=True ---
        orig_cfg2 = base / "orig_config2.json"
        quant_cfg2 = base / "quant_config2.json"
        orig_cfg2.write_text(
            json.dumps(
                {"model_type": "m", "quantization_config": {"bits": 4}, "custom_quant": {"x": 1}}, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        quant_cfg2.write_text(
            json.dumps(
                {"model_type": "m", "quantization_config": {"bits": 8}, "custom_quant": {"x": 2}}, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        # replace_default_ignores=True with custom_quant: quantization_config differences should surface
        cfg_rep2 = check_config_json_equal_except_quantization(
            orig_cfg2, quant_cfg2, ignore_key_names=["custom_quant"], replace_default_ignores=True
        )
        # quantization_config is NOT stripped (replace_default_ignores=True), so there will be a mismatch
        if cfg_rep2.get("ok"):
            failures.append(
                "check_config_json_equal_except_quantization replace_default_ignores: expected mismatch, got ok"
            )

        # --- check_non_quantized_tensors_md5_unchanged: read_error path (truncated data region) ---
        # The output shard has a valid header (so collect_tensor_metadata succeeds) but the
        # data bytes are truncated, triggering a read_error when _md5_safetensors_tensor_payload
        # attempts to seek and read past the end of file.
        src_trunc = base / "src_trunc"
        out_trunc = base / "out_trunc"
        src_trunc.mkdir()
        out_trunc.mkdir()
        trunc_payload = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        _write_mock_safetensors_shard(
            src_trunc / "model.safetensors",
            [("model.weight", "U8", [len(trunc_payload)], trunc_payload)],
        )
        # Build a shard whose header claims 8-byte payload but only contains 2 bytes of data.
        import struct as _struct

        trunc_header_dict = {
            "model.weight": {"dtype": "U8", "shape": [len(trunc_payload)], "data_offsets": [0, len(trunc_payload)]}
        }
        trunc_header_bytes = json.dumps(trunc_header_dict, separators=(",", ":")).encode("utf-8")
        with (out_trunc / "model.safetensors").open("wb") as fh:
            fh.write(_struct.pack("<Q", len(trunc_header_bytes)))
            fh.write(trunc_header_bytes)
            fh.write(b"\xde\xad")  # only 2 bytes instead of 8 — read will hit EOF mid-tensor
        qconf_trunc: dict[str, Any] = {"exclude": ["model.weight"], "max_samples": 10}
        nm_trunc = check_non_quantized_tensors_md5_unchanged(src_trunc, out_trunc, qconf_trunc)
        # Should fail because payload read hits unexpected EOF
        if nm_trunc.get("ok"):
            failures.append("check_non_quantized_tensors_md5_unchanged read_error path: expected failure")
        trunc_reasons = {m.get("reason") for m in nm_trunc.get("mismatches", [])}
        if "read_error" not in trunc_reasons:
            failures.append(f"check_non_quantized_tensors_md5_unchanged: expected read_error reason, got {nm_trunc}")

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
    # Dev-only: embedded mock checks for ``__all__`` exports; not an end-user CLI.
    raise SystemExit(_selftest_exported_symbols())
