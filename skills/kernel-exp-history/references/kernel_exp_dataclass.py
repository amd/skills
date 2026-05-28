# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import tempfile
import time
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional
import uuid

# Global JSON "database" location
DB_PATH = Path("kernel_experiments_db.json")

Status = Literal["new", "running", "done", "failed", "timeout"]


def _uuid64() -> str:
    """Generate a 64-bit hex uuid string."""
    return f"{uuid.uuid4().int & ((1 << 64) - 1):016x}"


@dataclass
class KernelExperiment:
    score: float  # avg speedup (1.0 = no speedup, 2.0 = 2x faster)
    raw_result: str  # per-shape speedups or notes
    dtype_sig: str  # fp16, bf16, fp32, bf8, etc.
    env: str  # GPU model, ROCm version, etc.
    is_buggy: bool
    error_message: str  # error type + message when is_buggy is True
    change_summary: str
    detailed_description: str
    code_change: str  # diff patch string
    base_commit: str  # upstream commit id (not local)
    operator_sig: str  # which files/kernels are affected
    profiling_info: str
    status: Status
    id: str = field(default_factory=_uuid64)
    pid: str = field(
        default="",
        metadata={"comment": "Parent experiment id; set manually, do not auto-generate."},
    )  # Parent experiment id; set manually when linking lineage.
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict) -> "KernelExperiment":
        return KernelExperiment(**data)


def _ensure_db_exists() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        DB_PATH.write_text("{}", encoding="utf-8")


@contextmanager
def _locked_db(exclusive: bool):
    mode = "a+"  # ensure file exists and is open for locking
    with DB_PATH.open(mode, encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _atomic_write_json(path: Path, content: Dict[str, Dict]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(content, tmp, indent=2, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _load_db() -> Dict[str, Dict]:
    _ensure_db_exists()
    with _locked_db(exclusive=False):
        with DB_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)


def _save_db(db: Dict[str, Dict]) -> None:
    with _locked_db(exclusive=True):
        _atomic_write_json(DB_PATH, db)


def create_experiment(exp: KernelExperiment) -> None:
    db = _load_db()
    if exp.id in db:
        raise ValueError(f"Experiment with id '{exp.id}' already exists")
    db[exp.id] = exp.to_dict()
    _save_db(db)


def get_experiment(exp_id: str) -> Optional[KernelExperiment]:
    db = _load_db()
    if exp_id not in db:
        return None
    return KernelExperiment.from_dict(db[exp_id])


def list_experiments() -> List[KernelExperiment]:
    db = _load_db()
    return [KernelExperiment.from_dict(v) for v in db.values()]


def top_experiments(max_results: int = 20) -> List[Dict[str, object]]:
    """
    Return experiments sorted by score desc, containing only key fields.
    """
    experiments = list_experiments()
    filtered = [exp for exp in experiments]
    filtered.sort(key=lambda e: e.score, reverse=True)
    top_n = filtered[: max(0, max_results)]
    keys = [
        "base_commit",
        "change_summary",
        "detailed_description",
        "dtype_sig",
        "env",
        "id",
        "operator_sig",
        "profiling_info",
        "raw_result",
        "score",
    ]
    return [{k: getattr(exp, k) for k in keys} for exp in top_n]


def update_experiment(exp_id: str, **changes) -> KernelExperiment:
    db = _load_db()
    if exp_id not in db:
        raise KeyError(f"Experiment with Id '{exp_id}' not found")
    current = db[exp_id]
    current.update(changes)
    db[exp_id] = current
    _save_db(db)
    return KernelExperiment.from_dict(current)


def delete_experiment(exp_id: str) -> None:
    db = _load_db()
    if exp_id not in db:
        raise KeyError(f"Experiment with Id '{exp_id}' not found")
    del db[exp_id]
    _save_db(db)


def test_insert_example() -> KernelExperiment:
    """Insert a sample experiment entry for quick sanity checks."""
    sample = KernelExperiment(
        pid="(Parent experiment id)",
        score=1.25,
        raw_result="shape=128x128 speedup=1.3; shape=256x256 speedup=1.2",
        dtype_sig="fp16",
        env="MI300X, ROCm 7.0.0",
        is_buggy=False,
        error_message="",
        change_summary="tuned block size and vectorized loads",
        detailed_description="Adjusted kernel launch for better wave occupancy on MI300X.",
        code_change="(diff patch here)",
        base_commit="abcdef1234567890",
        operator_sig="attention_ragged.cu: paged_attention_ll4mi",
        profiling_info="SQ busy 75%, TCP 65%, TCC 55%",
        status="new",
    )
    create_experiment(sample)
    return sample


if __name__ == "__main__":
    exp = test_insert_example()
    print(f"Inserted sample experiment with id: {exp.id}")
