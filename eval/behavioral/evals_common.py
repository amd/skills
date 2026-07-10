"""Shared helpers for the JSON-driven behavioral evals.

Both the local executor (`run_evals.py`) and the CI matrix builder
(`.github/scripts/select_behavioral.py`) import this module so the two agree
byte-for-byte on:

  * where a skill's eval file lives (`skills/<skill>/evals/evals.json`),
  * how the optional per-test ``runners`` field resolves to concrete GitHub
    Actions runner-label sets, and
  * the canonical string id used to name each runner target.

Schema (``skills/<skill>/evals/evals.json``)
--------------------------------------------
A single JSON object::

    {
      "skill": "local-ai-use",              # optional; defaults to the folder name
      "default_runners": [ <target>, ... ], # optional; per-file fallback for tests
      "tests": [ <test>, ... ]
    }

Each ``<test>``::

    {
      "id": "generate-image-of-a-cat",      # unique within the file
      "prompt": "…the user turn to run…",
      "setup": {                            # optional; staged before the run
        "files": { "main.py": "…contents…" }
      },
      "runners": [ <target>, ... ],         # optional; see below
      "expect": {
        "logs_contains":      [ "…substring…" ],   # deterministic
        "workspace_contains": [ "out.png" ],       # deterministic
        "should":             [ "…statement…" ],   # LLM-judged, must be TRUE
        "should_not":         [ "…statement…" ]    # LLM-judged, must be FALSE
      }
    }

Runner targets
--------------
``runners`` (and the file-level ``default_runners``) is a list of *targets*.
A target selects one runner and produces one CI job; every target listed on a
test runs that test. A target may be written as:

  * a string  — ``"Windows"`` or ``"halo"`` (a single extra label), or
  * a list    — ``["halo", "Windows"]`` (all labels must be present on the
                runner; ANDed together).

``self-hosted`` is always implied and prepended, and the alias ``halo`` expands
to the real ``strix_halo`` runner label. When a test omits ``runners`` the
file's ``default_runners`` is used; when that is also omitted the repo default
(Strix-Halo Linux + Windows) applies, matching the pre-JSON matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SKILLS_DIR = REPO_ROOT / "skills"

# Labels every self-hosted target carries. Kept separate from the per-target
# labels so authors write only the distinguishing bits ("Windows", "halo").
BASE_LABELS: tuple[str, ...] = ("self-hosted",)

# Author-friendly aliases -> real GitHub runner labels.
LABEL_ALIASES: dict[str, str] = {
    "halo": "strix_halo",
}

# Used when neither the test nor the file declares runners. Reproduces the
# original hardcoded matrix: Strix Halo on both Linux and Windows.
GLOBAL_DEFAULT_RUNNERS: list[list[str]] = [
    ["strix_halo", "Linux"],
    ["strix_halo", "Windows"],
]


def evals_path(skill: str) -> Path:
    """Path to a skill's eval file (may or may not exist)."""
    return SKILLS_DIR / skill / "evals" / "evals.json"


def has_evals(skill: str) -> bool:
    """A skill is testable when it has both a SKILL.md and an evals.json."""
    return (SKILLS_DIR / skill / "SKILL.md").is_file() and evals_path(skill).is_file()


def load_evals(skill: str) -> dict:
    """Load and lightly validate a skill's eval file."""
    path = evals_path(skill)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    if not isinstance(data.get("tests"), list):
        raise ValueError(f"{path}: missing 'tests' array")
    return data


def all_testable_skills() -> list[str]:
    """Every skill under skills/ that ships an evals.json."""
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(p.name for p in SKILLS_DIR.iterdir() if p.is_dir() and has_evals(p.name))


def _normalize_group(target) -> list[str]:
    """Turn one target (string or list) into an ordered, de-duped label list.

    Applies aliases and prepends the always-on base labels. Order is preserved
    so the labels read naturally; duplicates (e.g. an alias colliding with a
    base label) are dropped.
    """
    if isinstance(target, str):
        raw = [target]
    elif isinstance(target, list):
        raw = [str(x) for x in target]
    else:
        raise ValueError(f"runner target must be a string or list, got {target!r}")

    labels: list[str] = []
    for label in [*BASE_LABELS, *raw]:
        resolved = LABEL_ALIASES.get(label, label)
        if resolved not in labels:
            labels.append(resolved)
    if len(labels) <= len(BASE_LABELS):
        raise ValueError(f"runner target {target!r} adds no distinguishing label")
    return labels


def target_id(labels: list[str]) -> str:
    """Stable id for a resolved label set (order-independent).

    Excludes the base labels so the id reads as the distinguishing part, e.g.
    ``["self-hosted","strix_halo","Windows"]`` -> ``"strix_halo+Windows"``.
    """
    distinctive = sorted(l for l in labels if l not in BASE_LABELS)
    return "+".join(distinctive)


def resolve_targets(test: dict, file_default: list | None) -> dict[str, list[str]]:
    """Resolve a test's runner targets to ``{target_id: labels}``.

    Falls back file-default -> global-default when the test omits ``runners``.
    """
    raw = test.get("runners")
    if raw is None:
        raw = file_default if file_default is not None else GLOBAL_DEFAULT_RUNNERS
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"test {test.get('id')!r}: 'runners' must be a non-empty list")

    out: dict[str, list[str]] = {}
    for target in raw:
        labels = _normalize_group(target)
        out[target_id(labels)] = labels
    return out


def skill_targets(data: dict) -> dict[str, list[str]]:
    """Union of every runner target across a skill's tests -> ``{id: labels}``."""
    file_default = data.get("default_runners")
    targets: dict[str, list[str]] = {}
    for test in data["tests"]:
        targets.update(resolve_targets(test, file_default))
    return targets


def tests_for_target(data: dict, tid: str) -> list[dict]:
    """The tests in ``data`` that should run on the target with id ``tid``."""
    file_default = data.get("default_runners")
    return [t for t in data["tests"] if tid in resolve_targets(t, file_default)]
