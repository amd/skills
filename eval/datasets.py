# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Per-skill eval datasets: discovery, parsing, and structural validation.

Every skill owns one dataset at ``skills/<name>/evals/evals.json``. A case in
that file is a single user prompt plus what should happen when the agent
receives it, and the same case feeds both run modes:

  * **routing** -- the whole catalog is installed and only ``expect_skill`` is
    graded ("did the right skill fire, and only then?").
  * **behavior** -- just this skill is installed, the run goes to completion,
    and ``should`` / ``should_not`` / ``logs_contain`` / ``files_exist`` are
    graded ("once it fired, did it do the job?").

Writing the prompt once for both is the point: a routing prompt that nothing
grades is a prompt nobody maintains, and a behavioral test that re-asserts
routing with a substring match is a worse version of a check this module
already models as a field.

The folder is the identity. ``skills/serving-llms-on-epyc/evals/evals.json``
is a dataset about ``serving-llms-on-epyc``, so nothing inside restates that,
and ``expect_skill`` defaults to the owning skill. The minimum valid case is::

    {"id": "epyc-vllm-zentorch", "prompt": "Serve Llama 3.1 8B with zentorch."}

Set ``expect_skill`` explicitly only to say something different: ``null`` for
"no skill should fire", or another skill's name to assert a handoff.

Prompt categories are derived rather than declared, because the file a case
lives in already carries the distinction:

  * ``expect_skill`` names a skill        -> ``positive``
  * ``expect_skill: null`` in a skill's dataset -> ``near_miss`` (its owner
    wrote it precisely because it sits close to that skill)
  * a case in the shared pool             -> ``unrelated`` (belongs to no skill)

Stdlib only, so the runner needs no ``pip install``. ``load_machine`` is the
one exception and imports PyYAML lazily; nothing on the run path calls it.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
SKILLS_DIR = REPO_ROOT / "skills"

# One dataset per skill, beside the skill it describes.
DATASET_RELPATH = Path("evals") / "evals.json"
HOOKS_RELPATH = Path("evals") / "hooks.py"
MACHINE_RELPATH = Path("evals") / "machine.yml"

# Prompts that belong to no skill's domain. They are the "unrelated" control
# group for every skill at once, so they live centrally instead of being
# copy-pasted into each dataset.
SHARED_NEGATIVES = EVAL_DIR / "negatives.json"

# Tier 0, the bar every skill clears before it can ship. Cheap to meet (five
# prompts, no hardware, no assertions) and enforced structurally so a thin
# dataset fails validation without spending a single token.
MIN_POSITIVE_CASES = 3
MIN_NEGATIVE_CASES = 2

# `additionalProperties: false`, by hand. A mistyped key would otherwise be
# silently dropped -- and since `expect_skill` defaults to the owning skill,
# a typo'd `expect_skil: null` would quietly become a positive case.
CASE_KEYS = {
    "id",
    "prompt",
    "expect_skill",
    "should",
    "should_not",
    "logs_contain",
    "files_exist",
    "workspace",
    "note",
}
# `$schema` is allowed so a dataset can point editors at
# eval/schema/evals.schema.json for autocomplete; the runner ignores it.
DATASET_KEYS = {"cases", "_comment", "$schema"}

# JSON has no comments, so `note` is the sanctioned place for one. The runner
# ignores it; without it owners annotate fields that are not free text.
_STRING_LISTS = ("should", "should_not", "logs_contain", "files_exist")

# Distinguishes "the owner omitted expect_skill" (default: this skill) from
# "the owner wrote expect_skill: null" (no skill should fire).
_ABSENT = object()


@dataclass
class Case:
    """One prompt and everything that should be true after the agent sees it."""

    id: str
    prompt: str
    # The skill whose dataset this came from; None for the shared pool.
    skill: str | None
    # The skill that must activate, or None when nothing should.
    expect_skill: str | None
    should: list[str] = field(default_factory=list)
    should_not: list[str] = field(default_factory=list)
    logs_contain: list[str] = field(default_factory=list)
    files_exist: list[str] = field(default_factory=list)
    # Directory (relative to the skill root) whose contents seed the workspace.
    workspace: str | None = None
    note: str = ""

    @property
    def category(self) -> str:
        """Reporting bucket, derived from the expectation and the source file.

        Kept out of the file format on purpose: an owner who has to classify a
        prompt will eventually classify one wrong, and every input needed to
        do it correctly is already here.
        """
        if self.expect_skill is not None:
            return "positive"
        return "near_miss" if self.skill else "unrelated"

    @property
    def has_behavior(self) -> bool:
        """Whether this case grades anything beyond the routing decision."""
        return bool(self.should or self.should_not or self.logs_contain or self.files_exist)


def dataset_path(skill: str) -> Path:
    return SKILLS_DIR / skill / DATASET_RELPATH


def hooks_path(skill: str) -> Path:
    return SKILLS_DIR / skill / HOOKS_RELPATH


def machine_path(skill: str) -> Path:
    return SKILLS_DIR / skill / MACHINE_RELPATH


def catalog_skills() -> list[str]:
    """Every skill under ``skills/``, published or not.

    Deliberately not read from ``.claude-plugin/marketplace.json``: routing is
    measured against the catalog the agent can see, and an unpublished skill
    still competes for prompts the moment someone installs the repo. Gating
    coverage on publication also meant a skill got its first routing test only
    after it shipped, which is exactly the wrong order.
    """
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(
        path.name
        for path in SKILLS_DIR.iterdir()
        if path.is_dir() and (path / "SKILL.md").is_file()
    )


def skills_with_datasets() -> list[str]:
    """Skills that ship an eval dataset (and so can be run or gated on)."""
    return [skill for skill in catalog_skills() if dataset_path(skill).is_file()]


def _parse_cases(payload: object, skill: str | None, source: Path, errors: list[str]) -> list[Case]:
    """Turn one parsed dataset file into cases, appending any problems found."""
    where = source.name

    if not isinstance(payload, dict):
        errors.append(f"{where}: top level must be an object with a `cases` array.")
        return []

    unknown = sorted(set(payload) - DATASET_KEYS)
    if unknown:
        errors.append(f"{where}: unknown top-level key(s): {', '.join(unknown)}.")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        errors.append(f"{where}: `cases` must be a non-empty array.")
        return []

    cases: list[Case] = []
    for index, entry in enumerate(raw_cases):
        label = f"{where}: cases[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label} must be an object.")
            continue

        unknown = sorted(set(entry) - CASE_KEYS)
        if unknown:
            errors.append(
                f"{label} has unknown key(s): {', '.join(unknown)}. "
                f"Allowed: {', '.join(sorted(CASE_KEYS))}."
            )

        case_id = entry.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            errors.append(f"{label} is missing a non-empty string `id`.")
            continue
        case_id = case_id.strip()

        prompt = entry.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(f"{label} (`{case_id}`) is missing a non-empty string `prompt`.")
            continue

        raw_expect = entry.get("expect_skill", _ABSENT)
        if raw_expect is _ABSENT:
            # The common case: a positive for the skill that owns the file.
            expect = skill
        elif raw_expect is None:
            expect = None
        elif isinstance(raw_expect, str) and raw_expect.strip():
            expect = raw_expect.strip()
        else:
            errors.append(
                f"{label} (`{case_id}`): `expect_skill` must be a skill name or null."
            )
            continue

        if skill is None and expect is not None:
            errors.append(
                f"{label} (`{case_id}`): shared negatives belong to no skill, so "
                "`expect_skill` must be null."
            )
            continue

        lists: dict[str, list[str]] = {}
        malformed = False
        for key in _STRING_LISTS:
            value = entry.get(key, [])
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                errors.append(
                    f"{label} (`{case_id}`): `{key}` must be an array of non-empty strings."
                )
                malformed = True
                continue
            lists[key] = [item.strip() for item in value]
        if malformed:
            continue

        workspace = entry.get("workspace")
        if workspace is not None and (not isinstance(workspace, str) or not workspace.strip()):
            errors.append(f"{label} (`{case_id}`): `workspace` must be a directory path.")
            continue

        note = entry.get("note", "")
        if not isinstance(note, str):
            errors.append(f"{label} (`{case_id}`): `note` must be a string.")
            continue

        if expect is None and lists["should"]:
            # A negative case ends with no skill loaded, so "the agent should
            # have done X" is grading the base model, not the skill.
            errors.append(
                f"{label} (`{case_id}`): a case expecting no skill cannot use "
                "`should`; use `should_not` to pin down what must not happen."
            )
            continue

        if skill is None and any(lists[key] for key in _STRING_LISTS):
            # Behavior mode stages exactly one skill, and these cases belong to
            # none, so the assertions would be silently skipped rather than run.
            errors.append(
                f"{label} (`{case_id}`): shared negatives are routing-only and "
                "cannot carry behavioral assertions. Move the case into the "
                "dataset of the skill it should not trigger."
            )
            continue

        cases.append(
            Case(
                id=case_id,
                prompt=prompt.strip(),
                skill=skill,
                expect_skill=expect,
                should=lists["should"],
                should_not=lists["should_not"],
                logs_contain=lists["logs_contain"],
                files_exist=lists["files_exist"],
                workspace=workspace.strip() if isinstance(workspace, str) else None,
                note=note,
            )
        )
    return cases


def load_dataset(skill: str, errors: list[str] | None = None) -> list[Case]:
    """Cases from one skill's dataset. Raises SystemExit on error unless collecting."""
    collected: list[str] = [] if errors is None else errors
    path = dataset_path(skill)
    if not path.is_file():
        collected.append(f"{skill}: missing {DATASET_RELPATH.as_posix()}.")
        cases: list[Case] = []
    else:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            collected.append(f"{skill}/{path.name}: invalid JSON: {exc}")
            payload = None
        cases = _parse_cases(payload, skill, path, collected) if payload is not None else []

    if errors is None and collected:
        raise SystemExit("error: " + "\n       ".join(collected))
    return cases


def load_shared_negatives(errors: list[str] | None = None) -> list[Case]:
    """The catalog-wide `unrelated` control group."""
    collected: list[str] = [] if errors is None else errors
    if not SHARED_NEGATIVES.is_file():
        return []
    try:
        payload = json.loads(SHARED_NEGATIVES.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        collected.append(f"{SHARED_NEGATIVES.name}: invalid JSON: {exc}")
        payload = None
    cases = _parse_cases(payload, None, SHARED_NEGATIVES, collected) if payload is not None else []

    if errors is None and collected:
        raise SystemExit("error: " + "\n       ".join(collected))
    return cases


def load_all_cases(errors: list[str] | None = None) -> list[Case]:
    """Every case in the repo: each skill's dataset plus the shared pool.

    Routing grades against this whole set at once, which is where the coverage
    comes from: a positive case for skill Y is an implicit negative for skill
    X, so N owners each writing a handful of prompts about their own domain
    produce N-squared routing coverage without coordinating.
    """
    cases: list[Case] = []
    for skill in skills_with_datasets():
        cases.extend(load_dataset(skill, errors))
    cases.extend(load_shared_negatives(errors))
    return cases


def duplicate_ids(cases: list[Case]) -> list[str]:
    """Case ids used more than once. Ids are repo-wide because routing pools them."""
    return sorted(cid for cid, count in Counter(c.id for c in cases).items() if count > 1)


def filter_cases(cases: list[Case], only: str) -> list[Case]:
    """Narrow `cases` to a comma-separated list of case ids or skill names."""
    if not only.strip():
        return cases
    wanted = {token.strip() for token in only.split(",") if token.strip()}
    selected = [
        case
        for case in cases
        if case.id in wanted or case.skill in wanted or (case.expect_skill or "") in wanted
    ]
    if not selected:
        raise SystemExit(f"error: --only '{only}' matched no cases")
    return selected


def validate_all() -> list[str]:
    """Every structural problem across every dataset, as human-readable strings.

    Run by CI before any tokens are spent, so a malformed dataset fails in
    seconds rather than halfway through a paid run.
    """
    errors: list[str] = []
    cases = load_all_cases(errors)
    catalog = set(catalog_skills())

    for case_id in duplicate_ids(cases):
        errors.append(
            f"duplicate case id `{case_id}`. Ids are repo-wide because routing "
            "pools every skill's cases into one run."
        )

    for case in cases:
        if case.expect_skill is not None and case.expect_skill not in catalog:
            errors.append(
                f"case `{case.id}`: `expect_skill` names `{case.expect_skill}`, "
                "which has no directory under skills/."
            )
        if case.workspace:
            if case.skill is None:
                errors.append(f"case `{case.id}`: shared negatives cannot stage a workspace.")
            elif not (SKILLS_DIR / case.skill / case.workspace).is_dir():
                errors.append(
                    f"case `{case.id}`: `workspace` points at "
                    f"`{case.skill}/{case.workspace}`, which is not a directory."
                )

    for skill in catalog:
        errors.extend(tier0_errors(skill, [c for c in cases if c.skill == skill]))
    return errors


def tier0_errors(skill: str, cases: list[Case]) -> list[str]:
    """Whether `skill` meets the mandatory coverage bar."""
    if not dataset_path(skill).is_file():
        return [
            f"{skill}: no eval dataset. Every skill needs "
            f"`skills/{skill}/{DATASET_RELPATH.as_posix()}` with at least "
            f"{MIN_POSITIVE_CASES} prompts that should trigger it and "
            f"{MIN_NEGATIVE_CASES} that should not. Copy eval/TEMPLATE.json to start."
        ]

    errors: list[str] = []
    # Counted against *this* skill, so a case handed off to a neighbour
    # (`expect_skill` naming another skill) is a negative here, not a positive.
    positive = sum(1 for c in cases if c.expect_skill == skill)
    negative = sum(1 for c in cases if c.expect_skill != skill)
    if positive < MIN_POSITIVE_CASES:
        errors.append(
            f"{skill}: {positive} positive case(s); Tier 0 needs at least "
            f"{MIN_POSITIVE_CASES}. Add prompts a real user would type."
        )
    if negative < MIN_NEGATIVE_CASES:
        errors.append(
            f"{skill}: {negative} near-miss case(s); Tier 0 needs at least "
            f"{MIN_NEGATIVE_CASES}. Add prompts close to this skill's domain "
            "that should NOT trigger it (`\"expect_skill\": null`)."
        )
    return errors


def load_machine(skill: str) -> dict:
    """Runner requirements for `skill`, or ``{}`` when it runs anywhere.

    Absent is the common case and means the default runners. Only skills that
    need specific hardware ship ``evals/machine.yml``, and they ship it beside
    the skill so the declaration is owned by the person who knows about the
    hardware rather than by a list inside a CI workflow.

    PyYAML is imported here rather than at module scope: nothing on the run
    path needs this, so the runner stays dependency-free.
    """
    path = machine_path(skill)
    if not path.is_file():
        return {}
    import yaml  # noqa: PLC0415 -- keeps the runner stdlib-only

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"error: {path} must be a YAML mapping.")
    return data
