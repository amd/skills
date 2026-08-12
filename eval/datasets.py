# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Per-skill eval datasets: discovery, parsing, and structural validation.

Every skill owns one dataset at ``skills/<name>/evals/evals.json``, holding an
``evaluations`` array. Each evaluation is a user prompt, a yes/no answer to
"should this skill fire?", and -- when the answer is yes -- what should be
true once it has::

    {
      "id": "epyc-vllm-zentorch",
      "skill_should_trigger": true,
      "prompt": "Serve Llama 3.1 8B with zentorch."
    }

There are two run modes to satisfy:

  * **routing** -- the whole catalog is installed and only the trigger
    decision is graded ("did the right skill fire, and only then?"). Every
    evaluation runs here.
  * **behavior** -- just this skill is installed, the run goes to completion,
    and ``expected_behavior`` / ``unexpected_behavior`` / ``logs_contain`` /
    ``files_exist`` are graded ("once it fired, did it do the job?"). Only a
    triggering evaluation can run here, and only if it asserts something.

One prompt graded by both is the point: a routing prompt that nothing grades
is a prompt nobody maintains, and a behavioral test that re-asserts routing
with a substring match is a worse version of a check this module already
models as a field.

``skill_should_trigger: false`` makes the evaluation routing-only. No skill
loads for it, so there is no behavior phase to hang an assertion or a staged
workspace off, and those fields are rejected rather than silently ignored:
such an evaluation is an ``id``, a ``prompt``, the flag, and maybe a ``note``.

The folder is the identity, so no evaluation names a skill;
``skills/serving-llms-on-epyc/evals/evals.json`` is about
``serving-llms-on-epyc`` and ``skill_should_trigger`` refers to it. A prompt
that should trigger a *different* skill belongs in that skill's dataset:
routing installs the whole catalog at once, so it is the same assertion either
way, and filing it under the neighbour keeps ``false`` meaning "nothing fires".

Prompt categories are derived rather than declared, because the flag and the
file a prompt lives in already carry the distinction:

  * ``skill_should_trigger: true``               -> ``positive``
  * ``false`` in a skill's own dataset           -> ``near_miss`` (its owner
    wrote it precisely because it sits close to that skill)
  * an evaluation in the shared pool             -> ``unrelated`` (belongs to
    no skill's domain)

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

# A dataset is one array of evaluations, and every evaluation answers the
# routing question outright rather than leaving it to be inferred.
EVALUATIONS_KEY = "evaluations"
TRIGGER_KEY = "skill_should_trigger"

# `additionalProperties: false`, by hand. A mistyped key would otherwise be
# silently dropped, quietly turning an expectation into no expectation at all.
#
# The two shapes take different fields, and the difference is not a style
# choice. An evaluation with `skill_should_trigger: false` is graded on exactly
# one thing -- that nothing fired -- and no skill is ever loaded for it, so
# there is no behavior phase to hang an assertion or a staged workspace off.
# Those are a prompt and nothing more.
TRIGGER_CASE_KEYS = {
    "id",
    "prompt",
    TRIGGER_KEY,
    "expected_behavior",
    "unexpected_behavior",
    "logs_contain",
    "files_exist",
    "workspace",
    "note",
}
NO_TRIGGER_CASE_KEYS = {"id", "prompt", TRIGGER_KEY, "note"}

DATASET_KEYS = {EVALUATIONS_KEY, "comment"}

# JSON has no comments, so `note` is the sanctioned place for one. The runner
# ignores it; without it owners annotate fields that are not free text.
_STRING_LISTS = ("expected_behavior", "unexpected_behavior", "logs_contain", "files_exist")


@dataclass
class Case:
    """One prompt and everything that should be true after the agent sees it."""

    id: str
    prompt: str
    # The skill whose dataset this came from; None for the shared pool.
    skill: str | None
    skill_should_trigger: bool
    expected_behavior: list[str] = field(default_factory=list)
    unexpected_behavior: list[str] = field(default_factory=list)
    logs_contain: list[str] = field(default_factory=list)
    files_exist: list[str] = field(default_factory=list)
    # Directory (relative to the skill root) whose contents seed the workspace.
    workspace: str | None = None
    note: str = ""

    @property
    def expect_skill(self) -> str | None:
        """The skill that must activate, or None when nothing should.

        Derived, never written down: the owning folder names the skill and
        `skill_should_trigger` says whether it should fire.
        """
        return self.skill if self.skill_should_trigger else None

    @property
    def category(self) -> str:
        """Reporting bucket, derived from the flag and the source file.

        Kept out of the file format on purpose: an owner who has to classify a
        prompt will eventually classify one wrong, and every input needed to
        do it correctly is already here.
        """
        if self.skill_should_trigger:
            return "positive"
        return "near_miss" if self.skill else "unrelated"

    @property
    def has_behavior(self) -> bool:
        """Whether this case grades anything beyond the routing decision."""
        return bool(
            self.expected_behavior
            or self.unexpected_behavior
            or self.logs_contain
            or self.files_exist
        )


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


def _parse_case(entry: object, skill: str | None, label: str, errors: list[str]) -> Case | None:
    """Turn one array element into a Case, appending any problems found."""
    if not isinstance(entry, dict):
        errors.append(f"{label} must be an object.")
        return None

    should_trigger = entry.get(TRIGGER_KEY)
    if not isinstance(should_trigger, bool):
        errors.append(
            f"{label} needs `{TRIGGER_KEY}`: true if this prompt should activate "
            "the skill that owns this file, false if no skill should fire at all."
        )
        return None

    if skill is None and should_trigger:
        errors.append(
            f"{label}: the shared pool belongs to no skill, so every evaluation "
            f"in it needs `{TRIGGER_KEY}: false`. A prompt that should trigger a "
            "skill belongs in that skill's dataset."
        )
        return None

    allowed = TRIGGER_CASE_KEYS if should_trigger else NO_TRIGGER_CASE_KEYS
    unknown = set(entry) - allowed
    # Called out separately from a plain typo: these are real fields on the
    # wrong kind of evaluation, and the reason they are rejected is worth saying.
    misplaced = sorted(unknown & (TRIGGER_CASE_KEYS - NO_TRIGGER_CASE_KEYS))
    if misplaced:
        errors.append(
            f"{label} uses {', '.join(f'`{k}`' for k in misplaced)}, which only "
            f"apply when `{TRIGGER_KEY}` is true. An evaluation expecting nothing "
            "to fire is graded on that alone -- no skill is ever loaded for it, "
            "so there is no behavior phase to assert anything about."
        )
    unknown = sorted(unknown - set(misplaced))
    if unknown:
        errors.append(
            f"{label} has unknown key(s): {', '.join(unknown)}. "
            f"Allowed here: {', '.join(sorted(allowed))}."
        )

    case_id = entry.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        errors.append(f"{label} is missing a non-empty string `id`.")
        return None
    case_id = case_id.strip()

    prompt = entry.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        errors.append(f"{label} (`{case_id}`) is missing a non-empty string `prompt`.")
        return None

    lists: dict[str, list[str]] = {}
    for key in _STRING_LISTS:
        value = entry.get(key, [])
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            errors.append(
                f"{label} (`{case_id}`): `{key}` must be an array of non-empty strings."
            )
            return None
        lists[key] = [item.strip() for item in value]

    workspace = entry.get("workspace")
    if workspace is not None and (not isinstance(workspace, str) or not workspace.strip()):
        errors.append(f"{label} (`{case_id}`): `workspace` must be a directory path.")
        return None

    note = entry.get("note", "")
    if not isinstance(note, str):
        errors.append(f"{label} (`{case_id}`): `note` must be a string.")
        return None

    return Case(
        id=case_id,
        prompt=prompt.strip(),
        skill=skill,
        skill_should_trigger=should_trigger,
        expected_behavior=lists["expected_behavior"],
        unexpected_behavior=lists["unexpected_behavior"],
        logs_contain=lists["logs_contain"],
        files_exist=lists["files_exist"],
        workspace=workspace.strip() if isinstance(workspace, str) else None,
        note=note,
    )


def _parse_cases(payload: object, skill: str | None, source: Path, errors: list[str]) -> list[Case]:
    """Turn one parsed dataset file into cases, appending any problems found."""
    where = source.name

    if not isinstance(payload, dict):
        errors.append(f"{where}: top level must be an object with an `{EVALUATIONS_KEY}` array.")
        return []

    unknown = sorted(set(payload) - DATASET_KEYS)
    if unknown:
        errors.append(f"{where}: unknown top-level key(s): {', '.join(unknown)}.")

    raw = payload.get(EVALUATIONS_KEY)
    if not isinstance(raw, list) or not raw:
        errors.append(f"{where}: `{EVALUATIONS_KEY}` must be a non-empty array.")
        return []

    cases: list[Case] = []
    for index, entry in enumerate(raw):
        case = _parse_case(entry, skill, f"{where}: {EVALUATIONS_KEY}[{index}]", errors)
        if case is not None:
            cases.append(case)
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
    selected = [case for case in cases if case.id in wanted or case.skill in wanted]
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
        # Only reachable for a triggering evaluation, which is the only kind
        # that owns a skill and the only kind allowed to stage anything.
        if case.workspace and case.skill:
            if not (SKILLS_DIR / case.skill / case.workspace).is_dir():
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
            f"{MIN_POSITIVE_CASES} evaluations where `{TRIGGER_KEY}` is true "
            f"and {MIN_NEGATIVE_CASES} where it is false. "
            "Copy eval/TEMPLATE.json to start."
        ]

    errors: list[str] = []
    positive = sum(1 for c in cases if c.skill_should_trigger)
    negative = len(cases) - positive
    if positive < MIN_POSITIVE_CASES:
        errors.append(
            f"{skill}: {positive} evaluation(s) with `{TRIGGER_KEY}: true`; "
            f"Tier 0 needs at least {MIN_POSITIVE_CASES}. Add prompts a real "
            "user would type."
        )
    if negative < MIN_NEGATIVE_CASES:
        errors.append(
            f"{skill}: {negative} evaluation(s) with `{TRIGGER_KEY}: false`; "
            f"Tier 0 needs at least {MIN_NEGATIVE_CASES}. Add prompts close to "
            "this skill's domain that should NOT trigger it."
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
