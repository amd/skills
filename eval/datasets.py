# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Per-skill eval datasets: discovery, parsing, and structural validation.

Every skill owns one dataset at ``skills/<name>/evals/evals.json``, holding
two arrays of prompts:

  * ``expected_matches``    -- this skill should trigger.
  * ``expected_no_matches`` -- no skill should trigger.

A case is a user prompt plus what should be true after the agent sees it, and
the same case feeds both run modes:

  * **routing** -- the whole catalog is installed and only the trigger
    decision is graded ("did the right skill fire, and only then?").
  * **behavior** -- just this skill is installed, the run goes to completion,
    and ``should`` / ``should_not`` / ``logs_contain`` / ``files_exist`` are
    graded ("once it fired, did it do the job?").

Writing the prompt once for both is the point: a routing prompt that nothing
grades is a prompt nobody maintains, and a behavioral test that re-asserts
routing with a substring match is a worse version of a check this module
already models as a field.

The folder is the identity and the array is the expectation, so no case names
a skill. ``skills/serving-llms-on-epyc/evals/evals.json`` is a dataset about
``serving-llms-on-epyc``, and a case under its ``expected_matches`` is::

    {"id": "epyc-vllm-zentorch", "prompt": "Serve Llama 3.1 8B with zentorch."}

A prompt that should trigger a *different* skill belongs in that skill's
``expected_matches``, not here: routing installs the whole catalog at once, so
it is the same assertion either way, and filing it under the neighbour keeps
"no skill should fire" meaning exactly that.

Prompt categories are derived rather than declared, because the array a case
lives in and the file it lives in already carry the distinction:

  * ``expected_matches``                        -> ``positive``
  * ``expected_no_matches`` in a skill's dataset -> ``near_miss`` (its owner
    wrote it precisely because it sits close to that skill)
  * a case in the shared pool                    -> ``unrelated`` (belongs to
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

# The two arrays a dataset is made of. The name of the array a case sits in is
# the whole expectation, which is why no case carries a field naming a skill.
MATCH_KEY = "expected_matches"
NO_MATCH_KEY = "expected_no_matches"

# `additionalProperties: false`, by hand. A mistyped key would otherwise be
# silently dropped, quietly turning an expectation into no expectation at all.
#
# The two arrays take different fields. A case where nothing should trigger has
# no skill to grade, so every assertion that asks whether something *happened*
# -- `should`, `logs_contain`, `files_exist` -- would be grading the base
# model. Only `should_not` makes sense there.
MATCH_CASE_KEYS = {
    "id",
    "prompt",
    "should",
    "should_not",
    "logs_contain",
    "files_exist",
    "workspace",
    "note",
}
NO_MATCH_CASE_KEYS = {"id", "prompt", "should_not", "workspace", "note"}

# `$schema` is allowed so a dataset can point editors at
# eval/schema/evals.schema.json for autocomplete; the runner ignores it.
DATASET_KEYS = {MATCH_KEY, NO_MATCH_KEY, "_comment", "$schema"}

# JSON has no comments, so `note` is the sanctioned place for one. The runner
# ignores it; without it owners annotate fields that are not free text.
_STRING_LISTS = ("should", "should_not", "logs_contain", "files_exist")


@dataclass
class Case:
    """One prompt and everything that should be true after the agent sees it."""

    id: str
    prompt: str
    # The skill whose dataset this came from; None for the shared pool.
    skill: str | None
    # Which array this came from: expected_matches, or expected_no_matches.
    expects_match: bool
    should: list[str] = field(default_factory=list)
    should_not: list[str] = field(default_factory=list)
    logs_contain: list[str] = field(default_factory=list)
    files_exist: list[str] = field(default_factory=list)
    # Directory (relative to the skill root) whose contents seed the workspace.
    workspace: str | None = None
    note: str = ""

    @property
    def expect_skill(self) -> str | None:
        """The skill that must activate, or None when nothing should.

        Derived, never written down: the owning folder names the skill and the
        array says whether it should fire.
        """
        return self.skill if self.expects_match else None

    @property
    def category(self) -> str:
        """Reporting bucket, derived from the array and the source file.

        Kept out of the file format on purpose: an owner who has to classify a
        prompt will eventually classify one wrong, and every input needed to
        do it correctly is already here.
        """
        if self.expects_match:
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


def _parse_case(
    entry: object, skill: str | None, expects_match: bool, label: str, errors: list[str]
) -> Case | None:
    """Turn one array element into a Case, appending any problems found."""
    allowed = MATCH_CASE_KEYS if expects_match else NO_MATCH_CASE_KEYS

    if not isinstance(entry, dict):
        errors.append(f"{label} must be an object.")
        return None

    unknown = set(entry) - allowed
    # Called out separately from a plain typo: these are real fields, used in
    # the wrong array, and the reason they are rejected is worth saying.
    misplaced = sorted(unknown & (MATCH_CASE_KEYS - NO_MATCH_CASE_KEYS))
    if misplaced:
        errors.append(
            f"{label} uses {', '.join(f'`{k}`' for k in misplaced)}, which "
            f"only work under `{MATCH_KEY}`. No skill loads for a prompt in "
            f"`{NO_MATCH_KEY}`, so an assertion that something happened would "
            "grade the base model. Use `should_not` to pin down what must not."
        )
    unknown = sorted(unknown - set(misplaced))
    if unknown:
        array = MATCH_KEY if expects_match else NO_MATCH_KEY
        errors.append(
            f"{label} has unknown key(s): {', '.join(unknown)}. "
            f"A case in `{array}` allows: {', '.join(sorted(allowed))}."
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

    if skill is None and any(lists[key] for key in _STRING_LISTS):
        # Behavior mode stages exactly one skill, and these cases belong to
        # none, so the assertions would be silently skipped rather than run.
        errors.append(
            f"{label} (`{case_id}`): shared negatives are routing-only and "
            "cannot carry behavioral assertions. Move the case into the "
            f"`{NO_MATCH_KEY}` of the skill it should not trigger."
        )
        return None

    return Case(
        id=case_id,
        prompt=prompt.strip(),
        skill=skill,
        expects_match=expects_match,
        should=lists["should"],
        should_not=lists["should_not"],
        logs_contain=lists["logs_contain"],
        files_exist=lists["files_exist"],
        workspace=workspace.strip() if isinstance(workspace, str) else None,
        note=note,
    )


def _parse_cases(payload: object, skill: str | None, source: Path, errors: list[str]) -> list[Case]:
    """Turn one parsed dataset file into cases, appending any problems found."""
    where = source.name

    if not isinstance(payload, dict):
        errors.append(
            f"{where}: top level must be an object with `{MATCH_KEY}` and "
            f"`{NO_MATCH_KEY}` arrays."
        )
        return []

    unknown = sorted(set(payload) - DATASET_KEYS)
    if unknown:
        errors.append(f"{where}: unknown top-level key(s): {', '.join(unknown)}.")

    # The shared pool describes no skill, so it has nothing to match.
    if skill is None and payload.get(MATCH_KEY):
        errors.append(
            f"{where}: the shared pool holds prompts no skill should answer, so "
            f"it cannot have `{MATCH_KEY}`. A prompt that should trigger a skill "
            "belongs in that skill's dataset."
        )

    cases: list[Case] = []
    empty = True
    for key, expects_match in ((MATCH_KEY, True), (NO_MATCH_KEY, False)):
        raw = payload.get(key, [])
        if not isinstance(raw, list):
            errors.append(f"{where}: `{key}` must be an array.")
            continue
        empty = empty and not raw
        if skill is None and expects_match:
            continue  # already reported above; parsing it would only add noise
        for index, entry in enumerate(raw):
            case = _parse_case(entry, skill, expects_match, f"{where}: {key}[{index}]", errors)
            if case is not None:
                cases.append(case)

    if empty:
        errors.append(f"{where}: needs at least one case in `{MATCH_KEY}` or `{NO_MATCH_KEY}`.")
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
            f"{MIN_POSITIVE_CASES} prompts under `{MATCH_KEY}` and "
            f"{MIN_NEGATIVE_CASES} under `{NO_MATCH_KEY}`. "
            "Copy eval/TEMPLATE.json to start."
        ]

    errors: list[str] = []
    positive = sum(1 for c in cases if c.expects_match)
    negative = len(cases) - positive
    if positive < MIN_POSITIVE_CASES:
        errors.append(
            f"{skill}: {positive} case(s) in `{MATCH_KEY}`; Tier 0 needs at "
            f"least {MIN_POSITIVE_CASES}. Add prompts a real user would type."
        )
    if negative < MIN_NEGATIVE_CASES:
        errors.append(
            f"{skill}: {negative} case(s) in `{NO_MATCH_KEY}`; Tier 0 needs at "
            f"least {MIN_NEGATIVE_CASES}. Add prompts close to this skill's "
            "domain that should NOT trigger it."
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
