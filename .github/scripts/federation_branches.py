"""Which branch of a product repo federation tracks.

Shared by `federate_skills.py`, which clones the branch, `federation_guard.py`,
which checks that the branch is the one approved, and `record_skill_owner.py`,
which records the approved branch. Standard library only, because
`record_skill_owner.py` runs without the others' dependencies.

A branch is one of:

- a plain branch name, e.g. `main` or `develop`, tracked as it moves;
- a release pattern with a single `*` standing for a version number, e.g.
  `release/*`. It resolves to the matching branch with the highest version,
  so `release/0.12` wins over `release/0.9`, and a new `release/0.13` is
  picked up by the next federation run after it is pushed. Branches whose `*` part is not a version
  (`release/next`, `release/0.13-wip`) never match.

Omitting the branch means `main`.
"""

from __future__ import annotations

import re

DEFAULT_BRANCH = "main"
WILDCARD = "*"

# Deliberately narrower than what git accepts: a name that needs quoting, or
# that git would read as an option or a range, has no business in a registry.
BRANCH_RE = re.compile(r"^(?!-)(?!.*\.\.)(?!.*//)[A-Za-z0-9._/*-]+(?<![/.])$")
VERSION_RE = re.compile(r"v?(\d+(?:\.\d+)*)")


def validate_branch(value: object, where: str = "branch") -> str:
    """Return `value` as a branch, or raise ValueError saying what is wrong."""
    if value is None:
        return DEFAULT_BRANCH
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty branch name, got {value!r}.")
    branch = value.strip()
    if not BRANCH_RE.match(branch) or branch.endswith(".lock"):
        raise ValueError(
            f"{where} {branch!r} is not a branch name. Use a name such as "
            f"{DEFAULT_BRANCH!r}, or a release pattern such as 'release/*'."
        )
    if branch.count(WILDCARD) > 1:
        raise ValueError(
            f"{where} {branch!r} has more than one '*'. A pattern has a single "
            "'*' standing for the version, as in 'release/*'."
        )
    return branch


def is_pattern(branch: str) -> bool:
    return WILDCARD in branch


def describe(branch: str) -> str:
    """Human wording for a branch, for pull request bodies and comments."""
    if is_pattern(branch):
        return f"the newest `{branch}` branch"
    return f"`{branch}`"


def latest_matching(pattern: str, branches: list[str]) -> str | None:
    """Return the highest-versioned branch matching `pattern`, or None."""
    prefix, _, suffix = pattern.partition(WILDCARD)
    best: tuple[tuple[int, ...], str] | None = None
    for name in branches:
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        middle = name[len(prefix) : len(name) - len(suffix)]
        match = VERSION_RE.fullmatch(middle)
        if not match:
            continue
        version = tuple(int(part) for part in match.group(1).split("."))
        candidate = (version, name)
        if best is None or candidate > best:
            best = candidate
    return best[1] if best else None
