"""Which branch of a product repo federation tracks.

Used by `federate_skills.py`, which clones the branch, and by
`federation_guard.py`, which links to it. Standard library only.

A branch is one of:

- a plain branch name, e.g. `main` or `develop`, tracked as it moves;
- a release pattern with a single `*` standing for a version, e.g.
  `release/*`. It resolves to the matching branch with the highest version,
  so a newly pushed release is picked up on the next run.

What the `*` may stand for is a version number, optionally `v`-prefixed and
optionally followed by an `alpha`, `beta`, or `rc` pre-release tag:
`0.13`, `v1.2.3`, `0.13-rc1`, `0.13rc2`, `0.13-beta.1`. Versions are compared
numerically, and a pre-release sorts below its final release but above
anything older:

    release/0.12 < release/0.13-rc1 < release/0.13-rc2 < release/0.13 < release/0.14-rc1

Branches whose `*` part is not a version (`release/next`, `release/0.13-wip`)
never match.

Branch creation time is deliberately not used: git does not record it, and
the newest commit date would jump back to an old release whenever a fix is
backported to it.

Omitting the branch means `main`.
"""

from __future__ import annotations

import re

DEFAULT_BRANCH = "main"
WILDCARD = "*"

# Deliberately narrower than what git accepts: a name that needs quoting, or
# that git would read as an option or a range, has no business in a registry.
BRANCH_RE = re.compile(r"^(?!-)(?!.*\.\.)(?!.*//)[A-Za-z0-9._/*-]+(?<![/.])$")
VERSION_RE = re.compile(
    r"v?(?P<release>\d+(?:\.\d+)*)"
    r"(?:[-.]?(?P<pre>alpha|beta|rc)[-.]?(?P<pre_number>\d+)?)?",
    re.IGNORECASE,
)
PRE_RELEASE_RANK = {"alpha": 0, "beta": 1, "rc": 2}


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


def version_key(text: str) -> tuple | None:
    """Sort key for a version such as `0.13` or `0.13-rc1`, or None if not one."""
    match = VERSION_RE.fullmatch(text)
    if not match:
        return None
    release = [int(part) for part in match.group("release").split(".")]
    # `1.0` and `1` are the same release.
    while len(release) > 1 and release[-1] == 0:
        release.pop()
    pre = match.group("pre")
    if pre is None:
        return (tuple(release), 1, 0, 0)
    return (
        tuple(release),
        0,
        PRE_RELEASE_RANK[pre.lower()],
        int(match.group("pre_number") or 0),
    )


def latest_matching(pattern: str, branches: list[str]) -> str | None:
    """Return the highest-versioned branch matching `pattern`, or None."""
    prefix, _, suffix = pattern.partition(WILDCARD)
    best: tuple[tuple, str] | None = None
    for name in branches:
        if len(name) <= len(prefix) + len(suffix):
            continue
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        key = version_key(name[len(prefix) : len(name) - len(suffix)])
        if key is None:
            continue
        candidate = (key, name)
        if best is None or candidate > best:
            best = candidate
    return best[1] if best else None
