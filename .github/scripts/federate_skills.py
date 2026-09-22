#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0"]
# ///
"""Vendor skills from the external repositories declared in `.github/federation.json`.

Each entry in that file names a GitHub repo, the branch to track, and the
exact path of every skill folder to vendor from it. The branch defaults to
`main`. It can also be a release pattern such as `release/*`, which tracks
the matching branch with the highest version (see `federation_branches.py`).
Tags and commits cannot be tracked, and the federation guard only lets a
pull request set the branch that the repo's owners approved in
`.github/skill_owners.json`, so a source repo cannot route unreviewed work
into the catalog through a side branch.

For each declared skill, the script:

1. Resolves the source's branch (picking the newest release branch for a
   pattern) and shallow-clones the repo at it into a temp directory, using
   sparse-checkout so only the declared skill paths are fetched.
2. Hashes the upstream skill folder and compares it against the hash
   recorded in the vendored copy's `.federated.json`. A skill whose
   upstream contents are unchanged is left byte-for-byte as it is on
   disk, so a run that finds no upstream change produces no diff at all
   and therefore no pull request. If a skill is re-vendored anyway and
   the resulting files match the copy already on disk, the previous
   marker is restored, because a diff consisting only of a commit and a
   hash is noise. Deleting a marker forces that skill to be re-vendored.
3. Copies changed skill folders into `skills/<name>/` as a mirror of
   upstream, except for the skill's top-level `evals/` folder, which is
   outside federation entirely: upstream's copy is never imported and the
   catalog's copy is never touched. Anything else the source repo does not
   ship is removed.
3b. Optionally vendors the skill under a different local catalog name (the
   `as:` field on a skill entry). Federated skills follow a
   `<projectrepo>-<skill>` naming convention in this catalog (e.g. the
   `analysis-orchestrator` skill from TraceLens is vendored as
   `tracelens-analysis-orchestrator`), so the local folder, marketplace
   entry, and the SKILL.md `name` frontmatter are all set to the `as:`
   value. The upstream folder name is still used to locate the skill in
   its source repo.
4. Writes `.federated.json` inside each copy with the source repo, the
   tracked branch (plus the branch it resolved to, for a pattern), the
   resolved commit, and the content hash, so we can tell
   vendored skills apart from skills authored in this repo and detect the
   next real upstream change.
5. Rewrites relative markdown links that point outside the copied skill
   folder (e.g. `examples/foo.yaml`, `docs/bar.md`) into absolute
   github.com URLs pinned to the imported commit, so the offline link
   checker doesn't flag them as missing local files. Links to files that
   were actually copied into the skill folder are left untouched.
6. Synthesizes a minimal `skill-card.md` (Description, Owner, License)
   from the source metadata when the upstream copy doesn't already ship
   one, so the imported skill satisfies the card validation gate (see
   docs/skill-requirements.md).
7. Adds each declared skill to the bundle's `skills` array in
   `.claude-plugin/marketplace.json` (as a `./skills/<name>` path) so it
   ships in the single AMD plugin.

Nothing is ever deleted here. A vendored skill that is no longer declared
in `.github/federation.json` is reported and left alone; removing it is a
deliberate pull request by a maintainer.

Usage:
    uv run .github/scripts/federate_skills.py                 # vendor changed skills
    uv run .github/scripts/federate_skills.py --check-catalog # validate the file
    uv run .github/scripts/federate_skills.py --only tracelens-analysis-orchestrator

The `--only` flag (repeatable) restricts the run to the named *local*
skill folder(s) (the `as:` name when one is set); other skills in the
catalog are left untouched.

`--summary-json PATH` writes a machine-readable record of the run
(including a ready-made pull request title and body) for the calling
workflow to consume.

The companion GitHub Actions workflow `federate-skills` runs this script
nightly and on demand, and opens a pull request with the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import federation_branches as branches  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CATALOG_FILE = REPO_ROOT / ".github" / "federation.json"
SKILLS_DIR = REPO_ROOT / "skills"
CLAUDE_MARKETPLACE = REPO_ROOT / ".claude-plugin" / "marketplace.json"
MARKER_FILENAME = ".federated.json"
# Never part of a skill's content: these appear only in a working copy, and
# hashing them would make change detection depend on whether someone happened
# to run the tests before the importer.
IGNORED_DIR_NAMES = {"__pycache__", ".pytest_cache"}
# Top-level skill folders federation does not carry. `evals/` is the catalog's
# to own: the datasets are maintained and run here, so importing upstream's
# copy would overwrite them and hashing either copy would tie change detection
# to a folder federation no longer moves. Matched at the skill root only, so a
# nested `agents/evals/` is still ordinary content.
UNFEDERATED_DIR_NAMES = {"evals"}
# The bundle references each published skill as `./skills/<name>` in the
# marketplace plugin entry's `skills` array.
SKILLS_PATH_PREFIX = "./skills/"
CARD_FILENAME = "skill-card.md"

FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<frontmatter>.*?)\n---\s*\n?(?P<body>.*)\Z",
    re.DOTALL,
)
# The `name:` line inside a SKILL.md frontmatter block. Used to rewrite the
# frontmatter `name` when a skill is vendored under a different local name.
NAME_FIELD_RE = re.compile(r"(?m)^(?P<key>name[ \t]*:[ \t]*)(?P<value>.*)$")
# Inline markdown links and images: `[text](target)` / `![alt](target)`,
# with an optional `"title"` after the target. The `target` group captures
# everything up to whitespace or the closing paren.
MARKDOWN_LINK_RE = re.compile(
    r"(?P<prefix>!?\[[^\]]*\]\()(?P<target>[^)\s]+)(?P<suffix>(?:\s+\"[^\"]*\")?\))"
)
# Anything with an explicit URI scheme (https://, mailto:, etc.).
URI_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
# Marketplace descriptions are read by humans browsing the catalog; truncate
# very long SKILL.md descriptions so the listing stays readable. The full
# description is still available in the vendored SKILL.md.
MARKETPLACE_DESCRIPTION_MAX = 320

REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
SOURCE_KEYS = {"repo", "license", "branch", "skills"}
SKILL_KEYS = {"path", "as", "marketplace_description"}
# Keys that would pin a source to something other than a branch. They are
# rejected with a pointed message rather than ignored, so an entry that tries
# to track a tag or commit fails loudly instead of silently tracking `main`.
PINNING_KEYS = {"ref", "tag", "commit", "rev", "revision"}


@dataclass
class SkillSpec:
    """One skill folder to vendor, addressed by its full path in the source repo."""

    path: str
    local_name: str | None = None
    marketplace_description_override: str | None = None

    @property
    def folder(self) -> str:
        """Upstream folder name (the last path segment)."""
        return posixpath.basename(self.path)

    @property
    def dest_name(self) -> str:
        """Local catalog name: the `as:` override, or the upstream folder."""
        return self.local_name or self.folder


@dataclass
class Source:
    repo: str
    license: str
    skills: list[SkillSpec] = field(default_factory=list)
    # As declared: a branch name, or a pattern such as `release/*`.
    branch: str = branches.DEFAULT_BRANCH

    @property
    def name(self) -> str:
        """Stable slug for the source, derived from `owner/repo`."""
        return re.sub(r"[^a-z0-9]+", "-", self.repo.lower()).strip("-")


@dataclass
class ImportResult:
    source: Source
    folder: str
    path: str
    commit: str
    updated: bool
    # The branch actually cloned: `source.branch` itself, or the newest
    # release branch it matched.
    branch: str = branches.DEFAULT_BRANCH
    skill_description: str = ""
    marketplace_description: str = ""

    @property
    def short_commit(self) -> str:
        return self.commit[:7]


def parse_federation(catalog: Path) -> list[Source]:
    """Parse and validate `.github/federation.json`.

    Unknown keys are rejected rather than ignored: the file is hand-edited by
    contributors registering a skill, and a typo that silently vendors the
    wrong thing (or nothing) is worse than a failed run.
    """
    if not catalog.exists():
        raise FileNotFoundError(f"Federation file not found: {catalog}")
    try:
        data = json.loads(catalog.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{catalog} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{catalog} must contain a JSON object.")
    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"{catalog} must define a non-empty `sources` array.")

    sources: list[Source] = []
    for idx, raw in enumerate(raw_sources):
        where = f"sources[{idx}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{where} must be an object.")
        pinning = PINNING_KEYS & set(raw)
        if pinning:
            raise ValueError(
                f"{where} sets {sorted(pinning)}, but federated sources track a "
                "branch, never a tag or commit. Use `branch` instead (e.g. "
                f"{branches.DEFAULT_BRANCH!r} or 'release/*'), or leave it out "
                f"to track {branches.DEFAULT_BRANCH!r}."
            )
        unknown = set(raw) - SOURCE_KEYS
        if unknown:
            raise ValueError(
                f"{where} has unknown key(s) {sorted(unknown)}; expected "
                f"{sorted(SOURCE_KEYS)}."
            )
        repo = raw.get("repo")
        if not isinstance(repo, str) or not REPO_RE.match(repo):
            raise ValueError(
                f'{where}.repo must be a GitHub "<owner>/<repo>" string, got '
                f"{repo!r}."
            )
        branch = branches.validate_branch(raw.get("branch"), f"{where}.branch")

        skills_raw = raw.get("skills")
        if not isinstance(skills_raw, list) or not skills_raw:
            raise ValueError(
                f"{where} ({repo}) must list at least one skill under `skills`."
            )

        skills: list[SkillSpec] = []
        for sk_idx, sk in enumerate(skills_raw):
            sk_where = f"{where}.skills[{sk_idx}]"
            if not isinstance(sk, dict):
                raise ValueError(
                    f"{sk_where} must be an object with a `path` key. Every "
                    "skill in a repo is declared individually."
                )
            unknown = set(sk) - SKILL_KEYS
            if unknown:
                raise ValueError(
                    f"{sk_where} has unknown key(s) {sorted(unknown)}; expected "
                    f"{sorted(SKILL_KEYS)}."
                )
            path = sk.get("path")
            if not isinstance(path, str) or not path.strip("/"):
                raise ValueError(
                    f"{sk_where}.path must be the skill folder's path inside "
                    f'{repo}, e.g. "skills/my-skill".'
                )
            path = path.replace("\\", "/").strip("/")
            if ".." in path.split("/"):
                raise ValueError(f"{sk_where}.path must not contain '..': {path!r}")
            local_name = sk.get("as")
            if local_name is not None and (
                not isinstance(local_name, str) or "/" in local_name
            ):
                raise ValueError(
                    f"{sk_where}.as must be a plain local folder name, got "
                    f"{local_name!r}."
                )
            skills.append(
                SkillSpec(
                    path=path,
                    local_name=local_name,
                    marketplace_description_override=sk.get("marketplace_description"),
                )
            )

        sources.append(
            Source(
                repo=repo,
                license=raw.get("license") or "UNKNOWN",
                skills=skills,
                branch=branch,
            )
        )
    return sources


def run(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a command, raise on failure, return stdout."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def remote_branches(repo: str) -> list[str]:
    """Names of every branch on `repo`, read without cloning."""
    out = run(["git", "ls-remote", "--heads", f"https://github.com/{repo}.git"])
    prefix = "refs/heads/"
    names = []
    for line in out.splitlines():
        _, _, ref = line.partition("\t")
        if ref.startswith(prefix):
            names.append(ref[len(prefix) :])
    return names


def resolve_branch(source: Source) -> str:
    """The branch to clone for `source`: its own, or its newest release branch."""
    if not branches.is_pattern(source.branch):
        return source.branch
    resolved = branches.latest_matching(source.branch, remote_branches(source.repo))
    if resolved is None:
        raise ValueError(
            f"{source.repo} has no branch matching {source.branch!r} with a "
            "version number in place of the '*'."
        )
    return resolved


def shallow_clone(repo: str, branch: str, sub_paths: Iterable[str], dest: Path) -> str:
    """Sparse + blobless clone of `repo` at `branch`, restricted to `sub_paths`.

    Returns the resolved commit SHA. Sparse-checkout avoids pulling the
    whole repo when only a few sub-trees are needed (the AMD-AGI/Apex tree
    is large; we only want `tools/skills`).

    Line endings are forced to LF so the checkout is byte-identical
    everywhere: the content hashes computed from it drive change detection,
    and they must mean the same thing on a maintainer's Windows machine and
    on a Linux runner.

    Background maintenance is disabled because it outlives the clone: a
    blobless clone hands commit-graph writing to a detached process that
    keeps creating files under `.git/objects/info/`, which then races the
    removal of the temp directory this clone lives in.
    """
    url = f"https://github.com/{repo}.git"
    run(
        [
            "git",
            "-c",
            "gc.auto=0",
            "-c",
            "maintenance.auto=false",
            "-c",
            "fetch.writeCommitGraph=false",
            "clone",
            "--filter=blob:none",
            "--sparse",
            "--no-checkout",
            "--single-branch",
            "--branch",
            branch,
            url,
            str(dest),
        ]
    )
    run(["git", "config", "core.autocrlf", "false"], cwd=dest)
    run(["git", "config", "core.eol", "lf"], cwd=dest)
    run(["git", "sparse-checkout", "set", "--cone", *sub_paths], cwd=dest)
    run(["git", "checkout", branch], cwd=dest)
    return run(["git", "rev-parse", "HEAD"], cwd=dest)


def content_hash(root: Path, exclude: Iterable[str] = ()) -> str:
    """Hash every file under `root` (relative paths and bytes).

    Change detection compares this against the hash stored in the vendored
    copy's marker, so a skill is re-vendored when its upstream folder's
    contents change, not merely when its source repo gets some unrelated
    commit. Without that distinction every commit to a large product repo
    would churn the vendored copies (the marker and the commit-pinned links)
    and open a pull request with no substance in it.

    Files are fed to the digest ordered by their POSIX relative path rather
    than by sorting `Path` objects, because `Path` comparison is
    case-insensitive on Windows: `sorted()` puts `SKILL.md` after `agents/`
    on a maintainer's machine and before it on a Linux runner, which would
    hash identical files to different digests.

    Files under an unfederated top-level folder are left out on both sides
    of the comparison, since federation neither reads nor writes them.

    `exclude` names relative paths to leave out of the digest.
    """
    skipped = set(exclude)
    entries = sorted(
        (
            (path.relative_to(root).as_posix(), path)
            for path in root.rglob("*")
            if path.is_file()
        ),
        key=lambda entry: entry[0],
    )
    digest = hashlib.sha256()
    for rel, path in entries:
        parts = rel.split("/")
        if (
            rel in skipped
            or IGNORED_DIR_NAMES.intersection(parts[:-1])
            or (len(parts) > 1 and parts[0] in UNFEDERATED_DIR_NAMES)
        ):
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def list_repo_files(clone_dir: Path, commit: str) -> set[str]:
    """Return every tracked path in the repo at `commit` (POSIX style).

    Uses `git ls-tree`, which reads tree objects only, so it works even on a
    blob-filtered, sparse checkout without fetching file contents.
    """
    out = run(["git", "ls-tree", "-r", "--name-only", commit], cwd=clone_dir)
    return {line.strip() for line in out.splitlines() if line.strip()}


def _should_skip_target(target: str) -> bool:
    """True for targets that are not repo-relative file paths.

    Skips absolute URLs (`https://...`), scheme links (`mailto:`), in-page
    anchors (`#section`), root-absolute paths (`/foo`), and protocol-relative
    URLs (`//host/...`).
    """
    t = target.strip()
    if not t:
        return True
    if t[0] in "#/":
        return True
    if URI_SCHEME_RE.match(t):
        return True
    return False


def rewrite_external_references(
    skill_dir: Path,
    repo_skill_path: str,
    repo_files: set[str],
    repo: str,
    commit: str,
    log: list[str],
) -> None:
    """Rewrite relative links that escape the skill folder into GitHub URLs.

    A vendored skill often links to files that live elsewhere in its source
    repo (e.g. `examples/foo.yaml`, `docs/bar.md`). Those paths don't exist
    inside the copied skill folder, so the offline link checker flags them as
    missing files. For each such link we point at the upstream repo on
    github.com, pinned to the imported `commit`.

    Links that resolve to a file actually present inside the skill folder
    (e.g. `reference.md`) are left untouched so they keep working locally.
    """
    repo_skill_path = repo_skill_path.strip("/")

    def replace_in(text: str) -> tuple[str, list[tuple[str, str]]]:
        rewrites: list[tuple[str, str]] = []

        def _sub(match: re.Match[str]) -> str:
            target = match.group("target")
            if _should_skip_target(target):
                return match.group(0)
            path_part, sep, anchor = target.partition("#")
            frag = sep + anchor if sep else ""
            if not path_part:
                return match.group(0)

            # Resolve the link both as the markdown spec would (relative to
            # the file's folder in the repo) and relative to the repo root,
            # since skill docs often write repo-root-relative paths.
            skill_rel = posixpath.normpath(posixpath.join(repo_skill_path, path_part))
            root_rel = posixpath.normpath(path_part)

            within_skill = skill_rel == repo_skill_path or skill_rel.startswith(
                repo_skill_path + "/"
            )
            if within_skill and skill_rel in repo_files:
                # Genuine intra-skill link; it was copied, leave it local.
                return match.group(0)

            if skill_rel in repo_files:
                chosen = skill_rel
            else:
                chosen = root_rel

            # Can't map something that points above the repo root.
            if chosen.startswith("..") or chosen.startswith("/"):
                return match.group(0)

            url = f"https://github.com/{repo}/blob/{commit}/{chosen}{frag}"
            rewrites.append((target, url))
            return f"{match.group('prefix')}{url}{match.group('suffix')}"

        return MARKDOWN_LINK_RE.sub(_sub, text), rewrites

    for md_path in sorted(skill_dir.rglob("*.md")):
        original = md_path.read_text(encoding="utf-8")
        updated, rewrites = replace_in(original)
        if updated != original:
            md_path.write_text(updated, encoding="utf-8")
            rel = md_path.relative_to(skill_dir.parent).as_posix()
            for old, new in rewrites:
                log.append(f"    [{rel}] {old} -> {new}")


def parse_frontmatter(text: str) -> dict:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}
    try:
        data = yaml.safe_load(match.group("frontmatter"))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def truncate_description(text: str, limit: int = MARKETPLACE_DESCRIPTION_MAX) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    # Cut at the last sentence boundary that still fits.
    cut = text[: limit - 1]
    last_period = cut.rfind(". ")
    if last_period >= int(limit * 0.6):
        return cut[: last_period + 1]
    return cut.rstrip(",;:") + "…"


def read_marker(skill_dir: Path) -> dict:
    """Parse a vendored skill's `.federated.json`, or return {} if absent.

    A corrupt marker reads as {} so the next run re-vendors the skill and
    rewrites the marker cleanly.
    """
    marker = skill_dir / MARKER_FILENAME
    if not marker.exists():
        return {}
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def find_federated_skills() -> dict[str, dict]:
    """Return {skill_folder_name: parsed marker JSON} for every existing
    skill that has a `.federated.json` marker."""
    found: dict[str, dict] = {}
    if not SKILLS_DIR.exists():
        return found
    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir() or skill_dir.name.startswith("."):
            continue
        if (skill_dir / MARKER_FILENAME).exists():
            found[skill_dir.name] = read_marker(skill_dir)
    return found


def is_up_to_date(
    marker: dict,
    source: Source,
    spec: SkillSpec,
    upstream_hash: str,
) -> bool:
    """True when the vendored copy already matches this upstream folder.

    The repo and path are compared alongside the hash because they change
    what the vendored copy should contain: markdown links are rewritten to
    absolute URLs under the source repo, so re-pointing a skill at a
    different repo or path has to re-vendor even if the files are identical.
    The declared branch is compared rather than the one a pattern resolved
    to, so a new release branch with the same skill contents is not a bump.
    """
    return (
        bool(upstream_hash)
        and marker.get("content_hash") == upstream_hash
        and marker.get("repo") == source.repo
        and marker.get("path") == spec.path
        and marker.get("ref") == source.branch
    )


def copy_skill(src: Path, dest: Path) -> None:
    """Mirror the upstream skill folder into `dest`.

    Anything upstream does not ship is removed, so a re-import is also how
    an upstream deletion propagates. The exception is the top-level folders
    in `UNFEDERATED_DIR_NAMES`: the copy already in the catalog is kept as
    it is, and upstream's is not imported over it.
    """
    if dest.is_dir():
        for entry in dest.iterdir():
            if entry.is_dir() and entry.name in UNFEDERATED_DIR_NAMES:
                continue
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    elif dest.exists():
        dest.unlink()

    src_root = Path(src)

    def ignore(directory: str, names: list[str]) -> set[str]:
        # `copytree` calls this for every directory it walks; only the skill's
        # own top level is exempt, so compare against the root it started at.
        if Path(directory) != src_root:
            return set()
        return {name for name in names if name in UNFEDERATED_DIR_NAMES}

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, ignore=ignore, dirs_exist_ok=True)


def write_marker(
    skill_dir: Path,
    source: Source,
    branch: str,
    commit: str,
    relative_path: str,
    upstream_hash: str,
) -> None:
    marker = {
        "source": source.name,
        "repo": source.repo,
        "ref": source.branch,
    }
    if branch != source.branch:
        marker["resolved_ref"] = branch
    marker |= {
        "commit": commit,
        "path": relative_path,
        "license": source.license,
        "content_hash": upstream_hash,
        "imported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (skill_dir / MARKER_FILENAME).write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )


def write_card(skill_dir: Path, source: Source, description: str) -> None:
    """Write a minimal skill-card.md unless the upstream copy shipped one.

    Federated skills are copied wholesale (`copy_skill` does rmtree +
    copytree), so any card authored here would be wiped on re-import. When
    upstream doesn't provide a card, synthesize one from the source metadata
    so the imported skill still satisfies the card validation gate.
    """
    card = skill_dir / CARD_FILENAME
    if card.exists():
        return
    owner_org = source.repo.split("/")[0]
    license_text = source.license or f"See [{source.repo}](https://github.com/{source.repo})"
    card.write_text(
        "# Skill Card\n\n"
        "## Description\n\n"
        f"{description}\n\n"
        "## Owner\n\n"
        f"{owner_org} (federated from "
        f"[{source.repo}](https://github.com/{source.repo}))\n\n"
        "## License\n\n"
        f"{license_text}\n",
        encoding="utf-8",
    )


def rewrite_skill_name(skill_dir: Path, new_name: str, log: list[str]) -> None:
    """Set the SKILL.md frontmatter `name` to `new_name`.

    Upstream ships its own `name` (e.g. `analysis-orchestrator`), but this
    repo's validator requires the frontmatter `name` to match the skill's
    directory name. When a skill is vendored under a different local name
    (the `as:` field), rewrite the frontmatter so the imported copy stays
    valid without hand-editing after every refresh.
    """
    skill_md = skill_dir / "SKILL.md"
    text = skill_md.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    if not match:
        return
    fm_start, fm_end = match.span("frontmatter")
    frontmatter = match.group("frontmatter")

    new_frontmatter, count = NAME_FIELD_RE.subn(
        lambda m: f"{m.group('key')}{new_name}", frontmatter, count=1
    )
    if count == 0:
        # No `name:` line to rewrite; prepend one so the copy stays valid.
        new_frontmatter = f"name: {new_name}\n{frontmatter}"
    if new_frontmatter == frontmatter:
        return
    skill_md.write_text(text[:fm_start] + new_frontmatter + text[fm_end:], encoding="utf-8")
    log.append(f"    [SKILL.md] name -> {new_name}")


def update_publish_list(declared: Iterable[str]) -> bool:
    """Sync the bundle's `skills` array in `.claude-plugin/marketplace.json`.

    AMD ships a single curated plugin whose `skills` array lists the published
    skills as `./skills/<name>` paths. Newly vendored federated skills are
    added so they ship in the bundle. The existing curation order is
    preserved; freshly added skills are appended in sorted order for a
    deterministic diff.

    Returns True when the file was modified.
    """
    data = json.loads(CLAUDE_MARKETPLACE.read_text(encoding="utf-8"))
    plugins = data.get("plugins")
    if not isinstance(plugins, list) or not plugins or not isinstance(plugins[0], dict):
        raise ValueError(
            f"{CLAUDE_MARKETPLACE.relative_to(REPO_ROOT)} must define a bundle "
            "plugin entry to sync federated skills into."
        )
    entry = plugins[0]
    skills = entry.get("skills")
    if not isinstance(skills, list):
        skills = []

    present = {
        s[len(SKILLS_PATH_PREFIX) :].strip("/")
        for s in skills
        if isinstance(s, str) and s.startswith(SKILLS_PATH_PREFIX)
    }
    additions = sorted(
        f"{SKILLS_PATH_PREFIX}{name}" for name in declared if name not in present
    )
    new_skills = skills + additions

    changed = new_skills != skills
    if changed:
        entry["skills"] = new_skills
        CLAUDE_MARKETPLACE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return changed


def import_source(
    source: Source,
    log: list[str],
) -> list[ImportResult]:
    results: list[ImportResult] = []
    # A failed cleanup must not fail the run: everything of value has already
    # been copied out by then, and leaving a temp directory behind is cheaper
    # than a red nightly job.
    with tempfile.TemporaryDirectory(
        prefix="amd-skills-import-", ignore_cleanup_errors=True
    ) as tmpdir:
        tmp_path = Path(tmpdir) / source.name
        branch = resolve_branch(source)
        if branch != source.branch:
            log.append(f"[{source.name}] {source.branch} resolved to {branch}")
        log.append(f"[{source.name}] cloning {source.repo}@{branch}")
        commit = shallow_clone(
            source.repo, branch, [spec.path for spec in source.skills], tmp_path
        )
        log.append(f"[{source.name}] resolved to commit {commit}")
        repo_files = list_repo_files(tmp_path, commit)

        for spec in source.skills:
            src_skill = tmp_path / spec.path
            if not src_skill.is_dir():
                raise FileNotFoundError(
                    f"Skill path {spec.path!r} not found in "
                    f"{source.repo}@{branch}."
                )
            skill_md = src_skill / "SKILL.md"
            if not skill_md.exists():
                raise FileNotFoundError(
                    f"Skill {spec.path!r} from {source.repo} has no SKILL.md."
                )

            dest_name = spec.dest_name
            dest_skill = SKILLS_DIR / dest_name
            marker = read_marker(dest_skill)
            upstream_hash = content_hash(src_skill)
            if is_up_to_date(marker, source, spec, upstream_hash):
                log.append(
                    f"[{source.name}] skills/{dest_name} is up to date "
                    f"(unchanged upstream); left untouched"
                )
                results.append(
                    ImportResult(
                        source=source,
                        folder=dest_name,
                        path=spec.path,
                        commit=marker.get("commit", commit),
                        updated=False,
                        branch=branch,
                    )
                )
                continue

            frontmatter = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            description = frontmatter.get("description") or ""
            if not isinstance(description, str) or not description.strip():
                raise ValueError(
                    f"Skill {spec.path!r} from {source.repo} has no "
                    "non-empty `description` in its SKILL.md frontmatter."
                )
            marketplace_description = (
                spec.marketplace_description_override
                or truncate_description(description)
            )

            renamed = f" (as {dest_name})" if dest_name != spec.folder else ""
            log.append(
                f"[{source.name}] vendoring {spec.path} -> "
                f"skills/{dest_name}{renamed}"
            )
            marker_path = dest_skill / MARKER_FILENAME
            previous_marker = (
                marker_path.read_bytes() if marker_path.is_file() else None
            )
            previous_content = (
                content_hash(dest_skill, exclude=[MARKER_FILENAME])
                if dest_skill.is_dir()
                else None
            )

            copy_skill(src_skill, dest_skill)
            write_marker(dest_skill, source, branch, commit, spec.path, upstream_hash)
            write_card(dest_skill, source, marketplace_description)
            rewrite_skill_name(dest_skill, dest_name, log)
            rewrite_external_references(
                dest_skill,
                spec.path,
                repo_files,
                source.repo,
                commit,
                log,
            )

            # The marker is bookkeeping, not content. If re-vendoring produced
            # the same files as the copy already on disk, put the old marker
            # back: a pull request whose entire diff is a hash and a timestamp
            # tells a reviewer nothing. A marker still naming the branch the
            # source used to track is kept out of that, or it would never move.
            if (
                previous_marker is not None
                and marker.get("ref") == source.branch
                and previous_content
                == content_hash(dest_skill, exclude=[MARKER_FILENAME])
            ):
                marker_path.write_bytes(previous_marker)
                log.append(
                    f"[{source.name}] skills/{dest_name} re-vendored to "
                    f"identical content; marker left at "
                    f"{marker.get('commit', '?')[:7]}"
                )
                results.append(
                    ImportResult(
                        source=source,
                        folder=dest_name,
                        path=spec.path,
                        commit=marker.get("commit", commit),
                        updated=False,
                        branch=branch,
                    )
                )
                continue

            results.append(
                ImportResult(
                    source=source,
                    folder=dest_name,
                    path=spec.path,
                    commit=commit,
                    updated=True,
                    branch=branch,
                    skill_description=description.strip(),
                    marketplace_description=marketplace_description,
                )
            )
    return results


def report_undeclared(
    declared: set[str],
    existing: dict[str, dict],
    log: list[str],
) -> None:
    """Note vendored skills that no source declares any more.

    They are reported rather than deleted: removing a shipped skill is a
    decision for a maintainer's pull request, not a side effect of a
    scheduled run.
    """
    for name, marker in sorted(existing.items()):
        if name in declared:
            continue
        log.append(
            f"[undeclared] skills/{name} is vendored (from "
            f"{marker.get('repo', '?')}) but no longer declared in "
            f"{CATALOG_FILE.name}; remove it in a pull request if that is "
            "intended"
        )


def commit_url(repo: str, commit: str) -> str:
    return f"https://github.com/{repo}/commit/{commit}"


def pr_title(updated: list[ImportResult]) -> str:
    """Compose the pull request title for a run.

    A single bumped skill — the common nightly case — reads
    "Bump `<skill>` to `<short commit>`". No bumps means no pull request,
    hence no title.
    """
    if not updated:
        return ""
    if len(updated) == 1:
        result = updated[0]
        return f"Bump `{result.folder}` to `{result.short_commit}`"
    return f"Bump {len(updated)} federated skills"


def pr_body(
    updated: list[ImportResult],
    unchanged: list[ImportResult],
) -> str:
    lines = [
        "Automated federation refresh driven by `.github/federation.json`.",
        "",
        "Each source is tracked at the branch its entry names "
        f"(`{branches.DEFAULT_BRANCH}` unless set), and a skill is re-vendored "
        "only when the contents of its upstream folder change.",
    ]
    if updated:
        lines += ["", "**Bumped**", ""]
        for result in sorted(updated, key=lambda r: r.folder):
            lines.append(
                f"- `{result.folder}` to "
                f"[`{result.short_commit}`]"
                f"({commit_url(result.source.repo, result.commit)}) "
                f"from `{result.source.repo}/{result.path}` "
                f"on `{result.branch}`"
                + (
                    f" (newest `{result.source.branch}`)"
                    if result.branch != result.source.branch
                    else ""
                )
            )
    if unchanged:
        lines += ["", "**Unchanged**", ""]
        for result in sorted(unchanged, key=lambda r: r.folder):
            lines.append(f"- `{result.folder}`")
    lines += [
        "",
        "Each vendored skill carries a `.federated.json` marker recording the "
        "source repo, the tracked branch, the resolved commit, and the content "
        "hash used for change detection.",
    ]
    return "\n".join(lines) + "\n"


def build_summary(results: list[ImportResult]) -> dict:
    """Machine-readable record of the run for the calling workflow."""
    updated = [r for r in results if r.updated]
    unchanged = [r for r in results if not r.updated]
    return {
        "changed": bool(updated),
        "title": pr_title(updated),
        "body": pr_body(updated, unchanged),
        "updated": [
            {
                "skill": r.folder,
                "repo": r.source.repo,
                "path": r.path,
                "branch": r.branch,
                "commit": r.commit,
                "short_commit": r.short_commit,
            }
            for r in sorted(updated, key=lambda r: r.folder)
        ],
        "unchanged": sorted(r.folder for r in unchanged),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=CATALOG_FILE,
        help=f"Path to the federation file (default: {CATALOG_FILE}).",
    )
    parser.add_argument(
        "--check-catalog",
        action="store_true",
        help=(
            "Validate the federation file and print what it declares, "
            "without cloning anything."
        ),
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="SKILL",
        help=(
            "Vendor only the named skill folder (repeatable). Skills not "
            "named here are left untouched."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        metavar="PATH",
        help="Write a JSON summary of the run (including a PR title and body).",
    )
    args = parser.parse_args(argv)

    sources = parse_federation(args.catalog)

    if args.check_catalog:
        seen: set[str] = set()
        for source in sources:
            for spec in source.skills:
                if spec.dest_name in seen:
                    raise ValueError(
                        f"Skill name collision: {spec.dest_name!r} is declared "
                        f"more than once in {args.catalog}."
                    )
                seen.add(spec.dest_name)
                print(
                    f"{source.repo}@{source.branch} {spec.path} "
                    f"-> skills/{spec.dest_name}"
                )
        print("")
        print(f"{args.catalog} is valid: {len(seen)} skill(s) declared.")
        return 0

    only = set(args.only or [])
    if only:
        known = {spec.dest_name for source in sources for spec in source.skills}
        unknown = only - known
        if unknown:
            raise ValueError(
                "--only names skill(s) not present in the catalog: "
                + ", ".join(sorted(unknown))
            )
        for source in sources:
            source.skills = [s for s in source.skills if s.dest_name in only]
        sources = [source for source in sources if source.skills]
    log: list[str] = []
    declared: set[str] = set()
    all_results: list[ImportResult] = []

    SKILLS_DIR.mkdir(exist_ok=True)
    existing_federated = find_federated_skills()

    for source in sources:
        for spec in source.skills:
            if spec.dest_name in declared:
                raise ValueError(
                    f"Skill name collision: {spec.dest_name!r} is declared by "
                    f"more than one source in {args.catalog}."
                )
            declared.add(spec.dest_name)
        all_results.extend(import_source(source, log))

    # With --only the skills the caller didn't name are deliberately ignored,
    # so they would all look undeclared; skip the report in that case.
    if not only:
        report_undeclared(declared, existing_federated, log)

    publish_changed = update_publish_list(declared)

    for line in log:
        print(line)

    summary = build_summary(all_results)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    print("")
    print(f"Bumped: {len(summary['updated'])} skill(s)")
    print(f"Already up to date: {len(summary['unchanged'])} skill(s)")
    print(f"Publish list: {'changed' if publish_changed else 'unchanged'}")
    if summary["title"]:
        print(f"Pull request title: {summary['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
