#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0"]
# ///
"""Import skills from external repositories listed in `.github/scripts/sources.yml`.

For each source, the script:

1. Shallow-clones the repo at the pinned `ref` into a temp directory,
   using sparse-checkout so only the configured `path` is fetched.
2. Copies each named skill folder into `skills/<skill>/`. When refreshing an
   existing import, a local `evals/` subdirectory is kept if the upstream
   skill folder does not ship one (catalog-authored behavioral tests).
   Sources that set `resolve_wrappers: true` may expose lightweight public
   wrappers whose body points at a repository-owned implementation SKILL.md.
   The importer resolves that entrypoint, copies its adjacent resources, and
   bundles any non-public helper skills as one-level references so the
   installed catalog skill remains self-contained.
2b. Optionally vendors the skill under a different local catalog name (the
   `as:` field on a skill entry). Federated skills follow a
   `<projectrepo>-<skill>` naming convention in this catalog (e.g. the
   `analysis-orchestrator` skill from TraceLens is vendored as
   `tracelens-analysis-orchestrator`), so the local folder, marketplace
   entry, and the SKILL.md `name` frontmatter are all set to the `as:`
   value. The upstream folder name is still used to locate the skill in
   its source repo.
3. Writes `.federated.json` inside each copy with source metadata so we
   can tell vendored skills apart from skills authored in this repo.
4. Rewrites relative markdown links that point outside the copied skill
   folder (e.g. `examples/foo.yaml`, `docs/bar.md`) into absolute
   github.com URLs pinned to the imported commit, so the offline link
   checker doesn't flag them as missing local files. Links to files that
   were actually copied into the skill folder are left untouched.
5. Synthesizes a minimal `skill-card.md` (Description, Owner, License)
   from the source metadata when the upstream copy doesn't already ship
   one, so the imported skill satisfies the card validation gate (see
   docs/skill-cards.md).
6. Adds each imported skill to the bundle's `skills` array in
   `.claude-plugin/marketplace.json` (as a `./skills/<name>` path) so it
   ships in the single AMD plugin.
7. Removes any previously imported skill (one with a `.federated.json`)
   that is no longer listed in `.github/scripts/sources.yml`, and drops it
   from the bundle's `skills` array.

Usage:
    uv run .github/scripts/import_external_skills.py            # write changes
    uv run .github/scripts/import_external_skills.py --dry-run  # report only
    uv run .github/scripts/import_external_skills.py --only magpie-kernel-evaluator

The `--only` flag (repeatable) restricts the run to the named *local*
skill folder(s) (the `as:` name when one is set): other skills in the
catalog are skipped and pruning is limited to the named skills, so
unrelated federated skills are never removed.

The companion GitHub Actions workflow `import-external-skills` calls this
script on manual dispatch and opens a pull request with the result.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import shutil
import subprocess
import tempfile
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CATALOG_FILE = Path(__file__).resolve().parent / "sources.yml"
SKILLS_DIR = REPO_ROOT / "skills"
CLAUDE_MARKETPLACE = REPO_ROOT / ".claude-plugin" / "marketplace.json"
MARKER_FILENAME = ".federated.json"
# Local-only subdirectories preserved across re-import when absent upstream.
PRESERVE_IF_ABSENT_UPSTREAM = ("evals",)
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

# Exact public-wrapper body supported by `resolve_wrappers`. Requiring the
# whole body to match keeps resolution explicit and prevents prose from being
# interpreted as a filesystem instruction.
SKILL_POINTER_RE = re.compile(
    r"\A\s*Read and follow the instructions in `(?P<path>[^`]+/SKILL\.md)`\.\s*\Z"
)
# Runtime snippets in repository-native skills sometimes assume execution from
# the source checkout. Standalone catalog copies instead resolve resources from
# the loaded SKILL.md location.
SKILL_DIR_ASSIGNMENT_RE = re.compile(
    r"(?m)^(?P<indent>[ \t]*)SKILL_DIR=\.claude/skills-impl/[^\s]+[ \t]*$"
)


@dataclass
class SkillSpec:
    folder: str
    local_name: str | None = None
    marketplace_description_override: str | None = None
    aliases: dict[str, str] | None = None

    @property
    def dest_name(self) -> str:
        """Local catalog name: the `as:` override, or the upstream folder."""
        return self.local_name or self.folder


@dataclass
class Source:
    name: str
    repo: str
    ref: str
    path: str
    license: str
    skills: list[SkillSpec]
    resolve_wrappers: bool = False
    aliases: dict[str, str] | None = None
    selected_skills: list[SkillSpec] | None = None

    @property
    def import_skills(self) -> list[SkillSpec]:
        return self.selected_skills if self.selected_skills is not None else self.skills


@dataclass
class ResolvedSkill:
    spec: SkillSpec
    public_dir: Path
    implementation_dir: Path
    public_path: str
    implementation_path: str
    public_frontmatter: dict
    implementation_name: str


@dataclass
class ImportResult:
    source: Source
    folder: str
    commit: str
    skill_description: str
    marketplace_description: str


def parse_sources(catalog: Path) -> list[Source]:
    if not catalog.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog}")
    data = yaml.safe_load(catalog.read_text(encoding="utf-8")) or {}
    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"{catalog} must define a non-empty `sources` list.")

    sources: list[Source] = []
    for idx, raw in enumerate(raw_sources):
        if not isinstance(raw, dict):
            raise TypeError(f"sources[{idx}] must be a mapping.")
        try:
            name = raw["name"]
            repo = raw["repo"]
            ref = raw["ref"]
            path = raw["path"]
        except KeyError as exc:
            raise ValueError(
                f"sources[{idx}] is missing required key: {exc.args[0]!r}"
            ) from None

        license_str = raw.get("license", "UNKNOWN")
        source_aliases = raw.get("aliases")
        if source_aliases is not None and (
            not isinstance(source_aliases, dict)
            or not all(
                isinstance(old, str) and isinstance(new, str)
                for old, new in source_aliases.items()
            )
        ):
            raise ValueError(
                f"sources[{idx}].aliases must be a string-to-string mapping."
            )
        skills_raw = raw.get("skills") or []
        if not isinstance(skills_raw, list) or not skills_raw:
            raise ValueError(
                f"sources[{idx}] ({name!r}) must list at least one skill under "
                "`skills:`."
            )

        skills: list[SkillSpec] = []
        for sk_idx, sk in enumerate(skills_raw):
            if isinstance(sk, str):
                skills.append(SkillSpec(folder=sk))
            elif isinstance(sk, dict) and "name" in sk:
                aliases = sk.get("aliases")
                if aliases is not None and (
                    not isinstance(aliases, dict)
                    or not all(
                        isinstance(old, str) and isinstance(new, str)
                        for old, new in aliases.items()
                    )
                ):
                    raise ValueError(
                        f"sources[{idx}].skills[{sk_idx}].aliases must be a "
                        "string-to-string mapping."
                    )
                skills.append(
                    SkillSpec(
                        folder=sk["name"],
                        local_name=sk.get("as"),
                        marketplace_description_override=sk.get(
                            "marketplace_description"
                        ),
                        aliases=aliases,
                    )
                )
            else:
                raise ValueError(
                    f"sources[{idx}].skills[{sk_idx}] must be a string or a "
                    "mapping with at least a `name` key."
                )

        sources.append(
            Source(
                name=name,
                repo=repo,
                ref=ref,
                path=path.strip("/"),
                license=license_str,
                skills=skills,
                resolve_wrappers=bool(raw.get("resolve_wrappers", False)),
                aliases=source_aliases,
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


def shallow_clone(repo: str, ref: str, sub_path: str, dest: Path) -> str:
    """Sparse + shallow clone `repo` at `ref`, restricted to `sub_path`.

    Returns the resolved commit SHA. Sparse-checkout avoids pulling the
    whole repo when only one sub-tree is needed (the AMD-AGI/Apex tree is
    large; we only want `tools/skills`).
    """
    url = f"https://github.com/{repo}.git"
    run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--sparse",
            "--no-checkout",
            url,
            str(dest),
        ]
    )
    run(["git", "sparse-checkout", "set", "--cone", sub_path], cwd=dest)
    # `git checkout <ref>` resolves branches, tags, and full commit SHAs.
    run(["git", "checkout", ref], cwd=dest)
    return run(["git", "rev-parse", "HEAD"], cwd=dest)


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
    # Regex examples such as `[A-Za-z](\w+)` are not Markdown links even
    # though they match the lightweight link parser above.
    if "\\" in t:
        return True
    if t[0] in "#/":
        return True
    return bool(URI_SCHEME_RE.match(t))


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
    for md_path in sorted(skill_dir.rglob("*.md")):
        original = md_path.read_text(encoding="utf-8")
        local_rel = md_path.relative_to(skill_dir).as_posix()
        repo_file_path = posixpath.join(repo_skill_path, local_rel)
        updated, rewrites = rewrite_external_references_in_text(
            original,
            repo_file_path=repo_file_path,
            copied_repo_root=repo_skill_path,
            repo_files=repo_files,
            repo=repo,
            commit=commit,
        )
        if updated != original:
            md_path.write_text(updated, encoding="utf-8")
            rel = md_path.relative_to(skill_dir.parent).as_posix()
            for old, new in rewrites:
                log.append(f"    [{rel}] {old} -> {new}")


def rewrite_external_references_in_text(
    text: str,
    *,
    repo_file_path: str,
    copied_repo_root: str | None,
    repo_files: set[str],
    repo: str,
    commit: str,
) -> tuple[str, list[tuple[str, str]]]:
    """Rewrite markdown links in one file using its original repo location."""
    rewrites: list[tuple[str, str]] = []
    repo_file_dir = posixpath.dirname(repo_file_path)
    copied_repo_root = copied_repo_root.strip("/") if copied_repo_root else None

    def _sub(match: re.Match[str]) -> str:
        target = match.group("target")
        if _should_skip_target(target):
            return match.group(0)
        path_part, sep, anchor = target.partition("#")
        frag = sep + anchor if sep else ""
        if not path_part:
            return match.group(0)

        # Resolve as Markdown does first, then try repo-root-relative because
        # many product skills intentionally cite root paths such as docs/foo.md.
        file_rel = posixpath.normpath(posixpath.join(repo_file_dir, path_part))
        root_rel = posixpath.normpath(path_part)

        if copied_repo_root:
            within_copy = file_rel == copied_repo_root or file_rel.startswith(
                copied_repo_root + "/"
            )
            if within_copy and file_rel in repo_files:
                return match.group(0)

        chosen = file_rel if file_rel in repo_files else root_rel
        if chosen.startswith(("..", "/")):
            # Some upstream skills were written for a repository-specific
            # loader and contain one too few `..` components. When the
            # intended repository file is still unambiguous, recover it by
            # matching the non-parent suffix. This keeps the standalone copy
            # useful while pinning the correction to the imported commit.
            suffix = posixpath.normpath(path_part)
            while suffix.startswith("../"):
                suffix = suffix[3:]
            suffix_matches = sorted(
                candidate
                for candidate in repo_files
                if suffix and (candidate == suffix or candidate.endswith(f"/{suffix}"))
            )
            if len(suffix_matches) == 1:
                chosen = suffix_matches[0]
        if chosen.startswith(("..", "/")):
            return match.group(0)

        url = f"https://github.com/{repo}/blob/{commit}/{chosen}{frag}"
        rewrites.append((target, url))
        return f"{match.group('prefix')}{url}{match.group('suffix')}"

    return MARKDOWN_LINK_RE.sub(_sub, text), rewrites


def parse_frontmatter(text: str) -> dict:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}
    try:
        data = yaml.safe_load(match.group("frontmatter"))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _frontmatter_and_body(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"{path} has no valid YAML frontmatter block.")
    frontmatter = parse_frontmatter(text)
    if not frontmatter:
        raise ValueError(f"{path} has invalid or empty YAML frontmatter.")
    return frontmatter, match.group("body")


def _repo_relative_path(clone_dir: Path, raw_path: str) -> tuple[Path, str]:
    """Resolve and confine a wrapper target to the cloned repository."""
    normalized = posixpath.normpath(raw_path.strip().lstrip("/"))
    if normalized.startswith("../") or normalized in {"", ".", ".."}:
        raise ValueError(f"Unsafe wrapper target path: {raw_path!r}")
    target = (clone_dir / normalized).resolve()
    try:
        target.relative_to(clone_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Wrapper target escapes repository: {raw_path!r}") from exc
    return target, normalized


def resolve_source_skills(
    source: Source,
    clone_dir: Path,
    repo_files: set[str],
) -> list[ResolvedSkill]:
    """Resolve configured public folders to the directories that get copied."""
    src_root = clone_dir / source.path
    resolved: list[ResolvedSkill] = []
    for spec in source.skills:
        public_dir = src_root / spec.folder
        if not public_dir.is_dir():
            raise FileNotFoundError(
                f"Skill {spec.folder!r} not found under "
                f"{source.repo}/{source.path}@{source.ref}."
            )
        public_md = public_dir / "SKILL.md"
        if not public_md.exists():
            raise FileNotFoundError(
                f"Skill {spec.folder!r} from {source.repo} has no SKILL.md."
            )
        public_frontmatter, public_body = _frontmatter_and_body(public_md)
        public_path = f"{source.path}/{spec.folder}".strip("/")
        implementation_dir = public_dir
        implementation_path = public_path

        if source.resolve_wrappers:
            pointer = SKILL_POINTER_RE.fullmatch(public_body)
            if pointer is None:
                raise ValueError(
                    f"Skill {spec.folder!r} from {source.repo} must contain only "
                    "a supported `Read and follow .../SKILL.md` wrapper body when "
                    "`resolve_wrappers` is enabled."
                )
            implementation_md, implementation_md_path = _repo_relative_path(
                clone_dir, pointer.group("path")
            )
            if (
                implementation_md_path not in repo_files
                or not implementation_md.is_file()
            ):
                raise FileNotFoundError(
                    f"Wrapper target {implementation_md_path!r} for "
                    f"{spec.folder!r} does not exist in {source.repo}@{source.ref}."
                )
            implementation_dir = implementation_md.parent
            implementation_path = posixpath.dirname(implementation_md_path)

        implementation_frontmatter, _ = _frontmatter_and_body(
            implementation_dir / "SKILL.md"
        )
        implementation_name = implementation_frontmatter.get("name")
        if not isinstance(implementation_name, str) or not implementation_name:
            raise ValueError(
                f"Implementation for {spec.folder!r} has no non-empty `name`."
            )
        resolved.append(
            ResolvedSkill(
                spec=spec,
                public_dir=public_dir,
                implementation_dir=implementation_dir,
                public_path=public_path,
                implementation_path=implementation_path,
                public_frontmatter=public_frontmatter,
                implementation_name=implementation_name,
            )
        )
    return resolved


def index_implementation_skills(source_root: Path) -> dict[str, Path]:
    """Index non-wrapper skill implementations by frontmatter name."""
    index: dict[str, Path] = {}
    for skill_md in sorted(source_root.rglob("SKILL.md")):
        try:
            frontmatter, body = _frontmatter_and_body(skill_md)
        except ValueError:
            continue
        if SKILL_POINTER_RE.fullmatch(body):
            continue
        name = frontmatter.get("name")
        if not isinstance(name, str) or not name:
            continue
        previous = index.get(name)
        if previous is not None and previous != skill_md:
            raise ValueError(
                f"More than one implementation named {name!r}: "
                f"{previous} and {skill_md}"
            )
        index[name] = skill_md
    return index


def _replace_skill_aliases(text: str, aliases: dict[str, str]) -> str:
    for upstream_name, public_name in sorted(
        aliases.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if upstream_name != public_name:
            text = re.sub(
                rf"(?<![a-z0-9-]){re.escape(upstream_name)}(?![a-z0-9-])",
                public_name,
                text,
            )
    return text


def _mentioned_implementation_skills(
    text: str, implementation_index: dict[str, Path]
) -> set[str]:
    return {
        name
        for name in implementation_index
        if re.search(rf"(?<![a-z0-9-]){re.escape(name)}(?![a-z0-9-])", text)
    }


def _rewrite_standalone_runtime_paths(
    text: str, implementation_path: str | None = None
) -> str:
    text = SKILL_DIR_ASSIGNMENT_RE.sub(
        lambda match: (
            f"{match.group('indent')}# Set SKILL_DIR to the directory containing "
            "this installed SKILL.md."
        ),
        text,
    )
    if implementation_path:
        text = text.replace(
            f"`{implementation_path}/`", "the installed skill directory"
        )
        text = text.replace(f"`{implementation_path}`", "the installed skill directory")
    return text.replace(
        "Resolve `SKILL_DIR` from the repo root",
        "Resolve `SKILL_DIR` from the loaded skill location",
    )


def bundle_supporting_skills(
    skill_dir: Path,
    *,
    root_implementation_name: str,
    implementation_index: dict[str, Path],
    public_aliases: dict[str, str],
    clone_dir: Path,
    repo_files: set[str],
    repo: str,
    commit: str,
    log: list[str],
) -> list[str]:
    """Bundle non-public helper procedures referenced by a public skill."""
    root_parts: list[str] = []
    for path in sorted(skill_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        if path == skill_dir / "SKILL.md":
            match = FRONTMATTER_RE.match(text)
            text = match.group("body") if match else text
        root_parts.append(text)
    root_text = "\n".join(root_parts)
    public_implementations = set(public_aliases)
    pending = sorted(
        _mentioned_implementation_skills(root_text, implementation_index)
        - public_implementations
        - {root_implementation_name}
    )
    bundled: dict[str, str] = {}

    while pending:
        name = pending.pop(0)
        if name in bundled:
            continue
        source_md = implementation_index[name]
        text = source_md.read_text(encoding="utf-8")
        text = _replace_skill_aliases(text, public_aliases)
        repo_file_path = source_md.relative_to(clone_dir).as_posix()
        text = _rewrite_standalone_runtime_paths(
            text, posixpath.dirname(repo_file_path)
        )
        text, rewrites = rewrite_external_references_in_text(
            text,
            repo_file_path=repo_file_path,
            copied_repo_root=None,
            repo_files=repo_files,
            repo=repo,
            commit=commit,
        )
        bundled[name] = text
        for old, new in rewrites:
            log.append(f"    [references/{name}.md] {old} -> {new}")

        dependencies = (
            _mentioned_implementation_skills(text, implementation_index)
            - public_implementations
            - {root_implementation_name, name}
            - set(bundled)
        )
        pending = sorted(set(pending) | dependencies)

    if not bundled:
        return []
    references = skill_dir / "references"
    references.mkdir(exist_ok=True)
    for name, text in sorted(bundled.items()):
        (references / f"{name}.md").write_text(text, encoding="utf-8")
        log.append(f"    bundled internal helper {name}")
    return sorted(bundled)


def rewrite_frontmatter(
    skill_dir: Path,
    *,
    name: str,
    description: str,
    log: list[str],
) -> None:
    """Apply the public wrapper's routing metadata to a copied implementation."""
    skill_md = skill_dir / "SKILL.md"
    text = skill_md.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"Copied implementation {skill_md} has invalid frontmatter.")
    description_lines = textwrap.wrap(
        " ".join(description.split()),
        width=96,
        break_long_words=False,
        break_on_hyphens=False,
    )
    serialized = "\n".join(
        [
            f"name: {name}",
            "description: >-",
            *[f"  {line}" for line in description_lines],
        ]
    )
    skill_md.write_text(
        f"---\n{serialized}\n---\n\n{match.group('body').lstrip()}",
        encoding="utf-8",
    )
    log.append(f"    [SKILL.md] public routing metadata -> {name}")


def add_catalog_portability_section(
    skill_dir: Path,
    *,
    source: Source,
    commit: str,
    bundled_helpers: list[str],
) -> None:
    """Explain source-path and bundled-helper behavior in standalone installs."""
    skill_md = skill_dir / "SKILL.md"
    text = skill_md.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    if not match:
        return
    body = match.group("body")
    section = [
        "## Catalog portability",
        "",
        (
            f"This standalone skill is federated from `{source.repo}` at commit `{commit}`. "
            "Resolve bundled files relative to this `SKILL.md`. Repository-relative paths "
            f"such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source]"
            f"(https://github.com/{source.repo}/tree/{commit}); use a local Quark checkout "
            "when available, otherwise consult that pinned source."
        ),
    ]
    if bundled_helpers:
        section.extend(
            [
                "",
                (
                    "When the workflow names one of these internal procedures, read its "
                    "bundled reference before carrying out that step:"
                ),
                "",
                *[f"- [`{name}`](references/{name}.md)" for name in bundled_helpers],
            ]
        )
    section_text = "\n".join(section) + "\n\n"
    first_heading = re.search(r"(?m)^# .+\n", body)
    if first_heading:
        insert_at = first_heading.end()
        body = body[:insert_at] + "\n" + section_text + body[insert_at:].lstrip("\n")
    else:
        body = section_text + body
    skill_md.write_text(
        text[: match.start("body")] + body,
        encoding="utf-8",
    )


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


def find_federated_skills() -> dict[str, dict]:
    """Return {skill_folder_name: parsed marker JSON} for every existing
    skill that has a `.federated.json` marker."""
    found: dict[str, dict] = {}
    if not SKILLS_DIR.exists():
        return found
    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir() or skill_dir.name.startswith("."):
            continue
        marker = skill_dir / MARKER_FILENAME
        if marker.exists():
            try:
                found[skill_dir.name] = json.loads(marker.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # Treat a corrupt marker as "managed by this script" so the
                # next run will overwrite or remove it cleanly.
                found[skill_dir.name] = {}
    return found


def copy_skill(src: Path, dest: Path, log: list[str] | None = None) -> None:
    with tempfile.TemporaryDirectory(prefix="amd-skills-preserve-") as tmpdir:
        preserved: dict[str, Path] = {}
        if dest.exists():
            for subdir in PRESERVE_IF_ABSENT_UPSTREAM:
                local_subdir = dest / subdir
                if local_subdir.is_dir() and not (src / subdir).exists():
                    preserved_path = Path(tmpdir) / subdir
                    shutil.copytree(local_subdir, preserved_path)
                    preserved[subdir] = preserved_path
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        for subdir, preserved_path in preserved.items():
            shutil.copytree(preserved_path, dest / subdir)
            if log is not None:
                log.append(f"    preserved local {subdir}/ (absent in upstream import)")


def write_marker(
    skill_dir: Path,
    source: Source,
    commit: str,
    relative_path: str,
    implementation_path: str | None = None,
) -> None:
    marker = {
        "source": source.name,
        "repo": source.repo,
        "ref": source.ref,
        "commit": commit,
        "path": relative_path,
        "license": source.license,
        "imported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if implementation_path and implementation_path != relative_path:
        marker["implementation_path"] = implementation_path
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
    license_text = (
        source.license or f"See [{source.repo}](https://github.com/{source.repo})"
    )
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
    skill_md.write_text(
        text[:fm_start] + new_frontmatter + text[fm_end:], encoding="utf-8"
    )
    log.append(f"    [SKILL.md] name -> {new_name}")


def update_publish_list(
    imported: Iterable[str],
    removed: Iterable[str],
    dry_run: bool,
) -> bool:
    """Sync the bundle's `skills` array in `.claude-plugin/marketplace.json`.

    AMD ships a single curated plugin whose `skills` array lists the published
    skills as `./skills/<name>` paths. Newly imported federated skills are added
    so they ship in the bundle, and skills that were pruned from
    `.github/scripts/sources.yml` are removed. The existing curation order is
    preserved; freshly added skills are appended in sorted order for a
    deterministic diff.

    Returns True when the file was modified (or would be in a dry run).
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

    removed_paths = {f"{SKILLS_PATH_PREFIX}{name}" for name in removed}
    kept = [s for s in skills if s not in removed_paths]
    present = {
        s[len(SKILLS_PATH_PREFIX) :].strip("/")
        for s in kept
        if isinstance(s, str) and s.startswith(SKILLS_PATH_PREFIX)
    }
    additions = sorted(
        f"{SKILLS_PATH_PREFIX}{name}" for name in imported if name not in present
    )
    new_skills = kept + additions

    changed = new_skills != skills
    if changed and not dry_run:
        entry["skills"] = new_skills
        CLAUDE_MARKETPLACE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return changed


def import_source(
    source: Source,
    dry_run: bool,
    log: list[str],
) -> list[ImportResult]:
    results: list[ImportResult] = []
    with tempfile.TemporaryDirectory(prefix="amd-skills-import-") as tmpdir:
        tmp_path = Path(tmpdir) / source.name
        log.append(f"[{source.name}] cloning {source.repo}@{source.ref}")
        commit = shallow_clone(source.repo, source.ref, source.path, tmp_path)
        log.append(f"[{source.name}] resolved to commit {commit}")
        repo_files = list_repo_files(tmp_path, commit)

        src_root = tmp_path / source.path
        if not src_root.is_dir():
            raise FileNotFoundError(
                f"Path {source.path!r} not found in {source.repo}@{source.ref}."
            )

        resolved_skills = resolve_source_skills(source, tmp_path, repo_files)
        implementation_index = (
            index_implementation_skills(src_root) if source.resolve_wrappers else {}
        )
        public_aliases = {
            resolved.implementation_name: resolved.spec.dest_name
            for resolved in resolved_skills
        }
        selected_names = {spec.dest_name for spec in source.import_skills}

        for resolved in resolved_skills:
            spec = resolved.spec
            if spec.dest_name not in selected_names:
                continue
            skill_aliases = dict(public_aliases)
            skill_aliases.update(source.aliases or {})
            skill_aliases.update(spec.aliases or {})
            description = resolved.public_frontmatter.get("description") or ""
            if not isinstance(description, str) or not description.strip():
                raise ValueError(
                    f"Skill {spec.folder!r} from {source.repo} has no "
                    "non-empty `description` in its SKILL.md frontmatter."
                )
            description = _replace_skill_aliases(description, skill_aliases)
            marketplace_description = (
                spec.marketplace_description_override
                or truncate_description(description)
            )

            dest_name = spec.dest_name
            dest_skill = SKILLS_DIR / dest_name
            action = "would import" if dry_run else "importing"
            renamed = f" (as {dest_name})" if dest_name != spec.folder else ""
            log.append(
                f"[{source.name}] {action} {spec.folder} -> skills/{dest_name}{renamed}"
            )
            if not dry_run:
                copy_skill(resolved.implementation_dir, dest_skill, log)

                if source.resolve_wrappers:
                    text_suffixes = {
                        ".md",
                        ".py",
                        ".json",
                        ".yaml",
                        ".yml",
                        ".rst",
                        ".txt",
                    }
                    for text_path in sorted(dest_skill.rglob("*")):
                        if (
                            not text_path.is_file()
                            or text_path.suffix not in text_suffixes
                        ):
                            continue
                        text = text_path.read_text(encoding="utf-8")
                        text = _replace_skill_aliases(text, skill_aliases)
                        if text_path.suffix == ".md":
                            text = _rewrite_standalone_runtime_paths(
                                text, resolved.implementation_path
                            )
                        text_path.write_text(text, encoding="utf-8")

                rewrite_external_references(
                    dest_skill,
                    resolved.implementation_path,
                    repo_files,
                    source.repo,
                    commit,
                    log,
                )
                if source.resolve_wrappers:
                    bundled_helpers = bundle_supporting_skills(
                        dest_skill,
                        root_implementation_name=resolved.implementation_name,
                        implementation_index=implementation_index,
                        public_aliases=skill_aliases,
                        clone_dir=tmp_path,
                        repo_files=repo_files,
                        repo=source.repo,
                        commit=commit,
                        log=log,
                    )
                    rewrite_frontmatter(
                        dest_skill,
                        name=dest_name,
                        description=description.strip(),
                        log=log,
                    )
                    add_catalog_portability_section(
                        dest_skill,
                        source=source,
                        commit=commit,
                        bundled_helpers=bundled_helpers,
                    )
                else:
                    rewrite_skill_name(dest_skill, dest_name, log)

                write_marker(
                    dest_skill,
                    source,
                    commit,
                    resolved.public_path,
                    resolved.implementation_path,
                )
                write_card(dest_skill, source, marketplace_description)

            results.append(
                ImportResult(
                    source=source,
                    folder=dest_name,
                    commit=commit,
                    skill_description=description.strip(),
                    marketplace_description=marketplace_description,
                )
            )
    return results


def prune_orphans(
    declared: set[str],
    existing: dict[str, dict],
    dry_run: bool,
    log: list[str],
) -> list[str]:
    removed: list[str] = []
    for name, marker in existing.items():
        if name in declared:
            continue
        log.append(
            f"[orphan] removing skills/{name} (previously imported from "
            f"{marker.get('repo', '?')}@{marker.get('ref', '?')})"
        )
        if not dry_run:
            shutil.rmtree(SKILLS_DIR / name)
        removed.append(name)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and report the planned changes without writing them.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=CATALOG_FILE,
        help=f"Path to the catalog file (default: {CATALOG_FILE}).",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="SKILL",
        help=(
            "Import only the named skill folder (repeatable). When set, "
            "skills not named here are left untouched and pruning is "
            "restricted to the named skills, so other federated skills are "
            "never removed."
        ),
    )
    args = parser.parse_args(argv)

    sources = parse_sources(args.catalog)

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
            source.selected_skills = [s for s in source.skills if s.dest_name in only]
        sources = [source for source in sources if source.import_skills]
    log: list[str] = []
    declared: set[str] = set()
    all_results: list[ImportResult] = []

    SKILLS_DIR.mkdir(exist_ok=True)
    existing_federated = find_federated_skills()

    for source in sources:
        for spec in source.import_skills:
            if spec.dest_name in declared:
                raise ValueError(
                    f"Skill name collision: {spec.dest_name!r} is listed by "
                    "more than one source in .github/scripts/sources.yml."
                )
            declared.add(spec.dest_name)
        all_results.extend(import_source(source, args.dry_run, log))

    # With --only we deliberately ignore skills the user didn't name, so
    # restrict orphan pruning to just those skills. Otherwise every other
    # federated skill would look like an orphan and be deleted.
    prunable = (
        {name: marker for name, marker in existing_federated.items() if name in only}
        if only
        else existing_federated
    )
    pruned = prune_orphans(declared, prunable, args.dry_run, log)
    imported_names = {result.folder for result in all_results}
    publish_changed = update_publish_list(imported_names, pruned, args.dry_run)

    for line in log:
        print(line)

    print()
    print(f"Imported: {len(all_results)} skill(s)")
    print(f"Removed orphans: {len(pruned)}")
    print(
        "Publish list: "
        f"{'changed' if publish_changed else 'unchanged'}"
        f"{' (dry run)' if args.dry_run and publish_changed else ''}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
