#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0"]
# ///
"""Import skills from external repositories listed in `.github/scripts/sources.yml`.

For each source, the script:

1. Shallow-clones the repo at the pinned `ref` into a temp directory,
   using sparse-checkout so only the configured `path` is fetched.
2. Copies each named skill folder into `skills/<skill>/`.
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
6. Updates `.claude-plugin/marketplace.json` with an entry per imported
   skill (using the SKILL.md `description` as the marketplace blurb,
   unless the source declares an override).
7. Removes any previously imported skill (one with a `.federated.json`)
   that is no longer listed in `.github/scripts/sources.yml`.

Usage:
    uv run .github/scripts/import_external_skills.py            # write changes
    uv run .github/scripts/import_external_skills.py --dry-run  # report only

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
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CATALOG_FILE = Path(__file__).resolve().parent / "sources.yml"
SKILLS_DIR = REPO_ROOT / "skills"
CLAUDE_MARKETPLACE = REPO_ROOT / ".claude-plugin" / "marketplace.json"
MARKER_FILENAME = ".federated.json"
CARD_FILENAME = "skill-card.md"

FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<frontmatter>.*?)\n---\s*\n?(?P<body>.*)\Z",
    re.DOTALL,
)
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


@dataclass
class SkillSpec:
    folder: str
    marketplace_description_override: str | None = None


@dataclass
class Source:
    name: str
    repo: str
    ref: str
    path: str
    license: str
    skills: list[SkillSpec]


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
            raise ValueError(f"sources[{idx}] must be a mapping.")
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
                skills.append(
                    SkillSpec(
                        folder=sk["name"],
                        marketplace_description_override=sk.get(
                            "marketplace_description"
                        ),
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


def copy_skill(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def write_marker(
    skill_dir: Path,
    source: Source,
    commit: str,
    relative_path: str,
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


def update_marketplace(results: Iterable[ImportResult], dry_run: bool) -> bool:
    """Sync `.claude-plugin/marketplace.json` with the imported skills.

    Returns True when the file was modified (or would be modified in a dry
    run).
    """
    data = json.loads(CLAUDE_MARKETPLACE.read_text(encoding="utf-8"))
    plugins = data.setdefault("plugins", [])
    by_name = {p.get("name"): p for p in plugins if isinstance(p, dict)}

    changed = False
    for result in results:
        name = result.folder
        entry = by_name.get(name)
        expected = {
            "name": name,
            "source": f"./skills/{name}",
            "description": result.marketplace_description,
        }
        if entry is None:
            plugins.append(expected)
            by_name[name] = expected
            changed = True
            continue
        # Drop the legacy `skills` key: each skill directory ships its
        # SKILL.md at the plugin root, so a single-skill plugin auto-loads
        # without an explicit skills path.
        if "skills" in entry:
            del entry["skills"]
            changed = True
        for key, value in expected.items():
            if entry.get(key) != value:
                entry[key] = value
                changed = True

    # Drop entries that point at skills that no longer exist on disk so
    # the importer also cleans up the marketplace when an entry is
    # removed from `.github/scripts/sources.yml`.
    existing_dirs = {p.name for p in SKILLS_DIR.iterdir() if p.is_dir()}
    pruned = [p for p in plugins if not isinstance(p, dict) or p.get("name") in existing_dirs]
    if len(pruned) != len(plugins):
        plugins[:] = pruned
        changed = True

    plugins.sort(key=lambda p: p.get("name", ""))

    if changed and not dry_run:
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

        for spec in source.skills:
            src_skill = src_root / spec.folder
            if not src_skill.is_dir():
                raise FileNotFoundError(
                    f"Skill {spec.folder!r} not found under "
                    f"{source.repo}/{source.path}@{source.ref}."
                )
            skill_md = src_skill / "SKILL.md"
            if not skill_md.exists():
                raise FileNotFoundError(
                    f"Skill {spec.folder!r} from {source.repo} has no SKILL.md."
                )
            frontmatter = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            description = frontmatter.get("description") or ""
            if not isinstance(description, str) or not description.strip():
                raise ValueError(
                    f"Skill {spec.folder!r} from {source.repo} has no "
                    "non-empty `description` in its SKILL.md frontmatter."
                )
            marketplace_description = (
                spec.marketplace_description_override
                or truncate_description(description)
            )

            dest_skill = SKILLS_DIR / spec.folder
            relative_path = f"{source.path}/{spec.folder}"
            action = "would import" if dry_run else "importing"
            log.append(f"[{source.name}] {action} {spec.folder} -> skills/{spec.folder}")
            if not dry_run:
                copy_skill(src_skill, dest_skill)
                write_marker(dest_skill, source, commit, relative_path)
                write_card(dest_skill, source, marketplace_description)
                rewrite_external_references(
                    dest_skill,
                    relative_path,
                    repo_files,
                    source.repo,
                    commit,
                    log,
                )

            results.append(
                ImportResult(
                    source=source,
                    folder=spec.folder,
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
) -> int:
    removed = 0
    for name, marker in existing.items():
        if name in declared:
            continue
        log.append(
            f"[orphan] removing skills/{name} (previously imported from "
            f"{marker.get('repo', '?')}@{marker.get('ref', '?')})"
        )
        if not dry_run:
            shutil.rmtree(SKILLS_DIR / name)
        removed += 1
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
    args = parser.parse_args(argv)

    sources = parse_sources(args.catalog)
    log: list[str] = []
    declared: set[str] = set()
    all_results: list[ImportResult] = []

    SKILLS_DIR.mkdir(exist_ok=True)
    existing_federated = find_federated_skills()

    for source in sources:
        for spec in source.skills:
            if spec.folder in declared:
                raise ValueError(
                    f"Skill name collision: {spec.folder!r} is listed by "
                    "more than one source in .github/scripts/sources.yml."
                )
            declared.add(spec.folder)
        all_results.extend(import_source(source, args.dry_run, log))

    pruned = prune_orphans(declared, existing_federated, args.dry_run, log)
    marketplace_changed = update_marketplace(all_results, args.dry_run)

    for line in log:
        print(line)

    print("")
    print(f"Imported: {len(all_results)} skill(s)")
    print(f"Removed orphans: {pruned}")
    print(
        "Marketplace: "
        f"{'changed' if marketplace_changed else 'unchanged'}"
        f"{' (dry run)' if args.dry_run and marketplace_changed else ''}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
