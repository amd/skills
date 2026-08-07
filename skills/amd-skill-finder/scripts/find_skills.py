#!/usr/bin/env python3
"""Find AMD skills and route AMD/ROCm questions to curated source projects.

The default path is read-only and deterministic: inspect installed skills, read
the current AMD catalog when available, and rank a bundled curated registry.
Pass --live to search code in the top routed GitHub repositories. Arbitrary
GitHub repository search is disabled unless --general-github is also supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SKILL_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SKILL_DIR / "data"
REGISTRY_PATH = DATA_DIR / "sources.json"
SNAPSHOT_PATH = DATA_DIR / "catalog-snapshot.json"
CATALOG_REPO = "amd/skills"
SELF_SKILL_NAME = "amd-skill-finder"
CATALOG_MANIFEST_URL = (
    "https://raw.githubusercontent.com/amd/skills/main/.claude-plugin/marketplace.json"
)
RAW_SKILL_URL = (
    "https://raw.githubusercontent.com/amd/skills/main/skills/{name}/SKILL.md"
)
SECRET_PATTERNS = (
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}", re.IGNORECASE),
)
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "can",
    "do",
    "for",
    "from",
    "help",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "skill",
    "skills",
    "the",
    "this",
    "to",
    "use",
    "using",
    "want",
    "with",
}
TIER_LABELS = {
    "amd-official": "AMD official",
    "reviewed-upstream": "reviewed upstream",
    "user-specified": "user-specified",
}


class FinderError(RuntimeError):
    """Expected user-facing finder failure."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise FinderError(f"Expected a JSON object in {path}")
    return data


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    registry = _read_json(path)
    if registry.get("schema_version") != 1:
        raise FinderError(f"Unsupported registry schema in {path}")
    if not isinstance(registry.get("projects"), list):
        raise FinderError(f"Registry {path} is missing a projects array")
    return registry


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9+#.]+", text.lower()))


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def tokens(text: str) -> set[str]:
    return {
        _stem(token)
        for token in normalize(text).split()
        if len(token) >= 2 and token not in STOP_WORDS
    }


def phrase_present(phrase: str, normalized_query: str) -> bool:
    phrase = normalize(phrase)
    return bool(phrase) and f" {phrase} " in f" {normalized_query} "


def reject_secrets(query: str) -> None:
    if any(pattern.search(query) for pattern in SECRET_PATTERNS):
        raise FinderError(
            "The query appears to contain a credential or private key. Remove "
            "secrets before sending search terms to a catalog or GitHub."
        )


def _parse_frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"\A---\r?\n(.*?)\r?\n---", text, re.DOTALL)
    if not match:
        return {}
    lines = match.group(1).splitlines()
    values: dict[str, str] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        field = re.match(r"^(name|description):\s*(.*)$", line)
        if not field:
            index += 1
            continue
        key, raw = field.groups()
        if raw in {">", ">-", "|", "|-"}:
            folded: list[str] = []
            index += 1
            while index < len(lines) and (
                not lines[index].strip() or lines[index].startswith((" ", "\t"))
            ):
                if lines[index].strip():
                    folded.append(lines[index].strip())
                index += 1
            values[key] = " ".join(folded)
            continue
        values[key] = raw.strip().strip("'\"")
        index += 1
    return values


def _catalog_from_directory(
    skills_dir: Path, names: Iterable[str] | None = None
) -> list[dict[str, str]]:
    if not skills_dir.is_dir():
        return []
    selected = set(names) if names is not None else None
    entries: list[dict[str, str]] = []
    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        name = skill_md.parent.name
        if selected is not None and name not in selected:
            continue
        metadata = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        if metadata.get("name") and metadata.get("description"):
            entries.append(
                {
                    "name": metadata["name"],
                    "description": metadata["description"],
                    "origin": "local catalog checkout",
                    "url": f"https://github.com/amd/skills/tree/main/skills/{name}",
                }
            )
    return entries


def _repo_root() -> Path | None:
    candidate = SKILL_DIR.parents[1]
    if (candidate / ".claude-plugin" / "marketplace.json").is_file():
        return candidate
    return None


def local_catalog() -> list[dict[str, str]]:
    root = _repo_root()
    if root is None:
        return []
    manifest_path = root / ".claude-plugin" / "marketplace.json"
    try:
        manifest = _read_json(manifest_path)
        plugins = manifest.get("plugins", [])
        paths = plugins[0].get("skills", []) if plugins else []
        names = [str(path).rstrip("/").split("/")[-1] for path in paths]
    except (OSError, ValueError, IndexError, AttributeError):
        names = None
    return _catalog_from_directory(root / "skills", names)


def _fetch_text(url: str, timeout: int = 8) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "amd-skill-finder/1.0", "Accept": "text/plain"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def remote_catalog() -> list[dict[str, str]]:
    manifest = json.loads(_fetch_text(CATALOG_MANIFEST_URL))
    plugins = manifest.get("plugins", [])
    if not plugins:
        raise FinderError("The AMD catalog manifest has no plugin entries")
    names = [
        str(path).rstrip("/").split("/")[-1] for path in plugins[0].get("skills", [])
    ]
    entries: list[dict[str, str]] = []
    for name in names:
        metadata = _parse_frontmatter(_fetch_text(RAW_SKILL_URL.format(name=name)))
        if metadata.get("name") and metadata.get("description"):
            entries.append(
                {
                    "name": metadata["name"],
                    "description": metadata["description"],
                    "origin": "live AMD catalog",
                    "url": f"https://github.com/amd/skills/tree/main/skills/{name}",
                }
            )
    return entries


def snapshot_catalog() -> list[dict[str, str]]:
    snapshot = _read_json(SNAPSHOT_PATH)
    return [
        {
            "name": str(entry["name"]),
            "description": str(entry["description"]),
            "origin": "bundled catalog snapshot",
            "url": f"https://github.com/amd/skills/tree/main/skills/{entry['name']}",
        }
        for entry in snapshot.get("skills", [])
    ]


def installed_roots(extra_roots: Iterable[str] = ()) -> list[Path]:
    roots = [
        Path.cwd() / ".claude" / "skills",
        Path.cwd() / ".agents" / "skills",
        Path.home() / ".claude" / "skills",
        Path.home() / ".codex" / "skills",
    ]
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        roots.append(Path(codex_home) / "skills")
    roots.extend(Path(value).expanduser() for value in extra_roots)
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def installed_skills(extra_roots: Iterable[str] = ()) -> list[dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    for root in installed_roots(extra_roots):
        for entry in _catalog_from_directory(root):
            entry["origin"] = f"installed at {root}"
            entry["url"] = str(root / entry["name"])
            entries.setdefault(entry["name"], entry)
    return sorted(entries.values(), key=lambda entry: entry["name"])


def has_amd_signal(query: str, registry: dict[str, Any]) -> bool:
    normalized_query = normalize(query)
    return any(
        phrase_present(str(signal), normalized_query)
        for signal in registry.get("amd_signals", [])
    )


def score_catalog_entry(query: str, entry: dict[str, str]) -> tuple[int, list[str]]:
    normalized_query = normalize(query)
    query_tokens = tokens(query)
    name = entry["name"]
    document = f"{name} {entry['description']}"
    document_tokens = tokens(document)
    score = 0
    reasons: list[str] = []
    if phrase_present(name, normalized_query):
        score += 35
        reasons.append(f"explicit skill name `{name}`")
    browse_phrases = (
        "browse amd skills",
        "list amd skills",
        "show amd skills",
        "what amd skills",
        "which amd skills",
    )
    if any(phrase_present(phrase, normalized_query) for phrase in browse_phrases):
        score += 10
        reasons.append("AMD catalog browse request")
    overlap = sorted(query_tokens & document_tokens)
    if overlap:
        score += min(24, 4 * len(overlap))
        reasons.append("matched " + ", ".join(overlap[:5]))
    if "skill" in normalize(query).split():
        score += 2
    return score, reasons


def score_project(
    query: str, project: dict[str, Any], amd_query: bool
) -> tuple[int, list[str]]:
    normalized_query = normalize(query)
    query_tokens = tokens(query)
    score = 0
    reasons: list[str] = []
    project_names = [str(project["id"]), str(project["display_name"])]
    for name in project_names:
        if phrase_present(name, normalized_query):
            score += 18
            reasons.append(f"explicit project `{project['display_name']}`")
            break
    name_overlap = query_tokens & tokens(" ".join(project_names))
    if name_overlap:
        score += 4 * len(name_overlap)
    for keyword in project.get("keywords", []):
        keyword = str(keyword)
        if phrase_present(keyword, normalized_query):
            score += 10 if " " in normalize(keyword) else 8
            reasons.append(f"matched `{keyword}`")
            continue
        overlap = query_tokens & tokens(keyword)
        score += min(3, len(overlap))
    for category in project.get("categories", []):
        if phrase_present(str(category).replace("-", " "), normalized_query):
            score += 6
            reasons.append(f"matched {category} domain")
    if (
        score > 0
        and amd_query
        and any(
            repo.get("tier") == "amd-official"
            for repo in project.get("repositories", [])
        )
    ):
        score += 8
    return score, list(dict.fromkeys(reasons))[:4]


def route_projects(
    query: str,
    registry: dict[str, Any],
    scope: str = "curated",
    limit: int = 5,
) -> list[dict[str, Any]]:
    amd_query = has_amd_signal(query, registry)
    excluded = {repo.lower() for repo in registry.get("excluded_repositories", [])}
    routed: list[dict[str, Any]] = []
    for project in registry["projects"]:
        repositories = [
            repo
            for repo in project.get("repositories", [])
            if str(repo.get("repo", "")).lower() not in excluded
        ]
        if scope == "amd" and not any(
            repo.get("tier") == "amd-official" for repo in repositories
        ):
            continue
        score, reasons = score_project(query, project, amd_query)
        if score < 7:
            continue
        repositories.sort(key=lambda repo: int(repo.get("priority", 0)), reverse=True)
        routed.append(
            {
                "type": "source_project",
                "installable": False,
                "id": project["id"],
                "name": project["display_name"],
                "description": project["description"],
                "categories": project.get("categories", []),
                "score": score,
                "why": reasons or ["matched project vocabulary"],
                "repositories": [
                    {
                        **repo,
                        "url": f"https://github.com/{repo['repo']}",
                        "tier_label": TIER_LABELS.get(
                            repo.get("tier"), repo.get("tier")
                        ),
                    }
                    for repo in repositories
                ],
            }
        )
    routed.sort(
        key=lambda item: (
            item["score"],
            max((repo.get("priority", 0) for repo in item["repositories"]), default=0),
            item["name"],
        ),
        reverse=True,
    )
    return routed[:limit]


def _rank_skills(
    query: str, entries: list[dict[str, str]], installed: bool
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for entry in entries:
        score, reasons = score_catalog_entry(query, entry)
        if score < 7:
            continue
        result: dict[str, Any] = {
            "type": "installed_skill" if installed else "installable_skill",
            "installable": not installed,
            "name": entry["name"],
            "description": entry["description"],
            "score": score + (60 if installed else 45),
            "why": reasons,
            "origin": entry["origin"],
            "url": entry["url"],
        }
        if not installed:
            result["install_command"] = (
                f"npx skills add amd/skills --skill {entry['name']} --global --yes"
            )
        ranked.append(result)
    ranked.sort(key=lambda item: (item["score"], item["name"]), reverse=True)
    return ranked


def _deduplicate_catalog(entries: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return list({entry["name"]: entry for entry in entries}.values())


def _gh_json(arguments: list[str], timeout: int = 25) -> Any:
    gh = shutil.which("gh")
    if not gh:
        raise FinderError("Live GitHub search requires the `gh` CLI")
    try:
        process = subprocess.run(
            [gh, *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise FinderError(f"GitHub search timed out after {timeout}s") from exc
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip()
        raise FinderError(f"GitHub search failed: {detail[:400]}")
    return json.loads(process.stdout or "[]")


def _search_terms(query: str) -> list[str]:
    ordered: list[str] = []
    for token in normalize(query).split():
        if token in STOP_WORDS or len(token) < 3:
            continue
        if token not in ordered:
            ordered.append(token)
    ordered.sort(
        key=lambda value: (any(char.isdigit() for char in value), len(value)),
        reverse=True,
    )
    return ordered[:4]


def live_code_search(
    query: str,
    projects: list[dict[str, Any]],
    extra_repositories: Iterable[str] = (),
    repository_limit: int = 5,
    result_limit: int = 2,
) -> tuple[list[dict[str, Any]], list[str]]:
    repositories: list[tuple[str, str, str]] = []
    for project in projects:
        for repo in project["repositories"]:
            repositories.append((repo["repo"], project["id"], repo["tier"]))
    repositories.extend(
        (repo, repo.lower().replace("/", "-"), "user-specified")
        for repo in extra_repositories
    )
    unique_repositories = list(dict.fromkeys(repositories))[:repository_limit]
    terms = _search_terms(query)
    results: list[dict[str, Any]] = []
    warnings: list[str] = []
    for repository, project_id, tier in unique_repositories:
        matches: list[dict[str, Any]] = []
        for term in terms[:2]:
            try:
                payload = _gh_json(
                    [
                        "search",
                        "code",
                        term,
                        "--repo",
                        repository,
                        "--limit",
                        str(result_limit),
                        "--json",
                        "path,repository,textMatches,url",
                    ]
                )
            except (FinderError, json.JSONDecodeError) as exc:
                warnings.append(f"{repository}: {exc}")
                break
            if isinstance(payload, list) and payload:
                matches = payload
                break
        for match in matches[:result_limit]:
            path = str(match.get("path", ""))
            lower_path = path.lower()
            if lower_path.endswith("skill.md"):
                result_type = "embedded_skill"
            elif lower_path.startswith("docs/") or lower_path.endswith(".md"):
                result_type = "guide"
            elif "example" in lower_path:
                result_type = "code_example"
            else:
                result_type = "source_code"
            fragments = match.get("textMatches", []) or []
            fragment = ""
            if fragments and isinstance(fragments[0], dict):
                fragment = " ".join(str(fragments[0].get("fragment", "")).split())[:240]
            results.append(
                {
                    "type": result_type,
                    "installable": False,
                    "project_id": project_id,
                    "repository": repository,
                    "source_tier": tier,
                    "path": path,
                    "url": match.get("url"),
                    "excerpt": fragment,
                }
            )
    return results, list(dict.fromkeys(warnings))


def general_repository_search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    payload = _gh_json(
        [
            "search",
            "repos",
            query,
            "--limit",
            str(limit),
            "--json",
            "fullName,url,description,stargazersCount,updatedAt",
        ]
    )
    return [
        {
            "type": "unreviewed_repository",
            "installable": False,
            "name": entry.get("fullName"),
            "url": entry.get("url"),
            "description": entry.get("description") or "",
            "stars": entry.get("stargazersCount"),
            "updated_at": entry.get("updatedAt"),
            "source_tier": "unreviewed",
        }
        for entry in payload
    ]


def find(
    query: str,
    *,
    scope: str = "curated",
    kind: str = "all",
    limit: int = 5,
    offline: bool = False,
    live: bool = False,
    general_github: bool = False,
    extra_repositories: Iterable[str] = (),
    extra_installed_roots: Iterable[str] = (),
    live_repository_limit: int = 5,
    live_result_limit: int = 2,
) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise FinderError("Search query cannot be empty")
    reject_secrets(query)
    registry = load_registry()
    warnings: list[str] = []

    installed_results: list[dict[str, Any]] = []
    catalog_results: list[dict[str, Any]] = []
    if kind in {"all", "skills"}:
        installed = [
            entry
            for entry in installed_skills(extra_installed_roots)
            if entry["name"] != SELF_SKILL_NAME
        ]
        installed_results = _rank_skills(query, installed, installed=True)[:limit]

        catalog = local_catalog()
        if not catalog and not offline:
            try:
                catalog = remote_catalog()
            except (FinderError, OSError, ValueError, urllib.error.URLError) as exc:
                warnings.append(f"Live AMD catalog unavailable: {exc}")
        if not catalog:
            catalog = snapshot_catalog()
        installed_names = {entry["name"] for entry in installed}
        catalog = [
            entry
            for entry in _deduplicate_catalog(catalog)
            if entry["name"] not in installed_names and entry["name"] != SELF_SKILL_NAME
        ]
        catalog_results = _rank_skills(query, catalog, installed=False)[:limit]

    source_results: list[dict[str, Any]] = []
    if kind in {"all", "sources"} and scope != "catalog":
        source_results = route_projects(query, registry, scope=scope, limit=limit)

    live_results: list[dict[str, Any]] = []
    if live and not offline:
        live_results, live_warnings = live_code_search(
            query,
            source_results,
            extra_repositories=extra_repositories,
            repository_limit=live_repository_limit,
            result_limit=live_result_limit,
        )
        warnings.extend(live_warnings)
    elif extra_repositories:
        warnings.append("Explicit repositories require --live for code search")

    general_results: list[dict[str, Any]] = []
    if general_github:
        if offline or not live:
            warnings.append(
                "General GitHub search requires --live and cannot run offline"
            )
        else:
            try:
                general_results = general_repository_search(query, limit=limit)
            except (FinderError, json.JSONDecodeError) as exc:
                warnings.append(str(exc))

    return {
        "query": query,
        "scope": scope,
        "catalog_repository": CATALOG_REPO,
        "installed_skills": installed_results,
        "catalog_skills": catalog_results,
        "source_projects": source_results,
        "live_matches": live_results,
        "unreviewed_repositories": general_results,
        "warnings": list(dict.fromkeys(warnings)),
    }


def _markdown_link(label: str, url: str | None) -> str:
    return f"[{label}]({url})" if url else label


def render_markdown(result: dict[str, Any]) -> str:
    lines = ["# AMD skill finder results", "", f"Query: `{result['query']}`"]

    sections = (
        ("Installed skills", result["installed_skills"]),
        ("AMD catalog skills", result["catalog_skills"]),
    )
    for heading, entries in sections:
        if not entries:
            continue
        lines.extend(["", f"## {heading}", ""])
        for entry in entries:
            lines.append(f"- **{entry['name']}** — {entry['description']}")
            lines.append(
                f"  Provenance: {entry['origin']}. Why: {'; '.join(entry['why'])}."
            )
            if entry.get("install_command"):
                lines.append(f"  Install after approval: `{entry['install_command']}`")

    if result["source_projects"]:
        lines.extend(["", "## Curated source projects", ""])
        for project in result["source_projects"]:
            lines.append(f"- **{project['name']}** — {project['description']}")
            repo_links = [
                f"{_markdown_link(repo['repo'], repo['url'])} ({repo['tier_label']}, {repo['role']})"
                for repo in project["repositories"]
            ]
            lines.append(f"  Sources: {', '.join(repo_links)}.")
            lines.append(
                f"  Why: {'; '.join(project['why'])}. Result type: source project, not installable."
            )

    if result["live_matches"]:
        lines.extend(["", "## Live repository matches", ""])
        for match in result["live_matches"]:
            label = f"{match['repository']}:{match['path']}"
            lines.append(
                f"- {_markdown_link(label, match.get('url'))} "
                f"— `{match['type']}`, {TIER_LABELS.get(match['source_tier'], match['source_tier'])}"
            )
            if match.get("excerpt"):
                lines.append(f"  {match['excerpt']}")

    if result["unreviewed_repositories"]:
        lines.extend(["", "## Unreviewed GitHub fallback", ""])
        for entry in result["unreviewed_repositories"]:
            lines.append(
                f"- {_markdown_link(str(entry['name']), entry.get('url'))} — "
                f"{entry['description']} (unreviewed; inspect before use)"
            )

    if not any(
        result[key]
        for key in (
            "installed_skills",
            "catalog_skills",
            "source_projects",
            "live_matches",
            "unreviewed_repositories",
        )
    ):
        lines.extend(["", "No strong match was found in the selected scope."])

    if result["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result["warnings"])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="AMD/ROCm task or capability to find")
    parser.add_argument(
        "--scope",
        choices=("catalog", "amd", "curated"),
        default="curated",
        help="catalog only, AMD-owned sources, or all curated sources",
    )
    parser.add_argument(
        "--kind",
        choices=("all", "skills", "sources"),
        default="all",
        help="return skills, source projects, or both",
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="maximum results per primary section"
    )
    parser.add_argument(
        "--offline", action="store_true", help="avoid all network access"
    )
    parser.add_argument(
        "--live", action="store_true", help="search code in routed repositories with gh"
    )
    parser.add_argument(
        "--general-github",
        action="store_true",
        help="also search unreviewed GitHub repositories (requires --live)",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        metavar="OWNER/REPO",
        help="also search an explicit repository; repeat as needed (requires --live)",
    )
    parser.add_argument(
        "--installed-root",
        action="append",
        default=[],
        metavar="PATH",
        help="additional directory containing installed skill folders",
    )
    parser.add_argument(
        "--live-repos",
        type=int,
        default=5,
        help="maximum repositories for live code search",
    )
    parser.add_argument(
        "--live-results",
        type=int,
        default=2,
        help="maximum code matches per repository",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 1 or args.live_repos < 1 or args.live_results < 1:
        print("Limits must be positive integers", file=sys.stderr)
        return 2
    try:
        result = find(
            args.query,
            scope=args.scope,
            kind=args.kind,
            limit=args.limit,
            offline=args.offline,
            live=args.live,
            general_github=args.general_github,
            extra_repositories=args.repo,
            extra_installed_roots=args.installed_root,
            live_repository_limit=args.live_repos,
            live_result_limit=args.live_results,
        )
    except FinderError as exc:
        print(f"amd-skill-finder: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(render_markdown(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
