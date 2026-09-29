#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# ///
"""Install this catalog with the `skills` CLI and check what lands on disk.

Everything else in CI reads the repo; this is the one check that goes through
`npx skills add`, the path users actually take. It catches what only shows up
after the CLI has discovered, cloned, copied, and linked the skills:

  discover     `add --list` finds exactly the skill folders under skills/, and
               groups the ones the plugin manifest publishes apart from the rest.
               Catches a nested SKILL.md (e.g. under evals/) leaking out as a
               skill, a skill the CLI cannot parse, or a duplicate folder.
  install      every skill installs for several agents; each installed folder
               holds exactly the git-tracked files of its source folder, with
               the same contents, and tooling files keep LF endings on every OS.
  reinstall    installing a second time succeeds and leaves the same tree.
  remove       removing one skill clears its folder, its agent links, and its
               lock entry, and leaves the others alone.
  global       `-g` writes under the user's home and nothing into the project.
  update       `update -y` succeeds on a remote install (local-path installs
               are not updatable, so this is skipped for them).

Run from the repo root, e.g.:

    uv run .github/scripts/test_cli_install.py --source .
    uv run .github/scripts/test_cli_install.py --source amd/skills --mode copy
    uv run .github/scripts/test_cli_install.py \\
        --source https://github.com/amd/skills/tree/<sha>/skills/local-ai-use \\
        --expect local-ai-use --scenarios install reinstall remove global update

`--source` is passed to the CLI as-is. The expected files are read from the git
checkout at `--repo-root`, so it must be the same revision the source resolves to.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CLI_VERSION = "1.7.0"

# Agents whose project folders are checked after an install. cursor and codex
# share the universal `.agents/skills` folder, which is also where the CLI keeps
# its canonical copy; claude-code and windsurf each get their own folder.
AGENT_DIRS = {
    "claude-code": ".claude/skills",
    "cursor": ".agents/skills",
    "codex": ".agents/skills",
    "windsurf": ".windsurf/skills",
}
CANONICAL_DIR = ".agents/skills"
GLOBAL_AGENT = "claude-code"
GLOBAL_AGENT_DIR = ".claude/skills"

# .gitattributes pins these to LF because they run on Linux; a CRLF copy breaks
# a shebang or a YAML parser on the user's machine.
LF_ONLY_SUFFIXES = {".sh", ".py", ".yml", ".yaml"}

ALL_SCENARIOS = ["discover", "install", "reinstall", "remove", "global", "update"]
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")
# The CLI draws its tree with `|` on a terminal and `│` when piped.
LIST_NAME = re.compile(r"^[|│]\s{4}([a-z0-9][a-z0-9._-]*)\s*$")
UNGROUPED_LABEL = "General"


class CheckFailed(Exception):
    pass


class Cli:
    """Runs one pinned `skills` CLI with telemetry off and an isolated home."""

    def __init__(self, version: str, work: Path) -> None:
        npx = shutil.which("npx")
        if not npx:
            raise SystemExit("npx not found on PATH; install Node.js first.")
        self.npx = npx
        self.package = f"skills@{version}"
        self.home = work / "home"
        self.home.mkdir(parents=True, exist_ok=True)
        real_home = Path.home()
        self.env = os.environ.copy()
        # The fake home isolates where skills land, but npm and git still need
        # the real proxy and CA settings to reach the registry and GitHub.
        for var, config in (
            ("npm_config_userconfig", real_home / ".npmrc"),
            ("GIT_CONFIG_GLOBAL", real_home / ".gitconfig"),
        ):
            if config.is_file() and var not in self.env:
                self.env[var] = str(config)
        self.env.update(
            {
                # CI installs must never reach the skills.sh install counts.
                "DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1",
                "NO_COLOR": "1",
                "CI": "1",
                "GIT_TERMINAL_PROMPT": "0",
                "HOME": str(self.home),
                "USERPROFILE": str(self.home),
                "npm_config_update_notifier": "false",
            }
        )
        if "npm_config_cache" not in os.environ:
            local = os.environ.get("LOCALAPPDATA")
            cache = Path(local) / "npm-cache" if os.name == "nt" and local else real_home / ".npm"
            self.env["npm_config_cache"] = str(cache)
        if version != "latest":
            self.env["npm_config_prefer_offline"] = "true"

    def run(self, cwd: Path, *args: str, timeout: int = 600) -> str:
        cmd = [self.npx, "-y", self.package, *args]
        print(f"    $ npx {self.package} {' '.join(args)}", flush=True)
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                cwd=cwd,
                env=self.env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise CheckFailed(f"`{' '.join(args)}` timed out after {timeout}s") from exc
        out = ANSI.sub("", (proc.stdout or "") + (proc.stderr or ""))
        if proc.returncode != 0:
            tail = "\n".join(out.strip().splitlines()[-30:])
            raise CheckFailed(
                f"`{' '.join(args)}` exited {proc.returncode}:\n{tail}"
            )
        return out


def tracked_skills(repo_root: Path) -> dict[str, list[str]]:
    """Map each skill folder under skills/ to its git-tracked files."""
    out = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z", "--", "skills"],
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    skills: dict[str, list[str]] = {}
    for path in filter(None, out.split("\0")):
        parts = path.split("/", 2)
        if len(parts) == 3:
            skills.setdefault(parts[1], []).append(parts[2])
    return {
        name: sorted(files)
        for name, files in sorted(skills.items())
        if "SKILL.md" in files
    }


def published_skills(repo_root: Path) -> set[str]:
    manifest = repo_root / ".claude-plugin" / "marketplace.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        path.rstrip("/").rsplit("/", 1)[-1]
        for plugin in data.get("plugins", [])
        for path in plugin.get("skills", [])
    }


def parse_list(output: str) -> dict[str, str]:
    """Map skill name -> group label from `skills add <src> --list` output."""
    groups: dict[str, str] = {}
    group = UNGROUPED_LABEL
    listing = False
    for line in output.splitlines():
        stripped = line.strip()
        if not listing:
            listing = "Available Skills" in stripped
            continue
        match = LIST_NAME.match(line.rstrip())
        if match:
            groups[match.group(1)] = group
        elif stripped[:1].isalnum():
            group = stripped
    return groups


def files_under(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    )


def check_tree(installed: Path, source: Path, expected: list[str]) -> list[str]:
    if not (installed / "SKILL.md").is_file():
        return [f"{installed}: no SKILL.md"]
    errors: list[str] = []
    actual = files_under(installed)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing:
        errors.append(f"{installed}: missing {missing}")
    if extra:
        errors.append(f"{installed}: unexpected {extra}")
    for rel in sorted(set(expected) & set(actual)):
        got = (installed / rel).read_bytes()
        want = (source / rel).read_bytes()
        if got.replace(b"\r\n", b"\n") != want.replace(b"\r\n", b"\n"):
            errors.append(f"{installed / rel}: contents differ from the source")
        elif Path(rel).suffix in LF_ONLY_SUFFIXES and b"\r\n" in got:
            errors.append(f"{installed / rel}: CRLF line endings in a tooling file")
    return errors


def same_dir(a: Path, b: Path) -> bool:
    return os.path.realpath(a) == os.path.realpath(b)


class Suite:
    def __init__(self, args: argparse.Namespace) -> None:
        local = Path(args.source).expanduser()
        self.remote = not local.exists()
        # Each scenario runs in its own scratch project, so a relative path
        # would resolve against that project instead of the caller's cwd.
        self.source = args.source if self.remote else str(local.resolve())
        self.mode = args.mode
        self.agents = args.agent
        self.repo_root = args.repo_root.resolve()
        catalog = tracked_skills(self.repo_root)
        if not catalog:
            raise SystemExit(f"No tracked skills under {self.repo_root / 'skills'}")
        unknown = sorted(set(args.expect) - set(catalog))
        if unknown:
            raise SystemExit(f"--expect names skills not in the checkout: {unknown}")
        self.catalog = catalog
        self.expected = {n: catalog[n] for n in (args.expect or catalog)}
        self.whole_catalog = not args.expect
        self.work = Path(tempfile.mkdtemp(prefix="skills-cli-test-"))
        self.cli = Cli(args.cli_version, self.work)

    def project(self, name: str) -> Path:
        path = self.work / "projects" / name
        path.mkdir(parents=True)
        return path

    def add_args(self, *extra: str) -> list[str]:
        args = ["add", self.source, *extra]
        for agent in self.agents:
            args += ["-a", agent]
        if self.mode == "copy":
            args.append("--copy")
        return [*args, "-y"]

    def install_all(self, project: Path) -> None:
        self.cli.run(project, *self.add_args("--skill", "*"))

    def check_installed(self, project: Path, names: dict[str, list[str]]) -> None:
        errors: list[str] = []
        canonical_root = project / CANONICAL_DIR
        present = sorted(p.name for p in canonical_root.iterdir()) if canonical_root.is_dir() else []
        if sorted(names) != present:
            errors.append(
                f"{canonical_root}: expected skills {sorted(names)}, found {present}"
            )
        for name, files in names.items():
            source = self.repo_root / "skills" / name
            canonical = canonical_root / name
            errors += check_tree(canonical, source, files)
            for agent in self.agents:
                agent_dir = project / AGENT_DIRS[agent] / name
                if agent_dir == canonical:
                    continue
                if self.mode == "symlink" and not same_dir(agent_dir, canonical):
                    errors.append(f"{agent_dir}: not linked to {canonical}")
                errors += check_tree(agent_dir, source, files)
        lock = project / "skills-lock.json"
        if lock.is_file():
            locked = set(json.loads(lock.read_text(encoding="utf-8")).get("skills", {}))
            if locked != set(names):
                errors.append(
                    f"{lock}: lists {sorted(locked)}, expected {sorted(names)}"
                )
        else:
            errors.append(f"{lock}: not written")
        if errors:
            raise CheckFailed("\n".join(errors))

    def discover(self) -> str:
        if not self.whole_catalog:
            return "skipped: --expect narrows the source to part of the catalog"
        out = self.cli.run(self.project("discover"), "add", self.source, "--list")
        groups = parse_list(out)
        errors: list[str] = []
        found, want = set(groups), set(self.catalog)
        if found != want:
            errors.append(
                f"--list found {sorted(found)}; skills/ has {sorted(want)}"
                f" (missing {sorted(want - found)}, extra {sorted(found - want)})"
            )
        published = published_skills(self.repo_root)
        grouped = {n for n, g in groups.items() if g != UNGROUPED_LABEL}
        if grouped != published & found:
            errors.append(
                f"plugin group holds {sorted(grouped)}; the manifest publishes"
                f" {sorted(published)}"
            )
        if errors:
            raise CheckFailed("\n".join(errors))
        return f"{len(found)} skills, {len(grouped)} in the plugin group"

    def install(self) -> str:
        project = self.project("install")
        self.install_all(project)
        self.check_installed(project, self.expected)
        listed = self.cli.run(project, "list")
        missing = [n for n in self.expected if n not in listed]
        if missing:
            raise CheckFailed(f"`skills list` does not show {missing}")
        files = sum(len(f) for f in self.expected.values())
        return f"{len(self.expected)} skills, {files} files, agents {', '.join(self.agents)}"

    def reinstall(self) -> str:
        project = self.project("reinstall")
        self.install_all(project)
        self.install_all(project)
        self.check_installed(project, self.expected)
        return "second install left an identical tree"

    def remove(self) -> str:
        project = self.project("remove")
        self.install_all(project)
        target = min(self.expected)
        self.cli.run(project, "remove", target, "-y")
        leftovers = [
            str(project / d / target)
            for d in {CANONICAL_DIR, *(AGENT_DIRS[a] for a in self.agents)}
            if os.path.lexists(project / d / target)
        ]
        if leftovers:
            raise CheckFailed(f"`remove {target}` left {leftovers}")
        rest = {n: f for n, f in self.expected.items() if n != target}
        if rest:
            self.check_installed(project, rest)
        return f"removed {target}, {len(rest)} left intact"

    def global_install(self) -> str:
        project = self.project("global")
        target = min(self.expected)
        args = ["add", self.source, "--skill", target, "-a", GLOBAL_AGENT, "-g"]
        if self.mode == "copy":
            args.append("--copy")
        self.cli.run(project, *args, "-y")
        installed = self.cli.home / GLOBAL_AGENT_DIR / target
        errors = check_tree(installed, self.repo_root / "skills" / target, self.expected[target])
        written = sorted(p.name for p in project.iterdir())
        if written:
            errors.append(f"`-g` also wrote into the project: {written}")
        if errors:
            raise CheckFailed("\n".join(errors))
        return f"{target} under ~/{GLOBAL_AGENT_DIR}"

    def update(self) -> str:
        if not self.remote:
            return "skipped: local-path installs are not updatable"
        project = self.project("update")
        self.install_all(project)
        self.cli.run(project, "update", "-y")
        self.check_installed(project, self.expected)
        return "update ran clean against the installed revision"

    def run(self, scenarios: list[str]) -> int:
        steps = {
            "discover": self.discover,
            "install": self.install,
            "reinstall": self.reinstall,
            "remove": self.remove,
            "global": self.global_install,
            "update": self.update,
        }
        print(f"source={self.source} mode={self.mode} cli={self.cli.package}")
        print(f"checking against {self.repo_root} ({len(self.expected)} skills)\n")
        try:
            self.cli.run(self.work, "--version", timeout=300)
        except CheckFailed as exc:
            print(f"The CLI did not start, so no scenario can run:\n{exc}")
            return 1
        results: list[tuple[str, bool, str, float]] = []
        for name in scenarios:
            print(f"[{name}]", flush=True)
            start = time.monotonic()
            try:
                detail = steps[name]()
                ok = True
            except CheckFailed as exc:
                detail, ok = str(exc), False
            elapsed = time.monotonic() - start
            results.append((name, ok, detail, elapsed))
            print(f"  {'PASS' if ok else 'FAIL'} ({elapsed:.0f}s) {detail}\n", flush=True)
        shutil.rmtree(self.work, ignore_errors=True)
        self.summarize(results)
        failed = [r[0] for r in results if not r[1]]
        print(f"Summary: {len(results) - len(failed)}/{len(results)} passed"
              + (f"; failed: {', '.join(failed)}" if failed else ""))
        return 1 if failed else 0

    def summarize(self, results: list[tuple[str, bool, str, float]]) -> None:
        path = os.environ.get("GITHUB_STEP_SUMMARY")
        if not path:
            return
        lines = [
            f"### `skills add {self.source}` ({self.mode}, {self.cli.package})",
            "",
            "| Scenario | Result | Time | Detail |",
            "|---|---|---|---|",
        ]
        for name, ok, detail, elapsed in results:
            first = detail.splitlines()[0] if detail else ""
            lines.append(
                f"| {name} | {'pass' if ok else '**fail**'} | {elapsed:.0f}s"
                f" | {first.replace('|', '/')} |"
            )
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", required=True,
                        help="Source passed to `skills add` (path, owner/repo, or URL).")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT,
                        help="Git checkout at the revision the source resolves to.")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--cli-version",
                        default=os.environ.get("SKILLS_CLI_VERSION", DEFAULT_CLI_VERSION),
                        help="`skills` npm version, or `latest`.")
    parser.add_argument("--agent", action="append", choices=sorted(AGENT_DIRS),
                        help="Agents to install for (repeatable; default: all four).")
    parser.add_argument("--expect", action="append", default=[],
                        help="Skill the source should yield (repeatable; default: all).")
    parser.add_argument("--scenarios", nargs="+", choices=ALL_SCENARIOS,
                        default=ALL_SCENARIOS)
    args = parser.parse_args(argv)
    args.agent = args.agent or list(AGENT_DIRS)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    return Suite(args).run(args.scenarios)


if __name__ == "__main__":
    raise SystemExit(main())
