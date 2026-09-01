# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Isolated Codex CLI setup shared by routing and behavior evals."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path

from datasets import SKILLS_DIR

PLUGIN_NAME = "amd-skills-eval"


def codex_env(home: Path) -> dict[str, str]:
    """Return a subprocess environment with an isolated Codex home."""
    env = dict(os.environ)
    env["CODEX_HOME"] = str(home)
    return env


def _run_setup(cmd: list[str], env: dict[str, str]) -> None:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        env=env,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or f"exit code {proc.returncode}").strip()
        raise RuntimeError(f"Codex plugin setup failed: {detail[:1000]}")


class CodexInstall(AbstractContextManager[Path]):
    """Install exactly ``skills`` as a local plugin in a throwaway Codex home."""

    def __init__(self, skills: list[str]) -> None:
        self.skills = skills
        self.root: Path | None = None
        self.home: Path | None = None

    def __enter__(self) -> Path:
        codex_bin = shutil.which("codex")
        if not codex_bin:
            raise RuntimeError("'codex' CLI not found on PATH")
        if not self.skills:
            raise RuntimeError("cannot build a Codex eval plugin with no skills")

        self.root = Path(tempfile.mkdtemp(prefix="codex-eval-"))
        try:
            self.home = self.root / "home"
            source = self.root / "plugin"
            (source / ".codex-plugin").mkdir(parents=True)
            (source / ".agents" / "plugins").mkdir(parents=True)
            (source / "skills").mkdir()

            skill_paths: list[str] = []
            for skill in self.skills:
                src = SKILLS_DIR / skill
                if not (src / "SKILL.md").is_file():
                    raise FileNotFoundError(
                        f"skill '{skill}' not found at {src / 'SKILL.md'}"
                    )
                shutil.copytree(src, source / "skills" / skill)
                skill_paths.append(f"./skills/{skill}")

            plugin = {
                "name": PLUGIN_NAME,
                "version": "0.0.0-eval",
                "description": "Temporary isolated plugin for AMD skill evaluations.",
                "skills": skill_paths,
            }
            marketplace = {
                "name": PLUGIN_NAME,
                "plugins": [
                    {
                        "name": PLUGIN_NAME,
                        "source": {"source": "local", "path": "./"},
                        "policy": {"installation": "AVAILABLE"},
                    }
                ],
            }
            (source / ".codex-plugin" / "plugin.json").write_text(
                json.dumps(plugin, indent=2) + "\n", encoding="utf-8"
            )
            (source / ".agents" / "plugins" / "marketplace.json").write_text(
                json.dumps(marketplace, indent=2) + "\n", encoding="utf-8"
            )

            self.home.mkdir()
            env = codex_env(self.home)
            _run_setup(
                [codex_bin, "plugin", "marketplace", "add", str(source), "--json"],
                env,
            )
            _run_setup(
                [codex_bin, "plugin", "add", f"{PLUGIN_NAME}@{PLUGIN_NAME}", "--json"],
                env,
            )
            return self.home
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc) -> None:
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
        self.root = None
        self.home = None


def install(skills: list[str]) -> CodexInstall:
    """Return a context manager for an isolated Codex plugin installation."""
    return CodexInstall(skills)


def exec_command(
    workspace: Path,
    *,
    model: str | None,
    effort: str | None,
    sandbox: str,
) -> list[str]:
    """Build a non-interactive Codex JSONL command for one eval prompt."""
    codex_bin = shutil.which("codex")
    if not codex_bin:
        raise RuntimeError("'codex' CLI not found on PATH")
    cmd = [
        codex_bin,
        "exec",
        "--json",
        "--ephemeral",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--sandbox",
        sandbox,
        "--cd",
        str(workspace),
    ]
    if sandbox == "workspace-write":
        cmd.append("--approve-for-me")
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--config", f'model_reasoning_effort="{effort}"']
    cmd.append("-")
    return cmd


def check_api_reachable(
    home: Path, model: str | None = None, effort: str | None = None, timeout: int = 60
) -> tuple[bool, str]:
    """Confirm Codex can authenticate and reach its API from an isolated home."""
    workspace = Path(tempfile.mkdtemp(prefix="codex-preflight-"))
    try:
        proc = subprocess.run(
            exec_command(workspace, model=model, effort=effort, sandbox="read-only"),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            input="Reply with the single word: ok",
            timeout=timeout,
            env=codex_env(home),
        )
    except subprocess.TimeoutExpired:
        return False, f"API preflight timed out after {timeout}s (is the network reachable?)"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or f"exit code {proc.returncode}").strip()
        return False, detail[:500]
    return True, "ok"
