# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Simple eval runner: invokes Claude Code with a prompt and reports time + token usage.

Usage:
    python claude_eval.py "your prompt here"
    python claude_eval.py --prompt-file path/to/prompt.txt
    echo "your prompt" | python claude_eval.py -
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"
SKILLS_DIR = REPO_ROOT / "skills"


@dataclass
class EvalResult:
    prompt: str
    model: str | None
    effort: str | None
    skill: str | None
    wall_time_s: float
    duration_ms: int | None
    duration_api_ms: int | None
    num_turns: int | None
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    total_input_tokens: int
    total_cost_usd: float | None
    is_error: bool
    result_text: str
    session_id: str | None


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.prompt == "-" or args.prompt is None:
        data = sys.stdin.read().strip()
        if not data:
            sys.exit("error: no prompt provided (stdin empty)")
        return data
    return args.prompt


def list_available_skills() -> list[str]:
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(
        p.name for p in SKILLS_DIR.iterdir() if p.is_dir() and (p / "SKILL.md").is_file()
    )


@contextlib.contextmanager
def staged_skill_dir(skill: str | None) -> Iterator[Path | None]:
    """Stage a temp directory of the form ``<tmp>/.claude/skills/<skill>/...``
    so Claude Code's normal skill discovery picks it up via ``--add-dir``.

    Per the Claude Code docs:

        The `--add-dir` flag grants file access rather than configuration
        discovery, but skills are an exception: `.claude/skills/` within
        an added directory is loaded automatically.

    This registers the skill (name + description go into the skill listing)
    without injecting its full body into the prompt — Claude only loads the
    body when it decides to use the skill, or when invoked as ``/<skill>``.
    """
    if not skill:
        yield None
        return

    skill_src = SKILLS_DIR / skill
    if not (skill_src / "SKILL.md").is_file():
        available = list_available_skills()
        hint = f" Available skills: {', '.join(available)}." if available else ""
        sys.exit(f"error: skill '{skill}' not found at {skill_src / 'SKILL.md'}.{hint}")

    tmp_root = Path(tempfile.mkdtemp(prefix="eval-skill-"))
    try:
        dest = tmp_root / ".claude" / "skills" / skill
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(skill_src, dest)
        yield tmp_root
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def run_claude(
    prompt: str,
    model: str | None,
    effort: str | None,
    skill: str | None,
    extra_args: list[str],
    yolo: bool = False,
) -> tuple[float, dict]:
    claude_bin = shutil.which("claude")
    if not claude_bin:
        sys.exit("error: 'claude' CLI not found on PATH")

    with staged_skill_dir(skill) as skill_root:
        cmd = [claude_bin, "-p", prompt, "--output-format", "json"]
        if model:
            cmd += ["--model", model]
        if effort:
            cmd += ["--effort", effort]
        if skill_root is not None:
            cmd += ["--add-dir", str(skill_root)]
        if yolo:
            # Bypass all tool-permission prompts so the model can actually run
            # shell, edit files, etc. unattended. Without this, ``claude -p``
            # silently degrades to "I would have run X" because there is no
            # interactive user to approve tool calls.
            cmd += ["--dangerously-skip-permissions"]
        cmd += extra_args

        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            stdin=subprocess.DEVNULL,
        )
        elapsed = time.perf_counter() - start

    stdout = (proc.stdout or "").strip()
    try:
        payload = json.loads(stdout) if stdout else None
    except json.JSONDecodeError:
        payload = None

    if payload is None:
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        if stdout:
            sys.stderr.write(stdout + "\n")
        sys.exit(f"error: claude exited with code {proc.returncode} and produced no JSON output")

    return elapsed, payload


def build_result(
    prompt: str,
    model: str | None,
    effort: str | None,
    skill: str | None,
    elapsed_s: float,
    payload: dict,
) -> EvalResult:
    usage = payload.get("usage") or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)

    return EvalResult(
        prompt=prompt,
        model=model,
        effort=effort,
        skill=skill,
        wall_time_s=round(elapsed_s, 3),
        duration_ms=payload.get("duration_ms"),
        duration_api_ms=payload.get("duration_api_ms"),
        num_turns=payload.get("num_turns"),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
        total_input_tokens=input_tokens + cache_creation + cache_read,
        total_cost_usd=payload.get("total_cost_usd"),
        is_error=bool(payload.get("is_error", False)),
        result_text=payload.get("result", ""),
        session_id=payload.get("session_id"),
    )


def print_human(result: EvalResult) -> None:
    print("=" * 60)
    print("Claude Code Eval Result")
    print("=" * 60)
    print(f"Prompt:           {result.prompt[:120]}{'...' if len(result.prompt) > 120 else ''}")
    print(f"Model:            {result.model or '(default)'}")
    print(f"Effort:           {result.effort or '(default)'}")
    print(f"Skill:            {result.skill or '(none)'}")
    print(f"Wall time:        {result.wall_time_s:.3f} s")
    if result.duration_ms is not None:
        print(f"Reported time:    {result.duration_ms / 1000:.3f} s (api: {(result.duration_api_ms or 0) / 1000:.3f} s)")
    print(f"Turns:            {result.num_turns}")
    print(f"Input tokens:     {result.input_tokens}")
    print(f"  + cache write:  {result.cache_creation_input_tokens}")
    print(f"  + cache read:   {result.cache_read_input_tokens}")
    print(f"  = total in:     {result.total_input_tokens}")
    print(f"Output tokens:    {result.output_tokens}")
    if result.total_cost_usd is not None:
        print(f"Cost (USD):       ${result.total_cost_usd:.6f}")
    print(f"Error:            {result.is_error}")
    print("-" * 60)
    print("Response:")
    print(result.result_text)
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a prompt on Claude Code and measure time + tokens.")
    parser.add_argument("prompt", nargs="?", help="The prompt to send (use '-' to read from stdin).")
    parser.add_argument("--prompt-file", help="Read the prompt from a file.")
    parser.add_argument(
        "--model",
        default="sonnet",
        help="Model alias (e.g. sonnet, opus, haiku) or full name (e.g. claude-sonnet-4-6). Default: sonnet.",
    )
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high", "max"],
        default="high",
        help="Reasoning effort level for the session. Default: high.",
    )
    parser.add_argument(
        "--skill",
        default=None,
        help=(
            "Name of a skill under skills/ to expose to the model "
            "(its SKILL.md is appended to the system prompt). "
            "Omit to run with no skill. Use --list-skills to see options."
        ),
    )
    parser.add_argument(
        "--list-skills",
        action="store_true",
        help="Print the names of available skills under skills/ and exit.",
    )
    parser.add_argument(
        "--yolo",
        "--dangerously-skip-permissions",
        dest="yolo",
        action="store_true",
        help=(
            "Pass --dangerously-skip-permissions to claude, so the model can "
            "use shell / edit / write tools without per-call approval. "
            "Required for any eval whose prompt actually wants the model to "
            "run commands (otherwise claude -p degrades to memory-only answers)."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON to stdout instead of the human-readable summary.")
    parser.add_argument(
        "--output",
        help=(
            "Path to write the result JSON file. Defaults to "
            "eval/runs/<timestamp>-<model>-<effort>.json. "
            "Pass an empty string ('') to skip writing a file."
        ),
    )
    args, extra_args = parser.parse_known_args()
    extra_args = [a for a in extra_args if a != "--"]

    if args.list_skills:
        skills = list_available_skills()
        if not skills:
            print("(no skills found under skills/)")
        else:
            for name in skills:
                print(name)
        return

    prompt = read_prompt(args)

    elapsed, payload = run_claude(
        prompt, args.model, args.effort, args.skill, extra_args, yolo=args.yolo
    )
    result = build_result(prompt, args.model, args.effort, args.skill, elapsed, payload)

    serialized = json.dumps(asdict(result), indent=2)

    output_path: Path | None
    if args.output is None:
        DEFAULT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        skill_part = f"-{args.skill}" if args.skill else ""
        filename = f"{stamp}-{args.model}-{args.effort}{skill_part}.json"
        output_path = DEFAULT_RUNS_DIR / filename
    elif args.output == "":
        output_path = None
    else:
        output_path = Path(args.output)
        if output_path.parent and not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path is not None:
        output_path.write_text(serialized, encoding="utf-8")

    if args.json:
        print(serialized)
    else:
        print_human(result)
        if output_path is not None:
            print(f"Saved JSON to: {output_path}")


if __name__ == "__main__":
    main()
