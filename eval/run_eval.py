"""Simple eval runner: invokes Claude Code with a prompt and reports time + token usage.

Usage:
    python run_eval.py "your prompt here"
    python run_eval.py --prompt-file path/to/prompt.txt
    echo "your prompt" | python run_eval.py -
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EvalResult:
    prompt: str
    model: str | None
    effort: str | None
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


def run_claude(
    prompt: str,
    model: str | None,
    effort: str | None,
    extra_args: list[str],
) -> tuple[float, dict]:
    claude_bin = shutil.which("claude")
    if not claude_bin:
        sys.exit("error: 'claude' CLI not found on PATH")

    cmd = [claude_bin, "-p", prompt, "--output-format", "json"]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
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
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of human-readable output.")
    parser.add_argument("--output", help="Also write the result JSON to this file.")
    args, extra_args = parser.parse_known_args()
    extra_args = [a for a in extra_args if a != "--"]
    prompt = read_prompt(args)

    elapsed, payload = run_claude(prompt, args.model, args.effort, extra_args)
    result = build_result(prompt, args.model, args.effort, elapsed, payload)

    if args.output:
        Path(args.output).write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print_human(result)


if __name__ == "__main__":
    main()
