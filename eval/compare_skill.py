# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Compare the same prompt with and without a given skill.

This is a thin wrapper around ``claude_eval.py``: it runs the prompt twice
(or ``--trials N`` times per side), once with the skill exposed and once
without, then prints a side-by-side KPI table and writes a JSON report.

Usage:
    python eval/compare_skill.py "your prompt" --skill local-ai-use
    python eval/compare_skill.py --prompt-file eval/prompts/foo.txt --skill init
    python eval/compare_skill.py "..." --skill review --trials 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Reuse the single-run primitives from the sibling script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from claude_eval import (  # noqa: E402
    EvalResult,
    build_result,
    list_available_skills,
    read_prompt,
    run_claude,
)

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"

# (key, human label, value format string). Keys that aren't on EvalResult
# directly (e.g. duration_s) are derived in ``to_metrics``.
METRICS: list[tuple[str, str, str]] = [
    ("wall_time_s", "Wall time (s)", "{:.3f}"),
    ("duration_s", "Reported time (s)", "{:.3f}"),
    ("duration_api_s", "API time (s)", "{:.3f}"),
    ("num_turns", "Turns", "{:.2f}"),
    ("input_tokens", "Input tokens", "{:.1f}"),
    ("cache_creation_input_tokens", "Cache write tokens", "{:.1f}"),
    ("cache_read_input_tokens", "Cache read tokens", "{:.1f}"),
    ("total_input_tokens", "Total input tokens", "{:.1f}"),
    ("output_tokens", "Output tokens", "{:.1f}"),
    ("total_cost_usd", "Cost (USD)", "${:.6f}"),
]


def to_metrics(r: EvalResult) -> dict[str, float | None]:
    """Project an ``EvalResult`` onto the metric keys we care about."""
    d = asdict(r)
    d["duration_s"] = (r.duration_ms / 1000.0) if r.duration_ms is not None else None
    d["duration_api_s"] = (
        (r.duration_api_ms / 1000.0) if r.duration_api_ms is not None else None
    )
    return d


def aggregate(results: list[EvalResult], keys: list[str]) -> dict[str, dict]:
    metric_rows = [to_metrics(r) for r in results]
    out: dict[str, dict] = {}
    for key in keys:
        values = [m.get(key) for m in metric_rows if m.get(key) is not None]
        if not values:
            out[key] = {"mean": None, "stdev": None, "values": []}
            continue
        out[key] = {
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return out


def run_trials(
    prompt: str,
    model: str,
    effort: str,
    skill: str | None,
    trials: int,
    runs_dir: Path | None,
    label: str,
    yolo: bool = False,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for i in range(trials):
        sys.stderr.write(f"[{label}] trial {i + 1}/{trials} running ...\n")
        sys.stderr.flush()
        elapsed, payload = run_claude(prompt, model, effort, skill, [], yolo=yolo)
        r = build_result(prompt, model, effort, skill, elapsed, payload)
        if r.is_error:
            sys.stderr.write(
                f"[{label}] trial {i + 1} returned is_error=true; "
                "including in results anyway.\n"
            )
        if runs_dir is not None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            skill_part = f"-{skill}" if skill else "-noskill"
            filename = f"{stamp}-{model}-{effort}{skill_part}-{label}-t{i + 1}.json"
            (runs_dir / filename).write_text(
                json.dumps(asdict(r), indent=2), encoding="utf-8"
            )
        results.append(r)
    return results


def fmt_value(v: float | int | None, fmt_str: str) -> str:
    if v is None:
        return "-"
    return fmt_str.format(v)


def fmt_delta(v: float | int | None, fmt_str: str) -> str:
    if v is None:
        return "-"
    sign = "+" if v >= 0 else "-"
    return sign + fmt_str.format(abs(v))


def fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "-"
    return f"{pct:+.1f}%"


def build_comparison(
    without_summary: dict[str, dict],
    with_summary: dict[str, dict],
) -> dict[str, dict]:
    comp: dict[str, dict] = {}
    for key, _label, _fmt in METRICS:
        wo = without_summary.get(key, {}).get("mean")
        wi = with_summary.get(key, {}).get("mean")
        delta = (wi - wo) if (wo is not None and wi is not None) else None
        pct = (delta / wo * 100.0) if (delta is not None and wo not in (None, 0)) else None
        comp[key] = {
            "without_mean": wo,
            "with_mean": wi,
            "delta": delta,
            "pct_change": pct,
        }
    return comp


def print_table(skill: str, trials: int, comparison: dict[str, dict]) -> None:
    headers = ["Metric", "Without skill", f"With '{skill}'", "Δ", "Δ %"]
    rows: list[list[str]] = []
    for key, label, fmt_str in METRICS:
        c = comparison[key]
        rows.append(
            [
                label,
                fmt_value(c["without_mean"], fmt_str),
                fmt_value(c["with_mean"], fmt_str),
                fmt_delta(c["delta"], fmt_str),
                fmt_pct(c["pct_change"]),
            ]
        )

    widths = [
        max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))
    ]
    sep = "  "

    def render(row: list[str]) -> str:
        cells = []
        for i, cell in enumerate(row):
            cells.append(cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]))
        return sep.join(cells)

    print("=" * (sum(widths) + sep_total(widths, sep)))
    print(f"Skill comparison: '{skill}'   (trials per side: {trials})")
    print("=" * (sum(widths) + sep_total(widths, sep)))
    print(render(headers))
    print(sep.join("-" * w for w in widths))
    for row in rows:
        print(render(row))
    print("=" * (sum(widths) + sep_total(widths, sep)))


def sep_total(widths: list[int], sep: str) -> int:
    return len(sep) * (len(widths) - 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the same prompt with and without a given skill. "
            "Wraps claude_eval.py and produces a side-by-side KPI table."
        )
    )
    parser.add_argument(
        "prompt", nargs="?", help="The prompt to send (use '-' to read from stdin)."
    )
    parser.add_argument("--prompt-file", help="Read the prompt from a file.")
    parser.add_argument(
        "--skill",
        required=True,
        help="Name of a skill under skills/ to compare against the no-skill baseline.",
    )
    parser.add_argument(
        "--model",
        default="sonnet",
        help="Model alias or full name. Default: sonnet.",
    )
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high", "max"],
        default="high",
        help="Reasoning effort level. Default: high.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials per side. KPIs are reported as means. Default: 1.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Path for the comparison JSON report. Default: "
            "eval/runs/<timestamp>-compare-<skill>.json. "
            "Pass '' to skip writing a report file."
        ),
    )
    parser.add_argument(
        "--no-save-runs",
        action="store_true",
        help="Don't save individual per-trial JSON files (only the comparison report).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the comparison JSON to stdout instead of the human-readable table.",
    )
    parser.add_argument(
        "--yolo",
        "--dangerously-skip-permissions",
        dest="yolo",
        action="store_true",
        help=(
            "Pass --dangerously-skip-permissions to claude on every trial. "
            "Without this, claude -p stops at the first tool-permission prompt "
            "and just answers from memory, which makes any skill that wants to "
            "run shell / scripts look like a no-op."
        ),
    )

    args = parser.parse_args()

    available = list_available_skills()
    if args.skill not in available:
        hint = f" Available skills: {', '.join(available)}." if available else ""
        sys.exit(f"error: skill '{args.skill}' not found under skills/.{hint}")

    if args.trials < 1:
        sys.exit("error: --trials must be >= 1")

    prompt = read_prompt(args)

    runs_dir: Path | None
    if args.no_save_runs:
        runs_dir = None
    else:
        runs_dir = DEFAULT_RUNS_DIR
        runs_dir.mkdir(parents=True, exist_ok=True)

    without_results = run_trials(
        prompt,
        args.model,
        args.effort,
        None,
        args.trials,
        runs_dir,
        "no-skill",
        yolo=args.yolo,
    )
    with_results = run_trials(
        prompt,
        args.model,
        args.effort,
        args.skill,
        args.trials,
        runs_dir,
        args.skill,
        yolo=args.yolo,
    )

    metric_keys = [k for k, _l, _f in METRICS]
    without_summary = aggregate(without_results, metric_keys)
    with_summary = aggregate(with_results, metric_keys)
    comparison = build_comparison(without_summary, with_summary)

    report = {
        "prompt": prompt,
        "model": args.model,
        "effort": args.effort,
        "skill": args.skill,
        "trials": args.trials,
        "without_skill": {
            "runs": [asdict(r) for r in without_results],
            "summary": without_summary,
        },
        "with_skill": {
            "runs": [asdict(r) for r in with_results],
            "summary": with_summary,
        },
        "comparison": comparison,
    }

    serialized = json.dumps(report, indent=2)

    output_path: Path | None
    if args.output is None:
        DEFAULT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = DEFAULT_RUNS_DIR / f"{stamp}-compare-{args.skill}.json"
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
        print_table(args.skill, args.trials, comparison)
        if output_path is not None:
            print(f"Saved comparison report to: {output_path}")


if __name__ == "__main__":
    main()
