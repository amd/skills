#!/usr/bin/env python3
"""Write a bump's eval results into the skill's `skill-card.md`.

Called by the `skill-card-evals` workflow once the `evals` run on a
`Bump <skill>` pull request finishes. It reads that run's results, renders a
short `## Evaluation` section, and puts it at the end of the vendored card:

    uv run .github/scripts/skill_card_evals.py \
        --repo amd/skills --run-id 40786582946 \
        --skill tracelens-analysis-orchestrator

`--print` writes the section to stdout instead of touching the card.

Three sources feed the section, all read through the `gh` CLI:

* the conclusion of skillscope's structural job, which checks SKILL.md's
  format, the skill card's required sections, the eval dataset, and that the
  skill's internal links resolve;
* the `routing-report` artifact, when routing ran, for this skill's recall,
  precision, and near-miss cases plus the accuracy of the whole routing room;
* the logs of this skill's `Behavioral (<skill> on <os>)` jobs, one per OS,
  since skillscope prints each case's result there and uploads no artifact.

The section sits between two HTML comments, so a re-run replaces it in place.
A bump re-vendors the skill folder wholesale, which resets the card to the
upstream copy, so each bump's card carries that bump's results and no other.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

START = "<!-- skill-card-evals:start -->"
END = "<!-- skill-card-evals:end -->"
MARKER_FILENAME = ".federated.json"
# skillscope's discover job, which runs the structural checks before selecting
# what to grade.
STRUCTURAL_JOB = "Check dataset structure and select runs"

CASE_RE = re.compile(r"\[(PASS|FAIL)\] ([\w.-]+): (\d+)/(\d+) checks in ([\d.]+)s")
SUMMARY_RE = re.compile(r"cases passed\*\* .*? on `([^`]+)` \(effort `([^`]+)`\)")


def behavioral_job_label(job_name: str, skill: str) -> str | None:
    """`Linux` for `evals / Behavioral (<skill> on Linux)`, else None.

    Scoped-hardware legs carry their environment, as in
    `Behavioral (<skill> on Linux, behavioral-instinct)`; it is kept in the
    label so two legs on the same OS stay apart.
    """
    match = re.search(rf"Behavioral \({re.escape(skill)} on ([^,)]+)(?:, ([^)]+))?\)", job_name)
    if not match:
        return None
    os_name, environment = match.groups()
    return f"{os_name} ({environment})" if environment else os_name


def parse_behavioral_log(text: str) -> dict:
    """Case results and the model they ran on, from one behavioral job's log."""
    cases = [
        {
            "id": m.group(2),
            "passed": m.group(1) == "PASS",
            "met": int(m.group(3)),
            "total": int(m.group(4)),
            "elapsed_s": float(m.group(5)),
        }
        for m in CASE_RE.finditer(text)
    ]
    summary = SUMMARY_RE.search(text)
    return {
        "cases": cases,
        "model": summary.group(1) if summary else None,
        "effort": summary.group(2) if summary else None,
    }


def routing_rows(report: dict, skill: str) -> list[tuple[str, str]]:
    """This skill's routing rows, or none when it is not in the routing room."""
    stats = report.get("per_skill", {}).get(skill)
    if not stats:
        return []

    def ratio(part: int, whole: int) -> str:
        return f"{part}/{whole} ({round(100 * part / whole)}%)" if whole else "n/a"

    rows = [
        ("Routing recall (triggers when it should)", ratio(stats["correct"], stats["expected"])),
        ("Routing precision (no false triggers)", ratio(stats["correct"], stats["fired_total"])),
    ]
    near_misses = [
        c for c in report.get("cases", [])
        if c.get("category") == "near_miss" and c.get("skill") == skill
    ]
    if near_misses:
        ignored = sum(1 for c in near_misses if c.get("passed"))
        rows.append(("Near-miss prompts correctly ignored", f"{ignored}/{len(near_misses)}"))
    return rows


def behavioral_cell(leg: dict, expected_cases: int, label: str | None) -> str:
    """One OS's aggregate; `label` is None when only one OS ran.

    `expected_cases` is every case seen on any OS, so a leg that crashed
    partway reads as short of the others rather than as a clean pass.
    """
    cases = leg["cases"]
    passed = sum(1 for c in cases if c["passed"])
    met = sum(c["met"] for c in cases)
    total = sum(c["total"] for c in cases)
    elapsed = round(sum(c["elapsed_s"] for c in cases))
    body = f"{passed}/{expected_cases} cases ({met}/{total} checks), {elapsed}s"
    if leg.get("conclusion") == "failure" and len(cases) < expected_cases:
        body += ", job failed"
    return body if label is None else f"{label} {body}"


def render(
    skill: str,
    commit: str,
    run_url: str,
    routing: dict | None,
    behavioral: dict[str, dict],
    structural: str | None = None,
) -> str:
    """The `## Evaluation` section, markers included.

    `structural` is the conclusion of the job that runs skillscope's
    structural checks, or None if the run had no such job.
    """
    rows = []
    if structural:
        result = {"success": "Passed", "failure": "Failed"}.get(structural, structural)
        rows.append(("Structure (SKILL.md format, skill card, eval dataset, internal links)", result))
    rows += routing_rows(routing, skill) if routing else []

    case_ids: list[str] = []
    for leg in behavioral.values():
        for case in leg["cases"]:
            if case["id"] not in case_ids:
                case_ids.append(case["id"])
    labels = list(behavioral)
    if len(labels) == 1:
        rows.append(("Behavioral", behavioral_cell(behavioral[labels[0]], len(case_ids), None)))
    elif labels:
        rows.append(("Behavioral", " · ".join(
            behavioral_cell(behavioral[label], len(case_ids), label) for label in labels
        )))

    model = next((leg["model"] for leg in behavioral.values() if leg["model"]), None)
    effort = next((leg["effort"] for leg in behavioral.values() if leg["effort"]), None)
    if not model and routing:
        model, effort = routing["meta"].get("model"), routing["meta"].get("effort")

    footer = []
    if model:
        footer.append(f"Model: {model} ({effort})" if effort else f"Model: {model}")
    if len(labels) == 1:
        footer.append(f"Behavioral on {labels[0]}")
    if routing:
        totals = routing["totals"]
        accuracy = round(100 * totals["passed"] / totals["graded"]) if totals["graded"] else 0
        footer.append(
            f"Routing room: {len(routing['meta']['skills'])} published skills, "
            f"room accuracy {totals['passed']}/{totals['graded']} ({accuracy}%)"
        )
    elif labels:
        footer.append("Routing: not run for this bump")
    else:
        footer.append("No graded evals ran for this bump")

    lines = [START, f"## Evaluation — `{commit}` ([CI run]({run_url}))", ""]
    if rows:
        lines += ["| Check | Result |", "| --- | --- |"]
        lines += [f"| {check} | {result} |" for check, result in rows]
        lines.append("")
    lines.append(" · ".join(footer))
    lines.append(END)
    return "\n".join(lines) + "\n"


def apply(card: str, section: str) -> str:
    """Replace the card's section if it has one, else append it."""
    start, end = card.find(START), card.find(END)
    if start != -1 and end > start:
        return card[:start] + section + card[end + len(END):].lstrip("\n")
    return card.rstrip("\n") + "\n\n" + section


def gh(*args: str) -> str:
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def fetch(repo: str, run_id: str, skill: str) -> tuple[dict | None, dict[str, dict], str | None]:
    """The run's routing report (None if routing did not run), behavioral legs,
    and the structural checks' conclusion."""
    routing = None
    with tempfile.TemporaryDirectory() as tmp:
        try:
            gh("run", "download", run_id, "-R", repo, "-n", "routing-report", "-D", tmp)
        except subprocess.CalledProcessError:
            pass
        report = Path(tmp) / "routing-report.json"
        if report.is_file():
            routing = json.loads(report.read_text(encoding="utf-8"))

    jobs = gh(
        "api", "--paginate", f"repos/{repo}/actions/runs/{run_id}/jobs",
        "--jq", ".jobs[] | {id, name, conclusion}",
    )
    behavioral: dict[str, dict] = {}
    structural = None
    for line in jobs.splitlines():
        job = json.loads(line)
        if job["name"].endswith(STRUCTURAL_JOB):
            structural = job["conclusion"]
            continue
        label = behavioral_job_label(job["name"], skill)
        if label is None or job["conclusion"] in ("skipped", "cancelled", None):
            continue
        behavioral[label] = parse_behavioral_log(
            gh("api", f"repos/{repo}/actions/jobs/{job['id']}/logs")
        ) | {"conclusion": job["conclusion"]}
    return routing, dict(sorted(behavioral.items())), structural


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repo", required=True, help="owner/repo the run belongs to")
    parser.add_argument("--run-id", required=True, help="the `evals` workflow run")
    parser.add_argument("--skill", required=True, help="the bumped skill's folder name")
    parser.add_argument("--skills-dir", type=Path, default=Path("skills"))
    parser.add_argument("--commit", help="commit to name in the heading (default: the vendored marker's)")
    parser.add_argument("--print", action="store_true", help="print the section instead of editing the card")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    skill_dir = args.skills_dir / args.skill
    commit = args.commit
    if not commit:
        marker = skill_dir / MARKER_FILENAME
        commit = json.loads(marker.read_text(encoding="utf-8"))["commit"] if marker.is_file() else "unknown"
    run_url = f"https://github.com/{args.repo}/actions/runs/{args.run_id}"

    routing, behavioral, structural = fetch(args.repo, args.run_id, args.skill)
    section = render(args.skill, commit[:7], run_url, routing, behavioral, structural)
    if args.print:
        sys.stdout.write(section)
        return 0

    card = skill_dir / "skill-card.md"
    card.write_text(apply(card.read_text(encoding="utf-8"), section), encoding="utf-8")
    print(f"Wrote the evaluation section to {card}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
