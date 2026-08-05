#!/usr/bin/env python3
"""Parse HRR replay/capture logs into a structured finding (read-only)."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# --- regex library (diverse workloads) ---

RE_PROGRESS = re.compile(
    r"\[HRR progress\].*seq=(\d+).*kernels=(\d+).*d2h_pass=(\d+).*"
    r"d2h_fail=(\d+).*d2h_attempted=(\d+).*last=\"([^\"]+)\""
)
RE_FATAL_EVENT = re.compile(
    r"\[HRR\] Fatal: T(\d+) Event (\d+) \(([^)]+)\) returned (\d+) \(([^)]+)\)"
)
RE_FATAL_GPU = re.compile(
    r"\[HRR\] Fatal: GPU error after T(\d+) Event (\d+) \(([^)]+)\): (\d+) \(([^)]+)\)"
)
RE_FATAL_GENERIC = re.compile(r"\[HRR\] Fatal: ([^\n]+)")
RE_MAF = re.compile(
    r"Memory access fault by GPU node-(\d+).*on address (0x[0-9a-fA-F]+)\.\s*"
    r"Reason:\s*([^.\n]+)"
)
# The leading fields of this bracket vary by ROCm build -- some emit `host:`,
# some start at `GPU index:` -- so anchor on the two fields actually consumed
# rather than on the whole prefix.
RE_MEM_FAULT_ERR = re.compile(
    r"Memory Fault Error \[[^\]]*?faulting addr: (0x[0-9a-fA-F]+), kernel: ([^\]]+)\]"
)
RE_HANG = re.compile(r"HSA_STATUS_ERROR_(MEMORY_FAULT|ABORTED|EXCEPTION)")
RE_PASS = re.compile(r"\[HRR\] PASS\b")
RE_FAIL = re.compile(r"\[HRR\] FAIL\b")
RE_ARCHIVE_RECOVERED = re.compile(
    r"recovered (\d+) events|Archive : (\d+) events, (\d+) kernels, (\d+) blobs, (\d+) code objects"
)
# `--info` prints `Complete:     yes (clean shutdown)` or `Complete:     NO (...)`,
# so the verdict is case-mixed and always followed by an explanation.
RE_ARCHIVE_COMPLETE = re.compile(r"Complete:\s+(yes|no)\b", re.IGNORECASE)
# `--info` reports the archive as labelled fields, one per line; a replay run
# reports the same totals on a single `[HRR] Archive :` line.
RE_INFO_EVENTS = re.compile(r"^Events:\s+(\d+)\s*$", re.MULTILINE)
RE_INFO_KERNELS = re.compile(r"^Kernels:\s+(\d+)\s*$", re.MULTILINE)
# Short archives print no `Kernels:` total and report launches only in the API
# call-count block. Without this the total stays unknown and the single-kernel
# inference in finalize() can never fire.
RE_INFO_LAUNCH_COUNT = re.compile(r"\bhip\w*LaunchKernel\s+(\d+)\b")
# Rows of the `--info` "Kernel Summary" table: name, grid, block, with an
# optional leading id (some builds omit the id column). A memory fault can kill
# the replay before any per-launch attribution reaches the log, in which case
# the archive's kernel list is the only record of what ran. The name must start
# with a letter or underscore, which is what keeps a bare id from being read as
# a symbol now that the id is optional.
RE_INFO_KERNEL_ROW = re.compile(
    r"^[ \t]*(?:\d+[ \t]+)?([A-Za-z_][^\s\[]*)[ \t]+\[[\d,\s]+\][ \t]+\[[\d,\s]+\]",
    re.MULTILINE,
)
# `--sync-after-launch` and `--sync-after-event` print one line per replayed
# event. A GPU fault tears the process down before HRR writes its own Fatal
# line, so the last of these is often the only record of the failing dispatch.
RE_EVENT_PROGRESS = re.compile(
    r"^[ \t]*(?:\[HRR\][ \t]*)?Event (\d+):[ \t]*(\w+)"
    r"(?:[^\n]*?->[ \t]*Kernel '([^']+)')?",
    re.MULTILINE,
)
# PyTorch/ATen kernels reach the GPU through `<<<>>>` (hipLaunchByPtr) and pass
# device pointers inside by-value structs. Capture records those (arg encoding
# `value_kind == 3`) and replay translates them, plus a defensive rescan, so
# these kernels do replay faithfully on a current build. The detector is still a
# value-based heuristic, and an archive recorded before it landed carries no
# such offsets at all, so a fault on one of these symbols earns a caveat in the
# finding -- not a different verdict.
RE_ATEN_CHEVRON = re.compile(r"_ZN2at6native|at::native::")
# Emitted by the archive reader when the on-disk format and this hrr-playback
# build disagree. Nothing is replayed, so this outranks every other signal.
RE_VERSION_MISMATCH = re.compile(r"\[HRR\] Version mismatch: file=(\d+) reader=(\d+)")
RE_CAPTURE_MAF = RE_MAF
RE_SUBALLOC_OOB = re.compile(
    r"\[HRR\] SUBALLOC OOB: kernel arg\[(\d+)\] rec (0x[0-9a-fA-F]+)"
)
RE_D2H_SUMMARY = re.compile(
    r"D2H checks\s+: (\d+) pass.*?, (\d+) fail, (\d+) skipped"
)
RE_KERNARG = re.compile(r"kernarg_address=(0x[0-9a-fA-F]+)")
RE_GRID = re.compile(r"grid=\[([^\]]+)\], workgroup=\[([^\]]+)\]")
RE_CIJK = re.compile(r"(Cijk_[A-Za-z0-9_]+)")
RE_CAPTURE_HIP = re.compile(r"\[capture\] HIP_SO=(\S+)")


@dataclass
class Finding:
    outcome: str
    fault_class: str
    fault_address: str | None = None
    fault_reason: str | None = None
    failing_event_seq: int | None = None
    failing_call_index: int | None = None
    failing_thread: int | None = None
    failing_api: str | None = None
    kernel_name: str | None = None
    kernel_family: str | None = None
    kernarg_address: str | None = None
    grid: str | None = None
    workgroup: str | None = None
    gpu_node: str | None = None
    last_progress_kernel: str | None = None
    last_event_kernel: str | None = None
    kernels_launched: int | None = None
    d2h_pass: int | None = None
    d2h_fail: int | None = None
    d2h_attempted: int | None = None
    suballoc_oob_count: int = 0
    suballoc_oob_args: list[int] = field(default_factory=list)
    archive_events: int | None = None
    archive_kernels: int | None = None
    archive_kernel_names: list[str] = field(default_factory=list)
    archive_complete: str | None = None
    archive_format_version: int | None = None
    reader_format_version: int | None = None
    capture_hip_so: str | None = None
    sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classify(text: str, finding: Finding) -> str:
    # Order matters. A GPU memory fault also trips HRR's own abort line and an
    # HSA_STATUS_ERROR_MEMORY_FAULT, so the fault has to be classified before
    # the generic abort and hang branches or every fault reads as an API error.
    if RE_VERSION_MISMATCH.search(text):
        return "archive_version_mismatch"
    if RE_MAF.search(text) or RE_MEM_FAULT_ERR.search(text):
        reason = (finding.fault_reason or "").lower()
        if "read-only" in reason:
            return "read_only_page_fault"
        return "illegal_memory_access"
    if RE_PASS.search(text):
        if finding.d2h_fail and finding.d2h_fail > 0:
            return "nan_inf_divergence"
        return "replay_pass"
    if "out of memory" in text.lower() or "hipErrorOutOfMemory" in text:
        return "replay_oom"
    if RE_FATAL_EVENT.search(text) or RE_FATAL_GPU.search(text) or RE_FATAL_GENERIC.search(text):
        return "replay_fatal_api"
    if RE_HANG.search(text):
        return "hang"
    if RE_FAIL.search(text) or (finding.d2h_fail and finding.d2h_fail > 0):
        return "nan_inf_divergence"
    if "Replay aborted" in text or "aborting replay" in text:
        return "replay_aborted"
    return "unknown"


def _kernel_family(name: str | None) -> str | None:
    if not name:
        return None
    if name.startswith("Cijk_"):
        m = re.search(r"_MT(\d+x\d+x\d+)", name)
        sk = "_SK3_" if "_SK3_" in name else ("_SK2_" if "_SK2_" in name else None)
        parts = ["hipblaslt_gemm"]
        if m:
            parts.append(f"MT{m.group(1)}")
        if sk:
            parts.append("streamk" if "SK3" in sk else "streamk_variant")
        return "/".join(parts)
    if name.startswith("_ZN"):
        return "pytorch_kernel"
    return "other"


def parse_text(text: str, source: str, finding: Finding) -> Finding:
    finding.sources.append(source)

    for m in RE_CAPTURE_HIP.finditer(text):
        finding.capture_hip_so = m.group(1)

    for m in RE_ARCHIVE_RECOVERED.finditer(text):
        g = m.groups()
        if g[0]:
            finding.archive_events = int(g[0])
        if len(g) >= 5 and g[1]:
            finding.archive_events = int(g[1])
            finding.archive_kernels = int(g[2])

    m = RE_ARCHIVE_COMPLETE.search(text)
    if m:
        finding.archive_complete = m.group(1).lower()

    m = RE_INFO_EVENTS.search(text)
    if m:
        finding.archive_events = int(m.group(1))
    m = RE_INFO_KERNELS.search(text)
    if m:
        finding.archive_kernels = int(m.group(1))
    elif finding.archive_kernels is None:
        m = RE_INFO_LAUNCH_COUNT.search(text)
        if m:
            finding.archive_kernels = int(m.group(1))

    for name in RE_INFO_KERNEL_ROW.findall(text):
        # The `--info` table truncates long names to its column width, and a
        # truncated symbol is worse than none: it cannot be looked up or given
        # to a kernel developer.
        if name.endswith("...") or name in finding.archive_kernel_names:
            continue
        finding.archive_kernel_names.append(name)

    m = RE_VERSION_MISMATCH.search(text)
    if m:
        finding.archive_format_version = int(m.group(1))
        finding.reader_format_version = int(m.group(2))
        finding.notes.append(
            f"archive format v{m.group(1)} cannot be read by this hrr-playback "
            f"(reader v{m.group(2)}); nothing was replayed"
        )

    oob_args: set[int] = set()
    for m in RE_SUBALLOC_OOB.finditer(text):
        finding.suballoc_oob_count += 1
        oob_args.add(int(m.group(1)))
    finding.suballoc_oob_args = sorted(oob_args)

    last_prog = None
    for m in RE_PROGRESS.finditer(text):
        finding.failing_event_seq = int(m.group(1))
        finding.kernels_launched = int(m.group(2))
        finding.d2h_pass = int(m.group(3))
        finding.d2h_fail = int(m.group(4))
        finding.d2h_attempted = int(m.group(5))
        last_prog = m.group(6)
    finding.last_progress_kernel = last_prog

    for m in (RE_FATAL_EVENT, RE_FATAL_GPU):
        hit = m.search(text)
        if hit:
            finding.failing_thread = int(hit.group(1))
            finding.failing_call_index = int(hit.group(2))
            finding.failing_api = hit.group(3)
            break

    last_event = None
    for m in RE_EVENT_PROGRESS.finditer(text):
        last_event = m
    if last_event is not None:
        # An HRR Fatal line names the failing event exactly; this is only the
        # last event that started, so it never overrides one.
        if finding.failing_call_index is None:
            finding.failing_call_index = int(last_event.group(1))
            finding.failing_api = last_event.group(2)
        finding.last_event_kernel = last_event.group(3)

    m = RE_MAF.search(text)
    if m:
        finding.gpu_node = m.group(1)
        finding.fault_address = m.group(2)
        finding.fault_reason = m.group(3).strip()

    m = RE_MEM_FAULT_ERR.search(text)
    if m:
        finding.fault_address = finding.fault_address or m.group(1)
        finding.kernel_name = m.group(2).strip()

    if not finding.kernel_name:
        cijk = RE_CIJK.search(text)
        if cijk:
            finding.kernel_name = cijk.group(1)

    m = RE_KERNARG.search(text)
    if m:
        finding.kernarg_address = m.group(1)

    m = RE_GRID.search(text)
    if m:
        finding.grid = m.group(1)
        finding.workgroup = m.group(2)

    m = RE_D2H_SUMMARY.search(text)
    if m:
        finding.d2h_pass = int(m.group(1))
        finding.d2h_fail = int(m.group(2))

    finding.kernel_family = _kernel_family(finding.kernel_name)
    return finding


def finalize(finding: Finding, corpus: str) -> Finding:
    """Set the verdict from every input at once.

    Classification has to run over the whole corpus rather than per input: a
    `--analyze` run parses the replay log and then the archive `--info` dump,
    and the `--info` text alone carries no verdict, so classifying per input
    would let it overwrite the replay's outcome with UNKNOWN.
    """
    if RE_VERSION_MISMATCH.search(corpus):
        finding.outcome = "UNREADABLE"
    elif RE_MAF.search(corpus) or RE_MEM_FAULT_ERR.search(corpus):
        finding.outcome = "MAF"
    elif RE_PASS.search(corpus):
        finding.outcome = "PASS"
    elif RE_FAIL.search(corpus):
        finding.outcome = "FAIL"
    elif "aborting replay" in corpus or RE_FATAL_EVENT.search(corpus):
        finding.outcome = "ABORT"
    else:
        finding.outcome = "UNKNOWN"

    finding.fault_class = _classify(corpus, finding)

    # A clean replay has no implicated kernel. The archive still lists the
    # kernels it ran, and a GEMM name matched out of that listing would sit in
    # the report next to a PASS as if it were a culprit.
    if finding.fault_class == "replay_pass":
        finding.kernel_name = None
        finding.kernel_family = None
        finding.last_event_kernel = None
        return finding

    # Under --sync-after-launch the last launch to start is the one that
    # faulted, so it stands in when the runtime's fault line carried no kernel.
    if not finding.kernel_name and finding.last_event_kernel:
        finding.kernel_name = finding.last_event_kernel
        finding.notes.append(
            f"kernel name taken from the last launch to start before the fault "
            f"({finding.last_event_kernel}); the runtime fault line named no "
            f"kernel. Valid only because the replay ran with "
            f"--sync-after-launch."
        )

    # A memory fault can tear the process down before HRR attributes the
    # failing dispatch, leaving a replay log with no kernel at all. When the
    # archive holds exactly one kernel, that kernel is the one that faulted.
    # Anything more than one stays unknown: guessing among several would be
    # exactly the plausible-but-unevidenced answer the skill forbids.
    if (
        not finding.kernel_name
        and finding.archive_kernels == 1
        and len(finding.archive_kernel_names) == 1
    ):
        finding.kernel_name = finding.archive_kernel_names[0]
        finding.notes.append(
            f"kernel name inferred from the archive, which contains exactly one "
            f"kernel ({finding.kernel_name}); the replay log carried no "
            f"per-launch attribution. Re-run with --sync-after-launch to confirm "
            f"the faulting dispatch directly."
        )

    # A `<<<>>>`-launched ATen kernel passes device pointers inside by-value
    # structs. Current capture records those offsets and replay translates them,
    # so this is not automatically a recording artefact, but the detector is a
    # heuristic and an archive taken before it landed carries no offsets at all.
    # Both failure modes look exactly like a workload fault, so flag the
    # ambiguity rather than resolving it either way.
    if finding.fault_class in ("illegal_memory_access", "read_only_page_fault") and (
        RE_ATEN_CHEVRON.search(finding.kernel_name or "")
    ):
        finding.notes.append(
            "the faulting kernel is an ATen kernel launched through <<<>>> "
            "(hipLaunchByPtr), which passes device pointers inside by-value "
            "structs. Replay translates those via a value-based heuristic, and "
            "an archive recorded before that support landed has none recorded "
            "at all, so an untranslated pointer here would fault exactly like a "
            "workload defect. Confirm against the user's original failure "
            "signature before reporting this as their bug."
        )

    finding.kernel_family = _kernel_family(finding.kernel_name)
    return finding


def run_archive_info(archive: Path, hrr_playback: str | None) -> str:
    play = hrr_playback or "hrr-playback"
    try:
        proc = subprocess.run(
            [play, str(archive), "--info"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return proc.stdout + proc.stderr
    except FileNotFoundError:
        return ""
    except subprocess.TimeoutExpired:
        return "[timeout running hrr-playback --info]"


def parse_sweep_tsv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return rows
    header = lines[0].split("\t")
    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split("\t")
        rows.append(dict(zip(header, cols)))
    return rows


def render_markdown(f: Finding, sweep: list[dict[str, Any]] | None = None) -> str:
    lines = [
        "# HRR replay finding",
        "",
        "## Summary",
        f"- **Outcome**: {f.outcome}",
        f"- **Fault class**: `{f.fault_class}`",
        f"- **Kernel**: `{f.kernel_name or 'unknown'}`",
        f"- **Kernel family**: `{f.kernel_family or 'unknown'}`",
        "",
        "## Fault details",
        f"- **Fault address**: `{f.fault_address or 'n/a'}`",
        f"- **Fault reason**: {f.fault_reason or 'n/a'}",
        f"- **Failing event seq**: {f.failing_event_seq or 'n/a'}",
        f"- **Failing call index**: {f.failing_call_index or 'n/a'}",
        f"- **Failing API**: {f.failing_api or 'n/a'}",
        f"- **Kernarg address**: `{f.kernarg_address or 'n/a'}`",
        f"- **GPU node**: {f.gpu_node or 'n/a'}",
        f"- **Grid / workgroup**: {f.grid or 'n/a'} / {f.workgroup or 'n/a'}",
        "",
        "## Replay progress at fault",
        f"- **Kernels launched**: {f.kernels_launched or 'n/a'}",
        f"- **D2H**: pass={f.d2h_pass or 0} fail={f.d2h_fail or 0} attempted={f.d2h_attempted or 0}",
        f"- **Last progress kernel**: `{f.last_progress_kernel or 'n/a'}`",
        f"- **Last launch before fault**: `{f.last_event_kernel or 'n/a'}`",
        "",
        "## Archive / capture",
        f"- **Events**: {f.archive_events or 'n/a'}",
        f"- **Kernels (archive)**: {f.archive_kernels or 'n/a'}",
        f"- **Kernel names (archive)**: {', '.join(f.archive_kernel_names) or 'n/a'}",
        f"- **Complete**: {f.archive_complete or 'n/a'}",
        f"- **Capture HIP**: `{f.capture_hip_so or 'n/a'}`",
        f"- **Suballoc OOB reports**: {f.suballoc_oob_count} (args: {f.suballoc_oob_args or []})",
        "",
        "## Sources",
    ]
    for s in f.sources:
        lines.append(f"- `{s}`")
    if f.notes:
        lines.extend(["", "## Notes"])
        lines.extend(f"- {n}" for n in f.notes)
    if sweep:
        lines.extend(["", "## Multi-run sweep"])
        lines.append("| run | gpu | outcome | fault_addr |")
        lines.append("|-----|-----|---------|------------|")
        for r in sweep:
            lines.append(
                f"| {r.get('run','')} | {r.get('gpu','')} | {r.get('outcome','')} | {r.get('fault_addr','')} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", action="append", default=[], help="Replay or capture log (repeatable)")
    ap.add_argument("--archive", help="HRR archive pid-* directory for --info")
    ap.add_argument("--sweep-tsv", help="multi-replay sweep summary TSV")
    ap.add_argument("--hrr-playback", help="Path to hrr-playback binary")
    ap.add_argument("--format", choices=("json", "markdown"), default="markdown")
    ap.add_argument("-o", "--output", help="Write report to file")
    args = ap.parse_args()

    if not args.log and not args.archive and not args.sweep_tsv:
        ap.error("provide --log, --archive, and/or --sweep-tsv")

    finding = Finding(outcome="UNKNOWN", fault_class="unknown")
    chunks: list[str] = []
    for log_path in args.log:
        p = Path(log_path)
        if not p.is_file():
            finding.notes.append(f"log not found: {p}")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        chunks.append(text)
        parse_text(text, str(p), finding)

    if args.archive:
        arch = Path(args.archive)
        info = run_archive_info(arch, args.hrr_playback)
        if info:
            chunks.append(info)
            parse_text(info, f"{arch} (--info)", finding)
        else:
            finding.notes.append("hrr-playback --info unavailable; archive path recorded only")
            finding.sources.append(str(arch))

    finalize(finding, "\n".join(chunks))

    sweep = parse_sweep_tsv(Path(args.sweep_tsv)) if args.sweep_tsv else None
    if sweep:
        finding.notes.append(f"multi-replay sweep: {len(sweep)} runs")

    out = (
        json.dumps(finding.to_dict(), indent=2)
        if args.format == "json"
        else render_markdown(finding, sweep)
    )
    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
