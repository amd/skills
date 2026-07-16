#!/usr/bin/env python3
"""
local-ai-privacy: file redaction gate.

  python3 redact.py INPUT [INPUT ...] -o OUTPUT_DIR

Reads the ORIGINAL files, redacts them with a local Lemonade model plus
deterministic regex detectors, and writes masked copies to OUTPUT_DIR. In the
skill flow this script is the ONLY thing that ever opens the originals; the
cloud model reads just the copies, so raw PII never enters the conversation
and therefore never reaches the cloud.

stdout is part of the trust boundary: it prints masked paths, entity type
counts, and status lines ONLY - never file content, never raw values. Error
messages print exception class names, not messages, because a message can
echo content.

Fail-closed rules:
  - Lemonade or the redaction model unavailable -> exit 2 before touching any
    file. There is deliberately no regex-only fallback.
  - LLM discovery fails for a file after retries -> that file is not written.
  - Every output is re-scanned with the regex detectors; a file with any
    remaining structured-PII hit is withheld (not written).
  - PDFs with no text layer, images without OCR, unsupported binaries -> the
    original is NEVER copied to the output dir; the receipt says so.

Exit codes:
  0  every file redacted (possibly some unsupported files skipped - see receipt)
  1  partial: some files withheld or failed - analyze only what was written
  2  preflight failed (Lemonade / model / paths); nothing was processed
  3  ran, but nothing could be redacted
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from redaction_core import (  # noqa: E402
    LemonadeClient, RedactionSession, regex_discover,
)

SKILL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = SKILL_DIR / "data" / "deidentify-prompt.txt"
DEFAULT_LEMONADE_URL = "http://localhost:13305"
DEFAULT_MODEL = "Qwen3.6-35B-A3B-NoThinking"

TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".xml",
              ".yaml", ".yml", ".html", ".htm", ".log", ".ini", ".toml"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

logging.basicConfig(level=logging.WARNING, format="[redact] %(message)s")


def say(msg: str) -> None:
    print(f"[redact] {msg}", flush=True)


def fail(code: int, *lines: str) -> None:
    for l in lines:
        say(l)
    sys.exit(code)


# ---------------------------------------------------------------------------
# Collect input files
# ---------------------------------------------------------------------------

def collect_files(inputs: list[Path], out_dir: Path) -> list[tuple[Path, Path]]:
    """Return (absolute path, output-relative path) pairs. Hidden files,
    symlinks, and anything under out_dir are skipped."""
    files: list[tuple[Path, Path]] = []
    for inp in inputs:
        if inp.is_file():
            files.append((inp.resolve(), Path(inp.name)))
            continue
        for f in sorted(inp.rglob("*")):
            if not f.is_file() or f.is_symlink():
                continue
            rel = f.relative_to(inp)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if out_dir in f.resolve().parents:
                continue
            files.append((f.resolve(), Path(inp.name) / rel))
    return files


# ---------------------------------------------------------------------------
# Text extraction (all local; no file content on stdout)
# ---------------------------------------------------------------------------

def _looks_texty(path: Path) -> bool:
    return b"\0" not in path.open("rb").read(8192)


def _pdf_text(path: Path) -> tuple[str | None, str | None]:
    if shutil.which("pdftotext"):
        r = subprocess.run(["pdftotext", "-q", str(path), "-"],
                           capture_output=True, timeout=120)
        if r.returncode == 0:
            return r.stdout.decode("utf-8", errors="replace"), None
        return None, "pdftotext could not read this PDF"
    try:
        from pypdf import PdfReader
    except ImportError:
        return None, ("no PDF text extractor - install poppler-utils "
                      "(pdftotext) or 'pip install pypdf'")
    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages), None


def _docx_text(path: Path) -> tuple[str | None, str | None]:
    import html
    import re
    import zipfile
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    xml = re.sub(r"</w:p>", "\n", xml)
    return html.unescape(re.sub(r"<[^>]+>", "", xml)), None


def _image_text(path: Path) -> tuple[str | None, str | None]:
    if not shutil.which("tesseract"):
        return None, ("image - no OCR available (install tesseract-ocr); "
                      "original NOT copied")
    r = subprocess.run(["tesseract", str(path), "stdout"],
                       capture_output=True, timeout=120)
    if r.returncode != 0:
        return None, "OCR failed on this image; original NOT copied"
    return r.stdout.decode("utf-8", errors="replace"), None


def extract_text(path: Path) -> tuple[str | None, str, str | None]:
    """Return (text, kind, skip_reason). text is None when the file must be
    skipped; kind is one of text/pdf/docx/image/binary."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        text, err = _pdf_text(path)
        return text, "pdf", err
    if ext == ".docx":
        text, err = _docx_text(path)
        return text, "docx", err
    if ext in IMAGE_EXTS:
        text, err = _image_text(path)
        return text, "image", err
    if ext in TEXT_EXTS or (ext == "" and _looks_texty(path)):
        if not _looks_texty(path):
            return None, "binary", "binary content; original NOT copied"
        return (path.read_bytes().decode("utf-8", errors="replace"),
                "text", None)
    return None, "binary", "unsupported file type; original NOT copied"


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------

def extract_one(abs_path: Path, rel: Path, session: RedactionSession,
                llm: LemonadeClient, max_bytes: int) -> tuple[dict, str | None]:
    """Phase 1: extract text and ingest it into the session map. Returns the
    receipt entry plus the text (None when the file won't be written)."""
    entry = {"file": str(rel), "status": "", "kind": "", "output": None,
             "entities": {}, "note": None}
    try:
        text, kind, skip_reason = extract_text(abs_path)
        entry["kind"] = kind
        if text is None:
            entry.update(status="skipped", note=skip_reason)
            return entry, None
        if len(text) > max_bytes:
            entry.update(status="skipped",
                         note=f"larger than {max_bytes // 1_000_000} MB limit")
            return entry, None
        if not text.strip():
            entry.update(status="skipped",
                         note="no extractable text (scanned/empty?); "
                              "original NOT copied")
            return entry, None
        if not session.ingest(text, llm):
            entry.update(status="failed",
                         note="local redaction model did not answer; "
                              "file NOT copied (fail closed)")
            return entry, None
        return entry, text
    except Exception as exc:
        # Exception messages can echo file content - class name only.
        entry.update(status="failed",
                     note=f"error: {type(exc).__name__}; file NOT copied")
        return entry, None


def write_one(entry: dict, text: str, rel: Path, out_dir: Path,
              session: RedactionSession, taken: set[str]) -> None:
    """Phase 2, after the map is complete across ALL files: mask, scrub-check,
    and write. Output names are masked with full knowledge, so a name can't
    leak a value that only appeared inside another file's content."""
    try:
        masked = session.mask(text)
        leftovers = regex_discover(masked)
        if leftovers:
            types = sorted({t for _, t in leftovers})
            entry.update(status="withheld",
                         note=f"post-check found {len(leftovers)} unmasked "
                              f"{'/'.join(types)} pattern(s); file NOT copied")
            return

        out_rel = session.mask(str(rel))
        if entry["kind"] in ("pdf", "docx", "image"):
            out_rel += ".txt"       # we emit extracted text, not the original
        while out_rel in taken:
            p = Path(out_rel)
            out_rel = str(p.with_name(p.stem + "-dup" + p.suffix))
        taken.add(out_rel)

        dest = out_dir / out_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(masked, encoding="utf-8")
        entry.update(status="redacted", output=out_rel,
                     entities=session.counts_in(text))
        if entry["kind"] in ("pdf", "docx"):
            entry["note"] = "text extracted; layout not preserved"
        elif entry["kind"] == "image":
            entry["note"] = "OCR text only; the image itself is NOT copied"
    except Exception as exc:
        # Exception messages can echo file content - class name only.
        entry.update(status="failed",
                     note=f"error: {type(exc).__name__}; file NOT copied")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Redact copies of files locally before cloud analysis.")
    ap.add_argument("inputs", nargs="+", help="files and/or directories")
    ap.add_argument("-o", "--output", required=True,
                    help="directory for redacted copies (created if missing; "
                         "must be outside the inputs)")
    ap.add_argument("--lemonade-url", default=DEFAULT_LEMONADE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt-file", default=str(DEFAULT_PROMPT),
                    help="redaction system prompt (default: de-identification)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=32468)
    ap.add_argument("--max-file-mb", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.output).resolve()
    inputs = [Path(p) for p in args.inputs]
    for p in inputs:
        if not p.exists():
            fail(2, f"input not found: {p}")
        if p.is_dir() and (out_dir == p.resolve()
                           or p.resolve() in out_dir.parents):
            fail(2, "output dir must be OUTSIDE the input dir "
                    f"(got {out_dir} inside {p})")

    files = collect_files(inputs, out_dir)
    if not files:
        fail(2, "no files found under the given inputs")

    prompt_file = Path(args.prompt_file)
    if not prompt_file.exists():
        fail(2, f"prompt file not found: {prompt_file}")

    llm = LemonadeClient(args.lemonade_url, args.model,
                         prompt_file.read_text(encoding="utf-8").strip(),
                         max_tokens=args.max_tokens, retries=args.retries)

    # Preflight - fail closed BEFORE reading any file. No regex-only fallback.
    if not llm.health():
        fail(2, f"Lemonade Server not reachable at {args.lemonade_url}",
                "Start it with:  lemonade serve", "Then re-run this command.")
    installed = llm.installed_models()
    if installed is not None and args.model not in installed:
        fail(2, f"redaction model '{args.model}' is not installed in Lemonade",
                f"Pull it with:  lemonade pull {args.model}",
                "Then re-run this command.")
    say(f"preflight OK - Lemonade at {args.lemonade_url}, model {args.model}")

    session = RedactionSession()

    # Filenames leak identity too ("john-smith-labs.csv"): discover on the
    # path listing first so every printed path below is already masked.
    listing = "\n".join(str(rel) for _, rel in files)
    if not session.ingest(listing, llm):
        fail(2, "redaction model reachable but not answering usably; "
                "nothing was processed. Check Lemonade logs and retry.")

    out_dir.mkdir(parents=True, exist_ok=True)
    say(f"{len(files)} file(s) -> {out_dir}")

    # Phase 1: extract + ingest everything, so the placeholder map is complete
    # before any output name or content is fixed.
    max_bytes = args.max_file_mb * 1_000_000
    results, texts = [], []
    for a, r in files:
        entry, text = extract_one(a, r, session, llm, max_bytes)
        results.append(entry)
        texts.append(text)

    # Filenames echo identity in mangled form ("John Smith" in a CSV named
    # john-smith-labs.csv); register those variants so paths mask too.
    for _, rel in files:
        session.alias_variants_in(str(rel))

    # Phase 2: mask, scrub-check, write.
    taken: set[str] = set()
    for (a, r), entry, text in zip(files, results, texts):
        if text is not None:
            write_one(entry, text, r, out_dir, session, taken)

    # Receipt paths: masked with the complete map (idempotent on placeholders).
    for e in results:
        e["file"] = session.mask(e["file"])

    # Receipt - masked paths, statuses, counts. Never content, never values.
    for e in results:
        if e["status"] == "redacted":
            ents = ", ".join(f"{n} {t}" for t, n in sorted(e["entities"].items()))
            extra = f" - {e['note']}" if e["note"] else ""
            say(f"REDACTED  {e['output']}  ({ents or 'no entities found'}){extra}")
        else:
            say(f"{e['status'].upper():9} {e['file']} - {e['note']}")

    n = {s: sum(1 for e in results if e["status"] == s)
         for s in ("redacted", "skipped", "withheld", "failed")}
    totals: dict[str, int] = {}
    for e in results:
        for t, c in e["entities"].items():
            totals[t] = totals.get(t, 0) + c
    say("-" * 60)
    say(f"{n['redacted']} redacted, {n['skipped']} skipped, "
        f"{n['withheld']} withheld, {n['failed']} failed; "
        f"{sum(totals.values())} entities masked"
        + (" (" + ", ".join(f"{t} {c}" for t, c in sorted(totals.items())) + ")"
           if totals else ""))

    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "tool": "local-ai-privacy redact.py",
        "model": args.model,
        "files": results,
        "totals": {**n, "entities": totals},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    say(f"manifest: {out_dir / 'manifest.json'}")
    masked_inputs = " ".join(f"'{session.mask(str(p))}'" for p in inputs)
    say(f"verify locally (never via the assistant): diff -r {masked_inputs} "
        f"'{out_dir}'")

    if n["redacted"] == 0:
        fail(3, "nothing could be redacted - do NOT analyze the originals")
    if n["withheld"] or n["failed"]:
        fail(1, "PARTIAL: analyze ONLY the redacted copies above; report the "
                "withheld/failed files to the user")
    sys.exit(0)


if __name__ == "__main__":
    main()
