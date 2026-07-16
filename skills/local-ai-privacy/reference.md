# Local AI Privacy: Reference

One flow, three scripts, one shared engine:

| File | Role |
|---|---|
| `scripts/redact.py` | The redaction gate - the only thing that ever opens the originals |
| `scripts/harden.py` | Locks deny/allow permission rules into `~/.claude/settings.json` |
| `scripts/redaction_core.py` | Shared engine: placeholder map, regex detectors, local-LLM client |
| `data/deidentify-prompt.txt` | Default prompt: hide *who*, keep *what* |
| `data/redaction-prompt.txt` | Strict prompt: flag anything sensitive at all |

## Contents

- [redact.py: usage and flags](#redactpy-usage-and-flags)
- [What gets detected](#what-gets-detected)
- [Placeholder format](#placeholder-format)
- [File-type handling](#file-type-handling)
- [Exit codes](#exit-codes)
- [Manifest schema](#manifest-schema)
- [harden.py and hardening](#hardenpy-and-hardening)
- [Model picker](#model-picker)
- [Limitations](#limitations)

---

## redact.py: usage and flags

Reads the ORIGINAL files, redacts them locally, writes masked copies to an
output directory, and prints a receipt. Its stdout is part of the trust
boundary: only masked paths, statuses, and entity **type counts** are
printed - never file content, never raw values, and error messages show
exception class names only (a message could echo content).

```bash
python3 redact.py INPUT [INPUT ...] -o OUTPUT_DIR
```

| Flag | Default | Meaning |
|---|---|---|
| `-o, --output` | (required) | Directory for redacted copies. Must be outside the inputs. |
| `--lemonade-url` | `http://localhost:13305` | Lemonade server |
| `--model` | `Qwen3.6-35B-A3B-NoThinking` | Redaction model (see [Model picker](#model-picker)) |
| `--prompt-file` | `data/deidentify-prompt.txt` | Redaction system prompt |
| `--retries` | `3` | LLM retries per chunk before failing closed |
| `--max-tokens` | `8192` | Local model reply cap (raise for thinking models) |
| `--max-file-mb` | `20` | Files larger than this are skipped |

Hidden files, symlinks, and anything under the output dir are skipped.
`LEMONADE_API_KEY` is honored if set. No third-party Python packages are
required.

**The prompt matters:** the default `deidentify-prompt.txt` redacts *who*
(names, contacts, IDs, DOB, facility names, secrets) but keeps *what* (lab
values, medications, diagnoses, diet, visit dates), so the copies stay
analyzable. For maximum-strictness redaction of anything sensitive at all -
including clinical content, proprietary code, and business figures - pass
`--prompt-file data/redaction-prompt.txt`. Both are plain text; edit them
to widen or narrow detection, no code change needed.

## What gets detected

Three passes feed one placeholder map (same value → same placeholder across
all files in a run):

1. **Deterministic regex** - SSN, email, phone, credit card (Luhn-validated),
   IPv4, `MRN: <value>`, `DOB: <value>`. These can't be missed by a flaky
   model, and they double as the **post-redaction scrub check**: every output
   is re-scanned, and a file with any remaining hit is withheld instead of
   written.
2. **Local LLM** (open-ended) - names, addresses, relatives, facility names,
   freeform secrets; anything the prompt describes, in 8,000-char chunks.
3. **Name-token aliasing** (deterministic) - once a person-typed value like
   "Margaret Walsh" or "Dr. Anita Krishnan" is in the map, its component
   words ("Margaret", "Walsh", "Krishnan") and initials ("MW", "AK") are
   masked everywhere at word boundaries, mapped to the same placeholder. The
   LLM reliably flags a person in structured fields ("Patient:", "From:")
   but can miss casual prose mentions ("Hi Margaret", a "- AK" sign-off);
   this pass makes every later mention deterministic once the person has
   been found once.

Filenames are ingested too: `john-smith-labs.csv` becomes
`[NAME_1]-labs.csv` in the output dir, the receipt, and the manifest, so
identity can't leak through paths.

## Placeholder format

`[TYPE_N]` - the entity category in caps plus a 1-based counter unique per
type (`[NAME_1]`, `[SSN_2]`). Placeholders are assigned by the engine, not
the model, and are stable for the whole run: the same real value always gets
the same placeholder in every file. The engine refuses to re-redact its own
tokens, so placeholders never nest (`[SSN_1]` → `[SSN_2]` → …). The
value↔placeholder map lives only in RAM and is discarded when the script
exits - nothing sensitive is ever written to disk.

## File-type handling

| Type | Handling |
|---|---|
| `.txt .md .csv .tsv .json .xml .yaml .html .log .ini .toml`, extensionless text | Redacted as text |
| `.pdf` | Text extracted via `pdftotext` (poppler) or `pypdf`, redacted, written as `<name>.pdf.txt`. No extractor installed, or no text layer → **skipped, never copied** |
| `.docx` | Text extracted from the XML, written as `<name>.docx.txt` |
| Images (`.png .jpg .tiff` …) | OCR via `tesseract` if installed, redacted text written as `<name>.<ext>.txt`. The image itself is **never copied**. No tesseract → skipped |
| Other binaries | Skipped, never copied |

## Exit codes

| Code | Meaning | Claude's obligation |
|---|---|---|
| `0` | Everything redacted (unsupported files may be skipped - see receipt) | Analyze the output dir |
| `1` | Partial: some files withheld (post-check hit) or failed (LLM unusable) | Analyze only what was written; lead with the withheld/failed list |
| `2` | Preflight failed (Lemonade down, model missing, bad paths); nothing processed | Surface the printed fix; **never** touch the originals |
| `3` | Ran, but nothing could be redacted | Same as 2 |

## Manifest schema

`OUTPUT_DIR/manifest.json` - all paths masked, no raw values anywhere:

```json
{
  "created": "2026-07-14T12:00:00",
  "tool": "local-ai-privacy redact.py",
  "model": "Qwen3.6-35B-A3B-NoThinking",
  "files": [
    {"file": "health-data/[NAME_1]-labs.csv", "status": "redacted",
     "kind": "text", "output": "health-data/[NAME_1]-labs.csv",
     "entities": {"NAME": 2, "SSN": 1}, "note": null}
  ],
  "totals": {"redacted": 4, "skipped": 1, "withheld": 0, "failed": 0,
             "entities": {"NAME": 8, "SSN": 2}}
}
```

`status` is one of `redacted | skipped | withheld | failed`.

## harden.py and hardening

- **Automatic (harden.py):** the skill flow runs
  `harden.py <originals> <output-dir>` as its FIRST tool call, before the
  gate ever opens a file — so even if redaction fails and the user retries
  in a fresh session, that session already has the block in place. It
  merges two rules into `~/.claude/settings.json`: **deny** `Read` on the
  originals (the harness then mechanically blocks reads, independent of
  model behavior) and **allow** `Read` on the copies (no more prompts for
  them). Idempotent, preserves all other settings, writes
  `settings.json.bak` first, refuses to write anything that doesn't
  re-parse, and never touches a malformed settings file. Paths under home
  become `~/...` rules; others `//abs` rules. Effective from the **next**
  session - permissions snapshot at session start. Each protected folder
  accumulates its own rule pair.
- **Zero-trust:** run the `redact.py` command in your own terminal *before*
  prompting Claude, then share only the output dir. No model of any kind is
  in the loop while the originals are open.
- **Verify:** `diff -r <originals> <output-dir>` locally - the receipt
  prints the exact command. Redacted copies contain only `[TYPE_N]`
  placeholders, so reviewing them is safe.
- **Residual gap:** a `Read` deny rule does not cover `cat`/`python3 -c`
  through Bash. The skill's hard rules forbid that, and Bash prompts make
  it visible, but the mechanically airtight option remains the zero-trust
  variant.

## Model picker

| Model | Approx size | Notes |
|---|---|---|
| `Qwen3.6-35B-A3B-NoThinking` | ~18 GB | **Default.** 35B MoE with ~3B active params - strong PII coverage at near-small-model speed, and no reasoning overhead so it returns clean JSON. |
| `Qwen3.6-35B-A3B-ThinkingCoder` | ~18 GB | Same base, but emits `<think>` reasoning first. Higher latency and needs a larger `--max-tokens`; usually not worth it for extraction. |
| `Qwen3-1.7B-GGUF` | ~1.1 GB | Lightweight fallback. Fastest and lowest RAM, but weaker coverage of unusual/edge-case PII. The regex pass and post-check still backstop the structured kinds. |

Switch with `--model NAME` (or `LOCALAI_REDACTION_MODEL` is honored as a
default by older setups). Pull first if needed: `lemonade pull <model>`.
For a thinking model, also raise `--max-tokens` (e.g. `16384`) so reasoning
can't truncate the JSON.

## Limitations

- **Prompt text is NOT protected** - only file contents. The user must refer
  to files by path, never paste contents into the prompt.
- The "never read or list the originals" rule is behavioral (SKILL.md), not
  mechanical: a directory listing of the input folder would put unmasked
  filenames into the conversation. The harden.py deny rule makes reads
  mechanically impossible from the next session onward; approve Bash
  commands touching the input path only if they are the `redact.py` or
  `harden.py` invocations.
- Visual redaction of images is not attempted: images are OCR'd (text only)
  or skipped; originals are never copied either way.
- Ordinary dates, ages, and state/country names are kept by default (needed
  for trend analysis); a determined re-identifier can correlate them. Edit
  `data/deidentify-prompt.txt` to tighten.
- Secrets in base64/obfuscated encodings are not detected.
- The LLM pass can miss unusual identifiers; the regex post-check only
  guarantees the *structured* kinds never slip through. Name-token aliasing
  closes prose mentions of people the LLM found at least once, but a person
  it never flags anywhere still slips, as do all-lowercase prose mentions of
  a name ("margaret said…"). The `diff -r` verify step exists precisely so
  the user can eyeball the rest.
- Token aliasing is deliberately over-eager: a surname that is also a common
  capitalized word (a patient named June, a Dr. Park) will be masked in
  every context, including dates and places. That is the safe failure
  direction, but it can degrade analysis of those spans.
- For strict compliance (HIPAA safe harbor, PCI-DSS), layer a dedicated
  deterministic scanner on top.
