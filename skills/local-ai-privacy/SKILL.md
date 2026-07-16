---
name: local-ai-privacy
description: >-
  Redacts sensitive data locally (via a Lemonade model) before it reaches the
  cloud LLM. Use when the user wants files or folders containing PII, patient
  or health data, customer records, credentials, or confidential documents
  analyzed by Claude but with personal data stripped first; mentions HIPAA,
  GDPR, de-identification, or data confidentiality; or asks to "redact before
  sending to cloud", "mask sensitive data", "keep my data local", or "protect
  my privacy with Claude Code". A local script writes redacted copies of the
  files; the cloud model only ever reads the copies.
  Do not use for image generation, TTS, or STT - use local-ai-use instead.
allowed-tools: Bash(lemonade:*), Bash(python:*), Bash(python3:*)
---

# Local AI Privacy (local redaction gate)

The user wants files analyzed WITHOUT their sensitive data leaving the
machine. Anthropic receives only what enters this conversation - so the
guarantee is simple: **raw file content must never appear in any tool input
or tool output.** A local script (`redact.py`) opens the originals, redacts
them with a local Lemonade model plus deterministic regex detectors, and
writes masked copies. You work exclusively with the copies.

```
user's files ──► redact.py (local only) ──► redacted copies ──► you ──► insights
 (originals)     regex + local LLM          [NAME_1], [SSN_1]   read copies only
      ▲
      └── NEVER opened by you. Only redact.py touches these.
```

## Hard rules

1. **NEVER read OR list the originals.** No Read, Grep, cat, head - and no
   `ls`, Glob, or tree on the input folder either: filenames are PII too
   (`margaret-walsh-bloodwork.csv`), and a directory listing puts them in
   your context unmasked. Don't inspect "what's in there" or "check the
   formats" - `redact.py` classifies file types itself, and its receipt is
   your only source of truth about the inputs. Run it knowing nothing but
   the path the user gave you.
2. **Fail closed.** If `redact.py` exits 2 or 3, stop and help the user fix
   the problem it printed. Never analyze the originals as a fallback, and
   never offer to.
3. **Partial results (exit 1):** analyze only what was written to the output
   dir, and lead your final answer with which files were withheld or skipped
   and why.
4. Placeholders like `[NAME_1]`, `[SSN_2]` are opaque. Copy them verbatim;
   never guess what is behind one.
5. If the user pasted sensitive values directly into their prompt, tell them
   that text already reached the cloud before any skill could run. Only file
   contents are protected by this flow - advise them to refer to files by
   path and never paste contents.

## Flow

### 0. Set expectations (one sentence, before any tool call)

Tell the user what they are about to see, so permission prompts inform
instead of alarm. Something like:

> Here's what will happen: first I'll lock in a permission rule so future
> sessions are blocked from reading your originals at all, then a local
> script redacts them - it's the only thing that touches your originals.
> After it finishes I'll only read files under `…-redacted`. If any
> permission prompt ever asks to read or list your original folder, deny
> it.

### 1. Lock in the guardrails (before touching any data)

`<INPUT>` is the path the user gave (files or folders, anywhere on disk);
the output dir is always `<INPUT>-redacted`, a sibling of the input (never
inside it). **If the prompt doesn't contain a concrete path, ask the user
for it - NEVER search for it.** A `find`/`ls`/Glob hunt for "the health
data folder" prints unmasked folder and file names into the conversation,
which is the exact leak this skill exists to prevent. Same rule when a
given path turns out not to exist (harden.py will tell you): report the
error verbatim and ask, don't explore.

Then, as your FIRST tool call:

```bash
python3 ~/.claude/skills/local-ai-privacy/scripts/harden.py <INPUT> <INPUT>-redacted
```

(If this skill is installed somewhere other than `~/.claude/skills` — e.g.
a project's `.claude/skills` — use that path instead. Never locate the
scripts by searching.)

It merges two rules into `~/.claude/settings.json` - deny `Read` on the
originals, allow `Read` on the copies - idempotently, with a backup,
effective from the next session. Running it first means that even if the
redaction below fails and the user retries in a fresh session, that session
already has the mechanical block in place. Each new workspace gets its own
rule pair. If harden.py itself fails (e.g. a malformed settings.json),
report it, continue with the flow - the hard rules above still protect this
run - and tell the user to fix settings.json. Never edit settings.json
yourself.

### 2. Run the gate

```bash
python3 ~/.claude/skills/local-ai-privacy/scripts/redact.py <INPUT> -o <INPUT>-redacted
```

The script checks its own prerequisites first and exits 2 with the fix if
they are missing:
- Lemonade not running → user runs `lemonade serve`
- model missing → `lemonade pull Qwen3.6-35B-A3B-NoThinking` (one-time,
  ~18 GB - confirm with the user before pulling)

Then re-run the same command. Everything it prints is already masked - file
names included.

### 3. Analyze the copies

Read files under the output dir only (skip `manifest.json` unless you need
per-file status). Text extracted from PDFs/DOCX/images appears as `.txt`
files. Do the analysis the user asked for.

### 4. Close with the redaction receipt

End your answer with a short receipt taken from the script output: how many
files were redacted, entity counts by type (e.g. `NAME 8, SSN 2, DOB 1`),
any skipped/withheld files and what the user should do about them, what
`harden.py` locked in, and the local verify command the script printed
(`diff -r <originals> <copies>`) so the user can confirm the masking with
their own eyes - never offer to run that diff yourself.

## Zero-trust variant (mention when the data is especially sensitive)

The user can run the exact same `redact.py` command in their own terminal
*before* ever prompting, then share only the output dir. No model of any
kind sees the originals during redaction.

## Reference

CLI flags, file-type handling, exit codes, manifest schema, the model
picker, and troubleshooting: [reference.md](reference.md).
