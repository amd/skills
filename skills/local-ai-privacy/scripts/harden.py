#!/usr/bin/env python3
"""
local-ai-privacy: lock in permission rules for a redaction workspace.

  python3 harden.py <originals_path> <redacted_output_path>

Adds two rules to ~/.claude/settings.json:
  deny  Read(<originals>/**)  — the harness mechanically blocks reads of the
                                originals, independent of model behavior
  allow Read(<redacted>/**)   — reading the masked copies stops prompting

Run automatically by the skill after a successful redaction. Idempotent:
existing rules are never duplicated, other settings are preserved, the file
is backed up to settings.json.bak before writing, and nothing is written
unless the result re-parses as valid JSON (a malformed settings.json would
silently disable ALL user settings). Rules take effect from the NEXT
Claude Code session — permissions are snapshotted at session start.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def rule_path(p: Path) -> str:
    """Permission-rule form of a path: ~/... when under home, //... otherwise."""
    p = p.expanduser().resolve()
    try:
        return "~/" + p.relative_to(Path.home()).as_posix()
    except ValueError:
        return "//" + p.as_posix().lstrip("/")


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: harden.py <originals_path> <redacted_output_path>")
        sys.exit(2)
    originals, redacted = Path(sys.argv[1]), Path(sys.argv[2])
    if not originals.exists():
        print(f"[harden] originals path not found: {originals}")
        sys.exit(2)

    deny_rule = f"Read({rule_path(originals)}/**)"
    allow_rule = f"Read({rule_path(redacted)}/**)"

    settings_file = Path.home() / ".claude" / "settings.json"
    try:
        settings = (json.loads(settings_file.read_text(encoding="utf-8"))
                    if settings_file.exists() else {})
    except json.JSONDecodeError:
        print(f"[harden] {settings_file} is not valid JSON — fix it first; "
              "nothing was changed")
        sys.exit(1)

    perms = settings.setdefault("permissions", {})
    changed = []
    for key, rule in (("deny", deny_rule), ("allow", allow_rule)):
        rules = perms.setdefault(key, [])
        if rule not in rules:
            rules.append(rule)
            changed.append(f"{key}  {rule}")

    if not changed:
        print("[harden] rules already present — nothing to do")
        return

    text = json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
    json.loads(text)  # refuse to write anything that doesn't re-parse
    if settings_file.exists():
        settings_file.with_suffix(".json.bak").write_text(
            settings_file.read_text(encoding="utf-8"), encoding="utf-8")
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    settings_file.write_text(text, encoding="utf-8")

    for c in changed:
        print(f"[harden] added {c}")


if __name__ == "__main__":
    main()
