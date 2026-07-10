#!/usr/bin/env python3
"""
local-ai-privacy: teardown.

Undoes everything start.py did:
  1. Kills the background proxy process.
  2. Restores ANTHROPIC_BASE_URL in ~/.claude/settings.json to the original cloud URL.

Run this when you no longer want redaction.
"""

from __future__ import annotations

import json
import os
import platform
import signal
import sys
from pathlib import Path

CONF_DIR      = Path.home() / ".claude" / "skills" / "local-ai-privacy"
CONF_FILE     = CONF_DIR / "proxy.conf"
PID_FILE      = CONF_DIR / "proxy.pid"
SETTINGS_FILE = Path.home() / ".claude" / "settings.json"
PROXY_URL     = "http://127.0.0.1:8080"


def _print(msg: str) -> None:
    print(f"[local-ai-privacy] {msg}", flush=True)


def kill_proxy() -> None:
    if not PID_FILE.exists():
        _print("No proxy PID file found — proxy may not be running.")
        return
    pid = int(PID_FILE.read_text().strip())
    try:
        if platform.system() == "Windows":
            os.kill(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
        _print(f"Stopped proxy (PID {pid})")
    except ProcessLookupError:
        _print(f"Proxy process {pid} was not running.")
    except PermissionError:
        _print(f"Could not stop proxy (PID {pid}) — try killing it manually.")
    PID_FILE.unlink(missing_ok=True)


def restore_settings() -> None:
    if not SETTINGS_FILE.exists():
        _print("settings.json not found — nothing to restore.")
        return

    settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    env = settings.get("env", {})

    if env.get("ANTHROPIC_BASE_URL") != PROXY_URL:
        _print("settings.json does not point at the proxy — nothing to restore.")
        return

    # Get original cloud URL from conf file
    original_url = None
    if CONF_FILE.exists():
        conf = json.loads(CONF_FILE.read_text(encoding="utf-8"))
        original_url = conf.get("cloud_url")

    if original_url:
        env["ANTHROPIC_BASE_URL"] = original_url
        _print(f"Restored ANTHROPIC_BASE_URL → {original_url}")
    else:
        # Fall back: remove the key entirely so Claude Code uses its default
        env.pop("ANTHROPIC_BASE_URL", None)
        _print("Removed ANTHROPIC_BASE_URL from settings.json (using Claude Code default)")

    settings["env"] = env
    SETTINGS_FILE.write_text(
        json.dumps(settings, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    _print("Stopping local-ai-privacy...")
    kill_proxy()
    restore_settings()
    _print("Done. Restart Claude Code to connect directly to the cloud again.")


if __name__ == "__main__":
    main()
