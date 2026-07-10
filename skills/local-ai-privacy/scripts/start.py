#!/usr/bin/env python3
"""
local-ai-privacy: one-shot setup.

Run this once to enable transparent redaction for all future Claude Code sessions:

  1. Reads the current ANTHROPIC_BASE_URL from ~/.claude/settings.json
  2. Saves it to ~/.claude/skills/local-ai-privacy/proxy.conf so the proxy
     knows where to forward clean requests.
  3. Starts proxy.py as a windowless background process (survives terminal close).
  4. Patches ANTHROPIC_BASE_URL in settings.json to point at the proxy.

After this script exits, restart Claude Code once.
Every future session will automatically flow through the proxy — no user action needed.

To undo everything: python stop.py

Requirements: Python 3.10+, requests
"""

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SKILL_DIR   = Path(__file__).resolve().parent.parent
PROXY_PY    = SKILL_DIR / "scripts" / "proxy.py"
CONF_FILE   = Path.home() / ".claude" / "skills" / "local-ai-privacy" / "proxy.conf"
SETTINGS_FILE = Path.home() / ".claude" / "settings.json"

PROXY_PORT  = 8080
PROXY_HOST  = "127.0.0.1"
PROXY_URL   = f"http://{PROXY_HOST}:{PROXY_PORT}"

LEMONADE_URL    = "http://localhost:13305"
REDACTION_MODEL = "Qwen3-1.7B-GGUF"


def _print(msg: str) -> None:
    print(f"[local-ai-privacy] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def check_lemonade() -> None:
    url = LEMONADE_URL + "/api/v1/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            if r.status != 200:
                raise RuntimeError(f"status {r.status}")
    except Exception as exc:
        _print(f"FAIL: Lemonade Server not reachable at {url}: {exc}")
        _print("Start it with:  lemonade serve")
        sys.exit(1)
    _print(f"Lemonade Server reachable at {LEMONADE_URL}")


def check_redaction_model() -> None:
    url = LEMONADE_URL + "/api/v1/models?show_all=true"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        ids = [m["id"] for m in data.get("data", [])]
        if REDACTION_MODEL not in ids:
            _print(f"FAIL: Redaction model '{REDACTION_MODEL}' not found in Lemonade.")
            _print(f"Pull it with:  lemonade pull {REDACTION_MODEL}")
            sys.exit(1)
    except Exception as exc:
        _print(f"WARNING: Could not verify model list ({exc}) — proceeding anyway.")
    _print(f"Redaction model {REDACTION_MODEL} available")


def check_requests() -> None:
    try:
        import requests  # noqa: F401
    except ImportError:
        _print("FAIL: 'requests' package not found.")
        _print("Install with:  pip install requests")
        sys.exit(1)


# ---------------------------------------------------------------------------
# settings.json helpers
# ---------------------------------------------------------------------------

def read_settings() -> dict:
    if not SETTINGS_FILE.exists():
        return {}
    return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))


def write_settings(data: dict) -> None:
    SETTINGS_FILE.write_text(
        json.dumps(data, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def get_current_base_url(settings: dict) -> str:
    return settings.get("env", {}).get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")


# ---------------------------------------------------------------------------
# proxy.conf
# ---------------------------------------------------------------------------

def write_conf(cloud_url: str) -> None:
    CONF_FILE.parent.mkdir(parents=True, exist_ok=True)
    conf = {
        "cloud_url":       cloud_url,
        "lemonade_url":    LEMONADE_URL,
        "redaction_model": REDACTION_MODEL,
        "proxy_port":      PROXY_PORT,
    }
    CONF_FILE.write_text(json.dumps(conf, indent=2), encoding="utf-8")
    _print(f"Saved proxy config to {CONF_FILE}")


# ---------------------------------------------------------------------------
# Start proxy as a background process
# ---------------------------------------------------------------------------

def proxy_is_running() -> bool:
    try:
        with urllib.request.urlopen(
            f"{PROXY_URL}/__proxy_health", timeout=2
        ) as r:
            return r.status == 200
    except Exception:
        return False


def start_proxy() -> None:
    if proxy_is_running():
        _print("Proxy already running — skipping launch")
        return

    # pythonw.exe on Windows runs without a console window and survives terminal close.
    # On Linux/macOS use python with nohup.
    python_exe = sys.executable
    if platform.system() == "Windows":
        pythonw = Path(python_exe).parent / "pythonw.exe"
        if pythonw.exists():
            python_exe = str(pythonw)

    cmd = [python_exe, str(PROXY_PY), "--conf", str(CONF_FILE)]

    log_file = CONF_FILE.parent / "proxy.log"

    if platform.system() == "Windows":
        # DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP: survives parent exit
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf, stderr=lf,
                creationflags=flags,
                close_fds=True,
            )
    else:
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf, stderr=lf,
                start_new_session=True,
                close_fds=True,
            )

    # Write PID so stop.py can kill it later
    pid_file = CONF_FILE.parent / "proxy.pid"
    pid_file.write_text(str(proc.pid), encoding="utf-8")
    _print(f"Proxy started (PID {proc.pid}), log: {log_file}")

    # Wait up to 10 s for the proxy to become reachable
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if proxy_is_running():
            _print(f"Proxy ready at {PROXY_URL}")
            return
        time.sleep(0.5)

    _print("WARNING: Proxy did not respond within 10 s — check the log:")
    _print(f"  {log_file}")


# ---------------------------------------------------------------------------
# Patch settings.json
# ---------------------------------------------------------------------------

def patch_settings(settings: dict) -> None:
    env = settings.setdefault("env", {})
    if env.get("ANTHROPIC_BASE_URL") == PROXY_URL:
        _print("settings.json already points at the proxy — no change")
        return
    env["ANTHROPIC_BASE_URL"] = PROXY_URL
    write_settings(settings)
    _print(f"Patched settings.json: ANTHROPIC_BASE_URL → {PROXY_URL}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _print("Starting local-ai-privacy setup...")

    check_requests()
    check_lemonade()
    check_redaction_model()

    settings  = read_settings()
    cloud_url = get_current_base_url(settings)

    if cloud_url == PROXY_URL:
        # Already patched from a previous run — just make sure proxy is alive
        _print("settings.json already points at proxy.")
        conf = json.loads(CONF_FILE.read_text()) if CONF_FILE.exists() else {}
        cloud_url = conf.get("cloud_url", "https://api.anthropic.com")
    else:
        _print(f"Cloud endpoint: {cloud_url}")

    write_conf(cloud_url)
    start_proxy()
    patch_settings(settings)

    print()
    _print("Setup complete. Now:")
    _print("  1. Restart Claude Code (close and reopen, or press Ctrl+C then run 'claude' again)")
    _print("  2. Every prompt you type will be redacted locally before reaching the cloud")
    _print("  3. To stop: python stop.py  (restores your original cloud connection)")
    print()


if __name__ == "__main__":
    main()
