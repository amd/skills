#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""One-shot setup for the `local-ai-use` skill.

Performs the setup steps from SKILL.md:

  1. Ensures modern Lemonade is installed and its background service (the
     `lemond` daemon) is reachable on http://localhost:13305 (override with
     --host / --port or LEMONADE_HOST / LEMONADE_PORT). If no modern
     `lemonade` CLI is found, the latest version is installed on the user's
     behalf. The daemon auto-starts on install and is managed by the OS
     service manager, so this script never runs a `serve` command; it waits
     for the service to come up and, if it does not, prints the OS-specific
     start command and exits non-zero.
  2. Writes the routing rule from `templates/local-ai-rule.md` into
     <workspace>/AGENTS.md, between stable BEGIN/END markers so re-runs
     replace the block in place rather than appending.

Modern Lemonade (v10.1.0+) unified everything under the single `lemonade`
CLI (subcommands `status`, `pull`, `run`, ...) driving an always-on `lemond`
service. Older/incompatible builds of `lemonade` -- whichever way they were
installed (an old .msi/.deb, or the pip `lemonade-sdk` package) -- lack the
service-control subcommands and are NOT supported here. The CLI check below
therefore probes *capability* (does `lemonade status` work?) rather than
trusting the binary name, so an old incompatible CLI on PATH is reported
clearly instead of being driven into a 90-second dead end.

Setup never downloads models: the default image/TTS/STT models are pulled
on first use, by the installed AGENTS.md rule (see its failure
handling). This keeps setup fast and offline-friendly.

The script is idempotent: a second run on a fully configured workspace only
re-runs the healthcheck. It exits non-zero on any unrecoverable failure.
Pass --no-install to refuse the automatic install (it then just reports the
missing CLI and exits non-zero, the old behaviour).

Constants are documented inline; nothing is magical.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# Defaults match the system-wide Lemonade Server install. Both the CLI
# (LEMONADE_HOST / LEMONADE_PORT) and the OpenAI-compatible HTTP endpoints
# bind to these by default.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 13305

# The Lite Collection from Lemonade OmniRouter. Picked because each default
# fits in under ~5 GB and runs on commodity CPU hardware, so the savings vs.
# cloud calls are real on a typical developer laptop. See SKILL.md for upgrade
# paths.
DEFAULT_IMAGE_MODEL = "SD-Turbo"
DEFAULT_TTS_MODEL = "kokoro-v1"
DEFAULT_STT_MODEL = "Whisper-Tiny"

# Stable markers around the rule block in AGENTS.md. The script rewrites the
# region between these markers in place; do not change the marker strings or
# every existing AGENTS.md will get a duplicate block on the next run.
BEGIN_MARKER = "<!-- BEGIN amd-skills:local-ai-use -->"
END_MARKER = "<!-- END amd-skills:local-ai-use -->"

SKILL_DIR = Path(__file__).resolve().parent.parent
RULE_TEMPLATE = SKILL_DIR / "templates" / "local-ai-rule.md"

# The *full* Windows installer: Lemonade plus the desktop app and the
# always-on `lemond` service. `releases/latest/download/<asset>` always
# resolves to the newest published asset of that exact name, so we never have
# to pin a version.
WINDOWS_MSI_URL = (
    "https://github.com/lemonade-sdk/lemonade/releases/latest/download/lemonade.msi"
)
# Default per-user install location used by lemonade.msi. The CLI is added to
# the *user* PATH in the registry, which the current process will not see, so
# we also probe this tree directly after installing.
WINDOWS_INSTALL_DIR = Path(
    os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
) / "lemonade_server"

# GitHub release metadata, used to resolve the versioned macOS .pkg asset
# (its filename embeds the version, so there is no stable latest/download URL).
GITHUB_LATEST_RELEASE_API = (
    "https://api.github.com/repos/lemonade-sdk/lemonade/releases/latest"
)

# Ubuntu/Debian install: the stable PPA. The apt package is named
# `lemonade-server` (the CLI you then run is `lemonade`); installing it pulls
# in the `lemond` service, which the OS auto-starts. Run as a single shell
# pipeline so one sudo prompt covers the whole thing.
LINUX_APT_INSTALL = (
    "sudo add-apt-repository -y ppa:lemonade-team/stable && "
    "sudo apt-get update && "
    "sudo apt-get install -y lemonade-server"
)

# Modern Lemonade exposes a single `lemonade` CLI (it drives the always-on
# `lemond` service). We deliberately do NOT fall back to the deprecated
# `lemonade-server` CLI or the pip "eval" build; instead we verify the CLI we
# find is actually the modern one via a capability probe (see find_cli).
CLI_NAME = "lemonade"

# Docs URL to point users at when they must install/upgrade Lemonade by hand.
INSTALL_DOCS_URL = "https://lemonade-server.ai/docs/guide/install/"


def _default_workspace() -> Path:
    """Workspace root for AGENTS.md.

    Defaults to cwd, but if launched from inside an agent's skill folder
    (the universal `<.dot-config>/skills/<skill>/` layout used by Claude,
    Cursor, Codex, Gemini, etc.), climb out to the real workspace root so
    AGENTS.md is never buried inside the skill folder.
    """
    cwd = Path.cwd().resolve()
    for parent in cwd.parents:
        if parent.name == "skills" and parent.parent.name.startswith("."):
            return parent.parent.parent
    return cwd


def _print(msg: str) -> None:
    """Single-line, prefix-tagged status print so the agent's output stays parseable."""
    print(f"[local-ai-use] {msg}", flush=True)


def _http_get(url: str, timeout_s: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout_s) as r:  # noqa: S310
        return r.status, r.read()


def _candidate_clis() -> list[str]:
    """Every `lemonade` executable we can find, PATH first.

    On Windows the MSI updates the *user* PATH in the registry, which the
    current process will not have inherited, so we also probe the default
    per-user install tree for the executable.
    """
    candidates: list[str] = []
    found = shutil.which(CLI_NAME)
    if found:
        candidates.append(found)
    if platform.system() == "Windows" and WINDOWS_INSTALL_DIR.exists():
        for exe in WINDOWS_INSTALL_DIR.rglob(f"{CLI_NAME}.exe"):
            exe_str = str(exe)
            if exe_str not in candidates:
                candidates.append(exe_str)
    return candidates


def is_modern_cli(cli: str) -> bool:
    """True if `cli` is the modern Lemonade CLI (drives the `lemond` service).

    Capability probe, not a name check: the modern `lemonade` CLI exposes a
    `status` subcommand that reports on the service. An older or otherwise
    incompatible `lemonade` on PATH -- regardless of how it was installed (an
    old .msi/.deb, or the pip `lemonade-sdk` build) -- does not recognise the
    subcommand and errors with an argparse "invalid choice" instead. Running
    `lemonade status` is cheap, does not mutate anything, and tells us both
    that the CLI is modern AND whether the service is already up, so we reuse
    it as the single discriminator.

    A modern CLI prints "Server is running..." or "Server is not running"
    (exiting 0 or 1 accordingly). We key off that phrasing rather than the
    exit code alone.
    """
    try:
        result = subprocess.run(
            [cli, "status"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    blob = f"{result.stdout}\n{result.stderr}".lower()
    # Modern `lemonade status` always reports on the server, whether or not it
    # is running. An old/incompatible CLI never prints this phrasing (it errors
    # with "invalid choice: 'status'"), so the presence of the phrase is a
    # positive, unambiguous signal that this is the modern CLI.
    return "server is running" in blob or "server is not running" in blob


def find_cli() -> tuple[str | None, str | None]:
    """Locate a Lemonade CLI and classify it.

    Returns ``(modern_cli, stale_cli)``:

    - ``(path, None)``  a modern, capable `lemonade` CLI was found -> use it.
    - ``(None, path)``  a `lemonade` executable exists but is NOT the modern
                        CLI (old pip "eval" build, or otherwise incompatible)
                        -> the caller should guide the user to upgrade.
    - ``(None, None)``  no `lemonade` executable found at all -> install it.
    """
    stale: str | None = None
    for cli in _candidate_clis():
        if is_modern_cli(cli):
            return cli, None
        stale = stale or cli
    return None, stale


def install_lemonade() -> None:
    """Install the latest version of Lemonade for the current OS.

    Raises RuntimeError on any unrecoverable failure so the caller can report
    a clean message and fall back to the manual install link.
    """
    system = platform.system()
    if system == "Windows":
        _install_windows()
    elif system == "Linux":
        _install_linux()
    elif system == "Darwin":
        _install_macos()
    else:
        raise RuntimeError(
            f"No automatic installer for this OS ({system}). "
            f"Install manually: {INSTALL_DOCS_URL}"
        )


def _download(url: str, dest: Path) -> None:
    _print(f"downloading {url}")
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"download failed ({url}): {exc}") from exc


def _run(cmd: list[str] | str, *, shell: bool = False) -> None:
    """Run an install command, surfacing a clean error on failure."""
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    _print(f"running: {printable}")
    result = subprocess.run(cmd, shell=shell)  # noqa: S602,S603
    if result.returncode != 0:
        raise RuntimeError(f"command failed (exit {result.returncode}): {printable}")


def _install_windows() -> None:
    """Silently install the full lemonade.msi (server + desktop app)."""
    msi = Path(tempfile.gettempdir()) / "lemonade.msi"
    _download(WINDOWS_MSI_URL, msi)
    # /qn = silent, per-user (no elevation needed). The MSI registers the CLI
    # and Start Menu shortcut and pulls the full app payload.
    _run(["msiexec", "/i", str(msi), "/qn"])
    _print("Lemonade installed.")


def _install_linux() -> None:
    """Install the stable PPA server plus the desktop frontend on apt distros."""
    if shutil.which("apt-get") is None:
        raise RuntimeError(
            "Automatic install only supports apt-based distros (Ubuntu/Debian). "
            f"Install manually: {INSTALL_DOCS_URL}"
        )
    if os.geteuid() != 0 and shutil.which("sudo") is None:  # type: ignore[attr-defined]
        raise RuntimeError(
            "Need root (or sudo) to install system packages. "
            f"Install manually: {INSTALL_DOCS_URL}"
        )
    _run(LINUX_APT_INSTALL, shell=True)
    _print("Lemonade installed.")


def _install_macos() -> None:
    """Download the latest signed .pkg and install it system-wide."""
    pkg_url = _resolve_macos_pkg_url()
    pkg = Path(tempfile.gettempdir()) / "Lemonade.pkg"
    _download(pkg_url, pkg)
    _run(["sudo", "installer", "-pkg", str(pkg), "-target", "/"])
    _print("Lemonade installed.")


def _resolve_macos_pkg_url() -> str:
    """Resolve the versioned macOS .pkg download URL from the latest release."""
    req = urllib.request.Request(
        GITHUB_LATEST_RELEASE_API, headers={"Accept": "application/vnd.github+json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15.0) as r:  # noqa: S310
            data = json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(f"could not query latest release: {exc}") from exc
    for asset in data.get("assets", []):
        name = asset.get("name", "")
        if name.endswith("-Darwin.pkg"):
            return asset["browser_download_url"]
    raise RuntimeError(
        "No macOS .pkg asset found in the latest release. "
        f"Install manually: {INSTALL_DOCS_URL}"
    )


def service_start_hint() -> str:
    """OS-specific command to (re)start the `lemond` service by hand.

    Modern Lemonade has no `lemonade serve`; the `lemond` daemon is managed by
    the OS service manager and auto-starts on install. If it is somehow down,
    the user starts it via their service manager, not via the CLI.
    """
    system = platform.system()
    if system == "Linux":
        # v11 ships a system service; older 10.x used a --user unit. Offer both.
        return (
            "sudo systemctl start lemond   "
            "(or, for a per-user install: systemctl --user start lemond)"
        )
    if system == "Darwin":  # macOS
        return "sudo launchctl load /Library/LaunchDaemons/com.lemonade.server.plist"
    if system == "Windows":
        return (
            "start the Lemonade tray app from the Start menu, or run "
            "`Start-Service lemond` (or `net start lemond`) in an elevated shell"
        )
    return f"start the Lemonade service for your OS; see {INSTALL_DOCS_URL}"


def uninstall_hint() -> str:
    """OS-specific ways to remove an old/incompatible Lemonade.

    An incompatible `lemonade` could come from any install channel, so we do
    not assume pip. We list the removal path for each channel and let the user
    apply whichever one matches how they installed it.
    """
    system = platform.system()
    if system == "Linux":
        return (
            "remove it however it was installed: `sudo apt remove lemonade-server` "
            "(apt/PPA) or `pip uninstall lemonade-sdk` (pip)"
        )
    if system == "Darwin":  # macOS
        return (
            "remove it however it was installed: delete the installed "
            "`Lemonade.app`/receipt from a .pkg install, or "
            "`pip uninstall lemonade-sdk` (pip)"
        )
    if system == "Windows":
        return (
            "remove it however it was installed: for an .msi install, run "
            "`winget uninstall -e --id AMD.LemonadeServer` or uninstall Lemonade "
            "Server from Settings > Apps > Installed apps; or, if it came from "
            "pip, `pip uninstall lemonade-sdk`"
        )
    return "remove the old Lemonade using your platform's package manager or `pip uninstall lemonade-sdk`"


def wait_for_server(host: str, port: int, timeout_s: float = 90.0) -> bool:
    """Poll /api/v1/health until it answers 200 or we hit the timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if check_server_reachable(host, port):
            return True
        time.sleep(2.0)
    return False


def check_server_reachable(host: str, port: int) -> bool:
    """Return True if /api/v1/health responds 200 within 3 seconds."""
    url = f"http://{host}:{port}/api/v1/health"
    try:
        status, _ = _http_get(url, timeout_s=3.0)
        return status == 200
    except (urllib.error.URLError, OSError):
        return False


def render_rule_block(
    *,
    host: str,
    port: int,
    image_model: str,
    tts_model: str,
    stt_model: str,
) -> str:
    """Read the rule template and fill in endpoint/model choices.

    The template already includes BEGIN/END markers and matches the constants
    at the top of this file. We re-validate that here so a future template
    edit cannot silently drift away from the markers the writer relies on.
    """
    if not RULE_TEMPLATE.exists():
        raise FileNotFoundError(
            f"Rule template missing: {RULE_TEMPLATE}. "
            "Did the skill folder get partially copied?"
        )
    text = RULE_TEMPLATE.read_text(encoding="utf-8")
    if BEGIN_MARKER not in text or END_MARKER not in text:
        raise ValueError(
            "Rule template is missing the BEGIN/END markers; refuse to write "
            "AGENTS.md because re-runs would append duplicate blocks."
        )
    endpoint_host = "localhost" if host in {"127.0.0.1", "::1"} else host
    base_root = f"http://{endpoint_host}:{port}"
    replacements = {
        "{{LEMONADE_BASE_ROOT}}": base_root,
        "{{LEMONADE_BASE_URL}}": f"{base_root}/api/v1",
        "{{IMAGE_MODEL}}": image_model,
        "{{TTS_MODEL}}": tts_model,
        "{{STT_MODEL}}": stt_model,
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    unresolved = sorted(set(re.findall(r"\{\{[A-Z_]+\}\}", text)))
    if unresolved:
        raise ValueError(
            "Rule template still has unresolved placeholders: "
            + ", ".join(unresolved)
        )
    return text.strip() + "\n"


def upsert_agents_md(
    workspace: Path,
    *,
    host: str,
    port: int,
    image_model: str,
    tts_model: str,
    stt_model: str,
) -> Path:
    """Write or replace the rule block inside <workspace>/AGENTS.md."""
    target = workspace / "AGENTS.md"
    block = render_rule_block(
        host=host,
        port=port,
        image_model=image_model,
        tts_model=tts_model,
        stt_model=stt_model,
    )

    if not target.exists():
        target.write_text(
            "# Agent instructions\n\n"
            "Project-scoped rules picked up automatically by Cursor, Claude Code,\n"
            "Codex, Gemini CLI, and other AGENTS.md-aware coding agents.\n\n"
            f"{block}",
            encoding="utf-8",
        )
        _print(f"created {target}")
        return target

    existing = target.read_text(encoding="utf-8")
    if BEGIN_MARKER in existing and END_MARKER in existing:
        before, _, rest = existing.partition(BEGIN_MARKER)
        _, _, after = rest.partition(END_MARKER)
        # Strip trailing newline noise around the spliced region so we don't
        # accumulate blank lines on every re-run.
        new = before.rstrip() + "\n\n" + block + after.lstrip()
        if new == existing:
            _print(f"AGENTS.md rule already up to date at {target}")
            return target
        target.write_text(new, encoding="utf-8")
        _print(f"updated rule block in {target}")
        return target

    # No existing block: append with a separating blank line.
    if not existing.endswith("\n"):
        existing += "\n"
    target.write_text(existing + "\n" + block, encoding="utf-8")
    _print(f"appended rule block to {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=_default_workspace(),
        help="Workspace root where AGENTS.md should be written (default: workspace root, auto-detected).",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("LEMONADE_HOST", DEFAULT_HOST),
        help="Lemonade Server host (default: 127.0.0.1 / $LEMONADE_HOST).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("LEMONADE_PORT", str(DEFAULT_PORT))),
        help="Lemonade Server port (default: 13305 / $LEMONADE_PORT).",
    )
    parser.add_argument(
        "--image-model",
        default=DEFAULT_IMAGE_MODEL,
        help=f"Image generation model written into AGENTS.md, pulled on first use (default: {DEFAULT_IMAGE_MODEL}).",
    )
    parser.add_argument(
        "--tts-model",
        default=DEFAULT_TTS_MODEL,
        help=f"Text-to-speech model written into AGENTS.md, pulled on first use (default: {DEFAULT_TTS_MODEL}).",
    )
    parser.add_argument(
        "--stt-model",
        default=DEFAULT_STT_MODEL,
        help=f"Speech-to-text model written into AGENTS.md, pulled on first use (default: {DEFAULT_STT_MODEL}).",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Do not auto-install Lemonade; just report and exit non-zero if the CLI or service is missing.",
    )
    args = parser.parse_args(argv)

    cli, stale = find_cli()

    # An old/incompatible `lemonade` on PATH shadows the modern CLI. It may
    # have come from any install channel (old .msi/.deb or pip), so we never
    # assume one. We also never drive or auto-remove it -- we guide the user,
    # because a shadowing binary will keep hiding a freshly installed modern
    # CLI until it is removed.
    if cli is None and stale is not None:
        _print(f"FAIL: found an old/incompatible Lemonade CLI at {stale}.")
        _print(
            "It is missing the modern `lemonade status` command, so it predates "
            "the unified CLI (v10.1.0) and cannot be used by this skill."
        )
        _print(f"Uninstall it first so it stops shadowing the modern CLI: {uninstall_hint()}.")
        _print(
            "Then re-run this skill to install the latest Lemonade for you, or "
            f"install it yourself: {INSTALL_DOCS_URL}"
        )
        return 2

    if cli is None:
        if args.no_install:
            _print("FAIL: the `lemonade` CLI is not on PATH (--no-install set).")
            _print(f"Install Lemonade manually: {INSTALL_DOCS_URL}")
            return 2
        _print("`lemonade` CLI not found; installing the latest version of Lemonade.")
        try:
            install_lemonade()
        except RuntimeError as exc:
            _print(f"FAIL: automatic install did not complete: {exc}")
            return 2
        cli, stale = find_cli()
        if cli is None:
            _print("FAIL: install finished but a modern `lemonade` CLI is still not found.")
            _print(
                "Open a new shell so PATH refreshes and re-run, or install "
                f"manually: {INSTALL_DOCS_URL}"
            )
            return 2
    _print(f"using Lemonade CLI: {cli}")

    # Modern Lemonade auto-starts the `lemond` service on install; there is no
    # `lemonade serve`. If it is not up yet (e.g. still starting right after a
    # fresh install), poll briefly, then guide the user to start the OS
    # service rather than trying to spawn it ourselves.
    if not check_server_reachable(args.host, args.port):
        if args.no_install:
            _print(
                f"FAIL: Lemonade Server is not responding at "
                f"http://{args.host}:{args.port}/api/v1/health (--no-install set)."
            )
            _print(f"Start the service: {service_start_hint()}")
            return 3
        _print("Lemonade service not reachable yet; waiting for it to come up.")
        if not wait_for_server(args.host, args.port):
            _print(
                f"FAIL: the Lemonade service did not become reachable at "
                f"http://{args.host}:{args.port}/api/v1/health."
            )
            _print(f"Start it manually, then re-run: {service_start_hint()}")
            _print(f"If it is not installed, see {INSTALL_DOCS_URL}")
            return 3

    _print(f"server reachable at http://{args.host}:{args.port}")

    upsert_agents_md(
        args.workspace.resolve(),
        host=args.host,
        port=args.port,
        image_model=args.image_model,
        tts_model=args.tts_model,
        stt_model=args.stt_model,
    )
    _print("done. Future image, TTS, and STT requests now route to local Lemonade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
