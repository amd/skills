# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Regression tests for CLI detection, port discovery, and installer choice in
../setup_local_ai.py.

`lemonade` is the only valid CLI, but the name alone does not prove a binary is
the modern build that drives the `lemond` service. An older `lemonade` left on
PATH answers to the name and then fails every subsequent command, so the script
selects by capability (`lemonade status`) instead. These cases pin that
behaviour, plus the install paths: winget on Windows and the Homebrew cask on
macOS, each falling back to a downloaded installer only when the package
manager is absent.

The same `status` subcommand also reports where the service actually bound, so
the last case pins that the script asks rather than assuming DEFAULT_PORT: a
server on another port must not read as "not running".

Cases needing a real Lemonade install are skipped when it is missing, so the
suite still runs on a bare machine.

Run standalone; no pytest or third-party dependency required:

    python3 scripts/tests/test_setup_local_ai.py
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import stat
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "setup_local_ai.py"
WINDOWS = platform.system() == "Windows"

# What an older `lemonade` prints for `status`: argparse rejects the subcommand.
STALE_CLI_STDERR = (
    "usage: lemonade [-h] [-i INPUT] ...\n"
    "lemonade: error: argument {}: invalid choice: 'status'\n"
)

# What the modern CLI prints; the phrase is the discriminator.
MODERN_CLI_STDOUT = "Server is running on port 13305\n\nVersion    11.7.0\n"

# A service on a non-default port, as `status` reports it. The WebSocket line
# is included because it is a second port in the same output that endpoint
# discovery must not pick up.
OTHER_PORT = 45999
OTHER_PORT_STDOUT = (
    f"Server is running on port {OTHER_PORT}\n\n"
    "Property            Value\n"
    "Version             11.7.0\n"
    "WebSocket Port      9001\n"
)


def _load_setup():
    spec = importlib.util.spec_from_file_location("setup_local_ai_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = _load_setup()

FAILURES: list[str] = []


def check(name: str, condition: bool) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}")
    if not condition:
        FAILURES.append(name)


def skip(name: str, why: str) -> None:
    print(f"  SKIP  {name} ({why})")


def _install_fake(directory: Path, name: str, *, stdout="", stderr="", code=0) -> Path:
    """Create an executable named `name` with fixed output and exit code."""
    directory.mkdir(parents=True, exist_ok=True)
    impl = directory / f"{name.replace('-', '_')}_impl.py"
    impl.write_text(
        "import sys\n"
        f"sys.stdout.write({stdout!r})\n"
        f"sys.stderr.write({stderr!r})\n"
        f"sys.exit({code})\n",
        encoding="utf-8",
    )
    if WINDOWS:
        launcher = directory / f"{name}.bat"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "{impl}" %*\r\n', "utf-8")
    else:
        launcher = directory / name
        launcher.write_text(f'#!/bin/sh\nexec {sys.executable} {impl} "$@"\n', "utf-8")
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    return launcher


def test_only_one_cli_name() -> None:
    print("\n[1] `lemonade` is the only CLI name the script will look for")
    check("CLI_NAME is lemonade", setup.CLI_NAME == "lemonade")
    source = SCRIPT.read_text(encoding="utf-8")
    for dead in ("lemonade-server-dev", "lemonade_sdk", "pip install", "pip uninstall"):
        check(f"no reference to {dead}", dead not in source)
    # The apt *package* is still named lemonade-server even though the CLI is
    # `lemonade`, so that one string is expected to survive.
    check(
        "apt package name retained",
        "lemonade-server" in setup.LINUX_APT_INSTALL,
    )


def test_capability_probe(tmpdir: Path, real_cli: str | None) -> None:
    print("\n[2] a CLI is judged by `status`, not by its name")
    stale = _install_fake(tmpdir, "lemonade", stderr=STALE_CLI_STDERR, code=2)
    check("old CLI rejected", not setup.is_modern_cli(str(stale)))

    modern_dir = tmpdir / "modern"
    modern_dir.mkdir()
    modern = _install_fake(modern_dir, "lemonade", stdout=MODERN_CLI_STDOUT)
    check("modern CLI accepted", setup.is_modern_cli(str(modern)))

    if real_cli:
        check("real installed CLI accepted", setup.is_modern_cli(real_cli))
    else:
        skip("real installed CLI accepted", "Lemonade not installed")


def test_stale_cli_is_reported(tmpdir: Path) -> None:
    print("\n[3] an old CLI on PATH is reported, never driven")
    original_path = os.environ["PATH"]
    original_dir = setup.WINDOWS_INSTALL_DIR
    # Hide the real Windows install tree, which find_cli probes outside PATH,
    # so this case sees only the stale fake.
    setup.WINDOWS_INSTALL_DIR = tmpdir / "no-such-install"
    os.environ["PATH"] = str(tmpdir)
    try:
        modern, stale = setup.find_cli()
        check("no modern CLI returned", modern is None)
        check("stale CLI surfaced to the caller", stale is not None)

        # A stale CLI earlier on PATH still wins the name lookup, so setup must
        # keep reporting it rather than quietly using a modern one further
        # along: the AGENTS.md rule's own `lemonade pull` would hit the stale
        # binary too, and only uninstalling it fixes that.
        os.environ["PATH"] = str(tmpdir) + os.pathsep + str(tmpdir / "modern")
        modern, stale = setup.find_cli()
        check("shadowing stale CLI still reported", modern is None and stale is not None)

        # Once it is gone, the modern CLI is picked up with no other change.
        os.environ["PATH"] = str(tmpdir / "modern")
        modern, stale = setup.find_cli()
        check("modern CLI used once the stale one is removed", modern is not None)
    finally:
        os.environ["PATH"] = original_path
        setup.WINDOWS_INSTALL_DIR = original_dir


def test_install_paths() -> None:
    print("\n[4] installs use the current package managers")
    check("winget id is AMD.LemonadeServer", setup.WINDOWS_WINGET_ID == "AMD.LemonadeServer")
    check("brew cask is lemonade-server", setup.MACOS_BREW_CASK == "lemonade-server")
    check(
        "Ubuntu uses the stable PPA",
        "ppa:lemonade-team/stable" in setup.LINUX_APT_INSTALL,
    )
    check(
        "install docs point at the current guide",
        setup.INSTALL_DOCS_URL == "https://lemonade-server.ai/docs/guide/install/",
    )

    source = SCRIPT.read_text(encoding="utf-8")
    # The downloadable installers must stay reachable as fallbacks, but only
    # after the package manager has been tried.
    check("MSI kept as Windows fallback", "lemonade.msi" in source)
    check("pkg kept as macOS fallback", "-Darwin.pkg" in source)
    check(
        "winget tried before the MSI",
        source.index('shutil.which("winget")') < source.index("WINDOWS_MSI_URL, msi"),
    )
    check(
        "brew tried before the pkg",
        source.index('shutil.which("brew")') < source.index("_resolve_macos_pkg_url()\n    pkg ="),
    )


def test_no_serve_command() -> None:
    print("\n[5] the script never invokes a `serve` subcommand")
    source = SCRIPT.read_text(encoding="utf-8")
    check('no `"serve"` argument built', '"serve"' not in source)
    check("service start hint provided instead", bool(setup.service_start_hint()))


def test_uninstall_hint_is_current() -> None:
    print("\n[6] uninstall guidance matches the supported channels")
    hint = setup.uninstall_hint()
    check("no pip in the hint", "pip" not in hint)
    expected = {
        "Windows": "winget uninstall",
        "Linux": "apt remove",
        "Darwin": "brew uninstall",
    }.get(platform.system())
    if expected:
        check(f"hint mentions {expected}", expected in hint)
    else:
        skip("platform-specific hint", f"untested OS {platform.system()}")


def test_port_discovery(tmpdir: Path, real_cli: str | None) -> None:
    print("\n[7] the bound port is discovered, not assumed to be the default")
    json_cli = str(
        _install_fake(tmpdir / "json", "lemonade", stdout=f'{{"port": {OTHER_PORT}}}\n')
    )
    check("port read from `status --json`", setup.discover_port(json_cli) == OTHER_PORT)

    # Builds without `--json` fall back to the printed line, where the
    # WebSocket port must not be mistaken for the API port.
    text_cli = str(_install_fake(tmpdir / "text", "lemonade", stdout=OTHER_PORT_STDOUT))
    check("port read from plain `status`", setup.discover_port(text_cli) == OTHER_PORT)

    stopped = str(
        _install_fake(tmpdir / "stopped", "lemonade", stdout="Server is not running\n")
    )
    check("a stopped service reports no port", setup.discover_port(stopped) is None)

    check("discovery beats DEFAULT_PORT", setup.resolve_port(json_cli, None, None) == OTHER_PORT)
    check("an explicit port is never second-guessed", setup.resolve_port(json_cli, None, 9) == 9)
    # The CLI here knows nothing about another machine's config.
    check(
        "a remote host keeps the default port",
        setup.resolve_port(json_cli, "workstation.local", None) == setup.DEFAULT_PORT,
    )
    check("no CLI falls back to the default", setup.resolve_port(None, None, None) == setup.DEFAULT_PORT)

    # Nothing listens on OTHER_PORT, so this only pins that the wait follows
    # the port `status` reports instead of the one it started from.
    check(
        "the wait loop follows the reported port",
        setup.wait_for_server(setup.DEFAULT_HOST, 1, cli=json_cli, timeout_s=1.0)
        == (False, OTHER_PORT),
    )

    # `status` already returns the beacon's answer; scanning for beacons here
    # would also surface Lemonade servers on other machines on the LAN.
    check('no `"scan"` argument built', '"scan"' not in SCRIPT.read_text(encoding="utf-8"))

    if real_cli:
        check("real CLI reports a port or none", setup.discover_port(real_cli) != 0)
    else:
        skip("real CLI reports a port or none", "Lemonade not installed")


def main() -> int:
    real_cli = shutil.which(setup.CLI_NAME)
    if real_cli and not setup.is_modern_cli(real_cli):
        real_cli = None
    print(f"Lemonade CLI: {real_cli or 'not installed (some cases skipped)'}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        test_only_one_cli_name()
        test_capability_probe(tmpdir, real_cli)
        test_stale_cli_is_reported(tmpdir)
        test_install_paths()
        test_no_serve_command()
        test_uninstall_hint_is_current()
        test_port_discovery(tmpdir, real_cli)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
