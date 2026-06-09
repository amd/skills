#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Per-modality setup for the `local-ai-use` skill.

The skill covers two independent modality groups — `image` (image generation)
and `speech` (text-to-speech + speech-to-text). You pick which one(s) to set up
so a request for one modality never pulls the other's models.

Performs the setup steps from SKILL.md, scoped to the selected modality/ies:

  1. Confirms the system-wide Lemonade Server is installed and reachable on
     http://localhost:13305 (override with --host / --port or LEMONADE_HOST /
     LEMONADE_PORT).
  2. Pulls only the selected modality's default models if missing
     (image: SD-Turbo; speech: kokoro-v1 + Whisper-Tiny).
  3. Writes the routing rule from `templates/local-ai-rule.md` into
     <workspace>/AGENTS.md, keeping only the selected modality sections, between
     stable BEGIN/END markers so re-runs replace the block in place.

Usage: `setup_local_ai.py image`, `setup_local_ai.py speech`, or
`setup_local_ai.py all` (both). The installed rule reflects exactly the
modality/ies passed, so to add a modality later re-run with the full set.

The script is idempotent: re-running with the same modality/ies only re-runs the
healthcheck. It exits non-zero on any unrecoverable failure.

Constants are documented inline; nothing is magical.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
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

# The two modality groups the skill can set up. `all` is a shorthand for both.
# Each rule section in the template is wrapped in `<!-- modality:NAME -->` /
# `<!-- /modality:NAME -->`; unselected sections are stripped before writing.
MODALITIES = ("image", "speech")

# Stable markers around the rule block in AGENTS.md. The script rewrites the
# region between these markers in place; do not change the marker strings or
# every existing AGENTS.md will get a duplicate block on the next run.
BEGIN_MARKER = "<!-- BEGIN amd-skills:local-ai-use -->"
END_MARKER = "<!-- END amd-skills:local-ai-use -->"

SKILL_DIR = Path(__file__).resolve().parent.parent
RULE_TEMPLATE = SKILL_DIR / "templates" / "local-ai-rule.md"

INSTALL_URL = "https://lemonade-server.ai/install_options.html"


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


def check_cli_installed() -> bool:
    """Return True if the `lemonade` CLI is on PATH."""
    return shutil.which("lemonade") is not None


def check_server_reachable(host: str, port: int) -> bool:
    """Return True if /api/v1/health responds 200 within 3 seconds."""
    url = f"http://{host}:{port}/api/v1/health"
    try:
        status, _ = _http_get(url, timeout_s=3.0)
        return status == 200
    except (urllib.error.URLError, OSError):
        return False


def list_downloaded_models(host: str, port: int) -> set[str]:
    """Return the set of locally downloaded model IDs.

    Uses `lemonade list --downloaded` (CLI) and falls back to
    GET /api/v1/models when the CLI lacks the flag. Returning an empty set is
    treated as "could not determine" by the caller, which still attempts the
    pulls; `lemonade pull` is itself idempotent.
    """
    try:
        out = subprocess.run(
            ["lemonade", "list", "--downloaded", "--json"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
        data = json.loads(out)
        return {m.get("id", "") for m in data if isinstance(m, dict)}
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
        pass

    try:
        status, body = _http_get(
            f"http://{host}:{port}/api/v1/models",
            timeout_s=5,
        )
        if status == 200:
            data = json.loads(body)
            return {
                m.get("id", "") for m in data.get("data", [])
                if isinstance(m, dict) and m.get("downloaded")
            }
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        pass

    return set()


def pull_model(model: str) -> bool:
    """Run `lemonade pull <model>`. Returns True on success."""
    _print(f"pulling {model}...")
    try:
        subprocess.run(
            ["lemonade", "pull", model],
            check=True,
            # Stream output so the user sees the download progress instead of
            # staring at a frozen prompt; SD-Turbo is several GB.
            stdout=None, stderr=None,
            # SD-Turbo is the largest pull at ~5 GB. 30 minutes is generous
            # for a slow connection; below that we'd false-positive on real
            # downloads.
            timeout=30 * 60,
        )
        return True
    except subprocess.CalledProcessError as exc:
        _print(f"pull failed for {model} (exit {exc.returncode})")
        return False
    except subprocess.TimeoutExpired:
        _print(f"pull timed out for {model} after 30 minutes")
        return False


def _select_modality_sections(text: str, modalities: set[str]) -> str:
    """Keep only the selected `<!-- modality:NAME -->` sections in the template.

    Unselected sections are removed entirely; for selected sections the marker
    comments themselves are stripped, leaving their content in place.
    """
    for name in MODALITIES:
        open_tag = f"<!-- modality:{name} -->"
        close_tag = f"<!-- /modality:{name} -->"
        if name in modalities:
            text = text.replace(open_tag + "\n", "").replace(open_tag, "")
            text = text.replace("\n" + close_tag, "").replace(close_tag, "")
        else:
            # Drop the whole region, including a trailing blank line if present.
            text = re.sub(
                re.escape(open_tag) + r".*?" + re.escape(close_tag) + r"\n?",
                "",
                text,
                flags=re.DOTALL,
            )
    return text


def render_rule_block(
    *,
    host: str,
    port: int,
    modalities: set[str],
    image_model: str,
    tts_model: str,
    stt_model: str,
) -> str:
    """Read the rule template, keep the selected modalities, and fill in values.

    The template includes BEGIN/END markers and per-modality section markers. We
    re-validate the BEGIN/END markers here so a future template edit cannot
    silently drift away from the markers the writer relies on.
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
    text = _select_modality_sections(text, modalities)
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
    # Any placeholder left here belongs to a kept section but had no value;
    # placeholders inside dropped sections are already gone.
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
    modalities: set[str],
    image_model: str,
    tts_model: str,
    stt_model: str,
) -> Path:
    """Write or replace the rule block inside <workspace>/AGENTS.md."""
    target = workspace / "AGENTS.md"
    block = render_rule_block(
        host=host,
        port=port,
        modalities=modalities,
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
        "modality",
        nargs="+",
        choices=(*MODALITIES, "all"),
        help="Which modality/ies to set up: 'image', 'speech', or 'all' (both). "
        "The installed rule reflects exactly what you pass.",
    )
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
        "--skip-pull",
        action="store_true",
        help="Do not pull missing models; just verify and write AGENTS.md.",
    )
    parser.add_argument(
        "--image-model",
        default=DEFAULT_IMAGE_MODEL,
        help=f"Image generation model to pull and write into AGENTS.md (default: {DEFAULT_IMAGE_MODEL}).",
    )
    parser.add_argument(
        "--tts-model",
        default=DEFAULT_TTS_MODEL,
        help=f"Text-to-speech model to pull and write into AGENTS.md (default: {DEFAULT_TTS_MODEL}).",
    )
    parser.add_argument(
        "--stt-model",
        default=DEFAULT_STT_MODEL,
        help=f"Speech-to-text model to pull and write into AGENTS.md (default: {DEFAULT_STT_MODEL}).",
    )
    args = parser.parse_args(argv)

    modalities = set(MODALITIES) if "all" in args.modality else set(args.modality)

    if not check_cli_installed():
        _print("FAIL: `lemonade` is not on PATH.")
        _print(f"Install Lemonade Server first: {INSTALL_URL}")
        return 2

    if not check_server_reachable(args.host, args.port):
        _print(
            f"FAIL: Lemonade Server is not responding at "
            f"http://{args.host}:{args.port}/api/v1/health."
        )
        _print(
            "Start it: on Windows launch the Lemonade Start Menu shortcut; "
            "on Linux run `sudo systemctl start lemonade-server`."
        )
        return 3

    _print(f"server reachable at http://{args.host}:{args.port}")
    _print(f"setting up: {', '.join(sorted(modalities))}")

    if not args.skip_pull:
        # Pull only the models for the selected modality/ies.
        wanted = []
        if "image" in modalities:
            wanted.append(args.image_model)
        if "speech" in modalities:
            wanted += [args.tts_model, args.stt_model]
        downloaded = list_downloaded_models(args.host, args.port)
        for model in dict.fromkeys(wanted):
            if model in downloaded:
                _print(f"already downloaded: {model}")
                continue
            if not pull_model(model):
                # Surface the failure but keep going so the user at least gets
                # the rule installed for the models that did succeed.
                _print(
                    f"continuing without {model}; the rule will reference it "
                    "but calls will 404 until you pull it."
                )

    upsert_agents_md(
        args.workspace.resolve(),
        host=args.host,
        port=args.port,
        modalities=modalities,
        image_model=args.image_model,
        tts_model=args.tts_model,
        stt_model=args.stt_model,
    )
    _print(
        "done. Future "
        + " and ".join(sorted(modalities))
        + " requests now route to local Lemonade."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
