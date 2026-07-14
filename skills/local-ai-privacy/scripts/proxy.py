#!/usr/bin/env python3
"""
local-ai-privacy: redacting proxy server.

Started automatically by start.py — do not run this directly.

Reads config from ~/.claude/skills/local-ai-privacy/proxy.conf:
  {
    "cloud_url":       "https://api.anthropic.com",
    "lemonade_url":    "http://localhost:13305",
    "redaction_model": "Qwen3.6-35B-A3B-NoThinking",
    "proxy_port":      8317
  }

Endpoints
---------
GET  /health        -> 200 {"status":"ok"}  (liveness for start.py / the skill)
POST /redact        -> local-only redaction preview. Body {"text": "..."} (or a
                       full Anthropic request). Returns the placeholders that
                       WILL be applied, WITHOUT ever calling the cloud. Handy to
                       run from a terminal to confirm masking works.
POST /v1/messages   -> the load-bearing path. Redacts system + every message +
                       tool blocks, forwards ONLY the redacted body to the real
                       cloud endpoint, then rehydrates placeholders in the
                       response locally before returning it to Claude Code.
everything else     -> passed through unchanged (e.g. GET /v1/models).

Placeholder consistency
-----------------------
A single in-memory map is shared across /redact and /v1/messages for the life of
the proxy process, so the same real value always maps to the same placeholder
([SSN_1], [EMAIL_2], ...) on every request, and rehydration on the way back is
unambiguous. The map lives only in RAM — the real values never touch disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[proxy] %(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("proxy")

# ---------------------------------------------------------------------------
# Config (loaded once at startup from proxy.conf)
# ---------------------------------------------------------------------------

SKILL_DIR = Path(__file__).resolve().parent.parent
REDACTION_PROMPT_FILE = SKILL_DIR / "data" / "redaction-prompt.txt"

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}

# Response headers we must NOT echo downstream: `requests` decodes the body for
# us (gzip/deflate), and rehydration changes its length, so the upstream
# Content-Encoding / Content-Length no longer describe what we send. Forwarding
# them makes Claude Code try to gunzip plain text -> ZlibError.
_SKIP_RESP_HEADERS = {
    "transfer-encoding", "connection", "keep-alive",
    "content-encoding", "content-length",
}

# Local model context is small; discover PII in line-aligned chunks of this size.
DISCOVERY_CHUNK_CHARS = 8000

PLACEHOLDER_NOTE = (
    "NOTE: some values in this conversation have been replaced with redaction "
    "placeholders — an ALL-CAPS type name and a number in square brackets. Each "
    "one is an opaque stand-in for a real value you cannot see. Use ONLY the "
    "placeholder tokens that already appear in the conversation, copy them "
    "verbatim wherever that value belongs, and never invent new placeholder "
    "tokens or change their numbers."
)

_cfg: dict = {}


def _load_conf(conf_path: Path) -> None:
    if not conf_path.exists():
        log.error("Config file not found: %s", conf_path)
        log.error("Run start.py first to generate it.")
        sys.exit(1)
    conf = json.loads(conf_path.read_text(encoding="utf-8"))
    _cfg.update(conf)
    log.info("Cloud endpoint  : %s", _cfg["cloud_url"])
    log.info("Lemonade URL    : %s", _cfg["lemonade_url"])
    log.info("Redaction model : %s", _cfg["redaction_model"])


# ---------------------------------------------------------------------------
# Session placeholder map (in-memory, shared across /redact and /v1/messages)
# The BaseHTTPServer handles one request at a time, so no locking is required.
# ---------------------------------------------------------------------------

_value_to_ph: dict[str, str] = {}   # "123-45-6789" -> "[SSN_1]"
_ph_to_value: dict[str, str] = {}   # "[SSN_1]"      -> "123-45-6789"
_type_counts: dict[str, int] = {}   # "SSN"          -> 1
_seen_chunks: set[str] = set()      # sha256 of text chunks already discovered

_IDX_RE = re.compile(r"_\d+$")

# Matches a token the proxy itself produced, e.g. [SSN_1] or [INTERNAL_HOST_12].
# These must never be treated as new PII, or the proxy re-redacts its own output
# and placeholders nest/relabel ([SSN_1] -> [SSN_2] -> ... ) across turns.
_PLACEHOLDER_RE = re.compile(r"^\[[A-Z][A-Z0-9_]*_\d+\]$")


def _clean_label(label: str) -> str:
    """Normalise a freeform model label to a clean UPPERCASE TYPE.

    Handles `EMAIL`, `email`, `Email Address`, `[SSN]`, `SSN_0` -> `EMAIL`/`SSN`.
    """
    k = str(label).strip().strip("[]").strip()
    k = _IDX_RE.sub("", k)            # drop any trailing _<n> index
    k = re.sub(r"[^A-Za-z_]", "_", k) # non-letters become underscores
    k = re.sub(r"_+", "_", k).strip("_")
    return k.upper() or "SECRET"


def _ph_type(ph: str) -> str:
    return _IDX_RE.sub("", ph.strip("[]")).upper()


def _placeholder_for(value: str, typ: str) -> str:
    """Return the stable placeholder for a value, minting a new one if unseen."""
    if value in _value_to_ph:
        return _value_to_ph[value]
    # Never redact a token we already produced: the bracketed form "[SSN_1]",
    # the bare inner form "SSN_1" the model sometimes extracts, or any current
    # placeholder value. Otherwise the proxy re-redacts its own output and
    # placeholders nest/relabel ([SSN_1] -> [SSN_2] -> ...).
    v = value.strip()
    if (value in _ph_to_value
            or _PLACEHOLDER_RE.match(v)
            or f"[{v.strip('[]')}]" in _ph_to_value):
        return value
    typ = (typ or "SECRET").upper()
    n = _type_counts.get(typ, 0) + 1
    _type_counts[typ] = n
    ph = f"[{typ}_{n}]"
    _value_to_ph[value] = ph
    _ph_to_value[ph] = value
    return ph


def _mask_str(s: str) -> str:
    """Replace every known real value in s with its placeholder (longest first
    so overlapping values don't clobber each other)."""
    if not s or not _value_to_ph:
        return s
    for val in sorted(_value_to_ph, key=len, reverse=True):
        if val and val in s:
            s = s.replace(val, _value_to_ph[val])
    return s


def _rehydrate_str(s: str) -> str:
    """Replace placeholders with their real values (longest first so [NAME_1]
    does not corrupt [NAME_12])."""
    if not s or not _ph_to_value:
        return s
    for ph in sorted(_ph_to_value, key=len, reverse=True):
        if ph in s:
            s = s.replace(ph, _ph_to_value[ph])
    return s


# A placeholder like [INTERNAL_HOST_12] is at most ~20 chars; never hold more.
_MAX_PLACEHOLDER = 64


def _split_safe(buf: str) -> tuple[str, str]:
    """Split streamed text into (safe, held).

    A placeholder can be split across streaming chunks (`[SSN` then `_1]`), so
    we must not rehydrate a chunk in isolation. `held` is the trailing part that
    might be the start of an incomplete placeholder ("[" with no "]" yet);
    `safe` is everything before it and can be rehydrated and flushed now.
    """
    idx = buf.rfind("[")
    if idx == -1 or "]" in buf[idx:]:
        return buf, ""                     # no open bracket pending
    if len(buf) - idx > _MAX_PLACEHOLDER:
        return buf, ""                     # too long to be a placeholder; flush
    return buf[:idx], buf[idx:]


# ---------------------------------------------------------------------------
# Local model: discover sensitive (value, type) pairs
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _load_redaction_prompt() -> str:
    if not REDACTION_PROMPT_FILE.exists():
        raise FileNotFoundError(f"Missing: {REDACTION_PROMPT_FILE}")
    return REDACTION_PROMPT_FILE.read_text(encoding="utf-8").strip()


def _extract_json(content: str) -> str:
    """Pull the JSON value (array or object) out of a noisy small-model reply
    (strip <think> blocks, markdown fences, and any surrounding prose)."""
    content = _THINK_RE.sub("", content).strip()
    if content.startswith("```"):
        content = "\n".join(
            l for l in content.splitlines() if not l.startswith("```")
        ).strip()
    starts = [p for p in (content.find("["), content.find("{")) if p != -1]
    if starts:
        start = min(starts)
        end = max(content.rfind("]"), content.rfind("}"))
        if end > start:
            content = content[start:end + 1]
    return content


def _discover_chunk(text: str) -> tuple[list[tuple[str, str]], bool]:
    """Ask the local model for sensitive (value, type) pairs in one text chunk.

    Returns (pairs, ok). ok is False only when the model was unreachable or kept
    returning unparseable output across all retries — in which case the caller
    MUST fail closed (never forward the original text to the cloud). The model
    is non-deterministic, so we retry a few times; each attempt usually parses.
    """
    url = _cfg["lemonade_url"].rstrip("/") + "/api/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("LEMONADE_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": _cfg["redaction_model"],
        "messages": [
            {"role": "system", "content": _load_redaction_prompt()},
            {"role": "user",   "content": text},
        ],
        "temperature": 0,
        # Headroom so a larger model (or a thinking model's reasoning) can't get
        # truncated into invalid JSON, which would fail closed and block a turn.
        "max_tokens": int(_cfg.get("redaction_max_tokens", 8192)),
    }

    attempts = max(1, int(_cfg.get("redaction_retries", 3)))
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("Lemonade call failed (attempt %d/%d): %s",
                        attempt, attempts, exc)
            continue

        raw = resp.json()["choices"][0]["message"]["content"].strip()
        try:
            result = json.loads(_extract_json(raw))
        except json.JSONDecodeError:
            log.warning("Non-JSON from redaction model (attempt %d/%d)",
                        attempt, attempts)
            continue

        # Preferred contract: a JSON array of {"text","label"}. Stay tolerant of
        # a dict wrapper or the older {"mapping": {...}} shape just in case.
        if isinstance(result, list):
            items = result
        elif isinstance(result, dict) and "mapping" in result:
            items = [{"text": v, "label": k} for k, v in result["mapping"].items()]
        elif isinstance(result, dict):
            items = result.get("sensitive") or result.get("entities") or []
        else:
            items = []

        pairs = []
        for it in items:
            if not isinstance(it, dict):
                continue
            val = it.get("text") or it.get("value")
            label = it.get("label") or it.get("type") or "SECRET"
            # Skip <3-char spans: masking a 1-2 char substring would smear
            # across unrelated text.
            if isinstance(val, str) and len(val.strip()) >= 3:
                pairs.append((val, _clean_label(label)))
        return pairs, True

    log.error("Redaction failed after %d attempt(s) — blocking (fail closed)",
              attempts)
    return [], False


def _chunk_by_lines(text: str, size: int) -> list[str]:
    chunks, cur, cur_len = [], [], 0
    for line in text.splitlines(keepends=True):
        if cur_len + len(line) > size and cur:
            chunks.append("".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += len(line)
    if cur:
        chunks.append("".join(cur))
    return chunks or [text]


def _ingest(text: str) -> bool:
    """Discover sensitive values in text and add them to the session map.

    Returns False if discovery could not be completed for any chunk (caller
    must fail closed). Chunks already discovered in a previous turn are skipped
    so we don't re-scan the whole transcript every request.
    """
    for chunk in _chunk_by_lines(text, DISCOVERY_CHUNK_CHARS):
        h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        if h in _seen_chunks:
            continue
        pairs, ok = _discover_chunk(chunk)
        if not ok:
            return False
        for value, typ in pairs:
            _placeholder_for(value, typ)
        _seen_chunks.add(h)
    return True


# ---------------------------------------------------------------------------
# Anthropic request structure: gather text / apply mask over system + messages
# + tool blocks. ids / names / types are left untouched.
# ---------------------------------------------------------------------------

def _deep_collect(obj, out: list) -> None:
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, list):
        for x in obj:
            _deep_collect(x, out)
    elif isinstance(obj, dict):
        for v in obj.values():
            _deep_collect(v, out)


def _deep_mask(obj):
    if isinstance(obj, str):
        return _mask_str(obj)
    if isinstance(obj, list):
        return [_deep_mask(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _deep_mask(v) for k, v in obj.items()}
    return obj


def _gather_request_text(body: dict) -> str:
    out: list[str] = []
    sys_ = body.get("system")
    if isinstance(sys_, str):
        out.append(sys_)
    elif isinstance(sys_, list):
        for b in sys_:
            if isinstance(b, dict) and b.get("type") == "text":
                out.append(b.get("text", ""))
    for msg in body.get("messages", []):
        c = msg.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            for blk in c:
                if not isinstance(blk, dict):
                    continue
                t = blk.get("type")
                if t == "text":
                    out.append(blk.get("text", ""))
                elif t == "tool_use":
                    _deep_collect(blk.get("input", {}), out)
                elif t == "tool_result":
                    _deep_collect(blk.get("content", ""), out)
    return "\n".join(x for x in out if x)


def _mask_request(body: dict) -> None:
    """Mutate body in place: mask system, every message, and tool blocks."""
    sys_ = body.get("system")
    if isinstance(sys_, str):
        body["system"] = _mask_str(sys_)
    elif isinstance(sys_, list):
        for b in sys_:
            if isinstance(b, dict) and b.get("type") == "text":
                b["text"] = _mask_str(b.get("text", ""))
    for msg in body.get("messages", []):
        c = msg.get("content")
        if isinstance(c, str):
            msg["content"] = _mask_str(c)
        elif isinstance(c, list):
            for blk in c:
                if not isinstance(blk, dict):
                    continue
                t = blk.get("type")
                if t == "text":
                    blk["text"] = _mask_str(blk.get("text", ""))
                elif t == "tool_use":
                    blk["input"] = _deep_mask(blk.get("input", {}))
                elif t == "tool_result":
                    blk["content"] = _deep_mask(blk.get("content", ""))


def _inject_placeholder_note(body: dict) -> None:
    """Tell the cloud model to preserve [TYPE_N] tokens. Only when something was
    actually redacted."""
    if not _ph_to_value:
        return
    sys_ = body.get("system")
    if sys_ is None:
        body["system"] = PLACEHOLDER_NOTE
    elif isinstance(sys_, str):
        body["system"] = sys_ + "\n\n" + PLACEHOLDER_NOTE
    elif isinstance(sys_, list):
        sys_.append({"type": "text", "text": PLACEHOLDER_NOTE})


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class ProxyHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # handled by our logger

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json_error(self, code: int, msg: str) -> None:
        self._send_json(code, {"error": {"type": "proxy_error", "message": msg}})

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    # --- routing ---

    def _route(self) -> str:
        """Path without query string or trailing slash, e.g. `/v1/messages`."""
        return self.path.split("?", 1)[0].rstrip("/")

    def _relay_headers(self, headers) -> None:
        for k, v in headers.items():
            if k.lower() in _SKIP_RESP_HEADERS:
                continue
            self.send_header(k, v)

    def do_GET(self):
        if self._route() in ("/health", "/__proxy_health"):
            self._send_json(200, {"status": "ok"})
        else:
            self._passthrough()

    def do_POST(self):
        path = self._route()
        if path.endswith("/redact"):
            self._handle_redact()
        elif path.endswith("/messages"):
            self._handle_messages()
        else:
            self._passthrough()

    # --- /redact: local-only preview, never touches the cloud ---

    def _handle_redact(self) -> None:
        raw = self._read_body()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json_error(400, "Invalid JSON")
            return

        text = body.get("text")
        if text is None:
            text = _gather_request_text(body)

        if not text or not text.strip():
            self._send_json(200, {"ok": True, "masked": text or "", "entities": []})
            return

        try:
            ok = _ingest(text)
        except Exception as exc:
            log.error("Redaction preview error: %s", exc)
            ok = False
        if not ok:
            self._send_json_error(
                502,
                "local-ai-privacy: redaction preview could not be completed. "
                "Check that the Lemonade server is running, then retry.",
            )
            return

        entities = [
            {"type": _ph_type(ph), "original": val, "placeholder": ph}
            for val, ph in _value_to_ph.items()
            if val in text
        ]
        log.info("Preview: %d entit%s in submitted text",
                 len(entities), "y" if len(entities) == 1 else "ies")
        self._send_json(200, {"ok": True, "masked": _mask_str(text),
                              "entities": entities})

    # --- /v1/messages: redact -> forward -> rehydrate ---

    def _handle_messages(self) -> None:
        raw = self._read_body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json_error(400, "Invalid JSON")
            return

        stream = body.get("stream", False)

        # 1. discover + mask system, messages, and tool blocks
        text = _gather_request_text(body)
        if text.strip():
            try:
                ok = _ingest(text)
            except Exception as exc:
                log.error("Redaction error (%s) — blocking request (fail closed)", exc)
                ok = False
            if not ok:
                self._send_json_error(
                    502,
                    "local-ai-privacy: redaction could not be completed, so this "
                    "request was blocked to keep unredacted data off the cloud. "
                    "Check that the Lemonade server is running and see proxy.log, "
                    "then retry.",
                )
                return
            _mask_request(body)
            _inject_placeholder_note(body)
            if _value_to_ph:
                log.info("Forwarding redacted request (%d known entities)",
                         len(_value_to_ph))

        # 2. forward ONLY the redacted body to the real cloud endpoint.
        # Preserve the original path + query (e.g. /v1/messages?beta=true).
        cloud_url = _cfg["cloud_url"].rstrip("/") + self.path
        fwd_headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in HOP_BY_HOP}
        fwd_headers["Content-Type"] = "application/json"
        clean_body = json.dumps(body).encode("utf-8")
        fwd_headers["Content-Length"] = str(len(clean_body))

        try:
            cloud_resp = requests.post(
                cloud_url, data=clean_body, headers=fwd_headers,
                stream=True, timeout=300,
            )
        except requests.RequestException as exc:
            self._send_json_error(502, f"Cloud error: {exc}")
            return

        # 3. relay response, rehydrating placeholders locally. The body is always
        # decoded (and possibly rewritten), so drop upstream encoding/length.
        if stream:
            self.send_response(cloud_resp.status_code)
            self._relay_headers(cloud_resp.headers)
            self.end_headers()
            self._stream_sse(cloud_resp)
        else:
            data = self._rehydrated_json(cloud_resp)
            self.send_response(cloud_resp.status_code)
            self._relay_headers(cloud_resp.headers)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _write_raw(self, s: str) -> None:
        self.wfile.write(s.encode("utf-8"))
        self.wfile.flush()

    def _emit_event(self, etype: str, data: dict) -> None:
        self._write_raw("event: %s\ndata: %s\n\n" % (etype, json.dumps(data)))

    def _emit_text_delta(self, index: int, text: str) -> None:
        self._emit_event("content_block_delta", {
            "type": "content_block_delta", "index": index,
            "delta": {"type": "text_delta", "text": text},
        })

    def _stream_sse(self, cloud_resp) -> None:
        # Fast path: nothing was redacted, so relay bytes verbatim.
        if not _ph_to_value:
            try:
                for chunk in cloud_resp.iter_content(chunk_size=8192):
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        # Rehydrating path: reassemble events (which span several lines) and
        # buffer per-index text so a placeholder split across chunks is restored.
        buffers: dict[int, str] = {}
        lines: list[str] = []
        try:
            for line in cloud_resp.iter_lines(decode_unicode=True):
                if line != "":
                    lines.append(line)
                    continue
                if lines:
                    self._process_sse_event(lines, buffers)
                    lines = []
            if lines:
                self._process_sse_event(lines, buffers)
            # Flush anything still held back.
            for i, held in buffers.items():
                if held:
                    self._emit_text_delta(i, _rehydrate_str(held))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _process_sse_event(self, lines: list[str], buffers: dict[int, str]) -> None:
        raw = "\n".join(lines) + "\n\n"
        data = None
        for ln in lines:
            if ln.startswith("data:"):
                ds = ln[5:].strip()
                if ds and ds != "[DONE]":
                    try:
                        data = json.loads(ds)
                    except json.JSONDecodeError:
                        data = None
        if data is None:
            self._write_raw(raw)
            return

        t = data.get("type")
        delta = data.get("delta", {}) if isinstance(data.get("delta"), dict) else {}

        if t == "content_block_delta" and delta.get("type") == "text_delta":
            i = data.get("index", 0)
            buffers[i] = buffers.get(i, "") + delta.get("text", "")
            safe, held = _split_safe(buffers[i])
            buffers[i] = held
            if safe:
                self._emit_text_delta(i, _rehydrate_str(safe))
            return

        if t == "content_block_stop":
            i = data.get("index", 0)
            if buffers.get(i):
                self._emit_text_delta(i, _rehydrate_str(buffers[i]))
                buffers[i] = ""
            self._write_raw(raw)
            return

        if t in ("message_delta", "message_stop"):
            for i, held in list(buffers.items()):
                if held:
                    self._emit_text_delta(i, _rehydrate_str(held))
                    buffers[i] = ""
            self._write_raw(raw)
            return

        self._write_raw(raw)

    def _rehydrated_json(self, cloud_resp) -> bytes:
        """Return the response body decoded and with placeholders swapped back."""
        data = cloud_resp.content  # requests decodes gzip/deflate for us
        if _ph_to_value:
            try:
                b = json.loads(data)
                for blk in b.get("content", []):
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        blk["text"] = _rehydrate_str(blk.get("text", ""))
                data = json.dumps(b).encode("utf-8")
            except (json.JSONDecodeError, KeyError):
                pass
        return data

    def _passthrough(self) -> None:
        raw = self._read_body()
        cloud_url = _cfg["cloud_url"].rstrip("/") + self.path
        fwd_headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in HOP_BY_HOP}
        if raw:
            fwd_headers["Content-Length"] = str(len(raw))

        try:
            resp = requests.request(
                self.command, cloud_url,
                data=raw or None, headers=fwd_headers,
                stream=True, timeout=60,
            )
        except requests.RequestException as exc:
            self._send_json_error(502, f"Cloud error: {exc}")
            return

        data = resp.content  # decoded by requests; emit as-is with correct length
        self.send_response(resp.status_code)
        self._relay_headers(resp.headers)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True, help="Path to proxy.conf")
    args = parser.parse_args()

    _load_conf(Path(args.conf))

    port = int(_cfg.get("proxy_port", 8317))
    server = HTTPServer(("127.0.0.1", port), ProxyHandler)

    log.info("Proxy listening on http://127.0.0.1:%d", port)
    log.info("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Proxy stopped.")


if __name__ == "__main__":
    main()
