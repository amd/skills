#!/usr/bin/env python3
"""
local-ai-privacy: redacting proxy server.

Started automatically by start.py — do not run this directly.

Reads config from ~/.claude/skills/local-ai-privacy/proxy.conf:
  {
    "cloud_url":       "https://llm-api.amd.com/Anthropic",
    "lemonade_url":    "http://localhost:13305",
    "redaction_model": "Qwen3-1.7B-GGUF",
    "proxy_port":      8080
  }

For every POST /v1/messages from Claude Code:
  1. Extracts user message text.
  2. Calls local Lemonade model to redact sensitive content.
  3. Rewrites the request body with masked text.
  4. Forwards the clean request to the real cloud endpoint.
  5. Streams the response back, re-substituting placeholders in text deltas.

GET /__proxy_health returns 200 so start.py can verify the proxy is up.
All other endpoints are passed through unchanged (e.g. GET /v1/models).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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
# Redaction helpers
# ---------------------------------------------------------------------------

def _load_redaction_prompt() -> str:
    if not REDACTION_PROMPT_FILE.exists():
        raise FileNotFoundError(f"Missing: {REDACTION_PROMPT_FILE}")
    return REDACTION_PROMPT_FILE.read_text(encoding="utf-8").strip()


def _redact(text: str) -> tuple[str, dict[str, str]]:
    """Send text to local Lemonade. Returns (masked_text, mapping)."""
    url = _cfg["lemonade_url"].rstrip("/") + "/api/v1/chat/completions"
    redaction_prompt = _load_redaction_prompt()

    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("LEMONADE_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": _cfg["redaction_model"],
        "messages": [
            {"role": "system", "content": redaction_prompt},
            {"role": "user",   "content": text},
        ],
        "temperature": 0,
        "max_tokens": 4096,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Lemonade call failed (%s) — forwarding original", exc)
        return text, {}

    content = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if model wrapped the JSON
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(l for l in lines if not l.startswith("```")).strip()

    try:
        result  = json.loads(content)
        masked  = result.get("masked", text)
        mapping = result.get("mapping", {})
        if mapping:
            log.info("Redacted entities: %s", result.get("entities", list(mapping.keys())))
        else:
            log.info("No sensitive entities found")
        return masked, mapping
    except json.JSONDecodeError:
        log.warning("Non-JSON from redaction model — forwarding original")
        return text, {}


def _extract_text(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        c = msg.get("content", "")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    parts.append(blk.get("text", ""))
    return "\n".join(parts)


def _apply_mask(messages: list[dict], mapping: dict[str, str]) -> list[dict]:
    if not mapping:
        return messages
    reverse = {v: f"[{k}]" for k, v in mapping.items()}

    def _m(s: str) -> str:
        for orig, ph in reverse.items():
            s = s.replace(orig, ph)
        return s

    out = []
    for msg in messages:
        if msg.get("role") != "user":
            out.append(msg)
            continue
        msg = dict(msg)
        c = msg.get("content", "")
        if isinstance(c, str):
            msg["content"] = _m(c)
        elif isinstance(c, list):
            msg["content"] = [
                {**blk, "text": _m(blk.get("text", ""))}
                if isinstance(blk, dict) and blk.get("type") == "text"
                else blk
                for blk in c
            ]
        out.append(msg)
    return out


def _resubstitute(text: str, mapping: dict[str, str]) -> str:
    for k, v in mapping.items():
        text = text.replace(f"[{k}]", v)
    return text


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class ProxyHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # handled by our logger

    def _send_json_error(self, code: int, msg: str) -> None:
        body = json.dumps({"error": {"type": "proxy_error", "message": msg}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # Health check so start.py can poll
    def _handle_health(self) -> None:
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/__proxy_health":
            self._handle_health()
        else:
            self._passthrough()

    def do_POST(self):
        if self.path.rstrip("/").endswith("/messages"):
            self._handle_messages()
        else:
            self._passthrough()

    def _handle_messages(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw    = self.rfile.read(length)

        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json_error(400, "Invalid JSON")
            return

        messages = body.get("messages", [])
        stream   = body.get("stream", False)

        # --- redact user messages ---
        text = _extract_text(messages)
        mapping: dict[str, str] = {}
        if text.strip():
            try:
                _, mapping = _redact(text)
                if mapping:
                    body["messages"] = _apply_mask(messages, mapping)
            except Exception as exc:
                log.warning("Redaction error (%s) — forwarding original", exc)

        # --- forward to cloud ---
        cloud_url = _cfg["cloud_url"].rstrip("/") + "/v1/messages"
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

        # --- relay response ---
        self.send_response(cloud_resp.status_code)
        for k, v in cloud_resp.headers.items():
            if k.lower() in ("transfer-encoding", "connection"):
                continue
            self.send_header(k, v)
        self.end_headers()

        if stream:
            self._stream_sse(cloud_resp, mapping)
        else:
            self._relay_json(cloud_resp, mapping)

    def _stream_sse(self, cloud_resp, mapping: dict) -> None:
        try:
            for line in cloud_resp.iter_lines(decode_unicode=True):
                out = line
                if mapping and line and line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str and data_str != "[DONE]":
                        try:
                            evt = json.loads(data_str)
                            if evt.get("type") == "content_block_delta":
                                delta = evt.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    delta["text"] = _resubstitute(
                                        delta.get("text", ""), mapping)
                                    evt["delta"] = delta
                                    out = "data: " + json.dumps(evt)
                        except (json.JSONDecodeError, KeyError):
                            pass
                self.wfile.write((out + "\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _relay_json(self, cloud_resp, mapping: dict) -> None:
        try:
            data = cloud_resp.content
            if mapping:
                try:
                    b = json.loads(data)
                    for blk in b.get("content", []):
                        if isinstance(blk, dict) and blk.get("type") == "text":
                            blk["text"] = _resubstitute(blk.get("text", ""), mapping)
                    data = json.dumps(b).encode("utf-8")
                except (json.JSONDecodeError, KeyError):
                    pass
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _passthrough(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw    = self.rfile.read(length) if length else b""

        cloud_url   = _cfg["cloud_url"].rstrip("/") + self.path
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

        self.send_response(resp.status_code)
        for k, v in resp.headers.items():
            if k.lower() in ("transfer-encoding", "connection"):
                continue
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(resp.content)
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

    port = int(_cfg.get("proxy_port", 8080))
    server = HTTPServer(("127.0.0.1", port), ProxyHandler)

    log.info("Proxy listening on http://127.0.0.1:%d", port)
    log.info("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Proxy stopped.")


if __name__ == "__main__":
    main()
