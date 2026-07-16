#!/usr/bin/env python3
"""
local-ai-privacy: shared redaction engine.

Used by redact.py (the file redaction gate): writes redacted copies of
files so only masked content ever enters a cloud conversation. Kept as a
separate module so other front-ends (a CLI, a proxy, a pipeline stage) can
reuse the same detection without touching file handling.

Detection is two-pass, feeding one placeholder map:
  1. regex_discover(): deterministic patterns for structured PII (SSN, email,
     phone, credit card via Luhn, IPv4, MRN/DOB-in-context). redact.py also
     re-runs these on its OUTPUT as a scrub check and withholds any file that
     still matches.
  2. LemonadeClient.discover(): a local LLM finds unstructured PII (names,
     addresses, freeform secrets) chunk by chunk.

Standard library only (urllib), so redact.py needs no pip installs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.error
import urllib.request

log = logging.getLogger("redaction")

# Local model context is small; discover PII in line-aligned chunks of this size.
DISCOVERY_CHUNK_CHARS = 8000

# ---------------------------------------------------------------------------
# Labels and placeholders
# ---------------------------------------------------------------------------

_IDX_RE = re.compile(r"_\d+$")

# Matches a token this engine itself produced, e.g. [SSN_1] or [INTERNAL_HOST_12].
# These must never be treated as new PII, or output gets re-redacted and
# placeholders nest/relabel ([SSN_1] -> [SSN_2] -> ...).
PLACEHOLDER_RE = re.compile(r"^\[[A-Z][A-Z0-9_]*_\d+\]$")


def clean_label(label) -> str:
    """Normalise a freeform model label to a clean UPPERCASE TYPE.

    Handles `EMAIL`, `email`, `Email Address`, `[SSN]`, `SSN_0` -> `EMAIL`/`SSN`.
    Capped at 24 chars so a runaway label can't smuggle content into receipts.
    """
    k = str(label).strip().strip("[]").strip()
    k = _IDX_RE.sub("", k)            # drop any trailing _<n> index
    k = re.sub(r"[^A-Za-z_]", "_", k) # non-letters become underscores
    k = re.sub(r"_+", "_", k).strip("_")
    return (k.upper() or "SECRET")[:24]


def ph_type(ph: str) -> str:
    """`[SSN_1]` -> `SSN`."""
    return _IDX_RE.sub("", ph.strip("[]")).upper()


# ---------------------------------------------------------------------------
# Pass 1: deterministic detectors for structured PII
# ---------------------------------------------------------------------------

def _luhn_ok(value: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", value)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _ipv4_ok(value: str) -> bool:
    if any(int(p) > 255 for p in value.split(".")):
        return False
    return value not in ("127.0.0.1", "0.0.0.0", "255.255.255.255")


# (TYPE, pattern, validator). If the pattern has a capture group, group 1 is
# the sensitive value (used for context-anchored patterns like "MRN: <value>").
REGEX_DETECTORS: list[tuple[str, re.Pattern, object]] = [
    ("SSN", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"), None),
    ("EMAIL", re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None),
    # Requires separators between groups, so ISO dates (4-2-2) don't match.
    ("PHONE", re.compile(
        r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"), None),
    ("CARD", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"), _luhn_ok),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), _ipv4_ok),
    ("MRN", re.compile(
        r"\b(?:MRN|medical record(?:\s+(?:number|no\.?))?)\s*[:#]?\s*"
        r"([A-Za-z]*\d[A-Za-z0-9-]{4,})", re.IGNORECASE), None),
    ("DOB", re.compile(
        r"\b(?:DOB|date of birth|birth date)\s*[:#]?\s*"
        r"(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2}"
        r"|[A-Z][a-z]+ \d{1,2},? \d{4})", re.IGNORECASE), None),
]


def regex_discover(text: str) -> list[tuple[str, str]]:
    """Return deterministic (value, TYPE) pairs found in text.

    Also serves as the post-redaction scrub check: run it on masked output —
    any hit means something structured slipped through and the file must be
    withheld.
    """
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for typ, pat, check in REGEX_DETECTORS:
        for m in pat.finditer(text):
            val = m.group(1) if m.groups() else m.group(0)
            if val in seen:
                continue
            if check is not None and not check(val):
                continue
            if PLACEHOLDER_RE.match(val.strip()):
                continue
            seen.add(val)
            found.append((val, typ))
    return found


# ---------------------------------------------------------------------------
# Pass 2: local LLM discovery via Lemonade
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json(content: str) -> str:
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


def chunk_by_lines(text: str, size: int = DISCOVERY_CHUNK_CHARS) -> list[str]:
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


class LemonadeClient:
    """Minimal OpenAI-compatible chat client for a local Lemonade server."""

    def __init__(self, base_url: str, model: str, prompt: str,
                 max_tokens: int = 8192, retries: int = 3,
                 timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.retries = max(1, retries)
        self.timeout = timeout
        self.api_key = os.environ.get("LEMONADE_API_KEY", "")

    def _get_json(self, path: str, timeout: int):
        req = urllib.request.Request(self.base_url + path)
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def health(self) -> bool:
        try:
            return self._get_json("/api/v1/health", timeout=5) is not None
        except Exception:
            return False

    def installed_models(self) -> list[str] | None:
        """Model ids Lemonade can serve right now, or None if unknowable."""
        try:
            data = self._get_json("/api/v1/models", timeout=10)
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return None

    def discover(self, text: str) -> tuple[list[tuple[str, str]], bool]:
        """Ask the local model for sensitive (value, TYPE) pairs in one chunk.

        Returns (pairs, ok). ok is False only when the model was unreachable
        or kept returning unparseable output across all retries — the caller
        MUST fail closed (never let the original text reach the cloud). The
        model is non-deterministic, so each retry usually parses.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            # Headroom so a larger model (or a thinking model's reasoning)
            # can't get truncated into invalid JSON, which would fail closed.
            "max_tokens": self.max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(1, self.retries + 1):
            try:
                req = urllib.request.Request(
                    self.base_url + "/api/v1/chat/completions",
                    data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read())
                raw = resp["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                log.warning("Lemonade call failed (attempt %d/%d): %s",
                            attempt, self.retries, type(exc).__name__)
                continue

            try:
                result = json.loads(extract_json(raw))
            except json.JSONDecodeError:
                log.warning("Non-JSON from redaction model (attempt %d/%d)",
                            attempt, self.retries)
                continue

            # Preferred contract: a JSON array of {"text","label"}. Stay
            # tolerant of a dict wrapper or {"mapping": {...}} just in case.
            if isinstance(result, list):
                items = result
            elif isinstance(result, dict) and "mapping" in result:
                items = [{"text": v, "label": k}
                         for k, v in result["mapping"].items()]
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
                # Skip <3-char spans: masking a 1-2 char substring would
                # smear across unrelated text.
                if isinstance(val, str) and len(val.strip()) >= 3:
                    pairs.append((val, clean_label(label)))
            return pairs, True

        log.error("Redaction failed after %d attempt(s) — fail closed",
                  self.retries)
        return [], False


# ---------------------------------------------------------------------------
# Session: the value -> placeholder map both passes feed
# ---------------------------------------------------------------------------

_HONORIFICS = {"dr", "mr", "mrs", "ms", "miss", "prof", "md", "do", "rd",
               "rn", "np", "pa", "phd", "jr", "sr", "ii", "iii"}

# Placeholder types that denote a person. Only these get token aliases —
# deriving tokens from e.g. FACILITY values would mask words like "Medicine".
_PERSON_TYPE_HINTS = ("NAME", "PATIENT", "DOCTOR", "PROVIDER", "PRESCRIBER",
                      "PERSON", "PHYSICIAN", "NURSE", "SPOUSE")


class RedactionSession:
    """One placeholder map shared across every text ingested in a session, so
    the same real value always maps to the same placeholder ([SSN_1], ...)
    across all files in a run. Values live only in RAM; nothing here is ever
    written to disk."""

    def __init__(self):
        self.value_to_ph: dict[str, str] = {}   # "123-45-6789" -> "[SSN_1]"
        self.ph_to_value: dict[str, str] = {}   # "[SSN_1]"     -> "123-45-6789"
        self.type_counts: dict[str, int] = {}   # "SSN"         -> 1
        self._seen_chunks: set[str] = set()     # sha256 of chunks already scanned
        # Word-boundary aliases derived from person names: "Margaret" and
        # "MW" both -> "[NAME_1]" once "Margaret Walsh" is known. The LLM
        # reliably flags a person SOMEWHERE (a "Patient:" field) but misses
        # casual prose mentions ("Hi Margaret", a "— AK" sign-off); these
        # make every later mention deterministic.
        self._token_aliases: dict[str, str] = {}

    def _derive_name_tokens(self, value: str, ph: str) -> None:
        if not any(h in ph_type(ph) for h in _PERSON_TYPE_HINTS):
            return
        words = [w for w in re.split(r"[^A-Za-z]+", value)
                 if w and w.lower() not in _HONORIFICS]
        for w in words:
            if len(w) >= 3 and w[0].isupper() and w not in self.value_to_ph:
                self._token_aliases.setdefault(w, ph)
        if len(words) >= 2:
            initials = "".join(w[0] for w in words if w[0].isupper())
            if 2 <= len(initials) <= 4:
                self._token_aliases.setdefault(initials, ph)

    def placeholder_for(self, value: str, typ: str) -> str:
        """Return the stable placeholder for a value, minting one if unseen."""
        if value in self.value_to_ph:
            return self.value_to_ph[value]
        # Never redact a token we already produced: the bracketed form
        # "[SSN_1]", the bare inner form "SSN_1" the model sometimes extracts,
        # or any current placeholder value.
        v = value.strip()
        if (value in self.ph_to_value
                or PLACEHOLDER_RE.match(v)
                or f"[{v.strip('[]')}]" in self.ph_to_value):
            return value
        typ = (typ or "SECRET").upper()
        n = self.type_counts.get(typ, 0) + 1
        self.type_counts[typ] = n
        ph = f"[{typ}_{n}]"
        self.value_to_ph[value] = ph
        self.ph_to_value[ph] = value
        self._derive_name_tokens(value, ph)
        return ph

    def mask(self, s: str) -> str:
        """Replace every known real value in s with its placeholder (longest
        first so overlapping values don't clobber each other), then mask
        derived name tokens at word boundaries."""
        if not s:
            return s
        for val in sorted(self.value_to_ph, key=len, reverse=True):
            if val and val in s:
                s = s.replace(val, self.value_to_ph[val])
        # Token pass: word-boundary regex so "AK" never fires inside "LEAK",
        # and lookarounds also exclude [ _ ] so a token can't corrupt an
        # already-placed placeholder like [NAME_1].
        for tok in sorted(self._token_aliases, key=len, reverse=True):
            if tok in s:
                s = re.sub(rf"(?<![A-Za-z\[]){re.escape(tok)}(?![A-Za-z_\]])",
                           self._token_aliases[tok], s)
        return s

    def ingest(self, text: str, llm: LemonadeClient | None) -> bool:
        """Discover sensitive values in text and add them to the map.

        Runs the regex pass always and the LLM pass when a client is given.
        Returns False if LLM discovery could not be completed for any chunk —
        the caller must fail closed. Chunks already scanned are skipped by
        hash, so repeated content isn't re-scanned.
        """
        for chunk in chunk_by_lines(text):
            h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            if h in self._seen_chunks:
                continue
            for value, typ in regex_discover(chunk):
                self.placeholder_for(value, typ)
            if llm is not None:
                pairs, ok = llm.discover(chunk)
                if not ok:
                    return False
                for value, typ in pairs:
                    self.placeholder_for(value, typ)
            self._seen_chunks.add(h)
        return True

    def alias_variants_in(self, s: str) -> None:
        """Register case/separator variants of known values that appear in s,
        mapping them to the SAME placeholder as the original.

        Filenames echo identity in mangled form: "John Smith" in a file's
        content shows up as "john-smith-labs.csv" in its name. Exact-string
        masking can't bridge that, so before masking a path, call this to
        catch lowercase and -/_/. joined variants of every known value.
        """
        low = s.lower()
        for val, ph in list(self.value_to_ph.items()):
            base = val.lower()
            for sep in ("-", "_", ".", "", " "):
                var = base.replace(" ", sep)
                if len(var) >= 3 and var in low:
                    idx = low.find(var)
                    actual = s[idx:idx + len(var)]
                    if actual not in self.value_to_ph:
                        self.value_to_ph[actual] = ph
        # Lowercased single name tokens too: "margaret-notes.txt".
        for tok, ph in list(self._token_aliases.items()):
            var = tok.lower()
            if len(var) >= 3 and var in low:
                idx = low.find(var)
                actual = s[idx:idx + len(var)]
                if actual not in self.value_to_ph:
                    self.value_to_ph[actual] = ph

    def counts_in(self, text: str) -> dict[str, int]:
        """Distinct known values present in text, tallied by TYPE — safe to
        print or send to the cloud (types and counts only, never values)."""
        counts: dict[str, int] = {}
        if not text:
            return counts
        for val, ph in self.value_to_ph.items():
            if val in text:
                t = ph_type(ph)
                counts[t] = counts.get(t, 0) + 1
        return counts
