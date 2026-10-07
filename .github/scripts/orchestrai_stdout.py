#!/usr/bin/env python3
"""Publish complete test streams, redacting private data rather than lines.

This is a public-output boundary, not a grader parser. Raw control-plane logs,
agent transcript attachments and hardware inventories are not inputs to it.
"""

from __future__ import annotations

import ipaddress
import json
import re
import unicodedata
from collections.abc import Iterable
from urllib.parse import unquote, urlsplit

from orchestrai_logs import ANSI_RE, CASE_RE, private_log_values, redact_private_values

# Refuse oversized streams explicitly instead of publishing a misleading tail.
MAX_STREAM_BYTES = 16_000_000
UNAVAILABLE = "[Test stream unavailable: transport limit exceeded.]"
PUBLIC_DOMAINS = (
    "github.com",
    "githubusercontent.com",
    "githubassets.com",
    "nodejs.org",
    "nodesource.com",
    "npmjs.org",
    "npmjs.com",
    "pypi.org",
    "pythonhosted.org",
    "python.org",
    "huggingface.co",
    "hf.co",
    "pytorch.org",
    "ubuntu.com",
    "debian.org",
    "lemonade-server.ai",
    "gpuopen.com",
)
PUBLIC_AMD_HOSTS = {"amd.com", "www.amd.com", "rocm.docs.amd.com", "docs.amd.com"}
# Any RFC scheme can carry userinfo, including database and vendor schemes.
URL = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*:(?:\\?/){2}[^\s<>\"']+")
PRIVATE_HOST = re.compile(
    r"(?i)\b(?:[a-z0-9_-]+\.)+(?:amd\.com|internal|local|corp|lan)\b(?::\d+)?"
)
IDENTITY = re.compile(
    r"(?im)(?<![\w-])(?:host[ _-]?name|host|machine(?:[ _-]name)?|actor(?:[ _-]name)?|executionnode|"
    r"user[ _-]?name|computername|serial(?:[ _-]number)?|executing user|user|session[ _-]?id|job[ _-]?id|run[ _-]?id|"
    r"device[_-]?tags?|machine[_-]?tags?)\b(?P<key_quote>[\"']?)[ \t]*[:=][ \t]*"
)
MACHINE_NAME = re.compile(
    r"\b(?:[A-Z][A-Z0-9]+(?:-[A-Z0-9]+){2,}|"
    r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+-[A-Za-z0-9]*[0-9][A-Za-z0-9]*)\b(?!\.\d)"
)
PRIVATE_PATH = re.compile(
    r"(?i)(?P<quote>[\"'])(?P<quoted>(?:[a-z]:[\\/]|\\\\|"
    r"/(?:home|root|tmp|var|opt|mnt|srv|Users|private|workspace|builds|\[REDACTED\])/|\[REDACTED\][\\/])[^\r\n\"']+)(?P=quote)|"
    r"(?P<bare>(?<![a-z0-9])(?:[a-z]:[\\/](?:Program Files(?: \(x86\))?[\\/]|Users[\\/][^\\/\r\n<>\"']+[\\/])?|\\\\)[^\s<>\"']+|"
    r"(?<![\w:/])/(?:home|root|tmp|var|opt|mnt|srv|Users|private|workspace|builds|\[REDACTED\])(?=/|$|[\s<>\"'])(?:/[^\s<>\"']*)?|"
    r"\[REDACTED\][\\/][^\s<>\"']+)"
)
PUBLIC_TEST_PATH = re.compile(
    r"(?:testcases/)?L4-sys/skills/(?:linux|windows)/sys_func-skills_behavioral"
)
PUBLIC_TEST_SUFFIX = re.compile(r"(?:^|/)(" + PUBLIC_TEST_PATH.pattern + r")\Z")
FILE_NAME = re.compile(
    r"[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}\.(?:py|ps1|sh|js|mjs|cjs|exe|"
    r"json|yaml|yml|md|txt|log|whl|zip|gz|toml|cfg|ini|so|dll|pyd)(?::\d+(?::\d+)?)?",
    re.IGNORECASE,
)
COMMON_PATH_NAMES = {
    "skills",
    "skillscope",
    "venv",
    "artifacts",
    "dependencies",
    "bin",
    "scripts",
    "node_modules",
    "cache",
    "logs",
    "results",
    "site-packages",
    "python",
    "python3",
    "pip",
    "pip3",
    "nodejs",
}
CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?im)(?<![\w-])(?:[a-z0-9_-]*(?:api[_-]?key|password|passwd|secret|token|access[_-]key(?:[_-]id)?|private[_-]key|cookie)|"
    r"authorization|ocp-apim-subscription-key|anthropic_custom_headers|"
    r"x-api-key|subscription[_-]?key|set-cookie|credentials|"
    r"llm_gateway_(?:key|url|user))\b(?P<key_quote>[\"']?)[ \t]*[:=][ \t]*"
)
END_LINE = re.compile(r"[\r\n]")
FIELD_BOUNDARY = re.compile(
    r"[,;\r\n}\]]|[ \t]+(?=[\"']?[A-Za-z][A-Za-z0-9_-]*[\"']?[ \t]*[:=])"
)
JSON_DECODER = json.JSONDecoder()
JSON_FIELD = re.compile(r'[ \t\r\n]*"(?:\\.|[^"\\])*"[ \t\r\n]*:')
AUTH = re.compile(r"(?i)\b(?:bearer|basic)\s+[^\s,;\"']+")
CLI_SECRET = re.compile(
    r"(?i)(--(?:api[-_]key|password|passwd|secret|token|access[-_]key|subscription[-_]key)(?:\s+|=))[^\r\n]+"
)
OPAQUE = re.compile(r"(?<![\w])[A-Za-z0-9_+/=-]{28,}(?![\w])")
HEX_VALUE = re.compile(r"(?<![a-zA-Z0-9])[0-9a-fA-F]{28,}(?![a-zA-Z0-9])")
ALPHANUM_VALUE = re.compile(r"(?<![a-zA-Z0-9])[a-zA-Z0-9]{28,}(?![a-zA-Z0-9])")
UUID_VALUE = re.compile(
    r"(?<![a-zA-Z0-9])[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}(?![a-zA-Z0-9])"
)
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
ADDRESS = re.compile(r"(?<![\w])(?:\d{1,3}\.){3}\d{1,3}(?![\w])")
IPV6 = re.compile(r"(?<![\w:])(?:[0-9a-fA-F]{0,4}:){2,}[0-9a-fA-F:.%_-]*(?![\w:])")
MAC = re.compile(r"(?i)\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b")


def _public_host(host: str) -> bool:
    return (
        host in PUBLIC_AMD_HOSTS
        or any(
            host == domain or host.endswith("." + domain) for domain in PUBLIC_DOMAINS
        )
        or host in {"localhost", "127.0.0.1", "::1"}
    )


def _url(match: re.Match) -> str:
    try:
        # JSON may escape each slash without changing the URL's meaning.
        parsed = urlsplit(match[0].replace("\\/", "/"))
        if not _public_host((parsed.hostname or "").lower()) or "@" in unquote(
            parsed.netloc
        ):
            return "[PRIVATE URL REDACTED]"
        # Query strings/fragments may carry signed URLs, auth or tenant IDs.
        return match[0].split("?", 1)[0].split("#", 1)[0]
    except ValueError:
        return "[PRIVATE URL REDACTED]"


def _quoted_end(value: str, start: int) -> int:
    quote = value[start]
    index = start + 1
    while index < len(value):
        if value[index] == "\\":
            index += 2
        elif value[index] == quote:
            return index + 1
        else:
            index += 1
    # An unterminated secret string cannot safely reveal the remainder.
    return len(value)


def _line_end(value: str, start: int) -> int:
    end = END_LINE.search(value, start)
    return end.start() if end else len(value)


def _continued_end(value: str, start: int, end: int) -> int:
    # YAML blocks and folded headers can put a secret on indented next lines.
    line_start = value.rfind("\n", 0, start) + 1
    line_prefix = value[line_start:start]
    indent = len(line_prefix) - len(line_prefix.lstrip(" \t"))
    while end < len(value) and value[end] in "\r\n":
        next_start = end + (2 if value[end : end + 2] == "\r\n" else 1)
        next_end = value.find("\n", next_start)
        next_end = len(value) if next_end < 0 else next_end
        line = value[next_start:next_end]
        next_indent = len(line) - len(line.lstrip(" \t"))
        if line.strip() and next_indent <= indent:
            break
        end = next_end
    return end


def _assignment_end(
    value: str, start: int, *, credential: bool = False, structured: bool = False
) -> int:
    """Locate a scalar/container without consuming adjacent structured fields."""
    index = start
    while index < len(value) and value[index].isspace():
        index += 1
    if index == len(value):
        return index
    if value[index] in "\"'":
        end = _quoted_end(value, index)
        if credential and not structured:
            # Header/shell values may contain several quoted segments. Only a
            # quoted mapping key establishes a safe adjacent-field boundary.
            return _continued_end(value, start, _line_end(value, end))
        return end
    if value[index] in "[{":
        stack: list[str] = []
        while index < len(value):
            char = value[index]
            if char in "\"'":
                index = _quoted_end(value, index)
                continue
            if char in "[{":
                stack.append("]" if char == "[" else "}")
            elif char in "]}":
                if not stack or char != stack.pop():
                    return len(value)
                if not stack:
                    if credential and not structured:
                        return _continued_end(value, start, _line_end(value, index + 1))
                    return index + 1
            index += 1
        return len(value)
    # Plain assignments stop at field delimiters, or an adjacent assignment.
    boundary = END_LINE if credential and not structured else FIELD_BOUNDARY
    end = boundary.search(value, index)
    index = end.start() if end else len(value)
    return index if structured else _continued_end(value, start, index)


def _structured_assignment(value: str, match: re.Match) -> bool:
    """Require a JSON key, valid value and following object-field boundary."""
    if match["key_quote"] != '"' or value[match.start() - 1 : match.start()] != '"':
        return False
    before_key = match.start() - 2
    while before_key >= 0 and value[before_key].isspace():
        before_key -= 1
    if before_key < 0 or value[before_key] not in "{,":
        return False
    start = match.end()
    while start < len(value) and value[start].isspace():
        start += 1
    try:
        _, end = JSON_DECODER.raw_decode(value, start)
    except (ValueError, RecursionError):
        return False
    while end < len(value) and value[end].isspace():
        end += 1
    if end == len(value):
        return False
    if value[end] == ",":
        return bool(JSON_FIELD.match(value, end + 1))
    return value[end] == "}"


def _redact_assignments(value: str, pattern: re.Pattern, notice: str) -> str:
    parts: list[str] = []
    offset = 0
    for match in pattern.finditer(value):
        if match.start() < offset:
            continue
        if pattern is CREDENTIAL_ASSIGNMENT:
            line_start = value.rfind("\n", 0, match.start()) + 1
            line_end = value.find("\n", match.end())
            line = value[line_start : line_end if line_end >= 0 else len(value)].rstrip(
                "\r"
            )
            # Case IDs can mention api-key, but only this exact numeric harness
            # footer is attributable. Known secrets were already masked above.
            case = CASE_RE.fullmatch(line)
            if case and not case["error"]:
                continue
        end = _assignment_end(
            value,
            match.end(),
            credential=pattern is CREDENTIAL_ASSIGNMENT,
            structured=_structured_assignment(value, match),
        )
        raw = value[match.end() : end]
        scalar = raw.lstrip()
        quote = scalar[0] if scalar[:1] in {'"', "'"} else match["key_quote"]
        if not quote and scalar[:1] in {"[", "{"} and not scalar.startswith(notice):
            quote = '"'
        parts.append(value[offset : match.end()])
        parts.append(quote + notice + quote)
        # Retain line boundaries even when one secret spans several lines.
        parts.append("".join(re.findall(r"\r\n|\r|\n", raw)))
        offset = end
    parts.append(value[offset:])
    return "".join(parts)


def _opaque_token(value: str) -> bool:
    # Long keys need not mix letter cases/digits or use a large alphabet. Only
    # uniform repetitive diagnostics (e.g. a long line of x's) are exempt here.
    # This is a conservative shape rule, not proof that other text is secret-free.
    return len(set(value.lower())) >= 2


def _address(match: re.Match) -> str:
    try:
        address = ipaddress.ip_address(match[0].split("%", 1)[0])
    except ValueError:
        return match[0]  # Version numbers and timestamps are not addresses.
    return match[0] if address.is_loopback else "[ADDRESS REDACTED]"


def public_commits(environ: dict) -> set[str]:
    """Only workflow-verified public revisions can bypass opaque-value masking.

    Do not trust actor labels such as 'commit', or an allowlist in its manifest:
    a credential can also be forty hexadecimal characters.
    """
    return {
        value.lower()
        for key in ("PUBLIC_SKILLS_COMMIT", "PUBLIC_SKILLSCOPE_COMMIT")
        if isinstance(value := environ.get(key), str)
        and re.fullmatch(r"[0-9a-fA-F]{40}", value)
    }


def _path(match: re.Match) -> str:
    raw = match["quoted"] or match["bare"]
    path = raw.replace("\\", "/")
    trimmed = path.rstrip("),;].")
    punctuation = path[len(trimmed) :]
    basename = trimmed.rstrip("/").rsplit("/", 1)[-1]
    # Reveal no private directory hierarchy, share name or username. Retain a
    # plain filename/common tool name, or the fixed public adapter test path.
    public_test = PUBLIC_TEST_SUFFIX.search(trimmed)
    suffix = public_test[1] if public_test else ""
    if not suffix and (
        FILE_NAME.fullmatch(basename)
        or basename.lower() in COMMON_PATH_NAMES
        or re.fullmatch(r"(?i)python\d{2,3}|python3\.\d{1,2}", basename)
    ):
        suffix = basename
    result = "<local>" + ("/" + suffix if suffix else "") + punctuation
    quote = match["quote"] or ""
    return quote + result + quote


def sanitize_stream(
    raw: object,
    private_values: Iterable[str] = (),
    *,
    verified_commits: Iterable[str] = (),
) -> str:
    """Retain safe line context; redact recognized private data.

    Known values (including multiline secrets and driver-map URLs) are replaced
    before structural matching. The same boundary runs again at the verdict job.
    GitHub masking alone is insufficient: node-bound secrets are not repo secrets.
    """
    if not isinstance(raw, str):
        return ""
    if len(raw.encode("utf-8")) > MAX_STREAM_BYTES:
        return UNAVAILABLE + "\n"
    value = ANSI_RE.sub("", raw)
    value = "".join(
        char
        for char in value
        if char in "\n\r\t" or not unicodedata.category(char).startswith("C")
    )
    value = redact_private_values(value, private_values)
    # A missing END marker must not expose the remainder of a private key.
    value = re.sub(
        r"-----BEGIN [^\r\n]*PRIVATE KEY-----[\s\S]*?(?:-----END [^\r\n]*PRIVATE KEY-----|\Z)",
        "[PRIVATE KEY REDACTED]",
        value,
    )
    value = _redact_assignments(value, CREDENTIAL_ASSIGNMENT, "[REDACTED]")
    value = CLI_SECRET.sub(r"\1[REDACTED]", value)
    value = AUTH.sub("[AUTHORIZATION REDACTED]", value)
    value = re.sub(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", "[ACCESS KEY REDACTED]", value)
    value = URL.sub(_url, value)
    value = PRIVATE_HOST.sub(
        lambda match: (
            match[0]
            if _public_host(match[0].split(":", 1)[0].lower())
            else "[HOST REDACTED]"
        ),
        value,
    )
    value = _redact_assignments(value, IDENTITY, "[IDENTITY REDACTED]")
    value = MACHINE_NAME.sub("[IDENTITY REDACTED]", value)
    value = PRIVATE_PATH.sub(_path, value)
    value = EMAIL.sub("[EMAIL REDACTED]", value)
    value = ADDRESS.sub(_address, value)
    value = MAC.sub("[HARDWARE ADDRESS REDACTED]", value)
    value = IPV6.sub(_address, value)
    commits = {
        sha.lower()
        for sha in verified_commits
        if isinstance(sha, str) and re.fullmatch(r"[0-9a-fA-F]{40}", sha)
    }
    # Normalized filenames may contain credentials or opaque actor IDs.
    # Check their components too, not only the complete slash-containing token.
    value = HEX_VALUE.sub(
        lambda match: (
            match[0] if match[0].lower() in commits else "[OPAQUE VALUE REDACTED]"
        ),
        value,
    )
    value = UUID_VALUE.sub("[OPAQUE VALUE REDACTED]", value)
    value = ALPHANUM_VALUE.sub(
        lambda match: (
            "[OPAQUE VALUE REDACTED]"
            if match[0].lower() not in commits and _opaque_token(match[0])
            else match[0]
        ),
        value,
    )
    # The fixed public test path is not a base64 credential just because it
    # contains a slash, uppercase L, and the digit 4. Other slash-containing
    # high-entropy strings still receive credential masking.
    value = OPAQUE.sub(
        lambda match: (
            match[0]
            if match[0].lower() in commits
            or PUBLIC_TEST_PATH.fullmatch(match[0].lstrip("/"))
            else (
                "[OPAQUE VALUE REDACTED]"
                if re.fullmatch(r"[0-9a-fA-F]{28,}", match[0])
                or re.fullmatch(
                    r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", match[0]
                )
                or match[0].startswith(("sk-", "ghp_", "gho_", "github_pat_", "eyJ"))
                or (
                    re.search(r"[A-Z]", match[0])
                    and re.search(r"[a-z]", match[0])
                    and re.search(r"[0-9]", match[0])
                )
                else match[0]
            )
        ),
        value,
    )
    # Never allow actor output to issue Actions commands or forge groups/masks.
    return value.replace("::", ": :")


def public_streams(
    test: dict,
    private_values: Iterable[str] = (),
    *,
    verified_commits: Iterable[str] = (),
) -> dict:
    return {
        stream: sanitize_stream(
            test.get(stream), private_values, verified_commits=verified_commits
        )
        for stream in ("stdout", "stderr")
        if isinstance(test.get(stream), str)
    }


def manifest_streams(item: dict, environ: dict) -> dict:
    streams = item.get("public_streams")
    if not isinstance(streams, dict):
        return {}
    return public_streams(
        streams,
        private_log_values({}, {}, environ),
        verified_commits=public_commits(environ),
    )


def stream_coverage(streams: dict) -> str:
    if any(UNAVAILABLE in value for value in streams.values()):
        return "Incomplete (transport limit exceeded)"
    available = [stream for stream in ("stdout", "stderr") if stream in streams]
    if len(available) == 2:
        return "stdout + stderr received (redacted)"
    if available:
        return f"{available[0]} received; other stream unavailable"
    return "Test streams unavailable"
