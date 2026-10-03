#!/usr/bin/env python3
"""Publish complete test streams, redacting private data rather than lines.

This is a public-output boundary, not a grader parser. Raw control-plane logs,
agent transcript attachments and hardware inventories are not inputs to it.
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from collections.abc import Iterable
from urllib.parse import urlsplit

from orchestrai_logs import ANSI_RE, private_log_values

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
URL = re.compile(r"(?i)\b(?:https?|ssh|ftp|git)://[^\s<>\"']+")
PRIVATE_HOST = re.compile(
    r"(?i)\b(?:[a-z0-9_-]+\.)+(?:amd\.com|internal|local|corp|lan)\b(?::\d+)?"
)
IDENTITY = re.compile(
    r"(?im)\b(?:host[ _-]?name|host|machine(?:[ _-]name)?|actor(?:[ _-]name)?|executionnode|"
    r"user[ _-]?name|computername|serial(?:[ _-]number)?|executing user|user|session[ _-]?id|job[ _-]?id|run[ _-]?id|"
    r"device[_-]?tags?|machine[_-]?tags?)\b[\"']?\s*[:=]\s*[^\r\n]+"
)
MACHINE_NAME = re.compile(r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+){2,}\b")
PRIVATE_PATH = re.compile(
    r"(?i)(?P<quote>[\"'])(?P<quoted>(?:[a-z]:[\\/]|\\\\|"
    r"/(?:home|root|tmp|var|opt|mnt|srv|Users|private|workspace|builds)/)[^\r\n\"']+)(?P=quote)|"
    r"(?P<bare>(?<![a-z0-9])(?:[a-z]:[\\/](?:Program Files(?: \(x86\))?[\\/]|Users[\\/][^\\/\r\n<>\"']+[\\/])?|\\\\)[^\s<>\"']+|"
    r"(?<![\w:/])/(?:home|root|tmp|var|opt|mnt|srv|Users|private|workspace|builds)(?=/|$|[\s<>\"'])(?:/[^\s<>\"']*)?)"
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
    r"(?im)((?<![\w-])(?:[a-z0-9_]*(?:api[_-]?key|password|passwd|secret|token|access_key(?:_id)?|private_key|cookie)|"
    r"authorization|ocp-apim-subscription-key|anthropic_custom_headers|"
    r"x-api-key|subscription[_-]?key|set-cookie|credentials|"
    r"llm_gateway_(?:key|url|user))\b[\"']?\s*[:=]\s*)[^\r\n]+"
)
AUTH = re.compile(r"(?i)\b(?:bearer|basic)\s+[^\s,;\"']+")
CLI_SECRET = re.compile(
    r"(?i)(--(?:api[-_]key|password|passwd|secret|token|access[-_]key|subscription[-_]key)(?:\s+|=))[^\r\n]+"
)
OPAQUE = re.compile(r"(?<![\w])[A-Za-z0-9_+/=-]{28,}(?![\w])")
HEX_VALUE = re.compile(r"(?<![a-zA-Z0-9])[0-9a-fA-F]{28,}(?![a-zA-Z0-9])")
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
        parsed = urlsplit(match[0])
        if (
            not _public_host((parsed.hostname or "").lower())
            or parsed.username
            or parsed.password
        ):
            return "[PRIVATE URL REDACTED]"
        # Query strings/fragments may carry signed URLs, auth or tenant IDs.
        return match[0].split("?", 1)[0].split("#", 1)[0]
    except ValueError:
        return "[PRIVATE URL REDACTED]"


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
    """Retain every line; redact credentials and private infrastructure data.

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
    for private in sorted(set(private_values), key=len, reverse=True):
        if isinstance(private, str) and len(private) >= 4:
            value = value.replace(private, "[REDACTED]")
    # A missing END marker must not expose the remainder of a private key.
    value = re.sub(
        r"-----BEGIN [^\r\n]*PRIVATE KEY-----[\s\S]*?(?:-----END [^\r\n]*PRIVATE KEY-----|\Z)",
        "[PRIVATE KEY REDACTED]",
        value,
    )
    value = CREDENTIAL_ASSIGNMENT.sub(r"\1[REDACTED]", value)
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
    value = IDENTITY.sub("[IDENTITY REDACTED]", value)
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
