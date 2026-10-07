"""Public, fixed-schema evidence that the Windows adapter checked its driver."""

from __future__ import annotations

import re

from orchestrai_logs import ANSI_RE, TIMESTAMP_PREFIX_RE


MARKER = re.compile(
    r"\[skills-readiness\] windows_amd_display="
    r"(ready|missing|unhealthy|query_failed) compute_backend=not_probed"
)
READINESS_DIAGNOSTICS = {
    "missing": "Windows AMD display adapter is missing.",
    "unhealthy": "Windows AMD display adapter or driver is unhealthy.",
    "query_failed": "Windows AMD display readiness query failed.",
    "not_reported": "Windows GPU readiness was not confirmed by the test adapter.",
    "invalid": "Windows GPU readiness evidence was invalid.",
}
STATES = {"ready", *READINESS_DIAGNOSTICS}


def safe_readiness(item: dict) -> dict:
    """Never copy hardware inventory or arbitrary adapter fields into reports."""
    value = item.get("readiness")
    if not isinstance(value, dict):
        return {}
    state = value.get("windows_amd_display")
    if not isinstance(state, str) or state not in STATES:
        return {}
    if value.get("compute_backend") != "not_probed":
        return {}
    return {"windows_amd_display": state, "compute_backend": "not_probed"}


def readiness_description(item: dict) -> str:
    readiness = safe_readiness(item)
    if not readiness:
        return ""
    state = readiness["windows_amd_display"]
    label = "ready" if state == "ready" else state.replace("_", " ")
    return (
        f"Windows AMD display driver: {label}. "
        "Compute backend was not probed by preflight; GPU workload coverage "
        "depends on the selected evaluation."
    )


def enforce_windows_readiness(item: dict, test: dict) -> None:
    """A current Windows plan cannot accept evidence from a preflight-free adapter.

    This marker describes the adapter's device checks, not evidence of actual
    kernel execution. Only complete, standalone adapter records are accepted.
    """
    if item.get("os") != "Windows" or item.get("gpu_preflight_required") is not True:
        return
    records: list[str] = []
    invalid = False
    for stream in ("stdout", "stderr"):
        value = test.get(stream)
        if not isinstance(value, str):
            continue
        for raw in value.splitlines():
            line = TIMESTAMP_PREFIX_RE.sub("", ANSI_RE.sub("", raw)).strip()
            if not line.startswith("[skills-readiness]"):
                continue
            match = MARKER.fullmatch(line)
            if match:
                records.append(match[1])
            else:
                invalid = True
    state = (
        "invalid"
        if invalid or len(records) > 1
        else records[0]
        if records
        else "not_reported"
    )
    item["readiness"] = {
        "windows_amd_display": state,
        "compute_backend": "not_probed",
    }
    if state != "ready" and item.get("status") in {"passed", "failed"}:
        item["status"] = "error"
        item["error"] = READINESS_DIAGNOSTICS[state]
