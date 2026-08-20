#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""
Does vLLM support this model's architecture? -- so the skill checks real vLLM
support instead of blanket-blocking multimodal.

Reads the model's `architectures` from its HF config.json, then checks them
against vLLM's model registry for the pinned vLLM version. The registry comes
from the version-pinned registry.py on GitHub (no vLLM install needed); if that
is unreachable it falls back to an importable local `vllm`. Generation endpoints
(text + multimodal) are supported; pooling/embedding/reranker and non-LLM
architectures are not chat/completion endpoints and are rejected.

It also inspects the model repo for a chat template (`chat_template.jinja`, or
`chat_template` in `tokenizer_config.json`) and picks the client endpoint:
`/v1/chat/completions` when a usable template is present, else `/v1/completions`
for raw prompts. A multimodal model with no usable template cannot serve and is
rejected (`launchable: false`).

    check_model.py --model-id Qwen/Qwen3-0.6B
    check_model.py --model-id <id> --vllm-version 0.25.1

Exit 0 if vLLM serves it as a generation endpoint (or support is undeterminable
-- launch confirms), 1 if it is positively unsupported OR a multimodal model has
no usable chat template. JSON to stdout. Env: HF_TOKEN for gated models.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

HF = "https://huggingface.co"
GH_RAW = "https://raw.githubusercontent.com/vllm-project/vllm"
REG_PATH = "vllm/model_executor/models/registry.py"
TOKENIZER_CFG = "tokenizer_config.json"
CHAT_TEMPLATE_FILE = "chat_template.jinja"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "epyc.json"
DEFAULT_VLLM_VERSION = json.loads(DATA_PATH.read_text(encoding="utf-8"))["vllm_version"]

# registry.py dict name -> kind we care about
_SECTIONS = {
    "_TEXT_GENERATION_MODELS": "text",
    "_TRANSFORMERS_BACKEND_MODELS": "text",
    "_MULTIMODAL_MODELS": "multimodal",
    "_EMBEDDING_MODELS": "pooling",
    "_POOLING_MODELS": "pooling",
    "_CROSS_ENCODER_MODELS": "pooling",
}


def _get(url, token=None):
    """GET text from a URL. Returns (text, error_message)."""
    headers = {"User-Agent": "check-model/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
            return r.read().decode("utf-8"), None
    except urllib.error.HTTPError as e:
        return None, {401: "not found or gated (set HF_TOKEN)",
                      403: "access denied -- accept the model license on HuggingFace",
                      404: "not found"}.get(e.code, f"HTTP {e.code}")
    except Exception as e:
        return None, str(e)


def model_architectures(model, rev, token):
    """Architectures declared in the model's HF config.json. Returns (list, error)."""
    text, err = _get(f"{HF}/{model}/resolve/{rev}/config.json", token)
    if text is None:
        return None, err
    try:
        cfg = json.loads(text)
    except ValueError:
        return None, "config.json is not valid JSON"
    return cfg.get("architectures") or [], None


def classify_template_field(ct):
    """Classify a tokenizer_config `chat_template` value (pure).

    Returns (status, selected_name, names):
    - a non-empty string -> ("present", None, [])
    - a list of {name, template} -> "present" if it has a single template or a
      "default", else "ambiguous" (multiple named, no default)
    - anything else / empty -> ("absent", None, [])
    """
    if isinstance(ct, str) and ct.strip():
        return "present", None, []
    if isinstance(ct, list) and ct:
        names = [e.get("name") for e in ct if isinstance(e, dict) and e.get("name")]
        if "default" in names:
            return "present", "default", names
        if len(names) == 1:
            return "present", names[0], names
        return "ambiguous", None, names
    return "absent", None, []


def chat_template_info(model, rev, token):
    """Inspect the model repo for a usable chat template. Returns a dict with
    `status` (present/ambiguous/absent/unknown), `source`, `selected_name`, `names`.

    A standalone `chat_template.jinja` (Transformers v5) takes precedence over the
    `chat_template` field in `tokenizer_config.json`. When neither file can be read
    (gated/offline) the status is `unknown`."""
    tmpl, terr = _get(f"{HF}/{model}/resolve/{rev}/{CHAT_TEMPLATE_FILE}", token)
    if tmpl is not None and tmpl.strip():
        return {"status": "present", "source": CHAT_TEMPLATE_FILE, "selected_name": None, "names": []}

    cfg_text, cerr = _get(f"{HF}/{model}/resolve/{rev}/{TOKENIZER_CFG}", token)
    if cfg_text is None:
        return {"status": "unknown", "source": None, "selected_name": None, "names": [],
                "detail": cerr or terr or "no tokenizer_config.json"}
    cfg = json.loads(cfg_text) if cfg_text.strip().startswith("{") else {}
    status, selected, names = classify_template_field(cfg.get("chat_template"))
    return {"status": status, "source": TOKENIZER_CFG if status != "absent" else None,
            "selected_name": selected, "names": names}


def endpoints_for(kind, template_status):
    """Pick client endpoints from modality + template status (pure).

    Returns (supported_endpoints, primary_endpoint, needs_template):
    - multimodal needs a chat template; without one it cannot serve (needs_template).
    - text with a usable template -> chat preferred, completions also available.
    - text without an auto-usable template (absent/ambiguous/unknown) -> completions
      (chat still possible via an explicit --chat-template)."""
    if kind == "multimodal":
        if template_status == "present":
            return ["chat_completions"], "chat_completions", False
        return [], None, True
    if template_status == "present":
        return ["chat_completions", "completions"], "chat_completions", False
    return ["completions"], "completions", False


def registry_from_github(version):
    """Parse vLLM's registry.py at v<version>. Returns ({arch: kind}, source) or (None, err)."""
    src, err = _get(f"{GH_RAW}/v{version}/{REG_PATH}")
    if src is None:
        return None, err
    reg, cur = {}, None
    for line in src.splitlines():
        s = line.strip()
        sec = re.match(r"^(_[A-Z0-9_]+_MODELS)\s*(?::[^=]+)?=\s*\{", s)
        if sec:
            cur = _SECTIONS.get(sec.group(1))
            continue
        if s.startswith("}"):
            cur = None
            continue
        if cur:
            key = re.match(r'^"([A-Za-z0-9_]+)"\s*:', s)
            if key:
                reg[key.group(1)] = cur
    return (reg or None), (f"github:v{version}" if reg else "registry.py had no parseable archs")


def registry_from_local():
    """Coarse fallback: an importable local `vllm` (text vs multimodal). Returns ({arch: kind}, source) or (None, None)."""
    snippet = (
        "import json;"
        "from vllm import ModelRegistry as R;"
        "a=list(R.get_supported_archs());"
        "mm=set(x for x in a if R.is_multimodal_model([x]));"
        "print(json.dumps({'archs':a,'mm':list(mm)}))"
    )
    r = subprocess.run(["python", "-c", snippet], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, timeout=60)
    if r.returncode != 0 or not r.stdout.strip():
        return None, None
    try:
        d = json.loads(r.stdout)
    except ValueError:
        return None, None
    mm = set(d.get("mm", []))
    return {a: ("multimodal" if a in mm else "text") for a in d.get("archs", [])}, "vllm-import"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument(
        "--vllm-version",
        default=DEFAULT_VLLM_VERSION,
        help=f"registry version (default: {DEFAULT_VLLM_VERSION} from data/epyc.json)",
    )
    a = p.parse_args()
    token = os.environ.get("HF_TOKEN", "")

    archs, aerr = model_architectures(a.model_id, a.revision, token)
    if not archs:
        # Cannot read the config (gated/offline) -- do not positively block; the
        # gating check and launch will catch real problems.
        print(json.dumps({"model_id": a.model_id, "supported": None, "kind": "undetermined",
                           "message": f"Could not read architectures ({aerr or 'none declared'}); support unverified. "
                                      "If gated, set HF_TOKEN. This does not bypass the gating/launch checks."}, indent=2))
        sys.exit(0)

    reg, source = registry_from_github(a.vllm_version)
    if reg is None:
        reg, source = registry_from_local()
    if reg is None:
        print(json.dumps({"model_id": a.model_id, "architectures": archs, "supported": None,
                           "kind": "undetermined",
                           "message": "Could not load vLLM's model registry (no network and no importable vllm); "
                                      "support unverified. vLLM confirms support at load (no-retry rule applies)."}, indent=2))
        sys.exit(0)

    kinds = [reg.get(arch) for arch in archs]
    known = [k for k in kinds if k]
    out = {"model_id": a.model_id, "architectures": archs, "registry_source": source}

    if not known:
        out.update(supported=False, kind="unsupported",
                   message=f"vLLM has no registry entry for {archs}; it cannot serve this model on any backend. Stop.")
        print(json.dumps(out, indent=2))
        sys.exit(1)

    if any(k in ("text", "multimodal") for k in known):
        kind = "multimodal" if "multimodal" in known else "text"
        tmpl = chat_template_info(a.model_id, a.revision, token)
        eps, primary, needs_tmpl = endpoints_for(kind, tmpl["status"])
        out.update(supported=True, kind=kind, chat_template=tmpl,
                   supported_endpoints=eps, primary_endpoint=primary)

        if kind == "multimodal" and needs_tmpl:
            out.update(launchable=False,
                       message=(f"{archs} is a multimodal model but no usable chat template was found "
                                f"(chat_template.status={tmpl['status']}). vLLM needs a chat template to "
                                "serve it; pass --chat-template <file> or choose a model that ships one. Stop."))
            print(json.dumps(out, indent=2))
            sys.exit(1)

        if primary == "chat_completions":
            msg = (f"vLLM supports {archs} as a {kind} generation endpoint; a chat template is present "
                   f"({tmpl['source']}). Serve /v1/chat/completions.")
        elif tmpl["status"] == "ambiguous":
            msg = (f"vLLM supports {archs} as a {kind} generation endpoint, but it ships multiple named chat "
                   f"templates with no 'default' ({tmpl['names']}). /v1/completions works now; for chat, pass "
                   "--chat-template <file> or select one of those names.")
        elif tmpl["status"] == "unknown":
            msg = (f"vLLM supports {archs} as a {kind} generation endpoint, but the chat template could not be "
                   "read (gated/offline). Defaulting to /v1/completions; if this is an instruct model with a "
                   "template, /v1/chat/completions will also work.")
        else:
            msg = (f"vLLM supports {archs} as a {kind} generation endpoint, but no chat template was found. "
                   "Serve base text via /v1/completions; for chat, pass --chat-template <file>.")
        if kind == "multimodal":
            msg += " A multimodal arch may still hit a GPU-only kernel on CPU; that surfaces at load (no-retry rule applies)."
        out.update(launchable=True, message=msg)
        print(json.dumps(out, indent=2))
        sys.exit(0)

    out.update(supported=False, kind="pooling",
               message=f"{archs} is a pooling/embedding/reranker model in vLLM, not a chat/completion endpoint. Stop.")
    print(json.dumps(out, indent=2))
    sys.exit(1)


if __name__ == "__main__":
    main()
