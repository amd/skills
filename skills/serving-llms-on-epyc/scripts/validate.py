#!/usr/bin/env python3
"""
Validate the environment before serving vLLM + zentorch on an EPYC CPU host.

Checks a container runtime (docker or podman); probes the SELECTED runtime
(container image if present, else conda/host) for its exact vLLM/zentorch/torch
versions and the active vLLM platform; applies the Venice stack-compatibility
gate; and checks host perf libraries (tcmalloc / OpenMP via LD_PRELOAD),
HF_TOKEN, and RAM. Each issue is error (blocks launch) / warning (degrades) /
advisory (info).

The stack probe distinguishes zentorch-accelerated serving (a Zen platform is
active) from an unaccelerated stock CPU platform. Pass `--generation` (from
detect.py) to enable the Venice gate: Venice on the validated default vLLM
proceeds; Venice on any other vLLM sets `requires_confirmation`; a non-Zen
platform is a hard error.

Usage:
    python3 scripts/validate.py
    python3 scripts/validate.py --image <image from data/epyc.json> --generation Venice

Exits 0 if no error-severity issues remain, 1 otherwise. JSON to stdout.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "epyc.json"
DEFAULT_VLLM_VERSION = json.loads(DATA_PATH.read_text(encoding="utf-8"))["vllm_version"]

# One-line probe run inside the SELECTED runtime (container or host). It reports
# the exact vLLM/zentorch/torch versions AND which vLLM platform is actually
# active, so we can tell zentorch-accelerated serving from an unaccelerated
# stock CPU platform. Both the in-tree `ZenCpuPlatform` and the out-of-tree
# zentorch platform carry "zen" in their module/class path; stock `CpuPlatform`
# does not -- so a name check is a version-robust "is zentorch active?" signal.
PROBE = (
    "import json,vllm,zentorch,torch;"
    "from vllm.platforms import current_platform as _p;"
    "_t=type(_p);_n=_t.__module__+'.'+_t.__name__;"
    "print(json.dumps({"
    "'vllm':vllm.__version__,"
    "'zentorch':getattr(zentorch,'__version__','unknown'),"
    "'torch':torch.__version__,"
    "'platform':_n,"
    "'zen_active':('zen' in _n.lower())}))"
)


def _sh(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", f"timed out after {timeout}s"


def _detect_runtime():
    """Pick an accessible container runtime: docker (daemon reachable) > podman
    (rootless). Returns (runtime, detail) or (None, why).

    Like serving-llms-on-instinct, an accessible runtime is a PREREQUISITE. We
    check and report a one-time fix; we never escalate privileges (no sudo).
    """
    if shutil.which("docker"):
        rc, _, err = _sh("docker ps -q")
        if rc == 0:
            return "docker", "docker reachable"
        last = (err or "docker ps failed").splitlines()[0][:120]
    else:
        last = "docker not installed"
    if shutil.which("podman"):
        rc, _, err = _sh("podman info --format '{{.Host.Arch}}'")
        if rc == 0:
            return "podman", "podman available (rootless)"
        last = (err or last).splitlines()[0][:120] if err else last
    return None, last


def _probe_stack(run_prefix, source):
    """Run PROBE in the selected environment and return (stack, error).

    `run_prefix` is the container run prefix (e.g. 'docker run --rm <image> ')
    for the container path, or '' to probe the local host/conda env. `source`
    labels which path was probed. Returns the parsed stack dict (with `source`)
    or (None, message) if the probe could not run or produced no JSON.
    """
    cmd = f'{run_prefix}python -c "{PROBE}"'
    rc, out, err = _sh(cmd, timeout=120)
    if rc != 0 or not out:
        return None, (err or "probe failed")[:200]
    js = next((ln for ln in out.splitlines() if ln.strip().startswith("{")), "")
    if not js:
        return None, "probe produced no JSON"
    stack = json.loads(js.strip())
    stack["source"] = source
    return stack, None


def stack_compatibility(generation, stack, default_vllm=DEFAULT_VLLM_VERSION):
    """Pure policy over the detected EPYC generation and the probed stack.

    Returns {"status", "message"} where status is one of:
    - "blocked": zentorch/Zen platform is not active -> serving would be
      unaccelerated stock CPU; a hard stop.
    - "confirmation_required": Venice on a vLLM other than the validated default
      -> warn, nudge to the pinned default image, and require explicit user
      confirmation before continuing.
    - "proceed": validated/expected stack.
    Returns None when there is no stack to judge.
    """
    if not stack:
        return None
    vllm_v = str(stack.get("vllm", "")).split("+")[0]
    if not stack.get("zen_active"):
        return {"status": "blocked",
                "message": (f"vLLM {vllm_v or '?'} is on the stock CPU platform "
                            f"({stack.get('platform', '?')}), not a Zen/zentorch platform -- zentorch "
                            "acceleration is NOT active. Enable zentorch or use the pinned container "
                            "image in data/epyc.json; do not serve an unaccelerated CPU stack.")}
    if generation == "Venice" and vllm_v != default_vllm:
        return {"status": "confirmation_required",
                "message": (f"Venice (6th Gen EPYC) is on vLLM {vllm_v or '?'}, which this recipe has "
                            f"NOT validated on Venice. The validated Venice stack is vLLM {default_vllm} "
                            "(the pinned container image in data/epyc.json). Switch to that image, or get "
                            f"the user's explicit OK to continue on vLLM {vllm_v or '?'} unvalidated.")}
    return {"status": "proceed",
            "message": f"Stack OK: vLLM {vllm_v or '?'} on {stack.get('platform', '?')} (zentorch active)."}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", default="", help="container image to check for (advisory)")
    p.add_argument("--generation", default="",
                   help="epyc_generation from detect.py; enables the Venice stack-compatibility gate")
    args = p.parse_args()

    issues = []
    stack = None  # the probed runtime stack for the SELECTED path

    # 1. Container runtime (prerequisite): docker > podman, else conda fallback.
    runtime, detail = _detect_runtime()
    conda_ok = _sh('python -c "import vllm, zentorch"')[0] == 0

    if runtime is None:
        if conda_ok:
            issues.append({"check": "container_runtime", "severity": "warning",
                           "message": f"No accessible container runtime ({detail}); using the conda/host path.",
                           "fix": "For the container path, make docker accessible or install rootless podman (see fix below)."})
        else:
            issues.append({"check": "container_runtime", "severity": "error",
                           "message": f"No accessible container runtime ({detail}) and no host vllm+zentorch.",
                           "fix": "One-time onboarding: add your user to the docker group "
                                  "(sudo usermod -aG docker $USER, then re-login) or start the daemon; "
                                  "OR install rootless podman; OR activate a conda env with vllm+zentorch."})

    # 2. Image present + (only if already pulled) zentorch inside it. The in-image
    #    import check runs ONLY when the image is local, so it never triggers a
    #    multi-GB pull just to validate.
    if runtime and args.image:
        repo = args.image.rsplit(":", 1)[0]  # strip the tag, keep any host:port/repo
        rc, out, _ = _sh(f"{runtime} images {repo} --format '{{{{.Repository}}}}:{{{{.Tag}}}}'")
        if args.image not in (out or ""):
            issues.append({"check": "image", "severity": "advisory",
                           "message": f"Image {args.image} not pulled yet; first launch will download it. "
                                      "RE-RUN validate.py after pulling so the stack-compatibility gate probes the real image.",
                           "fix": f"{runtime} pull {args.image}"})
        else:
            stack, perr = _probe_stack(f"{runtime} run --rm {args.image} ", "container")
            if stack:
                issues.append({"check": "image_stack", "severity": "advisory",
                               "message": f"Image stack: vLLM {stack.get('vllm')} / zentorch {stack.get('zentorch')} "
                                          f"/ torch {stack.get('torch')} on {stack.get('platform')}."})
            else:
                issues.append({"check": "image_stack", "severity": "warning",
                               "message": f"Image {args.image} is present but the stack probe failed inside it: {perr}",
                               "fix": "Use an image tag that bundles the zentorch plugin (see data/epyc.json)."})

    # 3. Host vllm+zentorch (for the conda path). Only probe the host when no
    #    container path was probed, since the container is preferred.
    if conda_ok:
        if stack is None:
            stack, perr = _probe_stack("", "host")
        if stack and stack.get("source") == "host":
            issues.append({"check": "host_stack", "severity": "advisory",
                           "message": f"Host stack: vLLM {stack.get('vllm')} / zentorch {stack.get('zentorch')} "
                                      f"/ torch {stack.get('torch')} on {stack.get('platform')}; conda path available."})
        elif stack is None:
            issues.append({"check": "host_stack", "severity": "warning",
                           "message": f"Host `import vllm, zentorch` reported ready but the stack probe failed: {perr}"})
    elif runtime:
        issues.append({"check": "host_stack", "severity": "advisory",
                       "message": "Host `import vllm, zentorch` not available; use the container path."})

    # 4. Venice stack-compatibility gate (needs a probed stack + a generation).
    compatibility = stack_compatibility(args.generation, stack)
    if compatibility:
        status = compatibility["status"]
        if status == "blocked":
            issues.append({"check": "stack_compatibility", "severity": "error",
                           "message": compatibility["message"]})
        elif status == "confirmation_required":
            issues.append({"check": "stack_compatibility", "severity": "warning",
                           "message": compatibility["message"],
                           "fix": f"Use the pinned vLLM {DEFAULT_VLLM_VERSION} image, or confirm to continue unvalidated."})
        else:
            issues.append({"check": "stack_compatibility", "severity": "advisory",
                           "message": compatibility["message"]})

    # 5. HF_TOKEN
    if not os.environ.get("HF_TOKEN"):
        issues.append({"check": "hf_token", "severity": "advisory",
                       "message": "HF_TOKEN not set. Required for gated models (Llama, Gemma); not needed for Qwen3.",
                       "fix": "export HF_TOKEN=hf_..."})

    # 6. RAM
    rc, out, _ = _sh("grep MemTotal /proc/meminfo | awk '{print int($2/1024/1024)}'")
    try:
        ram_gb = int(out)
    except ValueError:
        ram_gb = 0
    if 0 < ram_gb < 32:
        issues.append({"check": "ram", "severity": "warning",
                       "message": f"Only {ram_gb} GB RAM. CPU serving keeps weights + KV cache in RAM; large models may not fit.",
                       "fix": "Use a small model or a host with more RAM."})

    # 7. Perf libraries for the host/conda path (advisory). vLLM CPU wants
    #    libtcmalloc + libiomp (OpenMP) preloaded and warns otherwise. The
    #    container image sets these itself, so only check the host when the
    #    conda/host path is viable.
    if conda_ok:
        ld = os.environ.get("LD_PRELOAD", "")
        missing = [lib for lib in ("libtcmalloc", "libiomp") if lib not in ld]
        if missing:
            issues.append({"check": "perf_libs", "severity": "advisory",
                           "message": f"LD_PRELOAD is missing {', '.join(missing)}; vLLM CPU warns about this and throughput suffers without them (host/conda path).",
                           "fix": "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$CONDA_PREFIX/lib/libiomp5.so:$LD_PRELOAD"})

    errors = [i for i in issues if i["severity"] == "error"]
    requires_confirmation = bool(compatibility and compatibility["status"] == "confirmation_required")
    result = {
        "ready": len(errors) == 0,
        "requires_confirmation": requires_confirmation,
        "runtime": runtime,
        "runtime_detail": detail,
        "conda_path_available": conda_ok,
        "stack": stack,
        "compatibility": compatibility,
        "ram_gb": ram_gb,
        "errors": errors,
        "warnings": [i for i in issues if i["severity"] == "warning"],
        "advisories": [i for i in issues if i["severity"] == "advisory"],
    }
    print(json.dumps(result, indent=2))
    sys.exit(0 if len(errors) == 0 else 1)


if __name__ == "__main__":
    main()
