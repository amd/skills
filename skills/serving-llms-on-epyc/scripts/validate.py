#!/usr/bin/env python3
"""
Validate the environment before serving vLLM + zentorch on an EPYC CPU host.

Checks a container runtime (docker or podman), whether the vLLM+zentorch image
is present (and, if already pulled, that `import vllm, zentorch` works inside it),
a conda/host fallback (`import vllm, zentorch`), the host perf libraries
(tcmalloc / OpenMP via LD_PRELOAD), HF_TOKEN, and RAM. Each issue is error
(blocks launch) / warning (degrades) / advisory (info).

Usage:
    python3 scripts/validate.py
    python3 scripts/validate.py --image amdih/zendnn_zentorch:vllm_v0.22.0_zentorch_v2.11.0.1_ubuntu22.04_2026_ww23

Exits 0 if no error-severity issues remain, 1 otherwise. JSON to stdout.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", default="", help="container image to check for (advisory)")
    args = p.parse_args()

    issues = []

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
                           "message": f"Image {args.image} not pulled yet; first launch will download it (in-image zentorch check deferred to launch).",
                           "fix": f"{runtime} pull {args.image}"})
        else:
            rc, ver, err = _sh(f'{runtime} run --rm {args.image} '
                               f'python -c "import vllm,zentorch;print(vllm.__version__,zentorch.__version__)"', timeout=90)
            if rc == 0 and ver:
                issues.append({"check": "image_stack", "severity": "advisory",
                               "message": f"Image has vllm+zentorch ({ver})."})
            else:
                issues.append({"check": "image_stack", "severity": "warning",
                               "message": f"Image {args.image} is present but `import vllm, zentorch` failed inside it: {(err or 'unknown')[:120]}",
                               "fix": "Use an image tag that bundles the zentorch plugin (see data/epyc.json)."})

    # 3. Host vllm+zentorch (for the conda path)
    if conda_ok:
        _, ver, _ = _sh('python -c "import vllm,zentorch;print(vllm.__version__,zentorch.__version__)"')
        issues.append({"check": "host_stack", "severity": "advisory",
                       "message": f"Host vllm+zentorch importable ({ver}); conda path available."})
    elif runtime:
        issues.append({"check": "host_stack", "severity": "advisory",
                       "message": "Host `import vllm, zentorch` not available; use the container path."})

    # 4. HF_TOKEN
    if not os.environ.get("HF_TOKEN"):
        issues.append({"check": "hf_token", "severity": "advisory",
                       "message": "HF_TOKEN not set. Required for gated models (Llama, Gemma); not needed for Qwen3.",
                       "fix": "export HF_TOKEN=hf_..."})

    # 5. RAM
    rc, out, _ = _sh("grep MemTotal /proc/meminfo | awk '{print int($2/1024/1024)}'")
    try:
        ram_gb = int(out)
    except ValueError:
        ram_gb = 0
    if 0 < ram_gb < 32:
        issues.append({"check": "ram", "severity": "warning",
                       "message": f"Only {ram_gb} GB RAM. CPU serving keeps weights + KV cache in RAM; large models may not fit.",
                       "fix": "Use a small model or a host with more RAM."})

    # 6. Perf libraries for the host/conda path (advisory). vLLM CPU wants
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
    result = {
        "ready": len(errors) == 0,
        "runtime": runtime,
        "runtime_detail": detail,
        "conda_path_available": conda_ok,
        "ram_gb": ram_gb,
        "errors": errors,
        "warnings": [i for i in issues if i["severity"] == "warning"],
        "advisories": [i for i in issues if i["severity"] == "advisory"],
    }
    print(json.dumps(result, indent=2))
    sys.exit(0 if len(errors) == 0 else 1)


if __name__ == "__main__":
    main()
