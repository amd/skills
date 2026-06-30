#!/bin/bash
# One-time setup for the vendored multi-instance benchmark harness.
# Clones the external guidellm/ansible automation (redhat-et/vllm-cpu-perf-eval)
# into the harness dir and applies the bundled patch (rootless guidellm user fix,
# /tmp -> BENCH_TMPDIR redirect, local-model bind mount). Safe to re-run.
#
# No dependency on ZenDNN_tools or any other repo — everything lives under this
# skill. Run once before the first sweep.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HARNESS="${HARNESS:-$SKILL_DIR/harness}"
EVAL_DIR="$HARNESS/vllm-cpu-perf-eval"
PATCH="$HARNESS/vllm-cpu-perf-eval.patch"

[ -f "$HARNESS/run_sweep.sh" ] || { echo "ERROR: harness not found at $HARNESS" >&2; exit 1; }
[ -f "$PATCH" ] || { echo "ERROR: patch not found at $PATCH" >&2; exit 1; }

if [ ! -d "$EVAL_DIR/.git" ]; then
    echo "--- Cloning vllm-cpu-perf-eval (external ansible/guidellm automation) ---"
    git clone https://github.com/redhat-et/vllm-cpu-perf-eval.git "$EVAL_DIR"
else
    echo "--- vllm-cpu-perf-eval already present ($EVAL_DIR) ---"
fi

echo "--- Applying patch (use --3way; upstream drifts) ---"
if git -C "$EVAL_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "  Patch already applied; skipping."
else
    git -C "$EVAL_DIR" stash -q 2>/dev/null || true
    if git -C "$EVAL_DIR" apply --3way "$PATCH"; then
        echo "  Patch applied."
    else
        echo "ERROR: patch failed to apply. Resolve manually in $EVAL_DIR." >&2
        exit 1
    fi
fi

echo "--- Checking ansible collections ---"
need="containers.podman ansible.posix community.general"
have="$(ansible-galaxy collection list 2>/dev/null)"
for c in $need; do
    if echo "$have" | grep -qi "^$c "; then echo "  OK   $c"; else echo "  MISS $c (install: ansible-galaxy collection install $c)"; fi
done

echo ""
echo "Harness ready at: $HARNESS"
echo "Ansible playbook : $EVAL_DIR/automation/test-execution/ansible/llm-benchmark-concurrent-load.yml"
