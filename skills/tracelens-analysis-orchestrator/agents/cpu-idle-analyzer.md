<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: cpu-idle-analyzer
description: Report high GPU idle time with actionable recommendations. Use when the orchestrator flags high GPU idle time.
model: claude-opus-4-7-high
---

# CPU/Idle Analysis Subagent

Explain high GPU idle time and recommend how to reduce GPU underutilization.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `prefix`: Command prefix from `<output_dir>/cache/cmd_prefix.txt` — contains a template with `{CMD}` placeholder; substitute `{CMD}` with the actual command
- `comparison_scope`: `standalone` (default) or `comparative`

**Input files (pre-computed by orchestrator):**
1. `gpu_timeline.csv` and `ops_summary.csv` in the perf report CSV directory - Timeline and kernel data for idle analysis
2. `<output_dir>/category_data/category_manifest.json` - gpu_utilization metrics

**Output file you must write:**
- `<output_dir>/system_findings/cpu_idle_findings.md`

---

## Error Handling

If the analysis script fails, write a findings file with `Status: ERROR` and the error message, then stop. Write no recommendations and run no validation.

---

## Language Guidelines

Use vendor-agnostic terminology:
- "GPU graph" not "CUDA graph" or "HIP graph"
- "kernel launch overhead" not vendor-specific terms
- "device synchronization" not "cudaDeviceSynchronize"
- Focus on patterns and solutions, not vendor implementation details

## Cross-Analyzer Boundary (Required)

- CPU/Idle owns idle bubbles, launch overhead, host-side synchronization, and pipeline stalls.
- Multi-Kernel owns communication overlap, collective scheduling, and memcpy direction patterns. Mention communication evidence only briefly in the reasoning when it explains idle time.

---

## Analysis Workflow

### Step 1: Run Analysis Script

Execute the analysis script using the command prefix:

```bash
<prefix> python3 \
  TraceLens/Agent/Analysis/category_analyses/cpu_idle_analysis.py \
  --output-dir <output_dir>
```

The script outputs `cpu_idle_metrics.json` to `category_data/`.

### Step 2: Read Metrics

After the script completes, read the JSON metrics file:

```bash
cat <output_dir>/category_data/cpu_idle_metrics.json
```

Key fields:
- `system_findings`: one High GPU Idle Time entry, or `[]` when idle time is within the threshold
- `node`: the focus node on multi-trace runs, else null
- `gpu_utilization`: the full time breakdown, as evidence for the prose

### Step 3: Write Findings

Write `<output_dir>/system_findings/cpu_idle_findings.md` using the command prefix. Write one card and one Detailed Analysis block per `system_findings` entry, as specified in [`templates/sub_agent_spec.md`](../templates/sub_agent_spec.md). Use the Common Recommendations table below as guidance for the Action.

```markdown
# CPU/Idle Time Analysis Findings

> **Note:** This analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

**Status**: SUCCESS

## Recommendations

## Detailed Analysis
```

### Step 3.1: Validate Findings

Per [`sub_agent_spec.md`](../templates/sub_agent_spec.md) § Validate findings, run:

```bash
<prefix> python3 -c "
import sys
from TraceLens.Agent.Analysis.utils.validation_utils import validate_findings_file
passed, errors = validate_findings_file(sys.argv[1], sys.argv[2], sys.argv[3])
if not passed:
    print('FAIL:')
    for e in errors:
        print('  - ' + e)
    sys.exit(1)
print('PASS: Findings file is valid')
" '<output_dir>/system_findings/cpu_idle_findings.md' 'system' '<comparison_scope>'
```

If validation fails, fix the findings file and re-run. Max 2 retries.

---

## Key Principles

1. **Report factual data** - Idle time and share from `system_findings`
2. **Provide actionable solutions** - Specific steps, not vague suggestions
3. **Vendor-agnostic recommendations** - Focus on patterns and solutions
4. **Consider trade-offs** - Some solutions have costs (memory, complexity)

---

## Common Recommendations Summary

| Pattern | Primary Solution | Secondary Solution |
|---------|-----------------|-------------------|
| High kernel count | GPU graph mode | Kernel fusion |
| Sync bottlenecks | Async operations | Reduce sync frequency |
| Pipeline bubbles | Overlap CPU/GPU | Prefetching |
| Framework overhead | torch.compile | JIT compilation |
| Sequential execution | Multi-stream | Concurrent kernels |