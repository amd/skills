---
name: kernel-exp-history
description: This skill should be used when optimizing kernels in this repo and needing to consult past optimization experiments, or when recording the current optimization iteration back into the kernel experiment database.
---

# Kernel Experiment History

## Overview

Use the local kernel experiment database to look up prior optimization attempts and record new results after an optimization iteration completes.

## Workflow

### 1) Find prior experiments for inspiration

- Read `references/kernel_exp_dataclass.py` to understand the database helpers and schema.
- Start with `top_experiments(max_results=20)` to get a score-sorted list of high-impact experiments.
- If more context is needed, load full entries using `get_experiment(exp_id)` or `list_experiments()` and filter by `operator_sig`, `dtype_sig`, `env`, or `base_commit`.
- Summarize the most relevant patterns (block sizes, memory changes, profiling signals, etc.) before proposing new optimizations.

#### Query Examples

**Example 1: Find similar kernel optimizations**
```python
# Search for cache kernel optimizations
from kernel_exp_dataclass import list_experiments

experiments = list_experiments()
cache_exps = [e for e in experiments if 'cache' in e.operator_sig.lower()]

# Sort by score
cache_exps_sorted = sorted(cache_exps, key=lambda x: x.score, reverse=True)

print("Top cache kernel optimizations:")
for exp in cache_exps_sorted[:5]:
    print(f"  {exp.score:.4f}x - {exp.change_summary}")
```

**Example 2: Find best unroll factor**
```python
# Compare different unroll factors
unroll_exps = [e for e in experiments if 'unroll' in e.change_summary.lower()]

for exp in unroll_exps:
    factor = 'unknown'
    if 'unroll 4' in exp.detailed_description.lower():
        factor = '4'
    elif 'unroll 8' in exp.detailed_description.lower():
        factor = '8'
    print(f"Unroll {factor}: {exp.score:.4f}x - {exp.operator_sig[:50]}")
```

**Example 3: Learn from failures**
```python
# Find what NOT to do
failures = [e for e in experiments if e.score < 0.98 or e.is_buggy]

print("Failed optimizations (learn from these!):")
for exp in failures:
    print(f"  ❌ {exp.change_summary}")
    print(f"     Why: {exp.detailed_description[:100]}...")
```

### 2) Record the current optimization iteration

- After finishing the optimization iteration, write a concise summary of the changes and results.
- Populate all required fields on `KernelExperiment`, including:
  - `change_summary`, `detailed_description`, `raw_result`, `score`
  - `operator_sig`, `dtype_sig`, `env`, `base_commit`, `profiling_info`
  - `is_buggy`, `error_message`, `status`
  - `pid` if this iteration builds on a parent experiment (set manually)
- Call `create_experiment()` to append the entry to the database.

#### Field-by-Field Best Practices

**change_summary** (1 line, <80 chars):
- ✅ Good: "Applied #pragma unroll 4 to flash kernel - best result at +1.90%"
- ❌ Bad: "Made some changes to the kernel"
- Format: `<What> - <Result>` or `<What> - <Why it failed>`

**detailed_description** (multiple paragraphs):
Structure:
```
**Approach**: [What you tried]
- Specific technical details
- Why you thought it would work

**Result**: [What happened]
- Quantitative results
- Qualitative observations

**Why it worked/failed**: [Root cause analysis]
- Technical explanation
- Compare to similar attempts

**Key insight**: [Takeaway for future]
- What this taught you
- How to apply the lesson
```

**raw_result** (structured text):
```
Iteration N Results - [SUCCESS/REGRESSION/CRASH]:

**Overall**: X.XXXXx speedup = Y.YY% [IMPROVEMENT/REGRESSION]

**Per-kernel breakdown**:
- kernel_1: X.XXXXx (+Y.YY%)
- kernel_2: X.XXXXx (+Y.YY%)
...

**Summary**: X improvements, Y neutral, Z regressions

**Key finding**: [One-line takeaway]
```

**profiling_info** (even if not profiled):
- If profiled: Include key metrics (occupancy, bandwidth, bottleneck type)
- If NOT profiled: Explain why not, and what benchmarks showed

### 3) Update existing experiments (if needed)

- If you discover errors in previous recordings (e.g., false regression due to testing issues):
  - Use `update_experiment(exp_id, raw_result=..., score=..., detailed_description=...)`
  - Update the score to reflect corrected performance
  - Document the correction reason in detailed_description
- Common update scenarios:
  - Test methodology errors discovered
  - Performance re-measurement with better methodology
  - Bug fixes affecting correctness


## Score Guidelines

- Score = speedup ratio (e.g., 1.18 for 18% improvement)
- For regressions: score < 1.0 (e.g., 0.70 for 30% slower)
- Average across all tested configurations if performance varies

## The Value of Recording Failures

**Critical**: Record ALL iterations, especially failures!

**Why record failures?**
1. 🚫 **Prevent repetition**: Future you won't try the same failed approach
2. 📚 **Build institutional knowledge**: Team learns what doesn't work
3. 🔍 **Pattern recognition**: Multiple failures reveal deeper issues
4. 💡 **Negative results are results**: "X doesn't work" is valuable information

**Failure categories to track**:
- **Buggy** (`is_buggy=True`): Crashes, correctness errors
- **Regressive** (score < 1.0): Made things slower
- **Marginal** (0.99 < score < 1.01): No meaningful impact
- **Interference** (combined optimization worse than separate): Resource conflicts

**Example from cache kernel optimization**:
- Iteration 1 (-1.81%): Disrupted coalescing → Learned: preserve memory patterns
- Iteration 5 (CRASH): Manual unrolling bug → Learned: use pragmas, not manual
- Iteration 7 (-0.63%): Combined optimizations → Learned: resource interference real

## Notes

- Use `top_experiments()` first; fall back to full queries only when additional details are needed.
- Keep summaries short but specific enough to guide future optimization decisions.
