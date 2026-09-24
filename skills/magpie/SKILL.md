---
name: magpie
description: Benchmarks LLM inference and drives GPU kernel optimization with Magpie. Use when the user wants to benchmark vLLM, SGLang, or Atom; capture torch traces; post-process inference traces with TraceLens into prefill/decode and roofline reports; identify top bottleneck kernels or map profiler names to source; analyze or compare HIP, CUDA, PyTorch, or Triton kernels; validate and rank optimized variants; run local, container, or Ray workloads; or mentions Magpie, TraceLens, gap analysis, TTFT, TPOT, kernel evaluation, or AMD GPU optimization.
---

# Magpie

Use Magpie for three connected jobs:

1. Benchmark an inference workload and collect throughput, latency, and traces.
2. Analyze or compare GPU kernels for correctness and performance.
3. Drive an optimization loop from a benchmark bottleneck to source, candidate kernels, and end-to-end validation.

Describe only capabilities supported by the checked-out Magpie version. Do not infer support for an unverified ROCm, GPU, framework, or experimental integration.

## Choose the workflow

| User goal | Workflow |
|---|---|
| Evaluate one implementation | `analyze` |
| Rank two or more implementations | `compare` |
| Measure model-serving performance | `benchmark` |
| Find expensive kernels in existing traces | standalone gap analysis |
| Explain a profiled inference workload | benchmark → TraceLens post-processing → stage/roofline review |
| Optimize an end-to-end workload | benchmark → TraceLens/gap analysis → source mapping → analyze/compare → re-benchmark |

Use a YAML config for reproducible or multi-step work. Use inline CLI arguments for small exploratory runs.

## Preflight

1. Locate the Magpie repository or installed package.
2. Check the local interface before constructing commands:

   ```bash
   magpie --help
   magpie analyze --help
   magpie compare --help
   magpie benchmark --help
   magpie --gpu-info
   ```

3. Check required tools, model access, GPU visibility, writable output space, and container or Ray access as applicable.
4. Read the repository compatibility matrix before making version claims. Treat ROCm or hardware not listed there as unverified until tested.
5. Record the exact config, model revision, image, environment variables, GPU allocation, and Magpie commit for benchmark comparisons.

Run from the Magpie repository root, install with `pip install -e .`, or use `python -m Magpie` when the `magpie` entry point is unavailable.

## Analyze a kernel

Prefer a config when correctness or profiler settings matter:

```bash
magpie analyze --kernel-config path/to/kernel.yaml
```

For a quick single-kernel run:

```bash
magpie analyze path/to/kernel.hip --type hip --testcase "./run_test.sh"
```

Supported public kernel types are `hip`, `cuda`, `pytorch`, and `triton`. Use `--no-perf` only when the user wants correctness or execution validation without profiling.

Do not equate successful execution with numerical correctness. Supply a representative testcase whenever an optimized result will be accepted or rejected.

## Compare kernel variants

Compare at least two implementations and identify the baseline explicitly:

```bash
magpie compare --kernel-config path/to/compare.yaml
```

Keep inputs, tolerances, warmup, iteration count, GPU allocation, and profiler settings identical across candidates. Reject candidates that fail correctness before considering performance rankings.

For PyTorch without a testcase, Magpie's built-in check only verifies that each result is finite; it does not prove numerical equivalence between variants. Require a testcase for numerical validation.

## Benchmark inference

Prefer a checked-in benchmark config:

```bash
magpie benchmark --benchmark-config path/to/benchmark.yaml
```

The stable public CLI supports `vllm`, `sglang`, and `atom`. It supports direct `docker` and `local` run modes; use YAML configuration and the repository's Ray examples for distributed execution. Do not advertise integrations that exist only in internal enums or partial code paths as stable.

Enable profiling deliberately: profiler runs perturb latency and should not replace a clean baseline. Compare throughput, completed requests, TTFT, TPOT, ITL, and end-to-end latency using equivalent workloads.

## Post-process traces with TraceLens

Enable TraceLens in the profiled benchmark YAML; torch traces are its required input:

```yaml
benchmark:
  profiler:
    torch_profiler:
      enabled: true
    tracelens:
      enabled: true
      analysis_mode: inference
      analysis_stages: all
      export_format: csv
```

Use `analysis_mode: inference` for vLLM/SGLang. It splits the rank-0 trace into `prefilldecode`, `decode`, and `prefill` stages when available, runs TraceLens post-processing, and writes full stage reports plus compact `*_kernel_roofline_simple.csv` files under the benchmark workspace's `tracelens/` directory. For direct PyTorch trace reporting, use `analysis_mode: pytorch`.

Open the compact roofline CSVs first. Rank rows by `kernel_time_ms_sum` or `time_pct`; then use `roofline_bound`, arithmetic intensity, achieved TFLOP/s or TB/s, and `pct_roofline_mean` to form an optimization hypothesis. Confirm `benchmark_report.json.tracelens_analysis` has outputs and no error before treating post-processing as successful. Use `analysis_mode: pytorch` when the task specifically needs the legacy direct single-rank or multi-rank collective reports.

Magpie's integrated TraceLens stage produces CSV/Excel analysis artifacts, not an agent-written `analysis.md`. If the user requests a prioritized agentic report, pass the captured trace to the separate `tracelens-analysis-orchestrator` skill when installed; keep that result distinct from Magpie's benchmark report.

## Analyze existing traces and find source

Run standalone gap analysis with `--trace-dir` directly on `benchmark`:

```bash
magpie benchmark \
  --trace-dir path/to/torch_trace \
  --top-k 20 \
  --find-kernel-sources \
  --kernel-source-repos path/to/repository
```

Do not insert a `gap-analysis` positional token; it is not a CLI subcommand. Inspect the generated aggregate and per-rank CSVs, and preserve source-mapping confidence rather than assuming every normalized kernel name maps uniquely.

## Drive the optimization loop

1. Run an unprofiled baseline benchmark and save its config and report.
2. Repeat with torch profiling and TraceLens inference post-processing enabled.
3. Review stage-level TraceLens roofline summaries to classify dominant operations and likely compute, memory, or communication limits.
4. Run gap analysis over the representative steady-state window to rank concrete kernels.
5. Select bottlenecks by total contribution, not only single-dispatch duration.
6. Map the selected kernel to source and an executable testcase.
7. Generate isolated candidate implementations; preserve the baseline.
8. Use `analyze` for iteration, then `compare` with correctness gates to rank candidates.
9. Re-run the original unprofiled benchmark with the winning candidate and the same workload. Report both kernel-level and end-to-end changes, including regressions.

Stop before claiming success if correctness is unproven, the benchmark inputs changed, the source mapping is uncertain, or the end-to-end improvement is within run-to-run noise.

## Use MCP tools when available

Prefer Magpie MCP tools for structured agent workflows such as hardware inspection, kernel discovery, config generation, analyze/compare, optimization suggestions, result lookup, report comparison, Ray job management, and benchmark batches.

Do not pass a CLI `analyze_report.json` wrapper directly to an MCP tool that expects one result object's `performance_state` and `performance_result`. Do not assume every CLI option exists in MCP; kernel-source enrichment is currently exposed by the CLI gap-analysis path.

## Additional resources

- Full CLI reference: [reference.md](reference.md)
- Copy-paste command examples: [examples.md](examples.md)
