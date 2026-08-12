# Magpie reference

Use this reference after selecting a workflow in `SKILL.md`. Confirm flags against `magpie <mode> --help` because the checked-out version is authoritative.

## Public CLI surface

Global options precede the mode:

| Option | Values or purpose |
|---|---|
| `--config`, `-c` | Framework configuration file |
| `--verbose`, `-v` | Verbose logging |
| `--gpu-info` | Print detected GPU information and exit |
| `--environment`, `-e` | `local`, `container`, or `ray` |
| `--workers`, `-w` | Concurrent workers |
| `--docker-image` | Container image override |

### `analyze`

```text
magpie analyze [kernels ...] [--kernel-config FILE] [--testcase CMD]
               [--type hip|cuda|pytorch|triton] [--compile-cmd CMD]
               [--no-perf] [--output-dir DIR]
```

`analyze` requires a usable testcase from the CLI or config. Its report is written below a timestamped `analyze_*` workspace; `analyze_report.json` contains a `results` list rather than a single flat result.

### `compare`

```text
magpie compare [kernels ...] [--kernel-config FILE] [--testcase CMD]
               [--type hip|cuda|pytorch|triton] [--baseline INDEX]
               [--no-perf] [--output-dir DIR]
```

Provide at least two kernels. The report includes per-kernel results, comparison metrics, rankings, a winner, and a summary.

### `benchmark`

```text
magpie benchmark [vllm|sglang|atom] [--benchmark-config FILE]
                 [--model MODEL] [--precision fp8|fp16|bf16|fp4]
                 [--tp N] [--concurrency N] [--input-len N] [--output-len N]
                 [--torch-profiler] [--system-profiler]
                 [--run-mode docker|local] [--docker-image IMAGE]
                 [--timeout SECONDS] [--output-dir DIR]
```

Use YAML for advanced settings such as TraceLens, gap analysis, automatic GPU selection, persistent server lifecycle, Ray execution, profiler options, and framework-specific environment variables.

### Standalone gap analysis

```text
magpie benchmark --trace-dir DIR [--top-k N]
                 [--start-pct PCT] [--end-pct PCT]
                 [--min-duration-us US]
                 [--categories CATEGORY ...]
                 [--ignore-categories CATEGORY ...]
                 [--no-rank-csv] [--find-kernel-sources]
                 [--kernel-source-repos PATH ...]
```

`--trace-dir` accepts a torch-trace directory or a benchmark workspace containing one. Source enrichment can identify candidates from Triton JIT, CK Tile, Tensile, ATen, HIP, AITER, and Torch Inductor repositories. Treat a source match as evidence to verify, not a correctness guarantee.

## Kernel configuration

Single-kernel analyze configuration:

```yaml
kernel:
  id: candidate
  type: hip
  source_files:
    - ./kernel.hip
  working_dir: .
  compile_command: hipcc -O3 kernel.hip -o kernel
  testcase_command: ./run_test.sh
  env: {}
```

For compare, replace `kernel:` with `kernels:` and provide a list of at least two entries. Configuration may also override performance, correctness, Ray, and scheduler settings. Start from the repository's [kernel config example](https://github.com/AMD-AGI/Magpie/blob/main/Magpie/kernel_config.yaml.example) instead of inventing field names.

## Correctness and performance semantics

| Kernel type | Correctness guidance | Typical profiler |
|---|---|---|
| HIP | Provide a testcase; use Accordo when configured for binary replay/validation | rocprof-compute or Metrix |
| CUDA | Provide a testcase; configure the supported CUDA validation path | NCU |
| PyTorch | Require a testcase for numerical equivalence; without one, only finite-value sanity is checked | configured PyTorch/system path |
| Triton | Provide an executable script/testcase; built-in execution checks do not replace numeric assertions | Selected from detected GPU architecture |

Apply correctness as a hard gate before performance scoring. Keep tolerances explicit and appropriate to the datatype. Confirm optional profiler and correctness dependencies in the repository's [compatibility matrix](https://github.com/AMD-AGI/Magpie/blob/main/docs/reference/compatibility-matrix.md).

## Benchmark configuration and outputs

A benchmark config starts with `benchmark:` and commonly specifies:

- framework, model, precision, and tensor parallelism;
- concurrency and input/output lengths;
- Docker, local, or Ray execution settings;
- torch/system profiler and TraceLens settings;
- gap-analysis window, categories, top-k, and source finding;
- GPU selection constraints and server lifecycle;
- timeout, image, and environment variables.

Set the benchmark destination with the CLI `--output-dir` option. `output_dir`
is not a benchmark YAML field and would be ignored by `BenchmarkConfig`.

Important outputs may include:

- `config.yaml`: effective benchmark configuration snapshot, including workload, environment, execution, and profiler settings;
- `benchmark_report.json`: throughput, latency, execution status, and optional profiling/analysis results;
- `summary.txt`: human-readable summary;
- framework benchmark results such as `inferencex_result.json`;
- stdout/stderr logs;
- torch traces and profiler output;
- TraceLens reports;
- gap-analysis aggregate and per-rank CSVs, optionally enriched with source/test information.

Compare clean, unprofiled benchmark reports for final performance claims. Use each workspace's `config.yaml` to verify that the effective workloads are equivalent, and use profiled runs to explain the result.

## TraceLens post-processing

TraceLens consumes torch profiler traces after the inference workload completes. Configure it below `benchmark.profiler.tracelens`; `analysis_mode: inference` is the default and recommended mode for vLLM/SGLang, while `analysis_mode: pytorch` uses the direct PyTorch report flow.

Important inference settings:

| Setting | Purpose |
|---|---|
| `enabled` | Run TraceLens after trace capture |
| `analysis_mode` | `inference` or `pytorch` (`classic` aliases `pytorch`) |
| `analysis_stages` | `all` or selected `prefilldecode`, `decode`, and `prefill` stages |
| `auto_patch_runtime` | Build/reuse a TraceLens-ready Docker image when required |
| `tracelens_repo_path` | Pin the public TraceLens checkout used for runtime preparation |
| `extension_wheel_path` | Add an optional local TraceLens extension to the runtime |
| `cli_timeout_seconds` | Timeout for each post-processing command |
| `export_format` | `csv` or `excel` |
| `perf_report_enabled` | Generate the single-rank performance report |
| `multi_rank_report_enabled` | Generate collective and multi-rank analysis |
| `gpu_arch_config` | Provide an explicit architecture JSON for roofline modeling |

For Docker inference mode, Magpie runs post-processing in the resolved TraceLens-ready image after the benchmark container exits. The host needs Docker but does not need the TraceLens CLI on `PATH`. For other paths, verify the TraceLens CLI or configured runtime is available.

Expected workspace artifacts:

- `torch_trace/trace_split/`: representative inference stage windows;
- `tracelens/prefilldecode/`, `tracelens/decode_only/`, and `tracelens/prefill_only/`: full per-stage CSV reports when those stages exist;
- `tracelens/*_kernel_roofline_simple.csv`: compact stage summaries for first-pass review;
- `tracelens_rank0_csvs/`: direct single-rank reports in `pytorch` mode;
- `tracelens_collective_csvs/`: multi-rank collective and load-balance reports in `pytorch` mode;
- `benchmark_report.json.tracelens_analysis`: structured status and output-file paths.

Review the compact roofline files in this order:

1. Sort by `kernel_time_ms_sum` or `time_pct` to find dominant operations per inference stage.
2. Use `roofline_bound` and arithmetic intensity to classify compute- versus memory-bound work.
3. Check achieved TFLOP/s, achieved TB/s, and `pct_roofline_mean` for efficiency gaps.
4. Check `has_perf_model`; do not treat rows without a performance model as complete roofline evidence.
5. Join the operation-level finding with gap-analysis kernel timings and source mapping before choosing code to modify.

Integrated TraceLens post-processing does not create `analysis.md`. Use the separate `tracelens-analysis-orchestrator` skill for an agentic prioritized report when available, and label that report as a downstream artifact rather than Magpie-native output.

## CLI and MCP boundary

When the Magpie MCP server is connected, it may expose structured tools for:

- GPU hardware inspection and configuration;
- kernel discovery, config generation, analyze, compare, and optimization suggestions;
- benchmark execution, image listing, result listing/retrieval, and report comparison;
- gap analysis;
- Ray task status, result, cancellation, and listing;
- benchmark batches.

Observe these boundaries:

- CLI analyze reports wrap results in `results`; MCP optimization suggestions expect one structured result.
- CLI standalone gap analysis exposes kernel-source enrichment flags; do not assume the MCP gap tool accepts them.
- MCP report comparison requires compatible TraceLens rank-0 CSV artifacts rather than only two generic benchmark reports.
- If MCP is unavailable, use the CLI and inspect workspace files directly.

## Reproducibility checklist

Record:

- Magpie commit and command/config;
- GPU model, architecture, count, and visibility variables;
- ROCm/CUDA, driver, Python, framework, profiler, and container versions;
- model identifier and immutable revision when possible;
- tensor parallelism, concurrency, input/output lengths, precision, and random-range settings;
- warmup, iteration count, timeout, and profiler state;
- source commit and build flags for each kernel candidate.

Common environment inputs include model-access tokens and workload controls such as tensor parallelism, concurrency, input/output lengths, maximum model length, and GPU visibility. Never print secret token values.
