---
name: magpie
description: Performs GPU kernel correctness and performance evaluation and LLM inference benchmarking with Magpie. Analyzes single or multiple kernels (HIP/CUDA/PyTorch), compares kernel implementations, runs vLLM/SGLang benchmarks with profiling and TraceLens, and runs gap analysis on torch traces. Creates kernel config YAMLs, discovers kernels in a project, and queries GPU specs. Use when the user mentions Magpie, kernel analyze or compare, HIP/CUDA kernel evaluation, vLLM/SGLang benchmark, gap analysis, TraceLens, creating kernel configs, or discovering GPU kernels.
---

# Magpie

Magpie is a GPU kernel evaluation and LLM benchmarking framework. Use this skill when performing analyze, compare, benchmark, gap-analysis, or when creating kernel configs or discovering kernels without MCP.

**When describing Magpie's capabilities:** Describe only what is in this skill. Do not add project-specific, pipeline-specific, or other product/org names (e.g. do not mention any parent repo name).

## Entry point

- **CLI:** `magpie` or `python -m Magpie`. Run from the Magpie repo root (or with `PYTHONPATH` including the Magpie package).
- **Setup:** From repo root, `pip install -e .` (or `make install`).

## Analyze (single or multi-kernel)

Analyze kernel(s) for correctness and performance.

**With kernel config (recommended):**

```bash
magpie analyze --kernel-config path/to/kernel.yaml
```

**Inline (single kernel):**

```bash
magpie analyze path/to/kernel.hip --testcase "./run_test.sh"
```

- `-k`, `--kernel-config`: YAML with `kernel` or `kernels` (see template below).
- `-t`, `--testcase`: Command to run the test (required if not in config).
- `--type`: `hip` | `cuda` | `pytorch` (default: hip).
- `--compile-cmd`: Custom compile command.
- `--no-perf`: Skip performance profiling.
- `-o`, `--output-dir`: Output directory (default: `./results`).

**Config template (single kernel):** Use `kernel:` with `id`, `type`, `source_files`, `working_dir`, `testcase_command`, optional `compile_command`, `env`. See [Magpie/kernel_config.yaml.example](https://github.com/AMD-AGI/Magpie/blob/2a9263833f71755df2a93b466cdd3a9f803fc625/Magpie/kernel_config.yaml.example) and [examples/ck_gemm_add.yaml](https://github.com/AMD-AGI/Magpie/blob/2a9263833f71755df2a93b466cdd3a9f803fc625/examples/ck_gemm_add.yaml).

## Compare (multiple kernels)

Compare and rank multiple kernel implementations.

**With config:**

```bash
magpie compare --kernel-config path/to/compare.yaml
```

**Inline:**

```bash
magpie compare kernel1.hip kernel2.hip --testcase "./run_test.sh"
```

- `-k`, `--kernel-config`: YAML with `kernels:` list.
- `--baseline`: Index of baseline kernel (default: 0).
- `--no-perf`, `-o`: Same as analyze.

Example: [examples/ck_grouped_gemm_compare.yaml](https://github.com/AMD-AGI/Magpie/blob/2a9263833f71755df2a93b466cdd3a9f803fc625/examples/ck_grouped_gemm_compare.yaml).

## Benchmark (vLLM / SGLang)

Run framework-level LLM inference benchmarks with optional profiling and gap analysis.

**With config (recommended):**

```bash
magpie benchmark --benchmark-config examples/benchmarks/benchmark_vllm_dsr1.yaml
```

**CLI overrides:** `magpie benchmark [vllm|sglang] -m <model> --benchmark-config <yaml>` with optional:

- `-m`, `--model`: Model name or path.
- `-p`, `--precision`: fp8 | fp16 | bf16 | fp4 (default: fp8).
- `--tp`: Tensor parallel size (default: 1).
- `--concurrency`, `--input-len`, `--output-len`: Request and sequence settings.
- `--torch-profiler`, `--system-profiler`: Enable profilers.
- `--run-mode`: `docker` (default) or `local`.
- `--docker-image`, `--timeout`, `-o`: Override image, timeout (seconds), output dir.

Example configs: [examples/benchmarks/benchmark_vllm_dsr1.yaml](https://github.com/AMD-AGI/Magpie/blob/2a9263833f71755df2a93b466cdd3a9f803fc625/examples/benchmarks/benchmark_vllm_dsr1.yaml), [docs/benchmark.md](https://github.com/AMD-AGI/Magpie/blob/2a9263833f71755df2a93b466cdd3a9f803fc625/docs/benchmark.md).

## Gap analysis (standalone)

Run gap analysis on existing torch trace directories.

```bash
magpie benchmark gap-analysis --trace-dir path/to/torch_trace
```

- `--trace-dir`: Path to `torch_trace` dir or benchmark workspace (required).
- `--start-pct`, `--end-pct`: Analysis window 0–100 (default: 0, 100).
- `--top-k`: Top bottleneck kernels (default: 20).
- `--min-duration-us`: Minimum event duration (µs).
- `--categories`, `--ignore-categories`: Include/exclude event categories.

## GPU info

```bash
magpie --gpu-info
```

Shows vendor, architecture, compiler, profiler. No mode required.

## Create kernel config (no CLI)

When the user needs a kernel config file:

1. Emit YAML matching the structure in [Magpie/kernel_config.yaml.example](https://github.com/AMD-AGI/Magpie/blob/2a9263833f71755df2a93b466cdd3a9f803fc625/Magpie/kernel_config.yaml.example): `kernel:` with `id`, `type` (hip|cuda|pytorch), `source_files`, `working_dir`, `testcase_command`, and optionally `compile_command`, `env`.
2. Write the file to the user's requested path (e.g. `kernel_config.yaml`).
3. Run: `magpie analyze --kernel-config <that_file>`.

For **compare**, use `kernels:` as a list of kernel entries (each with `id`, `type`, `source_files`, etc.).

## Discover kernels (no CLI)

1. Scan the project for `.hip`, `.cu`, or PyTorch kernel files.
2. For each candidate, build a kernel config entry (id, type, source_files, working_dir, testcase_command if inferrable; otherwise prompt user).
3. Optionally write a combined config and run `magpie analyze -k <file>` or `magpie compare -k <file>`.

## Suggest optimizations (no CLI)

1. Read analyze or compare JSON output (from `-o` results or last run).
2. Use `performance_state`, `performance_result.summary`, and per-kernel stats (dispatch count, duration, utilization).
3. Suggest improvements (e.g. memory bandwidth, occupancy, kernel fusion) based on the metrics.

## List / get benchmark results (no CLI)

- **List:** Results live under the benchmark `--output-dir` (default: `./results`); each run has a timestamped workspace (e.g. `results/benchmark_vllm_<timestamp>/`).
- **Get result:** Open `benchmark_report.json` or `inferencex_result.json` in that workspace.
- **Compare runs:** Diff two workspace reports or run two benchmarks and compare; for TraceLens comparison use TraceLens tooling if available.

## Additional resources

- Full CLI reference: [reference.md](reference.md)
- Copy-paste command examples: [examples.md](examples.md)
