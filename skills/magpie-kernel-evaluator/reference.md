# Magpie CLI reference

Full flag reference for `magpie` / `python -m Magpie`. Global options apply before the mode.

## Global options

| Option | Description |
|--------|-------------|
| `--config`, `-c` | Framework configuration file (default: package config.yaml) |
| `--verbose`, `-v` | Verbose output |
| `--gpu-info` | Show detected GPU info and exit (no mode required) |
| `--environment`, `-e` | `local` \| `container` |
| `--workers`, `-w` | Number of concurrent workers |
| `--docker-image` | Docker image for container environment |

## analyze

| Option | Description |
|--------|-------------|
| `kernels` | Positional: kernel file(s) to analyze |
| `--kernel-config`, `-k` | Kernel configuration YAML |
| `--testcase`, `-t` | Testcase command |
| `--type` | `hip` \| `cuda` \| `pytorch` (default: hip) |
| `--compile-cmd` | Custom compile command |
| `--no-perf` | Skip performance profiling |
| `--output-dir`, `-o` | Output directory (default: ./results) |

## compare

| Option | Description |
|--------|-------------|
| `kernels` | Positional: kernel files to compare |
| `--kernel-config`, `-k` | Kernel configuration YAML (kernels list) |
| `--testcase`, `-t` | Testcase command (optional) |
| `--type` | `hip` \| `cuda` \| `pytorch` (default: hip) |
| `--baseline` | Baseline kernel index (default: 0) |
| `--no-perf` | Skip performance profiling |
| `--output-dir`, `-o` | Output directory (default: ./results) |

## benchmark

| Option | Description |
|--------|-------------|
| `framework` | Optional positional: `vllm` \| `sglang` |
| `--benchmark-config`, `-b` | Benchmark configuration YAML |
| `--model`, `-m` | Model name or path |
| `--precision`, `-p` | fp8 \| fp16 \| bf16 \| fp4 (default: fp8) |
| `--tp` | Tensor parallel size (default: 1) |
| `--concurrency` | Request concurrency (default: 32) |
| `--input-len` | Input sequence length (default: 1024) |
| `--output-len` | Output sequence length (default: 512) |
| `--torch-profiler` | Enable torch profiler |
| `--system-profiler` | Enable system profiler (rocprof/ncu) |
| `--run-mode` | `docker` \| `local` |
| `--docker-image` | Override Docker image |
| `--inferencex-path` | Path to InferenceX (auto-cloned if empty) |
| `--benchmark-script` | Override benchmark script name |
| `--timeout` | Timeout in seconds (default: 3600) |
| `--output-dir`, `-o` | Output directory (default: ./results) |

## benchmark gap-analysis

| Option | Description |
|--------|-------------|
| `--trace-dir` | Path to torch_trace dir or benchmark workspace (required) |
| `--start-pct` | Start of analysis window 0–100 (default: 0) |
| `--end-pct` | End of analysis window 0–100 (default: 100) |
| `--top-k` | Number of top bottleneck kernels (default: 20) |
| `--min-duration-us` | Minimum event duration in µs (default: 0) |
| `--categories` | Event categories to include (space-separated) |
| `--ignore-categories` | Event categories to exclude |

## Kernel config YAML (analyze)

Single kernel:

```yaml
kernel:
  id: "my_kernel"
  type: hip
  source_files: ["./kernel.hip"]
  working_dir: "."
  testcase_command: "./run_test.sh"
  # optional: compile_command, env
```

Multiple kernels (compare):

```yaml
kernels:
  - id: "v1"
    type: hip
    source_files: ["./v1.hip"]
    testcase_command: "./run_test.sh"
  - id: "v2"
    type: hip
    source_files: ["./v2.hip"]
    testcase_command: "./run_test.sh"
```

## Benchmark config YAML

Top-level key `benchmark:` with `framework`, `model`, `precision`, `envs`, `profiler`, `gap_analysis`, `timeout_seconds`, etc. See `examples/benchmarks/benchmark_vllm_dsr1.yaml` and `docs/how-to/benchmark.md`.
