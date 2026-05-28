# Magpie command examples

Copy-paste examples. Run from the Magpie repo root (or set `PYTHONPATH`).

## GPU info

```bash
magpie --gpu-info
# or
python -m Magpie --gpu-info
```

## Analyze (config file)

```bash
magpie analyze --kernel-config examples/ck_gemm_add.yaml
magpie analyze --kernel-config my_kernel.yaml -o ./out
```

## Analyze (inline)

```bash
magpie analyze ./kernels/matmul.hip --testcase "./run_test.sh"
magpie analyze ./kernel.hip -t "./run_test.sh" --type hip --no-perf
```

## Compare (config file)

```bash
magpie compare --kernel-config examples/ck_grouped_gemm_compare.yaml
```

## Compare (inline)

```bash
magpie compare kernel_v1.hip kernel_v2.hip --testcase "./run_test.sh"
magpie compare k1.hip k2.hip k3.hip -t "./test.sh" --baseline 0
```

## Benchmark (config file)

```bash
magpie benchmark --benchmark-config examples/benchmarks/benchmark_vllm_dsr1.yaml
magpie benchmark --benchmark-config examples/benchmarks/benchmark_sglang_dsr1.yaml -o ./results
```

## Benchmark (CLI overrides)

```bash
magpie benchmark vllm -m deepseek-ai/DeepSeek-R1-0528 -p fp8 --tp 8
magpie benchmark --benchmark-config examples/benchmarks/benchmark_vllm_dsr1.yaml --run-mode local --timeout 1800
```

## Gap analysis (standalone)

```bash
magpie benchmark gap-analysis --trace-dir ./results/benchmark_vllm_20260101_120000/torch_trace
magpie benchmark gap-analysis --trace-dir ./results/run_xyz/torch_trace --top-k 30 --start-pct 20 --end-pct 80
```

## Verbose and config

```bash
magpie -v analyze --kernel-config my.yaml
magpie --config path/to/config.yaml benchmark --benchmark-config examples/benchmarks/benchmark_vllm_dsr1.yaml
```
