---
name: aiter-reflection
description: This skill should be used when optimizing AMD GPU kernels on MI300 using the aiter project, including running op tests, benchmarking, iterating on kernel changes, and recording results in the kernel experiment database.
---

# Aiter Reflection

## Overview

Optimize AMD MI300 GPU kernels for correctness and performance using the aiter workflow, then record each iteration to the kernel experiment database.

## Workflow

### 1) Locate targets and understand tests

- Use the provided context to identify target kernel files, kernels, and their op tests.
- Run the op tests once to understand output format and verify correctness expectations. (Attention: Stucked background op test processes and lock files under jit folder may cause the op tests running failed; Op tests require JIT compiling, please be prepared to wait for a long time)

### 2) Build a benchmark shell script
- Come up with a new name for this iteration and create a folder logs/<new name>. Put the shell script under this folder
- Reuse the existing op_test python script
- Covers common shapes: 128, 256, 512, 1024, 2048, 4096 if applies
- Repeats each op test multiple times and reports the correctness and the average time consuming.
  - Use at least 100 iterations per configuration for reliable results
  - Include 10-20 warmup iterations to handle JIT compilation overhead
  - Add torch.cuda.synchronize() after each kernel call
  - Use fixed random seed for reproducibility
  - Use high-precision timing (time.perf_counter())
- Implements a robust timeout to avoid hangs.
- Outputs structured timing per shape.

### 3) Establish a baseline

- **Before testing**: Check for background GPU processes that may interfere
  - Use `rocm-smi` or `ps aux | grep python` to identify GPU tasks
  - Stop any unrelated GPU workloads
- Clear JIT compilation cache to ensure clean state
- Run the benchmark script using the `.venv` Python environment
- Save results under logs/<new name> folder with timestamp


### 4) Iterate on kernel optimization (one iteration)

- Read the kernel source, identify bottlenecks, and call `rocprof-compute` at least once to deepen bottleneck analysis.
- Use `kernel-exp-history` to review related optimization history and extract ideas.
- Modify the kernel file to improve performance for multiple shapes allowed.
- Save the changes: (git diff > logs/<new name>/iter<N>_diff.patch)
- Reinstall aiter and clear cache:
  - `python -m pip install -e . --no-build-isolation --no-deps --force-reinstall`
  - `rm -f aiter/jit/*.so && rm -rf aiter/jit/build ~/.aiter`
- Re-run the benchmark to measure the new performance.
- **If results seem suspicious** (unexpected regressions):
  - Verify no background processes are running
  - Re-test baseline with same methodology
  - Check if JIT compilation overhead affected measurements


### 5) Record the iteration

- **Document the results**:
  - Save detailed analysis in logs/<name>/iter<N>_analysis.md
  - Include performance comparison table
  - Document any issues encountered (false regressions, test methodology problems)

- Use `kernel-exp-history` to store in database
- **Verify result quality**: If showing unexpected regression, investigate before recording
- Restore the repo code to the `main` branch state after finishing the iteration


### 6) Repeat iterations

- Repeat step 4 for ten iterations (no stop), each time measuring and recording results.
