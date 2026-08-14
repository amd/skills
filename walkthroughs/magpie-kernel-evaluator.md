# AMD Skills Walkthroughs: `magpie-kernel-evaluator`

Magpie connects model-serving benchmarks to GPU kernel optimization. This
walkthrough follows the complete workflow:

```mermaid
flowchart LR
    A["Benchmark baseline"] --> B["Profile the same workload"]
    B --> C["TraceLens stage and roofline analysis"]
    C --> D["Find top-k kernels and source"]
    D --> E["Analyze the baseline kernel"]
    E --> F["Create and analyze candidates"]
    F --> G["Compare correct candidates"]
    G --> H["Integrate the winner"]
    H --> I["Repeat the clean benchmark"]
```

Magpie provides three public modes:

| Mode | Use it to | Primary artifact |
|---|---|---|
| `benchmark` | Measure vLLM, SGLang, or Atom; post-process traces with TraceLens; locate bottlenecks | `benchmark_report.json` |
| `analyze` | Validate and profile one HIP, CUDA, PyTorch, or Triton kernel | `analyze_report.json` |
| `compare` | Validate and rank two or more implementations | `compare_report.json` |

The main path requires a model-serving environment. If one is unavailable, use
the lightweight `vector_add` smoke test near the end of this file to verify that
the skill triggers and uses Magpie correctly.

## Prerequisites

- A Linux system with a Magpie-supported AMD GPU and ROCm installation
- Python 3.10 or later, Git, Node.js, and `npx`
- Claude Code installed and authenticated
- Docker or a prepared local model-serving environment
- Enough GPU memory and disk space for the selected workload
- `HF_TOKEN` when the model is gated
- `hipcc` for HIP kernel work
- Optional: `rocprof-compute` for kernel profiling and ranking

Confirm the basic GPU toolchain:

```bash
rocminfo | head
hipcc --version
```

## Step 1 — Install Magpie and the skill

```bash
git clone https://github.com/AMD-AGI/Magpie.git
cd Magpie
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
magpie --gpu-info
```

Before installing the AMD skill, ask Claude Code which skills it sees. Then
install `magpie-kernel-evaluator` with the
[`skills` CLI](https://github.com/vercel-labs/skills). Run all three commands in
your terminal; the `npx` one is a shell command, not a Claude prompt:

```bash
claude "Which skills can you see?" --model sonnet
npx skills add amd/skills --skill magpie-kernel-evaluator --agent claude-code
claude "Which skills can you see?" --model sonnet
```

The second response should include `magpie-kernel-evaluator`.

## Step 2 — Benchmark, run TraceLens, and identify bottleneck kernels

Magpie benchmark configs live under `examples/benchmarks/`. Select a config for
the available framework, model, GPU count, memory, and container. Before running
it, ask the agent to validate the experiment:

```text
Use the Magpie skill to review <benchmark-config> for this machine. Check the
framework, model access, image, visible GPUs, tensor parallel size, precision,
concurrency, sequence lengths, timeout, output paths, and disk space. Create a
clean baseline config and an equivalent profiled config. Enable the profiling
needed for torch traces and TraceLens inference post-processing only in the
profiled config. Analyze all available inference stages. Do not start until
both configs are internally consistent, and preserve the effective configs.
```

The profiled config should include:

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

Run an unprofiled baseline for the performance measurement and a separate
profiled run for diagnosis. Profiling can perturb latency, so do not use the
profiled result as the final baseline.

```bash
magpie benchmark \
  --benchmark-config <clean-benchmark-config> \
  --output-dir ./results/walkthrough-benchmark-baseline

magpie benchmark \
  --benchmark-config <profiled-benchmark-config> \
  --output-dir ./results/walkthrough-benchmark-profiled
```

Review request/token throughput, completed requests, TTFT, TPOT, ITL, and
end-to-end latency. Record the Magpie commit, model revision, image/framework
version, GPU model and count, tensor parallelism, precision, concurrency,
input/output lengths, warmup, and profiler state.

### Review TraceLens post-processing

After the profiled workload completes, Magpie invokes TraceLens functions to post-process the captured PyTorch traces.
For inference mode it splits representative stage windows and writes full
reports for available `prefilldecode`, `decode`, and `prefill` stages. Confirm
that `benchmark_report.json` contains a successful `tracelens_analysis` section,
then locate the compact roofline summaries:

```bash
find <profiled-workspace>/tracelens \
  -name '*_kernel_roofline_simple.csv' -print
```

Open these summaries before choosing a kernel. Within each stage:

1. Rank operations by `kernel_time_ms_sum` or `time_pct`.
2. Use `roofline_bound` and arithmetic intensity to distinguish compute- and
   memory-bound work.
3. Check achieved TFLOP/s, achieved TB/s, `pct_roofline_mean`, and
   `has_perf_model` before forming an optimization hypothesis.
4. If direct single-rank or multi-rank collective reports are specifically
   required, use a separate `analysis_mode: pytorch` run and review
   `tracelens_rank0_csvs/` or `tracelens_collective_csvs/`.

The full reports live under stage directories such as
`tracelens/prefilldecode/`, `tracelens/decode_only/`, and
`tracelens/prefill_only/`. TraceLens supplies operation- and stage-level
evidence; use gap analysis next to rank concrete kernel names and map them to
source.

Magpie's integrated TraceLens stage produces CSV/Excel artifacts, not an
agent-written `analysis.md`. For a separate prioritized agentic report, pass the
captured trace to `tracelens-analysis-orchestrator` when that skill is installed
and label its output as downstream analysis rather than a Magpie-native report.

If gap analysis and source enrichment were not enabled in the profiled config,
run them on the generated trace:

```bash
magpie benchmark \
  --trace-dir <profiled-workspace>/torch_trace \
  --start-pct 20 \
  --end-pct 80 \
  --top-k 20 \
  --find-kernel-sources \
  --kernel-source-repos <framework-repo> <rocm-library-repo>
```

`--trace-dir` is an option directly on `magpie benchmark`; `gap-analysis` is not
a positional subcommand.

Ask the agent to recommend a kernel using measured evidence:

```text
Read the Magpie TraceLens and gap-analysis outputs from <profiled-workspace>.
First identify the dominant inference stage and operation using the compact
roofline summaries, then recommend one concrete kernel using total GPU time,
trace-window share, call count, roofline evidence, category, source path, and
available test command. Separate measured evidence from assumptions. Reject
candidates that cannot be mapped or tested safely.
```

Prefer a kernel with high total GPU contribution, a confident source mapping,
and a deterministic correctness test—not one selected from call count alone.

## Step 3 — Analyze the baseline kernel

Use the mapped `source_file`, `test_file`, and `test_cmd` fields when available.
Create a Magpie kernel config that pins representative shapes, dtypes, target
architecture, compile flags, environment variables, working directory, and
testcase.

```text
Use the Magpie skill to create an analyze config for the selected bottleneck and
its representative benchmark shape. Preserve the original source. Run the
existing test first, then run Magpie analyze under
./results/walkthrough-kernel-baseline. Record the exact commands and environment.
Stop if numerical correctness cannot be established.
```

Run the config-driven analysis:

```bash
magpie analyze \
  --kernel-config <baseline-kernel-config> \
  --output-dir ./results/walkthrough-kernel-baseline
```

Review `analyze_report.json` for correctness before interpreting performance.

> For PyTorch kernels without a testcase, Magpie only checks that each result is
> finite; it does not prove numerical equivalence. Require a representative
> testcase before accepting an optimized implementation.

## Step 4 — Create and analyze optimization candidates

Ask the agent for a small number of changes tied to the measured bottleneck.
Examples include tile shape, memory access, occupancy, launch geometry, fusion,
or specialization for the observed input shape.

```text
Using the selected kernel source, testcase, and Magpie baseline report, propose
up to three hypothesis-driven optimization candidates. State the measured
bottleneck and expected tradeoff for each. Implement candidates separately,
preserve the baseline, and do not relax tolerances or remove correctness checks.
Run Magpie analyze on every candidate with identical shapes, dtypes, GPU,
compile flags, testcase, and profiler settings. Exclude incorrect candidates.
```

Write each run to a separate output directory:

```bash
magpie analyze \
  --kernel-config <candidate-a-config> \
  --output-dir ./results/walkthrough-candidate-a

magpie analyze \
  --kernel-config <candidate-b-config> \
  --output-dir ./results/walkthrough-candidate-b
```

Review generated code and commands before execution, especially device
selection, build flags, environment variables, and edits outside the selected
kernel.

## Step 5 — Compare the correct candidates

Build one `kernels:` config containing the baseline and only candidates that
passed analyze:

```yaml
kernels:
  - id: baseline
    type: hip
    source_files: ["<baseline-source>"]
    working_dir: <working-directory>
    compile_command: "<baseline-build-command>"
    testcase_command: "<representative-test-command>"
  - id: candidate_a
    type: hip
    source_files: ["<candidate-a-source>"]
    working_dir: <working-directory>
    compile_command: "<candidate-a-build-command>"
    testcase_command: "<representative-test-command>"
```

Then run:

```bash
magpie compare \
  --kernel-config <compare-config> \
  --baseline 0 \
  --output-dir ./results/walkthrough-kernel-compare
```

Review `compare_report.json`. Report the correctness vector, performance scores,
ranking, winner, variance, and measurement caveats. Never select an incorrect
implementation, even if it is faster. Repeat noisy measurements before
declaring a winner.

## Step 6 — Integrate the winner and repeat the benchmark

Integrate only the winning correct candidate into the framework or library used
by the original benchmark. Rebuild the environment with only the changes needed
to load that candidate.

```text
Integrate the winning correct candidate and rerun the original clean Magpie
benchmark. Keep the model, precision, GPU allocation, tensor parallelism,
concurrency, input/output lengths, warmup, and all unrelated software versions
identical. Write the result under ./results/walkthrough-benchmark-optimized and
compare it with the immutable clean baseline. Report kernel-level and
end-to-end changes, run-to-run variance, regressions, and paths to every Magpie
report.
```

```bash
magpie benchmark \
  --benchmark-config <optimized-clean-config> \
  --output-dir ./results/walkthrough-benchmark-optimized
```

Compare at least:

- request and token throughput
- TTFT, TPOT, ITL, and tail latency when available
- completed and failed requests
- the selected kernel's total time and trace share
- whether another kernel or system component became the new bottleneck

Do not claim success from isolated kernel speedup alone. The result must preserve
correctness and improve the equivalent clean end-to-end benchmark beyond normal
run-to-run noise.

## Step 7 — Review the outputs

Each `--output-dir` is a base directory. Magpie creates a timestamped workspace
inside it, so the end-to-end walkthrough should produce a structure like:

```text
results/
├── walkthrough-benchmark-baseline/
│   └── benchmark_<framework>_<timestamp>/
│       └── benchmark_report.json
├── walkthrough-benchmark-profiled/
│   └── benchmark_<framework>_<timestamp>/
│       ├── benchmark_report.json
│       ├── torch_trace/
│       │   └── trace_split/
│       ├── tracelens/
│       │   ├── prefilldecode/
│       │   ├── decode_only/
│       │   ├── prefill_only/
│       │   └── *_kernel_roofline_simple.csv
│       └── gap_analysis/
├── walkthrough-kernel-baseline/
│   └── analyze_<kernel-label>_<timestamp>/
│       └── analyze_report.json
├── walkthrough-candidate-a/
│   └── analyze_<kernel-label>_<timestamp>/
│       └── analyze_report.json
├── walkthrough-candidate-b/
│   └── analyze_<kernel-label>_<timestamp>/
│       └── analyze_report.json
├── walkthrough-kernel-compare/
│   └── compare_<timestamp>/
│       └── compare_report.json
└── walkthrough-benchmark-optimized/
    └── benchmark_<framework>_<timestamp>/
        └── benchmark_report.json
```

Use the workspace path printed by Magpie, or locate reports with `find`, rather
than assuming a fixed timestamped directory name.

A successful run demonstrates that the agent:

1. Triggered `magpie-kernel-evaluator` and used Magpie instead of an ad hoc flow.
2. Used TraceLens stage and roofline evidence plus gap analysis to select a
   concrete kernel.
3. Established correctness before interpreting performance.
4. Compared candidates under equivalent conditions.
5. Revalidated the winner in the original clean workload.

## Step 8 — Run a lightweight smoke test (optional)

When a model-serving environment is unavailable, use Magpie's included HIP
`vector_add` example to verify skill triggering and structured reports.

Ask the agent:

```text
Use the Magpie skill to evaluate examples/simple_hip_test/vector_add.hip. Compile
the example, run analyze with examples/simple_hip_test/analyze_default.yaml and
performance profiling disabled, then compare separate O2 and O3 builds using a
temporary kernels config. Prove both variants correct before discussing
performance. Write results under ./results/walkthrough-smoke-test and do not
replace Magpie with an ad hoc workflow.
```

The analyze portion is equivalent to:

```bash
hipcc -g -O2 \
  -o examples/simple_hip_test/vector_add \
  examples/simple_hip_test/vector_add.hip

magpie analyze \
  --kernel-config examples/simple_hip_test/analyze_default.yaml \
  --no-perf \
  --output-dir ./results/walkthrough-smoke-test
```

The testcase prints `PASSED: All 1024 elements correct.` when successful. The
agent should also produce a temporary `kernels:` compare config for the O2 and
O3 binaries and a `compare_report.json`. This small input validates the workflow
but is not evidence of a production speedup.

## Step 9 — Try the same task without the skill (optional)

Remove `magpie-kernel-evaluator`, start a fresh agent session, and repeat the
smoke test. Without the skill, an agent is more likely to bypass Magpie, invent
a config shape, compare performance before correctness, or omit the structured
reports needed for review.

For the complete command and configuration reference, read the
[`magpie-kernel-evaluator` skill](../skills/magpie-kernel-evaluator/SKILL.md).
