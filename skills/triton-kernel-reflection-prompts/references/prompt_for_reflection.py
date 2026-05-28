# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Concise reflection/prompts for Triton kernels on AMD/ROCm.

prompt = """
You are an expert in Triton GPU kernels for AMD/ROCm. Analyze failed tests and explain why they failed and how to fix.

Problem:
{problem}

Attempted solution:
{solution}

Test results:
{test_result}

Instructions:
- Think first, then output only the reflection.
- Do NOT change function names/signatures.
- Wrap the reflection in ```reflection ...``` (markdown code fence).
"""

prompt_exe = """
You are an expert in Triton GPU kernels for AMD/ROCm. Analyze runnable + correctness tests and explain fixes.

Problem:
{problem}

Attempted solution:
{solution}

Runnable test result:
{call_test_result}

Correctness test result:
{exe_test_result}

Instructions:
- Think first, then output only the reflection.
- Do NOT change function names/signatures.
- Wrap the reflection in ```reflection ...``` (markdown code fence).
"""

prompt_ga = """
You are an expert in Triton GPU kernels for AMD/ROCm. Summarize the current kernel's optimization strategy and how to improve performance.

Problem:
{problem}

Triton code:
{code}

Performance:
speedup: {latency}
efficiency(TFLOPS, GB/s): {efficiency}

Instructions:
- Note current tricks (tiling, fusion, memory access, autotune knobs).
- Suggest concrete next improvements.
- Wrap the reflection in ```reflection ...``` (markdown code fence).
"""

system_prompt = """Output JSON: {"reflection": "..."} with only the reflection text."""

prompt_extract_strategy = """
You are an expert in Triton kernels for AMD/ROCm. Compare implementations and extract why the better one wins; list key strategies.

Original problem:
{instruction}

Function signatures (must stay exact):
{function_signatures}

Implementations and results:
{top_programs}

Instructions:
- Identify the better implementation from results/reflections.
- Summarize why it wins.
- List the winning optimization strategies.
- Output as ```reflection ...``` (markdown code fence).
"""

prompt_evolve_reflect = """
You are an expert in Triton kernels for AMD/ROCm. Given history, current code, and errors, explain why it failed and how to fix.

Original problem:
{instruction}

Function signatures (must stay exact):
{function_signatures}

Metrics info:
{metrics_info}

History:
{evolution_history}

Current program:
{current_program}

Test result:
{test_result}

Reflection on current program:
{reflection}

Instructions:
- Focus on failure cause and concrete fixes.
- Keep AMD/ROCm safe (no CUDA-only features).
- Use ```reflection ...``` (markdown code fence).
"""

prompt_evolve_strategy_optimize = """
You are an expert in Triton kernels for AMD/ROCm. Summarize current optimization strategy, note bottlenecks, and propose how to beat it.

Original problem:
{instruction}

Function signatures (must stay exact):
{function_signatures}

Metrics info:
{metrics_info}

History:
{evolution_history}

Current program:
{current_program}

Test result:
{test_result}

Reflection on current program:
{reflection}

Instructions:
- Call out tuning knobs (BLOCK sizes, num_warps, num_stages), fusion, memory access, stability.
- Suggest specific next changes.
- Output as ```reflection ...``` (markdown code fence).
"""
