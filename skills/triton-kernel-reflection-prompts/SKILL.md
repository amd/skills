---
name: triton-kernel-reflection-prompts
description: Reflection/self-critique prompts for reviewing and fixing AMD-targeted Triton kernels after generation or test failures.
---

# AMD Kernel Reflection Prompts

- Use after a kernel run/test to drive structured self-review and fixes.
- Load `references/prompt_for_reflection.py` for the full reflection prompt and guidance.

## How to use
- Summarize failures/perf gaps, then feed the reflection prompt to propose patches.
- Follow the checklist: correctness first, then performance and readability.
- Keep AMD-focused advice: wave64 occupancy, LDS/bank conflict avoidance, coalesced and vectorized memory access.
- Output schema should include proposed code changes plus rationale for downstream tools.

## References
- `references/prompt_for_reflection.py`: Reflection prompt definitions.
