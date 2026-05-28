Search the kernel corpus for reusable Triton/HIP patterns without loading the full file.

- Corpus file: `skills/amd-kernel-patterns/references/train_crawl.json` (~24k lines, copied locally).
- Quick grep examples:
  - `rg -n "attention|flash" skills/amd-kernel-patterns/references/train_crawl.json`
  - `rg -n "layer[_-]?norm" ...`
  - `rg -n "activation" ...`
  - `rg -n "triton" ...`
  - `rg -n "hip" ...`
- After finding a hit, slice a small window with `sed -n 'start,endp'` to extract code + descriptions.
- Adapt to AMD: wave64 occupancy, LDS tiling, vectorized loads/stores, avoid bank conflicts, coalesced global access.
- Cite file and line numbers when reusing snippets; trim to only what you need.
