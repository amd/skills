## Keeping a long-lived PR mergeable: sync with a merge, then re-audit the auto-merge

A ROCm PR can sit in review for weeks, and upstream keeps moving underneath it. The maintainer eventually says "please resolve conflicts", which is a mechanical request with a non-mechanical trap in it.

Sync with a MERGE of upstream's default branch, never a rebase, once the branch is published. Rebasing rewrites shas a maintainer has already read and comments are anchored to, and it breaks the guarantee that the published tip is still an ancestor of what you push, which is what makes a later fast-forward of the PR branch possible at all. Merge, resolve, and let the merge commit say which conflicts were resolved and how -- that message is the maintainer's shortest path to re-reviewing only the delta.

The conflicts git reports are the easy half. The dangerous half is what it auto-merges cleanly, because upstream keeps writing code against the exact surfaces the port MULTIPLEXED, and a textually clean merge writes only the NVIDIA half of the new code. Two shapes recur (both hit apache/mahout#1399, an additive HIP backend for the QDP Rust/CUDA engine, after six weeks of upstream movement):

- Upstream adds an entry point to a file the port split into per-vendor backends. The port had turned one `extern "C"` block into three compile-time backends (real CUDA runtime, no-toolkit stubs, HIP wrappers); upstream added a `cudaGetDeviceCount` binding plus a helper calling it. The merge landed the binding in the two branches upstream knew about, and the HIP branch silently lacked it, so a helper written at module scope over those names failed to compile on ROCm. Whenever upstream grows a vendor-runtime surface the port fanned out, add the new name to EVERY backend in the same commit.
- Upstream adds GPU code or GPU tests behind the OS/toolkit predicate the port replaced. New test files gated on `#[cfg(target_os = "linux")]` (the CMake equivalents are `if(UNIX)` and `if(CUDA_FOUND)`) merge cleanly into a tree whose other GPU code moved to a port-introduced predicate covering ROCm on more than one OS. Nothing fails; the new coverage just evaporates from the ROCm configuration. This is invisible in CI that only builds one OS, and shows up as a test run on another platform reporting fewer tests than a sibling.

A third shape comes from the port itself. If the port changed what a shared helper does on AMD -- for example widened a `warp_reduce_sum` to fold the native 64 lanes on CDNA -- upstream code written later still calls it with CUDA semantics. LichtFeld-Studio's new rasterizer kernels treat a warp as one 32-lane pixel group and call the widened helper from it; on a 64-lane wavefront that sums two groups and double-counts the atomics, and nothing fails on wave32. For each helper the port changed, list its new callers in upstream's delta and check the width they assume.

A fourth is a textual conflict that resolves wrongly: upstream adds a CUDA-only compatibility shim (the CCCL 3.0 `CUB_VERSION` alias shims are the current one) directly after an include the port wrapped in a HIP guard, and git can pair the shim's `#endif` with the guard's. build-headers-macros.md, "Upstream CCCL 3.0 compat shims are CUDA-only by construction", has the resolution. (aihwkit)

So after every sync, read upstream's own delta rather than only the merged result: `git log <old-base>..upstream/main -- <ported subtree>` and `git diff <old-base> upstream/main -- <ported subtree>`, then check each addition against every abstraction the port introduced (vendor backends, platform cfgs, feature gates, shim headers). A useful closing check is that the merged tree differs from your previous published tip by EXACTLY upstream's delta plus the port-side lines you deliberately added -- anything else means the merge dropped or duplicated port intent.

Measure the drift itself before each review and again before opening the PR: `git fetch`
upstream's default branch and check that `git rev-list --count <port-branch>..origin/<default>`
is 0. **`git diff <default>...<branch>` will not tell you**: three dots diff from the MERGE
BASE, so a branch cut months ago shows a clean, plausible diff against code upstream has
since rewritten. GooFit sat 19 commits behind through two full review rounds this way; one
of those commits repointed the very submodule the port special-cased and rewrote the exact
CMake block the port wrapped, so the diff under review was not the diff that would have been
proposed. After bringing the branch up to date, look for two things besides the checks
above: whether upstream has independently adopted part of your change (drop your copy
rather than proposing it twice), and whether a project-wide configure probe now answers
differently (GooFit's `check_ipo_supported()` flipped NO -> YES across the sync, so every
conclusion drawn from the cached value was stale). Re-measure such a probe rather than
trusting a cache or an earlier review, and remember that what a probe reports and what the
build generates are different questions (build-cmake-link.md, "Settle any 'which flags
does this build actually use' question"). (GooFit)

Then confirm mergeability and publishability with two ancestry checks, not by reading the GitHub badge: `git merge-base --is-ancestor upstream/main <branch>` (upstream is contained, so merging back is a fast-forward) and `git merge-base --is-ancestor <published-tip> <branch>` (the reviewed tip is still an ancestor, so the PR branch can be fast-forwarded to this work).

## A patch you carry against a pinned dependency: check "unfixed upstream" per entry

A port that pins a third-party dependency and carries fixes against it will want to say so in the README and the patch header, and the natural sentence -- "these are bugs in <dep> and none of them is fixed upstream" -- is a claim about a moving target that nobody re-checks after the first round. Some of the entries may be fixes the dependency has ALREADY made, on commits later than the tag you pinned; you backported them, which is why they are in your patch, and calling them unfixed upstream is wrong in a way a maintainer can disprove in one click. Two other counts drift the same way: how many entries the patch has (count `^--- a/` and the header table, they can disagree) and which project each entry belongs to (a dependency's vendored `contrib/` copy of a third project is that project's bug, not the dependency's).

Check each entry against the dependency's default branch, not the patch as a whole, and record the commit you checked against with the date. Read-only GitHub API fetches of the patched paths plus a grep for the fixed expression is a few minutes for a dozen files, and it partitions the patch into backports (keep the hunk, it is needed at the pinned tag, but say why it is there) and genuine defects (keep the hunk AND report it). Then scope every upstream-visible sentence to the partition it is true of, and prefer sending the reader to the patch header over restating part of it in a README, so there is one place to keep correct. `diff-surfel-tracing` shipped "none of them is fixed at HEAD" over eleven entries; three of them were already on HIP RT's main branch, and it took three review rounds to stop restating the count wrongly. (diff-surfel-tracing)
