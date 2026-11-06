## 9. Environment and Debugging

### IR Dump
```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dumps python my_kernel.py
```
Produces numbered `.mlir` files per pipeline stage plus `final_isa.s`.

### Key Environment Variables
| Variable | Default | Description |
|---|---|---|
| `FLYDSL_DUMP_IR` | false | Dump IR at each stage |
| `FLYDSL_DEBUG_ENABLE_DEBUG_INFO` | false | Emit DWARF debug info (source-to-asm mapping) |
| `FLYDSL_RUNTIME_ENABLE_CACHE` | true | Enable kernel disk caching (in-memory cache is always active) |
| `FLYDSL_RUNTIME_CACHE_DIR` | ~/.flydsl/cache | Cache directory |
| `FLYDSL_COMPILE_OPT_LEVEL` | 2 | Optimization level (0-3) |
| `ARCH` | auto-detect | Override GPU architecture |

### Disk Cache Invalidation
The JIT disk cache is keyed by the traced MLIR and the toolchain fingerprint, so Python edits and rebuilt C++ passes invalidate it. Set `FLYDSL_RUNTIME_ENABLE_CACHE=0` only when debugging a suspected stale artifact:
```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 python my_kernel.py  # or: rm -rf ~/.flydsl/cache
```

### Source-to-Assembly Debug Info

FlyDSL supports source-to-assembly mapping for rocprofv3 ATT traces via the MLIR
`ensure-debug-info-scope-on-llvm-func` pass (equivalent to Triton's `add_di_scope`).

**How it works**:
1. FlyDSL's `FuncLocationTracker` generates MLIR `loc()` metadata pointing to Python source lines
2. The `ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}` pass converts MLIR locations into LLVM `DISubprogramAttr` / `DICompileUnitAttr` metadata
3. That metadata is carried through MLIR-to-LLVM-IR translation and emitted as `.debug_line` in the HSACO binary
4. rocprofv3 ATT reads `.debug_line` to produce `code.json` with `"source_file:line"` entries

**Pipeline position**: After `reconcile-unrealized-casts`, before `gpu-module-to-binary`:
```
... -> reconcile-unrealized-casts
    -> ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}  (conditional on enable_debug_info)
    -> gpu-module-to-binary{format=fatbin opts=""}
```

**Verification**: With `FLYDSL_DUMP_IR=1`, check `final_isa.s` for `.file` and `.loc` directives.
The PA decode kernel achieves 99.9% coverage (1109/1110 ISA instructions mapped to source).

**Key insight**: Without this pass, MLIR `loc()` metadata is silently dropped during MLIR-to-LLVM-IR
translation. The DI scope pass is what produces the debug info; everything downstream only carries it.

⚠ `opts=` is empty above because nothing routed through it ever reaches AMD codegen —
FlyDSL used to pass `-g` there, and it did **nothing**.
`gpu-module-to-binary`'s `opts=` string is consumed only by the XeVM and NVVM targets —
`TargetOptions::tokenizeCmdOptions()` has no ROCDL caller, and `ROCDL::assembleIsa()` takes no
flags parameter at all — so the whole string is discarded without a diagnostic. Debug info
survives purely because the DI scope pass wrote it into the IR earlier. See `/llvm` for the
other knobs that ride this same dead path.

### Autotune Module

FlyDSL includes a Triton-style autotune module at `python/flydsl/autotune.py`:

```python
from flydsl.autotune import autotune, Config, do_bench

@autotune(
    configs=[
        Config(block_dim=64, vec_width=4),
        Config(block_dim=128, vec_width=4),
        Config(block_dim=256, vec_width=4),
    ],
    key=['const_n'],     # re-tune when these arg values change
    warmup=5, rep=25,    # benchmark timing params
)
@flyc.jit
def myKernel(A, C, n: fx.Int32, const_n: fx.Constexpr[int],
             block_dim: fx.Constexpr[int], vec_width: fx.Constexpr[int],
             stream: fx.Stream = fx.Stream(None)):
    ...
```

- `Config` kwargs become `Constexpr` args injected into `@jit` call
- `Config.num_warps` and `waves_per_eu` are special compiler-level options
- First call benchmarks all configs; subsequent calls use cached best
- Disk cache at `~/.flydsl/autotune/{func_name}.json`
- `do_bench(fn, warmup=5, rep=25)` benchmarks using CUDA/HIP events, returns median ms

**IMPORTANT**: `waves_per_eu` does NOT work via `gpu-module-to-binary opts=`, and this is not a
limitation that could be lifted: `opts=` is never read on AMD at all (only XeVM and NVVM consume
it), and `--amdgpu-waves-per-eu` is not a command-line flag in the first place — it is an IR
function attribute, which `llc` rejects if passed as a flag. The hint works because
`RocmBackend.lower_compile_hints` *also* sets the `rocdl.waves_per_eu` attribute, which becomes
LLVM's `"amdgpu-waves-per-eu"="N"`.

⚠ `maxnreg` had no such second path — it reached only the dead `opts=` lane and was therefore
silently inert, so it has been removed: the hint now raises, as does `Config(maxnreg=...)`.
See `/llvm` before tuning occupancy, and to verify any knob actually reached codegen.

**DLTensorAdaptor bug**: Do NOT use `flyc.from_dlpack()` with pre-wrapped tensors when calling
a `@jit` function with varying `Constexpr` values. The `DLTensorAdaptor` caches MLIR types from
the first `ir.Context`, which become invalid when a new context is created (causes segfault).
Pass raw `torch.Tensor` objects instead.
