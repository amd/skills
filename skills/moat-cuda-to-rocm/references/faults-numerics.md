# Faults: floating point and integer arithmetic

Results that differ from the CUDA build or a CPU reference although nothing crashed.

<!-- toc:begin -->
Contents:

- Contraction and rounding
- Host math under the HIP compiler
- Conversions and multi-word integers
- Half-precision GEMM accuracy
<!-- toc:end -->

## Contraction and rounding

**clang for HIP defaults to `-ffp-contract=fast`** and forms FMAs ACROSS statements -- and
so does nvcc. nvcc contracts by default too (`--fmad=true`, and NVRTC likewise), and not
only within one expression: `float d = a*b; r = d + c;` compiles to one `fma.rn.f32` in PTX
and one FFMA in SASS (nvcc 12.8, sm_90; GPUMD), the same as the one-expression `a * b + c`.
`-ffp-contract=on` is therefore STRICTER than nvcc, not a match for it: it fuses only within
one source expression (clang emits `llvm.fmuladd` in the front end), while `fast` lets the
backend fuse whatever survives inlining. FMA formation still differs between the compilers,
so results can drift ~1 ULP from the CUDA build and CPU gold under either setting.
`-ffp-contract=off` is the `--fmad=false` equivalent. MATCH UPSTREAM's setting rather than
pinning one unconditionally: CV-CUDA's tests passed with `-ffp-contract=on`, while
LC-framework's nvcc recipe uses `-fmad=false -mno-fma -ffp-contract=off`, so pinning `on`
there would diverge. Choose `on` for what it guarantees: two template instantiations or call
sites built from the same source expressions form the same FMAs. GPUMD's
batched-vs-per-slot replica tests compare such instantiations bitwise and failed on gfx90a
under `fast`. It did not make CV-CUDA's `float` and `float4` instantiations of one template
agree (next entry). Integer-only components are unaffected.
(CV-CUDA OpWarpPerspective; LC-framework; GPUMD.)

**One contraction setting cannot satisfy both kinds of bit-exact test.** A project can have
tests that compare a kernel against a CPU gold (which want the contraction nvcc and the host
use) AND tests that compare two GPU code paths against each other byte-for-byte (which want
the same contraction in both instantiations). clang contracts a `float` instantiation and a
`float4` instantiation of the same template differently, so a per-plane path and a per-pixel
path that are bit-identical on NVIDIA differ by 1 ULP on ROCm under either setting. Measure
before choosing, and clear unrelated defects out of the sample first: in one suite a third of
the apparent contraction failures were an allocator ordering race and a test-fixture
use-after-free, and a decision taken on the raw count would have rested on the wrong
evidence. Treat the setting as a reviewed decision with numbers attached, and re-measure
after every functional fix that lands in the same suite. (CV-CUDA: `-ffp-contract=off`
removed most planar-versus-interleaved parity failures but regressed the cubic-versus-gold
case, so `on` was kept and the parity residual documented.)

**When bit-exact tests pin nvcc's results and no global setting is clean, read nvcc's fused
multiply-adds from the CUDA build and spell them out at each site, on the HIP device pass
only.** Three rules decide the spelling. (1) nvcc fuses `a * b + c` after inlining, so a
multiply and an add in different inlined functions still fuse: a vector-operator library's
`alpha * v + beta` is `operator+(operator*(alpha, v), beta)`, one FMA on nvcc, two roundings
under clang `-ffp-contract=on`. A helper instantiated for `float` then fuses where its
`float3` instantiation does not, and planar-versus-interleaved parity tests differ in the last
bit. A test gold written with `std::fma` is the tell. `fast` fixes those sites and breaks
others (in one suite about 75 of 183 bit-exact failures flipped and 39 new ones appeared). (2)
For a sum of two products, `a * b + c * d`, nvcc follows no fixed left or right rule, so read
its choice per site from the CUDA build's SASS (`cuobjdump -sass`, or `nvdisasm -g -c` on a
`-lineinfo` build to map instructions to source lines) instead of predicting it. With nvcc
12.8 the fused product changed with the element type, with the channel of a vector type
(`uchar3` fused the right product in channel x and the left one in y and z), with the column
inside one expression, with whether a product was hoisted out of a loop and reused, and with
a negated factor (`x * c0 + y * (-c1)` became `fma(x, c0, -(y * c1))`, while `x * (-c3) + y *
c4` in the same kernel became `fma(y, c4, -(x * c3))`). Each choice was fixed for a given
instantiation, column or channel, so a spelling per site on the HIP side can match it exactly.
Parity tests between two layouts that share the helper cannot tell the orders apart; a
temporary test that recomputes the outputs on the host with `std::fma` in nvcc's order shows
whether the HIP side now matches. (3) Keep host code unfused: `__host__ __device__` functions
also compile for the host, whose compiler does not contract in the CUDA build, and
host-evaluated golds expect that. A macro that expands to an explicit per-element fma under
`__HIP_DEVICE_COMPILE__` and to the original expression, operand order included, everywhere
else leaves the CUDA build's expression trees unchanged. Find the sites by scanning device
code for `acc += a * b`, `acc = acc + a * b` and blends written as one expression. (CV-CUDA
v0.19, gfx90a: 163 bit-exact failures to 0.)

**A test that fails deterministically on some AMD targets only, with an error that scales
with a sampling step, can be one ulp of contraction landing on a `ceilf`/`floorf` knife
edge.** Look for `ceilf`, `floorf` or integer casts of float quantities that are exact
integers in exact arithmetic (a ray length equal to a box dimension, a count derived from
length over step). LLVM contracts `a*a + b*b + c*c` differently per target: CDNA (gfx90a,
gfx942) put two squares into one `v_pk_mul_f32` and rounded both, while RDNA (gfx11, gfx12,
no packed FP32) built an FMA chain. The one-ulp difference becomes a whole extra sample. It
is not a wavefront-width bug, even when it splits along wave64/wave32 lines. Diagnose by
printing the integer from the device for one differing element, emulating the expression in
float32 with each contraction order, and forcing the order with `__fmaf_rn`/`__fmul_rn` in a
scratch build. The strict-IEEE CPU evaluation is not the reference: nvcc emitted
`fma(dz, dz, fma(dx, dx, dy*dy))` (PTX `mul.f32` then two `fma.rn.f32`), the order RDNA
happened to get. Compile the kernel with `nvcc -ptx` (the pip `nvidia-cuda-nvcc` and
`nvidia-nvvm` wheels are enough, no GPU needed), and if CUDA's order passes, write that order
with `fmaf` under the HIP guard; it pins the result on every AMD target without touching the
CUDA path. (TIGRE)

**AMDGPU can fold a float-to-half conversion into the preceding fma, rounding once where the
source rounds twice.** `__float2half_rn(__fmaf_rn(a, b, c))`, and any float-to-half
conversion of an fma result including a contracted `a * b + c`, can compile to one
`v_fma_mixlo_f16`/`v_fma_mixhi_f16`, which rounds straight to half instead of to float and
then to half. On gfx90a with HIP 7.16 (AMD clang 24), 230 of 4194304 random inputs gave a
different half, under every `-ffp-contract` setting. Symptom: F16 outputs one half-precision
unit off between two kernels that compute the same thing (the fold fires in one instantiation
and not the other), or against a CUDA reference. Check the ISA for `v_fma_mix*`. Workaround,
on the device pass only: an empty asm on the float result before the conversion (`asm("" :
"+v"(x));`), which costs nothing and keeps both roundings. (CV-CUDA resize and color
conversion, F16)

**`-ffast-math` is NOT the HIP translation of nvcc's `--use_fast_math`, and reaching for it
because the CUDA flags line has one is a real mistranslation.** nvcc's flag swaps single
precision transcendentals and division for the fast device intrinsics and leaves double
precision and IEEE exception behaviour alone. clang's `-ffast-math` additionally turns on
`-ffinite-math-only`, `-fassociative-math`, `-freciprocal-math`, `-fno-signed-zeros` and
`-ffp-contract=fast`, for DOUBLE precision as well. Geometric predicates -- continuous
collision detection, intersection and orientation tests, anything whose sign is the answer
-- are exactly what reassociation and no-NaN assumptions break, and the symptom is a solver
that stops converging rather than a wrong pixel. Translate `--use_fast_math` as at most
`-ffp-contract=<upstream's setting>` (first entry of this section), and leave everything
else IEEE. (GPU_IPC)

**`-ffast-math` implies `-ffinite-math-only` under clang; nvcc's `--use_fast_math` does
not assume finite math.** Translating the flag one for one therefore changes behaviour: any
algorithm that uses Infinity or NaN as a sentinel (depth sentinels, NaN clear values,
NaN-compare early-outs) silently produces nothing. The clang warning "use of infinity is
undefined behavior due to the currently enabled floating-point options" is the fingerprint;
treat it as fatal, not as noise. Drop the finite-math assumption on the HIP side
(`-fno-finite-math-only` after `-ffast-math`, or the individual fast-math flags the code
actually tolerates). (CuRast, whose runtime-compiled renderer drew nothing.)

**`__fsqrt_rn` is not always correctly rounded on gfx90a** (1 ULP high) where CUDA's
`sqrt.rn.f32` is IEEE correctly-rounded, so bit-exact tests pass on NVIDIA and fail on
gfx90a. Route the f32 device sqrt through the correctly-rounded f64 `__dsqrt_rn` and cast
back; CDNA has fast f64 sqrt so the cost is negligible. (CV-CUDA: OpNormalize, L2
PairwiseMatcher.)

**Device `cuda::min`/`cuda::max` NaN-selection can differ from host `std::min`/`std::max`.**
Host `std::min(a,b)` is `b<a?b:a` and `std::max(a,b)` is `a<b?b:a`; a device version written
`a<b?a:b` picks the OPPOSITE operand on a NaN compare, so tests filling buffers with random
bytes and comparing host gold against device output differ bit-for-bit. Spell the device
ternaries to match the host forms exactly. (CV-CUDA: OpMorphology CLOSE on RGBAf32.)

**An exact float-equality branch fed by approximate division can HANG, not just drift.**
HIP's `__fdividef` differs ~1 ULP from CUDA, so a downstream exact-equality test on the
quotient -- selecting a loop start or end index -- can flip on AMD. When that index feeds an
UNSIGNED loop bound via subtraction, the out-of-range value underflows to a huge count and
the loop never terminates: a data-dependent hang, not a wrong-by-1-ULP number. Do not gate
control flow on exact equality of approximate fp; use a tolerance, or cap the derived bound
at its provable geometric maximum. This is the class where numeric drift becomes a
CONTROL-FLOW failure. (TIGRE Siddon_projection: `Np` derived from `__fdividef` comparisons,
a 0.2s kernel ran >10 min at some angles; capped at `Nx+Ny+Nz+3`, a no-op for valid rays and
identical on CUDA.)

**A hand-written Jacobi/eigenvalue sweep replacing a library solver must normalize first.**
A port sometimes has to replace a library decomposition that does not compile for the
device (build-headers-macros.md, "Eigen DECOMPOSITIONS called from `__device__` code").
A cyclic Jacobi convergence test compares SUMS OF SQUARES (`off <= eps*eps*(on+off)`), so a
double input whose largest coefficient exceeds about 1.3e154 makes both sides `inf`, the
test reads true at sweep zero, and the routine returns the identity rotation -- laundering a
blown-up matrix into a plausible-looking result instead of letting the blow-up propagate to
where the caller would notice. Divide by `max|a_ij|` up front and scale the eigenvalues back
at the end; Eigen's own solver does this, which is why the original did not have the bug.
Two lines, and it applies to any hand-rolled replacement for a library decomposition.
(GPU_IPC `selfAdjointJacobi`.)

## Host math under the HIP compiler

**Host `min`/`max` on non-`int` values truncate to int under hipcc.** CUDA
declares host `min`/`max` for int, unsigned, long, long long, float and double. HIP's
`__clang_hip_math.h` declares only `min`/`max(int, int)` on the host by default. So host code in a
`.cu` file such as `max(0.25, min(100.0, fwhm))` or `max(atan(a), atan(b))` compiles
cleanly, converts both arguments to int and returns an int. Device code is unaffected,
because the device overloads cover float and double. Expect parameters silently rounded
down, a value below 1 clamped to 0, and geometry decisions taken on a zero angle. The only
build symptom is `-Wliteral-conversion` on fractional literals ("changes value from 0.25
to 0"), and only when a literal is involved; a variable argument, such as `x` in a
`__host__ __device__ float clamp(float x) { return min(max(x, 0.25f), 3.75f); }`, gets no
warning for itself. Unsigned and 64-bit arguments are hit too, which "the arguments are
integers" makes easy to miss: `max(3000000000u, 1u)` converts both to int and returns 1.
Detection: add
`-D__HIP_NO_HOST_MIN_MAX_IN_GLOBAL_NAMESPACE__ -include poison.h` to the real hipcc
command with `--cuda-host-only -fsyntax-only`. `poison.h` redefines the default pair with
`__attribute__((diagnose_if(true, "HOSTMINMAX", "warning")))`, so every host call that
lands on `(int, int)` is reported with its location. Define CUDA's full overload set the
same way, each overload naming its own signature, to see which calls really take floating
point. Use `diagnose_if`, not `deprecated`: a `deprecated` version missed LEAP's two
`ramp_filter.cu` sites (cause not traced). Fix: change the call sites to what CUDA's
overload calls. `crt/math_functions.hpp` implements `min`/`max(double, double)` and the
mixed float/double forms with `fmin`/`fmax`, and `(float, float)` with `fminf`/`fmaxf`.
Cast a float argument to double where CUDA took a mixed overload, e.g.
`fmax(0.25, fmin(100.0, double(fwhm)))`. That keeps CUDA's results, NaN handling included,
on every ROCm and on CUDA, without an `#ifdef`. Leave calls whose arguments are both `int`
alone; give unsigned, `long` and `long long` calls a small typed helper (or `std::min<T>` /
`std::max<T>` with the type spelled, in a TU no hipify pass rewrites -- see the `std::` note
below), so nothing reaches `(int, int)`.

`-D__HIP_DEFINE_EXTENDED_HOST_MIN_MAX__=1` is the macro the header offers, and it is a
supplement to that fix, not a replacement. It defaults to 0 in `__clang_hip_math.h` and, when
set, adds host overloads for the unsigned and 64-bit integer types and for float, double and
mixed float-double. Mixed signed/unsigned pairs need a second macro,
`__HIP_DEFINE_MIXED_HOST_MIN_MAX__`, which is off for a reason the header states (it compares
unsigned). The macro arrived in ROCm 7.2.0 (LLVM 0248d277ca); in 7.1.x and earlier the define
silently does nothing and the truncation stays. So a build that only adds the define is
correct on the ROCm it was tested on and wrong on an older one, with no diagnostic. If a
project does use it, make the older ROCm fail loudly with a host-pass assertion in the
compat header:

    #if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_DEVICE_COMPILE__)
    #include <type_traits>
    static_assert(std::is_same<decltype(min(1.0f, 2.0f)), float>::value,
                  "host min(float,float) resolves to min(int,int): fix the call sites or "
                  "build with ROCm >= 7.2 and -D__HIP_DEFINE_EXTENDED_HOST_MIN_MAX__=1");
    #endif

Either way, rerun the detection pass with no define: no host call may still land on
`(int, int)` with non-int arguments. Then confirm that the device code objects did not change
(`.text`/`.rodata` of the unbundled `.hip_fatbin`; the `__hip_cuid_*` names change with path and flags).
(LEAP: bilateral FWHMs truncated, intensity FWHM < 1 gave the identity; the modular-beam
Joseph backprojector chose its parallel kernel for cone data, failing adjointness and FBP;
the FBPlowpass window zeroed all but DC.) `std::min`/`std::max` are NOT safe either once a
hipify pass has run: hipify_torch (and torch's hipify, same table) rewrites them to
`::min`/`::max`, which land on the same `(int, int)` host pair, so a correct
`std::min(float, float)` in CUDA source truncates in the hipified copy. Run the detection
on the hipified sources, not the originals. (torch's hipify stopped rewriting `std::` math in
pytorch#200419, 2026-10; a torch older than that, and hipify_torch, still do.) When a build compiles host `.cpp` files as
HIP too, those files are in scope, but unqualified calls there usually bind to `std::min`
through `using namespace std` (a template exact match beats the int pair); to tell a real
truncation from that, report only calls that land on `(int, int)` with ROCm's default
pair AND on a non-int overload with CUDA's full set. (LEAP version two: 27 sites, 10 of
them new, including `1.0 / max(fabs(v_min), fabs(v_max))` dividing by zero in the cone
geometry setup.)

**A HIP workaround in shared host code must not change CUDA's numbers.** HIP declares
`rsqrtf` for the device only and host `min`/`max` only for `(int, int)` (the entry above),
and HIP's `__half`
converts to every floating type through a template, so a port that rewrites a host call for
HIP (`rsqrtf(x)` to `1.0f / std::sqrt(x)`, `std::pow(a, b)` to a cast through `double`,
`__half` to an integer through `float`) silently changes the CUDA build too: `1.0f / sqrtf`
rounds twice and moved 9 of 51 coefficients by 1 ULP, `pow` through double differed in the
last bit, and the float route made NaN/Inf half-to-int undefined where CUDA's
`__half2int_rz` saturates. Gate the rewrite on `__HIPCC__` and keep upstream's
expression in the `#else`. Do not narrow the gate to the toolchain where the error first
showed: `std::pow(__half, __half)` was ambiguous in the HIP host pass on Linux too (glibc
`double` vs libstdc++ `float`) once a `__half` kernel instantiated it. A `_MSC_VER` gate
was chosen because an older Linux build had compiled upstream's call, but that build had no
gate and predated the kernels that instantiate it, and the first Linux build with the gate
failed; key the gate to the cause (HIP's `__half`), not to the first OS that
failed; where HIP needs a substitute, write the formula CUDA's host math
uses (`(float)(1.0 / sqrt((double)x))` for host `rsqrtf`), and give conversions CUDA's
saturation and NaN rules. A compile-only CUDA build with a per-TU preprocessed diff against
upstream is what found these; HIP runs cannot. (LichtFeld-Studio)

## Conversions and multi-word integers

**HIP's OCP fp8 conversions do not compile for gfx942; E5M2 is the top byte of a half.**
`hip_fp8.h`'s `__hip_cvt_*` functions assert "type is unsupported by current target device"
for `__HIP_E5M2`/`__HIP_E4M3` when compiled for gfx942, which supports only the FNUZ encodings,
so CUDA code that stores data (a kv cache) as OCP E5M2 cannot use them there. E5M2 is exactly
the upper byte of an IEEE half: decode is `half_bits = byte << 8`; encode from half is
round-to-nearest-even on the low byte (`(mag + 0x7f + ((mag >> 8) & 1)) >> 8`) with saturation
to 0x7b and NaN to 0x7f; from float, rebias the exponent and round the mantissa to 2 bits, with
a separate subnormal branch below 2^-14. That is bit-identical to `torch.float8_e5m2` over every
half and bf16 pattern. An E5M2 value is also exact in bf16 (the upper half of its float bits),
which avoids a rounding conversion entirely. (ColossalAI, ROCm 10.2)

**Converting a NEGATIVE float to an UNSIGNED integer diverges silently between the back
ends** -- it is undefined behaviour, and the two hardware conversions pick different
answers. Measured on gfx90a, `(uint64_t)(-2.28)` is `0x00000000fffffffe`: the AMD lowering
of a 64-bit `fptoui` keeps only the low 32 bits of the two's-complement value. NVIDIA's
`cvt.rzi.u64.f64` saturates a negative source to zero. So a wrapped-negative idiom that
works on CUDA -- `(T)z + (flag & modulus)` to reduce a signed sample into `[0, q)`, or any
`(unsigned)x` relying on modular wraparound -- produces values near 2^32 on AMD. Grep for
`static_cast<unsigned...>(` / `(uint64_t)` applied to anything that can be negative, and
route the truncation through the signed type first: `(T)(std::make_signed_t<T>)x`. Note the
32-BIT case does NOT diverge: `(uint32_t)(-2.28)` clamps to 0 on both, so only the 64-bit
conversion needs the fix and a blanket sweep of the 32-bit sites would gratuitously change
CUDA behaviour. Symptom to recognise: an output whose magnitude clusters near 2^32 with
roughly half the values affected, i.e. exactly the negative half of a signed distribution.
The fix is NOT behaviour-neutral upstream (it replaces a saturated zero with the correct
negative representative), so say so in the commit message rather than presenting it as a
pure port change. (HEonGPU: RNGonGPU's Box-Muller Gaussian sampler; every ciphertext
decrypted to noise while plaintext encoding, which samples no error, passed -- which is the
tell that the fault is in a SAMPLER and not in the transform everything shares.)

### PTX carry-chain arithmetic (multi-word integers)

Big-integer and modular-reduction code reaches for PTX because CUDA exposes the carry flag
and C++ does not: `add.cc.u64` / `addc.cc.u64` / `subc.u64` chains, and `mul.lo.u64` +
`mul.hi.u64` pairs. None of it survives hipify, so it is hand-translated, and the
translation is easy to get subtly wrong in a way that compiles, runs, and silently corrupts
results. Guard the original with `#ifdef __HIP_PLATFORM_AMD__` and leave the NVIDIA path
byte-identical rather than replacing it for both.

The two replacements, with the predicates spelled out, since these are what get fumbled:

    // mul.lo/mul.hi -> 128-bit product
    lo = a * b;
    hi = __umul64hi(a, b);

    // add.cc/addc chain -> carry out of a + b + carry_in
    sum   = a + b + carry;
    carry = (sum < a || (carry && sum == a)) ? 1 : 0;

    // sub.cc/subc chain -> borrow out of a - b - borrow_in
    diff   = a - b - borrow;
    borrow = (a < b || (a == b && borrow)) ? 1 : 0;

The add predicate needs both clauses. With `carry_in = 1` and `b == 2^64 - 1` the sum equals
`a` exactly and a carry is nevertheless produced; testing only `sum < a` drops it. The
symmetric case is why the borrow predicate tests `a == b && borrow` and reads the ORIGINAL
`a` and `b`, not the computed difference.

**Establish the limb order from the asm operand numbering before writing anything.** Outputs
are numbered before inputs, so in

    : "=l"(v.y), "=l"(v.x)        // %0 = v.y, %1 = v.x
    : "l"(a), "l"(b)              // %2 = a,   %3 = b
    mul.lo.u64 %1, %2, %3;        // -> v.x is the LOW limb

a struct whose members are named `x` and `y` can carry low in either one, and the surrounding
C++ shift operators are the cross-check. Getting this backwards produces plausible-looking
garbage. (HEonGPU)

### Shift counts of 0 or >= 64 in hand-rolled 128-bit arithmetic

The same hand-rolled `uint128_t` (a pair of 64-bit limbs) that carries the PTX above almost
always implements `operator<<` / `operator>>` as three unguarded limb shifts:

    result.lo = value.lo >> shift;
    result.lo = (value.hi << (64 - shift)) | result.lo;   // shift == 0 -> hi << 64
    result.hi = value.hi >> shift;                        // shift >= 64 -> UB

Shifting a 64-bit value by 64 or more, or by `64 - 0`, is undefined in C++ and **the two back
ends resolve it differently**: PTX `shr.u64` / `shl.u64` clamp the count to the register
width and so yield 0 for any count >= 64, while AMDGPU's `v_lshrrev_b64` uses only the LOW 6
BITS of the count, so a shift of 64 acts as a shift of 0 and of 65 as a shift of 1. AMD
returns the wrong limb OR-ed into the right one at every count >= 64. This is the 64-bit
sibling of the 32-bit mask fault in faults-wavefront.md (a shift count that wraps at 32
because the mask was built in `int`).

This is data-dependent, not arch-dependent, so it hides until the shift count crosses 64.
Barrett reduction shifts by `modulus.bit + 3`, which reaches 64 at a 61-bit modulus: on
HEonGPU every modular multiplication was correct up to 60 bits and garbage at 61 and 62, so
CKKS (37-bit moduli) passed while BFV, whose decryption uses a 61-bit correction modulus,
decrypted to noise. **Sweep the modulus/exponent width, not just the input values, when
validating modular arithmetic** -- a probe that only tests the widths a passing scheme uses
proves nothing about the one that fails.

Fix by branching on `shift == 0`, `shift < 64` and `shift < 128` explicitly. No `#ifdef` is
needed, but **do not tell the maintainer his CUDA numbers are unchanged**: the equivalence
with PTX holds only for counts 0 through 64. At exactly 64 the clamp happens to give the
right answer -- `lo >> 64` is 0, `hi << (64 - 64)` is `hi`, so the low limb already comes out
as `hi` and the high limb as 0, which is what the explicit branch computes. From 65 up it
does not: `64 - shift` underflows, PTX clamps every term and returns zero for the whole
value, while the correct result (and what the explicit branch computes) is
`hi >> (shift - 64)` in the low limb. So the guard is a CUDA-side fix too, at any count
above 64 -- on HEonGPU that is a 62-bit modulus, which the class documents as supported.
Check the reachable count range against the type's documented limits before writing
"behaviour-preserving" in a commit message, and tell the maintainer about the CUDA-visible
change. Audit `ROTL64`-style rotate macros in the same file for the `n == 0` case.
(HEonGPU: GPU-NTT `modular_arith.cuh`.)

Before carrying such a fix, and again at every upstream sync, check whether upstream has
fixed the defect itself. A fix written in inline PTX does not reach the HIP build: give the
HIP branch a helper with exactly the result of upstream's PTX for every shift count the
callers can produce, and drop your own fix rather than keeping a second, different one.
(HEonGPU: upstream GPU-NTT later fixed the 61-bit case with PTX shift helpers, and the port
replaced its explicit branches with HIP helpers that return zero for counts of 64 or more.)

The same hardware difference bites 32-bit shifts whose count comes from data or from a count
that can reach the word width, not only lane-index masks (faults-wavefront.md). NVIDIA's `shl`
clamps the count; AMD uses its low five bits. Two forms: a data-derived shift in packed
counters, `1u << ((radix - first) << 4)` for radix values outside the current group, where the
code relies on the result being 0 (on ROCm the counters picked up bogus increments and ranks
ran past their buffers, a memory fault on the first frame); and a mask from a count that can be
exactly 32, `(1 << n) - 1` with `n` the children of a 32-wide tree node or the lanes of the last
sub-warp, which NVIDIA turns into all 32 lanes and AMD into 0. The second can be silent: an
empty mask fed a bounding-box reduction over no lanes, and a downstream overlap test happened
to report a hit on the resulting infinite box. Spell out the intended result (`n < 32 ? (1u <<
n) - 1 : ~0u`). A grep for mask expressions missed sites; what found them was listing every
shift whose count is not a literal and classifying each count's range (a script that strips
comments and takes the right operand up to the first lower-precedence operator; one project had
280 such sites). (PhysX)

## Half-precision GEMM accuracy

**rocBLAS `gemm_ex` rounds a 16-bit output once per k-chunk when k is long.** rocBLAS 5.7
(ROCm 10.1) splits k exactly when `max(lda, ldb) * k >= 2^30` for 2-byte types
(`gemm_tensile.hpp`: `safe_k = (2^31 - 1) / (max(ld_a, ld_b) * sizeof(Ti))`, split iff
`safe_k < k`), with the leading dimensions as passed whatever the transposes, ignoring strides
and batch count, into chunks of `safe_k`, and accumulates each chunk's product into D with
beta = 1. ROCm 10.2 (rocm-libraries #12219) narrows the split to the operands k strides
through, with the same 2^31-byte limit, so the 10.1 condition remains a safe superset. With a float16/bfloat16 D (`cublasGemmEx` -> `hipblasGemmEx` with compute type
f32) the output is rounded about k / 1023 times for a contiguous k-major operand, so the error
grows linearly with k: at k = 2^20 it is 25-50x a single rounding (fp16 7e-3, bf16 5e-2 relative),
and the 1025 chunk launches also take ~8 ms. `ROCBLAS_LAYER` logs only the outer call; a kernel
trace or `ROCBLAS_USE_HIPBLASLT=1` with the hipBLASLt log shows the chunks. A float D
(written to scratch, cast once) restores accuracy but is ~10x slower still; hipBLASLt
(`hipblasLtMatmul`, COMPUTE_32F, a few MiB of workspace) accumulates in float, rounds once and
runs these shapes in 45-85 us on MI300X, which is what torch uses there. Route only 16-bit
GEMMs with `max(lda, ldb) * k >= 2^30` to hipBLASLt so ordinary shapes keep their rocBLAS
path; strided batched (including stride 0 broadcast) works the same way. A regression test
needs k = 2^20 to clear a 4-ulp bound robustly; k = 2^15 splits only once. (oneflow: matmul
primitive, grouped_matmul_bias, fused self-attention.)

**gfx11 and gfx12 WMMA drift over long k, in every library, with any output type.**
Symptom: a half-precision (float16 or bfloat16) GEMM with a long k is several units of
roundoff off on RDNA3 and RDNA4 while the same call is accurate on CDNA. Every hipBLASLt
algorithm gives the same error, torch matches it digit for digit, rocBLAS does too wherever
it does not chunk k itself, and a float output does not help.

Cause: the float accumulator of the WMMA instruction (`v_wmma_f32_16x16x16_f16`) does not
round to nearest. The bias is toward zero and grows linearly with k: one wave carrying it
through 2^20 products is off by 3.4e-3 relative on gfx1100 and 1.7e-3 on gfx1201, against
3.1e-5 for an fp32 FMA chain (about 6-8 fp16 ulp at k = 2^20 on gfx1100, 2.5-3.7 on
gfx1201). The error also grows with the size of the partial sums, so a GEMM whose large
partial sums cancel is tens of ulp off even at k = 8192. gfx9 MFMA keeps fp32 accuracy.

Fix: choosing another algorithm does not help; shortening the chain does. Chunk k yourself,
gated on the k length on the WMMA families (not on `ld * k`). A GEMM whose k never exceeds
2^15 needs none of this.

- Cut k into equal chunks of at most 2^15 (chunks of about 2^13 measured slightly more
  accurate, 0.9-1.2 ulp, and no slower). Run them as batched GEMMs (batch stride = chunk
  along k) that write each chunk's product to its own slice IN THE 16-BIT TYPE OF D, then sum
  the slices in float into D, rounding once.
- Do not give the chunks a float D. On gfx1100 hipBLASLt picks a much smaller tile for a
  float D and runs 9 to 28 times slower.
- Do not rely on hipBLASLt's own split-k (`GemmTuning::setSplitK`). Its partials need
  m * n * split floats of workspace, and on gfx1201 (ROCm 10.1) the heuristic's first kernel
  for k-major A and small outputs is wrong by tens of ulp for many k above 2^17, with or
  without the split.
- A 16-bit slice limits each chunk's sum to that type's range and precision: a chunk can be
  inf where the chunks cancel to a finite result, and large chunk sums that cancel are
  rounded relative to their own size. So the kernel that sums the slices must recompute, from
  A and B as float dot products over all of k, every block of outputs where a slice is not
  finite or where a BOUND on the rounding error exceeds 3 units of roundoff of the output
  type (3 x 2^-11 float16, 3 x 2^-8 bfloat16) times the block's largest result. The bound is
  the sum of half an ulp of each slice in its type (0 below the smallest normal, so all-zero
  inputs never recompute) plus half an ulp of each float partial sum. It needs no host sync,
  works under graph capture and keeps true inf/NaN inputs non-finite.
- Do not replace the bound with a root-sum-square test. That bounds only the EXPECTED error;
  chunk sums that all round the same way pass it and lose 4 to 9 ulp.
- The bound almost never fires on full blocks of ordinary data, but often on blocks with few
  valid elements (thin outputs such as a matrix-vector product), so the recompute must be
  cheap: spread k over the block's threads with 16-byte loads and, when the output has few
  blocks, over several thread blocks whose partial sums are added in a fixed order. Watch the
  register count while doing it: a fully unrolled load loop took the kernel from 66 to 248
  VGPRs and made the common case 11% slower.
- Run each chunk on the library ordinary GEMMs use (rocBLAS was about 1.5x faster than
  hipBLASLt for large outputs on gfx1100) unless rocBLAS would split it again (the entry
  above). Fold the batch into the chunk calls, tile the slices over m and n to a memory
  budget, and if a library has no algorithm for one tile, run that tile alone through rocBLAS
  with a float D rather than failing the call after earlier tiles were written.

Limits: chunking bounds the chain length, not cancellation inside one chunk, which the slice
check cannot see; only a float accumulation without WMMA brings such cases to about 1 ulp,
at 7 to 200 times the cost. Realistic loss-scaled gradients stayed at 0.55-0.92 ulp, and
with the scheme above large weight gradients ran within about 1.1x of one plain GEMM.
(oneflow: long-k matmul on gfx1100 and gfx1201, ROCm 10.1; costs measured on gfx1100 only.)

**A GEMM workaround covers every entry point that reaches the library, fused epilogues included.**
oneflow routed its matmul, bmm and grouped GEMMs to the chunked long-k path but its cuBLASLt
compat (`cublasLtMatmul` on hipBLASLt, used by the fused linear and MLP kernels and their
backward) still sent the whole k to hipBLASLt: 5.7-8.4 float16 ulp on gfx1100 at k = 2^20 and
98.75 ulp on gfx1201 at k = 163840 (the ROCm 10.1 hipBLASLt DepthU-192 kernels, fixed in hipBLASLt
1.5.0). Grep for every call into the BLAS libraries, including Lt/fused wrappers, and put the
selection in the one compat function they share. Run the GEMM through the same path and apply
the epilogue (bias, ReLU, ReLU bit mask, masked gradient, bias gradient) afterwards in separate
kernels; the result then equals the unfused ops bit for bit, which also gives the test a check
that the path was taken (gate it on ROCm: cuBLASLt applies the epilogue before rounding). If the
long-k path needs another library's handle (rocBLAS here) and the compat only receives the Lt
handle, create the second handle with the first: a kernel may first run inside a graph capture,
where the 32 MiB a rocBLAS handle allocates at creation cannot be. (oneflow, 0e606373b.)
