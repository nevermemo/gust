# Semantics contract and known limitations

This page states what a `#[gpu]` kernel means on every GUST target, where the targets are allowed to differ, and
how sure each statement is. It is the executable-semantics contract of D22, narrowed to a defined domain by D38 and
confirmed by the owner in D39. The cases outside the domain are written as limitations for users: the owner's
position is that such differences are expected edge cases that developers avoid anyway, so GUST documents them and
does not try to make the targets match.

The CPU twin exists: `gust::cpu` (slice m3.b) is its backend and the `#[gpu]` macro emits the twin of every kernel with
a typed call (slice m3.c). The fixture tests assert the twin next to the GPU and the independent host reference, the
atomic functions run for real on the twin, and the ledger has a twin column (slice m3.d). `cargo xtask check-cpu`
(slice m3.c2) is a step of the Unit suite and runs the locked, GPU-free tests of the vector-add and polynomial
examples. What is still to come is marked **planned**: the remaining examples (m3.e1, m3.e2) and the `--cfg gust_dual`
checks and the cross-crate proof (m3.f1, m3.f2). A planned statement describes the intended behaviour and is not
evidence.

## 1. Scope

- **GPU target (exists).** A `#[gpu]` module is translated directly from the Rust syntax tree to Slang, `slangc`
  compiles the Slang to WGSL, and wgpu runs the WGSL. The measured adapter is NVIDIA GeForce RTX 5090 on Vulkan,
  driver 610.88, slangc 2026.13.1, 2026-10-24. The WGSL specification is the portable reference for this path. The
  contract does not extend to Slang's other targets: Slang itself leaves, for example, integer division by zero
  undefined. The test `the_emitted_wgsl_uses_plain_operators_and_conversions` checks that division, remainder,
  shifts, signed subtraction and multiplication, negation and the float-to-integer casts reach the WGSL as plain
  operators and `i32()` / `u32()` conversions; it does not check `+`, unsigned arithmetic or the float operators.
- **CPU twin (exists).** In a test build (`cfg(test)` of the crate that defines the kernel, or `--cfg gust_dual`, D26)
  the macro emits an `impl Run` for `gust::cpu::CpuDevice` that runs the rustc-checked body of each kernel, one
  invocation at a time in ascending `global_id.x`. A production build holds one target (D21): the twin is absent from
  it, which the check-ui fixture `typed-call-no-cpu-twin` proves for the defining crate (the cross-crate proof is
  planned, m3.f1). The production CPU target belongs to a later milestone (M5) and is not described here.
- **Not executed by the twin.** Kernels with a `Uniform<T>` parameter (they have no typed call), indirect dispatch
  and staged graphs stay GPU-only. A kernel can opt out explicitly with `#[kernel(cpu = "unsupported")]` (D26).
  Examples that use graphs will carry kernel-level twin evidence only (planned, m3.e1, m3.e2).
- **Language subset.** 32-bit `f32`, `i32`, `u32`, `bool` and module structs of them (README, "Current subset"). Float
  arithmetic is `+ - * /`, comparisons and casts: the validator accepts no other method than `.len()`,
  `.global_id()` and the `Option` / `Result` helpers, so there is no `sqrt`, `fma` or trigonometry to bound.
  The validator accepts `%` on floats (it rejects only an integer-literal zero as the right operand of `/` and `%`),
  and no bound is stated for it.
- **Independent references stay.** Tests of the translator keep an independent host reference as the expected value
  (D26); the twin is a third party in those tests, never the oracle.

WGSL section numbers and quotes on this page come from the WGSL editor's draft (gpuweb.github.io/gpuweb/wgsl), copy
read on 2026-10-24. A superscript in the spec is written with `^` here.

## 2. Labels

Every statement about GPU behaviour carries one of these, and they are never merged:

- **spec-allowed**: the WGSL specification defines the result or the allowed set, cited by section. Portable across
  adapters that run the WGSL that slangc emits for GUST. The test enforces the GPU value where the spec fixes one.
- **unspecified**: WGSL leaves the outcome open (indeterminate value, implementation-dependent, or a dynamic error;
  for example NaN converted to an integer). The GPU value in the ledger is an observation and not a requirement;
  the test prints a NOTE when it changes (shown with `--nocapture`, not in a default run) and never fails on it.
- **measured**: one adapter, one driver, one date (the adapter in section 1). Not portable.
- **proven**: a test pins the result on both targets and the pinned value lies inside the spec-allowed set. The
  statements that carry it are listed in section 8, each with its test; one matching observation never promotes a
  line. A GPU run against an independent host reference alone is labelled measured.
- **planned**: the twin will behave this way; nothing runs yet.

"Rust" in the ledger is the plain-Rust result in the crate that compiled the test. "Checks on" and "checks off" mean
the consuming build has `overflow-checks` enabled or not (D38); the test reads the mode at run time with the macro
`gust::cpu::overflow_checks!()`, which expands in the test crate, and asserts the matching column. The twin is
compiled in the crate that defines the kernel and so follows that build's mode; GUST pins no profile. The ledger is
one workspace profile, so the macro, the function `gust::cpu::overflow_checks()` and the twin agree there; the
cross-crate evidence (a dependent crate with a per-package override sees its own mode) is still planned, m3.f1.

## 3. The defined domain, and what lies outside it

D22 said integers are bit-exact across targets. D38 narrows that: the guarantee holds on the defined domain below
and nowhere else. Outside it the targets diverge, no diagnostic can prevent it because the operands are run-time
values, and GUST records both outcomes in the ledger (section 7) instead of removing the divergence.

D22 also said that entries of the known-mismatch file fail when an outcome changes. D39 narrows that for outcomes the
WGSL specification leaves open (NaN and infinity, subnormals, NaN conversions, out-of-bounds access), float edge cases
included: they are recorded as one-adapter observations and never fail a test. Outcomes the specification fixes, and
the plain-Rust outcomes, are still pinned and fail the test when they change.

Inside the domain the GPU and the twin give the same bits for integers, `bool` and control flow, and the
float rules of section 4 apply. The domain is:

1. `+`, `-`, `*` and unary `-` on `u32` and `i32` give a result that fits the type (KM-10).
2. Divisors are not zero, and `i32::MIN / -1` and `i32::MIN % -1` do not occur (KM-01 to KM-03).
3. Shift amounts are below 32 (KM-04). Bits shifted out by an in-range amount, such as `1u32 << 31`, are lost on
   both targets and are inside the domain.
4. `f32 as i32` and `f32 as u32` have a finite operand whose truncated value fits the target type (KM-05, KM-06).
5. Float operands, intermediate values and results are finite, and each is zero or normal (KM-07, KM-08).
6. Every index into a storage buffer is below its `len()`, and no two invocations race (section 5; KM-11).

What happens outside the domain, in one line each (the table has the exact values and the labels):

- Overflow: the GPU wraps modulo 2^32 and never reports it. WGSL section 6.2.3: "Expressions on concrete integer
  types that overflow produce a result that is modulo 2^bitwidth". Rust panics when `overflow-checks` is on and
  wraps when it is off.
- Division and remainder by zero, `MIN / -1`, `MIN % -1`: the GPU returns the dividend, 0, the dividend and 0
  (section 8.8). Rust panics in every profile.
- Shift by 32 or more: the GPU uses the amount modulo 32; section 8.10: "The number of bits to shift is the value of
  e2, modulo the bit width of e1." Rust panics when checks are on and masks when they are off.
- Out-of-range or NaN float-to-integer casts: Rust saturates to the integer's extremes with NaN as 0. The GPU clamps
  to the range that both types represent, so the top end differs (`2147483520`, `4294967040`), and NaN is
  indeterminate (section 15.7.6).
- Out-of-bounds access: the GPU raises a dynamic error and the outcome is open (section 6.5.7); Rust panics.

What users should do: keep run-time operands inside the domain by construction, or guard them in the kernel (test a
divisor before dividing, clamp before casting). A literal zero divisor is rejected at the Rust boundary with a
diagnostic; a run-time zero is not. There are no explicit `wrapping_*` operations in the dialect; they may become a
language feature of their own, not a way to close this gap. Do not rely on the twin to detect overflow on the GPU
path: the GPU wraps silently whatever the twin does.

**Limitation: arithmetic inside `gust` follows `gust`'s own overflow mode.** Rust fixes overflow behaviour when a
crate is compiled. An integer operation that the shadow body writes is compiled in the consumer's crate and follows
the consumer's mode (`gust::cpu::overflow_checks!()`). An integer operation that `gust` itself implements (a shadow
function or type provided by the library, as opposed to the kernel's own `+`) is compiled in `gust` and follows
`gust`'s mode (the function `gust::cpu::overflow_checks()`). The two modes differ only when a profile overrides
`overflow-checks` for one of the two packages. Today the only such operation is `atomic_add`. It never panics on
overflow, not even in a checked twin, unlike the kernel's own `+`: it wraps modulo 2^32 in every build, exactly as the
GPU does (the other atomic functions cannot overflow). So the case does not arise. This is documented, not fixed (D39).

The ledger test also runs controls inside the domain (integer cases, and float cases on exactly representable values,
so they say nothing about rounding). They must agree between the GPU and plain Rust in both overflow modes; a
disagreement there fails the test, because it would mean the domain is wrong.

## 4. Floats

D22 gives each float operation a documented error bound, because GPU `f32` is not strictly IEEE. The GPU column
below is what WGSL allows; the twin column is Rust's IEEE 754 binary32 with round-to-nearest-even, and Rust does not
fuse `a * b + c` by itself.

Quotes (WGSL editor's draft, section numbers as read):

- 15.7.4 (accuracy): "the result may be rounded up or down". The table in 15.7.4.1 lists `x + y`, `x - y`, `x * y`
  and `-x` as "Correctly rounded", and `x / y` as "2.5 ULP for |y| in the range [2^-126, 2^126]". Comparisons are
  "Correct result".
- 15.7.5 (fusion and reassociation): "An implementation may fuse operations if the transformed expression is at least
  as accurate as the original formulation." and "An implementation may reassociate operations."
- 15.7.2 (NaN and infinity): "Implementations may assume that overflow, infinities, and NaNs are not present during
  shader execution." A runtime expression that overflows or yields one then gives an indeterminate value (15.7.3),
  which section 8.2 defines as "some arbitrary implementation-chosen value of the static type".
- 15.7.2 (subnormals): inputs or outputs of the operations of 15.7.4 "may be flushed to zero". Other operations "are
  required to preserve subnormal numbers"; the arithmetic of this subset is in the first group. The same section
  lets an implementation ignore the sign of a zero.
- 15.7.6 (conversions): "for non-NaN cases, floating point to integer conversion clamps the value to be within the
  range of the target type, then rounds toward zero." For NaN: "If X is a NaN, the result is an indeterminate value
  in T." For integer to float, when the value is not representable: "WGSL does not specify whether the higher or
  lower representable value is chosen".

Per-operation bounds, with their input domains. "Domain" means finite operands and a finite result, each zero or
normal, unless the row says more. ULP is the unit in the last place of the result (15.7.4).
The last column is a spec-allowed
bound, not an observation.

| Operation | Domain | CPU twin | GPU, spec-allowed | Twin versus GPU |
| --- | --- | --- | --- | --- |
| `x + y`, `x - y`, `x * y` | domain | Nearest rounding: within 0.5 ULP of the exact result. | "Correctly rounded": either neighbour of the exact result, so within 1 ULP. | At most 1 ULP; identical when the exact result is representable. |
| `-x` | domain | Exact. | Exact ("Correctly rounded"). | Identical, except that the sign of a zero may differ (15.7.2). |
| `x / y` | domain, and `abs(y)` in `[2^-126, 2^126]` | Within 0.5 ULP. | Within 2.5 ULP. | At most 3 ULP. |
| `a * b + c`, any product feeding a sum | domain | Two roundings. | May be fused into one rounding (15.7.5). | No constant: derive it by error propagation (below). |
| a chain of additions or subtractions | domain | Evaluated in the written order. | May be regrouped, with no accuracy condition (15.7.5). | No constant: derive it over the worst grouping (below). |
| `x == y`, `x != y`, `x < y`, `x <= y`, `x > y`, `x >= y` | no NaN operand, no subnormal operand | IEEE 754. | "Correct result". | Identical truth values. |
| `i32 as f32`, `u32 as f32` | any | Nearest, ties to even. | Exact when the value is representable (all `abs(x)` up to 2^24); otherwise either neighbour (15.7.6). | Identical up to 2^24 in magnitude; above it at most 1 ULP, not measured. |
| `f32 as i32`, `f32 as u32` | truncated value fits the target type | Truncates toward zero. | Truncates toward zero (15.7.6). | Identical. Only in-range `f32 as i32` has a test (`numeric`, against a host reference, on the GPU and on the twin); no test covers `f32 as u32` inside the domain. |
| float `%` | accepted by the validator, not covered | Not stated. | WGSL inherits it from `x - y * trunc(x/y)` (15.7.4.1). | No bound stated. |
| NaN, infinity, subnormal operands or results, out-of-range casts | outside the domain | IEEE 754, kept subnormals, saturating casts. | Indeterminate, flushable, clamped (15.7.2, 15.7.6). | Ledger rows KM-05 to KM-08. |

Nothing in this table has been measured on a GPU except through the ledger rows, and the table is not a claim about
the RTX 5090. Several existing tests compare float results exactly with a host value (for example
`translated_semantics_execute_on_gpu` and two tests in `typed.rs`), but on exactly representable inputs, so no
existing test exercises float rounding. The first such test is the polynomial example's twin-versus-GPU assertion
(m3.c, below). None of this makes a row above proven.

**Limitation: reassociation.** WGSL 15.7.5 lets an implementation regroup a chain of additions with no accuracy
condition, so the GPU may add in a different order than the source. The twin adds in the written order.
Two correct programs can therefore differ by more than the per-operation bounds suggest, and a bound must be derived
over the worst grouping.

**Deriving a tolerance for a kernel (done for polynomial in m3.c; planned for the other examples, m3.e1).** An
expression-level tolerance is derived per test from these bounds and the test's input domain, and written in the test's doc comment. It is never an observed
maximum. Method: list the kernel's operations in order; mark the exact ones (multiplication by a power of two,
addition of 0, negation); bound each intermediate's magnitude from the input domain; give each inexact operation a
relative error of at most `u = 2^-24` on the twin and at most `2u` on the GPU for `+ - *` (one ULP of a normal
value is at most `2^-23` of its magnitude), and at most `5u` for `/` (2.5 ULP); propagate to first order; for
each chain of additions take the grouping with the largest partial sums on the GPU side (fusion never raises a bound;
reassociation can); the twin-versus-GPU bound is the sum of the two sides.
Observed maxima from measurement runs are recorded separately as observations on the measured adapter and never
become a tolerance. A sign flip near a root that stays inside the derived absolute bound is rounding, not a defect.

**Worked example (arithmetic on the kernel text, not a measurement).** The `evaluate` kernel of
`examples/polynomial` computes `squared = v * v`, `cubed = squared * v`, `curved = cubed - (2.0 * squared)`,
`linear = 0.5 * v` and `(curved + linear) + 1.0`, with inputs `k / 2048 - 1`, so `abs(v)` is at most 1. Multiplying by a
power of two is exact. Twin, in the written order: `squared` u, `cubed` 2u, `2.0 * squared` 2u, `curved` 7u (2u + 2u
plus u times its magnitude bound 3), plus `linear` 10.5u, plus 1.0 15u. GPU, with the worst grouping of the four
terms (magnitudes 1, 2, 0.5, 1): `squared` 2u, `cubed` 4u and `2.0 * squared` 4u from the products, then three
additions whose partial sums have magnitude at most 3, 4 and 4.5, costing 2u each time that magnitude, so 8u +
23u = 31u. The twin-versus-GPU bound is 15u + 31u = 46u, about 2.74e-6 absolute. The polynomial example's test
(slice m3.c) asserts 48u = 48 * 2^-24, about 2.86e-6, where the extra 2u is the margin for second-order terms.
Observation, not a tolerance: on the RTX 5090 the largest absolute twin-versus-GPU difference over the whole input
domain was 2.38e-7 (4u), and the twin equalled the host reference to the last bit.

## 5. Memory preconditions

A kernel is inside the contract only if both of these hold. A sequential CPU twin hides violations of the second.

- **In bounds.** Every index into a storage parameter is below its `len()`. A dispatch is rounded up to whole
  workgroups, so a kernel must guard the tail (the fixtures write `if i < out.len()`). An out-of-bounds access on the
  GPU is a dynamic error: WGSL section 6.5.7: "Executing an out-of-bounds access generates a dynamic error." The
  outcome is open: the invocation may trap, the load may return any value of the buffer or zero, the store may be
  dropped or may land in another location of the buffer. Memory accesses stay within the module's variables and the
  bound resources (section 2.2). Rust panics. KM-11 pins the observation (load returned 0, store dropped, with a
  4-element buffer and index 4).
- **Race-free.** Within one dispatch no two invocations write the same location, and no invocation reads a location
  that another writes, except through the atomic functions. On the GPU a race is a dynamic error, which section 2.2
  says "may or may not be detectable"; after a dynamic error the program "may not behave as described in the rest of
  this specification". Parameter alias rejection (D34) is a different guarantee and does not prevent this.
- **Why the twin cannot help.** The twin runs invocations one at a time in ascending `global_id.x`, so it
  produces one definite order. A racy kernel gives a clean twin result that matches a reference computed in the same
  order and proves nothing about the GPU, where the order is unspecified. Every reference in a fixture or example is
  therefore order-independent by construction: each written index has exactly one writer, or the kernel uses
  atomics.
- **Visibility.** The writes of a dispatch are visible after the dispatch completes. Within a dispatch, an invocation
  must not depend on another invocation's non-atomic writes.
- **Launch size (CPU backend).** The CPU backend (`gust::cpu`) rejects only a launch whose invocation total, rounded up
  to whole workgroups, is above 2^32 (`DeviceLimit`, because global ids are `u32`). The GPU backend also enforces the
  adapter's `max_compute_workgroups_per_dimension` (and any lower `with_limits` value). For example 5,000,000 elements
  at workgroup width 64 need 78125 workgroups: they run on the CPU backend and are `DeviceLimit` under wgpu's default
  limits (65535 workgroups). This is a documented difference (D39), not something to match; split the work into
  several dispatches for the GPU. The macro-emitted twin uses this backend (m3.c).

## 6. Atomics

- **GPU (exists).** `atomic_add`, `atomic_min`, `atomic_max`, `atomic_exchange` and `atomic_compare_exchange` are
  statement-only calls on `RWStructuredBuffer<u32>` or `<i32>` elements (`i32` has no min or max). WGSL section 17.8:
  "All atomic built-in functions use a relaxed memory ordering." and "No synchronization or ordering guarantees apply
  between atomic and non-atomic memory accesses". Ordering holds only among atomic operations on the same location,
  and GUST offers no stronger ordering. Do not use an atomic to publish non-atomic data to another invocation in the
  same dispatch.
- **The ordering rule.** An atomic whose final result does not depend on the order of the contending invocations
  gives the same result on both targets: `atomic_add` (it wraps modulo 2^32 on both targets by definition, its
  specification body in 17.8.3.1 is `old + v`, and the twin's version uses `wrapping_add`, unit-tested at
  `u32::MAX` and `i32::MAX` in `runtime.rs`; no kernel test drives an atomic add past the limit), `atomic_min` and
  `atomic_max`. A result that depends on which invocation goes first (`atomic_exchange`, a contended
  `atomic_compare_exchange`) is set-valued on the GPU: any invocation may win. The twin runs invocations in ascending
  order and so yields one member of that set: the exchange of 257 invocations ends at 256, the contended
  compare-exchange from 999 to `id + 1` at 1 (KM-09). That is the twin's order rule and nothing else: tests of such
  kernels accept the whole set on the GPU, never the twin's value, and the twin's value is asserted separately.
- **The CPU twin (exists, m3.d).** `atomic_add`, `atomic_min`, `atomic_max`, `atomic_exchange` and
  `atomic_compare_exchange` in `crates/gust/src/runtime.rs` are real sequential operations on `u32` and `i32`
  through a sealed helper trait, so a float receiver stays a rustc error (the validator already rejects it). Because the
  twin runs one invocation at a time, a plain read-modify-write is atomic. The shadow functions return the old value
  (the exchange and the arithmetic ones) or whether the swap happened (compare-exchange); in a kernel the GPU form is
  statement-only, so the returned value is not observable there. The twin's compare-exchange never fails
  spuriously.
- **Compare-exchange may fail spuriously.** WGSL's only compare-exchange is `atomicCompareExchangeWeak` (17.8.5):
  "The equality comparison may spuriously fail on some implementations." So even an uncontended compare-exchange
  that should succeed is not guaranteed to, and in the contended case the unchanged value (999 in the ledger case)
  is also a spec-allowed result. That the success case succeeds is a measured observation, not a guarantee.
- **Tested (GPU measured, twin exact).** `crates/gust-wgpu/tests/atomics.rs` checks a contended `atomic_add` sum, the
  contended minimum and maximum, the compare-exchange success and failure cases, and signed add, exchange and
  compare-exchange exactly, on the GPU and on the twin. The twin's final cell is compared directly with the GPU's in
  every test except the contended exchange and compare-exchange; in the signed sequence (add, exchange, compare-exchange
  on one cell) each step of both sides is pinned to the same literal (-4, -5, 11) and only the final state is compared
  directly. For the contended
  exchange the GPU must end in 0 to 256 and for the contended compare-exchange in 1 to 257 or at the unchanged 999
  (the weak compare-exchange may fail every attempt); the twin is asserted at 256 and at 1 respectively, and never
  compared with the GPU there. The returned old value of an atomic is never observable in GPU code (statement-only);
  the twin returns it and `runtime.rs` unit-tests it.
- **Known limitation of the GPU half.** The compare-exchange success case (a single invocation, 7 to 11) is asserted
  as succeeding. WGSL allows a spurious failure there too (17.8.5), so on the GPU that is a measured observation,
  not a guarantee; on the twin it holds by definition. Because the unchanged 999 is accepted for the contended case,
  a GPU compare-exchange that did nothing would pass the contended GPU assertion; the success case and the twin
  assertion are what catch a no-op.

## 7. Known-mismatch ledger

The ledger lists the constructs where the GPU and plain Rust (which the CPU target follows) give different results,
or where the GPU's result is unspecified, and that no diagnostic can prevent because the operands are run-time
values. Differences like these are expected and acceptable (D39): they are documented here, not removed.

Each row is pinned by `crates/gust-wgpu/tests/mismatch.rs`, which runs every case on the GPU, as plain Rust and on the
CPU twin (the macro-emitted twin of the same ledger kernel, called through the typed `dispatch`, one case per
dispatch); an id guard in that file keeps this table and the test in step. The column "Rust (and CPU twin)" holds the
plain-Rust result, and the test asserts that the twin's outcome is accepted by that same expectation, in the overflow
mode of the test crate: the twin runs the shadow body compiled in that crate, so it panics or wraps exactly as the
Rust column does. The twin is compared with Rust and never with the GPU, which D39 allows to differ. Each row's twin
outcome is therefore pinned, including for the rows where the specification leaves the GPU open (KM-05 to KM-08,
KM-11), where the GPU value stays an unenforced observation. A kernel without storage parameters is covered by a
separate test in the same file. Labels are defined in section 2. No ledger row is **proven**: a row records a
difference, and the proven label needs the two targets to agree.

| Id | Case | Rust (and CPU twin) | GPU | Label |
| --- | --- | --- | --- | --- |
| KM-01 | `x / 0` for `u32` and `i32` | Panics "attempt to divide by zero" in every profile. | Returns the dividend: `7 / 0 == 7`, `u32::MAX / 0 == u32::MAX`, `-7 / 0 == -7` (WGSL section 8.8). | spec-allowed, measured |
| KM-02 | `x % 0` for `u32` and `i32` | Panics "attempt to calculate the remainder with a divisor of zero" in every profile. | Returns 0 (WGSL section 8.8). | spec-allowed, measured |
| KM-03 | `i32::MIN / -1` and `i32::MIN % -1` | Panic "attempt to divide with overflow" and "attempt to calculate the remainder with overflow" in every profile. | `MIN / -1 == i32::MIN` and `MIN % -1 == 0` (WGSL section 8.8). | spec-allowed, measured |
| KM-04 | Shift by a run-time amount of 32 or more (`<<`, `>>` on `u32` and `i32`) | Checks on: panics "attempt to shift left with overflow" or "attempt to shift right with overflow". Checks off: the amount is masked to its low 5 bits. | The amount is masked to its low 5 bits (WGSL section 8.10): `1 << 32 == 1`, `1 << 33 == 2`, `1 << 64 == 1`, `0x8000_0000 >> 33 == 0x4000_0000`, `-8i32 >> 33 == -4`. | spec-allowed, measured |
| KM-05 | `f32 as i32` of a value outside the `i32` range, and of NaN | Saturates: `1.0e10 -> i32::MAX`, `-1.0e10 -> i32::MIN`, `NaN -> 0`, in every profile. | Finite values: the `i32` value closest to the truncated source that `f32` represents exactly (WGSL section 15.7.6), which differs from Rust at the top end: `1.0e10 -> 2147483520` and `2^31 -> 2147483520` (not `i32::MAX`), `-1.0e10 -> i32::MIN`. NaN: indeterminate; observed `i32::MIN`. | spec-allowed (finite), unspecified (NaN), measured |
| KM-06 | `f32 as u32` of a negative, too large or NaN value | Saturates: negative and NaN give 0, `1.0e10` and `2^32` give `u32::MAX`, in every profile. | Finite values (WGSL section 15.7.6, as KM-05): negatives such as `-1.0`, `-3.99` and `-1.0e10` give 0; `1.0e10` and `2^32` give 4294967040 (`0xffffff00`), not `u32::MAX`. NaN: indeterminate; observed 0. | spec-allowed (finite), unspecified (NaN), measured |
| KM-07 | NaN and infinity through arithmetic and comparison | IEEE 754: NaN propagates, `NaN != NaN` is true, ordered comparisons with NaN are false, `1.0 / 0.0` is `+inf`. | Unspecified (an operation that would produce NaN or infinity may give an indeterminate value; comparisons with NaN are not guaranteed to be unordered). Observed: NaN and infinity propagate and compare as in IEEE for every case except `NaN != NaN`, which is false. The sign and payload of a NaN are not preserved either: the GPU produced `0x7fffffff` in every case, Rust produces `0x7fc00000` or `0xffc00000`. Division by a divisor above 2^126 is also outside WGSL's division accuracy range: `3.0e38 / 3.0e38` gave 0 on the GPU, Rust gives 1. | unspecified, measured |
| KM-08 | Denormal (subnormal) `f32` operands and results | Kept: `1.0e-40 * 1.0` is `1.0e-40`, `1.0e-40 == 0.0` is false. | Unspecified (flushing to zero is allowed). Observed: operands and results flushed to zero (`1.0e-40 * 1.0`, `+ 0.0`, `* 2.0`, `/ 2.0` and `4.0e-38 / 4.0` give 0), and `1.0e-40 == 0.0` is true. | unspecified, measured |
| KM-09 | Contended `atomic_exchange` and `atomic_compare_exchange` (257 invocations, one cell) | Sequential, ascending invocation id: the exchange ends at 256; the compare-exchange from 999 to `id + 1` ends at 1. | Set-valued: the exchange ends at some id in 0 to 256, the compare-exchange at some value in 1 to 257, or at 999 (unchanged) because the weak compare-exchange may fail spuriously (WGSL section 17.8.5); the test accepts all of that set. Observed over 32 dispatches per run, three runs: 1 to 10 distinct final values per run, and never the sequential result (the exchange ended between 1 and 251, the compare-exchange between 7 and 137). | spec-allowed (the set), measured |
| KM-10 | Overflow of `+`, `-`, `*` and unary `-` on `u32` and `i32` | Checks on: panics ("attempt to add with overflow" and so on). Checks off: wraps modulo 2^32. | Wraps modulo 2^32 (WGSL section 6.2.3): `u32::MAX + 1 == 0`, `i32::MAX + 1 == i32::MIN`, `0u32 - 1 == u32::MAX`, `-i32::MIN == i32::MIN`. | spec-allowed, measured |
| KM-11 | Out-of-bounds index equal to the buffer length | Panics "index out of bounds" in every profile. | Unspecified (a dynamic error, WGSL section 6.5.7). Observed with a 4-element buffer and index 4: the load returned 0 and the store was dropped (the buffer was unchanged). | unspecified, measured |

Reading the table: the Rust column and every GPU value that the specification fixes (labelled spec-allowed) are
enforced, so the test fails when one changes and the row is edited in the same change. A value marked unspecified is
an observation: the test prints a NOTE when it changes and never fails (D39). Adding a construct to the ledger means
adding
its `KM-nn` row here and its case in `crates/gust-wgpu/tests/mismatch.rs`; the id guard fails if only one is done.

## 8. What is proven

A statement here has the **proven** label of section 2: one test asserts the same result on the GPU, on the CPU twin
and against an independent host reference, on the inputs the test uses, and the result lies inside the set that the
WGSL specification allows. It is a statement about those inputs and operations inside the defined domain (section 3),
not a general theorem. The twin half is `cfg(test)` code of the test crate (slice m3.d); the GPU half ran on the
adapter in section 1.

- **Integer, `bool` and control-flow results of the fixture kernels**, with element counts 1, 63, 64, 65 and 257:
  `semantics` (`translated_semantics_execute_on_gpu`: integers and the float branches on small exactly representable
  values), `numeric` (signed and unsigned division and remainder with non-zero divisors, the in-range `f32 as i32`
  cast, struct construction), `option`, `result` (the `ok` and `err` input sets), `loops` and `struct_assignment`.
- **Atomics** (`atomics.rs`): the contended `atomic_add` sum, the contended `atomic_min` and `atomic_max`, the failing
  compare-exchange, and signed `atomic_add` and `atomic_exchange`. Not proven: the succeeding compare-exchange (single
  and signed), because a weak compare-exchange may fail spuriously on the GPU (section 6), and the contended
  exchange and compare-exchange, where the targets are allowed to differ (KM-09).

Everything else on this page stays **spec-allowed**, **unspecified**, **measured** or **planned**. The floats of
section 4 remain measured on the polynomial example only; the signal-pipeline example (m3.e1) is where more float
evidence will come from.
