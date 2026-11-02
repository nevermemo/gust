# Semantics contract and known limitations

This page states what a `#[gpu]` kernel means on every GUST target, where the targets are allowed to differ, and
how sure each statement is. It is the executable-semantics contract of D22, narrowed to a defined domain by D38 and
confirmed by the owner in D39. The cases outside the domain are written as limitations for users: the owner's
position is that such differences are expected edge cases that developers avoid anyway, so GUST documents them and
does not try to make the targets match.

The CPU twin exists: `gust::cpu` (slice m3.b) is its backend and the `#[gpu]` macro emits the twin of every kernel with
a typed call (slice m3.c). The fixture tests assert the twin next to the GPU and the independent host reference, the
atomic functions run for real on the twin, and the ledger has a twin column (slice m3.d). `cargo xtask check-cpu`
(slice m3.c2) is a step of the Unit suite and runs the locked, GPU-free tests of the vector-add, polynomial,
signal-pipeline, particle-step, typed-pipeline, atomic-counter (slice m3.e1), staged-graph and component-pool (slice
m3.e2) examples. `cargo xtask check-dual` (slices m3.f1 and m3.f2) proves across crates that the twin is absent from a
production build and present in a `--cfg gust_dual` build, and, with an integer kernel whose `+` overflows, that a twin
runs in the overflow mode of the package that defines its kernel: the twin of a dependency's kernel panics when the
dependency has checks on while the dependent has them off, and the twin of a kernel the dependent defines itself wraps
(the dependent's own override does not reach a dependency's twin). The label **planned** is kept for a statement about behaviour that does not run yet; a planned statement
describes the intended behaviour and is not evidence.

## 1. Scope

- **GPU target (exists).** A `#[gpu]` module is translated directly from the Rust syntax tree to Slang, `slangc`
  compiles the Slang to WGSL, and wgpu runs the WGSL. The measured adapter is NVIDIA GeForce RTX 5090 on Vulkan,
  driver 610.88, slangc 2026.20, 2026-11-01 (the semantics were first measured on slangc 2026.13.1, 2026-10-24, and the
  Gpu suite passed again on 2026.20). The WGSL specification is the portable reference for this path. The
  contract does not extend to Slang's other targets: Slang itself leaves, for example, integer division by zero
  undefined. Every `/`, `%`, `/=` and `%=` whose divisor is not a literal lowers to the overloaded helper
  `__gust_div` or `__gust_rem` (`__gust_div_assign` and `__gust_rem_assign` with the place `inout`, so it is
  evaluated once), whose body is the plain operator, because slangc folds integer division and refuses a divisor that
  folds to zero even in code that no invocation runs (D74, cured in slice m6.dv); slangc keeps each helper as a WGSL
  function, so the WGSL calls `_gust_div_0(..)` and `_gust_rem_0(..)` where the operator stood, a literal divisor
  stays a plain operator (a literal zero is rejected by the validator, E0317), and the results do not change
  (`crates/gust-wgpu/tests/division.rs` compares them with a host reference and with KM-01 to KM-03). The test
  `the_emitted_wgsl_uses_plain_operators_and_conversions` checks that shifts, signed subtraction and multiplication,
  negation and the float-to-integer casts reach the WGSL as plain operators and `i32()` / `u32()` conversions, and that
  the bodies of the division helpers are the plain `/` and `%` on the helper's own parameters, in modules that name
  no matrix (a module that names `Mat2`, `Mat3` or
  `Mat4` lowers every `*` and `*=` to the overloaded helper `__gust_mul`, whose body is the same `*` for scalars and
  component-wise products and `mul` for a matrix product, and every unary `-` on a non-literal to `__gust_neg`, whose
  body is `-a` for scalars and vectors and `a * -1.0` for a matrix, because slangc's own `-m` is WGSL that naga
  refuses (`mat4x4<f32>(0) - m`) and `m * -1.0` equals negation bit for bit under IEEE arithmetic for every value but NaN
  (a device that drops the sign of a zero, which Vulkan allows without `SignedZeroInfNanPreserve`, or flushes a
  subnormal in a multiply, may differ from the twin's sign flip: D39, measured in section 4); slangc keeps each
  helper as a WGSL function, so the WGSL of such a module calls `_gust_mul_0(..)` where the plain `*` stood, and the
  WGSL of every other module is byte-identical to what it was before that lowering, apart from the division helpers
  above; slangc's WGSL for a float vector `-v` is `vec3<f32>(0) - v`, so by
  the IEEE rules a `+0` lane comes out `+0` where the twin gives `-0`, an edge-case difference left as it is (D39;
  not run on a GPU)); it does not check `+`, unsigned arithmetic or the float operators.
- **CPU twin (exists).** In a test build (`cfg(test)` of the crate that defines the kernel, or `--cfg gust_dual`, D26)
  the macro emits an `impl Run` for `gust::cpu::CpuDevice` that runs the rustc-checked body of each kernel, one
  invocation at a time in ascending `global_id.x`. A production build holds one target (D21): the twin is absent from
  it, which the check-ui fixture `typed-call-no-cpu-twin` proves for the defining crate and the `tests/dual` fixture of
  `check-dual` proves for a dependent crate (m3.f1).
  Under `GUST_PROFILE=cpu` the twin is the production CPU target (next item).
- **Production CPU target (exists, m5.e1 and m5.e2, D51).** `GUST_PROFILE=cpu` (never inferred) makes the CPU the one
  target of a build (D21). The macro emits the twin's `Run` impl for `CpuDevice` in every build and uses no artifact.
  The build holds no Slang compiler, helper or GPU runtime: the build script of `gust` refuses `slang-runtime`, and
  `gust-wgpu` fails to compile (`require_gpu_target`). The contract is the twin's, stated once for production:
  - *Execution.* One thread per dispatch. Invocations run in ascending `global_id.x`, in whole workgroups, as in the
    twin (sections 5 and 6). A result that is order-dependent on the GPU (a contended exchange or compare-exchange) is
    the sequential member of the GPU's set. Compare-exchange never fails spuriously here.
  - *Integers.* The same as the GPU inside the domain (section 3). Outside it the target follows the `overflow-checks`
    of the crate that defines the kernel (D43). In the default release profile `+ - *`, negation and shift amounts
    wrap and mask as the GPU does, and in a checked build they panic. Zero divisors and `MIN / -1` panic in every
    profile. Out-of-range float-to-integer casts saturate. These are the limitations of section 3 (D39).
  - *Floats.* IEEE 754 binary32, round to nearest even, in the written order, with no fusion (the twin column of
    section 4). Subnormals are kept, and NaN and infinity propagate. KM-07 and KM-08 differ from the GPU and stay
    unspecified there.
  - *Memory.* An out-of-bounds index panics (KM-11). A racy kernel cannot race here, so its CPU result is deterministic
    and proves nothing about the GPU (section 5).
  - *Limits.* Only the `u32` id range is enforced. The GPU's workgroup and storage limits are not emulated (section 5).
  - *Panics.* A panic in a kernel body follows the build's panic strategy. It unwinds out of `dispatch` and `submit`
    under `panic = "unwind"` and aborts the process under `panic = "abort"`. Nothing catches it.
  - *Parallelism.* None inside a dispatch. Independent dispatches on independent devices run on any thread the
    caller owns. A parallel strategy is a later decision (roadmap M7).
  - *Evidence.* `cargo xtask check-cpu-target` builds a fixture in release under `cpu`, runs three kernels against
    independent host references, scans the binary for the compiler, the helper and the GPU runtime, and checks the
    refusals. That is `Cpu` evidence, never GPU evidence (D42).
- **Uniforms (exist, m6.u1 host API and m6.u2 macro).** A uniform is one value in a constant buffer, bound to a
  `Uniform<T>` (alias `ConstantBuffer<T>`) parameter through a resource of kind `gust::kind::Uniform` (`GpuUniform<T>`
  on the GPU, `CpuUniform<T>` on the twin). The generated call names `Resource::Kind` in every bound (`Storage` for a
  buffer parameter, `Uniform` for a uniform one), so a uniform where a buffer is expected, or the reverse, is a compile
  error (the `typed-call-kind-mismatch` case of `tests/ui`); a hand-built call is still refused at submit
  (`BindingKind`). A uniform is read-only for the kernel and one element long whatever the dispatch's element count.
  `T` is a struct of the same `#[gpu]` module (GUST-E0206), declared `#[repr(C, align(16))]` by the macro with const
  assertions of its `Std140V1` offsets and size. A struct a uniform reaches is not also a storage element
  (GUST-E0207), holds no `Mat2` or `bool` (GUST-E0205), and is read one field at a time, never as a whole value
  (GUST-E0321). The twin runs the kernel with a copy of the value.
  Its ABI is `Std140V1` (`std140-v1` in the generated host text): scalars, 2- and 4-wide vectors, 4x4 matrices and
  structs of them rounded up to 16 bytes. `T` must also be padding-free, because the upload copies the bytes of the
  Rust value and padding is uninitialised: `create_uniform` and `CpuDevice::uniform` refuse a padded `T` with a message
  that names the fix, explicit padding fields, and the twin also refuses a uniform parameter whose rule is not
  `Std140V1`. The macro refuses at compile time a struct whose constant-buffer layout has a gap before a field or a
  tail (GUST-E0209), naming the explicit padding fields to write. A value is replaced by
  `HeadlessDevice::write_uniform` (`Queue::write_buffer`: after every earlier submission, before the next, so all
  dispatches of one later submission see the last value) or, ordered among the dispatches of one submission, by the
  copy node `StagedGraph::write_uniform` (a write of the uniform for the derived edges and the hazard check).
- **Not executed by the twin.** Indirect dispatch and staged graphs stay GPU-only. A kernel can opt out explicitly with
  `#[kernel(cpu = "unsupported")]` (D26). The examples that use them carry kernel-level twin evidence only (exists, m3.e2): `staged-graph` and
  `component-pool` run the twin of each of their kernels on the GPU's inputs to that kernel (read back where an earlier kernel produced them), and
  label each such test "kernel-level twin evidence; graph and indirect dispatch GPU-only". The graph, its derived
  dependency edges, the indirect launch that consumes the arguments and the pool growth are executed on the GPU alone,
  and no twin test covers them. The twin of `prepare_dispatch` does compute the arguments, and the test compares them
  with the GPU's.
- **Twin coverage of the examples (exists, m3.c to m3.e2).** Every kernel with a typed call in these examples runs on
  the twin next to the GPU and a host reference; none uses the opt-out. Integer, struct and copied outputs are exact;
  float outputs use the bound derived in section 4. The twin equals the host reference bit for bit in each of them.
  - `vector-add` (`add`), `polynomial` (`evaluate`): m3.c.
  - `signal-pipeline` (`transform`): derived bound.
  - `particle-step` (`scalar_order`, `field_order`, `snapshot`: exact; `step`: derived bounds).
  - `typed-pipeline` (`calibrate`, `classify`, `finalize`): derived bounds, each kernel compared on the inputs the GPU
    produced for the stage before; `severity` is exact after a stability check.
  - `atomic-counter` (`increment`): exact.
  - `staged-graph` (`transform`, `summarize`) and `component-pool` (`prepare_dispatch`: exact; `integrate`: derived
    bounds for `x`, `y` and `vy`, exact `vx`, `steps` and untouched particles), kernel level only (m3.e2). `summarize` is
    compared on the intermediate the GPU produced (`peak`, `above_threshold` and `count` exact, `total` derived);
    `integrate` on the allocation, the active count and the indirect arguments the GPU frame started from and
    produced. All four kernels have a twin, and none uses the opt-out.
- **Language subset.** 32-bit `f32`, `i32`, `u32`, `bool` and module structs of them (README, "Current subset"). Float
  arithmetic is `+ - * /`, comparisons and casts: the validator accepts no other method than `.len()`,
  `.global_id()` and the `Option` / `Result` helpers, so there is no `sqrt`, `fma` or trigonometry to bound.
  The validator accepts `%` on floats (it rejects only an integer-literal zero as the right operand of `/`, `%`, `/=` and `%=`),
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
one workspace profile, so the macro, the function `gust::cpu::overflow_checks()` and the twin agree there. The
cross-crate evidence is in `tests/dual`, run by `check-dual`: `overflow_mode_is_the_dependents_own` (m3.f1) shows that
a dependent with a per-package override sees its own mode through the macro, and the two `twin_overflow` tests show
what the twins do with `u32::MAX + 1`: the twin of the dependency's kernel follows the dependency's mode (a panic in
the dev profile, where only the dependent has checks off), the twin of the dependent's own kernel follows the
dependent's. Both outcomes are asserted per mode, in the dev profile; in the release profile every package wraps.

## 3. The defined domain, and what lies outside it

D22 said integers are bit-exact across targets. D38 narrows that: the guarantee holds on the defined domain below
and nowhere else. Outside it the targets diverge, no diagnostic can prevent it because the operands are run-time
values, and GUST records both outcomes in the ledger (section 7) instead of removing the divergence.

D22 also said that entries of the known-mismatch file fail when an outcome changes. D39 narrows that for outcomes the
WGSL specification leaves open (NaN and infinity, subnormals, NaN conversions, out-of-bounds access), float edge cases
included: they are recorded as one-adapter observations and never fail a test. Outcomes the specification fixes, and
the plain-Rust outcomes, are still pinned and fail the test when they change.

Inside the domain the GPU and the twin give the same bits for integers and `bool` values and take the same
control-flow path, with two exceptions that the domain does not remove. A decision derived from a float value (a
float-to-integer cast, a comparison or a branch on a float, even with every operand finite, normal and inside the
float bounds of section 4, as in the threshold guard at the end of that section) is exact only while no value lies
within its bound of the decision point, because the float results themselves differ by up to that bound, and the
integers and booleans computed from such a decision inherit the difference. Contended atomics are the second
exception: the twin runs the invocations in ascending order and yields one member of the set that the GPU may give
(section 6). The final value of a cell that receives only `atomic_add`, only `atomic_min` or only `atomic_max`
does not depend on the order and is identical; a cell that receives different kinds of operation is order-dependent.
The outcome of a contended `atomic_exchange` and of any `atomic_compare_exchange`, contended or not, is set-valued
on the GPU (it may fail spuriously, section 6; KM-09). The float rules of section 4 apply. The domain is:

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

The production CPU target (`GUST_PROFILE=cpu`, D51) reads the lines above like this. Overflow and shifts follow the
`overflow-checks` of the crate that defines the kernel (D43): in the default release profile they wrap and mask as
the GPU does, and in a checked build they panic. Division and remainder by zero and `MIN / -1` panic in every
profile. Float-to-integer casts saturate, as in Rust. An out-of-bounds access panics. These differences stay
documented limitations (D39); GUST does not try to match them.

What users should do: keep run-time operands inside the domain by construction, or guard them in the kernel (test a
divisor before dividing, clamp before casting). A literal zero divisor is rejected at the Rust boundary with a
diagnostic; a run-time zero is not. There are no explicit `wrapping_*` operations in the dialect; they may become a
language feature of their own, not a way to close this gap. Do not rely on the twin to detect overflow on the GPU
path: the GPU wraps silently whatever the twin does.

**Limitation: arithmetic inside `gust` follows `gust`'s own overflow mode.** Rust fixes overflow behaviour when a
crate is compiled. An integer operation that the shadow body writes is compiled in the crate that defines the kernel
and follows that crate's mode (`gust::cpu::overflow_checks!()` written there; a dependent that runs the twin of a
dependency's kernel gets the dependency's mode, not its own). An integer operation that `gust` itself implements (a shadow
function or type provided by the library, as opposed to the kernel's own `+`) is compiled in `gust` and follows
`gust`'s mode (the function `gust::cpu::overflow_checks()`). The two modes differ only when a profile overrides
`overflow-checks` for one of the two packages. Apart from the vector operations below, the only such operation is
`atomic_add`. It never panics on overflow, not even in a checked twin, unlike the kernel's own `+`: it wraps modulo
2^32 in every build, exactly as the GPU does (the other atomic functions cannot overflow). So the case does not arise
for atomics. This is documented, not fixed (D39).

**Limitation: vector operations follow two overflow modes.** The integer vector operations of `gust::math` (the
lane-wise arithmetic operators and `dot` on `u32` and `i32` lanes) are `#[inline]` functions in `gust`, so each crate
that uses one compiles its own copy. Measured in incremental dev and test builds (m6.a2): such an operation panics on
overflow only when both `gust` and the crate that compiles the copy have `overflow-checks` on; otherwise it wraps
modulo 2^32, as the GPU does. With one workspace profile the two modes agree, and a vector operation behaves like the
scalar operation beside it. Optimized non-incremental builds may inline code across crates differently, so a profile
that overrides `overflow-checks` for one of the two packages can see either mode there. This is documented, not fixed
(D39).

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

**Deriving a tolerance for a kernel (done for polynomial in m3.c and for the float kernels of the other examples in m3.e1 and m3.e2,
see "Derived kernels" below).** An
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

**Derived kernels (slices m3.e1 and m3.e2).** The same method, per kernel, with the per-operation numbers of the table above (a
rounding of at most `u` times the result's magnitude on the twin, `2u` on the GPU for `+ - *`, `u` and `5u` for `/`
with the divisor in `[2^-126, 2^126]`), the GPU side without fusion, first-order propagation, and each input's range
asserted by a test of the example (the ranges that the later stages of typed-pipeline and staged-graph receive from
the GPU, in their ignored tests, and the state that a component-pool frame starts from). The asserted bound is the derived sum rounded up to a whole `u` plus
1u for the second-order terms. The step-by-step tables are in the doc comment of each example's bound constant. The
last column is an observation on the RTX 5090 (debug profile, 2026-11-01), printed by the
ignored tests (`--nocapture`) and kept in the m3.e1 and m3.e2 slice logs; it is not a tolerance and no test uses it.

| Kernel (example) | Output | Derived | Asserted, absolute | Observed twin versus GPU |
| --- | --- | --- | --- | --- |
| `evaluate` (polynomial, slice m3.c) | `out` | 46u | 48u = 2.86e-6 | 2.38e-7 |
| `transform` (signal-pipeline) | `out` | 898.46u | 900u = 5.36e-5 | 7.63e-6 |
| `step` (particle-step) | `velocity.x` | 1.79u | 3u = 1.79e-7 | 0 |
| `step` (particle-step) | `velocity.y` | 6.57u | 8u = 4.77e-7 | 0 |
| `step` (particle-step) | `position.x` | 96.08u | 98u = 5.84e-6 | 1.49e-8 |
| `step` (particle-step) | `position.y` | 6.30u | 8u = 4.77e-7 | 0 |
| `step` (particle-step) | `energy` | 85.75u | 87u = 5.19e-6 | 4.77e-7 |
| `calibrate` (typed-pipeline) | `temperature` | 170.16u | 172u = 1.03e-5 | 1.91e-6 |
| `calibrate` (typed-pipeline) | `humidity` | 5.13u | 7u = 4.17e-7 | 5.96e-8 |
| `calibrate` (typed-pipeline) | `quality` | 8.03u | 10u = 5.96e-7 | 5.96e-8 |
| `classify` (typed-pipeline) | `score` | 60.96u | 63u = 3.76e-6 | 0 |
| `finalize` (typed-pipeline) | `adjusted_score` | 12.78u | 14u = 8.34e-7 | 0 |
| `transform` (staged-graph) | `intermediate` | 192u | 193u = 1.15e-5 | 1.19e-7 |
| `summarize` (staged-graph) | `total` | 3885u | 3886u = 2.32e-4 | 0 |
| `integrate` (component-pool) | `vy` | 18u | 19u = 1.13e-6 | 2.98e-8 |
| `integrate` (component-pool) | `x` | 52.5u | 54u = 3.22e-6 | 3.73e-9 |
| `integrate` (component-pool) | `y` | 72u | 73u = 4.35e-6 | 1.86e-9 |

Two properties of these kernels are worth stating. A chain of products has the same relative error in any grouping, and
every sum in them has two operands, so reassociation (15.7.5) cannot raise a bound; fusing a product into a sum only
removes a rounding. The one exception is `total` of staged-graph's `summarize`, which adds eight values: its bound is
derived over the left-deep grouping, the one with the largest partial sums, and the first addition (`0.0 + value`) is
exact. The kernels of the two graph examples are compared on identical inputs only (the twin is fed the GPU's
inputs to each kernel), so a bound covers one kernel's own roundings and not the rounding of the stage before. In the
staged-graph test the settings with exactly representable scale and offset give no rounding at all (observed 0); the
third set (scale 0.1, offset 0.3) is the one that rounds. A decision on a float (`severity` in `classify`, which compares `score` with 0.8 and 2.4) is
exact only while no value lies within the bound of its threshold; the typed-pipeline test checks that on the data it
uses, and a different input set could make the targets disagree on `severity`, as the ledger's float rows allow.

**Vectors and matrices (slice m6.g1).** The component-wise operators, `dot`, `mul_vec4` and the matrix products are
the scalar operations above applied lane by lane, so no new per-operation constant exists: a component-wise `+ - *`
is one rounding per lane (twin `u`, GPU `2u`, times the result's magnitude), `/` is `u` and `5u`, and negation,
`transpose`, swizzles, constructors, `from_cols` and `from_rows` move bits and are exact. A dot product and one
element of a matrix product (`m * v`, `m * m`, `mul_vec4`, `transpose(m) * v`) are sums of N products and take the
derivation of the doc comment of `crates/gust-wgpu/tests/vectors.rs`, over the worst grouping on the GPU side. The
test compares the GPU, the twin and an independent `f64` host reference, with integers exact.

| Operation (operands of magnitude at most 1 unless stated) | Derived twin | Derived GPU | Asserted twin, GPU | Observed twin versus GPU |
| --- | --- | --- | --- | --- |
| `a + b`, `a - b` per lane, `abs(r)` at most 2.5 | 2.5u | 5u | 2.5u, 5u | 0 |
| `a * b`, `a * k`, `k * a` per lane, `abs(r)` at most 1.5 | 1.5u | 3u | 1.5u, 3u | 0 |
| `a / b`, `a / k` per lane, `abs(r)` at most 2 | 2u | 10u | 2u, 10u | 2u |
| `dot`, `m * v`, `mul_vec4`, one element of `m * m`, N = 4 | 13u | 26u | 14u, 27u | 4u |
| the same with N = 3 (`Mat3`) | 8u | 16u | 9u, 17u | 4u |
| the same with N = 2 (`Mat2`, `Vec2` dot) | 4u | 8u | 5u, 9u | 2u |
| `-v`, `-m`: exact except the sign of a zero lane on WGSL (§1, D39); `transpose`, swizzles, `from_cols`, `from_rows`, integer vector operators | 0 | 0 | exact | 0 |

The last column is an observation on the RTX 5090 (debug profile, 2026-11-02), printed by `vectors` with
`--nocapture`; it is not a tolerance. A matrix product through a struct field and the `Vec4`/`Mat4` fields of a
`#[repr(C)]` struct (112 bytes, no padding) use the same bounds.

**Negation and the sign of zero (D39).** A module that names a matrix lowers `-m` to `m * -1.0` (section 1), which is the
twin's sign flip under IEEE arithmetic; `-v` is slangc's `vec(0) - v` on WGSL, which by IEEE would turn a `+0` lane into
`+0`. The test `matrix_negation_keeps_signed_zeros_and_subnormals_on_gpu` negates `Mat2`, `Mat4`, a `Mat3` column and a
`Vec4` whose lanes are zeros of both signs, subnormals of both signs, the smallest normal and ordinary values, and
compares bit for bit. Measured on the RTX 5090 (Vulkan, driver 610.88, slangc 2026.20, 2026-11-02, debug and release):
every `-m` lane equals the twin's sign flip, subnormals kept, and `-v` of `+0` gave `-0` and of `-0` gave `+0` (115
lanes), so this device shows no difference in either; another device or driver may.

**Limitation (D39): constant `Mat2::IDENTITY` and `Mat3::IDENTITY` on the SPIR-V pipeline.** On slangc 2026.20 a
kernel that multiplies by a compile-time constant `float2x2` or `float3x3` (what `Mat2::IDENTITY` and
`Mat3::IDENTITY` lower to) emits SPIR-V that crashes the process (segmentation fault, exit 139 in a probe harness
on the RTX 5090) when wgpu ingests it through naga's SPIR-V front end. The same source is correct as WGSL, correct
through SPIR-V passthrough, correct with a runtime `float3x3`, and correct for a constant `float4x4`. The executed
route of `gust-wgpu` is WGSL, which is why the `vectors` test uses the constants. The exported SPIR-V of that module
is valid (`spirv-val`, vulkan1.2, checked by hand for the six kernels of `vectors`) but must not be dispatched through
the SPIR-V ingestion path. This is a difference of toolchain behavior, documented and not chased.

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
- **What the twin does not model (later constructs).** The twin finishes one invocation before it starts the next, so
  it has no notion of a construct whose meaning spans invocations: workgroup barriers, subgroup operations and
  deliberate races (roadmap M7 plans barriers and shared memory, M8 subgroup operations). Before such a construct is accepted into the dialect, it needs a twin
  strategy written into this contract (what the twin does, and where it may differ from the GPU), or the kernels that
  use it must carry `#[kernel(cpu = "unsupported")]`, which leaves them without a twin. Running such a kernel
  one invocation at a time would give a clean result that says nothing about the GPU.
- **Visibility.** The writes of a dispatch are visible after the dispatch completes. Within a dispatch, an invocation
  must not depend on another invocation's non-atomic writes.
- **Production CPU target (D51).** Under `GUST_PROFILE=cpu` the kernels run on `gust::cpu::CpuDevice` in every build,
  one thread per dispatch. A racy kernel is still outside the contract: it gives one deterministic result that proves
  nothing about the GPU. The launch rule below holds as written, and the limits of the GPU are not emulated.
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
  is also a spec-allowed result. The contract (D47): on the GPU the success case is set-valued, the cell ends at
  the new value or stays unchanged (`{7, 11}` for 7 to 11, `{-5, 11}` for -5 to 11); on the twin it is exact (11).
  The failure case (compare against a value the cell does not hold) is exact on both, because a spurious failure
  only adds failures. The returned flag is never part of the contract: the GPU form is statement-only.
- **Tested (GPU measured, twin exact).** `crates/gust-wgpu/tests/atomics.rs` checks a contended `atomic_add` sum, the
  contended minimum and maximum, the compare-exchange failure case, and signed add and exchange exactly, on the GPU
  and on the twin, and the uncontended compare-exchange success cases (unsigned and signed) as a set on the GPU and
  exactly on the twin. The twin's final cell is compared directly with the GPU's in the contended add, min and max
  tests and the failure case, and not in the contended exchange and compare-exchange or the two success cases. In
  the signed sequence (add, exchange, compare-exchange on one cell) the add and exchange steps are pinned to the same
  literals (-4, -5) on both sides without comparing the sides, and the compare-exchange step is the set `{-5, 11}`
  on the GPU and 11 on the twin. For the contended
  exchange the GPU must end in 0 to 256 and for the contended compare-exchange in 1 to 257 or at the unchanged 999
  (the weak compare-exchange may fail every attempt); the twin is asserted at 256 and at 1 respectively, and never
  compared with the GPU there. The returned old value of an atomic is never observable in GPU code (statement-only);
  the twin returns it and `runtime.rs` unit-tests it.
- **Weak compare-exchange on the GPU (D47).** The compare-exchange success cases (a single invocation, 7 to 11 and
  -5 to 11) are asserted as the set `{7, 11}` and `{-5, 11}` on the GPU, because WGSL allows a spurious failure
  there too (17.8.5); on the twin they are asserted at 11, by definition. The RTX 5090 took the exchange branch in
  every measured run, so the spurious branch is exercised only by the GPU-free test of the two check functions that
  the GPU tests call, never by an adapter.
  Because the unchanged value is accepted in every compare-exchange case, a GPU compare-exchange that did nothing
  would pass all of the GPU assertions. The twin assertion catches a no-op twin; the failure case (exact, at 7, on
  both sides) catches a compare-exchange that always swaps, but cannot tell a no-op GPU from a correct one. A GPU
  compare-exchange that does nothing is therefore not detected by these tests; that is accepted for the same
  reason the set is (KM-09). At the emission level, the unit test `atomic_compare_exchange_on_rw_u32_is_accepted`
  in `gust-macros` checks that the call lowers to `.compareExchange()`; the `atomics` Slang golden covers the
  `atomic_add` kernel only and does not guard the compare-exchange.

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
| KM-09 | Contended `atomic_exchange` and `atomic_compare_exchange` (257 invocations, one cell), and the uncontended `atomic_compare_exchange` (one invocation) | Sequential, ascending invocation id: the exchange ends at 256; the compare-exchange from 999 to `id + 1` ends at 1. | Set-valued: the exchange ends at some id in 0 to 256, the compare-exchange at some value in 1 to 257, or at 999 (unchanged) because the weak compare-exchange may fail spuriously (WGSL section 17.8.5); the test accepts all of that set. The uncontended case (one invocation, `compare-exchange(cell, 999, id + 1)`) is set-valued too: the cell ends at 1 or stays 999 (WGSL section 17.8.5, D47), and the Rust and twin columns end at 1. Observed over 32 dispatches per run, three runs: 1 to 10 distinct final values per run, and, in the contended cases, never the sequential result (the exchange ended between 1 and 251, the compare-exchange between 7 and 137). The uncontended case ended at 1, the sequential result, in every dispatch, in both overflow modes. | spec-allowed (the set), measured |
| KM-10 | Overflow of `+`, `-`, `*` and unary `-` on `u32` and `i32` | Checks on: panics ("attempt to add with overflow" and so on). Checks off: wraps modulo 2^32. | Wraps modulo 2^32 (WGSL section 6.2.3): `u32::MAX + 1 == 0`, `i32::MAX + 1 == i32::MIN`, `0u32 - 1 == u32::MAX`, `-i32::MIN == i32::MIN`. | spec-allowed, measured |
| KM-11 | Out-of-bounds index equal to the buffer length | Panics "index out of bounds" in every profile. | Unspecified (a dynamic error, WGSL section 6.5.7). Observed with index 4: a load from a 4-element buffer returned 0, and a store into a 4-element buffer was dropped (the buffer was unchanged); the store is measured alone, with a longer `source` so that the kernel's own load is in bounds, and the twin's store row panics at the store because its load is in bounds. | unspecified, measured |

Reading the table: the Rust column and every GPU value that the specification fixes (labelled spec-allowed) are
enforced, so the test fails when one changes and the row is edited in the same change. A value marked unspecified is
an observation: the test prints a NOTE when it changes and never fails (D39). Adding a construct to the ledger means
adding
its `KM-nn` row here and its case in `crates/gust-wgpu/tests/mismatch.rs`; the id guard fails if only one is done.

Two kinds of adapter run the ledger, told apart by `device_type`: hardware, and software (`Cpu`, Mesa lavapipe in the
`gpu-software` job). The kind never changes what is enforced: the spec-allowed GPU values, the Rust column, the twin
column and every `DOMAIN` control hold on both, and a deviation of one of them on lavapipe is a red test, never a
second observation. Only the observations of the unenforced rows (KM-07, KM-08, KM-11 and the NaN casts of KM-05 and
KM-06) are kept per kind. The GPU column above is the hardware observation (RTX 5090, NVIDIA 610.88, Vulkan). The
software column is a separate pin, set only where lavapipe differs from it, and the test prints a NOTE, with the
driver, backend and `driver_info` of the adapter, when the running kind's own pin changes. A row without a software
pin (the hardware pin accepted m5.b0's value) is then compared only with the hardware pin, the NOTE says "no software
pin (the hardware pin accepted m5.b0's value)" for a difference from it, and nothing fails. The
software pins of the NaN rows record m5.b0's exact payload bits, so a lavapipe payload change prints a NOTE, while the
hardware pin accepts any NaN; neither ever fails, because the payload is not part of the contract. Measured on
lavapipe (llvmpipe, LLVM 20.1.2, Mesa 25.2.8, debug and release alike, 18 rows): KM-07 gives `0x7fc00000` (the operand
NaN, `f32::NAN`, propagating) from a NaN operand and `0xffc00000` (the x86 default NaN) from `inf * 0.0`,
`inf - inf`, `inf / inf` and `0.0 / 0.0`, where the RTX 5090 gives `0x7fffffff`; `3.0e38 / 3.0e38` gives 1
as Rust does, where the RTX 5090 gives 0; and KM-08 keeps denormals as Rust does (`1.0e-40 * 1.0` is `0x000116c2`,
that is `1.0e-40`, and `1.0e-40 == 0.0` is false), where the RTX 5090 flushes them. KM-09 keeps its set rule on both
kinds and gets no per-kind value: its observations vary between runs. Both are unspecified, measured observations of
one driver version each, not requirements, and the Rust and twin columns are unchanged.

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
- **Exact example kernels** (m3.c to m3.e2), each asserted equal on the GPU, on the twin and against a host reference:
  `vector-add` `add` on inputs whose sums are exact in `f32` (`twin_matches_reference_and_gpu`, counts 4, 65 and
  200); `atomic-counter` `increment`, a contended `atomic_add` of 1 whose sum does not depend on the order
  (`twin_matches_reference_and_gpu`, 1, 64, 257 and 1000 invocations); `particle-step` `scalar_order`, `field_order`
  (sums of multiples of 1/16, exact in `f32`) and `snapshot`, whose GPU output buffers are seeded with zeroed
  particles so that a kernel that wrote nothing or copied part of a struct would fail
  (`twin_matches_reference_and_gpu_for_the_exact_kernels`, counts 1, 63, 64, 65 and 257); `component-pool`
  `prepare_dispatch` (integer-only: the active count and the three indirect arguments equal on the GPU, on the twin
  and against a host formula over ten frames that clamp to the budget and to the live count,
  `twin_matches_reference_and_gpu_within_the_derived_bounds`; the clamp to the allocation has a twin-versus-host test
  only, `twin_frames_match_host_reference`). These statements are kernel level: the graph and the indirect launch
  that consume the arguments are GPU-only.
- **Scalar coretests** (`tests/fixtures/coretests.rs`, slice m5.d1): the scalar integer, `bool` and float cases
  transcribed from the Rust core tests, one per row in `docs/development/CORETESTS.md`, which also holds the pin, the
  exclusions and the acceptance of each row. Slice m5.d2 runs the same kernels on the GPU
  (`crates/gust-wgpu/tests/coretests.rs`, part of `check-feature gpu-semantics`): all 84 cases equal their literal on
  the CPU twin and on the GPU, on the RTX 5090 and on Mesa lavapipe (llvmpipe). A row is a statement of this section
  under the acceptance that page gives: exact integers, the 2.5 ULP set of float `/`, and the two-neighbour set of
  `i32 as f32` beyond +/-2^24.

Everything else on this page stays **spec-allowed**, **unspecified**, **measured** or **planned**. The floats of
section 4 remain derived and measured on the examples listed under "Derived kernels" (polynomial, signal-pipeline,
particle-step, typed-pipeline, staged-graph, component-pool) only. Those bounds are spec-allowed arithmetic checked by one GPU run, not a proof of
the GPU's rounding; the floats are not **proven**.
