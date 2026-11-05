# Differential program generator (`tests/difftest`)

Status: slices m5.c1a (the integer core), m5.c1a2 (floats v1, the exact-interval subset of R1-4), m5.c1b (the batch
runner `difftest-run`), m5.c1c (the shrinker) and m6.d (vectors and matrices, DIFFTEST_FORMAT 6). The crate generates
integer, boolean and `f32` kernels from a seed, and (since format 6) kernels that compute with the `gust::math` vectors
and matrices as locals and intermediates (buffers and results stay scalar), evaluates them with an independent
interpreter that tracks the defined domain (and, for floats, an interval of every value a conforming WGSL implementation
may hold), writes them as a `#[gust::gpu]` crate, and `difftest-run` builds a batch of them, runs every program on the
interpreter, the CPU twin and the GPU, and reports what disagreed. The crate itself is std only and has no GPU code: the
GPU code lives in the generated run crate, which depends on `gust-compute` and `gust-compute-wgpu` (library names `gust`
and `gust_wgpu`). `difftest-run --shrink` reduces one failing program (local use only; see Shrinking). A unit test is
evidence about the generator, the interpreter, the emitter and the classifier only; a GPU claim needs a `report.json` of
a `gpu` run, whose `mode`, `adapter` and `programs_executed` say what ran.

`tests/difftest` is a workspace of its own (own `Cargo.lock`, std only, `publish = false`), like `tests/ui` and
`tests/dual`; the root workspace does not contain it. Run it from that directory with the repository's usual cargo
setup:

```sh
cd tests/difftest
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked            # unit tests, the emitter round trips and the CPU short-circuit test (needs cargo and the cached registry, no GPU)
cargo test --locked -- --ignored   # the GPU short-circuit test (Gpu tier: Vulkan adapter, slangc, native reflection helper)
cargo run --locked --bin difftest-gen -- show 1 0     # the Rust text of program 0 of seed 1
cargo run --locked --bin difftest-run -- --seeds 1-2 --report report.json   # a measured batch on the GPU
cargo run --locked --bin difftest-run -- --seeds 1-4 --cpu-only             # interpreter against twin, no GPU
cargo run --locked --bin difftest-run -- --shrink 1:8 --report shrink.json   # local only: CI never calls it
```

## Program space

Types `u32`, `i32`, `bool`, and (since format 2) `f32`. One to three read-only input buffers of `u32`, `i32` or `f32`
(never `bool`) and one output buffer; the output element type is the result type, and a `bool` result is stored as
`u32` 0 or 1 (there is no `GpuPod` for `bool`, so no `bool` buffer). The kernel is `let i = id.x; if i < out.len() { <statements>; out[i] = <result>; }` over 1, 63, 64,
65, 257 or a random count below 4096 elements, workgroup size 64.

- Statements: `let` and `let mut` with an explicit type, assignment, `if`/`else`, and `for v in 0u32..N` with `N` a
  literal in 1 to 8. Statements nest two deep at most, so loops nest at most twice and a statement can run up to 64
  times per element.
- Expressions, depth at most 6: literals biased to boundaries (0, 1, -1, `i32::MIN`, `i32::MAX`, `u32::MAX`, powers
  of two, masks), locals, `i`, `a[i]` and other in-bounds reads (`a[(e) % a.len()]`, `a[a.len() - 1 - i]`),
  `+ - * / % & | ^ << >>`, unary `-` (`i32` only) and `!`, comparisons, `&&`, `||`, and `as` between `u32` and `i32`.
  The right operand of a shift is always `u32`. `bool as u32` is not generated (ARCHITECTURE.md: it is unproven).
- A divisor is never a literal zero, in any branch: the `#[gpu]` validator rejects `DivisionByLiteralZero` even where
  no input reaches (a test scans 20000 decoded programs for one). An all-ones divisor (`-1` for `i32`) is biased in,
  and then the dividend is often a plain read of an `i32` input, so that `i32::MIN / -1` is reached; seed 1 of the
  corpus discards one program for it (at format 2 it was seed 2, twice). Over seeds 1 to 300 the discard fired in 23
  seeds at format 2, about one seed in thirteen.
- Floats (the subset below): `f32` literals, locals and inputs; `+ - * /`, unary `-`, comparisons, `as` between `f32`
  and `i32`/`u32` under the conditions of the float rules, loop accumulators (`let mut acc`, then `acc = acc +/- term`
  first in a loop body of at most 8 trips per loop, but with the nesting above a loop-carried chain runs up to 64
  times, 65 terms; see the chain rule below), and at most 4 written terms in a `+`/`-` chain outside loops. Never float
  `%`, never a `bool` buffer, never an input outside the bounded set. An integer-only decode is kept as
  `Space::Integer`: it reads the same bytes as format 1 and decodes to the m5.c1a programs, and a test pins the m5.c1a
  discard counts of seeds 1 to 8 and the hashes of its 128 programs (emitted text and input data, with the format stamp read as 1), which equal those
of the m5.c1a seeds.txt: formats 2 and 3 change the stamp (and, in the float space, the programs), not the integer programs.
- No structs, helpers, `Option` or atomics.

### Vectors and matrices (format 6, slice m6.d)

A batch is 16 programs per seed: the first ten in-domain programs of the float space (the programs of format 5, in the
same order and byte for byte) and six of the vector space (`Space::Vector`, attempt numbers from
`VECTOR_ATTEMPT_BASE = 1_000_000`; the attempt number alone selects the space, so the reducer and the shrinker
re-decode a vector program without being told). The vector space is the float space plus these, as locals and
intermediates only; an input buffer, the output and `out_ty` stay scalar (`Program::check` refuses a vector result or
buffer), so the result file, the classifier and the shrinker see scalars.

- Types: `Vec2`/`Vec4`, `IVec2`/`IVec4`, `UVec2`/`UVec4` and `Mat2`/`Mat4` (square, column-major, `Mat4 { cols: [Vec4;
  4] }` layout). The 3-wide types are not generated: they have no buffer layout, and the 2- and 4-wide types are the
  ones the GPU runner can read back. A mutable vector local may be assigned a component (`v.x = e;`).
- Constructors: `Vec4::new(a, b, c, d)`, `Vec4::splat(s)`, `Mat4::from_cols(c0, c1, c2, c3)` and `Mat4::from_rows(..)`.
  Operators: the component-wise `+ - * /` of two vectors of one type, a vector with a scalar (`v * s`, `s * v`, `v /
  s`), unary `-` on `Vec` and `IVec` and on matrices, matrix `+ - *` and `m * s`, `s * m`, `m * v`, `m * m` (the `Mul`
  impls of `crates/gust/src/math.rs`; `Div` for integer vectors is the component-wise `/`). Reads back: `.x .y .z .w`,
  the swizzles `.xy()`, `.xyxy()`, `.wzyx()` (2 or 4 letters, from either width), `a.dot(b)`, `m.transpose()`. These
  have their own AST nodes (`Make`, `Splat`, `Comp`, `Swizzle`, `Dot`, `Transpose`, `AggBin`, `AggNeg`,
  `Stmt::AssignComp`), so every pattern over `Bin` and `Un` still sees scalars only; `ast::operator_counts` and the
  interpreter's coverage name them alike (`new Vec4`, `splat Vec4`, `comp Vec4`, `swizzle Vec4->2`, `dot Vec4`, `* Mat4
  Vec4`, `-Vec4`, `from_cols Mat4`, `transpose Mat2`, `assign comp`, ...).
- A vector or matrix local is initialised at depth 3 (a scalar at depth 5): their products multiply the bounds of the
  operands. A top-level vector local is folded into the result through a component (or, for `f32`, `dot(v, v)`), a
  matrix through a component of its product with a fresh vector, so that the values are live. A vector-space program
  without any vector or matrix operation is discarded as `no-aggregate` (its own class in the report) because it tests
  nothing new. That test runs before the domain filter: a vector-space attempt with no aggregate is counted as
  `no-aggregate` even when it would also have left the domain, so its domain label is not counted.
- **`Mat2` is never a compile-time constant.** SEMANTICS section 4: a constant `Mat2` or `Mat3` crashes on the SPIR-V
  route, layer not isolated (the harness runs the WGSL path, which does not, but the generator must hold on every path).
  `Program::check` refuses a `Mat2` constructor with no direct `f32` input read in its arguments (an `f32` `Input` node
  with no cast above it), and the generator puts one in (`vec2_from_input`) whenever the columns it drew have none. A
  constant `Mat4` is allowed. The guard is syntactic ("a direct `f32` `Input` read below the constructor") because the
  question that matters is whether a front end can fold the expression: a read behind a cast, such as `(in1[i] * 0u32)
  as f32`, can fold to a constant and does not count (a test pins that shape as refused).
- The reducer has no aggregate pass: pass 2 (`observed`) replaces scalar subexpressions only and skips vector
  and matrix values, so a shrunk vector program keeps its constructors and operators. The statement,
  trip-count, element-count, literal and byte-stream passes apply to it as to any program.

## The domain filter

The interpreter (`interp.rs`) evaluates the AST with checked operations, in Rust's evaluation order and with
short-circuiting `&&`/`||`. It reports the first of these, which are the integer conditions of D38 and
[SEMANTICS](../SEMANTICS.md) section 3 (items 1 to 3 and 6):

| Label | Condition |
| --- | --- |
| `overflow` | `+ - *` or unary `-` whose result does not fit `u32` or `i32` |
| `division-by-zero` | `/` or `%` with a zero divisor |
| `min-over-minus-one` | `i32::MIN / -1` or `i32::MIN % -1` |
| `shift-amount` | a shift by 32 or more |
| `index` | a buffer index not below `len()` |

A program is kept only if every element of its input stays in the domain at every evaluated sub-expression; otherwise
the whole program is discarded and the reason is counted. Besides these dynamic labels there is a static one,
`static` (`interp::static_violation`): any closed sub-expression made of literals only that leaves the domain (a
literal shift amount of 32 or more, a constant overflow, `MIN / -1`, a closed zero divisor) discards the program
even in dead code, in an `if false` branch or the unevaluated side of `&&`; so does a literal zero divisor or a literal
shift of 32 or more with a non-constant other operand. The reason is the shader compilers: slangc or WGSL
const-expression rules may refuse such a constant at shader-creation time although rustc, with the crate-level lint
`allow`, accepts it. Closed sub-expressions are folded eagerly, without short-circuiting, with the interpreter's own
`binary` and `unary`.

There is no rule for a divisor that slangc folds to zero without being a literal (the first iteration of a loop, `i * 0`,
`i - i`, a branch that a known condition takes). Format 4 (slice m5.c2) discarded those programs as `static`, because
slangc 2026.20 refused them with error E30002, "divide by zero", even in a branch that no element takes; since format 5
(slice m6.dv, D74) the macro lowers every `/` and `%` whose divisor is not a literal to a call of `__gust_div` or
`__gust_rem`, which slangc compiles in every shape the walk of format 4 modelled, so the programs are kept and the walk
(`first_iteration_zero_divisor`) and its two unit tests are gone. A literal zero divisor stays a plain operator and a
discard, as above. No discard remains for slangc folding a divisor, so there is no `compiler-limitation` class; every
`static` discard is a closed-constant rule. If a later Slang makes a helper call foldable again, `check-difftest` reports
the program as a `gpu-error` mismatch naming the seed, and the Slangc-tier shape table of
`crates/gust/tests/diagnostics.rs` fails first.

The vector operators inherit the domain component by component (`agg.rs`): the interpreter does each component with the
scalar `binary`, `unary` and `float_binary`, in the order the CPU twin performs them, so the first violation is that of
the lowest component and a vector is outside the domain as soon as one component is (`UVec2::new(1, u32::MAX) +
UVec2::splat(1)` is an `overflow`, `IVec2::new(4, MIN) / IVec2::new(2, -1)` a `min-over-minus-one`). `a.dot(b)` is `p0 +
p1 + ...` with `pk = a_k * b_k`, summed left to right (`Vec4::dot`), `(m * v)[i]` is `c0[i] * v[0] + c1[i] * v[1] + ...`
(`Mat4::mul_vec4`) and `(a * b).cols[j]` is `a * b.cols[j]` (`Mat4::mul_mat4`); an integer `dot` therefore overflows
where one of its products or partial sums does. `static_violation` folds a closed vector or matrix expression with the
same arithmetic (a closed `UVec2::splat(u32::MAX) + UVec2::splat(1)` in dead code is `static`), and a closed zero
component in the divisor of a vector quotient is `static` even when the dividend is not closed, as for a scalar.

Inside the domain the three parties (interpreter, CPU twin,
GPU) must agree bit for bit, so a program outside it is never asserted, and nothing here writes a ledger row (D39). A
program that reads no input buffer is discarded as `no-input-read` (it would be a constant).

## Float rules (v1, the exact-interval subset)

The integer filter is exact: inside the domain all three parties agree bit for bit. Floats are not, so the oracle for
a float program is an interval, not a value (R1-4 of the M5 plan). `float.rs` holds the arithmetic and `interp.rs`
applies it. For every float intermediate the interpreter carries the `f32` reference value (the written order, round to
nearest even, which is what the CPU twin computes) and an `f64` interval that must contain every value a conforming
WGSL implementation may hold for it. Numbers (`2^-50`, `2^-100`, ...) are the oracle's own choices, deliberately coarse;
the line numbers are `docs/SEMANTICS.md` and the section numbers are the WGSL specification's.

| Rule | Source |
| --- | --- |
| `+ - *`: the result lies within 1 ULP of the exact operation over the operand intervals ("either neighbour of the exact result"). The `f64` hull is widened by a relative `2^-50` per operation (8 times `f64`'s own rounding) plus 1 ULP of the result. | SEMANTICS.md:190; WGSL 15.7.4 |
| Unary `-` is exact (the sign of a zero may differ, which no comparison or cast here can see). | SEMANTICS.md:191; WGSL 15.7.2 |
| A maximal `+`/`-` chain is taken after inlining locals and unrolling loops (a compiler may reassociate through SSA values), and for any number `n` of terms it may be regrouped, so it is widened by `n - 1` roundings, each of at most one ULP of `S' = S + 2(n - 1) ULP(S)`, where `S` is the sum of the terms' maximal magnitudes (not by one ULP per written operation; `S` alone is unsound at a binade edge, where earlier roundings carry a partial sum into the next binade and double the ULP). The chain continues through a `let` variable and through a loop accumulator (`acc = acc +/- term` over `T` trips is a chain of `T + 1` terms, and with nested loops of up to 65). A product in the chain may be fused; the exact product is inside the product's interval, so a fused result is inside the chain interval. | SEMANTICS.md:193, 194; WGSL 15.7.5 |
| `x / y`: the divisor interval must lie inside `+-[2^-100, 2^100]` on one side of zero (so the GPU's divisor is non-zero and inside the `[2^-126, 2^126]` of the 2.5 ULP bound), and the quotient is widened by 3 ULP (2.5 rounded up). | SEMANTICS.md:192; WGSL 15.7.4 |
| `i32 as f32`, `u32 as f32` only for `abs(x)` up to `2^24`: beyond it the GPU may pick either neighbour and the twin does not, so the program is discarded (`16777217i32 as f32` is the first case; a closed cast of that kind is discarded even in dead code, like the integer `static` check). | SEMANTICS.md:196; WGSL 15.7.6 |
| `f32 as i32`, `f32 as u32` only when the interval truncates to one integer (`trunc(lo) == trunc(hi)`) and the truncated value lies at least one inside the target range (`[-(2^31-1), 2^31-2]` for `i32`, `[0, 2^32-2]` for `u32`; a value of -1 or below to `u32` is outside it, while a value in (-1, 0) truncates to 0 on every party and is accepted; D39, KM-06). | SEMANTICS.md:197; WGSL 15.7.6 |
| Comparisons (`== != < <= > >=`) only when the two operand intervals are disjoint, so every party gets the same truth value; `==` and `!=` included. | SEMANTICS.md:195; WGSL 15.7.2 |
| Every float intermediate must be exactly `[0, 0]` or inside `+-[2^-100, 2^100]` on one side of zero. Then no subnormal (which an implementation may flush) and no infinity or NaN (indeterminate) occurs on any party, and an interval that straddles zero is refused. | SEMANTICS.md:199; WGSL 15.7.2 |
| The relative width of the output interval must be at most `2^-10`; a wider one is "unboundable", because the comparison would not mean anything. | R1-4 (a choice of this oracle) |
| Float `%`, `bool` buffers, and inputs outside the bounded set (magnitudes in `[2^-20, 2^20]` or exactly zero, no negative zero) are never generated. | SEMANTICS.md:198 (float `%` has no stated bound) |
| Vectors and matrices (format 6): every component is a float intermediate under the rows above. A component-wise `+ - * /` is the scalar operation per component. `dot` of `N` components, `(m * v)[i]` and `(m * m)[i][j]` are a sum of `N` products composed in the twin's written order, so they are one `+` chain of `N` product terms: widened by `N - 1` roundings of `S'` (the rule above), where each product is itself within 1 ULP of its exact value. A GPU that fuses (`fma`) or regroups inside the sum stays inside that interval, as for a scalar chain; there is no separate bound for vectors, the chain rule is the bound. The 4-term limit on an aggregate `+`/`-` chain (`fit_chain` in `generator.rs`) is a cap of the generator, not part of the bound: `dot` counts as one term there, so `a.dot(b) + c.dot(d)` is an 8-product chain, which is still sound because `Fl` joins the chains. | SEMANTICS.md:193, 194 and section 4 "Vectors and matrices"; WGSL 15.7.5 |

Discard labels of the float rules, counted per seed beside the integer ones in `seeds.txt`:

| Label | Condition |
| --- | --- |
| `float-domain` | an intermediate outside the range rule, a divisor interval that touches zero or leaves `+-[2^-100, 2^100]`, or a float-to-integer cast whose range is outside the target |
| `comparison-overlap` | a comparison of two overlapping intervals |
| `unboundable` | a float-to-integer cast whose interval spans two integers, or an output interval wider than `2^-10` relative |
| `float-construct-not-in-v1` | an integer above `2^24` in magnitude cast to `f32` |

These are spelled with hyphens, like the integer labels (the plan names them "domain", "comparison overlap",
"unboundable", "float construct not in v1"). A closed sub-expression is folded in `static_violation` with the same
rules (a float constant outside the range rule, a closed float division by zero, an integer above `2^24` cast to
`f32`, a closed float cast out of the integer range) and discarded under the one label `static`, in dead code too; the
four labels above are the ones the evaluation itself raises, for values that depend on the input.

The float-space numbers that follow are those of format 5 (the committed corpus then: seeds 1 to 8, 128 kept programs,
507 discarded; m6.dv changed seed 4's programs only, against format 4; format 6 keeps the float space and its first ten
programs per seed, see Format 6 below for today's totals): 33 of the 128 modules had an `f32` output at format 3 (2 to 8
per seed), and the float discards are `float-domain` 93, `unboundable` 31, `comparison-overlap` 15 and
`float-construct-not-in-v1` 2 (the per-seed numbers are in `seeds.txt`). Float programs are therefore not rare among the
kept ones, but the range and width rules reject a visible share of the candidates; that is the price of an oracle that
never guesses.

What the intervals do not do: they are an over-approximation, so a program can be discarded although every party would
agree (`x - x` is refused because its interval straddles zero; a chain through a variable is widened as one chain even
where the variable's own chain was already finished). They are also not a measurement of any GPU: that the interval
contains every value a conforming implementation may hold is a claim about the WGSL text, which the tests below check
against an adversarial CPU model, not against hardware.

## Seeds and the format stamp

A program is a pure function of `(seed, attempt)`: `prng.rs` (xorshift64* seeded through SplitMix64) writes 4096 bytes,
and `generator.rs` decodes them into the AST; bytes past the end read as zero, which selects the first option of every
choice, so a truncated stream is still a valid, smaller program (the last pass of the shrinker relies on that).
`generate_batch` walks attempts 0, 1, 2, ... and keeps the in-domain programs until it has ten (the float space), then
walks attempts 1 000 000, 1 000 001, ... of the vector space until it has six more (16 in all; `Batch.vector_attempts`
and `Batch.vector_discarded` count the second walk apart, and `difftest-gen stats SEEDS` prints them). Before format 6
it kept sixteen programs of the float space.

`seeds.txt` pins seeds 1 to 8 (R1-13: 8 seeds, 16 programs each, 128 programs): per line `seed programs discarded
reason=count... hash=`, where the hash is FNV-1a 64 of the emitted `src/lib.rs` followed by the input data of every
program (type name, length and the 32-bit patterns, little endian), because the data is not part of the emitted text.
The test `corpus::tests::the_committed_corpus_is_what_the_generator_decodes_today` regenerates the file and fails on
any difference. A change to `prng.rs`, `generator.rs`, `ast.rs` or `emit.rs` that alters a line is
either a bug or a deliberate format change: bump `DIFFTEST_FORMAT` (`generator.rs`; it is stamped into every emitted
crate and `seeds.txt`) and regenerate with `cargo run --locked --bin difftest-gen -- seeds 1 8 > seeds.txt`. Do not
regenerate to make a test pass without deciding which of the two it is.

Format history: 1 the integer core (m5.c1a); 2 the `f32` program space (m5.c1a2); 3 the condition and literal biases of the
float space (m5.c1b). Why 3: the executed-operator coverage of the batch (below) showed that under the format-2 float
space a third of the `if` conditions were a bare `true` or `false` and only 2 of the 55 evaluated `if` statements of
seeds 1 to 8 were taken by some elements and skipped by others, so most branches were dead or one-sided over the data.
Format 3 never writes a bare literal condition, makes a condition often a comparison of an input element with a value
of the data (for `f32` the midpoint of two elements, so the intervals stay disjoint) or a test on the element index, and
makes one float-space literal in eight a value that an input holds. The `Space::Integer` decode is untouched (the
m5.c1a pins still pass). Measured on seeds 1 to 8, format 3: 94 of 100 `if` statements evaluated, 58 of them with both
sides taken; `&&`/`||` right operands evaluated 8604 times and skipped 4584 times.

Format 4 (slice m5.c2): the decoding is unchanged, the acceptance is not. The divisor rule (retired by format 5, below) discards, as `static`,
the programs whose integer divisor slangc folds to zero (the first iteration of its loops, a zero factor, identical
operands, the branch that a known condition takes), because slangc refuses them. Of `seeds.txt`: every hash changes
(the stamp is in the emitted text), the discard counts of seeds 2 to 8 change, and only seed 4 gets different programs,
since only there did a discarded attempt come out of the accepted set (86 discards become 91; format 3 had accepted
`s4_a26`, the program slangc refused). Fix round 2 widened the rule (zero factor, identical operands, known conditions,
wrapping arithmetic, `!` and `^` opaque) and kept the format: the 16 programs of each of the 8 seeds have the same
hashes before and after the widening (only the discard counts of seed 8 moved, `division-by-zero` 1 to 2 and `static`
14 to 13, and the pinned integer line of seed 4, `overflow` 25 to 24 and `static` 16 to 17), and format 4 was not
published before the widening. The Space::Integer pin
(`the_integer_space_decodes_to_the_programs_of_m5_c1a`) keeps its 8 hashes, so the integer programs that m5.c1a
accepted are all still accepted; six of its discard lines were relabelled.

Format 5 (slice m6.dv, D74): the divisor rule of format 4 is gone, because the macro lowers a `/` or `%` with a divisor
that is not a literal to `__gust_div` and `__gust_rem` and slangc compiles those in every shape (the rule's only reason
was slangc's error E30002). The decoding is unchanged and the acceptance goes back to format 3's: every seed holds its
format-3 programs again (`the_float_space_still_holds_the_format_3_programs_of_every_seed` today), the `Space::Integer`
pin has its m5.c1a discard counts back, and against format 4 only seed 4 gets different programs (`s4_a26`, the `v2 %
v2` that slangc refused, is accepted again: 91 discards become 86). Of the 124 `static` discards of seeds 1 to 8 at
format 4, 13 were the divisor rule's; 111 remain (the closed-constant rules). The counts of format 5 equal those of
format 3 (a3a7798), label for label. Against format 4 the differences are: `division-by-zero` +9 (17 to 26: a live zero
divisor the interpreter now meets, where the retired rule had discarded it as `static`), seed 5 has one more
`comparison-overlap` and one more `float-domain`, and seed 4 stops 5 attempts earlier (`s4_a26` is kept; 91 discards
become 86). The discard total of the corpus goes from 512 to 507.

Format 6 (slice m6.d): vectors and matrices. A batch is now ten float-space programs and six vector-space programs (see
Program space). The float space is untouched: its draws, and so the first ten programs of every seed, are those of
format 5 byte for byte (`a_batch_is_ten_float_programs_then_six_vector_programs` compares them with
`generate_batch_in(Space::Float, seed)`, and `the_float_space_still_holds_the_format_3_programs_of_every_seed` keeps the
format-3 hashes of that space), and the emitted text of a scalar program is unchanged. What changes, for all eight
seeds, is the last six programs of each: the old programs 11 to 16 (float-space attempts) are replaced by vector-space
ones, so every line of `seeds.txt` changes (programs 16, as before; the discard counts, the reasons and the hash; the
stamp is in the emitted text anyway). The corpus has 128 programs and 587 discards at format 6 (507 at format 5): 306
are float-space discards, of the 80 scalar programs kept, and 281 are vector-space discards, of the 48 vector programs
kept (329 vector attempts, 14.6 % kept). The new reason `no-aggregate` (31) is a vector-space program with no vector or
matrix operation. The vector space has its own byte reads (`Space::Vector`: input types weighted towards `f32` with at
least one `f32` buffer, aggregate types in `pick_ty`, the vector draws in `expr`, `float_expr` and `assign`), selected
by the attempt number, so a later change to it never moves a float-space program.

Measured at format 6 (2026-11-02, RTX 5090, NVIDIA 610.88, Vulkan): `check-difftest --seeds 1-8` (GPU) executed 128
of 128 programs with 0 mismatches in 116.1 s (cargo build 27.0 s, run 60.9 s; first dispatch 0.43 s mean, 0.61 s
max; warm dispatch 2.6 ms max), and `--cpu-only` executed 128 of 128 with 0 mismatches in 5.6 s. Discards by class,
both batches (the same generation): `domain` 464, `comparison-overlap` 16, `unboundable` 45,
`float-construct-not-in-v1` 3, `no-input-read` 28, `no-aggregate` 31.

**What seeds 1 to 8 write, and what they do not.** The 48 vector programs write all four products (`* Mat2 Vec2`,
`* Mat4 Vec4`, `* Mat2 Mat2`, `* Mat4 Mat4`), `dot` on five of the six vector types (not `IVec4`), and 34 of the 72
binary, negation and swizzle forms the generator can write. The other 38 never appear in the text of any program of
seeds 1 to 8, so the gated batch gives no GPU evidence for them: `* IVec4 IVec4`, `* UVec2 UVec2`, `* Mat2 f32`,
`* Mat4 f32`, `* f32 Mat2`, `* f32 Mat4`, `* UVec4 u32`, `* i32 IVec4`, `* u32 UVec4`; `+ IVec2 IVec2`,
`+ IVec4 IVec4`, `+ UVec2 UVec2`, `+ Mat2 Mat2`, `+ Mat4 Mat4`; `- IVec2 IVec2`, `- IVec4 IVec4`, `- UVec2 UVec2`,
`- UVec4 UVec4`, `- Mat2 Mat2`, `- Mat4 Mat4`; `-IVec2`, `-Mat2`; `/ Vec2 Vec2`, `/ Vec4 Vec4`, `/ IVec2 IVec2`,
`/ IVec4 IVec4`, `/ UVec2 UVec2`, `/ UVec4 UVec4`, `/ IVec4 i32`, `/ UVec2 u32`, `/ UVec4 u32`; and the swizzles
`IVec2->2`, `IVec4->2`, `UVec2->2`, `UVec2->4`, `UVec4->2`, `UVec4->4`, `Vec2->4`. The list is derived, not read off
by hand: `difftest-gen forms` prints the 72 names the generator can write (every `agg_bin_ty` row under the four
arithmetic operators, every negation, every swizzle shape), the `operator` names of the report's `coverage` are the
ones that appear in some program's text, and the list is `comm -23` of the first against the second (both sorted).
The unit test `the_generator_writes_every_binary_row_negation_and_swizzle_shape_within_the_first_seeds` is the
CPU-only guard that each of the 72 is reachable: seeds 1 to 72 write all of them, seeds 1 to 68 still miss
`- Mat4 Mat4`.

**The wider batch.** `check-difftest --seeds 1-64` (evidence, not a gate; 2026-11-02, the same device) executed
1024 of 1024 programs with 0 mismatches in 628.9 s (cargo build 180.8 s, run 427.4 s). Discards: 4609 in all
(`domain` 3637, `comparison-overlap` 149, `unboundable` 355, `float-construct-not-in-v1` 25, `no-input-read` 197,
`no-aggregate` 246). It writes 71 of the 72 forms (not `- Mat4 Mat4`) and executes every operator in its text but
`/ UVec4 UVec4`. So the matrix `+`, `-`, `m * s` and `s * m`, the integer vector-by-vector forms and the other
swizzles have 64-seed GPU evidence once, not on every run.

## The emitter

`emit.rs` writes `src/lib.rs` (one documented `#[gust::gpu] pub mod s<seed>_a<attempt>` per program with the kernel
`k`), plus `Cargo.toml` and the template `Cargo.lock`. Conventions:

- Every non-atomic operand is parenthesised, so the text never depends on precedence. A dropped parenthesis is
  invisible to rustc (`a << b + c` compiles), so the tests re-parse the emitted text with Rust's real precedence
  (`reparse.rs`) and require the same AST.
- `i32::MIN` has no literal: it is written `(-2147483647i32) - 1i32`, which is exact. A negated `i32` literal is a
  literal (`-5i32`).
- An `f32` literal is written as its exact decimal expansion with the `f32` suffix (`emit::f32_literal`: the value is
  `m * 2^e` with an integer `m`, expanded by repeated multiplication of a decimal digit vector by 2, or by 5 for a
  negative `e`), for example `0.100000001490116119384765625f32`. Every `f32` has a finite exact decimal expansion, and
  rustc rounds a literal to nearest, so the literal re-parses to the same bits; `reparse.rs` checks it for every float
  literal of the corpus with Rust's `str::parse::<f32>`, and the dialect's own check is the round-trip `cargo check`.
  A negative literal is not an atom (`(-0.5f32)` is parenthesised where it is an operand). A negative zero has no
  literal and is never generated.
- Vector forms (format 6) are written as the `gust::math` API spells them: `Vec4::new(a, b, c, d)`, `Vec4::splat(s)`,
  `Mat4::from_cols(..)`, `Mat4::from_rows(..)`, `v.x`, `v.xy()`, `v.wzyx()`, `a.dot(b)`, `m.transpose()`, and `v.x = e;`
  for a component assignment. A constructor, a component, a swizzle, `dot` and `transpose` are atoms (a call or a
  postfix binds tighter than any operator, so `-v0.x` is `-(v0.x)` and `(-v0).x` keeps its parentheses because `-v0` is
  an operation); the operators of two aggregates are parenthesised like the scalar ones. The emitted text of a scalar
  program is byte for byte what it was. `reparse.rs` reads every vector form back with Rust's own precedence
  (`the_emitted_text_of_every_vector_form_reads_back_as_the_same_program`, and the round trip over the corpus).
- The crate allows `arithmetic_overflow` and `unconditional_panic`: rustc denies by default an overflowing operation it
  can see statically even in a branch no input reaches, while the filter is per input.
- Text uses `\n` on every platform. The manifest and a path are written through `path_for_toml`.

## The template lockfile

Generated crates build with `cargo check --locked` (the round trip) and `cargo test --locked` (the runner) against `gust` and `gust-wgpu`
(feature `slang-runtime`) by path, with `GUST_PROFILE=dev`, in a directory under the target directory. The committed
`template/Cargo.lock` is what makes `--locked` work there. It belongs to `template/Cargo.toml`, which a test requires
to equal the emitter's manifest with the root `../../..` and the package `difftest-case`. A test requires every
registry package of the template lock (the `wgpu` entry included) to be the same version and checksum as in the root
`Cargo.lock`.

Every generated crate has its own package name, `difftest-case-<suffix>` (`package_name`), and `lock_for` writes the
template lock with that one name replaced. This is not cosmetic. Observed while writing the round-trip tests: with one
package name and one shared target directory, `cargo check` of a crate holding a module the validator rejects reported
success (and a later manual run replayed warnings of a different crate). Reproduced without GUST (Opus review of
m5.c1a): two dependency-free crates with the same name, in different directories sharing one target directory, get a
single fingerprint unit, so freshness is decided by file times and the second crate can pass unchecked. It is a cargo
behaviour, not a GUST bug (the likely mechanism is cargo's path-relative stable hash, which is `.` for each of them;
not traced in cargo's source). The rule is therefore one unique package name per generated crate directory sharing a
target directory; with distinct names the same control fails as it should. The runner keeps it: the package of a
run crate is `difftest-case-run-<16 hex digits>`, the FNV-1a 64 of the format and the seed list, so the same batch
always gets the same name (the build stays incremental) and two different batches never share one. Before it builds,
`difftest-run` reads the `Cargo.toml` of every crate directory under `<target>/difftest/` and refuses to go on when two
carry one package name (`harness::unique_package_names`; a test writes two such directories and requires the refusal).

Line endings do not matter to cargo (`--locked` accepts LF and CRLF) or to the tests: `lock_for` copies the template lock
byte for byte as it is checked out, and the round-trip test compares against that. Git stores the file with LF (the
repository checks out CRLF on Windows through `core.autocrlf`, like every other text file); a refresh below makes
cargo write LF. So the checked-out ending wins: after a refresh keep whatever ending the file had in the working tree
(CRLF on this machine) and let git normalize it, so that the diff shows only the content change. After a dependency
change in the root workspace, regenerate the template lock and review the diff before committing it:

```sh
cargo run --locked --manifest-path tests/difftest/Cargo.toml --bin difftest-run -- --refresh-lock
```

`--refresh-lock` copies the root `Cargo.lock` over `template/Cargo.lock` and runs `cargo metadata --offline` on the
template manifest, which prunes the lock to the template's packages (the old file is restored if that fails). It prints
`template lockfile unchanged` when the content (line endings aside) is the same as before. When `--locked` rejects a
run crate's lock, `difftest-run` stops with "template lockfile stale; run `difftest-run --refresh-lock` and commit
tests/difftest/template/Cargo.lock" instead of cargo's message.

## The batch runner (`difftest-run`)

```text
difftest-run --seeds 1-2 [--shapes grid,pixel] [--cpu-only] [--report PATH] [--target-dir DIR] [--repo-root DIR]
difftest-run --refresh-lock [--repo-root DIR]
```

`--shapes` takes `grid` (the default) or `grid,pixel`; see Shapes below. `--shrink` refuses it (the reducer carries no
pixel kernel).

`--seeds` takes a range, a list or both (`1-4`, `3`, `1,3,5-6`); the report goes to stdout without `--report`. The
target directory is `--target-dir`, else `CARGO_TARGET_DIR`, else `tests/difftest/target`. Exit status: 0 every program
agreed, 1 at least one disagreement (the report lists them), 2 the harness itself failed (a build error, a stale template
lockfile, a result file missing or malformed). `difftest-run` has no random-seed mode: the seeds are always named, so a failure is reproducible from the report.
`cargo xtask check-difftest --seeds random:N` draws N seeds itself and passes them on as a list (see Gates).

What a run does:

1. For every seed, `generate_batch` and the interpreter produce the planned programs (16 per seed) and their expected
   outputs; the number of programs must equal 16 times the number of seeds.
2. It writes the run crate `<target>/difftest/run-<hash>/`: `Cargo.toml` (package `difftest-case-run-<hash>`, see the
   template lockfile above), the template `Cargo.lock`, `src/lib.rs` (the kernels of `emit_crate_source`, byte for byte,
   then a `#[cfg(test)] mod difftest_run` with one runner per program) and `data/<module>.txt` (the element count, the
   input buffers and the interpreter output as decimal `u32` words, so that a 4096-element program is not a source
   literal). The CPU twin exists only under `cfg(test)` of the crate that defines the kernel, which is why the runner is
   a test of the run crate and not code of `difftest`.
3. It builds with `cargo test --no-run --locked` and runs `cargo test --locked`, under `GUST_PROFILE=dev` (the profile
   that makes `gust` compile and run Slang at run time), in the same target directory, with `DIFFTEST_OUT` naming the
   result directory and `DIFFTEST_CPU_ONLY` set to `1` or `0`. The generated test runs every program in order: the
   twin through the generated `k::dispatch` (inside `catch_unwind`), then, unless cpu-only, the GPU through
   `HeadlessDevice::dispatch_buffers` twice (the first compiles the kernel with slangc and the native helper and lets
   wgpu create the shader module and pipeline, the second gives the warm time) and a readback. A panic is recorded as
   that program's error. The test asserts nothing about values; it writes one `<module>.json` per program.
4. `difftest-run` requires exactly one result file per planned program (a missing, extra, half-written or foreign file is
   a harness failure, not a pass), parses them, classifies, and writes `report.json`.
5. `--cpu-only` never creates a `HeadlessDevice` and never touches the GPU: the interpreter is compared with the twin
   only, and `mode` in the report says `cpu-only`. It is the `unit`-tier batch.

Nothing is cached between runs except cargo's own incremental state: the same seeds give the same crate directory and
package name, so a second run of a batch only rebuilds what changed.

### Classification

The interpreter output is the reference. The CPU twin must equal it **bit for bit for every type, `f32` included**
(R1-4; a twin `-0.0` against an interpreter `0.0` is a `twin-vs-interpreter` mismatch, pinned by a unit test). The GPU
is held to the oracle: integers and booleans bit for bit, and an `f32` output must be finite and inside the element's
`[lo, hi]` interval, which compares by value (a GPU may return either zero, and `-0.0` lies in `[0, 0]`). One program can produce several
findings; each is a `mismatch` with its program name, seed, index, attempt, kind, the number of differing elements, the
first one, and a message with the values (and for `f32` their bits).

| Kind | Meaning |
| --- | --- |
| `twin-vs-interpreter` | the CPU twin disagrees with the interpreter in any bit |
| `gpu-vs-interpreter` | an integer or boolean GPU element differs from the interpreter |
| `gpu-outside-interval` | an `f32` GPU element is not finite, or outside its interval |
| `twin-error`, `gpu-error` | the twin or the GPU path panicked or returned an error (the message is kept) |
| `gpu-not-validated` | a GPU output without the mark that wgpu created the module (WGSL validation with wgpu's default capabilities) |
| `pixel-vs-grid` | (`--shapes grid,pixel` only; `u32`, `i32` and `bool` outputs) the GPU bits of the pixel kernel differ from the GPU bits of the grid kernel of the same program. An `f32` output has no such kind: each shape is judged against the interpreter's interval (`gpu-outside-interval`, with the `shape` field), and the bit difference between the shapes is only recorded under `shapes.float_pixel_vs_grid` (D128) |
| `transport` | the interpreter output stored in the result file is not the one the harness computed: a bug of the harness itself |

WGSL validation: every kernel's first dispatch makes wgpu create the shader module and the compute pipeline with the
default capabilities (never `Capabilities::all()`); a validation error panics, is caught and becomes a `gpu-error`. So
`wgsl_validated` is true exactly when the first dispatch succeeded; the harness does not compile each kernel to WGSL
a second time (that would double the slangc time, see the cost table).

### Shapes (`--shapes`, slice mx.p1c)

By default every program runs as the `grid` kernel `k` (a compute dispatch). With `--shapes grid,pixel` every module
also gets a second kernel `k_px` after `k`: the same parameters and body under `#[kernel(shape = pixel)]` (no
`workgroup_size`), run as a full-screen fragment pass with storage writes (D120, D127 Option S). The text of `k` is not
touched, so the committed seed hashes of the grid form are unchanged, and a batch with `pixel` is another crate (its
identity includes the shapes) from the same seeds without it.

- **Which programs can be pixel kernels is the macro's decision, not the harness's.** The batch builds every `k_px`; the
  macro's own classifier refuses a pin it cannot honour with a `GUST-E0723` to `GUST-E0728` error, the harness reads
  those from the build log, rebuilds once without the refused `k_px`, and reports each such program as `grid`-only with
  the macro's message as the reason (`shapes.grid_only`, below). It does not guess in advance. Measured (seeds 1 to 8,
  2026-11-05): 128 programs ran as pixel kernels, none stayed grid-only. A hand mutation records the other path: a
  `return;` at the top of one `k_px` gives exactly one refusal, `GUST-E0727`, at that module's attribute (the permanent
  test is `tests/roundtrip.rs::a_return_in_one_pixel_kernel_is_one_refusal_that_the_harness_parses`).
- **The pixel run.** The dispatch is one 1-D call of `count` elements (the element index is `y * 8192 + x`, so a corpus
  program of fewer than 4096 elements is one row). The output buffer starts as `0xA5A5A5A5` (the grid output starts at zero),
  so an element the pass never wrote differs from the grid result instead of matching it by accident. The CPU twin
  runs `k_px::dispatch`, the GPU runs it through `dispatch_buffers` twice (first and warm), as for the grid kernel.
- **Classification.** The pixel twin and the pixel GPU are held to the interpreter by the same rules as the grid ones.
  For `u32`, `i32` and `bool` outputs the pixel GPU bits must also equal the grid GPU bits when both exist
  (`pixel-vs-grid`). For `f32` they need not (D128): a device may fuse `a * b + c` and round `x / y` within 2.5 ulp
  differently in the compute and the fragment stage (SEMANTICS section 4), so each shape is judged against the
  interval only, and the bit difference is an observation in the report (programs compared, programs with a
  difference, elements, the largest distance in ulps, the first element). It is never a mismatch (D39). The record
  that makes this necessary: an earlier form of `examples/shape-bench`'s K3 (an unfused multiply-add chain fed by a
  division; measured 2026-11-05, not kept) differed between the shapes by up to 4 ulp in 9 % to 11 % of the words on the
  RTX 5090 (driver 610.88), with identical Slang, WGSL and SPIR-V for both stages, which makes it the driver's lowering
  per stage. The bench's K3 is now X-S1's form (no division, exact coefficients, no `fma`: the dialect has none) and
  writes the same bits in both shapes, as do the difftest corpus programs (below). A grid GPU error is not
  blamed on the pixel shape. Every mismatch has a `shape` field (`grid` or `pixel`; the grid findings of older reports
  have none and mean `grid`). A `pixel-vs-grid` finding cannot be shrunk: `--shrink` runs the grid kernel only.
- **What this does not show: stray fragments.** Two things already drop a write outside the `n` elements: every corpus
  kernel's own guard (`if i < out.len()`), and the device's robust buffer access (`robustBufferAccess2`), which makes an
  out-of-bounds write harmless and invisible. A fragment that ran outside the `n` elements, or twice, would therefore
  not change a value, and this batch claims value identity (integers) and interval membership (`f32`) only. It also cannot see a second row: every
  corpus program has `n < 8192`, so no program reaches a row after the first. The fragment-count claim (each element is
  written exactly once) is the atomic fragment-count test of slice mx.p1b (M2), not this batch.
- **How much of the examples is `grid`-only.** Plan risk 11 asked for the grid-only reasons over the examples as well
  as the corpus. Measured 2026-11-05 by pinning every `#[kernel(workgroup_size(..))]` of the ten examples that have one
  (21 kernels) to `shape = pixel` in a scratch edit, compiling, and taking the macro's reasons (it reports the first
  construct that rules a kernel out; the edit was reverted, and the refused kernel was put back before the next round so
  that the kernels after it were classified too): 4 kernels are accepted (`vector_add::add`, `polynomial::evaluate`,
  `signal_pipeline::transform` and the `web-vector-add` kernel) and 17 stay `grid`-only. By code: `GUST-E0723` 4
  (`atomic_add` twice, a storage texture, an element type that is not a scalar), `GUST-E0725` 3 (a write at an index
  other than the element's own: `mark_live`, `gather_particles`, `scatter_particles`), `GUST-E0726` 2 (a conditional
  write to the output: `staged::transform`, `staged::summarize`), `GUST-E0728` 8 (a component write, or a read of the
  output: `calibrate`, `scalar_order`, `field_order`, `prepare_dispatch`, `classify`, `integrate`, `step`, `finalize`).
  `GUST-E0724` and `GUST-E0727` did not occur. The generated corpus has no `grid`-only program (128 of 128 ran as
  pixel), because every generated kernel writes `out[id.x]` once, whole, on every path; the examples are the realistic
  counter-weight.
- **One module per program, both kernels.** A module that holds a `workgroup_size` kernel and a `shape = pixel` kernel
  needs the sibling-kernel fix of mx.p1b (E0707); the batch needs it and does not work around it.

Gates: `check-full` runs `check-difftest --seeds 1-8 --shapes grid,pixel`. The pull request list, the default
`check-difftest` and the Gpu suite stay grid-only (an xtask test pins that no suite but Full adds `--shapes`).

Measured 2026-11-05 (RTX 5090, NVIDIA 610.88, Vulkan, dev profile; other sessions held about 2 GiB of VRAM at 0 % to
6 % utilisation), seeds 1 to 8 with the `short_circuit` GPU test: grid only, 141 s (harness cargo build 26.0 s, run
99.2 s); `--shapes grid,pixel`, 257 s (build 61.1 s, run 159.2 s), 128 pixel programs, 0 grid-only, 0 mismatches. In the
pixel batch the `timing` totals were, over the 128 programs: grid first dispatches 74.4 s and pixel first dispatches
73.2 s (both include the Slang compile), grid warm dispatches 194 ms and pixel warm dispatches 329 ms (tiny kernels on a
shared GPU: a noisy smoke, no threshold). A second run after the Slang cache was warm took 137 s (build 21.7 s, run
105.3 s; first dispatches 48.8 s grid and 49.3 s pixel, warm 145 ms and 248 ms), with the `f32` observation of the
report: 0 of the 36 `f32` programs differ in a bit between the shapes (0 elements, 0 ulp). The per-shape time of larger workloads is measured by `examples/shape-bench`
(`plans/shapes/probe/bench/table.md`), not here.

### report.json

Keys: `tool`, `format`, `mode` (`gpu` or `cpu-only`), `passed`, `seeds`, `programs_requested` (16 per seed),
`programs_executed` (the result files whose runner finished; equal to `programs_requested` on a pass), `adapter`
(`null` for cpu-only; otherwise name, type, backend and driver of the adapter wgpu chose), `per_seed` (`programs`,
`attempts`, `discarded`), `discards`, `mismatch_count`, `mismatches`, `coverage` and `timing` (generation, interpreter,
cargo build and run seconds, twin and GPU first and warm dispatch times), and, since mx.p1c, `shapes`: `requested`
(`["grid"]` or `["grid","pixel"]`), `programs_by_shape` (`grid` and `pixel` counts) and `grid_only` (`total`,
`by_reason`, the macro's clause with the quoted constructs elided, and `programs`, each with its message). With `pixel`
the `timing` object also holds `pixel_twin_ms_total`, `pixel_gpu_first_dispatch_ms_total` and
`pixel_gpu_warm_dispatch_ms_total`. `cargo xtask check-difftest` requires the shapes it asked for, `grid` equal to the
programs executed, `pixel` plus `grid_only` equal to them, the reasons summing to `grid_only.total`, and at least one
pixel program when pixel was requested.

`discards` lists `total`, `by_reason` (the labels of the tables above, per run, summed over the seeds) and `by_class`,
which maps the labels to the four classes of R1-4 plus two: `domain` (the integer labels, `float-domain` and `static`),
`comparison-overlap`, `unboundable`, `float-construct-not-in-v1`, `no-input-read` (a fifth class, which the plan does
not name; a program without an input read is a constant, and the generator discards it) and `no-aggregate` (a sixth,
from format 6: a vector-space program with no vector or matrix operation in it). A label the classifier does
not know makes the run fail, so a new label cannot be silently counted under another class.

A `report.json` is deterministic for given seeds and mode except for `timing` and `adapter`; compare fields, not files.

### Coverage

`coverage` is measured by running the interpreter again with counting on, over the kept programs (so the domain filter
is already applied): for each operator and type (`+ u32`, `- f32`, `&& bool`, `as i32`, `for`, ...) the number of
programs that contain it in the text, the number of programs in which it is evaluated at least once, and the total
number of evaluations; the operators that appear in the text of some program but were never executed (`never_executed`);
the `if` statements (in the text, with a constant condition, evaluated, taken, with the else side taken, and with both
sides taken by some element); and how often the right operand of `&&`/`||` was evaluated and skipped. A written operator
in dead code is not counted as covered. A run of seeds 1 to 8 at format 3 executes every operator in its text
(`never_executed` is empty). At format 6 the vector operators have names of their own (`* Mat4 Vec4`, `dot Vec2`,
`swizzle Vec4->2`, `new Vec4`, `comp UVec2`, `transpose Mat2`, `assign comp`, ...). Seeds 1 to 8 execute every operator
in their text but two (`never_executed` is `* Vec2 Vec2` and `- Vec4 Vec4`: written in a branch or a chain that no
element reaches), and they write 34 of the 72 binary, negation and swizzle forms; which 38 they do not write is listed
under Format 6 above, with the command that derives the list. The corpus has one `transpose Mat2` and no `transpose
Mat4`.

### Short circuit

`&&` and `||` must not evaluate the right operand when the left decides, on every party. The test
`tests/short_circuit.rs` uses `out[i] = (i < 24) && (a[i + 40] == 7)` (and the `||` twin) over 64 elements with a
64-element input: for the elements 24 and above, `a[i + 40]` would read out of bounds. The interpreter's domain tracking
accepts the program only because it skips the right operand; the twin (Rust `&&`) skips it; the GPU run must not fail
and must give the same values (CPU run: `cargo test`; GPU run: `-- --ignored`). The GPU test is Gpu tier; `cargo xtask check-difftest` runs it (without `--cpu-only`). Measured on the RTX 5090: both programs run,
no GPU error, the 128 outputs agree with the hand-computed values.

A value comparison cannot prove that the GPU skipped the read: wgpu clamps out-of-bounds reads of a storage buffer
(robust buffer access) and the operands are pure, so evaluating both sides would give the same answer. The GPU test
therefore also reads the WGSL (through `DIFFTEST_WGSL_OUT`, which makes the generated test write each kernel's
`compile_wgsl` text) and requires that the one read of the input buffer sits inside the block guarded by the left
operand. Slang lowers `&&` in a statement context to a structured `if`/`else` (`if(i_0 < u32(24)) { _S2 = a_0[...] == u32(7); } else { _S2 = false; }`,
and for `||` the read is in the `else` arm of `if(i_0 >= u32(24))`), which is a short circuit by construction; there is no `select`. A
unit test of the check feeds it a hoisted read and requires the verdict "not guarded". What this does not show is the
behaviour of a different Slang version or of a driver compiler below WGSL; the claim is about the WGSL that Slang emits
here.

### Cost of a batch (spike S5.3, cost half)

Measured 2026-11-01 on the development PC (RTX 5090, NVIDIA 610.88, Vulkan, 4 build jobs), dev profile, seeds 1-2
(32 v1 kernels in one crate), before the format-3 bump (the figures do not depend on it; the format-3 GPU batch of the
same seeds through `difftest-run` is in the last column, from the saved report-gpu-seeds1-2.json). Estimates are those of the M5 plan.

| Quantity | Plan estimate | Measured (probe) | `difftest-run`, seeds 1-2 |
| --- | --- | --- | --- |
| cargo cold build (nothing cached) | 2 to 3 min | 83 s | not repeated |
| cargo warm build (dependencies cached, new crate text) | 20 to 60 s | 7.9 s | 0.8 s (nothing to rebuild; the crate's first build was 4.5 s) |
| slangc plus helper, per kernel | 0.3 to 0.8 s | 0.53 s mean, 0.76 s max | included below |
| first dispatch per kernel (compile, module, pipeline) | in the line above | 0.52 s mean | 0.60 s mean, 1.74 s max |
| GPU run, per kernel (warm dispatch) | under 10 s per batch | 2.0 ms mean, 3.4 ms max | 1.6 ms mean, 3.3 ms max |
| CPU twin, per kernel | under 1 s | 0.1 ms mean | 0.1 ms mean, 0.51 ms max |
| interpreter, per program | under 1 s | 0.35 ms mean | 0.26 ms mean |
| whole GPU batch of 32, warm | (outcome C if over 5 min) | about 26 s | 24 s (0.8 build + 23.0 run) |
| whole GPU batch of 32, cold | (outcome C if over 5 min) | about 101 s | not repeated |

Outcome A: every figure is at or below the plan's estimate and the largest batch figure is a third of the 5-minute line
of outcome C. In the units of R1-13: `unit` keeps 4 seeds CPU-only, `gpu-software` per pull request keeps 2 seeds, the
`check-full` gate keeps all 8 seeds, the nightly `gpu-software` list runs seeds 1 to 72 (since m6.z1), and shrinking
stays local. On the GitHub runner, seeds 1 to 8 took 45.5 s of `difftest-run` on lavapipe (run 41442659698, format
5). Not measured, so not claimed: the
`unit` job's cold build on a runner. The shrink half of S5.3 is measured under Shrinking below.

### Mutation checks

Each of these was applied by hand, observed to fail, and restored byte for byte (`cmp`) before review. Each also has
permanent tests that fail for the same reason (the second was re-run against the unit tests: 7 fail).

| Mutation | Observed | Permanent test |
| --- | --- | --- |
| delete one result file before classification | `expected exactly 16 result files, found 15`, exit 2 | `classify::tests::exactly_k_result_files_are_required...` |
| a wrong literal in the emitter (`u32` literals plus 2), in a `--cpu-only` batch | 8 of the 16 programs of seed 1 reported (`twin-vs-interpreter`, `twin-error`), each named, exit 1 | `emit::tests` (exact text, re-parse), `corpus::tests`, the integer-space pin |
| compare the GPU `f32` by bits (or the twin by value) | the `-0.0` tests fail | `classify::tests::the_gpu_f32_is_held_by_value...`, `the_twin_is_compared_bit_for_bit...` |
| one package name for different batches | the uniqueness tests fail | `harness::tests::every_distinct_batch...`, `two_crate_dirs_with_one_package_name_are_refused` |
| mx.p1c: a `return;` at the top of one `k_px` in a pixel run crate | `cargo check` fails with exactly one `GUST-E0727` at that module's `#[kernel(shape = pixel)]` line, and `shapes::refusals` finds that one refusal (so the harness would report that program `grid`-only with the macro's message) | `tests/roundtrip.rs::a_return_in_one_pixel_kernel_is_one_refusal_that_the_harness_parses` |
| m6.d (a): swap the operands of `m * v` in the interpreter (`agg::mat_vec` reads the matrix row-major) | 3 of the 18 `vector_tests` fail; a `--cpu-only` batch of seeds 1 to 8 reports 5 `twin-vs-interpreter` mismatches, each with its seed (s3_a1000019, s6_a1000057, s6_a1000058, s7_a1000038, s7_a1000048), exit 1 | `vector_tests::a_matrix_times_a_vector_uses_the_columns`, `a_matrix_times_a_matrix_multiplies_each_column`, `from_rows_and_transpose_swap_rows_and_columns` |
| m6.d (b): zero tolerance in `Fl::interval` (`w = 0`) | the batch stops before any program runs: the interpreter's own soundness check (`checked_value`) panics, "reference -85.83885 outside its interval (-85.83884811401367, -85.83884811401367)"; class: a bound (the interval is too narrow for the reference itself), not a program | `interp::checked_value` (debug build), the interval tests of `float.rs` |
| m6.d (c): the generator stops giving a `Mat2` an input read (a constant `Mat2` reaches the SPIR-V path) | 2 tests fail: the guard of `Program::check` refuses it ("a `Mat2` built from no input read could be a compile-time constant", seed 1 attempt 1000021), and the corpus pin is stale | `vector_tests::the_vector_space_is_deterministic_and_never_writes_a_constant_mat2`, `a_mat2_without_an_input_read_is_refused...` |

The m6.d mutations were restored with `cp backup file && touch file` and compared with `cmp` before the gates. The (b)
row is not a GPU-side finding: with zero tolerance the reference cannot sit inside its own interval, so the interpreter
refuses before any party runs. Mutation (b'), review of m6.d: narrow every output interval to the reference point
(`Some((reference, reference))` in `interp.rs`, so the assert holds) and run the GPU batch of seeds 1 to 8. It fails
with 8 `gpu-outside-interval` mismatches (s3_a37, s3_a44, s5_a1000003, s6_a10, s6_a1000003, s6_a1000031, s6_a1000057,
s6_a1000058): 561 of the 8589 `f32` output elements of the 36 `f32`-output programs differ from the written-order
reference on the RTX 5090, by a few ULP (fusing or regrouping). So the float comparison carries load on this device,
and an interval narrowed to the reference is caught. It was restored and compared with `cmp`. That the interval
contains what other devices hold is not measured (D39: a recorded limitation).

## Shrinking (slice m5.c1c)

`difftest-run --shrink <seed>:<index>` takes program `index` of the 16 that seed plans (the `index` field of a
mismatch in the batch report) and reduces it while it keeps failing the same way. It is a local tool: CI never calls
it and uploads the seed, the program text and the three outputs of a failing batch instead. Exit status: 0 a failure
record was written, 1 the program does not fail (nothing to shrink), 2 the harness failed. Options: `--cpu-only`,
`--report PATH` (the record; stdout without it), `--max-candidates N` (default 60), `--max-minutes M` (default 10).

One candidate is one build of a single-program crate `<target>/difftest/shrink-f<format>-s<seed>-i<index>/`, one test
run (interpreter, CPU twin and, unless `--cpu-only`, the GPU) and one classification, so it costs a cargo build plus a
GPU dispatch. The reducer itself (`reduce.rs`, std only) knows nothing about cargo: it takes a predicate
`Program -> Option<u32>`.

Size, against the 400-line limit of R1-12. The ruling of the coordinator is that the limit applies to the reducer
search, `reduce.rs`: 399 lines (351 without comments and blank lines). Two other parts are harness code that the
proptest fallback would need as well, so they are counted apart: the identity predicate, the failure record and the
executor in `shrink.rs` (302 lines), and the deadline in `harness.rs` (about 70 lines for the process-tree kill, the
output drains and the timed run, and about 15 more in `cargo_step`). Tests are in `reduce_tests.rs` and
`shrink_tests.rs`.

Passes, in this order, each to a fixed point, and the whole round repeats until no pass changes anything:

1. delta debugging over statements (chunks are dropped, a block is replaced by its body or by one branch);
2. replace a subexpression by the value the interpreter observed for the first failing element (the literal of that
   bit pattern, in the expression's type);
3. loop trip counts toward 1 (to 1, or to half);
4. element count toward the failing index plus one, rounded up to the workgroup size (64), else the index plus one;
5. literals (not `bool`) to 0, to 1 (-1 for a negative `i32`), or to the power of two at or below the literal (an
   `f32`: the sign and exponent with a zero mantissa). There is no halving step, so a literal that is already a power
   of two does not move: this is why the reproducer of the measurement below stops at 2^31 and not at 16. A halving
   or decrement candidate (about 27 more builds here) is a candidate improvement for m5.c2;
6. the byte stream: the original 4096 bytes are cut at 0, 1/8, 1/4, 1/2 and 3/4 of the bytes the program used and
   decoded again (a truncated stream reads zeros, which selects the first option of every choice, so it is still a
   valid and smaller program), which also removes trailing zeros.

Bounds, as built:

- Every candidate must be type-valid (`Program::check`) and pass the interpreter's domain filter (`discard_reason`)
  before it costs a build; those rejected there are not counted.
- Identity (`shrink.rs`, `reproduces`): the same classification, the same output type (the output buffer is `out`),
  and a mismatch that names an element. "The same classification" is the set of kinds of the mismatches that name an
  element: if the original is `{twin-vs-interpreter, gpu-vs-interpreter}`, a candidate that loses either one, or that
  gains another kind, is a different failure. A compile error, a validator rejection, an error of a party, a mismatch
  without an element, or a timeout is "does not reproduce". The failing element is the one of the first mismatch of
  the original's kind in `classify` order (the twin first).
- Accept only if strictly smaller by size, the tuple (statements, expression nodes, loop trips, elements, sum of
  literal magnitudes) in lexicographic order. This is what makes the search terminate.
- A cache keyed by the FNV hash of the emitted program text plus its inputs: a candidate seen before costs nothing and
  does not use the budget.
- One clock for the time budget. The original run comes first and has its own 120 s per cargo step; the 10 minutes
  start after it (`Limits::origin`, set by `shrink_with` once the original run has failed as expected), and `reduce`
  and the step deadlines both read that one origin. Each cargo step of a candidate has a deadline of the smaller of
  120 s and the time left (`step_deadline`), so the wall clock cannot overrun by more than the kill itself. A candidate
  is not started at all when less than `MIN_STEP` (1 s) is left: that ends the search with `stopped_by: time budget`,
  and it is neither run, nor counted as a timeout, nor taken from the candidate budget.
- On a timeout the process tree is killed (`taskkill /T /F` on Windows; elsewhere the child is spawned as the leader
  of a process group and the group is killed), the output so far is written to the step's log, and the candidate is
  "does not reproduce". After a kill the output readers are not joined, so a child that survived cannot block the
  search. On Unix, Ctrl-C is not forwarded: the child is in its own group, and std has no signal handling, so an
  interrupted `--shrink` can leave cargo running there until it finishes (`kill -- -<pid>` of its group stops it);
  on Windows the console delivers Ctrl-C to the children.
- Budget: 60 candidates or 10 minutes of search, whichever comes first (the record says which one stopped it).

The failure record (`--report`) is JSON with `tool`, `format`, `mode`, `adapter`, `seed`, `index`, `attempt`,
`classification`, `output_buffer`, `output_type`, `failing_element`, `candidates_used`, `cache_hits`, `accepted_steps`,
`stopped_by` (`fixed point`, `candidate budget` or `time budget`), `seconds`, `candidates` (the seconds and result of
each candidate that ran) and two blocks, `original` and `shrunk`. Each block has `program_text` (the emitter's output
at the time of the run: with a fault in the emitter, it is the faulted text), `size` (`statements`,
`expression_nodes`, `loop_trips`, `elements`, `literal_magnitudes`), `inputs` (type and decimal `u32` words per
buffer), and the three outputs: `interpreter`, `twin` and `gpu` (with `twin_error` / `gpu_error` when a party failed)
plus the `mismatches` of that program. The seed and index reproduce the original; the shrunk text is the program to
read.

### Cost of shrinking (spike S5.3, shrink half)

Measured 2026-11-01 on the development PC (RTX 5090, Vulkan, dev profile, warm target directory). A fault was injected
by hand into the emitter in the lane only (`u32` literals of 16 or more written one too large), seed 1 gave 8 mismatch
records for 4 of its 16 programs on the GPU, and program 8 (3393 elements, the most differing elements) was shrunk
with a 40-candidate budget. The fault was removed afterwards (the file hash equals the original). The same binary
was run twice, and the binary was rebuilt and run again after each of the two rounds of review fixes.

| Quantity | Plan estimate | Measured |
| --- | --- | --- |
| time per candidate | 5 to 15 s | means 9.0 s, 7.45 s (max 13.4 s), 6.97 s, 6.69 s (max 7.34 s) in runs 1 to 4 |
| candidates used | 15 to 30 | 17 in all four (9 accepted, 8 not; 4 more were cache hits), fixed point |
| total | under the 10 minute budget | 162.8 s, 133.9 s, 140.2 s, 126.0 s |
| size, original to shrunk | | 2 statements, 29 nodes, 3393 elements to 1 statement, 7 nodes, 1 element |

Outcome: **A (one program, one literal fault, warm, RTX 5090)** in the units of R1-13: both figures are inside the
estimate, so the reducer stays as built and no property test fallback is needed on this evidence. The fault
reproduces on nearly any literal of 16 or more, so the sample is favourable; a failure that needs loops, a cold
target directory or a runner without a GPU is not measured. The shrunk program keeps the one `let` that reads an
input (the domain filter discards a program that reads no input) and a result literal of 2^31: the interpreter
computes 2147483648, and the faulted emitter writes `2147483649u32` into the text of the record, which is what the
twin and the GPU computed. It stops at 2^31 because of pass 5 (above). The failure is a `twin-vs-interpreter` one
because the fault is in the emitted text, which the twin and the GPU share (the record's `mismatches` hold both).

### Shrinker mutation checks

Each was applied by hand to the lane, observed to fail in the unit tests, and restored (copy back, hash compared with
the saved original). The first four ran on the first version of the rule, (a) was re-run on the final one, and
(f) and (g) were added in fix round 2.

| Mutation | Observed |
| --- | --- |
| identity accepts any failure (a) | `only_the_same_classification...` and `a_candidate_that_gains...` fail |
| identity is one kind again (no set check) | `a_candidate_that_gains_or_drops...` fails |
| the strictly-smaller check is dropped (b) | `the_search_ends_at_a_fixed_point...` fails at its 20 s bound |
| a timeout counts as reproducing (c) | `a_build_failure_and_a_timeout...` and `candidates_that_time_out...` fail |
| the cache lookup never hits (d) | `a_candidate_seen_before_is_not_run_again` fails (a call counter) |
| deadlines count from the start of `shrink_with` again (f) | `an_original_run_that_takes_most...` fails |
| no `MIN_STEP` (g) | `no_candidate_is_started_with_less...` and `a_candidate_is_not_started...` fail |

The kill path has its own test, `harness::tests::a_hung_step_is_killed_with_its_children_and_returns_at_the_deadline`:
the test binary runs itself as a sleeping process that starts a sleeping copy of itself, the step returns at its 6 s
deadline with the output so far, and both processes are gone afterwards. It passed on Windows (the gate) and on
Linux (WSL Ubuntu 24.04, the process-group path, run by hand with `cargo test --offline`; not part of any gate).

## Gates (slice m5.c2)

`cargo xtask check-difftest` ([VALIDATION](VALIDATION.md), "The differential batch") is how the gates run this crate.
It runs `difftest-run` (nothing here is reimplemented), reads `report.json` by fields, and fails unless the report has
the mode and the seeds that were asked for, `programs_executed` equal to `16 x seeds`, no mismatch and, on the GPU
tier, an adapter. Every failure line starts with `check-difftest:` and names the seed. On the GPU tier it also runs
`cargo test --locked --manifest-path tests/difftest/Cargo.toml --test short_circuit -- --ignored` and requires the GPU
test to be reported `ok` by name. Both cargo calls get the cargo target directory of the run in `CARGO_TARGET_DIR`, so
the generated crate and the short-circuit test build beside the workspace's own artifacts. The report goes to
`<target>/difftest-report.json` (`--report PATH` moves it; the file is removed before the run, so a stale one cannot
pass). `--seeds` takes a list, a range or `random:N` (N from 1 to 64): the xtask draws N distinct seeds from the clock
in 10000 to 1000009999, prints them and runs them as a list, because `difftest-run` itself only takes named seeds;
rerun a failure with `--seeds <the printed list>`.

| Gate | Command | Seeds | Programs | Parties |
| --- | --- | --- | --- | --- |
| Unit suite (local), CI job `difftest` | `check-difftest --cpu-only --seeds 1-4`, after the harness crate's fmt, clippy and tests | 1 to 4 | 64 | interpreter, CPU twin |
| Gpu suite | `check-difftest` | 1 to 8 | 128 and the `short_circuit` GPU test | interpreter, CPU twin, GPU |
| `check-full`, last integration check | `check-difftest --seeds 1-8 --shapes grid,pixel` | 1 to 8 | 128 (each also as a `pixel` kernel) and the `short_circuit` GPU test | the same, GPU on the hardware adapter, and the pixel kernel against the grid kernel |
| `gpu-software`, pull request list | `check-difftest --seeds 1-2` | 1 to 2 | 32 and the `short_circuit` GPU test | the same, GPU on lavapipe |
| `gpu-software`, nightly list | `check-difftest --seeds 1-72` | 1 to 72 | 1152 and the `short_circuit` GPU test | the same, GPU on lavapipe |

Measured 2026-11-02 on the development PC (RTX 5090, Vulkan, dev profile), the whole command including the harness's
cargo build and run: the Unit batch (`--cpu-only --seeds 1-4`) takes 88 s cold (the dependency tree of the generated
crate, `wgpu` and the rest, not built yet; 78 s of it is the cargo build), 15.5 s when the dependencies are built and
the crate is new, and 3.3 s when the crate is unchanged. The GPU batch of seeds 1 to 2 with the short-circuit test takes
36.5 s; the GPU batch of all 8 seeds with the short-circuit test takes 85 s. These are PC numbers; no runner number
exists until the first CI run of the slice.

The first 8-seed GPU run (format 3) found a real limitation: seeds 1 to 3 and 5 to 8 passed, and seed 4 reported one
mismatch, program 5 (`s4_a26`), kind `gpu-error`. Slang refused the kernel at compile time with
`error[E30002]: divide by zero` on a generated `v2 % v2`, where `v2` is the counter of an unrolled `for v2 in 0u32..4u32`
loop and the statement sits in a branch no element takes. The interpreter accepted the program because the branch is
dead; slangc unrolled the loop and folded `0 % 0`. It was deterministic (seed 4 alone reproduced it) and the shrinker
could not reduce it (`--shrink 4:5` says the program "does not fail (1 mismatches without an element)": a compile error
has no differing element). Format 4 (fix round 1 of the slice) refused the shape in the generator and DIAGNOSTICS
wrote the limitation up; slice m6.dv (D74) cured it in the macro (`__gust_div` and `__gust_rem`) and format 5 took the
refusal back: `s4_a26` is kept again, and `check-difftest` runs it (seeds 1 to 8 on the GPU, below).

The harness crate has gates of its own in the Unit suite (fix round 1; the crate is a workspace of its own, so the
root's fmt, clippy and test never reached it): `cargo fmt --manifest-path tests/difftest/Cargo.toml -- --check`, `cargo
clippy --locked --manifest-path tests/difftest/Cargo.toml --all-targets -- -D warnings` and `cargo test --locked
--manifest-path tests/difftest/Cargo.toml` (the generator, interpreter, emitter, shrinker and corpus tests and the CPU
`short_circuit` test; the GPU one is `--ignored`). They are Unit steps, next to the CPU-only batch, and the four of
them are the steps that `cargo xtask test --suite Unit --skip-difftest` leaves out: CI runs them in the job `difftest`,
beside `unit`, so the `unit` and `slang` jobs keep their time (R4.10). On the development PC, whole commands including
the wrapper's cargo slot wait and the first compile of the profile: format check 49 s, clippy 55 s, tests 247 s (131
tests, 9 s of them running).

Fix round 2 added the same three cargo calls to `check-full` (`DIFFTEST_CARGO_CALLS` in `xtask/src/suite.rs`, run by
the Full mode of `verify` after the workspace tests, with a test of the exact list in `tests/verify_steps.rs`), so the
release gate reaches the harness crate's own fmt, clippy and tests as well. The CI job `difftest` sets one
`CARGO_TARGET_DIR` for the job and caches it (`actions/cache`, as gpu-software does, with `tests/difftest/**/Cargo.lock`
in the key): the xtask, the harness build, the `short_circuit` test (it reads the variable) and the run crate of the
batch (`difftest-run --target-dir`, which the xtask points at the same directory) build `wgpu` once there; the
round-trip test keeps its own nested directory under `<target>/tmp` on purpose (a nested cargo must not share the
directory of the cargo test that runs it). The first runner times, cold and warm, are not known until the job has run.

### The template lockfile went stale once

The first gate run of m5.c2 failed with "template lockfile stale": `tests/difftest/template/Cargo.lock` (m5.c1a)
predates the lockfile change of m4.g2, which added `serde`, `serde_core`, `serde_derive` and `termcolor` under
`codespan-reporting`. `difftest-run --refresh-lock` fixed it (60 added lines; line endings kept). The Unit suite now runs
the batch, so a drift of the root lock fails the Unit gate and CI at once instead of the next time somebody runs the
harness.

## Tests

- `prng`: the SplitMix64 reference vector, pinned xorshift64* outputs.
- `json` (the harness's own reader and writer; std only): values round trip, malformed text is rejected, words outside
  `u32` are rejected.
- `classify`: every discard label has a class and the domain class is the integer labels plus `float-domain` and
  `static`; the twin bit for bit (a twin `-0.0` against `0.0` is a mismatch), the interval check of a GPU `f32` (by value, either zero), bit equality of integers, the
  errors of a party, cpu-only skipping the GPU, a result file must hold the count and the outputs, and exactly K result
  files are required.
- `runsrc` and `harness`: the run crate source is the kernel source plus the test module, the data file, rewriting only
  what changed and dropping stale data, seed lists, unique package names (two directories with one name are refused), the
  stale-lock message, the template manifest, the report's discard counts by reason and class, executed-operator coverage.
- `interp` (coverage): executed against written operators, the right operand of `&&` counted only when it runs, a
  covered run equals a plain run.
- `interp`: hand-computed vectors for each domain condition, the normal cases around them, short-circuiting, index ==
  `len()`, loops and branches; the `static` check on dead code, nested constants, results and loops.
- `float`: the ULP function, the range rule, the widening of a single addition, of a 9-term chain (8 ULPs of the term
  sum) and of products and quotients, the divisor rule, disjoint comparisons, negation.
- `interp` (floats): the reference value in the written order, a product that is exactly zero, the `16777217` cast
  (discarded, also in dead code; `16777216` is accepted), comparison overlap, division and range, an unboundable
  difference, float-to-integer casts, a loop accumulator as a chain of `T + 1` terms, a chain through a variable,
  static float constants.
- `soundness` (test only; models, not hardware): for 6000 random small float programs (loops, divisions, chains,
  negation, up to depth 3), every element the interpreter accepts has its twin result, 12 adversarial `f32`
  evaluations and 12 `f64` evaluations inside `[lo, hi]`. The adversarial model unrolls loops and inlines variables,
  adds each maximal chain in one of three groupings (a left-deep fold in a random order, in the written order, or a
  balanced tree), fuses a product with the addition at random, and rounds every result to either `f32` neighbour of
  the exact value (a quotient to within 2.5 ULP); in half of the rounds it picks, at every rounding, the neighbour
  away from the twin. Random inputs include edge values (`+-2^20`, `+-2^-20`, `1 - 2^-24`, ...). The `f64` model adds
  in another random order. It requires at least 400 accepted programs, 100 with a loop and 40 with a division. Also: the polynomial
  worked example of SEMANTICS.md (the 31u GPU side and the 15u twin side around the exact value at `v = 1` lie inside
  the interval; its half width is 37u), the polynomial over the input domain away from its root and refusal near the
  root, a 9-term loop chain, and the two binade-edge chains of the Opus review (`1 - 2^-24`, `2^-26`, `2^-30`, and the
  same at scale 1024 with `2^-20`), fixed in `float` and run through the models in `soundness`.
- `generator`: determinism per seed, different seeds differ, batch size and discard accounting, every attempt and every
  truncated stream is well typed, operator coverage of the corpus for both types, no literal zero divisor in 20000 decoded
  programs, no static constant outside the domain and no literal shift of 32 or more in the 1600 accepted programs of
  seeds 1 to 100, the static discard being live.
- `vector_tests` (slice m6.d, 20 tests; the arithmetic is `agg.rs`): hand-computed `dot`, `m * v`, `m * m`, `from_rows`
  and `transpose` (a product is the columns, not the rows), a 4-wide product equal to the scalar sum of its row,
  swizzles, scalar operands and the order of a quotient, the integer domain per component, the float range rule per
  component, a component assignment, the `Mat2` guard (refused without a direct `f32` input read, also when the only
  read is behind a cast; a `Mat4` is not), ill-typed programs refused by `check`, a closed vector expression outside the
  domain is `static` also in dead code, operator names in the coverage, the emit/reparse round trip of every vector
  form, determinism of 600 vector attempts with `check()` on each and no constant `Mat2`, the vector programs of seeds 1
  to 8 passing the domain and using 20 named forms, the generator writing every one of the 72 binary, negation and
  swizzle forms within seeds 1 to 72 (the list is `ast::writable_aggregate_forms`), and the float space alone writing no
  vector. The generator tests of the older space (`the_float_space_still_holds_the_format_3_programs_of_every_seed`,
  `a_batch_is_ten_float_programs_then_six_vector_programs`) pin that the scalar programs did not move.
- `emit` and `reparse`: exact text of a module, parenthesisation table, re-parse equality for every program of the
  corpus, the template manifest, the template lock against the root lock.
- `corpus`: FNV reference vectors, the committed `seeds.txt`, the hash covering the input data, the `MIN / -1` discard
  inside seeds 1 to 8.
- `reduce` and `shrink` (with a fake predicate and a fake executor, no cargo): the size order, each pass on a small
  program, termination at a fixed point (the predicate asserts that every accepted candidate is strictly smaller), the
  cache (a predicate call counter), the candidate and time budgets, the filters (ill-typed and out-of-domain candidates
  never reach the predicate), the identity rule (same classification with an element; a build failure, a timeout, a
  different classification or an error never reproduces), `parse_target`, and an end-to-end shrink against a fake
  executor whose GPU is wrong exactly when the program contains a `u32` product; a candidate that gains or drops one
  party's mismatch is another failure; every run is given at most the time left of the budget as its deadline;
  the budget starts after the original run, and a candidate is not started with under `MIN_STEP` left.
- `harness` (deadline): a sleeping process and the sleeping copy of itself that it starts are killed at the deadline
  and gone afterwards, the step returns promptly with the output so far, and a finishing command returns all its output.
- `tests/short_circuit.rs`: see Short circuit above (a CPU test, an `#[ignore]`d GPU test, and the check of the WGSL guard). It builds its run crates in `CARGO_TARGET_DIR` when that is set, else under `CARGO_TARGET_TMPDIR`.
- `tests/roundtrip.rs`: the run crate of seeds 1 to 8 (128 kernels plus the generated test module) passes
  `cargo check --tests --locked`; all 128 corpus programs (the float ones included, so the dialect accepts `f32` buffers,
  arithmetic, comparisons and casts as emitted) in one crate pass `cargo check --locked`; a crate with a type error
  and a crate with a module the `#[gpu]` validator rejects (`while`) each fail with the expected message (the check is
  live). It runs `cargo check` directly, not through the repository's cargo lock wrapper, because the test itself runs
  inside the outer locked `cargo test` gate, which already holds the slot; its generated crates use their own target
  directory under `CARGO_TARGET_TMPDIR`.
