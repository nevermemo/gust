# Differential program generator (`tests/difftest`)

Status: slices m5.c1a (the integer core) and m5.c1a2 (floats v1, the exact-interval subset of R1-4). The crate
generates scalar integer, boolean and `f32` kernels from a seed, evaluates them with an independent interpreter that
tracks the defined domain (and, for floats, an interval of every value a conforming WGSL implementation may hold), and
writes them as a `#[gust::gpu]` crate. It has no GPU code and no runner: the batch runner and shrinking are the later
slices m5.c1b and m5.c1c of the M5 plan. Nothing here is GPU evidence; a passing unit test is evidence about the
generator, the interpreter and the emitter only.

`tests/difftest` is a workspace of its own (own `Cargo.lock`, std only, `publish = false`), like `tests/ui` and
`tests/dual`; the root workspace does not contain it. Run it from that directory with the repository's usual cargo
setup:

```sh
cd tests/difftest
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked            # unit tests and the emitter round trip (needs cargo and the cached registry, no GPU)
cargo run --locked --bin difftest-gen -- show 1 0     # the Rust text of program 0 of seed 1
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
  and then the dividend is often a plain read of an `i32` input, so that `i32::MIN / -1` is reached; seed 2 of the
  corpus discards one program for it. Over seeds 1 to 300 the discard fires in 23 seeds, about one seed in thirteen.
- Floats (the subset below): `f32` literals, locals and inputs; `+ - * /`, unary `-`, comparisons, `as` between `f32`
  and `i32`/`u32` under the conditions of the float rules, loop accumulators (`let mut acc`, then `acc = acc +/- term`
  first in a loop body of at most 8 trips per loop, but with the nesting above a loop-carried chain runs up to 64
  times, 65 terms; see the chain rule below), and at most 4 written terms in a `+`/`-` chain outside loops. Never float
  `%`, never a `bool` buffer, never an input outside the bounded set. An integer-only decode is kept as
  `Space::Integer`: it reads the same bytes as format 1 and decodes to the m5.c1a programs, and a test pins the m5.c1a
  discard counts of seeds 1 to 8 and the hashes of its 128 programs (emitted text and input data, with the format stamp read as 1), which equal those
of the m5.c1a seeds.txt: format 2 changes the stamp, not the integer programs.
- No structs, helpers, `Option` or atomics.

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
`binary` and `unary`. Inside the domain the three parties (interpreter, CPU twin,
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

The committed corpus (seeds 1 to 8, 128 kept programs, 374 discarded): 31 of the 128 modules have an `f32` output
(2 to 5 per seed), and the float discards are `float-domain` 64, `unboundable` 20, `comparison-overlap` 5 and
`float-construct-not-in-v1` 1 (the per-seed numbers are in `seeds.txt`). Float programs are therefore not rare among
the kept ones, but the range and width rules reject a visible share of the candidates; that is the price of an
oracle that never guesses.

What the intervals do not do: they are an over-approximation, so a program can be discarded although every party would
agree (`x - x` is refused because its interval straddles zero; a chain through a variable is widened as one chain even
where the variable's own chain was already finished). They are also not a measurement of any GPU: that the interval
contains every value a conforming implementation may hold is a claim about the WGSL text, which the tests below check
against an adversarial CPU model, not against hardware.

## Seeds and the format stamp

A program is a pure function of `(seed, attempt)`: `prng.rs` (xorshift64* seeded through SplitMix64) writes 4096 bytes,
and `generator.rs` decodes them into the AST; bytes past the end read as zero, which selects the first option of every
choice, so a truncated stream is still a valid, smaller program (the m5.c1c shrinker relies on that). `generate_batch`
walks attempts 0, 1, 2, ... and keeps the in-domain programs until it has 16.

`seeds.txt` pins seeds 1 to 8 (R1-13: 8 seeds, 16 programs each, 128 programs): per line `seed programs discarded
reason=count... hash=`, where the hash is FNV-1a 64 of the emitted `src/lib.rs` followed by the input data of every
program (type name, length and the 32-bit patterns, little endian), because the data is not part of the emitted text.
The test `corpus::tests::the_committed_corpus_is_what_the_generator_decodes_today` regenerates the file and fails on
any difference. A change to `prng.rs`, `generator.rs`, `ast.rs` or `emit.rs` that alters a line is
either a bug or a deliberate format change: bump `DIFFTEST_FORMAT` (`generator.rs`; it is stamped into every emitted
crate and `seeds.txt`) and regenerate with `cargo run --locked --bin difftest-gen -- seeds 1 8 > seeds.txt`. Do not
regenerate to make a test pass without deciding which of the two it is.

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
- The crate allows `arithmetic_overflow` and `unconditional_panic`: rustc denies by default an overflowing operation it
  can see statically even in a branch no input reaches, while the filter is per input.
- Text uses `\n` on every platform. The manifest and a path are written through `path_for_toml`.

## The template lockfile

Generated crates build with `cargo check --locked` (later `cargo test --locked`) against `gust` and `gust-wgpu`
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
target directory; with distinct names the same control fails as it should. The m5.c1b runner must keep the unique name
per run.

Line endings do not matter to cargo (`--locked` accepts LF and CRLF) or to the tests: `lock_for` copies the template lock
byte for byte as it is checked out, and the round-trip test compares against that. Git stores the file with LF (the
repository checks out CRLF on Windows through `core.autocrlf`, like every other text file); a refresh below makes
cargo write LF. So the checked-out ending wins: after a refresh keep whatever ending the file had in the working tree
(CRLF on this machine) and let git normalize it, so that the diff shows only the content change. After a dependency
change in the root workspace:

```sh
cd tests/difftest/template
cp ../../../Cargo.lock Cargo.lock
cargo metadata --format-version 1 --offline > /dev/null    # prunes the lock to the template's packages
```

Review the diff before committing it.

## Tests

- `prng`: the SplitMix64 reference vector, pinned xorshift64* outputs.
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
- `emit` and `reparse`: exact text of a module, parenthesisation table, re-parse equality for every program of the
  corpus, the template manifest, the template lock against the root lock.
- `corpus`: FNV reference vectors, the committed `seeds.txt`, the hash covering the input data, the `MIN / -1` discard
  inside seeds 1 to 8.
- `tests/roundtrip.rs`: all 128 corpus programs (the float ones included, so the dialect accepts `f32` buffers,
  arithmetic, comparisons and casts as emitted) in one crate pass `cargo check --locked`; a crate with a type error
  and a crate with a module the `#[gpu]` validator rejects (`while`) each fail with the expected message (the check is
  live). It runs `cargo check` directly, not through the repository's cargo lock wrapper, because the test itself runs
  inside the outer locked `cargo test` gate, which already holds the slot; its generated crates use their own target
  directory under `CARGO_TARGET_TMPDIR`.
