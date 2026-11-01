# Differential program generator (`tests/difftest`)

Status: slices m5.c1a (the integer core), m5.c1a2 (floats v1, the exact-interval subset of R1-4) and m5.c1b (the
batch runner `difftest-run`). The crate generates scalar integer, boolean and `f32` kernels from a seed, evaluates them
with an independent interpreter that tracks the defined domain (and, for floats, an interval of every value a
conforming WGSL implementation may hold), writes them as a `#[gust::gpu]` crate, and `difftest-run` builds a batch of
them, runs every program on the interpreter, the CPU twin and the GPU, and reports what disagreed. The crate itself is
std only and has no GPU code: the GPU code lives in the generated run crate, which depends on `gust` and `gust-wgpu`.
Shrinking is the later slice m5.c1c. A unit test is evidence about the generator, the interpreter, the emitter and the
classifier only; a GPU claim needs a `report.json` of a `gpu` run, whose `mode`, `adapter` and
`programs_executed` say what ran.

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

The committed corpus (seeds 1 to 8, 128 kept programs, 507 discarded, format 3): 33 of the 128 modules have an `f32`
output (2 to 8 per seed), and the float discards are `float-domain` 93, `unboundable` 31, `comparison-overlap` 15 and
`float-construct-not-in-v1` 2 (the per-seed numbers are in `seeds.txt`). Float programs are therefore not rare among
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

Format history: 1 the integer core (m5.c1a); 2 the `f32` program space (m5.c1a2); 3 the condition and literal biases of the
float space (m5.c1b). Why 3: the executed-operator coverage of the batch (below) showed that under the format-2 float
space a third of the `if` conditions were a bare `true` or `false` and only 2 of the 55 evaluated `if` statements of
seeds 1 to 8 were taken by some elements and skipped by others, so most branches were dead or one-sided over the data.
Format 3 never writes a bare literal condition, makes a condition often a comparison of an input element with a value
of the data (for `f32` the midpoint of two elements, so the intervals stay disjoint) or a test on the element index, and
makes one float-space literal in eight a value that an input holds. The `Space::Integer` decode is untouched (the
m5.c1a pins still pass). Measured on seeds 1 to 8, format 3: 94 of 100 `if` statements evaluated, 58 of them with both
sides taken; `&&`/`||` right operands evaluated 8604 times and skipped 4584 times.

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
difftest-run --seeds 1-2 [--cpu-only] [--report PATH] [--target-dir DIR] [--repo-root DIR]
difftest-run --refresh-lock [--repo-root DIR]
```

`--seeds` takes a range, a list or both (`1-4`, `3`, `1,3,5-6`); the report goes to stdout without `--report`. The
target directory is `--target-dir`, else `CARGO_TARGET_DIR`, else `tests/difftest/target`. Exit status: 0 every program
agreed, 1 at least one disagreement (the report lists them), 2 the harness itself failed (a build error, a stale template
lockfile, a result file missing or malformed). A random-seed mode (`--seeds random:N`) is not implemented; the seeds are
always named, so a failure is reproducible from the report.

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
| `transport` | the interpreter output stored in the result file is not the one the harness computed: a bug of the harness itself |

WGSL validation: every kernel's first dispatch makes wgpu create the shader module and the compute pipeline with the
default capabilities (never `Capabilities::all()`); a validation error panics, is caught and becomes a `gpu-error`. So
`wgsl_validated` is true exactly when the first dispatch succeeded; the harness does not compile each kernel to WGSL
a second time (that would double the slangc time, see the cost table).

### report.json

Keys: `tool`, `format`, `mode` (`gpu` or `cpu-only`), `passed`, `seeds`, `programs_requested` (16 per seed),
`programs_executed` (the result files whose runner finished; equal to `programs_requested` on a pass), `adapter`
(`null` for cpu-only; otherwise name, type, backend and driver of the adapter wgpu chose), `per_seed` (`programs`,
`attempts`, `discarded`), `discards`, `mismatch_count`, `mismatches`, `coverage` and `timing` (generation, interpreter,
cargo build and run seconds, twin and GPU first and warm dispatch times).

`discards` lists `total`, `by_reason` (the labels of the tables above, per run, summed over the seeds) and `by_class`,
which maps the labels to the four classes of R1-4 plus one: `domain` (the integer labels, `float-domain` and `static`),
`comparison-overlap`, `unboundable`, `float-construct-not-in-v1`, and `no-input-read` (a fifth class, which the plan does
not name; a program without an input read is a constant, and the generator discards it). A label the classifier does
not know makes the run fail, so a new label cannot be silently counted under another class.

A `report.json` is deterministic for given seeds and mode except for `timing` and `adapter`; compare fields, not files.

### Coverage

`coverage` is measured by running the interpreter again with counting on, over the kept programs (so the domain filter
is already applied): for each operator and type (`+ u32`, `- f32`, `&& bool`, `as i32`, `for`, ...) the number of
programs that contain it in the text, the number of programs in which it is evaluated at least once, and the total number
of evaluations; the operators that appear in the text of some program but were never executed (`never_executed`); the `if` statements (in the
text, with a constant condition, evaluated, taken, with the else side taken, and with both sides taken by some element);
and how often the right operand of `&&`/`||` was evaluated and skipped. A written operator in dead code is not
counted as covered. A run of seeds 1 to 8 at format 3 executes every operator in its text (`never_executed` is empty).

### Short circuit

`&&` and `||` must not evaluate the right operand when the left decides, on every party. The test
`tests/short_circuit.rs` uses `out[i] = (i < 24) && (a[i + 40] == 7)` (and the `||` twin) over 64 elements with a
64-element input: for the elements 24 and above, `a[i + 40]` would read out of bounds. The interpreter's domain tracking
accepts the program only because it skips the right operand; the twin (Rust `&&`) skips it; the GPU run must not fail
and must give the same values (CPU run: `cargo test`; GPU run: `-- --ignored`). The GPU test is Gpu tier; no gate runs it yet, and m5.c2's `check-difftest` runs it. Measured on the RTX 5090: both programs run,
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
nightly `gpu-software` and `check-full` keep all 8 seeds, and shrinking stays local. Not measured, so not claimed: the
GitHub runner and lavapipe (slangc time will dominate there too: 8 seeds are 128 kernels, about 68 s of slangc here), the
`unit` job's cold build on a runner, and the shrink half of S5.3 (time per candidate, candidates used), which belongs to
m5.c1c.

### Mutation checks

Each of these was applied by hand, observed to fail, and restored byte for byte (`cmp`) before review. Each also has
permanent tests that fail for the same reason (the second was re-run against the unit tests: 7 fail).

| Mutation | Observed | Permanent test |
| --- | --- | --- |
| delete one result file before classification | `expected exactly 16 result files, found 15`, exit 2 | `classify::tests::exactly_k_result_files_are_required...` |
| a wrong literal in the emitter (`u32` literals plus 2), in a `--cpu-only` batch | 8 of the 16 programs of seed 1 reported (`twin-vs-interpreter`, `twin-error`), each named, exit 1 | `emit::tests` (exact text, re-parse), `corpus::tests`, the integer-space pin |
| compare the GPU `f32` by bits (or the twin by value) | the `-0.0` tests fail | `classify::tests::the_gpu_f32_is_held_by_value...`, `the_twin_is_compared_bit_for_bit...` |
| one package name for different batches | the uniqueness tests fail | `harness::tests::every_distinct_batch...`, `two_crate_dirs_with_one_package_name_are_refused` |

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
- `emit` and `reparse`: exact text of a module, parenthesisation table, re-parse equality for every program of the
  corpus, the template manifest, the template lock against the root lock.
- `corpus`: FNV reference vectors, the committed `seeds.txt`, the hash covering the input data, the `MIN / -1` discard
  inside seeds 1 to 8.
- `tests/short_circuit.rs`: see Short circuit above (a CPU test, an `#[ignore]`d GPU test, and the check of the WGSL guard). It builds its run crates in `CARGO_TARGET_DIR` when that is set, else under `CARGO_TARGET_TMPDIR`.
- `tests/roundtrip.rs`: the run crate of seeds 1 to 8 (128 kernels plus the generated test module) passes
  `cargo check --tests --locked`; all 128 corpus programs (the float ones included, so the dialect accepts `f32` buffers,
  arithmetic, comparisons and casts as emitted) in one crate pass `cargo check --locked`; a crate with a type error
  and a crate with a module the `#[gpu]` validator rejects (`while`) each fail with the expected message (the check is
  live). It runs `cargo check` directly, not through the repository's cargo lock wrapper, because the test itself runs
  inside the outer locked `cargo test` gate, which already holds the slot; its generated crates use their own target
  directory under `CARGO_TARGET_TMPDIR`.
