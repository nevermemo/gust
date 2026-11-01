# Differential program generator (`tests/difftest`)

Status: slice m5.c1a, the integer core. The crate generates scalar integer and boolean kernels from a seed, evaluates
them with an independent interpreter that tracks the defined domain, and writes them as a `#[gust::gpu]` crate. It has
no GPU code and no runner: the batch runner, floats and shrinking are the later slices m5.c1b, m5.c1a2 and m5.c1c of
the M5 plan. Nothing here is GPU evidence; a passing unit test is evidence about the generator, the interpreter and the
emitter only.

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

Types `u32`, `i32`, `bool`. One to three read-only integer input buffers and one integer output buffer; the output
element type is the result type, and a `bool` result is stored as `u32` 0 or 1 (there is no `GpuPod` for `bool`, so
no `bool` buffer). The kernel is `let i = id.x; if i < out.len() { <statements>; out[i] = <result>; }` over 1, 63, 64,
65, 257 or a random count below 4096 elements, workgroup size 64.

- Statements: `let` and `let mut` with an explicit type, assignment, `if`/`else`, and `for v in 0u32..N` with `N` a
  literal in 1 to 8.
- Expressions, depth at most 6: literals biased to boundaries (0, 1, -1, `i32::MIN`, `i32::MAX`, `u32::MAX`, powers
  of two, masks), locals, `i`, `a[i]` and other in-bounds reads (`a[(e) % a.len()]`, `a[a.len() - 1 - i]`),
  `+ - * / % & | ^ << >>`, unary `-` (`i32` only) and `!`, comparisons, `&&`, `||`, and `as` between `u32` and `i32`.
  The right operand of a shift is always `u32`. `bool as u32` is not generated (ARCHITECTURE.md: it is unproven).
- A divisor is never a literal zero, in any branch: the `#[gpu]` validator rejects `DivisionByLiteralZero` even where
  no input reaches (a test scans 20000 decoded programs for one). An all-ones divisor (`-1` for `i32`) is biased in,
  and then the dividend is often a plain read of an `i32` input, so that `i32::MIN / -1` is reached; seed 2 of the
  corpus discards one program for it. Over seeds 1 to 300 the discard fires in 23 seeds, about one seed in thirteen.
- No floats (m5.c1a2), no structs, helpers, `Option` or atomics.

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
- `generator`: determinism per seed, different seeds differ, batch size and discard accounting, every attempt and every
  truncated stream is well typed, operator coverage of the corpus for both types, no literal zero divisor in 20000 decoded
  programs, no static constant outside the domain and no literal shift of 32 or more in the 1600 accepted programs of
  seeds 1 to 100, the static discard being live.
- `emit` and `reparse`: exact text of a module, parenthesisation table, re-parse equality for every program of the
  corpus, the template manifest, the template lock against the root lock.
- `corpus`: FNV reference vectors, the committed `seeds.txt`, the hash covering the input data, the `MIN / -1` discard
  inside seeds 1 to 8.
- `tests/roundtrip.rs`: all 128 corpus programs in one crate pass `cargo check --locked`; a crate with a type error
  and a crate with a module the `#[gpu]` validator rejects (`while`) each fail with the expected message (the check is
  live). It runs `cargo check` directly, not through the repository's cargo lock wrapper, because the test itself runs
  inside the outer locked `cargo test` gate, which already holds the slot; its generated crates use their own target
  directory under `CARGO_TARGET_TMPDIR`.
