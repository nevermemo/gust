# state-machine (gallery G7)

A tokenizer: a stream of ASCII symbols is cut into records of 16, and one invocation scans one record with two
finite-state machines at once, one that finds words (runs of letters) and one that finds numbers (runs of digits). Each
record gets how many words and numbers it holds and the length of the longest of each. A token that runs over the end of
a record ends there, because every record starts with both machines outside a token (the host reference does the same).

What it shows, in one kernel, the constructs of milestone 9 (`docs/SEMANTICS.md` sections 9, 10, 13, 14 and 16), which
run on the GPU and on the CPU twin:

- an **enum with `match`**: `State` (outside or inside a token) and `Class` (what a symbol is), and a `match` on the
  state in `transition`;
- a **tuple**: `transition` returns `(State, u32)` and `totals` returns `(u32, u32)`, and the caller takes them apart
  with `let (next, started) = ..`;
- a **trait with static dispatch**: `Classify` (implemented by `Ascii`) and `Rules` (implemented by `Words` and
  `Numbers`);
- a **generic method with trait bounds**: `Machine::feed<R: Rules, S: Classify>`, instantiated twice (words and
  numbers);
- **methods**: `feed` takes `&mut self` and is called on a `let mut` local, `totals` takes `&self`;
- a dispatch whose input is longer than its invocation count: `Independent(&text)` is the call-site statement that the
  text is not one element per invocation (16 symbols per record).

## Limits of the dialect that show in the source

Each one is a compile error or a missing form that the source works around, not a surprise at run time:

- An enum is not a field of a struct (GUST-E0211), so `Machine` keeps its state as the integer `State as u32` and `feed`
  turns it back into a `State`.
- A struct literal is accepted only as a `let` initializer or the right side of an assignment (GUST-E0810), so the
  arguments of `feed` are bound to locals first.
- A struct needs at least one field (GUST-E0105, empty struct), so the rule types `Words` and `Numbers`, which carry no
  data, each hold an unused `tag: u32`. Field-less enums would also satisfy the trait; structs with a tag were kept so
  that the two rules read as the types they are.
- There is no `RangeInclusive::contains` (`(48..=57).contains(&c)`), so `Ascii::class` writes the comparisons out (`c >=
  48 && c <= 57`) and carries `#[allow(clippy::manual_range_contains)]`, so that the crate still passes clippy with `-D
  warnings`.

## Tests

`cargo test` (CPU twin only): the kernel on `CpuDevice` against a host reference that never runs a state machine (it
splits each record at every symbol outside the token set), exact, for 1, 63, 64, 65 and 1000 records (the workgroup is
64, so the last workgroup is partial in all but 64); two hand-counted records; one guard that the generated stream has
words, numbers and tokens that run over a record boundary. `main` runs 1000 records on the GPU and asserts every record
against the same host reference before it prints.

All results are integers, so every comparison is exact and there is no tolerance. The arithmetic stays far inside the
defined domain of `docs/SEMANTICS.md` section 3 (counts of at most 16), so the D39 edge differences between the GPU and
the twin (overflow, division by zero, out-of-range casts) cannot occur here and none is claimed away.

Commands, from this directory (the tools are set up as for `hello-gust`):

```sh
cargo test                        # the CPU twin against the host reference; no GPU, no Slang
cargo run --features gust-dev     # development: compiles the kernel at run time through Slang, needs an adapter
cargo gust build                  # compiles the kernel with Slang and writes gust-artifacts/ (commit them)
cargo gust check                  # fails when gust-artifacts/ no longer matches the sources
cargo build --release             # no Slang needed: the kernel is embedded from gust-artifacts/
cargo run --release               # needs a Vulkan compute adapter
```

It prints the line `state-machine: 1000 records of 16 symbols, ..` with the totals (the stream is generated from a fixed
seed, so the numbers are the same on every run), then a line `adapter: <name>` with the GPU it ran on.