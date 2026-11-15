# variants (gallery G10)

One kernel, two bodies. `stats::tally` takes a stream of `u32` values and keeps three cells: their sum, how many are
over 100, and their maximum, each with an atomic operation. The **portable body** is the `#[kernel]`: every invocation
does its own `atomic_add`, `atomic_add` and `atomic_max`. The **subgroup body** is the `#[variant(of = tally,
requires(subgroups))]` beside it: the lanes of a subgroup first combine their values (`subgroup_add`, `subgroup_count`
of a condition, `subgroup_max`), and one lane (lane 0) does the three atomic operations for the whole subgroup, so the
cells see one operation per subgroup instead of one per element. Both bodies have the signature of the kernel, token for
token.

A build uses one body, chosen by `GUST_PROFILE` (D138): the one with the largest `requires` set that the capabilities of
the profile contain, else the portable body. `main` prints which one it was built with, from the descriptor of the
kernel:

```text
variants: body=portable, ..        # the default build (strict profile native): every adapter
variants: body=subgroups, ..       # GUST_PROFILE=native+subgroups: an adapter with subgroup operations
```

`cargo gust build` exports the artifact of **every** body, so a strict build of either profile runs from the committed
`gust-artifacts/` with no Slang (`cargo run --release` for the portable body, `GUST_PROFILE=native+subgroups cargo run
--release` for the subgroup one; in PowerShell set `$env:GUST_PROFILE` first). The program also prints the subgroup size
range of the adapter (a property of the pipeline, never assumed by the kernel) and a line `adapter: <name>`.

## What an adapter without subgroups does

The portable build runs on every adapter. A build for `native+subgroups` opens its device with the capability
(`HeadlessDevice::new()` uses the capabilities of the build), and an adapter that lacks it cannot open one: `new()`
returns `Error::MissingCapability`, which `main` prints (`variants: no device for body=subgroups: ..`) before it exits
with status 1. The program never runs a subgroup body on a device that was not opened with the capability, and never
gives a wrong result there; choosing the body is a build-time decision (D23), so there is no run-time fallback to the
portable body.

## Tests

`cargo test` runs the kernel on the CPU twin. A subgroup body has no CPU form, so **the twin always runs the portable
body**, and the test checks it against a host reference made of plain iterators, exact (all three cells are integers
below 2^32), for 1, 7, 63, 64, 65, 127, 129, 1000 and 100 000 values (the workgroup is 64 and subgroups are 8 to 128
lanes, so the last workgroup and the last subgroup are partial in most of them), plus one hand-computed case and a guard
that the data has values on both sides of the limit.

Before it prints, `main` also checks the sizes 1, 37 and 100 003 and then 100 000, each exact against the host
reference. 37 and 100 003 are not multiples of 8 or 32, so on subgroups of those sizes the last subgroup holds both real
values and lanes past the end of the data (which hold 0), the case the placement rule is about. The output line is
unchanged and reports the 100 000 values.

The subgroup body is checked on the GPU only, by `main` under `native+subgroups`, against the same host reference; the
portable body is checked on the GPU by `main` of the default build. So the evidence is: twin (portable body) equals the
host reference in `cargo test`; the GPU's portable body equals the host reference; the GPU's subgroup body equals the
host reference. The two GPU bodies and the twin are therefore equal through the reference at these sizes; no run
compares the subgroup body with the twin directly, because the two can not be in one build (a build has one body).
`cargo xtask check-gallery` runs both GPU builds.

The three results do not depend on the order in which the atomic operations happen (a sum, a count and a maximum), which
is why the twin, the GPU's portable body and the GPU's subgroup body are exact and equal whatever the subgroup size.

An invocation past the end of the data (the last workgroup is partial) holds 0 in the subgroup body and still takes part
in the subgroup operations, which the placement rule requires; a 0 changes none of the three results. The subgroup body
follows the discipline of D140: every subgroup operation is the whole value of a `let` at the top level of the body and
the bounds guard is a statement `if` around the load.

Commands, from this directory (the tools are set up as for `hello-gust`):

```sh
cargo test                                          # the CPU twin (portable body) against the host reference
cargo gust build                                    # writes the artifacts of both bodies (commit them)
cargo gust check                                    # fails when gust-artifacts/ no longer matches the sources
cargo run --release                                 # the portable body, any Vulkan compute adapter
GUST_PROFILE=native+subgroups cargo run --release   # the subgroup body, an adapter with subgroup operations
```
