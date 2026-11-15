```

## Capability variants and subgroups

Some devices offer more than the portable floor. A `#[variant]` function beside a kernel is an alternative body for a
device with a capability; the `#[kernel]` function stays the portable body, and the variant has the same signature
token for token and no options of its own. A build uses the one body with the largest `requires` set that its
capabilities contain, and the CPU twin and the `cpu` target always use the portable body. The one capability of this
release is `subgroups`: 21 `subgroup_*` functions (`subgroup_add`, `subgroup_ballot`, `subgroup_shuffle`,
`subgroup_size()`, `subgroup_lane()` and the rest), accepted only in the body of a kernel that requires it
(GUST-E0745), where every invocation of the subgroup reaches them (GUST-E0747), and in a kernel whose workgroup has a
multiple of 64 invocations (GUST-E0751). The lane count is a property of the pipeline, so a kernel reads
`subgroup_size()` and never assumes one. A crate's declared build profiles must each have a body for every kernel
(GUST-E0744).

G10, `variants`, tallies a stream into three cells. The portable body does up to three atomic operations per element;
the subgroup body combines the values of the subgroup first and lets one lane do the atomics:

```rust,no_run
# use gust::gpu;
