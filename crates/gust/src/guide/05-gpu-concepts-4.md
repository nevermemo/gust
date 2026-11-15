```

The capability a build has is named in `GUST_PROFILE` after its target, as in `native+subgroups` (the strict form that
ships) or `dev+subgroups` (the run-time-compile form the tests use); [chapter 6][ch6] says what the profile names mean.
Evidence for subgroup kernels is narrow: the RTX 5090 (subgroup size 32) and Mesa lavapipe in CI (size 8), and no browser
has run one. In the browser `web+subgroups` is refused at build time, so a web build runs the portable body.

## Floats

A float result of a kernel is not promised to be bit-identical to the CPU twin's. A device may fuse a multiply and an add
into one rounding, or flush a subnormal; SEMANTICS section 4 states the bounds and the cases that are left as documented
limitations (the project's D39 policy: such differences are written down, not chased). Compare floats with a tolerance,
and use exact comparisons where every value is an integer or a power of two.

Next, [chapter 6][ch6] and [chapter 8][`crate::guide::m9_language`]; the host halves of G5, G6 and G10 are in
[`gust_wgpu::guide::gpu_concepts_running`][ch5-running].
