# Benchmarks

Status: **no result is published.** This page holds the method, the rules a result must pass before it may be shown,
and the known limits of the measurements. It has no headline number, no ranking and no speed claim; whether the project
makes one is an open decision. The tables near the end are generated from filed results by
`cargo xtask check-bench --render`, and are empty until a result passes the rule.

<!-- bench-header:begin -->
> Every comparator (the hand-written WGSL, the rayon CPU backend and the rest) runs inside GUST's harness with the settings stated here, not in its authors' recommended configuration; corrections are welcome as issues against this repository.
>
> A figure describes one adapter, one driver, one operating system and one commit, measured as the methods below say. It is not a ranking, and this page makes no speed claim.
<!-- bench-header:end -->

## Workloads

The harness lives in `bench/` (a workspace of its own; see `bench/README.md` for every command). Eight families (W1 to W7 and W9), each
checked against a reference on every sample; a sample that fails its check fails the cell.

| id | what it is | correct when |
|---|---|---|
| W1 vector-add | `out = x + y`, f32, memory-bandwidth bound | bit-exact |
| W2 saxpy | `out = a * x + y`, f32, memory-bandwidth bound | within the derived bound below (two roundings of the backend's unit roundoff times `|a x| + |y|`) |
| W3 Horner polynomial | degree 4, 16 or 64, f32, arithmetic bound (`2 d` flops per 8 bytes) | within the derived bound below |
| W4 u32 atomic histogram | 256 or 65536 bins, uniform or Zipf keys, atomic-contention bound | every bin equals the serial count exactly |
| W5 particle step | 16 explicit-Euler steps of every particle of a struct buffer (a 28-byte std430 element: nested position and velocity pairs of `f32`, a `f32` mass, a `u32` tag, an `i32` charge), 2^16 or 2^20 particles, one invocation per particle | `mass`, `tag` and `charge` bit-exact; each of the four float fields within its own derived bound below |
| W6 staged pipeline | six kernels of `signal-pipeline`'s `transform` in a chain, f32, 2^20 or 2^22 elements, a buffer between stages; two GUST forms (`W6-graph`: `StagedGraph`; `W6-hand`: six dispatches, no graph) | the final stage within the composed derived bound below; every stage within its own bound in a unit test |
| W7 sum | `u32` (wrapping) and `f32`, 2^20 to 2^27 values, one output word, a reduction; GUST runs `gust-algo`'s public sum | `u32`: the wrapping sum exactly; `f32`: within the bound of the summation order the backend runs (below) |
| W9 naive matrix product | `C = A B` of two `N x N` f32 matrices, `N` from 256 to 2048, one output entry per invocation (the tiled form is not in this set) | every entry within the derived bound below |

**The unit roundoff, per backend.** Every f32 add and multiply is modelled as `fl(a op b) = (a op b)(1 + delta) + eta`
with `|eta| <= 2^-126` and `|delta| <= u`, where `u` is the backend's: `u = 2^-24` for the CPU backends (`serial`, rayon,
`gust-twin`: host arithmetic, IEEE round to nearest) and `u = 2^-23` for every GPU backend (GUST, its SPIR-V route and
the hand WGSL). `docs/SEMANTICS.md` section 4 lets a GPU add or multiply round to either neighbour, within 1 ULP, which is
a relative error of up to `2^-23`; `gust_algo::bound::C_GPU` encodes the same constant. The reference carries both bounds
and the check picks the one of the backend's kind (`Rounding`). W3's bound was first written with `2^-24` for the GPU
backends too; it was corrected with W7 and W9 (no backend had failed it, so this widens nothing that failed, it makes the
stated model the one the semantics grant).

**The W2 bound.** The kernel does one multiply and one add, `fl(fl(a x) + y)`. With the unit roundoff of "The unit roundoff,
per backend" and `p = a x` (exact in f64: two 24-bit significands) the computed value is `(p (1 + d1) + y)(1 + d2) + eta`,
`|d1|, |d2| <= u`, so against the exact `p + y` the error is at most `u (1 + u) |p| + u |p + y|`, which is at most
`((1 + u)^2 - 1) (|p| + |y|)` (the product form: two operations, `(1 + u)^2 - 1 = u (2 + u)`), plus the underflow term. A
fused multiply-add rounds once and has only less error. The f64 reference `a x + y` has one rounding, `2^-53 (|p| + |y|)`.
The check is

```text
|out - ref_f64| <= (((1 + u)^2 - 1) + 2^-53) (|a x| + |y|) + 2 * 2^-126
```

with `u = 2^-24` for the CPU backends and `u = 2^-23` for every GPU backend. It replaces an allowance of two f32 epsilons
(`2^-22`) of the same magnitude, which assumed round to nearest for every backend. The new factor is about `2^-23` for a CPU
backend (half of the old one) and about `2^-22` for a GPU (the old one, plus second-order terms), so the GPU's allowance is
not materially widened and the model is the one W3, W7 and W9 use. Tests show a plain and a fused result
inside the bound under both units (on inputs of their own with full 24-bit significands and a scale of 1.1, where at least
a tenth of the elements differ between the plain and the fused result and the worst plain error is a real fraction of the
bound), a value between the two bounds passing for a GPU and failing for a CPU backend, and a result one f32 step past
the bound failing; the bound is never widened to pass. The harness's own W2 inputs are exactly representable by
construction (`x` a multiple of 2^-11, a scale of 2.5, `y` a multiple of 2^-12), so every W2 result is exact and a W2 run
cannot observe a rounding or a contraction: the bound is checked on every sample, and it is the unit tests above that
show it holds when results do round.

**The W5 bound.** One invocation takes one particle and runs 16 steps in registers (`vx += fx dt; vy += fy dt; px += vx dt;
py += vy dt`, the position from the new velocity, `dt = 0.016`), then writes the particle back, so one dispatch is the
whole case and every dispatch of a cell writes the same output from the same input. `mass`, `tag` and `charge` are
copied; the check compares them as bits. The four float fields have no closed form that is both simple and tight over 16
compounded steps (a product-form `gamma` of the whole 16 steps would be rigorous but loose and growing), so the bound is the
forward error of the expression tree carried numerically along the f64 trajectory (`workloads::particles::exact_and_bound`;
the exact trajectory differs from the f64 one by about 1e-14 relative, which the factor `1 + 1e-9` below covers). With the
model of "The unit roundoff, per backend", a value is a pair `(v, e)`, the computed value within `e` of the exact `v`,
and

```text
add:  v = a.v + b.v    e = (a.e + b.e)(1 + u) + u |v| + eta
mul:  v = a.v * b.v    e = (|a.v| b.e + |b.v| a.e + a.e b.e)(1 + u) + u |v| + eta
```

The inputs are exact. The recurrence as written is the bound of the order the CPU adds in: Rust and the twin never
reassociate floats, and that is the `Rounding::Nearest` bound. A GPU may regroup a chain of additions with no accuracy
condition (`docs/SEMANTICS.md` section 4, WGSL 15.7.5), so for `Rounding::Either` every addition chain is bounded over
every grouping: `vx_k = vx_0 + c + ... + c` (`k + 1` terms, `c = fx dt`; likewise `vy_k`) and the final
`px = px_0 + vx_1 dt + ... + vx_16 dt` (17 terms; likewise `py`). For a chain `t_0 + ... + t_k` whose terms are each
computed within `e_i` (Higham, section 4.2, the product form, valid for any order and any binary grouping):

```text
e <= (1 + u)^k sum e_i + ((1 + u)^k - 1) sum |t_i| + k (1 + u)^k eta
```

(`particles::Chain`). Each `vx_k` is such a chain, and its bound enters the product `vx_k dt` that is a term of the
position chain. The order-free bound is never below the bound of the written order (a test shows it for every particle
of a run). Tests evaluate the step with every chain summed in reverse order, as a pairwise tree and largest term first,
and show each inside the GPU bound while differing from the written-order result. `dt` is the f32 nearest 0.016 on a CPU backend (a Rust literal, and the reference uses the same
f32); a GPU backend's shader compiler is only required to give that f32 or a neighbour, so the model gives the constant an
error `dt 2^-23` there. The check, per float field of every particle, is

```text
|out - ref_f64| <= (e_f32(u) + e_f64) * (1 + 1e-9)
```

with `u = 2^-24` for the CPU backends (`Rounding::Nearest`: serial, rayon-1, rayon-N, gust-twin, gust-cpu) and `u = 2^-23` for
every GPU backend (`Rounding::Either`: gust-gpu, gust-gpu-spirv, the hand WGSL), `eta = 2^-126`, and `e_f64` the same
recurrence with `u = 2^-53`, no `eta` and an exact `dt` (the reference's own error). The factor `1 + 1e-9` covers the
f64 evaluation of the bound itself. A fused multiply-add rounds once, so it has only less error than the multiply and
add it replaces, and a backend that contracts is inside the bound (a test shows plain and fused steps inside it under
both units); tests show a value between the two bounds passing for a GPU and failing for a CPU backend, a result one f32
step past the bound failing, and the bound tight (a small fraction of the field's scale, so it is not vacuous). The bound
is never widened to pass.

Limits of the W5 measurement. The example's second output (`energy`) is not part of this workload. The hand WGSL has the
same struct layout as GUST's (`shaders/particles.wgsl`); a test reads the layout that `GpuPod` derives for GUST's struct (size, alignment,
stride and each field's offset) and the field order of the shader text and asserts they agree, and another test, a separate
check, reads the committed Slang reflection of the step kernel (`bench/gust-bench-pre/gust-artifacts`) and checks stride 28 and the five
offsets. The GUST GPU backend reads the struct buffer back as `Particle` values and converts them to words inside its
end-to-end sample (a second allocation and a per-field copy), which the hand WGSL's raw word read-back does not do. So the
end-to-end timing of the two is not like for like, and the renderer shows no GUST end-to-end ratio, overhead row,
end-to-end bandwidth or GUST break-even for W5 (`Workload::end_to_end_comparable`, a property of the workload, with the
reason printed on the page); the dispatch timings stay. GUST's W5 end-to-end milliseconds include the struct conversion copy (the typed `Vec<Particle>` read-back converted to words), which is the reason the end-to-end ratio is hidden. `gust-cpu` and `gust-twin` convert the same way, so their W5
end-to-end times carry that copy too (they are in no GUST/hand pairing, but they do enter the rayon comparison). The quick check runs 2^16 particles with all 16 steps (the step count is part of the kernel, and a single step
would not exercise the compounding the bound is about). `wgsl-gust-shader` (GUST's generated WGSL run by the hand
harness) does not support W5, like W7 and W9.

**W6, the staged pipeline.** `examples/signal-pipeline`'s `transform` is six statements in one kernel. W6 makes each statement a
kernel of its own (`bench/gust-bench/src/workloads/signal_pipeline.rs`: `center`, `amplify`, `shift`, `rescale`, `energy`,
`combine`), `f32`, one invocation per element, a buffer between every two stages, 2^20 and 2^22 elements (quick: 2^20), so a
run is a chain of six dispatches and the workload is about what sequencing them costs. The plan's W6 row names six kernels;
the example has one kernel, `transform`, with six statements, and a one-node graph has no edges and nothing to sequence. W6
makes each statement a kernel, the only reading in which a graph has more than one node. The inputs are the
example's five periodic streams in the example's ranges. The traffic of the six passes is 11 reads and 6 writes of 4 bytes per
element (68 bytes), which is what the batch bandwidth columns divide by. The end-to-end bandwidth columns (GPU
and rayon-N alike) divide by 24 bytes per element instead: what an end-to-end sample moves over the bus is five uploads and
one read-back (20 + 4 bytes), and a fused CPU loop touches the same 24; the intermediates never leave the device.

There are two GUST forms, reported as two workload ids that share the kernels, the inputs and the bound:

| id | what runs | what one timed sample includes |
|---|---|---|
| `W6-graph` | six typed `Call`s added to a `StagedGraph` (`graph.run`, derived edges), `device.execute_graph` | building the graph, validating it (the edges are checked), encoding six compute passes into one command buffer, one submit and the wait. End to end adds five upload nodes (`graph.write`) and one read-back node (`graph.read`) in the same graph, each upload through a staging buffer the graph owns |
| `W6-hand` | the same six calls as one `dispatch_batch` of six `BufferDispatch`es, no graph | the six passes (one compute pass per dispatch) in one submit and the wait; no graph build, no validation of edges. End to end adds five `write_typed_buffer` uploads and the typed read-back |

Both forms of GUST and the hand WGSL do one submit with six compute passes per run; they differ in the host work around it.
The hand WGSL (`bench/shaders/signal_pipeline.wgsl`, six entry points of one module, one three-binding layout, one bind group per
stage, dispatched in hand order) is the baseline of both forms: `wgsl-prebuilt` builds its bind groups once, `wgsl-percall`
builds one per pass per run. The graph form has no GPU timestamps (`execute_graph` records none), so it has no batch
figure and no batch bandwidth (and it is outside the kill rule, see "The kill rule"); `W6-hand` has them: for W6,
`gpu_single` brackets one six-pass run and `gpu_batch_per_dispatch` is per six-pass run (one timestamp pair around the
`6 x batch` passes, divided by `batch`), not per compute pass. Both forms are
`same_kernel()` (the same six kernels in the same order, on every path) and `end_to_end_comparable()` (every buffer is
typed `f32` on every path, nothing is converted), so the renderer pairs them with the hand WGSL, and the pairing says
`W6-graph` includes the graph's per-call host work and `W6-hand` does not; the two ids are never merged into one row. The CPU
backends have no `StagedGraph`: `gust-cpu` and `gust-twin` run the six kernels one after another through `CpuDevice` (the
`W6-hand` form only), and `rayon-1`, `rayon-N` and `serial` compute one element's six stages at a time
(`signal_pipeline::element`, each stage rounded to `f32`, so the result is the same one a stage-by-stage run gives); none of them
is in the `W6-graph` table (so that table has no rayon-N row, and the renderer says so instead of stating a break-even).
`wgsl-gust-shader` does not run W6, like W5, W7 and W9, and neither do the order probes `wgsl-prebuilt+gust-device` and
`wgsl-prebuilt+gust-setup` (both need the timestamp-first resource order of GUST's single-kernel `prepare`, which W6's
staged path does not build, and `gust-setup` also needs GUST's generated shader). CubeCL is not in W6.

**The W6 bound.** Every operation is modelled as `fl(a op b) = (a op b)(1 + delta) + eta`, `|delta| <= u`, `|eta| <= 2^-126`,
with `u = 2^-24` for a CPU backend and `u = 2^-23` for a GPU (the unit roundoff section above); the division uses `u_div = u`
on a CPU (0.5 ULP) and `u_div = 2.5 * 2^-23` on a GPU (`docs/SEMANTICS.md` section 4, "within 2.5 ULP"; the example's `5u`).
The forward error of the tree is carried along the f64 trajectory as a pair `(v, e)`, the computed value within `e` of the
exact one:

```text
add:  e = (a.e + b.e)(1 + u) + u |v| + eta                     (a subtraction is the same)
mul:  e = (|a.v| b.e + |b.v| a.e + a.e b.e)(1 + u) + u |v| + eta
div:  e = ((a.e + |v| b.e) / (|b.v| - b.e)) (1 + u_div) + u_div |v| + eta
```

The five inputs are exact, and `0.25 * normalized` is a multiply by a power of two, exact (error `0.25 e`). The stages
are stored: each stage is a dispatch of its own, so a value is rounded to `f32` between any two operations, and no stage has
an addition chain longer than two, which is why the bound is not in the order-free form of W5 (nothing to regroup across a
stage, and a fused multiply-add rounds once, so it has only less error than the multiply and add it replaces). Each
stage's computed value is its operation applied to the previous stage's computed values, which is exactly what the
recurrence propagates. The reference is the same six operations in f64 with `u = 2^-53` (the division correctly rounded
too) and no `eta`. A stage's bound is the sum of the f32 model's final `e` and the f64 reference's, times `1 + 1e-9` (the
f64 evaluation of the bound itself). A timed sample checks the final stage against that composed bound
(`signal_pipeline::exact_and_bound`, the CPU bound or the GPU bound by the backend's `Rounding`, through the same
`Judging` as W3 and W9); a unit test checks every stage separately against its own (`signal_pipeline::stage_exact_and_bound`,
on the hand WGSL and on both GUST forms, from the intermediate buffers). The bound is never widened to pass. Tests show
a correct result passing, a result just past the bound failing, a value between the GPU and the CPU bounds passing for a
GPU and failing for a CPU backend, a GPU rounding every operation and the division by directed rounding inside the GPU
bound at every stage, and the bound a small multiple of the roundings it covers (not vacuous). The quick check runs
2^20 elements of both forms, checked on every sample.

**The W3 bound.** Higham (Accuracy and Stability of Numerical Algorithms, 2nd ed., section 5.1)
bounds Horner's rule by `gamma_{2d} S` with `S = sum |c_k| |x|^k`. The reference is Horner's rule in f64, which obeys the
same theorem with the f64 unit roundoff, so the check is

```text
|p - p_f64| <= (gamma_{2d} + gamma64_{2d}) * S + 2d * 2^-126
```

The last term allows underflow. A backend that fuses multiply-add is inside the bound (a test shows it) and a perturbed
result is outside it. The bound is never widened to make a backend pass.

**The W7 checks.** The `u32` sum is exact and is the wrapping one (modulo 2^32): `gust_algo::reduce::sum_wrapping` is
defined modulo 2^32 for every input, WGSL integer addition wraps, and the CPU backends use `wrapping_add`; the inputs are
32-bit hash words, so the true sum passes 2^32 by far and a lost or doubled element changes the word. The `f32` sum has
no single order, so each backend is judged by the bound of the order it runs, all of the form `gamma_K S` with
`S = sum |x_i|` (Higham, section 4.2: every value meets at most `K` roundings, whatever the grouping), plus the f64
reference's own error `(n - 1) 2^-53 S` (`gust_algo::bound::f64_reference_error` has the same formula; a test pins them
equal, and another that the f64 reference sum is exact for these inputs, so the first-order form is no looser than the
rigorous `gamma64_{n-1} S`):

```text
GUST's tree (gust-gpu, gust-gpu-spirv):  |s - s_f64| <= gust_algo::bound::f32_sum_bound_v1(n, 2^-23, S) + (n - 1) 2^-53 S
hand WGSL tree (wgsl-prebuilt/-percall): |s - s_f64| <= gamma_K(2^-23) S + (n - 1) 2^-53 S,  K = chain_1 + 8 + chain_2 + 8
serial, rayon-1, rayon-N (u = 2^-24):    |s - s_f64| <= ((1 + u)^{n-1} - 1) S + (n - 1) 2^-126 (1 + factor) + (n - 1) 2^-53 S
```

*GUST's tree.* `c = 2^-23` is the library's constant (`bound::C_GPU`). Which GUST schedule ran is not reported by the
public API (`limits::admits` is private): the harness repeats its test from the public device limits (workgroup memory
for the v1 kernels, the input within the storage binding, the first pass within the workgroup limit) and takes
`f32_sum_bound_v1` where it holds; where it does not it takes the larger of `f32_sum_bound` (v0) and the v1 bound. That
repeated test is an assumption of the harness (a test shows the branch and the fallback).

*The hand tree* has a known order, so it has its own bound (the plan's serial worst case still bounds it; a test shows
the tree bound is inside it, and this is a tightening the coordinator recorded). Pass one is a grid-stride reduction:
`blocks = min(ceil(n / 2048), max_compute_workgroups_per_dimension)` workgroups of 256 lanes, so the grid grows with `n`
(512 workgroups at 2^20, 65535 at 2^27 on an adapter whose per-dimension workgroup limit is 65535; the code takes the limit the device reports, so an adapter with a larger limit gets more workgroups and a smaller `K`). A lane adds `chain_1 = ceil(n / (256 blocks))` values one after the other, then
the 256 lane values of the workgroup are added in a tree of 8 rounds; pass two is one workgroup in which a lane adds
`chain_2 = ceil(blocks / 256)` partials and then the same 8-round tree. `K = chain_1 + 8 + chain_2 + 8` counts the first
add of a chain (into a zero accumulator, exact on IEEE hardware) too, the conservative reading of "either neighbour"; a
compiler's reassociation (WGSL section 15.7.5) does not lengthen a chain. K is 26 at 2^20 and, under that same 65535 limit, 281 at 2^27 (an adapter with a limit of 65536 or more gets `K = 280`).

*The host backends* (`serial`, `rayon-1`, `rayon-N`, `u = 2^-24`). The order of `serial` is sequential and of rayon
unknown, so they are judged by the serial worst case, which covers every order: `((1 + u)^{n-1} - 1) S` is Higham's
product form (`|prod(1 + delta_j) - 1| <= (1 + u)^k - 1` for every `k`), used everywhere in place of `gamma_{n-1}`: it is
never larger than `gamma_{n-1}`, exists for every `n` (`gamma_k` needs `k u < 1`, which fails above `n = 2^24`) and grows
with `n`. No tighter bound is invented for them. **Where the `f32` check is vacuous.** On `serial`, `rayon-1` and `rayon-N`
the factor is 0.065 at 2^20, 0.28 at 2^22, 0.65 at 2^23 and above 1 from 2^24 (1.7 at 2^24, then larger): from `n = 2^24`
the bound exceeds `S >= |s|`, so a result of `0.0` passes whatever the data, and 2^22 to 2^23 is weak (over a quarter of
`S`). The check on those three backends is therefore meaningful at 2^20 and 2^21 only. Correctness of the `serial` and
rayon cells at 2^22 to 2^27 rests on the same code passing at the small sizes (the code does not branch on `n`) and on
W7-`u32` being exact on the same loop shape (a different code path from the `f32` one, so this is supporting evidence, not coverage of the `f32` loop; a lost or doubled element changes the `u32` word). The `f32` check on
GUST's tree and the hand tree is meaningful at every size (the bound is a few times `1e-5 S` at most). `S` is itself
computed in f64, so the code scales it up by `1 / (1 - gamma64_{n-1})`.

The `f32` inputs are multiples of 2^-23 in `[-0.5, 1.5)`, so every addition rounds (a test shows a left-to-right sum
differs from the f64 one) and no partial sum overflows or goes subnormal. A backend that fails the bound fails the cell,
and a failure of the `f32` sum against its bound on any backend stops the slice that added it; the bound is never
widened to pass.

*The two algorithms differ.* GUST's v1 sum (`gust_algo::reduce`) gives a workgroup 8192 elements (256 lanes, each a chain
of up to 32 terms), tree-adds the lane values in 8 rounds into one partial per workgroup, and repeats over the partials:
one to three block-reduce dispatches (three at 2^27), with pooled partial buffers it zeroes on every call. The hand
reduction is two dispatches (above): one grid-stride pass whose grid grows with `n` and one workgroup. Remaining
differences: GUST's per-call pooling and zeroing, a different chain length and workgroup count, and (for `u32`) the
read-back inside the library call. They are different algorithms, so no GUST/hand ratio is a GUST overhead.

*What the timings of W7 are.* `gust-algo`'s sum exposes no GPU timestamps, so a W7 cell of GUST has no GPU time. Its
`dispatch_host` is `reduce::sum_on_device` for `f32` (the one-word result stays on the device, read back outside the timed
region, like the hand dispatch), but it still draws and zeroes its pooled partials per call, which the hand dispatch does
not. `u32` has no on-device sum, so W7-`u32`'s GUST `dispatch_host` is the public `sum_wrapping` call **including the
4-byte read-back**; the hand `dispatch_host` has none. The report shows no GUST/hand ratio for W7 and the kill rule leaves
W7 out (see "The kill rule").

*Devices.* Inputs above 2^25 words need a storage binding above the portable 128 MiB limit, so a run that includes them
(the default W7 ladder does) opens both GPU devices with the adapter's own limits (`LimitsPolicy::Adapter` for GUST, the
same three limits for the hand device), for every workload of that run. Both sides change the same way, so no pairing is
biased, but such a run is not directly comparable with an earlier run on the portable limits. Both grants are in the run
file's machine block (`wgpu_limits` and `wgpu_limits_hand_wgsl`).

**The W9 bound.** The entry `c_ij` is an inner product of length `N`: `N` multiplies and `N - 1` adds, in whatever order a
backend groups them. With the same rounding model as W3, Higham (section 3.1, eq. 3.5 and 3.13) gives `|C^ - C| <=
gamma_N |A| |B|` entrywise for the exact product `C` of the f32 inputs, and a fused multiply-add only has less error. The
reference is the product in f64 with `|A||B|` computed beside it (non-negative terms, relative error at most `gamma64_N`),
so the check, as W3's, is

```text
|c_ij - c_f64_ij| <= (gamma_N + gamma64_N) * (|A||B|)_ij * (1 + gamma64_N) + 2N * 2^-126
```

with `u = 2^-24` for the CPU backends (`serial`, rayon, `gust-twin`) and `u = 2^-23` for every GPU backend (GUST, its
SPIR-V route, the hand WGSL), as in "The unit roundoff, per backend" (the accumulation is a recursive sum of depth
`N - 1`, inside the theorem). The underflow term is first-order and moot here: the inputs are multiples of 2^-23, so the
smallest non-zero product is 2^-46 and nothing is subnormal. The matrices have full-mantissa entries in `[-1, 1)`, so the
products round. Tests show a plain result and a fused multiply-add result inside the bound (under both units), a result
between the two bounds passing for a GPU and failing for a CPU backend, and a result one f32 step past the bound outside;
the bound is never widened to pass. W9's size is the matrix side `N` (the matrices hold `N * N` words), and `--sizes` is
refused beside W9 above 4096 for that reason. Achieved bandwidth is not applicable to W9 (each matrix is read many times),
so the report shows none; a GFLOP/s column is not part of this slice.

## Backends and routes

| id | what it is |
|---|---|
| `gust-gpu` | the GUST kernel through `gust-wgpu`'s public API, WGSL route |
| `gust-gpu-spirv` | the same on the opt-in SPIR-V route (the `spirv-route` build) |
| `wgsl-prebuilt` | hand-written WGSL on plain wgpu, host side built the careful way: bind group and staging buffer made once |
| `wgsl-percall` | the same WGSL with a host side shaped like `gust-wgpu`'s: a bind group per dispatch, a fresh staging buffer, two submissions |
| `rustgpu-wgpu` | the `rust-gpu` build only: W1 to W4 written in Rust with rust-gpu 0.11.0, compiled on a pinned nightly to a committed SPIR-V module and loaded through wgpu's passthrough route; the host side of `wgsl-prebuilt` |
| `gust-twin`, `gust-cpu`, `gust-cpu-par` | GUST's CPU twin, the production CPU target and its element-wise parallel map |
| `rayon-1`, `rayon-N`, `serial` | `par_iter` on one thread and on all cores, and a plain loop |

A build holds one GUST route, so the two routes are separate result files, each with its own `wgsl-prebuilt` and
`wgsl-percall`, and are compared through their paired ratios to their own run's `wgsl-prebuilt`. `wgsl-percall`
also has three ablation switches (bind-group reuse, staging reuse, one submission); they run only in the ablation group,
whose rows are never part of a table.

### The rust-gpu comparator

`rustgpu-wgpu` answers a different question from the hand WGSL: what the same four kernels cost when written in Rust
and compiled by rust-gpu, the other Rust-to-GPU route. It is in a build of its own (`--features rust-gpu`, off by
default), never in the default build, CI or the lock.

- Toolchain: rust-gpu 0.11.0 (`spirv-std`, `spirv-builder`), which needs the pinned nightly `nightly-2026-08-15` with
  `rust-src`, `rustc-dev` and `llvm-tools`; the project calls itself "not yet production-ready". The shader crate and
  its builder are outside the bench workspace (own lock, own `rust-toolchain.toml`); the built module is committed with a
  manifest (toolchain, crate versions, entry points, sha256) and a test ties the bytes to it. The module targets
  Vulkan 1.1 (`spirv-val --target-env vulkan1.1` clean). `bench/README.md` has the rebuild command, with `--locked`;
  reproduction of the committed bytes was shown on one machine (the build machine), and a test in the default build
  checks the bytes against the manifest and the toolchain and crate pins against both `rust-toolchain.toml` and both
  `Cargo.toml` files. A run file records the module's identity under `versions.rust_gpu` (rust-gpu version, toolchain,
  target, the sha256 of the module the binary embeds).
- The Vulkan memory model, stated plainly: the module declares it (`OpCapability VulkanMemoryModel`,
  `OpMemoryModel Logical Vulkan`, as every rust-gpu Vulkan module does, and W4's queue-family-scope atomic needs it),
  and wgpu-hal 30.0.2 creates the device with `vulkanMemoryModel` off unless cooperative matrices are requested. That
  breaks a Vulkan valid-usage rule (VUID-VkShaderModuleCreateInfo-pCode-08740: a capability needs its feature), so the
  behaviour is undefined by the spec. It ran correctly on the two adapters tried (RTX 5090, Radeon 890M). Under the
  Khronos validation layer (Vulkan SDK 1.4.357) a functional quick run reports it for this module:
  VUID-VkShaderModuleCreateInfo-pCode-08740 and, for the queue-family scope of W4's atomic,
  VUID-RuntimeSpirv-vulkanMemoryModel-06266. Because Vulkan 1.3 requires the feature to be
  supported, the backend runs only on a physical device that reports Vulkan 1.3 or later
  (`WgpuContext::open_passthrough_on`; the adapter's feature query itself is not reachable through wgpu's HAL access
  without a Vulkan dependency of the bench, so the API version stands for it); on any other adapter it is refused with
  the reason. Not done here, and a condition of any publishable rust-gpu row: a device that actually enables
  `vulkanMemoryModel` (an own Vulkan device creation handed to wgpu, or an upstream wgpu change).
- Loading: `Features::PASSTHROUGH_SHADERS` and `create_shader_module_passthrough`, the call GUST's SPIR-V route makes,
  so neither naga nor wgpu's shader validation sees the module. The backend is refused, with the reason, only for a
  missing capability (the Vulkan 1.3 gate above, or passthrough shaders, which wgpu-hal sets on every Vulkan adapter,
  so that check is defensive): it is then left out of the run (`config.backends_not_run`), the cell is not run, never a
  fallback to WGSL, and naming it with `--backends` is an error. Any other failure to open the device fails the run.
- Kernels, each against its hand WGSL (`bench/shaders/`), all with workgroup size 256 and the hand WGSL's bindings:

  | workload | entry point | same as the hand WGSL | differs |
  |---|---|---|---|
  | W1 vector add (`f32`) | `w1_f32` | algorithm, guard on the output length | rust-gpu's own index checks (below) |
  | W2 saxpy | `w2_saxpy` | algorithm; the scalar is a one-element read-only storage buffer at binding 0, not a uniform, so neither side has a cheaper scalar path; guard on the output length and a non-empty scalar buffer | rust-gpu's own index checks |
  | W3 polynomial, degrees 4, 16, 64 | `w3_horner` | Horner's rule, the loop bounded by 64 that leaves at the degree, the three guards | rust-gpu's own index checks |
  | W4 histogram, four cases | `w4_histogram` | key assignment (a workgroup owns 1024 keys, each lane takes four, strided by 256), the bin guard; an atomic add per key | the add is `OpAtomicIAdd` at queue-family scope, the widest the module can declare without the `VulkanMemoryModelDeviceScope` capability (which wgpu does not enable), on a `&mut [u32]`, relaxed like the WGSL `atomicAdd`; queue-family scope itself needs the `vulkanMemoryModel` feature, which the device does not have, so it breaks VUID-RuntimeSpirv-vulkanMemoryModel-06266 (see the memory-model bullet) |
  | W5, W6, W7, W9 | none | | no rust-gpu kernel: the backend does not run them |

  A fifth entry point, `w1_u32`, is in the module and validated with it; no workload runs it. The "same algorithm"
  column is what was compared: each rust-gpu kernel against the hand WGSL of its workload, line by line (bindings,
  guards, loop shapes, key assignment); `Workload::same_kernel()` is about GUST and the hand WGSL and says nothing about
  this backend. W5 to W9 have no rust-gpu kernel at all. The GUST kill rule judges GUST against `wgsl-prebuilt` only and
  does not see this backend.
- Host side: the backend's own device (the run's third, after GUST's and the hand-WGSL one) on the same adapter
  selection as `wgsl-prebuilt`'s (default limits, timestamps), opened only when the run selects the backend, bind group
  and staging buffer built once, one submission per end-to-end sample. Every sample is judged by the same checker as
  every other GPU backend (the poison alone fails it).
- A run from the `rust-gpu` build is marked not publishable (`build.features` lists the feature): a third device and one
  more backend in the rotation have not been measured to leave the other rows unchanged. The mark is lifted only when
  both hold: (1) a run where the rust-gpu device cannot disturb the others (the backend in a group or a process of its
  own, or a measured no-interference result), and (2) a device that enables `vulkanMemoryModel` (above). No timing of
  this backend is claimed in this document.

### Bounds policy

An index outside a storage buffer is outside the benchmark's domain: no workload produces one, and the harness checks
every sample against the reference. The backends still differ in what the shader does if one occurs, and the table
should say so:

| backend | an out-of-range storage index |
|---|---|
| WGSL route (naga, `wgsl-prebuilt`, `wgsl-percall`, `gust-gpu`) | made safe by wgpu's shader runtime checks, which the `unchecked` probe turns off |
| GUST's SPIR-V route (`gust-gpu-spirv`, D174) | the route's own guards (an index clamp, on whenever the route is on), and naga-like behaviour where `robustBufferAccess2` is present; a D39 limitation (`docs/SEMANTICS.md` section 15) |
| `rustgpu-wgpu` (rust-gpu 0.11.0) | the invocation ends early: every slice index is guarded by an array-length test and the failing edge returns, so no load or store happens (read from the module's disassembly and from rust-gpu's source, where the panic paths of `core` become an abort that the linker turns into `OpReturn`; one dispatch of an unguarded W1 with 256 extra invocations on a 2^20-element buffer left the result intact and the device healthy, which is consistent with it but does not prove that the extra invocations wrote nothing) |

The hand kernels and the rust-gpu kernels also carry explicit guards on the output length, so for the workloads here the
two policies never run.

## The four timings

Per cell (backend, workload, size), each sampled 30 times after a warm-up, under a cap of 8 s per quantity and never below 5 samples (a cell with fewer
than 30 collected is marked F in the tables):

1. `end_to_end`: host inputs in, host output out, everything complete.
2. `dispatch_host`: one dispatch on resident data, host wall clock from the call to completion.
3. `gpu_single`: GPU timestamps around one compute pass.
4. `gpu_batch_per_dispatch`: timestamps from the first to the last of 16 back-to-back passes in one submission, divided
   by 16.

The GPU backends of a cell are sampled together in interleaved rounds, each round in a seeded random order, and each
backend is paired against `wgsl-prebuilt` round by round. The warm-up runs real load until the clock query reports P0,
capped at one second; P0 is recorded, not required.

## The publishable rule

A result is shown only if all of these hold. The harness decides items 1 to 3 and marks every cell and every run; a run
that fails is written with the header "shared machine, not publishable" and the reasons. Item 5 is checked by the
renderer.

1. The commit is an ancestor of `origin/main`, the work tree is clean (untracked files included) at the start and at the
   end of the run, and the build is a release build with none of the tainting features (`profile-counts`, `wgpu-trace`).
   A run that is `--quick`, `--functional`, `--cold-only`, `--sizes`, asks for fewer than 30 samples, changes the settle,
   the window, the sample cap or the warm-up (`--settle-secs`, `--window-secs`, `--cap-secs`, `--warmup-secs`), is on a
   software adapter, is in the ablation group or has probe backends is never publishable, each with a named blocker. So
   is a run whose compiled-in git HEAD differs from its run-time HEAD, or where either is unknown (build from the commit
   you run). A quantity with fewer than 5 collected samples blocks its cell.
2. Three complete runs of one session (the first start to the last end within 12 hours, distinct seeds), all checks
   passed, no failures, one route, one build, one adapter.
3. A quiet window before every cell. The harness settles for 3 s, then samples a 2 s window at 4 Hz (8 samples) of the
   Windows GPU-engine counters (per process and per adapter) and the process table (CPU time, creation time). The
   harness itself, its parent chain up to the first process that is not a shell or cargo, and the harness's descendants
   are excluded by process ancestry. Anything else is foreign: a process above `T_gpu` on the adapter under test in any
   sample, a named process (cargo, rustc, rustdoc, slangc, a model server, python, node, `vmmem`) that used CPU or
   started or ended during the window. A window that cannot be judged (a counter error, a short sample list, an
   unreadable process table, a counter row that is not a number, an unparsed counter instance, or an `invalid` reading
   of a counter that matters) counts as not quiet. The quiet windows of a publishable session are taken with the
   monitoring scripts stopped. A cell that is not quiet is retried, up to `N` times; every attempt is
   kept in the sampler sidecar.
4. The thresholds below, which are set from recorded data.
5. An attestation line in the commit that files the results (`Quiet-window attestation: ...`), and the harness's own start
   and end times in the file. The renderer refuses a file whose sidecar is missing or whose commit has no such line.

### The thresholds

<!-- bench-thresholds:begin -->
| Constant | Value |
|---|---|
| Thresholds version | 1 |
| `T_gpu` (foreign GPU-engine use, percent of the engine, any sample) | 5.0 |
| `T_cpu` (`vmmem` CPU growth, cpu-seconds per second) | not set (no `vmmem` process in any recorded window; a `vmmem` or `vmmemWSL` process present makes a window not quiet) |
| `N` (settle-and-sample attempts per cell) | 5 |
| Runs per aggregate | 3 |
| Session span, hours | 12.0 |
| Samples per quantity | 30 under a cap of 8.0 s per quantity, never below 5 |
<!-- bench-thresholds:end -->

These come from the sampler sidecars recorded in M10 (the values above are read from the harness source by the
renderer, so this page cannot drift from the code).

**The data.** Two sets of sidecars, all on one machine (an RTX 5090 desktop under Windows 11):

- a *quiet* set: 9 sidecars (three full runs and six quiet-window runs taken with the machine otherwise idle), 726
  windows, 174 of them GPU-group windows;
- a *shared* set: 17 sidecars (quick dry runs and smoke runs taken while other builds and tests ran), 510 windows.

For each window the maximum over samples of the largest foreign process's GPU-engine use on the adapter:

| Windows | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| quiet set, all (726) | 0.44 | 2.94 | 5.25 | 15.07 | 17.11 |
| quiet set, GPU-group windows (174) | 0.44 | 3.53 | 6.68 | 14.67 | 17.11 |

The harness's own use in its idle windows is 0.00 percent in all 1236 windows of both sets, so the 3 s settle is long
enough for the harness's own work to drain. The ambient desktop does produce short bursts (the System process's copy
engine and the session manager's `csrss`): 40 of 726 quiet windows (5.5 percent) are above 5 percent, 2.9 percent above
10, 1.2 percent above 15, none above 20. The shared set never recorded a sustained foreign GPU load either (its maximum
is 15.51), because the other lanes' work on that machine was CPU work (cargo and rustc grew 0.1 to 3.8 CPU-seconds per
second there, and none in any quiet window).

**What this does and does not separate.** `T_gpu` is 5 percent: above the median and the 90th percentile of the quiet
distribution, and at about its 94th. The ambient bursts above it are not handled by raising the threshold but by
retrying. Under the whole rule as implemented (the GPU rule, the named-process rule and the fail-closed rows), 87 of the
726 quiet windows fail (12.0 percent), and a failing window is followed by another failing one in 22 of 86 cases (25.6
percent). So a window sequence (one per CPU cell, one per GPU group) fails all five attempts with probability about
5e-4. A full run takes 96 such sequences (120 with the CPU twin), so about 5 (6) runs in 100 end not publishable, and
about 14 (17) aggregates of three runs in 100 contain one. The recording itself held one sequence of five failing
windows in a row, where this model expects about 0.4. That is on a machine where the monitoring scripts of the
recording still ran; 53 of the 87 failures are named-process windows. The figures come from the M10 premise runs'
sidecars (not in the repository).

That is the honest limit of the data: the recorded windows separate the harness's own idle noise (zero) and the desktop's brief bursts from a sustained
load by their persistence, not by a measured distribution of foreign load, because no sustained foreign GPU load was
ever recorded. The one known sustained shape is the harness's own `probe-adapters` busy loop, which reads up to 42
percent. A foreign process that uses the GPU lightly and steadily below 5 percent is not detected.

**`T_cpu` is not set.** The only process the CPU rule needs a threshold for is `vmmem` (the WSL and Hyper-V memory
process). It is read like any named process, but with no threshold the judge fails closed: a `vmmem` or `vmmemWSL`
process present in a window makes it not quiet, whatever its CPU time does. No such process existed in any recorded
window, so there is no data to place a threshold on. **Limit:** one sampler window taken with the WSL service up, but
with no running distribution confirmed, listed neither `vmmem` nor `vmmemWSL` (only `wslservice` and `vmcompute`), so
whether the process walk lists WSL's VM process by name is not established, and WSL's CPU load may not be seen by the CPU
rule. Named processes need no threshold; any CPU growth in a window counts. (The monitoring scripts of
the recording grew up to 0.49 CPU-seconds per second in 32 of 726 quiet windows and are on the named list for that
reason.)

## The kill rule

For each cell of the publishable group the harness takes the paired ratio of the GUST backend's GPU time to
`wgsl-prebuilt`'s (`gpu_single` and `gpu_batch_per_dispatch`). A ratio more than 10 percent off 1.0 on any cell marks the
run "C re-opened" (an evidence flag that the GUST path adds cost the plan did not expect); the run file lists the
offending cells. The rule judges only a complete run: a cell without a ratio counts as unpaired and the verdict is then
`null`, never `false`.

**Cells outside the rule.** A workload whose GUST and hand paths are not the same kernel is outside the rule, on both
sides (`Workload::same_kernel`; today only W7): GUST runs `gust-algo`'s public sum, which exposes no GPU timestamps, so its
cell has no paired GPU-time ratio by construction, and the hand reduction is another algorithm. Such a cell is neither
counted as unpaired nor judged (it would otherwise keep every run that includes W7 from being judged, `c_reopened` always
null), and it is **listed**: the rule's record in the run file has `excluded: [{backend, workload, n, reason}]` for each
of them, so a reader sees W7 was present and left out on purpose. This is a deviation from the plan's "any cell of the
publishable group" (decision D190). It does not affect whether a run is publishable.

W6-graph is outside the rule too, for a different reason and through a predicate of its own
(`Workload::gust_gpu_timestamps`, not `same_kernel`): `StagedGraph` runs through `execute_graph`, which records no GPU
timestamps, so the graph form has no GPU-time samples (`gpu_single` and `gpu_batch_per_dispatch` do not exist for it) and
no GPU-time pairing. Its cells are listed in `excluded` with that reason, and it is not counted as unpaired (it would
otherwise keep every full run from being judged). W6-graph stays `same_kernel`, so its dispatch and end-to-end ratios are
still shown and judged by the report; W6-hand has GPU timestamps and stays in the rule. This is a second deviation from
the plan's "any cell", recorded beside D190 by the coordinator; no `crates/` change was made for it.

**Reproducibility clause.** A gap counts only if it recurs: the same cell must be more than 10 percent off in all three
seeds of a confirming run (at least three runs with distinct seeds), with the control (`wgsl-percall`) inside its tolerance
on that cell in each of those runs. A cell that recurs with the control out of tolerance, or in fewer seeds, is listed and
not counted.

**The control's limits for the batch quantity.** `wgsl-percall` runs the same kernel text as `wgsl-prebuilt`, so its
distance from 1.0 is a noise floor for `gpu_single`. For `gpu_batch_per_dispatch` it is not a pure noise floor: it
differs from `wgsl-prebuilt` in the per-pass bind groups (percall builds a fresh one in each of the 16 passes, as GUST does;
prebuilt reuses one), so its batch excess may be a real effect of that and not noise. A "control out of tolerance" verdict
therefore voids a batch cell on a confounded control. A control that is identical in command stream for the batch
quantity would be a second `wgsl-prebuilt` instance, or the bind-group-reuse switch turned on. The kill rule's
re-opening does not gate whether a run is publishable; it is a recorded finding, and this limit is part of how it is read.

## Known outcome: the batch path (outcome D)

Closed as "unexplained, not tuned".

- An earlier quiet-window run on an RTX 5090, which the harness marks not publishable, showed GUST's 16-pass batched
  GPU time at 1.12 to 1.21 times hand WGSL's (a ratio of the two GPU times) on W1 at 1M and 4M elements and W2 at 1M,
  with `gpu_single` at parity and the control at 0.99 to 1.00.
- The wgpu API traces of GUST's batch and the hand batch differ in one per-pass item: GUST binds a fresh bind group in
  each pass and `wgsl-prebuilt` reuses one. `wgsl-percall` matches GUST's stream.
- In the confirming run (seeds 11, 22, 33; on the same RTX 5090, in runs the harness marks not publishable) the gap
  recurred on W1 1M only (1.275, 1.221, 1.248). The control was inside its tolerance in seed 33 only (1.106, 1.125, 1.081), so by the clause above the cell is listed, not counted. W1 4M (1.00)
  and W2 1M (1.03 to 1.08) did not recur.
- Neither the per-pass bind groups (GUST over percall is about 1.09 to 1.15) nor the pre-registered ablation `gust-setup`
  (GUST over gust-setup is about 0.97 to 1.11) accounts for the gap in all three seeds.
- A probe that runs the hand kernel on a separate device opened like GUST's sits in the same slow state as GUST (1.24 to
  1.29). That is an observation, not a cause, because the probe changes three things at once.
- No further ablation was run (the kill rule).

**Limitation as measured.** On an RTX 5090, in runs the harness marks not publishable, the ratio of GUST's batched GPU
time (16 dispatches) to hand WGSL's on vector-add at 1M elements is 1.22 to 1.28 in the last measurement. Over earlier
quiet runs the ratio reached 1.12 to 1.21 on vector-add and saxpy at 1M to 4M. The ratio for single-dispatch GPU time is
0.99 to 1.03. The cell is bimodal for every backend (about 6.1 and 7.9
microseconds per dispatch on W1 1M), and the identical-kernel control moved by up to 1.13 times on the same cell. The
excess is unexplained and not tuned. It is not in the generated kernel: running GUST's WGSL in the hand path gives 1.00
times.

## An observation, beside the method: the SPIR-V route on W3

On an RTX 5090, in runs the harness marks not publishable, the GPU-time ratio of the opt-in SPIR-V route to hand WGSL
on the polynomial workload is 0.80 at degree 16 and 0.70 at degree 64, both inside the derived bound. Most of the difference is wgpu's shader runtime checks on the WGSL route (naga's
loop-bound counter, absent from slangc's SPIR-V): the same ratio for hand WGSL without those checks is 0.81 and 0.80. At degree
16 the checks account for nearly all of the gap; the remainder at degree 64 (about 0.10) is observed, and its cause is not
identified. The arithmetic is the same in both routes (no fused multiply-add, no unrolling). This is an observation about
the method, not a result of this page, and under the rule as the harness implements it (two-sided, `group.rs`), a ratio that far below 1 on a cell of the publishable group re-opens the kill rule for its run; these runs are not publishable, and how the rule should treat the opt-in SPIR-V route is an open question for the owner's project review.

## Other limits of the measurements

- Cache residency: a cell whose working set fits in the GPU's L2 can read faster than DRAM; the harness flags every GPU
  figure above the adapter's DRAM peak (`dram_peak`, `above_dram_peak`). A figure above the peak is not a DRAM bandwidth.
- Two-state cells: some cells alternate between two speeds (W1 1M: about 6.1 and 7.9 microseconds). The report carries a
  gap test (`two_state_wide_gap`, 10 percent of the median between the 5th and 95th percentile of the batch quantity), which
  is calibrated on the m10 runs only and not fitted beyond them.
- The sampler idles the GPU once per GPU group, not before every backend; the CPU backends get a window each.
- P0 is not reached at 1M elements and below within the warm-up cap; the group is kept and marked.
- The sampler reads Windows counters; on other platforms a run cannot be judged and is never publishable.
- Comparators run inside this harness with the settings stated here (see the header).

## Results

<!-- bench-tables:begin -->
No results are filed. Nothing here is publishable until three complete runs of one session pass the rule above and are filed with the attestation line in their commit. The renderer refuses anything else.
<!-- bench-tables:end -->

## Reproducing

From `bench/` (the bench is a workspace of its own; every cargo command there runs with `GUST_PROFILE=dev`; its
`bench/target/` is gitignored, so the build leaves the tree clean). Build from a clean tree at the commit you will run:
the binary embeds its build commit and whether tracked files had uncommitted changes then, and a run whose build commit
differs from the checked-out one, or whose build was dirty (or whose build state is unknown), is not publishable.

```text
mkdir -p OUTSIDE_THE_WORK_TREE
GUST_PROFILE=dev cargo build --release --locked -p gust-bench
./target/release/gust-bench run --out OUTSIDE_THE_WORK_TREE/run1.json --label run1 --seed 1
```

Take the output path outside the work tree (a run that leaves the tree dirty is not publishable). The aggregate of three
runs is `gust-bench aggregate --json FILE --md FILE RUN.json...`. Filing is `bench/results/<adapter-slug>/<commit>-<route>.json` plus its
`.samples.json` (`<adapter-slug>` is the adapter's name in lower case with each run of characters other than letters and
digits made one hyphen, so "NVIDIA GeForce RTX 5090" is `nvidia-geforce-rtx-5090`; `<commit>` is the run's build commit,
at least its first 12 characters; `<route>` is the file's route), in a commit whose message carries the `Quiet-window attestation:` line. Then:

```text
cargo xtask check-bench --render          # writes the tables of this page
cargo xtask check-bench --render --check  # fails when the page is out of date (no workflow runs it; the xtask tests check the page against the renderer)
cargo xtask check-bench --functional --quick   # the correctness run on any adapter, no timing published
```
