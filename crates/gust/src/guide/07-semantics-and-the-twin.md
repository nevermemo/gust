# Semantics and the twin

A `#[gpu_module]` kernel is Rust that runs on a GPU, so two questions decide whether you can trust a result: what does the
kernel mean, and where may the GPU and plain Rust disagree. This chapter is the short version of
[SEMANTICS](https://github.com/nevermemo/gust/blob/main/docs/SEMANTICS.md), the page that states the contract. It
summarises sections 1 to 3 (scope, labels, the defined domain), the known-mismatch ledger (section 7) and the SPIR-V
route (section 15); it does not copy them, and where a number or a measurement below differs from that page, the page
is right.

## Three targets, one contract

A kernel can run in three ways, and the contract says what they share:

- **The GPU.** The macro translates the kernel's Rust syntax tree to Slang, `slangc` compiles the Slang to WGSL and
  wgpu runs the WGSL. The WGSL specification is the portable reference for what the GPU may do. The measurements in
  SEMANTICS were mostly made on one adapter (an NVIDIA RTX 5090 over Vulkan), and a few rows also have a software
  adapter (Mesa lavapipe); the page says which.
- **The CPU twin.** In a test build of the crate that defines the kernel (`cfg(test)`), or with `--cfg gust_dual`, the
  macro also emits an `impl Run` for `gust::cpu::CpuDevice` that runs the Rust body of the kernel, one invocation at a
  time in ascending `global_id.x`. A production build holds one target, so the twin is absent from it. The twin runs
  the body that rustc already checked, so it is a reference for what the program means, and it needs no GPU.
- **The production CPU target.** `GUST_PROFILE=cpu` makes the CPU the one target of a build: the kernels run as plain
  Rust and the build contains no Slang compiler, no helper and no GPU runtime ([chapter 6][ch6] has the profiles). Its
  contract is the twin's.

The contract is stated for a **defined domain**. Inside it the GPU and the twin give the same bits for integer and
`bool` results and take the same control-flow path, and the float results agree within bounds that SEMANTICS section 4
derives from the WGSL specification. Outside it the targets are allowed to differ, and GUST documents the difference
instead of removing it.

## How sure each statement is

SEMANTICS never merges its labels, and a claim should be read with its label:

- **spec-allowed**: the WGSL specification defines the result or the allowed set, so it is portable across adapters
  that run the WGSL GUST emits.
- **unspecified**: WGSL leaves the outcome open. The GPU value in the ledger is an observation and never a
  requirement.
- **measured**: one adapter, one driver, one date. Not portable.
- **proven**: a test pins the result on the GPU, on the twin and against an independent host reference, and the value
  lies inside the allowed set. It is a statement about the inputs the test uses, not a theorem.
- **planned**: the behaviour is intended and nothing runs yet. It is not evidence.

A run of the CPU twin is never evidence about a GPU. That is why the gallery programs compare a GPU result with a host
reference written in plain Rust that never calls the kernel, and use the twin only for the part that needs no adapter.

## The defined domain

The domain has six conditions. A kernel whose run-time values satisfy them has the same integer results on every
target:

1. `+`, `-`, `*` and unary `-` on `u32` and `i32` give a result that fits the type.
2. Divisors are not zero, and `i32::MIN / -1` and `i32::MIN % -1` do not occur.
3. Run-time shift amounts are in `0..32`. A constant amount outside that range that the macro can see is a compile
   error (GUST-E0335).
4. `f32 as i32` and `f32 as u32` have a finite operand whose truncated value fits the target type.
5. Float operands, intermediate values and results are finite, and each is zero or normal (not subnormal).
6. Every index into a storage buffer is below its `len()`, and no two invocations race.

Two things the domain does not remove. A decision derived from a float (a comparison, a branch or a cast) is exact
only while no value lies within the float error bound of the decision point, and the integers computed from that
decision inherit the difference. And some atomics are order-dependent: the twin runs the invocations in ascending order
and so gives one member of the set the GPU may give. Only a cell that receives only `atomic_add`, only `atomic_min` or
only `atomic_max` has the same final value on both. A contended `atomic_exchange`, and every `atomic_compare_exchange`
(contended or not, because the GPU's weak form may fail spuriously), is set-valued on the GPU, and so is a cell that
receives different kinds of operation.

The last condition is two rules, and the twin treats them differently. An out-of-range index panics on the twin with
"index out of bounds", where the GPU's outcome is open (the invocation may trap, a load may return zero or another
value of the buffer, a store may be dropped). A race is different: the twin runs one invocation to completion before it
starts the next, so a racy kernel gives a clean, repeatable twin result that proves nothing about the GPU, where the
order is unspecified. Keep every written location to one writer, or use the atomic functions.

## What happens outside the domain, and what to write instead

The differences are expected edge cases, and the project's policy for them (decision D39 in
[DECISIONS](https://github.com/nevermemo/gust/blob/main/docs/DECISIONS.md)) is that GUST documents them as
limitations and does not chase a match. In one line each:

- **Overflow** of the plain operators: the GPU wraps modulo 2^32 and never reports it. Rust panics when
  `overflow-checks` is on and wraps when it is off. The twin follows the build of the crate that defines the kernel.
- **Division or remainder by zero, `MIN / -1`, `MIN % -1`**: the GPU returns the dividend, 0, the dividend and 0 (the
  values the WGSL specification gives); Rust panics in every profile.
- **A run-time shift amount outside `0..32`**: the GPU uses the amount modulo 32; Rust panics with checks on and masks
  with checks off.
- **A float cast out of range, or NaN**: Rust saturates and turns NaN into 0. The GPU clamps to the range both types
  represent, so the top end differs, and NaN is indeterminate.
- **NaN, infinity and subnormals**: Rust follows IEEE 754 and keeps them. The WGSL specification lets an
  implementation assume NaN and infinity do not occur, and flush subnormals to zero.
- **An out-of-bounds index**: the GPU raises a dynamic error and the outcome is open; Rust panics.

Keep run-time operands inside the domain by construction, or guard them in the kernel: test a divisor before dividing,
clamp before casting. Where wrapping is what the program means (hashes, random number generators, counters), say so
with `wrapping_add`, `wrapping_sub`, `wrapping_mul` and `wrapping_neg` on `u32` and `i32`. Their result is inside the
domain for every input and is the same on the GPU, on the twin and in plain Rust, with overflow checks on or off.
There is no `wrapping_div`, because a zero divisor has no wrapped result. This kernel does both, and it guards the
index against the length of every buffer it reads or writes (condition 6). The doctest compiles it and does not run it
(`no_run`):

```rust,no_run
# use gust::gpu_module;
#[gpu_module]
mod hashing {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn mix(
        id: SV_DispatchThreadID,
        seed: StructuredBuffer<uint>,
        divisor: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < out.len() && i < seed.len() && i < divisor.len() {
            let d = divisor[i];
            let mut share = 0u32;
            if d != 0u32 {
                share = seed[i] / d;
            }
            out[i] = seed[i].wrapping_mul(2654435761u32).wrapping_add(share);
        }
    }
}
```

The plain operators keep their meaning, so an overflow that is a mistake still wraps silently on the GPU. Do not rely
on the twin to detect overflow on the GPU path.

## The known-mismatch ledger

SEMANTICS section 7 lists the constructs where the GPU and plain Rust differ, or where the GPU's result is
unspecified, and that no diagnostic can prevent because the operands are run-time values. Each row has an id, the Rust
result (which the CPU target follows, and which the twin must equal), the GPU result and a label. The rows are:

| Id | Case |
| --- | --- |
| KM-01, KM-02 | `x / 0` and `x % 0` |
| KM-03 | `i32::MIN / -1` and `i32::MIN % -1` |
| KM-04 | A shift by a run-time amount outside `0..32` |
| KM-05, KM-06 | `f32 as i32` and `f32 as u32` out of range, and NaN |
| KM-07, KM-08 | NaN and infinity; subnormal operands and results |
| KM-09 | Contended `atomic_exchange`, and `atomic_compare_exchange` contended or not |
| KM-10 | Overflow of `+`, `-`, `*` and unary `-` |
| KM-11 | An index equal to the buffer length |
| KM-12 | A texel address outside a texture |
| KM-13, KM-14 | Linear `sample` of a texture, and nearest `sample` within one ulp of a texel edge |

The test `crates/gust-wgpu/tests/mismatch.rs` in the repository runs every row on the GPU, as plain Rust and on the
twin, and an id guard keeps the table and the test in step. The Rust column and every GPU value that the WGSL
specification fixes are enforced, so the test fails when one changes. A value marked unspecified is an observation: the
test prints a note when it changes and never fails. The ledger also names one driver defect that is not a row: on the
RTX 5090 with NVIDIA driver 610.88 through Vulkan, a function-scope array can read zeros after an element at a dynamic
index is written back with a value the compiler can prove unchanged. GUST does not refuse or rewrite the form, because
the generated Slang, WGSL and SPIR-V are correct and an AMD Radeon 890M computed Rust's values. SEMANTICS section 7 and
the README ("Platforms") give the full pattern; it was measured on one device and driver.

## The SPIR-V route

By default the GPU path runs the WGSL that `slangc` emits, after naga checks and rewrites it. The opt-in Cargo feature
`spirv-route` of `gust-compute-wgpu` (it forwards to `gust-compute`, which forwards to the macros) builds Vulkan
compute and pixel pipelines from `slangc`'s own SPIR-V instead, through wgpu's passthrough shader modules. It is off
by default and Vulkan only, and its name is part of the 0.x promise
([chapter 11][`crate::guide::stability_and_platforms`]). A kernel stays on the WGSL route, and the device records the
reason in `HeadlessDevice::route_events`, when:

- the build has no `slang-runtime` (there is no SPIR-V to compile), or the kernel is a precompiled WGSL artifact, so a
  `native` or `web` build, which has no run-time compiler, never takes the route;
- the backend is not Vulkan, or the device lacks passthrough shaders;
- the SPIR-V compile fails, the module has push constants, or its bindings do not match the layout.

What differs on the route is the part to know. Passthrough has none of naga's run-time guards, so the macro emits
them as Slang text when the feature is on: an index into an array whose length it knows is clamped to the last
element, an integer division by a non-literal divisor goes through helpers that divide by one when the divisor is zero,
and workgroup memory is zeroed before the first statement. An index into a storage buffer is not clamped by GUST: it
keeps the device's behaviour, so an out-of-range index there is outside the domain like every other. A float-to-integer
conversion saturates as Rust does, where the WGSL route gave a different top value. A float result may differ by one
ulp where a driver contracts a multiply and an add. One limitation was measured on a software adapter (Mesa
lavapipe): with the route taken, `(x as f32) as i32` for `x = 16777217` gives `x`, where Rust gives the neighbour that
`x as f32` rounds to. It is recorded as a limitation and not chased, and SEMANTICS section 15 says what is known and
what is only a hypothesis about the cause.

Two build-graph consequences matter more than the differences. Cargo unifies features, so one dependency that enables
`gust-compute-wgpu/spirv-route` changes the emitted Slang of every kernel in that build. The content key of a
precompiled artifact hashes the Slang, so artifacts built by `cargo gust build` without the feature do not match a
build with it, and the reverse. Build the artifacts and the program with the same set of features.

## Replaying one call on the twin

When a kernel's result is wrong for one element, the twin lets you step through that element in a debugger.
Every kernel with a twin has a generated `capture(&gpu, &call)` (in a test build, or with `--cfg gust_dual`). Call it
with the same call you are about to dispatch, before the dispatch: it copies every bound buffer and uniform to the host
and returns a `gust::capture::Captured`, which owns the copies and runs the twin over them.

```rust,ignore
let call = update::step::call(count, &params, &input, &out);
let captured = update::step::capture(&gpu, &call)?;
update::step::dispatch(&gpu, count, &params, &input, &out)?;
// Run the body for element 1042 only (set a breakpoint in the kernel body first).
captured.run_element([1042, 0, 0])?;
let after = captured.buffer::<Particle>("out").unwrap().read();
```

`run_element(id)` runs one invocation, `run_until(id)` runs the twin's launch from its first invocation up to and
including `id` (x fastest, then y, then z), and `run_all()` runs the whole launch exactly as the twin does. The launch
covers every axis rounded up to whole workgroups, so `run_until` and `run_all` also run the padding ids, which a kernel
that does not guard them can write from; `run_element` takes only ids inside the extent. Every run starts from the
captured values of the read-write buffers, so runs do not add up; inputs (read-only buffers and uniforms) are not
restored, so you can edit one with `captured.buffer(..).unwrap().write(..)` to ask what a different input would give. An
id outside the launch extent is an error and runs nothing. A call the device would refuse (for example one buffer bound
as an input and an output) is refused by `capture` with the same error the dispatch gives. Four things to know:

- **Read-only buffers need an option.** A read-only storage buffer can only be copied if the device was opened with
  `DeviceOptions::default().with_capture(true)`; without it the capture fails with `PersistentBufferNotReadable` and a
  hint that names the option. Read-write buffers, `GpuVec`s, pool buffers and uniforms need nothing.
- **Not supported yet:** a kernel with a texture or a sampler returns an `Unsupported` error, and a kernel with
  `cpu = "unsupported"` (or none of whose bodies runs without a capability) returns `NoTwin`, because there is no twin
  to replay on.
- **The replay runs the twin's body.** For a kernel with variants that is the portable variant, not necessarily the
  body the GPU ran.
- **A replay reproduces the twin's order, not the GPU's schedule** (the GPU has none). For a kernel without races and
  atomics, `run_all` gives the GPU's result, bit for bit inside the defined domain. For a kernel with atomics or an
  order-dependent result, the replay is one member of the allowed set, not the history the GPU had; a race that
  corrupted the GPU run does not show up in the replay at all (see the paragraph on races above).

## Where to go next

- The kernel-side rules for atomics, workgroup memory and subgroups are in
  [chapter 5][`crate::guide::gpu_concepts`].
- The newer language features have their own sections of SEMANTICS (sections 9 to 14 and 16), summarised in
  [chapter 8][`crate::guide::m9_language`].
- What the promise covers, and how to report a result that disagrees with this chapter, is in
  [chapter 11][`crate::guide::stability_and_platforms`].
