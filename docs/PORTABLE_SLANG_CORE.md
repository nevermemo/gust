# Portability contract and evidence

Status: policy plus limited executable evidence. Last reviewed 2026-11-15.

| Tier | Meaning | Current evidence |
| --- | --- | --- |
| A: Portable GPU Core | Conservative Slang GPU subset across intended GPU targets; excludes CUDA-specific and Slang CPU execution assumptions | WGSL and SPIR-V compilation, and `spirv-val` on the exported SPIR-V (`cargo xtask check-artifacts`) |
| B: WebGPU engine profile | Explicit bounded bindings, portable resource layouts and synchronization, no native-only requirements | Native Vulkan wgpu execution of WGSL, and a browser run: `cargo xtask check-web` (`examples/web-vector-add`) and `gallery/web-hello` run the WGSL in headless Chrome (Chrome 156 on one RTX 5090 is the only browser and adapter measured) |
| C: Native optimization | Capability-gated binding arrays, nonuniform indexing, stronger atomics, multi-draw or mesh paths | Not implemented or tested here. The one capability gate that exists, `subgroups` (M8, Vulkan only, refused in the browser), is not on this list |

Tier labels are not certification. Metal, DXIL, browsers other than Chrome, and
additional vendors remain untested. A future machine-readable manifest should record
compiler version, target/options, adapter/features, fixture, compile/validate/execute
outcome, and unsupported reasons. `.ai/VALIDATION.json` is a local, untracked gate
artifact that `cargo xtask check-full` overwrites: it records command outcomes,
versions in stdout, and exported SPIR-V hashes, with the workspace root and home
directory redacted, and it is never committed (copy a summary into `STATUS.md`). It is
not that full cross-target capability matrix. Missing tools cannot count as successful
probes.

## Capability probe records

`gust::slang::probe(target)` compiles a known-good minimal compute kernel to
`target` and returns a `TargetProbe { target, supported, detail }` record. `supported`
is true only when `slangc` actually emitted the target; `detail` carries the success
marker or the compiler diagnostic on failure. This is the first machine-readable
capability record: downstream claims must cite a probe record, not an assumption.
The probe is capability evidence, not a correctness test. Cross-target discovery
(DXIL, Metal) and adapter/feature capture as probe records remain future work beyond
the completed T05 probe slice (T05 is complete; those are not part of its scope). The
browser run exists, but it is a run of kernels in Chrome (its result line names the
adapter), not a probe record.

## Resource baseline

Keep element data in `StructuredBuffer<T>` / `RWStructuredBuffer<T>` with fixed-size
elements. Store logical counts/capacities/offsets as separate metadata. Do not nest
an unsized geometry structure inside `RWStructuredBuffer<GeometryBuffer>`.

WGSL permits a runtime-sized array as a storage-buffer store type, or the last
member of the top-level storage structure; array elements cannot themselves be
runtime-sized. See the [WGSL type rules](https://www.w3.org/TR/WGSL/#runtime-sized)
and [Slang WGSL resource mapping](https://shader-slang.org/slang/user-guide/wgsl-target-specific).
Slang maps structured buffers into storage arrays. A custom top-level header plus
trailing array would need a separate, proven layout path; it is not today's ABI.

Portable references are indices, offsets, and IDs. A future `GpuPtr<T>` may carry
buffer identity plus offset, with bounds and lifetime rules; it is not a native
address or a promise to preserve arbitrary Rust pointer behavior.

Do not say “wgpu has no bindless or multi-draw.” Native capabilities exist, but
their requirements and backend support differ. Check [wgpu 30 feature flags](https://docs.rs/wgpu/30.0.2/wgpu/struct.Features.html)
(the workspace requires wgpu 30.0.1 or newer in the 30 line, a floor and not a pin; the lockfile resolves 30.0.2)
and enable them only after probing. The engine baseline must work without those
optional paths; native fast paths preserve the same output semantics.

## Numeric and ABI boundary

Today: 32-bit scalar arithmetic and recursively padding-free, four-byte-aligned
storage structs (`StorageV1`); booleans are not host POD. Beyond that the layouts are
`StorageV2` (2- and 4-wide vectors, 2x2 and 4x4 matrices, with natural alignment),
`Std140V1` for uniforms, and 2-D textures and samplers (`gust::texture`). Each is
gated by reflection: before a pipeline is created, the runtime compares every binding
and every field offset, size, alignment and stride with what Slang reports (D17, D91),
so a layout that disagrees is an error, not a silent upload. Float comparisons use explicit tolerances when operations can round
differently. No claim of general Rust integer overflow or float-to-int cast semantics.
Any wider ABI needs target layout tests before public buffer support.
