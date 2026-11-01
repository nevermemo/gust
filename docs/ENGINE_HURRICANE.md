# Hurricane: GPU-centric ECS engine destination

## Implemented now

Headless compute, persistent typed buffers, ordered batches, async jobs, timestamp
measurements, and particle/sensor examples. T07 adds an explicit, dependency-checked
staged graph. T08 adds host-driven component-pool growth by GPU copy, safe retirement,
and GPU-derived indirect compute arguments; see the contract below. No ECS scheduler,
rendering, entity lifecycle, compaction or GPU allocation framework is implemented.

## Researched direction

GPU-resident component pools for positions, velocities, transforms, animation,
visibility, and other parallel state. Entity IDs/generations provide stable identity
without exposing native pointers. Compare archetype, sparse-set, and hybrid storage,
including SoA/AoS choices, query costs, structural changes, and safe deletion. Do not
choose one mega-buffer by default. A scheduler may later infer system dependencies
from `Read<T>` / `Write<T>` and effect information; initial proofs use explicit edges.

Dynamic collections need separate buffer allocation, logical length, and capacity.
Host-driven geometric growth should allocate → copy live GPU bytes → refresh
bindings → retire old resources after safe completion. No resize readback and no
automatic shrink. Test zero length, growth boundaries, stale handles, device
ownership, and in-flight work. Pools, arenas, freelists, and compaction are future
options, not requirements for the first vector-like collection.

Geometry uses fixed-size vertex/index pools plus mesh metadata/count/offsets. Do
not use an unsized structure as a structured-buffer element (see [portability](PORTABLE_SLANG_CORE.md)).
Hierarchy can use parent indices plus levels/frontiers; ordinary recursion is not
the portable scheduling model. Textures/materials start with arrays/atlases,
material grouping and tables; native binding arrays require capability gates.

## Long-term aspiration

A frame may flow through input → gameplay → physics → animation → hierarchy and
transforms → culling/LOD → compaction → indirect arguments → rendering. This is
an organizing destination, not a committed monolithic pipeline. Portable rendering
may use grouped or repeated indirect draws; native multi-draw/mesh paths depend on
capabilities. Both should expose the same engine semantics without a CPU shader fallback.

## Small proofs before an engine framework

1. A component pool grows while retaining GPU-authored data, with explicit length
   and capacity and no host readback during growth.
2. Two typed systems declare read/write dependencies and cannot run out of order.
3. GPU active-count generation drives a validated indirect dispatch.
4. Culling writes compacted visibility and indirect arguments consumed by rendering.

Each proof needs a runnable example, failure cases, inspectable artifacts, and
measurements separating compile/setup, upload, dispatch, and readback. Benchmark
against independent CPU implementations; never conclude “GPU is better” from one
cold end-to-end timing or a debug build. Choose the first actual engine user and
observable unaided result before committing a broad ECS API.

## Component pool contract (T08, specified 2026-09-07)

`gust_wgpu::GpuPool<T>` is the first vector-like resident collection. It is
one typed storage allocation plus separate metadata; it is not an arena, freelist,
or entity table.

- **Capacity** is the allocation's element count, at least one. `capacity()` reports
  it exactly; kernels observe it through the bound buffer's `.len()`.
- **Logical length** (`len()`) is the number of live elements, host-tracked, never
  above capacity. Elements at or beyond `len()` are unspecified bytes. The pool
  mirrors the length into a one-element `count_buffer()` (`StructuredBuffer<uint>`)
  so kernels can read the active count on the GPU; the host writes it on every
  length change and never reads it back to make decisions.
- **Growth** is host-driven and geometric: `push` and `reserve` allocate a new buffer
  of `max(2 × capacity, required)`, encode a GPU→GPU copy of exactly `len × stride`
  live bytes into it, submit, and swap the pool's buffer. No readback, no
  re-upload of existing elements, no automatic shrink. Each growth increments
  `generation()` and returns a `GrowthRecord { old_capacity, new_capacity,
  copied_bytes }` as measurable evidence.
- **Retirement**: the previous allocation is kept in a retirement list with the copy
  submission's fence; `reclaim()` drops entries whose submission completed and
  reports how many. wgpu keeps in-flight resources alive regardless, so retirement
  is about observable, bounded lifetime, not memory safety. Growth while an earlier
  dispatch on the old buffer is still in flight is legal: queue submissions execute
  in order, so the copy observes that dispatch's writes.
- **Stale bindings** are prevented statically: `BufferBinding`s borrow the pool's
  buffer, and growth needs `&mut GpuPool`, so a binding cannot outlive the
  allocation it names. `generation()` exists for host bookkeeping, not as a runtime
  check for something the borrow checker already forbids.
- **Truncate/clear** only change the length and the count buffer (no GPU work).
  `pool_read()` copies back the live prefix only.
- **Indirect dispatch**: a buffer created with `create_indirect_buffer` (element type
  must be exactly three `uint` fields matching `DispatchIndirectArgs`; `INDIRECT`
  usage) may drive a `StagedGraph::dispatch_indirect` node. There is no host-known
  element count, so every binding must opt in with `independent_length()` and be
  non-empty (a placeholder allocation could be observed); layouts, access, and device
  are validated as for direct dispatches, and the args buffer counts as a read for
  hazard checking. The runtime cannot bound the dispatched count: wgpu zeroes an
  indirect dispatch whose workgroup count exceeds
  `max_compute_workgroups_per_dimension`, silently. Therefore the args must be
  derived on the GPU from the count buffer **clamped to the data buffer's `.len()`**
  (the allocation), and kernels must guard `i < active && i < buffer.len()`. A
  malformed count then degrades to "process the whole allocation", never to
  out-of-bounds access or a silent no-op. T09 supplies bounded loops. This proof
  predates atomics (T10), and the GPU-authored active count is the deferred T13 proof, so
  the count itself is the host-mirrored pool length; what the GPU derives without
  readback is the active count (count clamped by allocation and a settings budget)
  and the dispatch arguments.
