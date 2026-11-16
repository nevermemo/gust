# Execution graph and CPU/GPU domains

Status: researched direction. Current execution is host-driven batches and one
`StagedGraph` that runs nodes in insertion order; there is no graph compiler,
reordering, automatic partitioning, or transfer planner. Edges are derived only for
typed nodes, from the buffers and access modes their calls declare, never from
kernel bodies. Start with `examples/typed-pipeline`, which already proves three ordered
dispatches over different resident types, and extract only what another use requires.

A future graph describes high-level execution nodes and resource-version edges,
not shader instructions. Candidate nodes: CPU task, GPU direct/indirect dispatch,
render pass, upload, readback, copy, join, and continuation. Each node declares
reads/writes, execution domain, observable effects, and dependencies. Producer/consumer
ordering and resource transitions must be lowered for the selected backend; an
edge is not itself a universal GPU barrier.

## First bounded proof

One CPU settings update → typed GPU stage A → GPU stage B → small summary readback.
Verify output, ordering, intermediate GPU residency, and transfer byte counts.
Initially require explicit host-provided dependencies and reject invalid resources;
do not pretend the syn frontend can infer all effects or aliases.

Implemented (T07, 2026-09-07) as `gust_wgpu::StagedGraph` in
`crates/gust-wgpu/src/graph.rs` and `examples/staged-graph`. A graph holds
upload, dispatch, and readback nodes in insertion order with host-declared
dependencies (M2 added typed nodes with derived edges; see below).
`HeadlessDevice::execute_graph` validates before creating any GPU
object: dependencies must point at earlier nodes, every node's buffers must pass the
existing persistent-dispatch checks, and every earlier node that conflicts on a
buffer (either side writes) must be a transitive ancestor — otherwise
`GraphMissingDependency` names both nodes. Uploads are encoded as buffer copies so
they are ordered inside the same submission as the dispatches; readbacks copy to
staging and are mapped after the submission fence. The `GraphReport` records each
transfer's bytes, dispatch/workgroup counts, and the bytes of buffers that only
dispatches touched (resident). The example keeps 2·N·4 bytes resident, uploads 16
bytes, and reads back N/8 summaries, verified against an independent host reference
for N in {0, 1, 7, 8, 9, 63, 64, 65, 257, 1000} and across two settings updates.
Since m10.ts2 (D194), `HeadlessDevice::execute_graph_timed` runs the same graph and fills `GraphReport::gpu_elapsed`
with the GPU time from the first dispatch pass's begin to the last one's end (timestamps exist only at pass
boundaries, so copies between dispatches are inside the span); it holds the device's one timestamp profiler for the
run, and `execute_graph` never takes it.
Still true: nodes execute in insertion order; edges, declared or derived, are
checked, not used for scheduling. No reordering, no transfer planner.

`BufferBinding::independent_length()` was added for this proof: a one-element
settings buffer and an N→N/8 reduction are impossible under the shared
element-count rule. It is opt-in per binding, the kernel must guard that buffer
with its own `.len()`, and an empty independent buffer is rejected when the
dispatch has work. In typed calls the same opt-in is `Independent(&buffer)` at the
call site.

## Derived edges (M2)

The typed methods `write`, `run` (a typed `gust::Call`), `run_indirect` and `read`
derive the edges of the node they append; `upload`, `dispatch`, `dispatch_indirect`
and `readback` keep taking host-declared edges, and both kinds can be mixed in one
graph. Derivation reads the buffers a node touches and the access the kernel
declares for each (write for an upload, read for a readback, the descriptor's
access for a dispatch; the arguments buffer of an indirect dispatch counts as a
read). It never reads a kernel body, so the kernel's real behaviour can be
narrower than its declaration but not wider.

For every buffer a new node touches, its edges are the latest earlier node that
wrote the buffer and, when the new node writes it, every earlier node that touched
it since that writer. The result is sorted and deduplicated. It is conservative,
not a transitive reduction: a node may list an edge another edge already implies.
Nodes still run in insertion order, so an edge is a checked contract, not a
scheduling hint.

The whole graph is validated at execution, whichever way its edges were made: a
node with a missing edge is `GraphMissingDependency`, so a derived node does not
repair an invalid explicit node. `StagedGraph::dependencies(node)` returns a node's
edges (in declaration order for explicit nodes, ascending for derived ones),
`dependency_indices(node)` the same as plain indices, and `is_derived(node)` says
which kind it is.

`GraphReport` counts `derived_edges` (edges recorded on typed nodes) and
`declared_edges` (distinct edges the host declared on explicit nodes), alongside
the transfer, residency, dispatch and workgroup figures. `dispatches` counts the
dispatch nodes that produced GPU work, and `indirect_dispatches` those whose
workgroup count the GPU supplied; indirect dispatches are not added to
`workgroups`, because the host never learned that count.

An indirect dispatch (`run_indirect` or `dispatch_indirect`) is limit-checked on the
host for kernel shape (workgroup width, invocations per workgroup, storage buffer
count), binding indices and binding sizes only. The workgroup count
in the arguments buffer is not checked on the host. With wgpu's default instance
flags (`InstanceFlags::VALIDATION_INDIRECT_CALL`, which `HeadlessDevice` keeps) wgpu
silently zeroes a dispatch whose count exceeds the real device's per-dimension limit
on Vulkan (wgpu-core also needs the adapter's `INDIRECT_EXECUTION` and
`COMPUTE_SHADERS` downlevel flags, which Vulkan adapters set); without that flag the count reaches the GPU unvalidated. Either way the kernel
that writes the arguments must clamp. `HeadlessDevice::with_limits` does not apply
to indirect counts, because wgpu enforces the limits of the device it opened. An
arguments buffer that the same dispatch also binds read-write is rejected
(`AliasedBinding`); binding it read-only is accepted.

Async/await/spawn/join should ultimately mean dependencies and continuations.
Existing `WgpuJob` is an asynchronous submission handle, not shader async support.
Waiting must happen at a host boundary or a later dispatch, never as a resident
shader spinning for another workgroup.

Dynamic `while_gpu` candidates perform a bounded local budget, persist continuation
or frontier data, write the next active count, and build an indirect dispatch.
Termination, maximum work budgets, cancellation, and malformed counts need explicit
contracts before implementation. A dependency cycle is not fixed by a spin loop.

## Residency

Keep component/particle/geometry buffers on the GPU across stages and frames. Move
only required live-in/live-out values; intentional CPU work can use filesystem,
network, input, assets, or UI. A CPU node is not a substitute chosen silently when
a shader fails to compile. Buffer growth is a GPU-to-GPU copy, not a readback and
reupload. Ownership and safe retirement must account for in-flight submissions.

Automatic loop extraction, effect inference, Rust semantic integration, and graph
optimization are later milestones with independent tests and decision records.
