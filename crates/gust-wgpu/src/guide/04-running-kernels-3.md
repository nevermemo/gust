```

Floating-point results of a GPU kernel are within the bounds of section 4 of
[SEMANTICS](https://github.com/nevermemo/gust/blob/main/docs/SEMANTICS.md), not guaranteed bit for bit against the
CPU: a device may fuse a multiply and an add. G4 avoids the question by choosing powers of two and small integers, so
every value is exact; a program with arbitrary floats should compare with a tolerance, as G2's tests do.

## Staged graphs

`StagedGraph` is a graph of uploads, dispatches and read-backs, for programs that mix them. A node is an
upload (`write`), a typed call (`run`), an indirect call (`run_indirect`) or a read-back (`read`), and the graph derives
each node's edges from the buffers it touches and the access the kernel declares, so a later node waits for the last
writer of a buffer it reads, and a writer waits for the earlier readers. Explicit edges are available too, and the two
styles may be mixed. `device.execute_graph(&graph)` runs the nodes in insertion order as one submission, rejects a graph
whose edges do not cover the hazards its nodes create, and returns a report with the transfer bytes, the bytes that
stayed resident and the dispatch and edge counts. `device.execute_graph_timed(&graph)` runs the same graph and also reports `gpu_elapsed`, the GPU time from the first dispatch pass to the end of the last (`None` without timestamp queries or without a dispatch pass); `execute_graph` takes no timestamps and costs nothing extra. Uniforms can be written as nodes; a kernel with a texture or sampler
parameter cannot be a node.

## Resident data: `GpuVec`, pools and indirect dispatch

- `GpuVec<T>` is a vector that lives on one device. It is taken from that device's `BufferPool` (size classes, a 1 GiB
  cap on idle bytes, evicting the least recently returned first) and returned on drop. `from_slice` and `zeroed` use
  the default device and their `_on` forms a device you hold; `to_vec` reads it back; `&gpu_vec` goes where a typed
  call takes a buffer, and its binding shows the logical length, not the pooled size. Resident arguments of two devices
  are `Error::MixedDevices`.
- `GpuPool<T>` is a growable resident collection with an explicit capacity, a host-tracked logical length mirrored into
  a one-element count buffer for kernels, and geometric growth that copies the live prefix on the GPU, never through
  the host.
- A buffer from `create_indirect_buffer` can drive `StagedGraph::run_indirect`, where the GPU writes the workgroup
  count. The host never sees that count, so it is not checked: the kernel that writes the arguments must clamp. The
  contract is in
  [ENGINE_HURRICANE](https://github.com/nevermemo/gust/blob/main/docs/ENGINE_HURRICANE.md).

## Errors, limits and the older paths

Failures are `gust_wgpu::Error` values: a missing adapter, `DeviceLimit` (a workgroup size, a workgroup count, a buffer
or a binding over the device's limit, checked before compilation), `MissingCapability` (a capability the adapter lacks,
see [chapter 5][`crate::guide::gpu_concepts_running`]), the access and device-instance errors above and the like. The
earlier paths that take slices (`dispatch_f32`, `dispatch_u32`, `F32Binding`, `U32Binding` and the `*_f32_*` helpers) are
deprecated since 0.2.0 and still work; the typed calls replace them. A `BufferDispatch` built by hand is not
deprecated.

Next, [chapter 5][`crate::guide::gpu_concepts_running`] runs textures, workgroup memory and subgroup variants, and
[chapter 6][`crate::guide::profiles_artifacts_routing`] says how a kernel gets from source to a binary that needs no
Slang.
