```

Floating-point results of a GPU kernel are within the bounds of section 4 of
[SEMANTICS](https://github.com/nevermemo/gust/blob/main/docs/SEMANTICS.md), not guaranteed bit for bit against the
CPU: a device may fuse a multiply and an add. G4 avoids the question by choosing powers of two and small integers, so
every value is exact; a program with arbitrary floats should compare with a tolerance, as G2's tests do.

## Batches in your own encoder

A program that records its own passes on a shared device (`HeadlessDevice::adopt`) should not pay a queue submission
for every GUST batch: each extra submission costs host time every frame.
`device.encode_batch(&mut encoder, &dispatches)`
records a batch into the encoder you pass and never submits it, so a frame can be GUST batch, your own pass, GUST
batch, one `finish` and one `queue.submit`. `EncodedJob::submitted(&index)` takes the index that `queue.submit`
returned and gives back the same `WgpuJob` that `submit_batch` returns, which you wait on or poll. `submit_batch` is
those steps with an encoder of its own. `device.submit_batch_read(&dispatches, &clear, &buffer)` is the
one-submission form of a dispatch followed by a read-back (it also zeroes the buffers in `clear`, which saves the
separate submission that a recycled zeroed output otherwise costs, provided the output comes from
`uncleared_buffer`: `GpuVec::zeroed_on` still clears in a submission of its own); `encode_batch_read` is its encode form.
`device.gust_submissions()` counts the submissions GUST itself makes, so a test can state a budget; on a shared device
your own submissions are not in it.

The rules are those of one wgpu queue, which GUST does not change:

- Submissions run as if in submission order, so a later submission sees the writes of an earlier one with no barrier of
  yours; that is an order, not completion. Inside one encoder the order is the order of recording.
- `queue.write_buffer` lands before the next submission, so a write you queue after recording a batch and before your
  `queue.submit` is seen by that batch.
- Completion callbacks and mappings fire when the device is polled (by a blocking GUST call, by your poll, or by wgpu
  itself, possibly inside `queue.submit`); a mapping on a buffer the encoding writes is requested after your submission.
- GUST submits on its own account in `submit_batch`, `submit_batch_read`, `dispatch_batch_timed`, the read-backs
  (including the inspect read-back), `write_typed_buffer_timed`, the deprecated `dispatch_f32` and `dispatch_u32`, a
  recycled zeroed buffer's clear, pool growth and `execute_graph`; `encode_batch` and `EncodedJob::submitted` submit
  nothing. There is no timed `encode_batch`: the timestamp profiler's lock would have to be held across your submission.
- GUST cannot check that your encoder belongs to the GUST device (the one-instance rule of `adopt`), and it does not
  detect a wrong or early submission index: the job then waits for the submission you named.
- Keep every buffer the batch binds, and the `read` and `clear` buffers of the read forms, alive until your
  `queue.submit`: a pooled buffer dropped earlier can be handed out again and written before your submission runs.
- `EncodedRead::submitted` requests the read-back at once, so call it after your `queue.submit`. Called early, your
  submission is rejected (a buffer with a mapping pending); called with no submission behind it, the read-back returns
  stale bytes of an earlier read-back and nothing detects it. Drop an `EncodedRead` and its staging buffer is retired
  (never reused), because your encoder still copies into it. `PendingRead::finish` and `wait` refuse a device other
  than the one that made it (`Error::ForeignPersistentBuffer`).
- An error from `encode_batch` leaves your encoder as it was, except for the attachment of a `pixel` call (out of
  memory), after which you discard the encoder.

## Staged graphs

`StagedGraph` is a graph of uploads, dispatches and read-backs, for programs that mix them. A node is an
upload (`write`), a typed call (`run`), an indirect call (`run_indirect`) or a read-back (`read`), and the graph derives
each node's edges from the buffers it touches and the access the kernel declares, so a later node waits for the last
writer of a buffer it reads, and a writer waits for the earlier readers. Explicit edges are available too, and the two
styles may be mixed. `device.execute_graph(&graph)` runs the nodes in insertion order as one submission, rejects a graph
whose edges do not cover the hazards its nodes create, and returns a report with the transfer bytes, the bytes that
stayed resident and the dispatch and edge counts. `device.execute_graph_timed(&graph)` runs the same graph and also
reports `gpu_elapsed`, the GPU time from the first dispatch pass to the end of the last (`None` without timestamp
queries or without a dispatch pass); `execute_graph` takes no timestamps and costs nothing extra. Uniforms can be
written as nodes; a kernel with a texture or sampler parameter cannot be a node.

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
