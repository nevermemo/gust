# Running kernels

[Chapter 3][`gust::guide::authoring_model`] wrote kernels. This chapter runs them on a GPU with the headless wgpu
backend of this crate: how a kernel is called, how data gets to and from the device, how several calls are submitted
together, and the resident collections for programs that keep data on the device between calls. Every listing here is a
complete program that this crate's doctests compile (`no_run`: running one needs a Vulkan compute adapter). Each is a
packaged copy of an entry of the repository's `gallery/` directory, and `cargo xtask check-gallery` keeps the copy
identical to the entry.

## The device and its buffers

`HeadlessDevice::new()` opens a Vulkan compute adapter with wgpu's portable default limits, the ones a browser always
has. `HeadlessDevice::with_options` takes `DeviceOptions` to choose the adapter (high performance, low power, an index
or a name) and the limits policy (the portable ones, or the adapter's own, which is the default of
`default_device()`, the process-wide device used by calls that name none). By default the native backend asks wgpu for
Vulkan only; `DeviceOptions::with_backends` changes that.

Every kernel is built with wgpu's runtime checks on. `unsafe { DeviceOptions::default().unchecked_kernels() }` turns
them off for every kernel of the device it opens (`HeadlessDevice::shader_checks` reports it). With unchecked kernels
wgpu's runtime checks are off for every kernel of this device. An out-of-range index, and a loop that does not end, are
undefined behaviour on the GPU: other buffers of the device may be read or written, and the device may hang or be lost.
GUST does not add its own guards on this route. The CPU twin is unchanged and remains the reference; run a kernel
checked first. Integer division and remainder lose wgpu's guards too, so a zero divisor, and `MIN / -1` for a signed
type, are undefined behaviour as well. The option is measured on loop-heavy kernels on the RTX 5090 only; there was no
gain on the Radeon 890M. The setting covers GUST's own library kernels too: options passed to `set_default_options`
make the default device unchecked, so `gust-algo` operations and batch stages that index with values read from data run
unchecked as well. A device adopted with `HeadlessDevice::adopt` is always checked, the SPIR-V route is not affected, and
in a browser the option is accepted and ignored, because the browser checks every shader itself.

A typed buffer is created from a slice with `create_typed_buffer(name, &data, access)`, where `access` is
`GpuBufferAccess::ReadOnly` or `ReadWrite`, and it is read back with `read_typed_buffer`. The element type is any
`GpuPod`: 32-bit scalars, 2- and 4-wide vectors, 2x2 and 4x4 matrices, fixed-size arrays of the stored lengths, and
structs of these with no implicit padding. `try_create_typed_buffer` and `create_zeroed_buffer` check the device's
`max_buffer_size` first and return `Error::DeviceLimit`; `create_typed_buffer` itself does not.

## Sharing a device with a renderer

A renderer that presents to a window must present from the device GUST runs on, and a device GUST opened itself cannot
do that: it lives in a private `wgpu::Instance`, so configuring a surface of another instance with it panics inside
wgpu-core, or silently binds the surface to an unrelated device. So the renderer opens the device and GUST adopts it.
The renderer creates the instance, the surface and the adapter, asks `DeviceOptions::device_request(&adapter)` for the
features and limits GUST needs, adds its own (`request.features | wgpu::Features::FLOAT32_BLENDABLE`,
`request.limits.or_better_values_from(&mine)`), requests the device, and calls
`HeadlessDevice::adopt(&adapter, device, queue, gust::capability::enabled())`. `gust_wgpu::wgpu` is the wgpu behind
these types, so a renderer can name the same copy. Wrap the result in a `DeviceHandle` and call `set_default_device`
with it, and `default_device()`, the map calls, `par_for!` and `GpuVec::from_slice` run on the shared device too.

Rules of a shared device:

- Surface, adapter, device and queue come from one `wgpu::Instance`. GUST cannot check this.
- Adopt a device once and share the `DeviceHandle`: two adoptions of one device are two GUST devices whose buffers
  refuse to mix.
- GUST's blocking calls wait for every submission on the queue, so a GUST readback also waits for the renderer's frames
  in flight, and the renderer's callbacks can run inside a GUST call.
- An error scope the renderer holds on the same thread also catches errors from GUST's calls that cannot return one.
  GUST installs no uncaptured-error handler and no device-lost callback; the renderer's stay in force.
- GUST never calls `destroy`. The renderer's `destroy`, and a device loss, end both users.
- A feature the renderer did not enable degrades and is reported (`supports_gpu_timestamps`,
  `supports_coherent_storage`, `float32_filterable`); a capability whose feature is missing is
  `Error::MissingFeatures`, and an adapter that is not the device's is `Error::AdapterMismatch`.
- In a browser the page creates the instance, the canvas surface, the adapter and the device, adopts with an empty
  capability set, and calls `set_default_device` with an `Rc`.

`examples/window-share` in the repository is a window that does this and presents 60 frames whose colour comes from a
GUST kernel; it is a workspace of its own, built only by `cargo xtask check-window-example`.

## Looking at a buffer

`device.inspect(&buffer)` shows what a kernel wrote, with the field names of the element layout as columns:
`.range(100..110)` narrows it, printing it (`{}`) prints the table, `.table()` returns the rows, `.stats()` the
minimum, maximum and NaN count per field, `.to_json()` the same range as JSON, and `.snapshot()` copies the range so
that `snapshot.changed(&device)` lists the elements a later dispatch changed. Each call is one readback of the range,
so the buffer must be read-write; any other buffer returns `InspectError::Readback` with a hint. The `_async` forms
are the ones that exist on `wasm32`.

## Typed calls

For every kernel the `#[gpu_module]` macro generates three functions, and the first listing uses two of them:

- `call(element_count, resources..)` builds a reusable `gust::Call` that names the bound resources;
- `submit(device, element_count, resources..)` starts it and returns a job;
- `dispatch(device, element_count, resources..)` starts it and waits.

The resources are one reference each, in the kernel's parameter order: a buffer for a storage parameter and a
`GpuUniform<T>` from `create_uniform` for a `Uniform<T>`. The compiler checks that every argument is a buffer of the
element type its parameter declares, that there is one argument per resource, and that a buffer is not where a uniform
belongs (or the reverse). It does not check access (a read-only buffer where the kernel writes is
`Error::PersistentBufferAccess` before any GPU work), the order of two buffers of the same element type, or which
device a buffer came from (a buffer of another device is rejected at submit). At submit the backend also checks every
binding against the kernel descriptor and the launch against the device limits, before anything is compiled or encoded.

`element_count` is the number of invocations, and a buffer must hold exactly that many elements unless the argument is
wrapped as `Independent(&buffer)` at the call site. That wrapper says "this buffer has its own length", as a settings
buffer, a histogram's bins or a reduction's output does; the kernel must then guard it with its own `.len()`.
Dispatches may also be two- or three-dimensional: `call_extent`, `submit_extent` and `dispatch_extent` take an `Extent`,
and a launch past the device's 65 535 workgroups per dimension is cut by the host into pieces that run in one
submission, each invocation running once with the ids of an unsplit launch (a staged graph node keeps the
`DeviceLimit`, and so does a kernel whose split has neither a precompiled artifact nor a run-time compiler to build it;
the host never checks an indirect dispatch's count).

A kernel's pipeline is created once per device, the first time a launch of it has work: the first non-empty dispatch of
a typed call, a batch or a staged graph's first `execute`, and a batch sealed by `gust::Batch` over a non-empty extent
creates its stages' kernels at build; a stage sealed over an empty extent creates its kernel at its first non-empty
frame. The split variant of a launch past a workgroup limit is a second kernel, created when such a launch first runs.
`HeadlessDevice::precompile(&[&kernel::DESCRIPTOR, ..])` creates the named descriptors now, in order, and reports for
each whether it was created or already there and the CPU time it took (a "precompiled artifact" is the build-time WGSL;
this call creates the device's pipeline from it). Name the split variant (`kernel::DESCRIPTOR.split`) to create it
too. Later dispatches of those descriptors, typed or batched, are cache hits (`pipeline_cache_stats` counts them).
"Created" is the promise: a driver may do more work at the first submission, and in a browser the objects are
created but the browser may still be compiling.

The program below is gallery entry G2, `saxpy-uniform`. It dispatches once with the typed `dispatch`, then builds the
same launch as a value with `call`, changes the uniform with `write_uniform` and submits the call with `dispatch_call`:
the second dispatch sees the new value.

```rust,no_run
