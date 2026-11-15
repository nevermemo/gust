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
`default_device()`, the process-wide device used by calls that name none). The native backend asks wgpu for Vulkan
only.

A typed buffer is created from a slice with `create_typed_buffer(name, &data, access)`, where `access` is
`GpuBufferAccess::ReadOnly` or `ReadWrite`, and it is read back with `read_typed_buffer`. The element type is any
`GpuPod`: 32-bit scalars, 2- and 4-wide vectors, 2x2 and 4x4 matrices, fixed-size arrays of the stored lengths, and
structs of these with no implicit padding. `try_create_typed_buffer` and `create_zeroed_buffer` check the device's
`max_buffer_size` first and return `Error::DeviceLimit`; `create_typed_buffer` itself does not.

## Typed calls

For every kernel the `#[gpu]` macro generates three functions, and the first listing uses two of them:

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
and a one-dimensional launch covers at most `workgroup_size.x` times 65 535 elements on default limits.

The program below is gallery entry G2, `saxpy-uniform`. It dispatches once with the typed `dispatch`, then builds the
same launch as a value with `call`, changes the uniform with `write_uniform` and submits the call with `dispatch_call`:
the second dispatch sees the new value.

```rust,no_run
