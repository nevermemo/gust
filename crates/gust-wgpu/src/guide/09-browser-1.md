# The browser

A kernel that runs on a native Vulkan device can also run in a browser, on the page's own WebGPU adapter, from the same
Rust source. This chapter walks through the gallery entry `web-hello`
([`gallery/web-hello`](https://github.com/nevermemo/gust/tree/main/gallery/web-hello) in the repository): the sum of
`hello-gust` ([chapter 2][`gust::guide::install_and_first_program`]) as a page that adds two vectors on the GPU and
compares the result with a plain Rust reference. It covers what a browser changes in the calls you write, how the page
is built, what the repository checks on every change, and what that check cannot say.

## What a browser changes

On `wasm32-unknown-unknown`, wgpu uses its WebGPU backend, and nothing may block or read a clock. So this crate offers
only what can run there: `HeadlessDevice::new_async`, the buffer and upload functions, the typed `submit` calls and
`read_typed_buffer_async`. The blocking read-backs, the `_timed` functions, the blocking batch functions, the
execution-graph module and `HeadlessDevice::pool_read` are compiled out, so a call to one of them fails to compile. A
job cannot block either: its status is `Pending` until wgpu reports the submitted work done, and `wait` returns
`Error::WouldBlock`. The generated `dispatch` submits its work, which will run, and then returns that error at run time.
In a browser you submit, and then await a read-back.

For an element-wise map (a `#[gpu]` function on plain scalars) the macro generates the awaiting form for you:
`add::map_async(&a, &b).await` opens the default device (`default_device_async`, which in the browser is
asynchronous), uploads the slices, runs the generated kernel and awaits the read-back. The page below uses only that.
The blocking `map` and `map_on` that the macro generates on native targets are not emitted on `wasm32`, so
`add::map(..)` in a page is a compile error. Use `map_async`, `map_on_async`, or `map_gpu` with `GpuVec::to_vec_async`
(after `default_device_async` has run; until then `map_gpu` finds no adapter).

## The program

The kernel is a scalar function. `#[gpu]` on such a function keeps it as written, and adds a module of the same name
with the kernel and the calls that run it (`map_async` and, on native targets, `map`, among others). The reference is
plain Rust over slices; it never goes through the kernel, so a match is evidence about the kernel and not about itself.
The inputs are small integers, so every sum is exact in `f32` on any target and every comparison can be exact. Each
block of this chapter is a packaged copy of an anchored region of `gallery/web-hello/src/lib.rs`, and
`cargo xtask check-gallery` keeps the copy identical to it.

This first block is the host half of the page: the kernel and the reference. This crate's doctests compile it on the
host, with the real macro and backend, and run its empty `main` (the hidden `fn main() {}` stands in for the
`wasm32`-only code that uses it). Compiling it checks that the kernel is accepted and that its calls exist; it does not
run the kernel on any device.

```rust
