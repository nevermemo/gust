```

Three things to see here. The `if i < out.len()` guard is there because a dispatch is rounded up to whole workgroups,
so some invocations of the last workgroup lie past the end of the data: a kernel guards its tail (an out-of-range
access is outside the contract, see below). `p.a` reads one field of the uniform: a `Uniform<T>` takes a struct of the
module, and the macro lays it out as `Std140V1`, which means no implicit padding and a size that is a multiple of 16
bytes; that is why `Params` writes its three `pad` fields out. A struct with implicit padding is rejected at compile
time (GUST-E0208, and E0209 for a uniform). And the parameter order is the binding order: the host passes the uniform,
`x`, `y` and `out` in exactly this order.

The host side of this entry is in [chapter 4][ch4]. Its test module runs the same kernel on the CPU twin and compares
the result with an independent host reference, which is the pattern every gallery entry follows.

## Atomics

When many invocations update one location, a plain read-modify-write loses updates on a GPU. A kernel uses an atomic
instead. G3, `histogram`, counts keys into eight bins, so every bin is hit by thousands of invocations at once:

```rust,no_run
# use gust::gpu;
