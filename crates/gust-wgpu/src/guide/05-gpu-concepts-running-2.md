```

The staged graph refuses kernels with a texture or sampler parameter, binding one texture to two parameters is refused,
and a `pixel` call is one render pass of its own, in batch order.

## Workgroup memory and barriers (G6)

A kernel with workgroup memory has no CPU twin, so its evidence is the GPU run against an independent host reference.
`reduce` and `scan` below chain passes: each pass leaves one value per tile, and the next pass works on those, until
one tile is left (the sum) or the offsets are added back (the scan). Both use `Independent` for the buffers that are
shorter than the pass has invocations, and the program checks sizes below, at and above the tile of 64, ones that are
not a multiple of it, and 100 000, where three levels are needed.

```rust,no_run
