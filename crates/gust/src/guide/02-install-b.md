```

The host side creates a headless device, uploads three buffers, calls the kernel through its generated typed
`dispatch`, and reads the result back. It needs `gust-compute-wgpu`, which this crate cannot depend on, so the block is
marked `ignore` here; the whole program (imports, kernel and `main` together) is compiled by the doctests of
[`gust_wgpu::guide`](https://docs.rs/gust-compute-wgpu) (they run in the same test gates):

```rust,ignore
