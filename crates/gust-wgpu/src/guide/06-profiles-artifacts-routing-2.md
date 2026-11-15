```

There is no fallback: a GPU-routed kernel called on a handle with no GPU is `Error::NoGpu`, never a silent CPU run. A
`RoutedDevice` covers storage-buffer kernels only, and not with a GPU on `wasm32`. A kernel called on a named device is
a call-site change when its route flips (a CPU-routed kernel is called on `CpuDevice`, a GPU-routed one on
`HeadlessDevice`), which is what the routed handle removes.
