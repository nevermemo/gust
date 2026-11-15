```

The repository also has `gust-algo`, a workspace crate (not published) of device-resident `reduce`, `scan` and `compact`
over `GpuVec`, built from kernels of this kind.

## Capability variants (G10)

`HeadlessDevice::new()` opens the device with the capabilities of the build (`gust::capability::enabled()`). An adapter
that lacks one cannot open the device, which is `Error::MissingCapability` and never a wrong result; a kernel whose
`requires` the device was not opened with is the same error at launch, and a workgroup that is not whole subgroups is
`Error::SubgroupSize`. `HeadlessDevice::subgroup_sizes()` reports the adapter's range. `DESCRIPTOR.requires` tells the
program which body its build selected. The same program runs the portable body under the default profile and the
subgroup body under `GUST_PROFILE=native+subgroups`, from committed artifacts that hold both bodies, and checks the
three cells against a host reference either way.

```rust,no_run
