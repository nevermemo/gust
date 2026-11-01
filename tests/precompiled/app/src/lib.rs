// The library half of the precompiled fixture's `app`. It holds what only a dual build (`--cfg gust_dual`, D26) needs:
// the observer of E5 in `cargo xtask check-precompiled`, which builds this workspace with the cfg and the strict
// profile (`GUST_PROFILE=native`) together. The binary in `main.rs` is what E1 runs on the GPU; in a normal build this
// library is empty, so it adds nothing to that binary.

/// The host reference of `kernels::vector_add::add`: what the kernel computes for every element.
#[cfg(gust_dual)]
pub fn host_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(left, right)| left + right).collect()
}

/// Dispatches `kernels::vector_add::add` on the CPU twin (`gust::cpu::CpuDevice`) and compares the result with the
/// host reference. The twin exists only under the cfg, which is why this function does too; a build without it has
/// no such function (E3 of `check-dual` pins that for the workspace fixture).
#[cfg(gust_dual)]
pub fn twin_smoke() -> bool {
    use gust::cpu::CpuDevice;
    use kernels::vector_add;

    let device = CpuDevice::new();
    let a = device.buffer(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]);
    let b = device.buffer(&[10.0_f32, 20.0, 30.0, 40.0, 50.0]);
    let out = device.buffer(&[0.0_f32; 5]);
    if vector_add::add::dispatch(&device, 5, &a, &b, &out).is_err() {
        return false;
    }
    out.read() == host_add(&a.read(), &b.read()) && out.read() == [11.0, 22.0, 33.0, 44.0, 55.0]
}

#[cfg(all(test, gust_dual))]
mod tests {
    use kernels::vector_add;

    #[test]
    fn the_cpu_twin_matches_the_host_reference() {
        assert!(super::twin_smoke());
        // The strict profile embeds the committed artifact beside the twin: the descriptor of the dual build is the
        // one of the production build, plus the twin.
        // E5 observes the strict profile: with `GUST_PROFILE` dropped from the step, the artifact count above would still be
        // 1 (under `dev` too), but this name would not be `native`.
        assert_eq!(gust::profile::NAME, "native");
        assert_eq!(vector_add::add::DESCRIPTOR.artifacts.len(), 1);
    }
}
