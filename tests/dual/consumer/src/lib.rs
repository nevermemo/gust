// The consumer of the cross-crate fixture. Its tests dispatch `kernel_lib::kernels::add` on `gust::cpu::CpuDevice`.
// Without `--cfg gust_dual` that does not compile (the twin lives in `kernel-lib`, which this crate's `cfg(test)`
// does not reach): `cargo xtask check-dual` pins that error (E1) and, with the cfg, that the test passes (E2).

/// The host reference of `kernel_lib::kernels::add`: what the kernel computes for every element.
pub fn host_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(left, right)| left + right).collect()
}

#[cfg(test)]
mod tests {
    use gust::cpu::{CpuDevice, OverflowChecks};
    use kernel_lib::kernels;

    use super::host_add;

    #[test]
    fn twin_runs() {
        let device = CpuDevice::new();
        let a = device.buffer(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let b = device.buffer(&[10.0f32, 20.0, 30.0, 40.0, 50.0]);
        let out = device.buffer(&[0.0f32; 5]);
        kernels::add::dispatch(&device, 5, &a, &b, &out).unwrap();
        assert_eq!(out.read(), host_add(&a.read(), &b.read()));
        assert_eq!(out.read(), [11.0, 22.0, 33.0, 44.0, 55.0]);
    }

    /// A dependent with a per-package overflow override sees its own mode through the macro
    /// `gust::cpu::overflow_checks!()`, which expands in this crate, while the function answers for `gust`.
    /// `tests/dual/Cargo.toml` turns `overflow-checks` off for this package in the dev profile (so the test
    /// profile); `gust` keeps the dev default, on. The assertion on `gust`'s answer holds in the dev and release
    /// profiles (debug assertions on or off), the one on this crate's answer in both as well, because the
    /// release default is off.
    #[test]
    fn overflow_mode_is_the_dependents_own() {
        let in_gust = gust::cpu::overflow_checks();
        let in_consumer = gust::cpu::overflow_checks!();
        assert_eq!(in_consumer, OverflowChecks::Wrapping, "this package");
        let gust_is_checked = cfg!(debug_assertions);
        assert_eq!(
            in_gust,
            if gust_is_checked {
                OverflowChecks::Checked
            } else {
                OverflowChecks::Wrapping
            },
            "gust"
        );
        assert_eq!(in_gust != in_consumer, gust_is_checked);
    }
}
