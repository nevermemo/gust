// The consumer of the cross-crate fixture. Its tests dispatch `kernel_lib::kernels::add` on `gust::cpu::CpuDevice`.
// Without `--cfg gust_dual` that does not compile (the twin lives in `kernel-lib`, which this crate's `cfg(test)`
// does not reach): `cargo xtask check-dual` pins that error (E1) and, with the cfg, that the test passes (E2); the module `twin_overflow` (cfg only) also dispatches `kernel_lib::kernels::add_one` and this crate's own `own::add_one`, to show whose overflow mode a twin follows. Keep this header at three lines: e1-expected.txt pins the line of `twin_runs`' dispatch.

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

/// The overflow mode of a twin, observed by what it does with an addition that overflows (`u32::MAX + 1`), not by a
/// reporting probe. Only with the cfg: these tests dispatch `kernel_lib`'s kernel, which has no twin without it (E1
/// pins that error on `twin_runs` alone, so no other call of this kind may exist outside `gust_dual`).
///
/// A twin is compiled in the crate that defines its kernel, so it follows that package's `overflow-checks`, not the
/// dependent's. In the dev profile the two packages differ here: `tests/dual/Cargo.toml` switches checks off for this
/// package only, so `kernel-lib`'s twin panics and the twin of a kernel defined in this package wraps. In the release
/// profile both wrap.
#[cfg(all(test, gust_dual))]
mod twin_overflow {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use gust::cpu::{CpuBuffer, CpuDevice, Error, OverflowChecks};

    // A kernel of this package, in the shape of `kernel_lib::kernels::add_one`. Its twin is compiled here, under this
    // package's `cfg(test)`.
    #[gust::gpu]
    mod own {
        #[kernel(workgroup_size(64, 1, 1))]
        pub fn add_one(
            id: SV_DispatchThreadID,
            input: StructuredBuffer<uint>,
            mut out: RWStructuredBuffer<uint>,
        ) {
            let i = id.x;
            if i < out.len() {
                out[i] = input[i] + 1u32;
            }
        }
    }

    /// Dispatches `dispatch` on the single input `u32::MAX` and returns the output, or the panic message.
    fn add_one_to_the_maximum(
        dispatch: impl FnOnce(&CpuDevice, &CpuBuffer<u32>, &CpuBuffer<u32>) -> Result<(), Error>,
    ) -> Result<u32, String> {
        let device = CpuDevice::new();
        let input = device.buffer(&[u32::MAX]);
        let out = device.buffer(&[0u32]);
        catch_unwind(AssertUnwindSafe(|| {
            dispatch(&device, &input, &out)
                .unwrap_or_else(|error| panic!("the twin refused the call: {error}"));
        }))
        .map_err(|payload| {
            if let Some(text) = payload.downcast_ref::<&str>() {
                (*text).to_owned()
            } else if let Some(text) = payload.downcast_ref::<String>() {
                text.clone()
            } else {
                "<non-string panic payload>".to_owned()
            }
        })?;
        Ok(out.read()[0])
    }

    /// Checks on: the addition panics. Checks off: it wraps to 0.
    fn assert_outcome(mode: OverflowChecks, outcome: Result<u32, String>) {
        match mode {
            OverflowChecks::Checked => {
                let message = outcome.expect_err("checks on: u32::MAX + 1 must panic");
                assert!(
                    message.starts_with("attempt to add with overflow"),
                    "{message}"
                );
            }
            OverflowChecks::Wrapping => {
                assert_eq!(outcome, Ok(0), "checks off: u32::MAX + 1 wraps")
            }
            other => panic!("no expectation for the overflow mode {other:?}"),
        }
    }

    #[test]
    fn a_twin_follows_the_overflow_mode_of_the_package_that_defines_the_kernel() {
        let consumer = gust::cpu::overflow_checks!();
        let kernel_lib = kernel_lib::overflow_mode();
        assert_eq!(consumer, OverflowChecks::Wrapping, "this package");
        let kernel_lib_is_checked = cfg!(debug_assertions);
        assert_eq!(
            kernel_lib,
            if kernel_lib_is_checked {
                OverflowChecks::Checked
            } else {
                OverflowChecks::Wrapping
            },
            "kernel-lib"
        );
        assert_eq!(kernel_lib != consumer, kernel_lib_is_checked);
        // The twin of the dependency's kernel does what the dependency's mode says, which in the dev profile is not
        // what this package's mode would do.
        assert_outcome(
            kernel_lib,
            add_one_to_the_maximum(|device, input, out| {
                kernel_lib::kernels::add_one::dispatch(device, 1, input, out)
            }),
        );
    }

    #[test]
    fn a_twin_defined_in_the_dependent_follows_the_dependents_own_mode() {
        let consumer = gust::cpu::overflow_checks!();
        assert_eq!(consumer, OverflowChecks::Wrapping, "this package");
        assert_outcome(
            consumer,
            add_one_to_the_maximum(|device, input, out| {
                own::add_one::dispatch(device, 1, input, out)
            }),
        );
    }
}
