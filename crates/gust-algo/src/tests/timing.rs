//! The accumulate rule of `timed` (D194): no adapter needed. The GPU tests of the scope are in `gpu.rs`.

use std::time::Duration;

use crate::{
    Error, GpuTime, timed,
    timing::{Accum, note_batch, scope_is_open},
};

const MS: Duration = Duration::from_millis(1);

#[test]
fn a_scope_with_no_batch_has_no_time() {
    let result = timed(|| Ok(7)).unwrap();
    assert_eq!(result.value, 7);
    assert_eq!(result.time, GpuTime::default());
    assert_eq!(result.time.gpu_elapsed, None);
    assert_eq!(result.time.batches, 0);
}

#[test]
fn batches_add_up_and_the_count_follows() {
    let result = timed(|| {
        note_batch(Some(2 * MS), true);
        note_batch(Some(3 * MS), true);
        Ok(())
    })
    .unwrap();
    assert_eq!(result.time.gpu_elapsed, Some(5 * MS));
    assert_eq!(result.time.batches, 2);
}

#[test]
fn an_empty_batch_on_a_timestamp_device_adds_zero_and_one_without_support_poisons_the_scope() {
    let supported = timed(|| {
        note_batch(Some(2 * MS), true);
        note_batch(None, true);
        Ok(())
    })
    .unwrap();
    assert_eq!(supported.time.gpu_elapsed, Some(2 * MS));
    assert_eq!(supported.time.batches, 2);

    let unsupported = timed(|| {
        note_batch(Some(2 * MS), true);
        note_batch(None, false);
        Ok(())
    })
    .unwrap();
    assert_eq!(unsupported.time.gpu_elapsed, None);
    assert_eq!(unsupported.time.batches, 2);
}

#[test]
fn a_nested_scope_reports_its_own_time_and_the_outer_one_includes_it() {
    let outer = timed(|| {
        note_batch(Some(MS), true);
        let inner = timed(|| {
            note_batch(Some(4 * MS), true);
            Ok(())
        })?;
        assert_eq!(inner.time.gpu_elapsed, Some(4 * MS));
        assert_eq!(inner.time.batches, 1);
        note_batch(Some(MS), true);
        Ok(())
    })
    .unwrap();
    assert_eq!(outer.time.gpu_elapsed, Some(6 * MS));
    assert_eq!(outer.time.batches, 3);
}

#[test]
fn an_error_and_a_panic_end_the_scope_and_a_batch_outside_one_is_not_counted() {
    assert!(!scope_is_open());
    let error = timed(|| {
        assert!(scope_is_open());
        note_batch(Some(MS), true);
        Err::<(), _>(Error::Empty)
    });
    assert!(matches!(error, Err(Error::Empty)));
    assert!(!scope_is_open(), "an error left its scope open");

    let caught = std::panic::catch_unwind(|| {
        let _ = timed(|| -> Result<(), Error> {
            note_batch(Some(MS), true);
            panic!("inside the scope");
        });
    });
    assert!(caught.is_err());
    assert!(!scope_is_open(), "a panic left its scope open");

    // A nested inner scope that fails: the outer scope is still the open one, and after it ends nothing is open.
    let outer = timed(|| {
        let inner = timed(|| {
            note_batch(Some(MS), true);
            Err::<(), _>(Error::Empty)
        });
        assert!(inner.is_err());
        assert!(scope_is_open(), "the outer scope must still be open");
        Ok(())
    });
    assert!(outer.is_ok());
    assert!(!scope_is_open(), "a nested error left a scope open");

    // Nothing of the scopes above is left behind: a batch outside any scope counts nowhere, a new scope starts at 0.
    note_batch(Some(MS), true);
    let fresh = timed(|| Ok(())).unwrap();
    assert_eq!(fresh.time, GpuTime::default());
    assert!(!scope_is_open());
}

#[test]
fn an_enclosing_scope_counts_the_batches_of_an_inner_scope_that_failed() {
    let outer = timed(|| {
        let inner = timed(|| {
            note_batch(Some(3 * MS), true);
            Err::<(), _>(Error::Empty)
        });
        assert!(matches!(inner, Err(Error::Empty)));
        Ok(())
    })
    .unwrap();
    assert_eq!(outer.time.batches, 1);
    assert_eq!(outer.time.gpu_elapsed, Some(3 * MS));
}

/// The promise of the module doc, kept by a scan: every dispatch of a public operation goes through
/// `timing::dispatch_batch`, so none of the other production modules may call the device's batch functions itself.
#[test]
fn no_module_but_timing_dispatches_a_batch_itself() {
    let modules: [(&str, &str); 14] = [
        ("bound.rs", include_str!("../bound.rs")),
        ("compact.rs", include_str!("../compact.rs")),
        ("compact_kernels.rs", include_str!("../compact_kernels.rs")),
        ("cpu.rs", include_str!("../cpu.rs")),
        ("error.rs", include_str!("../error.rs")),
        ("evidence.rs", include_str!("../evidence.rs")),
        ("kernels.rs", include_str!("../kernels.rs")),
        ("lib.rs", include_str!("../lib.rs")),
        ("limits.rs", include_str!("../limits.rs")),
        ("reduce.rs", include_str!("../reduce.rs")),
        ("scan.rs", include_str!("../scan.rs")),
        ("scan_kernels.rs", include_str!("../scan_kernels.rs")),
        ("schedule.rs", include_str!("../schedule.rs")),
        ("variant.rs", include_str!("../variant.rs")),
    ];
    // The list covers every module lib.rs declares (timing and the test module aside), so a new module cannot be left
    // out of the scan.
    let lib = include_str!("../lib.rs");
    let declared: Vec<&str> = lib
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let rest = line
                .strip_prefix("pub mod ")
                .or_else(|| line.strip_prefix("mod "))?;
            rest.strip_suffix(';')
        })
        .filter(|name| !matches!(*name, "timing" | "tests"))
        .collect();
    assert!(declared.len() >= 13, "found only {declared:?} in lib.rs");
    for module in &declared {
        let file = format!("{module}.rs");
        assert!(
            modules.iter().any(|(name, _)| *name == file),
            "lib.rs declares `{module}`, which the source scan does not list; add `{file}`"
        );
    }
    for (name, source) in modules {
        // Test code of a module (a `#[cfg(test)]` item) may call the device directly.
        let production = source.split("#[cfg(test)]").next().unwrap_or(source);
        for needle in [
            ".dispatch_batch(",
            ".dispatch_batch_timed(",
            ".submit_batch(",
            "HeadlessDevice::dispatch_batch",
            "HeadlessDevice::submit_batch",
        ] {
            assert!(
                !production.contains(needle),
                "{name} calls `{needle}` itself; use crate::timing::dispatch_batch so that `timed` sees the batch"
            );
        }
    }
}

#[test]
fn the_accumulator_is_the_sum_of_its_batches() {
    let mut accum = Accum::default();
    accum.record(Some(MS), true);
    accum.record(Some(MS), true);
    assert_eq!(accum.time().gpu_elapsed, Some(2 * MS));
    assert_eq!(accum.time().batches, 2);
}
