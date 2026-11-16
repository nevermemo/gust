//! The accumulate rule of `timed` (D194): no adapter needed. The GPU tests of the scope are in `gpu.rs`.

use std::time::Duration;

use crate::{
    Error, GpuTime, timed,
    timing::{Accum, note_batch},
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
    let error = timed(|| {
        note_batch(Some(MS), true);
        Err::<(), _>(Error::Empty)
    });
    assert!(matches!(error, Err(Error::Empty)));

    let caught = std::panic::catch_unwind(|| {
        let _ = timed(|| -> Result<(), Error> {
            note_batch(Some(MS), true);
            panic!("inside the scope");
        });
    });
    assert!(caught.is_err());

    // Nothing of the two scopes above is left behind: a batch outside any scope counts nowhere, a new scope starts at 0.
    note_batch(Some(MS), true);
    let fresh = timed(|| Ok(())).unwrap();
    assert_eq!(fresh.time, GpuTime::default());
}

#[test]
fn the_accumulator_is_the_sum_of_its_batches() {
    let mut accum = Accum::default();
    accum.record(Some(MS), true);
    accum.record(Some(MS), true);
    assert_eq!(accum.time().gpu_elapsed, Some(2 * MS));
    assert_eq!(accum.time().batches, 2);
}
