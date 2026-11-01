include!("../../../tests/fixtures/atomics.rs");

use gust::{Independent, cpu::CpuDevice};
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

const INVOCATIONS: u32 = 257;

/// The sentinel of the contended compare-exchange. Besides a thread's `id + 1`, a conforming GPU may leave it
/// unchanged: WGSL's only compare-exchange is the weak one (17.8.5, "The equality comparison may spuriously fail on
/// some implementations"), so every attempt may fail (docs/SEMANTICS.md, KM-09).
const CAS_SENTINEL: u32 = 999;

/// Whether a weak compare-exchange on one cell ended in an allowed state: the exchange happened (`exchanged`) or the
/// comparison failed spuriously and the cell kept its value (`unchanged`). WGSL's only compare-exchange is the weak
/// one (17.8.5, "The equality comparison may spuriously fail on some implementations"), so even an uncontended
/// attempt that should succeed may leave the cell as it was (D47). Any other value, or a readback of the wrong
/// length, is a bug.
fn is_weak_compare_exchange_outcome<T: PartialEq>(
    actual: &[T],
    exchanged: T,
    unchanged: T,
) -> bool {
    matches!(actual, [only] if *only == exchanged || *only == unchanged)
}

/// The whole GPU assertion of the uncontended unsigned success case (cell 7, compare 7, new value 11): the cell ends
/// at 11 (exchanged) or stays 7 (spurious failure, WGSL 17.8.5; D47). Panics otherwise, and prints a NOTE on the
/// spurious branch. It is a function so that `the_weak_compare_exchange_checks_accept_exactly_the_allowed_values`
/// can feed it the spurious value and wrong values without a GPU: the RTX 5090 never takes the spurious branch.
fn check_cas_success(actual: &[u32]) {
    assert!(
        is_weak_compare_exchange_outcome(actual, 11u32, 7u32),
        "uncontended atomic_compare_exchange(7, 11) must end at 11 or, after a spurious failure, at 7, got \
         {actual:?}"
    );
    if actual == [7u32] {
        eprintln!(
            "NOTE atomics cas success: the weak compare-exchange failed spuriously; the cell is unchanged"
        );
    }
}

/// The whole GPU assertion of the signed success case (cell -5, compare -5, new value 11): the cell ends at 11 or
/// stays -5 (spurious failure). See `check_cas_success`.
fn check_signed_cas(actual: &[i32]) {
    assert!(
        is_weak_compare_exchange_outcome(actual, 11i32, -5i32),
        "signed atomic_compare_exchange(-5, 11) must end at 11 or, after a spurious failure, at -5, got {actual:?}"
    );
    if actual == [-5i32] {
        eprintln!(
            "NOTE atomics signed cas: the weak compare-exchange failed spuriously; the cell is unchanged"
        );
    }
}

/// 257 threads all increment the same shared `u32` counter.
/// The final value must equal exactly 257, matching an independent host reference, on the GPU and on the CPU twin.
#[test]
fn atomic_add_contention_produces_exact_sum() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("atomics adapter: {:?}", device.adapter_info());
    let counter =
        device.create_typed_buffer("atomics counter", &[0u32; 1], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &atomics::run::DESCRIPTOR,
            INVOCATIONS,
            &[BufferBinding::read_write(&counter).independent_length()],
        )
        .unwrap();
    let actual = device.read_typed_buffer(&counter).unwrap();
    assert_eq!(
        actual,
        vec![INVOCATIONS],
        "atomic_add under contention must produce exactly {INVOCATIONS}"
    );

    // The twin: the same typed call with the device swapped. A sequential read-modify-write gives the same sum.
    let cpu = CpuDevice::new();
    let twin = cpu.buffer(&[0u32; 1]);
    atomics::run::dispatch(&cpu, INVOCATIONS, Independent(&twin)).unwrap();
    assert_eq!(twin.read(), vec![INVOCATIONS], "twin atomic_add");
    assert_eq!(twin.read(), actual, "twin against the GPU");
}

/// 257 threads each attempt `atomic_min(counter, id.x)`.
/// Initial value is 1000; the minimum of {0..256} is 0.
#[test]
fn atomic_min_produces_minimum() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let counter =
        device.create_typed_buffer("atomics min", &[1000u32; 1], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &atomics::run_min::DESCRIPTOR,
            INVOCATIONS,
            &[BufferBinding::read_write(&counter).independent_length()],
        )
        .unwrap();
    let actual = device.read_typed_buffer(&counter).unwrap();
    assert_eq!(
        actual,
        vec![0u32],
        "atomic_min under contention must produce the minimum thread ID (0)"
    );

    let cpu = CpuDevice::new();
    let twin = cpu.buffer(&[1000u32; 1]);
    atomics::run_min::dispatch(&cpu, INVOCATIONS, Independent(&twin)).unwrap();
    assert_eq!(twin.read(), vec![0u32], "twin atomic_min");
    assert_eq!(twin.read(), actual, "twin against the GPU");
}

/// 257 threads each attempt `atomic_max(counter, id.x)`.
/// Initial value is 0; the maximum of {0..256} is 256.
#[test]
fn atomic_max_produces_maximum() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let counter = device.create_typed_buffer("atomics max", &[0u32; 1], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &atomics::run_max::DESCRIPTOR,
            INVOCATIONS,
            &[BufferBinding::read_write(&counter).independent_length()],
        )
        .unwrap();
    let actual = device.read_typed_buffer(&counter).unwrap();
    assert_eq!(
        actual,
        vec![INVOCATIONS - 1],
        "atomic_max under contention must produce the maximum thread ID ({})",
        INVOCATIONS - 1
    );

    let cpu = CpuDevice::new();
    let twin = cpu.buffer(&[0u32; 1]);
    atomics::run_max::dispatch(&cpu, INVOCATIONS, Independent(&twin)).unwrap();
    assert_eq!(twin.read(), vec![INVOCATIONS - 1], "twin atomic_max");
    assert_eq!(twin.read(), actual, "twin against the GPU");
}

/// 257 threads each `atomic_exchange(counter, id.x)`.
/// The GPU's final value is set-valued (any invocation may be the last writer): it must be in [0, 256]. The twin runs
/// the invocations in ascending order, so its last writer is invocation 256. That is the twin's order rule
/// (docs/SEMANTICS.md, section 6 and KM-09), not an expectation of the GPU, and the two are not compared.
#[test]
fn atomic_exchange_produces_valid_value() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let counter = device.create_typed_buffer(
        "atomics exchange",
        &[9999u32; 1],
        GpuBufferAccess::ReadWrite,
    );
    device
        .dispatch_buffers(
            &atomics::run_exchange::DESCRIPTOR,
            INVOCATIONS,
            &[BufferBinding::read_write(&counter).independent_length()],
        )
        .unwrap();
    let actual = device.read_typed_buffer(&counter).unwrap();
    let value = actual[0];
    assert!(
        value < INVOCATIONS,
        "atomic_exchange must produce a value in [0, {INVOCATIONS}), got {value}"
    );

    let cpu = CpuDevice::new();
    let twin = cpu.buffer(&[9999u32; 1]);
    atomics::run_exchange::dispatch(&cpu, INVOCATIONS, Independent(&twin)).unwrap();
    assert_eq!(
        twin.read(),
        vec![INVOCATIONS - 1],
        "twin atomic_exchange ends with the last writer in ascending order"
    );
}

/// 257 threads each attempt to replace the sentinel value 999 with `id + 1`.
///
/// The GPU's outcome is set-valued: WGSL's only compare-exchange is the weak one (17.8.5), which "may spuriously
/// fail on some implementations", so the cell ends at the winner's `id + 1` (1 to 257) or, if every attempt fails
/// spuriously, at the unchanged sentinel (KM-09). The twin never fails spuriously and runs in ascending order, so
/// invocation 0 wins and the cell ends at 1; that is the twin's order rule, not an expectation of the GPU. Because
/// the sentinel is accepted for the GPU, a no-op GPU atomic would pass the GPU half of this test; the twin half is
/// the guard against a no-op twin, and a GPU run that leaves the sentinel prints a NOTE.
#[test]
fn atomic_compare_exchange_contention_changes_the_initial_value() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let counter = device.create_typed_buffer(
        "atomics cas",
        &[CAS_SENTINEL; 1],
        GpuBufferAccess::ReadWrite,
    );
    device
        .dispatch_buffers(
            &atomics::run_cas::DESCRIPTOR,
            INVOCATIONS,
            &[BufferBinding::read_write(&counter).independent_length()],
        )
        .unwrap();
    let actual = device.read_typed_buffer(&counter).unwrap();
    let value = actual[0];
    assert!(
        value == CAS_SENTINEL || (1..=INVOCATIONS).contains(&value),
        "atomic_compare_exchange must end at a thread value in [1, {INVOCATIONS}] or, when every weak attempt \
         fails spuriously, at the unchanged sentinel {CAS_SENTINEL}, got {value}"
    );
    if value == CAS_SENTINEL {
        eprintln!("NOTE atomics cas: every attempt failed spuriously; the sentinel is unchanged");
    }

    let cpu = CpuDevice::new();
    let twin = cpu.buffer(&[CAS_SENTINEL; 1]);
    atomics::run_cas::dispatch(&cpu, INVOCATIONS, Independent(&twin)).unwrap();
    assert_eq!(
        twin.read(),
        vec![1u32],
        "twin atomic_compare_exchange: invocation 0 wins in ascending order"
    );
}

#[test]
fn atomic_compare_exchange_has_deterministic_success_and_failure_cases() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let cpu = CpuDevice::new();

    // The success case succeeds on the twin by definition. On the GPU the weak compare-exchange may fail spuriously
    // even without contention (WGSL 17.8.5; docs/SEMANTICS.md, section 6; D47), so the cell ends at 11 (exchanged)
    // or at 7 (unchanged). The twin is pinned at 11 and is not compared with the GPU. No success flag is asserted:
    // the GPU form is statement-only. A GPU compare-exchange that did nothing is not detected here (it leaves 7,
    // which is allowed); the twin pin catches a no-op twin and the failure case below catches an always-swap.
    let success = device.create_typed_buffer(
        "atomics cas success",
        &[7u32; 1],
        GpuBufferAccess::ReadWrite,
    );
    device
        .dispatch_buffers(
            &atomics::run_cas_success::DESCRIPTOR,
            1,
            &[BufferBinding::read_write(&success).independent_length()],
        )
        .unwrap();
    let actual_success = device.read_typed_buffer(&success).unwrap();
    check_cas_success(&actual_success);
    let twin_success = cpu.buffer(&[7u32; 1]);
    atomics::run_cas_success::dispatch(&cpu, 1, Independent(&twin_success)).unwrap();
    assert_eq!(twin_success.read(), vec![11u32], "twin cas success");

    // The failure case fails on every conforming implementation: a spurious failure only adds failures.
    let failure = device.create_typed_buffer(
        "atomics cas failure",
        &[7u32; 1],
        GpuBufferAccess::ReadWrite,
    );
    device
        .dispatch_buffers(
            &atomics::run_cas_failure::DESCRIPTOR,
            1,
            &[BufferBinding::read_write(&failure).independent_length()],
        )
        .unwrap();
    let actual_failure = device.read_typed_buffer(&failure).unwrap();
    assert_eq!(actual_failure, vec![7u32]);
    let twin_failure = cpu.buffer(&[7u32; 1]);
    atomics::run_cas_failure::dispatch(&cpu, 1, Independent(&twin_failure)).unwrap();
    assert_eq!(twin_failure.read(), vec![7u32], "twin cas failure");
    assert_eq!(twin_failure.read(), actual_failure, "twin against the GPU");
}

#[test]
fn signed_atomic_add_exchange_and_compare_exchange_execute() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let counter =
        device.create_typed_buffer("signed atomics", &[-7i32; 1], GpuBufferAccess::ReadWrite);
    let binding = [BufferBinding::read_write(&counter).independent_length()];
    let cpu = CpuDevice::new();
    let twin = cpu.buffer(&[-7i32; 1]);

    device
        .dispatch_buffers(&atomics::run_signed_add::DESCRIPTOR, 1, &binding)
        .unwrap();
    assert_eq!(device.read_typed_buffer(&counter).unwrap(), vec![-4i32]);
    atomics::run_signed_add::dispatch(&cpu, 1, Independent(&twin)).unwrap();
    assert_eq!(twin.read(), vec![-4i32], "twin signed add");

    device
        .dispatch_buffers(&atomics::run_signed_exchange::DESCRIPTOR, 1, &binding)
        .unwrap();
    assert_eq!(device.read_typed_buffer(&counter).unwrap(), vec![-5i32]);
    atomics::run_signed_exchange::dispatch(&cpu, 1, Independent(&twin)).unwrap();
    assert_eq!(twin.read(), vec![-5i32], "twin signed exchange");

    device
        .dispatch_buffers(&atomics::run_signed_cas::DESCRIPTOR, 1, &binding)
        .unwrap();
    // The weak compare-exchange of -5 to 11 may fail spuriously (WGSL 17.8.5; D47): the cell ends at 11 or stays -5.
    // The twin is pinned at 11 and is not compared with the GPU.
    let signed_cas = device.read_typed_buffer(&counter).unwrap();
    check_signed_cas(&signed_cas);
    atomics::run_signed_cas::dispatch(&cpu, 1, Independent(&twin)).unwrap();
    assert_eq!(twin.read(), vec![11i32], "twin signed compare-exchange");
}

/// Whether `check` panics on `actual`.
fn rejects<T: std::panic::RefUnwindSafe>(check: fn(&[T]), actual: &[T]) -> bool {
    std::panic::catch_unwind(|| check(actual)).is_err()
}

/// The GPU assertions of the two success cases, without a GPU. Each check accepts exactly the exchanged value and
/// the unchanged one (the spurious branch: `7` unsigned, `-5` signed, which the RTX 5090 never takes) and rejects
/// every other value, an empty readback and a two-element readback. The GPU tests call these same functions, so
/// tightening a check back to the exchanged value, or changing one of its constants, fails here and not only on an
/// adapter that fails spuriously. Bypassing a check at its call site is not caught by this test.
#[test]
fn the_weak_compare_exchange_checks_accept_exactly_the_allowed_values() {
    check_cas_success(&[11u32]);
    check_cas_success(&[7u32]);
    for bad in [0u32, 1, 6, 8, 10, 12, 999, u32::MAX] {
        assert!(rejects(check_cas_success, &[bad]), "unsigned {bad}");
    }
    assert!(rejects(check_cas_success, &[]), "unsigned empty readback");
    assert!(
        rejects(check_cas_success, &[11, 11]),
        "unsigned two elements"
    );
    assert!(
        rejects(check_cas_success, &[7, 11]),
        "unsigned two elements"
    );

    check_signed_cas(&[11i32]);
    check_signed_cas(&[-5i32]);
    for bad in [-4i32, -6, 0, 5, 7, 10, 12, i32::MIN, i32::MAX] {
        assert!(rejects(check_signed_cas, &[bad]), "signed {bad}");
    }
    assert!(rejects(check_signed_cas, &[]), "signed empty readback");
    assert!(rejects(check_signed_cas, &[11, 11]), "signed two elements");
    assert!(rejects(check_signed_cas, &[-5, 11]), "signed two elements");
}

#[test]
fn atomics_compile_to_wgsl() {
    let descriptor = &atomics::run::DESCRIPTOR;
    assert!(
        gust::slang::compile_wgsl(descriptor)
            .unwrap()
            .contains("@compute")
    );
}
