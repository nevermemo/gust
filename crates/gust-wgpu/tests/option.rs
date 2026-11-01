include!("../../../tests/fixtures/option.rs");

use gust::cpu::CpuDevice;
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

/// Independent host reference. Ordinary Rust `Option` semantics the GPU lowering
/// (Slang `Optional<T>`) must reproduce: presence tests, `unwrap_or`, `if let`, and
/// helpers that return or accept an `Option`.
fn reference(x: u32) -> (u32, u32, (u32, u32)) {
    let half = if x.is_multiple_of(2) {
        Some(x / 2)
    } else {
        None
    };
    let flag = if half.is_some() { 1 } else { 2 };
    let missing = if x > 1000 { Some(x) } else { None };
    let flags = flag + missing.unwrap_or(3) * 10;
    let mut total = half.unwrap_or(7) + (x + 100);
    if let Some(h) = half {
        total += h;
    }
    let pair = if x < 8 { (x, x + 1) } else { (99, 0) };
    (flags, total, pair)
}

#[test]
fn option_lowering_executes_on_gpu() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("option adapter: {:?}", device.adapter_info());
    for count in [1, 63, 64, 65, 257] {
        // Covers both parities, values below 8 (Some pair) and above 1000 (Some missing).
        let input: Vec<u32> = (0..count).map(|i| ((i * 97) % 1150) as u32).collect();
        let input_buf =
            device.create_typed_buffer("option input", &input, GpuBufferAccess::ReadOnly);
        let flags = device.create_typed_buffer(
            "option flags",
            &vec![0u32; count],
            GpuBufferAccess::ReadWrite,
        );
        let values = device.create_typed_buffer(
            "option values",
            &vec![0u32; count],
            GpuBufferAccess::ReadWrite,
        );
        let pairs = device.create_typed_buffer(
            "option pairs",
            &vec![
                optional::Pair {
                    first: 0,
                    second: 0
                };
                count
            ],
            GpuBufferAccess::ReadWrite,
        );

        device
            .dispatch_buffers(
                &optional::run::DESCRIPTOR,
                count as u32,
                &[
                    BufferBinding::read_only(&input_buf),
                    BufferBinding::read_write(&flags),
                    BufferBinding::read_write(&values),
                    BufferBinding::read_write(&pairs),
                ],
            )
            .unwrap();

        let actual_flags = device.read_typed_buffer(&flags).unwrap();
        let actual_values = device.read_typed_buffer(&values).unwrap();
        let actual_pairs = device.read_typed_buffer(&pairs).unwrap();

        // The CPU twin: the same typed call with the device swapped (macro-emitted under `cfg(test)`).
        let cpu = CpuDevice::new();
        let twin_input = cpu.buffer(&input);
        let twin_flags = cpu.buffer(&vec![0u32; count]);
        let twin_values = cpu.buffer(&vec![0u32; count]);
        let twin_pairs = cpu.buffer(&vec![
            optional::Pair {
                first: 0,
                second: 0
            };
            count
        ]);
        optional::run::dispatch(
            &cpu,
            count as u32,
            &twin_input,
            &twin_flags,
            &twin_values,
            &twin_pairs,
        )
        .unwrap();
        let twin_flags = twin_flags.read();
        let twin_values = twin_values.read();
        let twin_pairs = twin_pairs.read();
        assert_eq!(twin_flags, actual_flags, "twin flags, count {count}");
        assert_eq!(twin_values, actual_values, "twin values, count {count}");
        for i in 0..count {
            assert_eq!(
                (twin_pairs[i].first, twin_pairs[i].second),
                (actual_pairs[i].first, actual_pairs[i].second),
                "twin pairs[{i}] against the GPU for input {}",
                input[i]
            );
        }

        for i in 0..count {
            let (flags, total, (first, second)) = reference(input[i]);
            assert_eq!(
                twin_flags[i], flags,
                "twin flags[{i}] for input {}",
                input[i]
            );
            assert_eq!(
                twin_values[i], total,
                "twin values[{i}] for input {}",
                input[i]
            );
            assert_eq!(
                (twin_pairs[i].first, twin_pairs[i].second),
                (first, second),
                "twin pairs[{i}] for input {}",
                input[i]
            );
            assert_eq!(actual_flags[i], flags, "flags[{i}] for input {}", input[i]);
            assert_eq!(
                actual_values[i], total,
                "values[{i}] for input {}",
                input[i]
            );
            assert_eq!(
                (actual_pairs[i].first, actual_pairs[i].second),
                (first, second),
                "pairs[{i}] for input {}",
                input[i]
            );
        }
    }
}

#[test]
fn option_compiles_to_wgsl() {
    let descriptor = &optional::run::DESCRIPTOR;
    assert!(
        gust::slang::compile_wgsl(descriptor)
            .unwrap()
            .contains("@compute")
    );
}
