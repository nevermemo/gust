include!("../../../tests/fixtures/semantics.rs");

use gust::cpu::CpuDevice;
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

#[test]
fn translated_semantics_execute_on_gpu() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    for count in [1, 63, 64, 65, 257] {
        let values: Vec<f32> = (0..count).map(|i| (i % 7) as f32 - 3.0).collect();
        let input =
            device.create_typed_buffer("semantics input", &values, GpuBufferAccess::ReadOnly);
        let floats = device.create_typed_buffer(
            "semantics floats",
            &vec![0.0f32; count],
            GpuBufferAccess::ReadWrite,
        );
        let integers = device.create_typed_buffer(
            "semantics integers",
            &vec![0u32; count],
            GpuBufferAccess::ReadWrite,
        );
        device
            .dispatch_buffers(
                &semantics::run::DESCRIPTOR,
                count as u32,
                &[
                    BufferBinding::read_only(&input),
                    BufferBinding::read_write(&floats),
                    BufferBinding::read_write(&integers),
                ],
            )
            .unwrap();
        let actual_floats = device.read_typed_buffer(&floats).unwrap();
        let actual_integers = device.read_typed_buffer(&integers).unwrap();

        // The CPU twin: the same typed call with the device swapped (macro-emitted under `cfg(test)`).
        let cpu = CpuDevice::new();
        let twin_input = cpu.buffer(&values);
        let twin_floats = cpu.buffer(&vec![0.0f32; count]);
        let twin_integers = cpu.buffer(&vec![0u32; count]);
        semantics::run::dispatch(
            &cpu,
            count as u32,
            &twin_input,
            &twin_floats,
            &twin_integers,
        )
        .unwrap();
        let twin_floats = twin_floats.read();
        let twin_integers = twin_integers.read();
        // Exact on these inputs (small values, exactly representable results), so twin == GPU holds bit for bit.
        assert_eq!(twin_floats, actual_floats, "twin floats, count {count}");
        assert_eq!(
            twin_integers, actual_integers,
            "twin integers, count {count}"
        );

        for (i, &x) in values.iter().enumerate() {
            // Independent host expectations, not execution of a shader shadow body.
            let expected = if x > 0.0 {
                x * 2.5 + 1.0
            } else if x < 0.0 {
                -x
            } else {
                7.0
            };
            let expected_float = expected + if i >= 2 { 1.0 } else { 0.0 };
            assert_eq!(actual_floats[i], expected_float);
            assert_eq!(twin_floats[i], expected_float, "twin float[{i}]");
            let i = i as u32;
            let expected = if i.is_multiple_of(2) {
                255 - (i % 256) + 2 + (i / 2) + 1
            } else {
                i + 1024
            };
            assert_eq!(actual_integers[i as usize], expected);
            assert_eq!(twin_integers[i as usize], expected, "twin integer[{i}]");
        }
    }
}

#[test]
fn semantics_compile_to_wgsl() {
    let descriptor = &semantics::run::DESCRIPTOR;
    assert!(
        gust::slang::compile_wgsl(descriptor)
            .unwrap()
            .contains("@compute")
    );
}
