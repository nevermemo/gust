include!("../../../tests/fixtures/numeric.rs");

use gust::cpu::CpuDevice;
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

/// Independent host reference. This is ordinary Rust, not execution of a shader
/// shadow body: it states the portable numeric semantics the GPU must match.
fn reference(a: i32, b: u32, f: f32) -> (i32, u32, i32, u32, u32) {
    let signed = a
        .wrapping_div(2)
        .wrapping_mul(100)
        .wrapping_add(a.wrapping_rem(3));
    let unsigned = b / 2 * 100 + b % 3;
    let cast = f as i32;
    // pair_out[i].first = (b + 2) + 1; pair_out[i].second = (b + 1) + 1
    let pair_first = b.wrapping_add(3);
    let pair_second = b.wrapping_add(2);
    (signed, unsigned, cast, pair_first, pair_second)
}

#[test]
fn numeric_semantics_and_struct_construction_execute_on_gpu() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    for count in [1, 63, 64, 65, 257] {
        let a: Vec<i32> = (0..count)
            .map(|i| {
                let n = (i % 13) as i32 - 6;
                // Include negative values to exercise truncation-toward-zero
                // division and dividend-sign modulo.
                if n % 2 == 0 { n } else { -n }
            })
            .collect();
        let b: Vec<u32> = (0..count).map(|i| (i % 17) as u32).collect();
        let f: Vec<f32> = (0..count)
            .map(|i| {
                let n = (i % 11) as f32 - 5.0;
                if (i % 2) == 0 { n + 0.7 } else { n - 0.3 }
            })
            .collect();

        let a_buf = device.create_typed_buffer("numeric a", &a, GpuBufferAccess::ReadOnly);
        let b_buf = device.create_typed_buffer("numeric b", &b, GpuBufferAccess::ReadOnly);
        let f_buf = device.create_typed_buffer("numeric f", &f, GpuBufferAccess::ReadOnly);
        let signed_out = device.create_typed_buffer(
            "numeric signed",
            &vec![0i32; count],
            GpuBufferAccess::ReadWrite,
        );
        let unsigned_out = device.create_typed_buffer(
            "numeric unsigned",
            &vec![0u32; count],
            GpuBufferAccess::ReadWrite,
        );
        let cast_out = device.create_typed_buffer(
            "numeric cast",
            &vec![0i32; count],
            GpuBufferAccess::ReadWrite,
        );
        let pair_out = device.create_typed_buffer(
            "numeric pair",
            &vec![
                numeric::Pair {
                    first: 0,
                    second: 0
                };
                count
            ],
            GpuBufferAccess::ReadWrite,
        );

        device
            .dispatch_buffers(
                &numeric::run::DESCRIPTOR,
                count as u32,
                &[
                    BufferBinding::read_only(&a_buf),
                    BufferBinding::read_only(&b_buf),
                    BufferBinding::read_only(&f_buf),
                    BufferBinding::read_write(&signed_out),
                    BufferBinding::read_write(&unsigned_out),
                    BufferBinding::read_write(&cast_out),
                    BufferBinding::read_write(&pair_out),
                ],
            )
            .unwrap();

        let actual_signed = device.read_typed_buffer(&signed_out).unwrap();
        let actual_unsigned = device.read_typed_buffer(&unsigned_out).unwrap();
        let actual_cast = device.read_typed_buffer(&cast_out).unwrap();
        let actual_pair = device.read_typed_buffer(&pair_out).unwrap();

        // The CPU twin: the same typed call with the device swapped (macro-emitted under `cfg(test)`).
        let cpu = CpuDevice::new();
        let twin_a = cpu.buffer(&a);
        let twin_b = cpu.buffer(&b);
        let twin_f = cpu.buffer(&f);
        let twin_signed = cpu.buffer(&vec![0i32; count]);
        let twin_unsigned = cpu.buffer(&vec![0u32; count]);
        let twin_cast = cpu.buffer(&vec![0i32; count]);
        let twin_pair = cpu.buffer(&vec![
            numeric::Pair {
                first: 0,
                second: 0
            };
            count
        ]);
        numeric::run::dispatch(
            &cpu,
            count as u32,
            &twin_a,
            &twin_b,
            &twin_f,
            &twin_signed,
            &twin_unsigned,
            &twin_cast,
            &twin_pair,
        )
        .unwrap();
        let twin_signed = twin_signed.read();
        let twin_unsigned = twin_unsigned.read();
        let twin_cast = twin_cast.read();
        let twin_pair = twin_pair.read();
        assert_eq!(twin_signed, actual_signed, "twin signed, count {count}");
        assert_eq!(
            twin_unsigned, actual_unsigned,
            "twin unsigned, count {count}"
        );
        assert_eq!(twin_cast, actual_cast, "twin cast, count {count}");
        for i in 0..count {
            assert_eq!(
                (twin_pair[i].first, twin_pair[i].second),
                (actual_pair[i].first, actual_pair[i].second),
                "twin pair[{i}] against the GPU"
            );
        }

        for i in 0..count {
            let (exp_signed, exp_unsigned, exp_cast, exp_first, exp_second) =
                reference(a[i], b[i], f[i]);
            assert_eq!(actual_signed[i], exp_signed, "signed[{i}]");
            assert_eq!(actual_unsigned[i], exp_unsigned, "unsigned[{i}]");
            assert_eq!(actual_cast[i], exp_cast, "cast[{i}]");
            // Field identity: `first` and `second` are assigned by name even
            // though the literal wrote `second` first.
            assert_eq!(actual_pair[i].first, exp_first, "pair.first[{i}]");
            assert_eq!(actual_pair[i].second, exp_second, "pair.second[{i}]");
            // The twin against the independent reference, not only against the GPU.
            assert_eq!(twin_signed[i], exp_signed, "twin signed[{i}]");
            assert_eq!(twin_unsigned[i], exp_unsigned, "twin unsigned[{i}]");
            assert_eq!(twin_cast[i], exp_cast, "twin cast[{i}]");
            assert_eq!(twin_pair[i].first, exp_first, "twin pair.first[{i}]");
            assert_eq!(twin_pair[i].second, exp_second, "twin pair.second[{i}]");
        }
    }
}

#[test]
fn numeric_compiles_to_wgsl() {
    let descriptor = &numeric::run::DESCRIPTOR;
    assert!(
        gust::slang::compile_wgsl(descriptor)
            .unwrap()
            .contains("@compute")
    );
}
