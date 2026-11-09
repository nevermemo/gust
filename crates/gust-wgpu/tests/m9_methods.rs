include!("../../../tests/fixtures/methods.rs");

use gust::cpu::CpuDevice;
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

// Expected values, written by hand from the fixture's rules (not computed by a function that mirrors the kernel).
//
// Circle i has radius (i % 7) + 1 and scale 0.5.
//   area  = 3 * r * r * 0.5   : r = 1..7 gives 1.5, 6, 13.5, 24, 37.5, 54, 73.5 (every product is exact in f32)
//   grown = (r + 2) * 0.5     : r = 1..7 gives 1.5, 2, 2.5, 3, 3.5, 4, 4.5
//
// Span i uses row (i % 8) of this table, and the input value of the same row:
//   row   lo  hi  input  clamp_into(input)  extent
//    0     0   3    5     3 (above hi)        3
//    1     1   5    0     1 (below lo)        4
//    2     2   4    2     2 (on lo)           2
//    3     3   9   20     9 (above hi)        6
//    4     4   4    4     4 (lo == hi)        0
//    5     0  10    7     7 (inside)         10
//    6     1   2    1     1 (on lo)           1
//    7     2   8    2     2 (on lo)           6
const AREAS: [f32; 7] = [1.5, 6.0, 13.5, 24.0, 37.5, 54.0, 73.5];
const GROWN: [f32; 7] = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
const LO: [u32; 8] = [0, 1, 2, 3, 4, 0, 1, 2];
const HI: [u32; 8] = [3, 5, 4, 9, 4, 10, 2, 8];
const INPUT: [u32; 8] = [5, 0, 2, 20, 4, 7, 1, 2];
const CLAMPED: [u32; 8] = [3, 1, 2, 9, 4, 7, 1, 2];
const EXTENTS: [u32; 8] = [3, 4, 2, 6, 0, 10, 1, 6];

#[test]
fn methods_execute_on_gpu_and_on_the_cpu_twin() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("methods adapter: {:?}", device.adapter_info());
    for count in [1, 63, 64, 65, 257] {
        let circles: Vec<methods::Circle> = (0..count)
            .map(|i| methods::Circle {
                radius: (i % 7 + 1) as f32,
                scale: 0.5,
            })
            .collect();
        let spans: Vec<methods::Span> = (0..count)
            .map(|i| methods::Span {
                lo: LO[i % 8],
                hi: HI[i % 8],
            })
            .collect();
        let input: Vec<u32> = (0..count).map(|i| INPUT[i % 8]).collect();

        let circles_buf =
            device.create_typed_buffer("methods circles", &circles, GpuBufferAccess::ReadOnly);
        let spans_buf =
            device.create_typed_buffer("methods spans", &spans, GpuBufferAccess::ReadOnly);
        let input_buf =
            device.create_typed_buffer("methods input", &input, GpuBufferAccess::ReadOnly);
        let areas = device.create_typed_buffer(
            "methods areas",
            &vec![0.0f32; count],
            GpuBufferAccess::ReadWrite,
        );
        let grown = device.create_typed_buffer(
            "methods grown",
            &vec![0.0f32; count],
            GpuBufferAccess::ReadWrite,
        );
        let clamped = device.create_typed_buffer(
            "methods clamped",
            &vec![0u32; count],
            GpuBufferAccess::ReadWrite,
        );
        let extents = device.create_typed_buffer(
            "methods extents",
            &vec![0u32; count],
            GpuBufferAccess::ReadWrite,
        );
        device
            .dispatch_buffers(
                &methods::run::DESCRIPTOR,
                count as u32,
                &[
                    BufferBinding::read_only(&circles_buf),
                    BufferBinding::read_only(&spans_buf),
                    BufferBinding::read_only(&input_buf),
                    BufferBinding::read_write(&areas),
                    BufferBinding::read_write(&grown),
                    BufferBinding::read_write(&clamped),
                    BufferBinding::read_write(&extents),
                ],
            )
            .unwrap();
        let actual_areas = device.read_typed_buffer(&areas).unwrap();
        let actual_grown = device.read_typed_buffer(&grown).unwrap();
        let actual_clamped = device.read_typed_buffer(&clamped).unwrap();
        let actual_extents = device.read_typed_buffer(&extents).unwrap();

        // The CPU twin: the same typed call with the device swapped (macro-emitted under `cfg(test)`).
        let cpu = CpuDevice::new();
        let twin_circles = cpu.buffer(&circles);
        let twin_spans = cpu.buffer(&spans);
        let twin_input = cpu.buffer(&input);
        let twin_areas = cpu.buffer(&vec![0.0f32; count]);
        let twin_grown = cpu.buffer(&vec![0.0f32; count]);
        let twin_clamped = cpu.buffer(&vec![0u32; count]);
        let twin_extents = cpu.buffer(&vec![0u32; count]);
        methods::run::dispatch(
            &cpu,
            count as u32,
            &twin_circles,
            &twin_spans,
            &twin_input,
            &twin_areas,
            &twin_grown,
            &twin_clamped,
            &twin_extents,
        )
        .unwrap();
        let twin_areas = twin_areas.read();
        let twin_grown = twin_grown.read();
        let twin_clamped = twin_clamped.read();
        let twin_extents = twin_extents.read();

        assert_eq!(actual_areas.len(), count);
        for i in 0..count {
            assert_eq!(
                actual_areas[i],
                AREAS[i % 7],
                "gpu areas[{i}], count {count}"
            );
            assert_eq!(
                actual_grown[i],
                GROWN[i % 7],
                "gpu grown[{i}], count {count}"
            );
            assert_eq!(
                actual_clamped[i],
                CLAMPED[i % 8],
                "gpu clamped[{i}], count {count}"
            );
            assert_eq!(
                actual_extents[i],
                EXTENTS[i % 8],
                "gpu extents[{i}], count {count}"
            );
            assert_eq!(
                twin_areas[i],
                AREAS[i % 7],
                "twin areas[{i}], count {count}"
            );
            assert_eq!(
                twin_grown[i],
                GROWN[i % 7],
                "twin grown[{i}], count {count}"
            );
            assert_eq!(
                twin_clamped[i],
                CLAMPED[i % 8],
                "twin clamped[{i}], count {count}"
            );
            assert_eq!(
                twin_extents[i],
                EXTENTS[i % 8],
                "twin extents[{i}], count {count}"
            );
        }
    }
}

#[test]
fn the_method_fixture_compiles_to_wgsl() {
    let descriptor = &methods::run::DESCRIPTOR;
    assert!(
        gust::slang::compile_wgsl(descriptor)
            .unwrap()
            .contains("fn ")
    );
}
