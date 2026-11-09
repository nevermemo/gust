include!("../../../tests/fixtures/methods.rs");

use gust::Run;
use gust::cpu::CpuDevice;
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

// Expected values, written by hand from the fixture's rules (not computed by a function that mirrors the kernel).
//
// Circle i has radius (i % 7) + 1 and scale 0.5.
//   area  = 3 * r * r * 0.5   : r = 1..7 gives 1.5, 6, 13.5, 24, 37.5, 54, 73.5 (every product is exact in f32)
//   grown = (r + 2) * 0.5     : r = 1..7 gives 1.5, 2, 2.5, 3, 3.5, 4, 4.5
//   gap   = r - 0.25          : r = 1..7 gives 0.75, 1.75, .. 6.75. The method's parameter is named `this`; it must not
//                               take the place of the receiver (that would give 0.0 for every circle).
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
//
// The read-write buffer `ring` holds circles of radius (i % 5) + 1 and scale 1, so the `&self` method on one of its
// elements gives 3 * r * r = 3, 12, 27, 48, 75. The uniform holds a disc of radius 2.5 and scale 0.5, so
// weight(4) = 2.5 * 4 + 0.5 = 10.5 for every invocation.
const AREAS: [f32; 7] = [1.5, 6.0, 13.5, 24.0, 37.5, 54.0, 73.5];
const GROWN: [f32; 7] = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
const GAPS: [f32; 7] = [0.75, 1.75, 2.75, 3.75, 4.75, 5.75, 6.75];
const LO: [u32; 8] = [0, 1, 2, 3, 4, 0, 1, 2];
const HI: [u32; 8] = [3, 5, 4, 9, 4, 10, 2, 8];
const INPUT: [u32; 8] = [5, 0, 2, 20, 4, 7, 1, 2];
const CLAMPED: [u32; 8] = [3, 1, 2, 9, 4, 7, 1, 2];
const EXTENTS: [u32; 8] = [3, 4, 2, 6, 0, 10, 1, 6];
const RING_AREAS: [f32; 5] = [3.0, 12.0, 27.0, 48.0, 75.0];
const WEIGHT: f32 = 10.5;

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
        let ring: Vec<methods::Circle> = (0..count)
            .map(|i| methods::Circle {
                radius: (i % 5 + 1) as f32,
                scale: 1.0,
            })
            .collect();
        let frame = methods::Frame {
            disc: methods::Disc {
                radius: 2.5,
                scale: 0.5,
                pad0: 0.0,
                pad1: 0.0,
            },
        };

        let circles_buf =
            device.create_typed_buffer("methods circles", &circles, GpuBufferAccess::ReadOnly);
        let spans_buf =
            device.create_typed_buffer("methods spans", &spans, GpuBufferAccess::ReadOnly);
        let input_buf =
            device.create_typed_buffer("methods input", &input, GpuBufferAccess::ReadOnly);
        let ring_buf =
            device.create_typed_buffer("methods ring", &ring, GpuBufferAccess::ReadWrite);
        let uniform = device.create_uniform("methods frame", &frame).unwrap();
        let blank = vec![
            methods::Results {
                area: -1.0,
                grown: -1.0,
                gap: -1.0,
                ring_area: -1.0,
                weight: -1.0,
                clamped: u32::MAX,
                extent: u32::MAX,
            };
            count
        ];
        let results =
            device.create_typed_buffer("methods results", &blank, GpuBufferAccess::ReadWrite);
        let call = methods::run::call(
            count as u32,
            &circles_buf,
            &spans_buf,
            &input_buf,
            &uniform,
            &ring_buf,
            &results,
        );
        device.dispatch_call(&call).unwrap();
        let actual = device.read_typed_buffer(&results).unwrap();

        // The CPU twin: the same typed call with the device swapped (macro-emitted under `cfg(test)`).
        let cpu = CpuDevice::new();
        let twin_circles = cpu.buffer(&circles);
        let twin_spans = cpu.buffer(&spans);
        let twin_input = cpu.buffer(&input);
        let twin_frame = cpu.uniform(frame).unwrap();
        let twin_ring = cpu.buffer(&ring);
        let twin_results = cpu.buffer(&blank);
        methods::run::dispatch(
            &cpu,
            count as u32,
            &twin_circles,
            &twin_spans,
            &twin_input,
            &twin_frame,
            &twin_ring,
            &twin_results,
        )
        .unwrap();
        let twin = twin_results.read();

        assert_eq!(actual.len(), count);
        for (name, got) in [("gpu", &actual), ("twin", &twin)] {
            for i in 0..count {
                let note = format!("{name} [{i}], count {count}");
                assert_eq!(got[i].area, AREAS[i % 7], "area {note}");
                assert_eq!(got[i].grown, GROWN[i % 7], "grown {note}");
                assert_eq!(got[i].clamped, CLAMPED[i % 8], "clamped {note}");
                assert_eq!(got[i].extent, EXTENTS[i % 8], "extent {note}");
                assert_eq!(got[i].gap, GAPS[i % 7], "gap {note}");
                assert_eq!(got[i].ring_area, RING_AREAS[i % 5], "ring area {note}");
                assert_eq!(got[i].weight, WEIGHT, "weight {note}");
            }
        }
    }
}

#[test]
fn the_method_fixture_compiles_to_wgsl() {
    let descriptor = &methods::run::DESCRIPTOR;
    let wgsl = gust::slang::compile_wgsl(descriptor).unwrap();
    // The kernel is a compute entry, and the methods are functions of the output that it calls under their own names.
    assert!(wgsl.contains("compute"), "{wgsl}");
    // Slang writes a member `m` of the struct `S` as the function `S_m_<n>` (read from the real WGSL). Four of these
    // six names are also field names of `Results`, so the bare name proves nothing: the owner prefix does.
    for name in [
        "Circle_area",
        "Circle_grown",
        "Circle_gap",
        "Disc_weight",
        "Span_extent",
        "Span_clamp_into",
    ] {
        assert!(
            wgsl.contains(&format!("fn {name}_")),
            "no function `{name}` in the WGSL\n{wgsl}"
        );
    }
}
