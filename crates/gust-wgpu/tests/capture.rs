//! Capture and replay (g2.cap): `capture` copies the inputs of one GPU call before the dispatch, and `run_element` and
//! `run_until` replay chosen invocations on the CPU twin over those copies. The GPU results come from a real dispatch;
//! every expected value is computed in this file from the inputs, not by the kernel or the twin.
//!
//! Float values are small multiples of powers of two and every operation is a product or a sum of such values, so
//! the GPU and the twin agree to the bit.

include!("../../../tests/fixtures/grid.rs");

mod common;

use gust::capture::{Error as CaptureError, ReplayError};
use gust::{Extent, Independent};
use gust_wgpu::{DeviceOptions, Error, GpuBufferAccess, HeadlessDevice};

#[gust::gpu_module]
mod sim {
    pub struct Particle {
        pub pos: float,
        pub vel: float,
        pub tag: uint,
    }

    pub struct Params {
        pub dt: float,
        pub gain: float,
        pub pad0: float,
        pub pad1: float,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn step(
        id: SV_DispatchThreadID,
        params: Uniform<Params>,
        input: StructuredBuffer<Particle>,
        mut out: RWStructuredBuffer<Particle>,
    ) {
        let i = id.x;
        if i < out.len() && i < input.len() {
            let p = input[i];
            out[i] = Particle {
                pos: p.pos + p.vel * params.dt,
                vel: p.vel * params.gain,
                tag: p.tag + 1u32,
            };
        }
    }
}

#[gust::gpu_module]
mod nap {
    #[kernel(workgroup_size(64, 1, 1), cpu = "unsupported")]
    pub fn zero(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        let i = id.x;
        if i < out.len() {
            out[i] = 0u32;
        }
    }
}

/// Not a multiple of the workgroup width (64), so the guard in the kernel matters.
const N: usize = 150;
const DT: f32 = 0.5;
const GAIN: f32 = 2.0;
const SENTINEL: (u32, u32, u32) = (0xBF80_0000, 0xBF80_0000, 999);

fn capture_device(tag: &str) -> HeadlessDevice {
    let device = HeadlessDevice::with_options(DeviceOptions::default().with_capture(true))
        .expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("{tag} adapter: {}", device.adapter_info().name);
    assert!(device.capture_enabled());
    device
}

fn input_values() -> Vec<sim::Particle> {
    (0..N)
        .map(|i| sim::Particle {
            pos: i as f32 * 0.5,
            vel: (i % 8) as f32 * 0.25,
            tag: i as u32,
        })
        .collect()
}

fn sentinel_values() -> Vec<sim::Particle> {
    (0..N)
        .map(|_| sim::Particle {
            pos: f32::from_bits(SENTINEL.0),
            vel: f32::from_bits(SENTINEL.1),
            tag: SENTINEL.2,
        })
        .collect()
}

/// The element a particle becomes, from its index alone.
fn expected(i: usize) -> (u32, u32, u32) {
    let input = &input_values()[i];
    (
        (input.pos + input.vel * DT).to_bits(),
        (input.vel * GAIN).to_bits(),
        input.tag + 1,
    )
}

fn bits(values: &[sim::Particle]) -> Vec<(u32, u32, u32)> {
    values
        .iter()
        .map(|p| (p.pos.to_bits(), p.vel.to_bits(), p.tag))
        .collect()
}

#[test]
fn replaying_the_whole_extent_gives_the_outputs_the_gpu_wrote() {
    let device = capture_device("capture whole");
    let params = device
        .create_uniform(
            "params",
            &sim::Params {
                dt: DT,
                gain: GAIN,
                pad0: 0.0,
                pad1: 0.0,
            },
        )
        .unwrap();
    let input = device.create_typed_buffer("input", &input_values(), GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("out", &sentinel_values(), GpuBufferAccess::ReadWrite);

    let call = sim::step::call(N as u32, &params, &input, &out);
    let captured = sim::step::capture(&device, &call).unwrap();
    // The dispatch happens after the capture and changes the GPU's buffer, not the copies.
    sim::step::dispatch(&device, N as u32, &params, &input, &out).unwrap();
    let gpu = bits(&device.read_typed_buffer(&out).unwrap());
    let want: Vec<_> = (0..N).map(expected).collect();
    assert_eq!(gpu, want, "the GPU result is what the inputs say");
    assert_eq!(
        bits(&captured.buffer::<sim::Particle>("out").unwrap().read()),
        bits(&sentinel_values()),
        "the copy of `out` holds the values from before the dispatch"
    );

    captured.run_until([N as u32 - 1, 0, 0]).unwrap();
    let twin = bits(&captured.buffer::<sim::Particle>("out").unwrap().read());
    assert_eq!(twin, gpu, "the twin's outputs are the GPU's outputs");

    // The inputs and the uniform were copied as they were.
    assert_eq!(
        bits(&captured.buffer::<sim::Particle>("input").unwrap().read()),
        bits(&input_values())
    );
    let copied = captured.uniform::<sim::Params>("params").unwrap().read();
    assert_eq!((copied.dt, copied.gain), (DT, GAIN));
    assert_eq!(captured.extent(), Extent::linear(N as u32));
    // A buffer of another element type, or a name that is no parameter, is not found.
    assert!(captured.buffer::<u32>("out").is_none());
    assert!(captured.buffer::<sim::Particle>("params").is_none());
    assert!(captured.buffer::<sim::Particle>("nothing").is_none());
}

#[test]
fn run_element_changes_only_that_elements_outputs() {
    let device = capture_device("capture element");
    let params = device
        .create_uniform(
            "params",
            &sim::Params {
                dt: DT,
                gain: GAIN,
                pad0: 0.0,
                pad1: 0.0,
            },
        )
        .unwrap();
    let input = device.create_typed_buffer("input", &input_values(), GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("out", &sentinel_values(), GpuBufferAccess::ReadWrite);
    let captured =
        sim::step::capture(&device, &sim::step::call(N as u32, &params, &input, &out)).unwrap();
    let read = || bits(&captured.buffer::<sim::Particle>("out").unwrap().read());
    let untouched = bits(&sentinel_values());

    captured.run_element([67, 0, 0]).unwrap();
    let after = read();
    for (i, element) in after.iter().enumerate() {
        if i == 67 {
            assert_eq!(*element, expected(67));
        } else {
            assert_eq!(
                *element, untouched[i],
                "element {i} keeps its captured value"
            );
        }
    }

    // Every run starts from the captured values: another element, not both.
    captured.run_element([3, 0, 0]).unwrap();
    let after = read();
    assert_eq!(after[3], expected(3));
    assert_eq!(after[67], untouched[67], "the earlier run does not linger");

    // `run_until` runs 0..=id and no more.
    captured.run_until([9, 0, 0]).unwrap();
    let after = read();
    for (i, element) in after.iter().enumerate() {
        if i <= 9 {
            assert_eq!(*element, expected(i), "element {i} is replayed");
        } else {
            assert_eq!(*element, untouched[i], "element {i} is after the id");
        }
    }
}

#[test]
fn ids_outside_the_extent_are_errors_and_run_nothing() {
    let device = capture_device("capture extent");
    let params = device
        .create_uniform(
            "params",
            &sim::Params {
                dt: DT,
                gain: GAIN,
                pad0: 0.0,
                pad1: 0.0,
            },
        )
        .unwrap();
    let input = device.create_typed_buffer("input", &input_values(), GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("out", &sentinel_values(), GpuBufferAccess::ReadWrite);
    let captured =
        sim::step::capture(&device, &sim::step::call(N as u32, &params, &input, &out)).unwrap();
    let extent = Extent::linear(N as u32);
    // The first id past the extent, and the ids of the padding of the last workgroup (150 rounds up to 192), and an id
    // on an axis the extent does not have.
    for id in [[N as u32, 0, 0], [191, 0, 0], [0, 1, 0], [0, 0, 1]] {
        assert_eq!(
            captured.run_element(id),
            Err(ReplayError::OutOfExtent { id, extent })
        );
        assert_eq!(
            captured.run_until(id),
            Err(ReplayError::OutOfExtent { id, extent })
        );
    }
    let read = || bits(&captured.buffer::<sim::Particle>("out").unwrap().read());
    assert_eq!(read(), bits(&sentinel_values()), "nothing ran");
    // The last element of the extent is inside it.
    captured.run_element([N as u32 - 1, 0, 0]).unwrap();
    assert_eq!(read()[N - 1], expected(N - 1));
}

#[test]
fn a_read_only_input_needs_the_capture_option() {
    let device = common::device_logged("capture hint");
    assert!(!device.capture_enabled());
    let params = device
        .create_uniform(
            "params",
            &sim::Params {
                dt: DT,
                gain: GAIN,
                pad0: 0.0,
                pad1: 0.0,
            },
        )
        .unwrap();
    let input = device.create_typed_buffer("input", &input_values(), GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("out", &sentinel_values(), GpuBufferAccess::ReadWrite);
    let call = sim::step::call(N as u32, &params, &input, &out);
    let error = sim::step::capture(&device, &call).unwrap_err();
    match &error {
        CaptureError::Read {
            parameter,
            error: Error::PersistentBufferNotReadable,
            hint: Some(hint),
        } => {
            assert_eq!(*parameter, "input");
            assert!(hint.contains("with_capture(true)"), "{hint}");
        }
        other => panic!("expected the hinted read error, got {other:?}"),
    }
    let text = error.to_string();
    assert!(text.contains("`input`"), "{text}");
    assert!(text.contains("with_capture(true)"), "{text}");

    // The same buffers on a device opened with the option capture.
    let device = capture_device("capture option");
    let params = device
        .create_uniform(
            "params",
            &sim::Params {
                dt: DT,
                gain: GAIN,
                pad0: 0.0,
                pad1: 0.0,
            },
        )
        .unwrap();
    let input = device.create_typed_buffer("input", &input_values(), GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("out", &sentinel_values(), GpuBufferAccess::ReadWrite);
    sim::step::capture(&device, &sim::step::call(N as u32, &params, &input, &out)).unwrap();
    // The fallible and the zeroed allocations carry the option too.
    let tried = device
        .try_create_typed_buffer("tried", &[1u32, 2], GpuBufferAccess::ReadOnly)
        .unwrap();
    let zeroed = device
        .create_zeroed_buffer::<u32>("zeroed", 4, GpuBufferAccess::ReadOnly)
        .unwrap();
    for buffer in [&tried, &zeroed] {
        assert!(
            buffer
                .as_wgpu_buffer()
                .usage()
                .contains(wgpu::BufferUsages::COPY_SRC)
        );
    }
}

#[test]
fn a_kernel_without_a_twin_cannot_be_captured() {
    let device = common::device_logged("capture no twin");
    let out = device.create_typed_buffer("out", &[7u32; 4], GpuBufferAccess::ReadWrite);
    let call = nap::zero::call(4, &out);
    let error = nap::zero::capture(&device, &call).unwrap_err();
    assert!(
        matches!(error, CaptureError::NoTwin { kernel: "zero" }),
        "{error:?}"
    );
    let text = error.to_string();
    assert!(
        text.contains("no twin") && text.contains("`zero`"),
        "{text}"
    );
}

#[test]
fn a_two_dimensional_replay_follows_the_twins_row_order_and_keeps_an_independent_buffer() {
    let device = capture_device("capture grid");
    // 20 x 18 cells: more than one 16 x 16 workgroup along each axis, and not a multiple of it.
    let (width, height) = (20u32, 18u32);
    let count = (width * height) as usize;
    let size = device.create_typed_buffer("size", &[width, height], GpuBufferAccess::ReadOnly);
    let unwritten = vec![u32::MAX; count];
    let output = |label| device.create_typed_buffer(label, &unwritten, GpuBufferAccess::ReadWrite);
    let (coords, locals, groups) = (output("coords"), output("locals"), output("groups"));
    let call = grid::cells::call_extent(
        Extent::new(width, height, 1),
        Independent(&size),
        &coords,
        &locals,
        &groups,
    );
    let captured = grid::cells::capture(&device, &call).unwrap();
    grid::cells::dispatch_extent(
        &device,
        Extent::new(width, height, 1),
        Independent(&size),
        &coords,
        &locals,
        &groups,
    )
    .unwrap();
    let gpu = device.read_typed_buffer(&coords).unwrap();

    // Up to (5, 2): rows 0 and 1 are complete, row 2 has x = 0..=5, and nothing else ran.
    captured.run_until([5, 2, 0]).unwrap();
    let partial = captured.buffer::<u32>("coords").unwrap().read();
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let reached = y < 2 || (y == 2 && x <= 5);
            let want = if reached { y * 65_536 + x } else { u32::MAX };
            assert_eq!(partial[i], want, "cell ({x}, {y})");
        }
    }
    // The last cell of the extent replays the whole launch, and equals the GPU.
    captured.run_until([width - 1, height - 1, 0]).unwrap();
    assert_eq!(captured.buffer::<u32>("coords").unwrap().read(), gpu);
    // One cell, with its local and group ids (16 x 16 workgroups): (17, 16) is local (1, 0) of group (1, 1).
    captured.run_element([17, 16, 0]).unwrap();
    let i = (16 * width + 17) as usize;
    assert_eq!(
        captured.buffer::<u32>("locals").unwrap().read()[i],
        1,
        "local id"
    );
    assert_eq!(
        captured.buffer::<u32>("groups").unwrap().read()[i],
        65_536 + 1,
        "group id"
    );
    assert_eq!(
        captured.buffer::<u32>("coords").unwrap().read()[i],
        16 * 65_536 + 17
    );
    // The independent `size` buffer (2 elements, not 360) was captured with its rule: the twin accepted the call.
    assert_eq!(
        captured.buffer::<u32>("size").unwrap().read(),
        [width, height]
    );
}

#[gust::gpu_module]
mod pad {
    /// Marks every element of an independent table the launch reaches, so the padding ids of the last workgroup show.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn mark(id: SV_DispatchThreadID, mut table: RWStructuredBuffer<uint>) {
        let i = id.x;
        if i < table.len() {
            table[i] = 1u32;
        }
    }
}

#[test]
fn run_all_is_the_whole_launch_padding_included_and_run_until_stops_at_its_id() {
    let device = capture_device("capture padding");
    // A table of 256 and an extent of 100: the launch is 128 invocations, and the guard is the table's length.
    let table = device.create_typed_buffer("table", &[0u32; 256], GpuBufferAccess::ReadWrite);
    let call = pad::mark::call(100, Independent(&table));
    let captured = pad::mark::capture(&device, &call).unwrap();
    pad::mark::dispatch(&device, 100, Independent(&table)).unwrap();
    let gpu = device.read_typed_buffer(&table).unwrap();
    let marked = |values: &[u32]| values.iter().take_while(|value| **value == 1).count();
    assert_eq!(marked(&gpu), 128, "the GPU's padding invocations write too");
    assert_eq!(gpu.iter().filter(|value| **value == 1).count(), 128);

    captured.run_all().unwrap();
    let twin = captured.buffer::<u32>("table").unwrap().read();
    assert_eq!(twin, gpu, "the whole launch equals the GPU's result");

    // `run_until` stops at its id: ids 0..=99 only.
    captured.run_until([99, 0, 0]).unwrap();
    let partial = captured.buffer::<u32>("table").unwrap().read();
    assert_eq!(marked(&partial), 100);
    assert_eq!(partial.iter().filter(|value| **value == 1).count(), 100);
}

#[test]
fn a_call_the_device_refuses_is_not_captured() {
    let device = capture_device("capture aliased");
    let params = device
        .create_uniform(
            "params",
            &sim::Params {
                dt: DT,
                gain: GAIN,
                pad0: 0.0,
                pad1: 0.0,
            },
        )
        .unwrap();
    let both = device.create_typed_buffer("both", &input_values(), GpuBufferAccess::ReadWrite);
    // One buffer as the read-only `input` and the read-write `out`.
    let call = sim::step::call(N as u32, &params, &both, &both);
    let dispatched = sim::step::dispatch(&device, N as u32, &params, &both, &both).unwrap_err();
    assert!(
        matches!(dispatched, Error::AliasedBinding { .. }),
        "{dispatched:?}"
    );
    match sim::step::capture(&device, &call) {
        Err(CaptureError::Call(error @ Error::AliasedBinding { .. })) => {
            assert_eq!(error.to_string(), dispatched.to_string());
        }
        other => panic!("expected the dispatch's aliasing error, got {other:?}"),
    }
}

#[test]
fn a_gpu_vec_input_captures_without_the_option_with_its_logical_length() {
    let handle = common::device_handle();
    let device = &*handle;
    eprintln!("capture vec adapter: {}", device.adapter_info().name);
    assert!(!device.capture_enabled());
    let params = device
        .create_uniform(
            "params",
            &sim::Params {
                dt: DT,
                gain: GAIN,
                pad0: 0.0,
                pad1: 0.0,
            },
        )
        .unwrap();
    let input = gust_wgpu::GpuVec::from_slice_on(&handle, &input_values()).unwrap();
    let out = device.create_typed_buffer("out", &sentinel_values(), GpuBufferAccess::ReadWrite);
    let captured =
        sim::step::capture(device, &sim::step::call(N as u32, &params, &input, &out)).unwrap();
    let copy = captured.buffer::<sim::Particle>("input").unwrap().read();
    assert_eq!(
        copy.len(),
        N,
        "the logical length, not the pool's size class"
    );
    assert_eq!(bits(&copy), bits(&input_values()));
    captured.run_all().unwrap();
    let twin = bits(&captured.buffer::<sim::Particle>("out").unwrap().read());
    assert_eq!(twin, (0..N).map(expected).collect::<Vec<_>>());
}

#[test]
fn an_edited_input_stays_edited_and_the_outputs_are_still_restored() {
    let device = capture_device("capture edit");
    let params = device
        .create_uniform(
            "params",
            &sim::Params {
                dt: DT,
                gain: GAIN,
                pad0: 0.0,
                pad1: 0.0,
            },
        )
        .unwrap();
    let input = device.create_typed_buffer("input", &input_values(), GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("out", &sentinel_values(), GpuBufferAccess::ReadWrite);
    let captured =
        sim::step::capture(&device, &sim::step::call(N as u32, &params, &input, &out)).unwrap();
    // "What if this input were different?": the edit survives later runs.
    let mut changed = input_values();
    changed[5].pos = 100.0;
    captured
        .buffer::<sim::Particle>("input")
        .unwrap()
        .write(&changed)
        .unwrap();
    captured.run_element([5, 0, 0]).unwrap();
    let read = || bits(&captured.buffer::<sim::Particle>("out").unwrap().read());
    assert_eq!(read()[5].0, (100.0 + changed[5].vel * DT).to_bits());
    // The output is restored: running another element does not keep element 5.
    captured.run_element([6, 0, 0]).unwrap();
    assert_eq!(read()[5], bits(&sentinel_values())[5]);
    assert_eq!(read()[6], expected(6));
}
