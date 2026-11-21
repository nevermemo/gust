//! A module-level `const` shared by the kernels of a `#[gpu]` module, on the GPU and on the CPU twin (slice m11.g3,
//! D205; Gpu tier).
//!
//! The module below holds four `pub const` items (a `u32`, a negative `i32`, an `f32` and a `bool`, plus a `u32` that is
//! the largest value) read by a helper and by two kernels. The macro lowers each to one `static const` line in the text
//! of every kernel and leaves the Rust item as it is, so the host reads the same items: the expected values below are
//! computed from `shared::WIDTH` and the others, never from a copy of the number. Every value is a small integer or a
//! binary fraction, so each result is exactly representable and the comparisons are exact.
//!
//! The twin is a second assertion and never GPU evidence.

use gust::{Independent, cpu::CpuDevice};
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

#[gust::gpu]
mod shared {
    pub const WIDTH: u32 = 4096;
    pub const OFFSET: i32 = -7;
    pub const SCALE: f32 = 0.25;
    pub const ON: bool = true;
    pub const OFF: bool = false;
    pub const MASK: u32 = 4294967295;
    pub const LOW: i32 = -2147483648;
    pub const LOW_SUFFIXED: i32 = -2147483648i32;
    pub const NEG_REAL: f32 = -1.5;
    pub const TWO: f32 = 2f32;

    pub struct Acc {
        pub n: u32,
    }

    impl Acc {
        // A method of a struct reads a module const.
        pub fn bumped(&self) -> u32 {
            self.n + WIDTH
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn extras(
        id: SV_DispatchThreadID,
        src: StructuredBuffer<u32>,
        mut lows: RWStructuredBuffer<i32>,
        mut lows_suffixed: RWStructuredBuffer<i32>,
        mut negs: RWStructuredBuffer<f32>,
        mut methods: RWStructuredBuffer<u32>,
        mut closures: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < src.len() {
            lows[i] = LOW + (src[i] as i32);
            lows_suffixed[i] = LOW_SUFFIXED + (src[i] as i32);
            negs[i] = NEG_REAL * TWO * (src[i] as f32);
            let acc = Acc { n: src[i] };
            methods[i] = acc.bumped();
            // A closure reads a module const without capturing it.
            let add = |x: u32| -> u32 { x + WIDTH };
            let once = add(src[i]);
            closures[i] = add(once - WIDTH);
        }
    }

    fn lift(x: u32) -> u32 {
        if ON { x + WIDTH } else { x }
    }

    fn shift(x: u32) -> i32 {
        (x as i32) + OFFSET
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn integers(
        id: SV_DispatchThreadID,
        src: StructuredBuffer<u32>,
        mut wide: RWStructuredBuffer<u32>,
        mut signed: RWStructuredBuffer<i32>,
    ) {
        let i = id.x;
        if i < src.len() {
            wide[i] = lift(src[i]) + WIDTH;
            signed[i] = shift(src[i]) + OFFSET;
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn reals(
        id: SV_DispatchThreadID,
        src: StructuredBuffer<u32>,
        mut scaled: RWStructuredBuffer<f32>,
        mut flags: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < src.len() {
            scaled[i] = (lift(src[i]) as f32) * SCALE;
            let mut on: u32 = 0u32;
            if ON {
                on = 1u32;
            }
            let mut off: u32 = 0u32;
            if OFF {
                off = 1u32;
            }
            flags[i] = ((src[i] ^ MASK) & 255u32) + on * 10u32 + off;
        }
    }
}

/// The element count of every dispatch: not a multiple of the workgroup width (64), so the guard matters.
const N: usize = 70;

fn sources() -> Vec<u32> {
    (0..N as u32).map(|k| k * 3).collect()
}

/// The host's expectation, computed from the same `pub const` items the shader reads.
struct Expected {
    wide: Vec<u32>,
    signed: Vec<i32>,
    scaled: Vec<f32>,
    flags: Vec<u32>,
}

fn expected(src: &[u32]) -> Expected {
    let on = u32::from(shared::ON);
    let off = u32::from(shared::OFF);
    Expected {
        wide: src
            .iter()
            .map(|&x| x + shared::WIDTH + shared::WIDTH)
            .collect(),
        signed: src
            .iter()
            .map(|&x| x as i32 + shared::OFFSET + shared::OFFSET)
            .collect(),
        scaled: src
            .iter()
            .map(|&x| (x + shared::WIDTH) as f32 * shared::SCALE)
            .collect(),
        flags: src
            .iter()
            .map(|&x| ((x ^ shared::MASK) & 255) + on * 10 + off)
            .collect(),
    }
}

#[test]
fn the_host_expectation_reads_the_constants_it_claims_to() {
    // The constants are what the test is about: a change to one must change the expectation, not leave it behind.
    assert_eq!(shared::WIDTH, 4096);
    assert_eq!(shared::OFFSET, -7);
    assert_eq!(shared::SCALE, 0.25);
    assert_eq!((shared::ON, shared::OFF), (true, false));
    assert_eq!(shared::MASK, u32::MAX);
    let e = expected(&[0, 3]);
    assert_eq!(e.wide, [8192, 8195]);
    assert_eq!(e.signed, [-14, -11]);
    assert_eq!(e.scaled, [1024.0, 1024.75]);
    assert_eq!(e.flags, [255 + 10, 252 + 10]);
}

#[test]
fn two_kernels_and_a_helper_read_the_same_module_constants_on_the_gpu_and_the_twin() {
    let src = sources();
    let want = expected(&src);

    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let info = device.adapter_info();
    eprintln!("adapter: {} backend: {:?}", info.name, info.backend);

    let source = device.create_typed_buffer("consts src", &src, GpuBufferAccess::ReadOnly);
    let wide = device.create_typed_buffer("consts wide", &[0u32; N], GpuBufferAccess::ReadWrite);
    let signed =
        device.create_typed_buffer("consts signed", &[0i32; N], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &shared::integers::DESCRIPTOR,
            N as u32,
            &[
                BufferBinding::read_only(&source).independent_length(),
                BufferBinding::read_write(&wide).independent_length(),
                BufferBinding::read_write(&signed).independent_length(),
            ],
        )
        .unwrap();
    let gpu_wide = device.read_typed_buffer(&wide).unwrap();
    let gpu_signed = device.read_typed_buffer(&signed).unwrap();
    assert_eq!(gpu_wide, want.wide, "gpu: integers, u32 constants");
    assert_eq!(gpu_signed, want.signed, "gpu: integers, i32 constants");

    let scaled =
        device.create_typed_buffer("consts scaled", &[0f32; N], GpuBufferAccess::ReadWrite);
    let flags = device.create_typed_buffer("consts flags", &[0u32; N], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &shared::reals::DESCRIPTOR,
            N as u32,
            &[
                BufferBinding::read_only(&source).independent_length(),
                BufferBinding::read_write(&scaled).independent_length(),
                BufferBinding::read_write(&flags).independent_length(),
            ],
        )
        .unwrap();
    let gpu_scaled = device.read_typed_buffer(&scaled).unwrap();
    let gpu_flags = device.read_typed_buffer(&flags).unwrap();
    assert_eq!(gpu_scaled, want.scaled, "gpu: reals, f32 constant");
    assert_eq!(gpu_flags, want.flags, "gpu: reals, bool and mask constants");
    eprintln!(
        "consts: elements={N} u32_exact=true i32_exact=true f32_exact=true bool_exact=true mask_exact=true"
    );

    // The twin: the same typed calls through `CpuDevice`, asserted against the host as well.
    let cpu = CpuDevice::new();
    let twin_src = cpu.buffer(&src);
    let twin_wide = cpu.buffer(&[0u32; N]);
    let twin_signed = cpu.buffer(&[0i32; N]);
    shared::integers::dispatch(
        &cpu,
        N as u32,
        Independent(&twin_src),
        Independent(&twin_wide),
        Independent(&twin_signed),
    )
    .unwrap();
    assert_eq!(twin_wide.read(), want.wide, "twin: integers, u32 constants");
    assert_eq!(
        twin_signed.read(),
        want.signed,
        "twin: integers, i32 constants"
    );
    let twin_scaled = cpu.buffer(&[0f32; N]);
    let twin_flags = cpu.buffer(&[0u32; N]);
    shared::reals::dispatch(
        &cpu,
        N as u32,
        Independent(&twin_src),
        Independent(&twin_scaled),
        Independent(&twin_flags),
    )
    .unwrap();
    assert_eq!(twin_scaled.read(), want.scaled, "twin: reals, f32 constant");
    assert_eq!(
        twin_flags.read(),
        want.flags,
        "twin: reals, bool and mask constants"
    );
}

#[test]
fn extreme_literals_a_struct_method_and_a_closure_read_constants_on_the_gpu_and_the_twin() {
    let src = sources();
    let lows: Vec<i32> = src.iter().map(|&x| shared::LOW + x as i32).collect();
    let lows_suffixed: Vec<i32> = src
        .iter()
        .map(|&x| shared::LOW_SUFFIXED + x as i32)
        .collect();
    let reals: Vec<f32> = src
        .iter()
        .map(|&x| shared::NEG_REAL * shared::TWO * x as f32)
        .collect();
    let methods: Vec<u32> = src.iter().map(|&x| x + shared::WIDTH).collect();
    let closures = methods.clone();
    assert_eq!(shared::LOW, i32::MIN);
    assert_eq!(shared::LOW_SUFFIXED, i32::MIN);

    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let info = device.adapter_info();
    eprintln!("adapter: {} backend: {:?}", info.name, info.backend);
    let source = device.create_typed_buffer("extras src", &src, GpuBufferAccess::ReadOnly);
    let g_lows = device.create_typed_buffer("extras lows", &[0i32; N], GpuBufferAccess::ReadWrite);
    let g_lows2 =
        device.create_typed_buffer("extras lows2", &[0i32; N], GpuBufferAccess::ReadWrite);
    let g_reals =
        device.create_typed_buffer("extras reals", &[0f32; N], GpuBufferAccess::ReadWrite);
    let g_methods =
        device.create_typed_buffer("extras methods", &[0u32; N], GpuBufferAccess::ReadWrite);
    let g_closures =
        device.create_typed_buffer("extras closures", &[0u32; N], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &shared::extras::DESCRIPTOR,
            N as u32,
            &[
                BufferBinding::read_only(&source).independent_length(),
                BufferBinding::read_write(&g_lows).independent_length(),
                BufferBinding::read_write(&g_lows2).independent_length(),
                BufferBinding::read_write(&g_reals).independent_length(),
                BufferBinding::read_write(&g_methods).independent_length(),
                BufferBinding::read_write(&g_closures).independent_length(),
            ],
        )
        .unwrap();
    assert_eq!(
        device.read_typed_buffer(&g_lows).unwrap(),
        lows,
        "gpu: i32::MIN, unsuffixed"
    );
    assert_eq!(
        device.read_typed_buffer(&g_lows2).unwrap(),
        lows_suffixed,
        "gpu: i32::MIN, suffixed"
    );
    assert_eq!(
        device.read_typed_buffer(&g_reals).unwrap(),
        reals,
        "gpu: negative and `2f32` f32 constants"
    );
    assert_eq!(
        device.read_typed_buffer(&g_methods).unwrap(),
        methods,
        "gpu: a struct method reads a constant"
    );
    assert_eq!(
        device.read_typed_buffer(&g_closures).unwrap(),
        closures,
        "gpu: a closure reads a constant"
    );

    let cpu = CpuDevice::new();
    let twin_src = cpu.buffer(&src);
    let t_lows = cpu.buffer(&[0i32; N]);
    let t_lows2 = cpu.buffer(&[0i32; N]);
    let t_reals = cpu.buffer(&[0f32; N]);
    let t_methods = cpu.buffer(&[0u32; N]);
    let t_closures = cpu.buffer(&[0u32; N]);
    shared::extras::dispatch(
        &cpu,
        N as u32,
        Independent(&twin_src),
        Independent(&t_lows),
        Independent(&t_lows2),
        Independent(&t_reals),
        Independent(&t_methods),
        Independent(&t_closures),
    )
    .unwrap();
    assert_eq!(t_lows.read(), lows, "twin: i32::MIN, unsuffixed");
    assert_eq!(t_lows2.read(), lows_suffixed, "twin: i32::MIN, suffixed");
    assert_eq!(
        t_reals.read(),
        reals,
        "twin: negative and `2f32` f32 constants"
    );
    assert_eq!(
        t_methods.read(),
        methods,
        "twin: a struct method reads a constant"
    );
    assert_eq!(
        t_closures.read(),
        closures,
        "twin: a closure reads a constant"
    );
}
