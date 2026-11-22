//! A value `match` as the initializer of an untyped `let`, on the GPU and on the CPU twin (slice m13.matchu; Gpu tier).
//!
//! The kernel below binds the value of a `match` on `i % 3` without a type annotation, in seven ways: a block arm, an
//! `i32`, an `f32`, a vector, a nested `match` in arms, a `let` that shadows an outer local of its own
//! name inside a block, and an arm that returns. The type of each local comes from its arms (the rule of an untyped
//! value `if`), the macro lowers the `let` to a local declared before a `switch` that assigns it, and the Rust code
//! stays as it is for the twin, so the twin is a second assertion (rustc gave the semantics) and never GPU evidence.
//!
//! The expectations are written out by hand from the rules in the kernel, not computed by a function that mirrors it.
//!
//! Index `i` of an element, `r = i % 3`:
//! - `plain`   = (r == 0: 2 i + 10, else 20), the first arm a block whose tail shows no type, the second arm `20u32`
//! - `chain`   = (r == 0: -1, r == 1: 2, else 3) * i, as `i32`
//! - `scalar`  = (r == 0: 0.5, r == 1: 1.5, else 2.5), all exact in `f32`
//! - `vector`  = (r == 0: Vec2(1, 2), else Vec2(0.5, 0.5)), then `x + 2 y`: 5 and 1.5
//! - `nested`  = (r == 0: a nested `match` (i < 30: 1, else 2), r == 1: a nested `match` (i even: 3, odd: 4), else 5)
//! - `shadow`  = with s = i + 1, the inner `s` is `2 s` (the arm for 0 is never taken)
//! - `early`   = (r == 0: the arm returns, so the element keeps its initial 0; r == 1: 11; else 12)

// The kernel lists one parameter per buffer.
#![allow(clippy::too_many_arguments)]

use gust::{Independent, cpu::CpuDevice};
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

#[gust::gpu]
mod value_match_untyped {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn arms(
        id: SV_DispatchThreadID,
        mut plain: RWStructuredBuffer<u32>,
        mut chain: RWStructuredBuffer<i32>,
        mut scalar: RWStructuredBuffer<f32>,
        mut vector: RWStructuredBuffer<f32>,
        mut nested: RWStructuredBuffer<u32>,
        mut shadow: RWStructuredBuffer<u32>,
        mut early: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < plain.len() {
            let r = i % 3u32;
            let p = match r {
                0 => {
                    let t = i * 2u32;
                    t + 10u32
                }
                _ => 20u32,
            };
            plain[i] = p;
            let c = match r {
                0 => -1i32,
                1 => 2i32,
                _ => 3i32,
            };
            chain[i] = c * (i as i32);
            let f = match r {
                0 => 0.5f32,
                1 => 1.5f32,
                _ => 2.5f32,
            };
            scalar[i] = f;
            let v = match r {
                0 => Vec2::new(1.0, 2.0),
                _ => Vec2::splat(0.5),
            };
            vector[i] = v.x + v.y * 2.0;
            let n = match r {
                0 => match i / 30u32 {
                    0 => 1u32,
                    _ => 2u32,
                },
                1 => match i % 2u32 {
                    0 => 3u32,
                    _ => 4u32,
                },
                _ => 5u32,
            };
            nested[i] = n;
            let s: u32 = i + 1u32;
            {
                let s = match s {
                    0 => 0u32,
                    _ => s * 2u32,
                };
                shadow[i] = s;
            }
            let e = match r {
                0 => {
                    return;
                }
                1 => 11u32,
                _ => 12u32,
            };
            early[i] = e;
        }
    }
}

/// Not a multiple of the workgroup width (64), so the guard matters; large enough to cross `i = 30` in the `if` arm.
const N: usize = 71;

struct Expected {
    plain: Vec<u32>,
    chain: Vec<i32>,
    scalar: Vec<f32>,
    vector: Vec<f32>,
    nested: Vec<u32>,
    shadow: Vec<u32>,
    early: Vec<u32>,
}

fn expected() -> Expected {
    let index: Vec<u32> = (0..N as u32).collect();
    Expected {
        plain: index
            .iter()
            .map(|&i| if i.is_multiple_of(3) { 2 * i + 10 } else { 20 })
            .collect(),
        chain: index
            .iter()
            .map(|&i| match i % 3 {
                0 => -(i as i32),
                1 => 2 * (i as i32),
                _ => 3 * (i as i32),
            })
            .collect(),
        scalar: index
            .iter()
            .map(|&i| match i % 3 {
                0 => 0.5,
                1 => 1.5,
                _ => 2.5,
            })
            .collect(),
        vector: index
            .iter()
            .map(|&i| if i.is_multiple_of(3) { 5.0 } else { 1.5 })
            .collect(),
        nested: index
            .iter()
            .map(|&i| match i % 3 {
                0 if i < 30 => 1,
                0 => 2,
                1 if i.is_multiple_of(2) => 3,
                1 => 4,
                _ => 5,
            })
            .collect(),
        shadow: index.iter().map(|&i| 2 * (i + 1)).collect(),
        early: index
            .iter()
            .map(|&i| match i % 3 {
                0 => 0,
                1 => 11,
                _ => 12,
            })
            .collect(),
    }
}

#[test]
fn the_host_expectation_has_the_values_the_comment_gives() {
    let want = expected();
    // Spot values worked out from the rules above.
    assert_eq!(&want.plain[..4], [10, 20, 20, 16]);
    assert_eq!(&want.chain[..4], [0, 2, 6, -3]);
    assert_eq!(&want.scalar[..3], [0.5, 1.5, 2.5]);
    assert_eq!(&want.vector[..3], [5.0, 1.5, 1.5]);
    assert_eq!(&want.nested[27..34], [1, 3, 5, 2, 4, 5, 2]);
    assert_eq!((want.shadow[0], want.shadow[70]), (2, 142));
    assert_eq!(&want.early[..4], [0, 11, 12, 0]);
}

#[test]
fn an_untyped_value_match_takes_its_type_from_its_arms_on_the_gpu_and_the_twin() {
    let want = expected();
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let info = device.adapter_info();
    eprintln!(
        "value_match_untyped adapter: {} backend: {:?}",
        info.name, info.backend
    );

    let plain = device.create_typed_buffer("mu plain", &[0u32; N], GpuBufferAccess::ReadWrite);
    let chain = device.create_typed_buffer("mu chain", &[0i32; N], GpuBufferAccess::ReadWrite);
    let scalar = device.create_typed_buffer("mu scalar", &[0f32; N], GpuBufferAccess::ReadWrite);
    let vector = device.create_typed_buffer("mu vector", &[0f32; N], GpuBufferAccess::ReadWrite);
    let nested = device.create_typed_buffer("mu nested", &[0u32; N], GpuBufferAccess::ReadWrite);
    let shadow = device.create_typed_buffer("mu shadow", &[0u32; N], GpuBufferAccess::ReadWrite);
    let early = device.create_typed_buffer("mu early", &[0u32; N], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &value_match_untyped::arms::DESCRIPTOR,
            N as u32,
            &[
                BufferBinding::read_write(&plain).independent_length(),
                BufferBinding::read_write(&chain).independent_length(),
                BufferBinding::read_write(&scalar).independent_length(),
                BufferBinding::read_write(&vector).independent_length(),
                BufferBinding::read_write(&nested).independent_length(),
                BufferBinding::read_write(&shadow).independent_length(),
                BufferBinding::read_write(&early).independent_length(),
            ],
        )
        .unwrap();
    assert_eq!(
        device.read_typed_buffer(&plain).unwrap(),
        want.plain,
        "gpu: block arm"
    );
    assert_eq!(
        device.read_typed_buffer(&chain).unwrap(),
        want.chain,
        "gpu: i32"
    );
    assert_eq!(
        device.read_typed_buffer(&scalar).unwrap(),
        want.scalar,
        "gpu: f32"
    );
    assert_eq!(
        device.read_typed_buffer(&vector).unwrap(),
        want.vector,
        "gpu: vector"
    );
    assert_eq!(
        device.read_typed_buffer(&nested).unwrap(),
        want.nested,
        "gpu: nested match"
    );
    assert_eq!(
        device.read_typed_buffer(&shadow).unwrap(),
        want.shadow,
        "gpu: shadowing let"
    );
    assert_eq!(
        device.read_typed_buffer(&early).unwrap(),
        want.early,
        "gpu: returning arm"
    );
    eprintln!("value_match_untyped: elements={N} values_exact=true");

    let cpu = CpuDevice::new();
    let twin_plain = cpu.buffer(&[0u32; N]);
    let twin_chain = cpu.buffer(&[0i32; N]);
    let twin_scalar = cpu.buffer(&[0f32; N]);
    let twin_vector = cpu.buffer(&[0f32; N]);
    let twin_nested = cpu.buffer(&[0u32; N]);
    let twin_shadow = cpu.buffer(&[0u32; N]);
    let twin_early = cpu.buffer(&[0u32; N]);
    value_match_untyped::arms::dispatch(
        &cpu,
        N as u32,
        Independent(&twin_plain),
        Independent(&twin_chain),
        Independent(&twin_scalar),
        Independent(&twin_vector),
        Independent(&twin_nested),
        Independent(&twin_shadow),
        Independent(&twin_early),
    )
    .unwrap();
    assert_eq!(twin_plain.read(), want.plain, "twin: block arm");
    assert_eq!(twin_chain.read(), want.chain, "twin: i32");
    assert_eq!(twin_scalar.read(), want.scalar, "twin: f32");
    assert_eq!(twin_vector.read(), want.vector, "twin: vector");
    assert_eq!(twin_nested.read(), want.nested, "twin: nested match");
    assert_eq!(twin_shadow.read(), want.shadow, "twin: shadowing let");
    assert_eq!(twin_early.read(), want.early, "twin: returning arm");
}
