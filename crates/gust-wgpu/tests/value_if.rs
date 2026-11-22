//! `if`/`else` as the value of a `let`, on the GPU and on the CPU twin (slice m12.ifx; Gpu tier).
//!
//! The kernel below binds the value of an `if` chain in six ways: a two-arm `if`, an `else if` chain of `i32`, a value
//! whose two branches each run an atomic add before their value, a vector, a nested `if`, and a `let` that shadows an
//! outer local of its own name inside a block. The macro lowers each to a local declared before the `if` and assigned
//! in the branch that runs, and leaves the Rust code as it is for the twin, so the twin is a second assertion (rustc
//! gave the semantics) and never GPU evidence.
//!
//! The expectations are written out by hand from the rules in the kernel, not computed by a function that mirrors it.
//! The atomic adds are the evidence that exactly one branch runs: each of the `N` items bumps one of two counters, so
//! the counters add up to `N` and split by the parity of the index; a lowering that ran both branches would count `N`
//! in each.
//!
//! Index `i` of an element, `r = i % 3`:
//! - `plain`   = (r == 0: 10, else 20) + i
//! - `chain`   = (r == 0: -1, r == 1: 2, else 3) * i, as `i32`
//! - `effect`  = (i even: 100 + i, i odd: 200 + i); `hits[0]` counts the even items and `hits[1]` the odd ones
//! - `vector`  = (r == 0: Vec2(1, 2), else Vec2(0.5, 0.5)), then `x + 2 y`: 5 and 1.5, both exact
//! - `nested`  = (i < 30: (r == 0: 1, else 2), i < 60: 3, else 4)
//! - `shadow`  = with s = i + 1: (s > 35: 2 s, else s)

// The kernel lists one parameter per buffer.
#![allow(clippy::too_many_arguments)]

use gust::{Independent, cpu::CpuDevice};
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

#[gust::gpu]
mod value_if {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn branches(
        id: SV_DispatchThreadID,
        mut plain: RWStructuredBuffer<u32>,
        mut chain: RWStructuredBuffer<i32>,
        mut effect: RWStructuredBuffer<u32>,
        mut hits: RWStructuredBuffer<u32>,
        mut vector: RWStructuredBuffer<f32>,
        mut nested: RWStructuredBuffer<u32>,
        mut shadow: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < plain.len() {
            let r = i % 3u32;
            let p: u32 = if r == 0u32 { 10u32 } else { 20u32 };
            plain[i] = p + i;
            let c = if r == 0u32 {
                -1i32
            } else if r == 1u32 {
                2i32
            } else {
                3i32
            };
            chain[i] = c * (i as i32);
            let e: u32 = if (i & 1u32) == 0u32 {
                atomic_add(&mut hits[0u32], 1u32);
                100u32 + i
            } else {
                atomic_add(&mut hits[1u32], 1u32);
                200u32 + i
            };
            effect[i] = e;
            let v: Vec2 = if r == 0u32 {
                Vec2::new(1.0, 2.0)
            } else {
                Vec2::splat(0.5)
            };
            vector[i] = v.x + v.y * 2.0;
            let n: u32 = if i < 30u32 {
                if r == 0u32 { 1u32 } else { 2u32 }
            } else if i < 60u32 {
                3u32
            } else {
                4u32
            };
            nested[i] = n;
            let s: u32 = i + 1u32;
            {
                let s: u32 = if s > 35u32 { s * 2u32 } else { s };
                shadow[i] = s;
            }
        }
    }
}

/// Not a multiple of the workgroup width (64), so the guard matters; odd, so the parities differ in count.
const N: usize = 71;

struct Expected {
    plain: Vec<u32>,
    chain: Vec<i32>,
    effect: Vec<u32>,
    hits: [u32; 2],
    vector: Vec<f32>,
    nested: Vec<u32>,
    shadow: Vec<u32>,
}

fn expected() -> Expected {
    let index: Vec<u32> = (0..N as u32).collect();
    Expected {
        plain: index
            .iter()
            .map(|&i| if i.is_multiple_of(3) { 10 + i } else { 20 + i })
            .collect(),
        chain: index
            .iter()
            .map(|&i| match i % 3 {
                0 => -(i as i32),
                1 => 2 * (i as i32),
                _ => 3 * (i as i32),
            })
            .collect(),
        effect: index
            .iter()
            .map(|&i| {
                if i.is_multiple_of(2) {
                    100 + i
                } else {
                    200 + i
                }
            })
            .collect(),
        // 71 items: indices 0, 2, .., 70 are even (36 of them), 1, 3, .., 69 are odd (35).
        hits: [36, 35],
        vector: index
            .iter()
            .map(|&i| if i.is_multiple_of(3) { 5.0 } else { 1.5 })
            .collect(),
        nested: index
            .iter()
            .map(|&i| match i {
                0..=29 if i.is_multiple_of(3) => 1,
                0..=29 => 2,
                30..=59 => 3,
                _ => 4,
            })
            .collect(),
        shadow: index
            .iter()
            .map(|&i| if i + 1 > 35 { 2 * (i + 1) } else { i + 1 })
            .collect(),
    }
}

#[test]
fn the_host_expectation_has_the_values_the_comment_gives() {
    let want = expected();
    // Spot values worked out from the rules above.
    assert_eq!(&want.plain[..4], [10, 21, 22, 13]);
    assert_eq!(&want.chain[..4], [0, 2, 6, -3]);
    assert_eq!(&want.effect[..3], [100, 201, 102]);
    assert_eq!(&want.vector[..3], [5.0, 1.5, 1.5]);
    assert_eq!(&want.nested[28..32], [2, 2, 3, 3]);
    assert_eq!(
        (want.nested[60], want.shadow[34], want.shadow[35]),
        (4, 35, 72)
    );
    assert_eq!(want.hits[0] + want.hits[1], N as u32);
}

#[test]
fn a_value_if_runs_exactly_one_branch_on_the_gpu_and_the_twin() {
    let want = expected();
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let info = device.adapter_info();
    eprintln!(
        "value_if adapter: {} backend: {:?}",
        info.name, info.backend
    );

    let plain = device.create_typed_buffer("if plain", &[0u32; N], GpuBufferAccess::ReadWrite);
    let chain = device.create_typed_buffer("if chain", &[0i32; N], GpuBufferAccess::ReadWrite);
    let effect = device.create_typed_buffer("if effect", &[0u32; N], GpuBufferAccess::ReadWrite);
    let hits = device.create_typed_buffer("if hits", &[0u32; 2], GpuBufferAccess::ReadWrite);
    let vector = device.create_typed_buffer("if vector", &[0f32; N], GpuBufferAccess::ReadWrite);
    let nested = device.create_typed_buffer("if nested", &[0u32; N], GpuBufferAccess::ReadWrite);
    let shadow = device.create_typed_buffer("if shadow", &[0u32; N], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &value_if::branches::DESCRIPTOR,
            N as u32,
            &[
                BufferBinding::read_write(&plain).independent_length(),
                BufferBinding::read_write(&chain).independent_length(),
                BufferBinding::read_write(&effect).independent_length(),
                BufferBinding::read_write(&hits).independent_length(),
                BufferBinding::read_write(&vector).independent_length(),
                BufferBinding::read_write(&nested).independent_length(),
                BufferBinding::read_write(&shadow).independent_length(),
            ],
        )
        .unwrap();
    assert_eq!(
        device.read_typed_buffer(&plain).unwrap(),
        want.plain,
        "gpu: two-arm"
    );
    assert_eq!(
        device.read_typed_buffer(&chain).unwrap(),
        want.chain,
        "gpu: else-if chain"
    );
    assert_eq!(
        device.read_typed_buffer(&effect).unwrap(),
        want.effect,
        "gpu: value of the taken branch"
    );
    assert_eq!(
        device.read_typed_buffer(&hits).unwrap(),
        want.hits,
        "gpu: exactly one branch ran per item"
    );
    assert_eq!(
        device.read_typed_buffer(&vector).unwrap(),
        want.vector,
        "gpu: vector"
    );
    assert_eq!(
        device.read_typed_buffer(&nested).unwrap(),
        want.nested,
        "gpu: nested"
    );
    assert_eq!(
        device.read_typed_buffer(&shadow).unwrap(),
        want.shadow,
        "gpu: shadowing let"
    );
    eprintln!("value_if: elements={N} branch_counts_exact=true values_exact=true");

    let cpu = CpuDevice::new();
    let twin_plain = cpu.buffer(&[0u32; N]);
    let twin_chain = cpu.buffer(&[0i32; N]);
    let twin_effect = cpu.buffer(&[0u32; N]);
    let twin_hits = cpu.buffer(&[0u32; 2]);
    let twin_vector = cpu.buffer(&[0f32; N]);
    let twin_nested = cpu.buffer(&[0u32; N]);
    let twin_shadow = cpu.buffer(&[0u32; N]);
    value_if::branches::dispatch(
        &cpu,
        N as u32,
        Independent(&twin_plain),
        Independent(&twin_chain),
        Independent(&twin_effect),
        Independent(&twin_hits),
        Independent(&twin_vector),
        Independent(&twin_nested),
        Independent(&twin_shadow),
    )
    .unwrap();
    assert_eq!(twin_plain.read(), want.plain, "twin: two-arm");
    assert_eq!(twin_chain.read(), want.chain, "twin: else-if chain");
    assert_eq!(twin_effect.read(), want.effect, "twin: taken branch");
    assert_eq!(twin_hits.read(), want.hits, "twin: one branch per item");
    assert_eq!(twin_vector.read(), want.vector, "twin: vector");
    assert_eq!(twin_nested.read(), want.nested, "twin: nested");
    assert_eq!(twin_shadow.read(), want.shadow, "twin: shadowing let");
}
