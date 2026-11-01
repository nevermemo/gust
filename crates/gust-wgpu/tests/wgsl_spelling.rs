//! WGSL spelling of two lowerings that were correct Slang and invalid WGSL (slice fx.wgsl).
//!
//! 1. slangc drops the parentheses around comparison arguments, so `f(u < 5, s > 0)` reaches WGSL as
//!    `f(u_0 < u32(5), s_0 > i32(0))`, and WGSL's template-list discovery reads `u_0 < .. >` as a template list.
//!    The kernel states the shapes that triggered it, with loop-carried locals so that slangc cannot fold or hoist
//!    the comparisons.
//! 2. An unsuffixed integer literal of 2^31 or more is typed int64 by slangc, which wgpu refuses. The dialect has
//!    no 64-bit integers, so such a literal can only be `u32`.
//!
//! The expected values are Rust literals worked out by hand, and the CPU twin must agree with them too.

// The kernels state their operands on purpose and keep the plain `return` form the subset accepts.
#![allow(clippy::needless_return, clippy::unnecessary_cast)]

use gust::cpu::CpuDevice;
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

#[gust::gpu]
mod wgsl_tmpl {
    fn both(x: bool, y: bool) -> bool {
        if x {
            return y;
        }
        return false;
    }

    fn all3(x: bool, y: bool, z: bool) -> bool {
        if x && y {
            return z;
        }
        return false;
    }

    fn successor(x: uint) -> uint {
        x + 1u32
    }

    fn flag(value: bool) -> uint {
        if value {
            return 1u32;
        }
        return 0u32;
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i: uint = id.x;
        if i == 0u32 {
            // Loop-carried locals: u = 0 + 1 + 2 + 3 = 6 and s = -3 + 4 = 1 after the loop.
            let mut u: uint = a[1];
            let mut s: int = a[2] as int;
            for k in 0u32..a[0] {
                u += k;
                s += 1i32;
            }
            // A call whose arguments are a `<` and then a `>` or `>=` comparison. Every comparison is written once
            // with its own constant: slangc hoists a repeated one into a temporary, which hides the problem.
            out[0] = flag(both(u < 5u32, s > 0));
            out[1] = flag(both(u < 7u32, s > -1));
            out[2] = flag(both(u < 8u32, s >= 1));
            out[3] = flag(both(u < 9u32, s >= 2));
            // `^` on two comparisons goes through `__gust_xor(a, b)`, a call as well.
            out[4] = flag((u < 3u32) ^ (s > -2));
            out[5] = flag((u < 10u32) ^ (s > -3));
            out[6] = flag((u < 11u32) ^ (s >= 3));
            // The `<` in the middle of a list, in a nested list, on a float, and with a call on its right.
            let f: float = u as float;
            out[7] = flag(all3(s > -4, u < 12u32, s >= -5));
            out[8] = flag(both(both(u < 13u32, s > -6), s >= -7));
            out[9] = flag(both(f < 6.5, s > -8));
            out[10] = flag(both(u < successor(u), s > -9));
            // A `<` after a `&&` in a non-last element.
            let t: bool = a[0] == 4u32;
            out[11] = flag(both(t && (u < 14u32), s > -10));
        }
    }
}

#[gust::gpu]
mod wgsl_lit {
    fn flag(value: bool) -> uint {
        if value {
            return 1u32;
        }
        return 0u32;
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i: uint = id.x;
        if i == 0u32 {
            let x: uint = a[0];
            let m: uint = a[1];
            let n: int = a[2] as int;
            // Unsuffixed, 2^31 or more: the dialect has no 64-bit integers, so these are `u32`.
            out[0] = x + 3000000000;
            out[1] = flag(m == 4294967295);
            out[2] = 2147483648;
            // The suffixed spelling is the reference.
            out[3] = x + 3000000000u32;
            out[4] = flag(x == 4294967295);
            out[5] = a[3] + 2147483648;
            out[6] = 0x8000_0000;
            // The edges that must keep their meaning: i32::MAX unsuffixed, and i32::MIN as a negated literal.
            out[7] = flag(n < 2147483647);
            out[8] = flag(n > -2147483648);
            out[9] = flag(m > 2147483647);
            out[10] = flag(m >= 2147483648);
        }
    }
}

/// Every buffer has one element per invocation, so both are 64 long; the kernels read and write the first few.
const COUNT: usize = 64;

fn input(values: &[u32]) -> Vec<u32> {
    let mut input = vec![0u32; COUNT];
    input[..values.len()].copy_from_slice(values);
    input
}

#[test]
fn comparison_arguments_do_not_form_wgsl_template_lists() {
    let input_data = input(&[4, 0, -3i32 as u32]);
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let a = device.create_typed_buffer("tmpl input", &input_data, GpuBufferAccess::ReadOnly);
    let out =
        device.create_typed_buffer("tmpl out", &vec![0u32; COUNT], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &wgsl_tmpl::run::DESCRIPTOR,
            64,
            &[
                BufferBinding::read_only(&a),
                BufferBinding::read_write(&out),
            ],
        )
        .unwrap();
    let actual = device.read_typed_buffer(&out).unwrap();

    let cpu = CpuDevice::new();
    let twin_input = cpu.buffer(&input_data);
    let twin_out = cpu.buffer(&vec![0u32; COUNT]);
    wgsl_tmpl::run::dispatch(&cpu, COUNT as u32, &twin_input, &twin_out).unwrap();
    let twin = twin_out.read();

    // u = 6 and s = 1: both(F, T), both(T, T), both(T, T), both(T, F), then F ^ T, T ^ T, T ^ F.
    // The four shapes after those are all true: u = 6.0 < 6.5 and u < u + 1, and t && (u < 14) with t true.
    let expected = [0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1];
    assert_eq!(twin[..12], expected, "CPU twin against the Rust literals");
    assert_eq!(actual[..12], expected, "GPU against the Rust literals");
}

#[test]
fn large_unsuffixed_literals_are_u32_on_gpu() {
    let input_data = input(&[1_000_000_000, u32::MAX, 5, 5]);
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let a = device.create_typed_buffer("lit input", &input_data, GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("lit out", &vec![0u32; COUNT], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &wgsl_lit::run::DESCRIPTOR,
            64,
            &[
                BufferBinding::read_only(&a),
                BufferBinding::read_write(&out),
            ],
        )
        .unwrap();
    let actual = device.read_typed_buffer(&out).unwrap();

    let cpu = CpuDevice::new();
    let twin_input = cpu.buffer(&input_data);
    let twin_out = cpu.buffer(&vec![0u32; COUNT]);
    wgsl_lit::run::dispatch(&cpu, COUNT as u32, &twin_input, &twin_out).unwrap();
    let twin = twin_out.read();

    let expected = [
        4_000_000_000, // 1e9 + 3e9
        1,             // u32::MAX == 4294967295
        2_147_483_648,
        4_000_000_000, // the suffixed spelling
        0,             // 1e9 != 4294967295
        2_147_483_653, // 5 + 2^31
        2_147_483_648, // 0x8000_0000
        1,             // 5 < i32::MAX
        1,             // 5 > i32::MIN
        1,             // u32::MAX > i32::MAX as a u32
        1,             // u32::MAX >= 2^31
    ];
    assert_eq!(twin[..11], expected, "CPU twin against the Rust literals");
    assert_eq!(actual[..11], expected, "GPU against the Rust literals");
}
