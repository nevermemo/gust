//! `bool ^ bool`, `bool ^= bool` and the other boolean binary operators on the GPU (slice fx.bxor).
//!
//! WGSL defines `^` only on integers, so a `bool ^` passed through to slangc fails at shader creation. The kernel
//! computes every input pair of each operator in both spellings, next to integer `^` and `^=` that must keep their
//! meaning. The expected values are Rust literals worked out by hand, and the CPU twin must agree with them too.

// The kernel states every operand pair on purpose, including `x ^ x`, and keeps the plain `return` form the subset
// accepts, so these two lints of the host-side view of the kernel body do not apply.
#![allow(clippy::eq_op, clippy::needless_return)]

use gust::cpu::CpuDevice;
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

#[gust::gpu]
mod bool_ops {
    // A `bool` field cannot exist: a module struct is `GpuPod`, and `bool` is not host-shareable. The `^=` on a
    // struct field is therefore an integer one.
    struct Words {
        first: uint,
        second: uint,
    }

    // `^` inside helpers: the value form, and the compound form on a parameter copy.
    fn xor_flag(x: bool, y: bool) -> bool {
        x ^ y
    }

    fn xor_assign_flag(x: bool, y: bool) -> bool {
        let mut r: bool = x;
        r ^= y;
        r
    }

    fn xor_word(x: uint, y: uint) -> uint {
        x ^ y
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
            // The four pairs are (false, false), (false, true), (true, false), (true, true).
            let f: bool = a[0] != 0u32;
            let t: bool = a[1] != 0u32;
            out[0] = flag(f ^ f);
            out[1] = flag(f ^ t);
            out[2] = flag(t ^ f);
            out[3] = flag(t ^ t);
            let mut x0: bool = f;
            x0 ^= f;
            let mut x1: bool = f;
            x1 ^= t;
            let mut x2: bool = t;
            x2 ^= f;
            let mut x3: bool = t;
            x3 ^= t;
            out[4] = flag(x0);
            out[5] = flag(x1);
            out[6] = flag(x2);
            out[7] = flag(x3);
            out[8] = flag(f & f);
            out[9] = flag(f & t);
            out[10] = flag(t & f);
            out[11] = flag(t & t);
            out[12] = flag(f | f);
            out[13] = flag(f | t);
            out[14] = flag(t | f);
            out[15] = flag(t | t);
            let mut y0: bool = f;
            y0 &= f;
            let mut y1: bool = f;
            y1 &= t;
            let mut y2: bool = t;
            y2 &= f;
            let mut y3: bool = t;
            y3 &= t;
            out[16] = flag(y0);
            out[17] = flag(y1);
            out[18] = flag(y2);
            out[19] = flag(y3);
            let mut z0: bool = f;
            z0 |= f;
            let mut z1: bool = f;
            z1 |= t;
            let mut z2: bool = t;
            z2 |= f;
            let mut z3: bool = t;
            z3 |= t;
            out[20] = flag(z0);
            out[21] = flag(z1);
            out[22] = flag(z2);
            out[23] = flag(z3);
            // A compound xor whose right side is an expression, and a nested use of the result.
            let mut w: bool = t;
            w ^= f ^ t;
            out[24] = flag(w);
            out[25] = flag((t ^ f) ^ (f ^ t));
            // Integer `^` and `^=` keep their meaning (uint, then int with a negative operand).
            let u: uint = a[2] ^ a[3];
            out[26] = u;
            let mut v: uint = a[2];
            v ^= a[3];
            out[27] = v;
            let p: int = (a[4] as int) ^ (a[5] as int);
            out[28] = p as uint;
            let mut q: int = a[4] as int;
            q ^= a[5] as int;
            out[29] = q as uint;
            // `^=` on a buffer element (a `uint` one: `bool` is not host-shareable), the place evaluated once.
            out[30] = a[2];
            out[30] ^= a[3];
            let k: uint = i + 31u32;
            out[k] = a[2];
            out[k] ^= a[4];
            // `^=` on the fields of a local struct.
            let mut s: Words = Words {
                first: a[2],
                second: a[3],
            };
            s.first ^= s.second;
            s.second ^= a[4];
            out[32] = s.first;
            out[33] = s.second;
            // `^` inside helpers.
            out[34] = flag(xor_flag(t, f));
            out[35] = flag(xor_flag(t, t));
            out[36] = xor_word(a[2], a[3]);
            out[37] = flag(xor_assign_flag(t, t));
            out[38] = flag(xor_assign_flag(f, t));
        }
    }
}

/// Every buffer has one element per invocation, so both are 64 long; the kernel reads and writes the first few.
const COUNT: usize = 64;

fn input_data() -> Vec<u32> {
    let mut input = vec![0u32; COUNT];
    input[..6].copy_from_slice(&[0, 1, 0b1100, 0b1010, u32::MAX, 5]);
    input
}

fn expected() -> Vec<u32> {
    vec![
        // ^, then ^=: (f, f), (f, t), (t, f), (t, t)
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        // &, then |
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        // &=, then |=
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        // t ^= (f ^ t) is true ^ true = false; (t ^ f) ^ (f ^ t) is true ^ true = false
        0,
        0,
        // uint: 0b1100 ^ 0b1010, twice; int: -1 ^ 5 = -6 as u32, twice
        0b0110,
        0b0110,
        0xFFFF_FFFA,
        0xFFFF_FFFA,
        // buffer elements: 0b1100 ^ 0b1010, then 0b1100 ^ u32::MAX
        0b0110,
        !0b1100u32,
        // struct fields: 0b1100 ^ 0b1010, then 0b1010 ^ u32::MAX
        0b0110,
        !0b1010u32,
        // helpers: t ^ f, t ^ t, 0b1100 ^ 0b1010, then the compound form on (t, t) and (f, t)
        1,
        0,
        0b0110,
        0,
        1,
    ]
}

#[test]
fn bool_and_integer_bitwise_operators_execute_on_gpu() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let input =
        device.create_typed_buffer("bool ops input", &input_data(), GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer(
        "bool ops out",
        &vec![0u32; COUNT],
        GpuBufferAccess::ReadWrite,
    );
    device
        .dispatch_buffers(
            &bool_ops::run::DESCRIPTOR,
            64,
            &[
                BufferBinding::read_only(&input),
                BufferBinding::read_write(&out),
            ],
        )
        .unwrap();
    let actual = device.read_typed_buffer(&out).unwrap();

    // The CPU twin: the same typed call with the device swapped (macro-emitted under `cfg(test)`).
    let cpu = CpuDevice::new();
    let twin_input = cpu.buffer(&input_data());
    let twin_out = cpu.buffer(&vec![0u32; COUNT]);
    bool_ops::run::dispatch(&cpu, COUNT as u32, &twin_input, &twin_out).unwrap();
    let twin = twin_out.read();

    assert_eq!(twin[..39], expected(), "CPU twin against the Rust literals");
    assert_eq!(actual[..39], expected(), "GPU against the Rust literals");
}
