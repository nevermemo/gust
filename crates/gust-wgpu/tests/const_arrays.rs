//! Module-level array constants read at dynamic indices, on the GPU and on the CPU twin (slice m12.carr; Gpu tier).
//!
//! The first module holds a 16-entry `u32` table, an 8-entry `i32` table and a 4-entry `f32` table. One kernel reads
//! each at indices that come from the element id, from buffer data and from the loop counter, inside a loop, and writes
//! sums. The second module reads a `u32` table through a helper, a method and a closure, a `bool` table as a condition,
//! the length of both tables and a repeat-form `i32` table. The expected values are computed on the host from the same
//! `pub const` items, never from a copy of the numbers, and every value is a small integer or a binary fraction, so
//! each result is exactly representable and the comparisons are exact. Every index is in range.
//!
//! The twin is a second assertion and never GPU evidence.

use gust::{Independent, cpu::CpuDevice};
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

#[gust::gpu]
mod tables {
    pub const PRIMES: [u32; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
    pub const STEPS: [i32; 8] = [-3, 5, -7, 11, -13, 17, -19, 23];
    pub const QUARTERS: [f32; 4] = [0.25, 0.5, 1.5, -2.0];

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn walk(
        id: SV_DispatchThreadID,
        picks: StructuredBuffer<u32>,
        mut sums: RWStructuredBuffer<u32>,
        mut steps: RWStructuredBuffer<i32>,
        mut reals: RWStructuredBuffer<f32>,
    ) {
        let i = id.x;
        if i < picks.len() {
            let mut a: u32 = PRIMES[(picks[i] % 16u32) as usize];
            let mut b: i32 = 0i32;
            let mut c: f32 = 0.0f32;
            for k in 0u32..8u32 {
                let p: u32 = (i + k * 3u32 + picks[i]) % 16u32;
                a += PRIMES[p as usize];
                b += STEPS[((p + k) % 8u32) as usize];
                c += QUARTERS[((i + k) % 4u32) as usize];
            }
            sums[i] = a;
            steps[i] = b;
            reals[i] = c;
        }
    }
}

#[gust::gpu]
mod lookups {
    pub const LUT: [u32; 6] = [10, 20, 30, 40, 50, 60];
    pub const BITS: [bool; 4] = [true, false, false, true];
    pub const NEG: [i32; 3] = [-1; 3];

    pub struct Acc {
        pub n: u32,
    }

    impl Acc {
        // A method of a struct reads an array constant.
        pub fn pick(&self, k: u32) -> u32 {
            LUT[((self.n + k) % 6u32) as usize]
        }
    }

    fn lookup(k: u32) -> u32 {
        LUT[(k % 6u32) as usize]
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn mixed(
        id: SV_DispatchThreadID,
        src: StructuredBuffer<u32>,
        mut packed: RWStructuredBuffer<u32>,
        mut bits: RWStructuredBuffer<u32>,
        mut negs: RWStructuredBuffer<i32>,
    ) {
        let i = id.x;
        if i < src.len() {
            let acc = Acc { n: src[i] };
            // A closure reads an array constant without capturing it.
            let add = |x: u32| -> u32 { x + LUT[(x % 3u32) as usize] };
            packed[i] = lookup(src[i]) + acc.pick(1u32) + add(src[i]) + (LUT.len() as u32);
            let mut on: u32 = 0u32;
            for k in 0u32..4u32 {
                if BITS[((src[i] + k) % 4u32) as usize] {
                    on += 1u32 << k;
                }
            }
            bits[i] = on + (BITS.len() as u32) * 100u32;
            negs[i] = NEG[(src[i] % 3u32) as usize] * (src[i] as i32);
        }
    }
}

/// The element count of every dispatch: not a multiple of the workgroup width (64), so the guard matters.
const N: usize = 70;

fn picks() -> Vec<u32> {
    (0..N as u32).map(|k| (k * 7 + 3) % 41).collect()
}

struct WalkExpected {
    sums: Vec<u32>,
    steps: Vec<i32>,
    reals: Vec<f32>,
}

fn walk_expected(picks: &[u32]) -> WalkExpected {
    let mut want = WalkExpected {
        sums: Vec::new(),
        steps: Vec::new(),
        reals: Vec::new(),
    };
    for (i, &pick) in picks.iter().enumerate() {
        let i = i as u32;
        let mut a = tables::PRIMES[(pick % 16) as usize];
        let mut b = 0i32;
        let mut c = 0.0f32;
        for k in 0..8u32 {
            let p = (i + k * 3 + pick) % 16;
            a += tables::PRIMES[p as usize];
            b += tables::STEPS[((p + k) % 8) as usize];
            c += tables::QUARTERS[((i + k) % 4) as usize];
        }
        want.sums.push(a);
        want.steps.push(b);
        want.reals.push(c);
    }
    want
}

struct MixedExpected {
    packed: Vec<u32>,
    bits: Vec<u32>,
    negs: Vec<i32>,
}

fn mixed_expected(src: &[u32]) -> MixedExpected {
    let mut want = MixedExpected {
        packed: Vec::new(),
        bits: Vec::new(),
        negs: Vec::new(),
    };
    for &x in src {
        let lookup = lookups::LUT[(x % 6) as usize];
        let pick = lookups::LUT[((x + 1) % 6) as usize];
        let add = x + lookups::LUT[(x % 3) as usize];
        want.packed
            .push(lookup + pick + add + lookups::LUT.len() as u32);
        let mut on = 0u32;
        for k in 0..4u32 {
            if lookups::BITS[((x + k) % 4) as usize] {
                on += 1 << k;
            }
        }
        want.bits.push(on + lookups::BITS.len() as u32 * 100);
        want.negs.push(lookups::NEG[(x % 3) as usize] * x as i32);
    }
    want
}

#[test]
fn the_host_expectation_reads_the_tables_it_claims_to() {
    // The tables are what the test is about: a change to one must change the expectation, not leave it behind.
    assert_eq!(tables::PRIMES.len(), 16);
    assert_eq!(tables::STEPS.len(), 8);
    assert_eq!(tables::QUARTERS.len(), 4);
    assert_eq!(lookups::NEG, [-1, -1, -1]);
    // Element 0: pick 3, i = 0. `a` starts at PRIMES[3] = 7 and adds PRIMES[(3k + 3) % 16] for k in 0..8.
    let want = walk_expected(&[3]);
    let a: u32 = 7
        + (0..8u32)
            .map(|k| tables::PRIMES[((3 * k + 3) % 16) as usize])
            .sum::<u32>();
    assert_eq!(want.sums, [a]);
    let mixed = mixed_expected(&[4]);
    // lookup(4) = 50, pick(1) = LUT[5] = 60, add(4) = 4 + LUT[1] = 24, len = 6; BITS[0..3] at 4, 5, 6, 7 mod 4.
    assert_eq!(mixed.packed, [50 + 60 + 24 + 6]);
    assert_eq!(mixed.bits, [(1 + 8) + 400]);
    assert_eq!(mixed.negs, [-4]);
}

#[test]
fn tables_read_at_dynamic_indices_in_a_loop_equal_the_host_on_the_gpu_and_the_twin() {
    let picks = picks();
    let want = walk_expected(&picks);

    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let info = device.adapter_info();
    eprintln!("adapter: {} backend: {:?}", info.name, info.backend);

    let source = device.create_typed_buffer("walk picks", &picks, GpuBufferAccess::ReadOnly);
    let sums = device.create_typed_buffer("walk sums", &[0u32; N], GpuBufferAccess::ReadWrite);
    let steps = device.create_typed_buffer("walk steps", &[0i32; N], GpuBufferAccess::ReadWrite);
    let reals = device.create_typed_buffer("walk reals", &[0f32; N], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &tables::walk::DESCRIPTOR,
            N as u32,
            &[
                BufferBinding::read_only(&source).independent_length(),
                BufferBinding::read_write(&sums).independent_length(),
                BufferBinding::read_write(&steps).independent_length(),
                BufferBinding::read_write(&reals).independent_length(),
            ],
        )
        .unwrap();
    assert_eq!(
        device.read_typed_buffer(&sums).unwrap(),
        want.sums,
        "gpu: 16-entry u32 table"
    );
    assert_eq!(
        device.read_typed_buffer(&steps).unwrap(),
        want.steps,
        "gpu: 8-entry i32 table"
    );
    assert_eq!(
        device.read_typed_buffer(&reals).unwrap(),
        want.reals,
        "gpu: 4-entry f32 table"
    );
    eprintln!("const_arrays: elements={N} u32_exact=true i32_exact=true f32_exact=true");

    // The twin: the same typed call through `CpuDevice`, asserted against the host as well.
    let cpu = CpuDevice::new();
    let twin_picks = cpu.buffer(&picks);
    let twin_sums = cpu.buffer(&[0u32; N]);
    let twin_steps = cpu.buffer(&[0i32; N]);
    let twin_reals = cpu.buffer(&[0f32; N]);
    tables::walk::dispatch(
        &cpu,
        N as u32,
        Independent(&twin_picks),
        Independent(&twin_sums),
        Independent(&twin_steps),
        Independent(&twin_reals),
    )
    .unwrap();
    assert_eq!(twin_sums.read(), want.sums, "twin: 16-entry u32 table");
    assert_eq!(twin_steps.read(), want.steps, "twin: 8-entry i32 table");
    assert_eq!(twin_reals.read(), want.reals, "twin: 4-entry f32 table");
}

#[test]
fn a_helper_a_method_a_closure_a_bool_table_and_len_read_array_constants_on_the_gpu_and_the_twin() {
    let src = picks();
    let want = mixed_expected(&src);

    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let info = device.adapter_info();
    eprintln!("adapter: {} backend: {:?}", info.name, info.backend);

    let source = device.create_typed_buffer("mixed src", &src, GpuBufferAccess::ReadOnly);
    let packed = device.create_typed_buffer("mixed packed", &[0u32; N], GpuBufferAccess::ReadWrite);
    let bits = device.create_typed_buffer("mixed bits", &[0u32; N], GpuBufferAccess::ReadWrite);
    let negs = device.create_typed_buffer("mixed negs", &[0i32; N], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &lookups::mixed::DESCRIPTOR,
            N as u32,
            &[
                BufferBinding::read_only(&source).independent_length(),
                BufferBinding::read_write(&packed).independent_length(),
                BufferBinding::read_write(&bits).independent_length(),
                BufferBinding::read_write(&negs).independent_length(),
            ],
        )
        .unwrap();
    assert_eq!(
        device.read_typed_buffer(&packed).unwrap(),
        want.packed,
        "gpu: a helper, a method, a closure and `.len()` read the table"
    );
    assert_eq!(
        device.read_typed_buffer(&bits).unwrap(),
        want.bits,
        "gpu: a bool table as a condition"
    );
    assert_eq!(
        device.read_typed_buffer(&negs).unwrap(),
        want.negs,
        "gpu: a repeat-form i32 table"
    );

    let cpu = CpuDevice::new();
    let twin_src = cpu.buffer(&src);
    let twin_packed = cpu.buffer(&[0u32; N]);
    let twin_bits = cpu.buffer(&[0u32; N]);
    let twin_negs = cpu.buffer(&[0i32; N]);
    lookups::mixed::dispatch(
        &cpu,
        N as u32,
        Independent(&twin_src),
        Independent(&twin_packed),
        Independent(&twin_bits),
        Independent(&twin_negs),
    )
    .unwrap();
    assert_eq!(twin_packed.read(), want.packed, "twin: helper and method");
    assert_eq!(twin_bits.read(), want.bits, "twin: bool table");
    assert_eq!(twin_negs.read(), want.negs, "twin: repeat-form table");
}
