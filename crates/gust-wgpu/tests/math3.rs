//! `f32::hypot`, `f32::mul_add` and integer `pow` in kernels (slice m11.math3), executed on the GPU.
//!
//! Every test compares the GPU with an independent host reference (`f64`, `i64` or exact integer arithmetic, never the
//! shader's shadow body) and asserts the CPU twin against the same reference. The twin is never GPU evidence. Each test
//! prints the adapter it ran on and the largest error it saw; the test code does not assert an adapter name.
//!
//! The bounds are derived from the WGSL accuracy rules (the same rows `docs/SEMANTICS.md` quotes), never fitted to an
//! adapter, and a row over its bound is a finding to report, not a bound to widen:
//!
//! - `pow` on integers is exact: the rows are those whose result fits the type (no overflow row), and the GPU and the
//!   twin must equal the host bit for bit.
//! - `hypot` is `__gust_hypot`: `m * sqrt(1.0 + q * q)` for `q = n / m`, `m` the larger magnitude, `n` the smaller.
//!   Take the relative error of each step with `u = 2^-23` (a ULP is at most `u` of the value, and at least `u / 2`):
//!   `q = n / m` is 2.5 ULP (WGSL `x / y` for `|y|` in `[2^-126, 2^126]`), so `2.5 u`; `q * q` is "correctly
//!   rounded", either neighbour (1 ULP), plus the doubled error of `q`, so `5 u + u = 6 u`, and `q * q <= 1`; `1 +
//!   q * q` is either neighbour again, at most `u` absolute on a value in `[1, 2]`, so `7 u` relative in all; `sqrt`
//!   halves the incoming relative error (`3.5 u`) and adds its own 6.5 ULP (`6.5 u`, the float table's row), `10 u`;
//!   the final product is either neighbour again, `1 u`. Total `11 u`, and since a ULP of the result is at least
//!   `u / 2` of it, that is at most `22` ULP, measured at the spacing of `max(|exact|, |actual|)`. Operands are zero or
//!   normal with `max(|x|, |y|)` in `[2^-126, 2^126]` and a finite result.
//! - `mul_add` is `fma(a, b, c)`, whose WGSL accuracy is "inherited from `x * y + z`", and the builtin's note allows an
//!   ordinary multiply followed by an ordinary addition. So the GPU result may differ from the exact one by one ULP of
//!   the (rounded) product `a * b` plus one ULP at `max(|exact|, |actual|)`. The twin is Rust's fused result: it must
//!   equal the `f64` reference rounded once, and the GPU is never asserted bit-equal to it (the count is printed).
//!
//! The inputs stay inside the domain each bound states; NaN, infinity and subnormal inputs are not fed.

mod common;

use common::{device_reported as device, ulp};

use gust::cpu::CpuDevice;
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

#[gust::gpu_module]
mod math3 {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn pow_u32(
        id: SV_DispatchThreadID,
        bases: StructuredBuffer<uint>,
        exps: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            out[i] = bases[i].pow(exps[i]);
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn pow_i32(
        id: SV_DispatchThreadID,
        bases: StructuredBuffer<int>,
        exps: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<int>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            out[i] = bases[i].pow(exps[i]);
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn hypot_rows(
        id: SV_DispatchThreadID,
        xs: StructuredBuffer<float>,
        ys: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            out[i] = xs[i].hypot(ys[i]);
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn mul_add_rows(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<float>,
        b: StructuredBuffer<float>,
        c: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            out[i] = a[i].mul_add(b[i], c[i]);
        }
    }
}

const COUNT: usize = 5000;

/// A deterministic stream of 32-bit values (xorshift32), so the inputs are the same on every adapter.
struct Stream(u32);

impl Stream {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// An integer in `[0, n)`.
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }

    /// A float in `[lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next() >> 8) as f32 / (1u32 << 24) as f32;
        lo + (hi - lo) * unit
    }

    /// A normal `f32` of a random sign whose binary exponent lies in `[lo, hi]`.
    fn normal(&mut self, lo: i32, hi: i32) -> f32 {
        let exponent = lo + self.below((hi - lo + 1) as u32) as i32;
        let mantissa = self.range(1.0, 2.0);
        let value = mantissa * 2f32.powi(exponent);
        if self.next() & 1 == 0 { value } else { -value }
    }
}

/// The spacing used by the float bounds: at the larger of the exact and the actual result.
fn spacing(exact: f64, actual: f32) -> f64 {
    ulp((exact.abs() as f32).max(actual.abs()))
}

fn buffer<T: gust::GpuPod>(
    device: &HeadlessDevice,
    name: &str,
    data: &[T],
) -> gust_wgpu::GpuBuffer<T> {
    device.create_typed_buffer(name, data, GpuBufferAccess::ReadOnly)
}

/// `b^e` in `i64`, for rows the generators keep inside the type: a base of magnitude 0 or 1 has a closed form (the
/// exponent can be 4 billion), every other base has `e <= 32`.
fn pow_reference(b: i64, e: u32) -> i64 {
    match (b, e) {
        (_, 0) => 1,
        (0, _) => 0,
        (1, _) => 1,
        (-1, e) => {
            if e % 2 == 0 {
                1
            } else {
                -1
            }
        }
        (b, e) => b.pow(e),
    }
}

/// The largest `m` with `m^e <= limit`, for `e >= 1`.
fn max_base(e: u32, limit: i64) -> i64 {
    let mut m = (limit as f64).powf(1.0 / f64::from(e)) as i64 + 2;
    while m > 0 && (m as i128).pow(e) > i128::from(limit) {
        m -= 1;
    }
    m
}

#[test]
fn integer_pow_is_exact_on_the_gpu_and_the_twin() {
    let device = device();
    // Unsigned rows: the edges first, then random bases with an exponent that keeps the result inside `u32`.
    let mut ub: Vec<u32> = vec![
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        10,
        255,
        256,
        65_535,
        65_536,
        u32::MAX,
        u32::MAX,
        2,
        3,
    ];
    let mut ue: Vec<u32> = vec![
        0,
        1,
        5,
        u32::MAX,
        0,
        4_000_000_000,
        u32::MAX,
        31,
        0,
        1,
        20,
        9,
        4,
        3,
        2,
        1,
        1,
        0,
        24,
        3,
    ];
    let mut stream = Stream(0x1234_5671);
    while ub.len() < COUNT {
        let e = stream.below(33);
        let (b, e) = if e == 0 {
            (stream.next(), 0)
        } else {
            let top = max_base(e, i64::from(u32::MAX)) as u64;
            // A bias toward the largest base that fits, where the multiplication is closest to overflow.
            let b = if stream.below(4) == 0 {
                top
            } else {
                u64::from(stream.next()) % (top + 1)
            };
            (b as u32, e)
        };
        ub.push(b);
        ue.push(e);
    }
    // Signed rows: both signs of base, `(-2)^31 = i32::MIN` among them, and the bases `-1`, `0` and `1` with huge exponents.
    let mut ib: Vec<i32> = vec![
        -2,
        2,
        -1,
        -1,
        1,
        0,
        0,
        -2,
        i32::MAX,
        i32::MIN,
        46_340,
        -46_340,
        -3,
        3,
        -1290,
        7,
        i32::MIN,
        i32::MAX,
        -1,
        1,
    ];
    let mut ie: Vec<u32> = vec![
        31,
        30,
        u32::MAX,
        4_294_967_294,
        u32::MAX,
        0,
        u32::MAX,
        0,
        1,
        1,
        2,
        2,
        19,
        19,
        3,
        11,
        0,
        0,
        0,
        0,
    ];
    while ib.len() < COUNT {
        let e = stream.below(32);
        let (b, e) = if e == 0 {
            (stream.next() as i32, 0)
        } else {
            // `i32::MAX` for an even exponent or a positive base, and for `e = 31` the base `-2` reaches `i32::MIN`.
            let top = max_base(e, i64::from(i32::MAX));
            let magnitude = if stream.below(4) == 0 {
                top
            } else {
                (u64::from(stream.next()) % (top as u64 + 1)) as i64
            };
            let negative = stream.next() & 1 == 1;
            let signed = if negative { -magnitude } else { magnitude };
            (signed as i32, e)
        };
        ib.push(b);
        ie.push(e);
    }
    assert_eq!(
        (ub.len(), ue.len(), ib.len(), ie.len()),
        (COUNT, COUNT, COUNT, COUNT)
    );
    // The host reference, and a check that no row overflows (the generators and the edges are meant never to).
    let uwant: Vec<u32> = (0..COUNT)
        .map(|i| {
            let r = pow_reference(i64::from(ub[i]), ue[i]);
            u32::try_from(r)
                .unwrap_or_else(|_| panic!("row {i}: {}^{} overflows u32", ub[i], ue[i]))
        })
        .collect();
    let iwant: Vec<i32> = (0..COUNT)
        .map(|i| {
            let r = pow_reference(i64::from(ib[i]), ie[i]);
            i32::try_from(r)
                .unwrap_or_else(|_| panic!("row {i}: {}^{} overflows i32", ib[i], ie[i]))
        })
        .collect();

    let (ub_buf, ue_buf) = (buffer(&device, "ub", &ub), buffer(&device, "ue", &ue));
    let (ib_buf, ie_buf) = (buffer(&device, "ib", &ib), buffer(&device, "ie", &ie));
    let uout = device.create_typed_buffer("uout", &vec![0u32; COUNT], GpuBufferAccess::ReadWrite);
    let iout = device.create_typed_buffer("iout", &vec![0i32; COUNT], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &math3::pow_u32::DESCRIPTOR,
            COUNT as u32,
            &[
                BufferBinding::read_only(&ub_buf),
                BufferBinding::read_only(&ue_buf),
                BufferBinding::read_write(&uout),
            ],
        )
        .unwrap();
    device
        .dispatch_buffers(
            &math3::pow_i32::DESCRIPTOR,
            COUNT as u32,
            &[
                BufferBinding::read_only(&ib_buf),
                BufferBinding::read_only(&ie_buf),
                BufferBinding::read_write(&iout),
            ],
        )
        .unwrap();
    let ugpu = device.read_typed_buffer(&uout).unwrap();
    let igpu = device.read_typed_buffer(&iout).unwrap();

    let cpu = CpuDevice::new();
    let (cub, cue) = (cpu.buffer(&ub), cpu.buffer(&ue));
    let (cib, cie) = (cpu.buffer(&ib), cpu.buffer(&ie));
    let cuout = cpu.buffer(&vec![0u32; COUNT]);
    let ciout = cpu.buffer(&vec![0i32; COUNT]);
    math3::pow_u32::dispatch(&cpu, COUNT as u32, &cub, &cue, &cuout).unwrap();
    math3::pow_i32::dispatch(&cpu, COUNT as u32, &cib, &cie, &ciout).unwrap();
    let utwin = cuout.read();
    let itwin = ciout.read();

    let (mut u_bad, mut i_bad) = (0usize, 0usize);
    for i in 0..COUNT {
        for (label, got) in [("gpu", ugpu[i]), ("twin", utwin[i])] {
            if got != uwant[i] {
                u_bad += 1;
                eprintln!(
                    "{label} u32 pow[{i}]: {}.pow({}) = {got}, expected {}",
                    ub[i], ue[i], uwant[i]
                );
            }
        }
        for (label, got) in [("gpu", igpu[i]), ("twin", itwin[i])] {
            if got != iwant[i] {
                i_bad += 1;
                eprintln!(
                    "{label} i32 pow[{i}]: {}.pow({}) = {got}, expected {}",
                    ib[i], ie[i], iwant[i]
                );
            }
        }
    }
    let big_u = uwant.iter().filter(|&&r| r > u32::MAX / 2).count();
    let min_hit = iwant.iter().filter(|&&r| r == i32::MIN).count();
    eprintln!(
        "math3 pow: rows={COUNT} u32_mismatches={u_bad} i32_mismatches={i_bad} u32_rows_above_2^31={big_u} i32_min_rows={min_hit}"
    );
    assert_eq!(u_bad, 0, "u32 pow rows differ from the host");
    assert_eq!(i_bad, 0, "i32 pow rows differ from the host");
}

/// The `hypot` operands: edges first, then pairs of the same, near and far magnitudes.
fn hypot_inputs() -> (Vec<f32>, Vec<f32>) {
    let two = |e: i32| 2f32.powi(e);
    let mut xs: Vec<f32> = vec![
        0.0,
        0.0,
        3.0,
        -3.0,
        3.0,
        1.0e30,
        1.0e30,
        -1.0e30,
        1.0e-30,
        1.0e-30,
        1.0e30,
        1.0e30,
        -1.0e30,
        two(126),
        two(125),
        two(126),
        two(-126),
        two(-126),
        7.0,
        -7.0,
        1.0e30,
        0.1,
        65_536.0,
    ];
    let mut ys: Vec<f32> = vec![
        0.0,
        5.0,
        0.0,
        4.0,
        -4.0,
        1.0e30,
        -1.0e30,
        1.0e30,
        1.0e-30,
        1.0e30,
        1.0e-30,
        1.0e29,
        2.0e30,
        0.0,
        two(125),
        two(126),
        0.0,
        two(-126),
        7.0,
        7.0,
        3.0e30,
        0.2,
        65_536.0,
    ];
    let mut stream = Stream(0x00c0_ffee);
    while xs.len() < COUNT {
        // The larger operand: any exponent from `2^-110` to `2^120`; the other is 0, the same, or smaller by up to
        // 2^-40 (and then rounded to a normal value or zero).
        let big = stream.normal(-110, 120);
        let small = match stream.below(8) {
            0 => 0.0,
            1 => big,
            2 => -big,
            _ => {
                let s = stream.range(1.0, 2.0) * 2f32.powi(-(stream.below(41) as i32));
                let s = if stream.next() & 1 == 0 { s } else { -s };
                let s = big * s;
                if s.abs() < f32::MIN_POSITIVE { 0.0 } else { s }
            }
        };
        if stream.next() & 1 == 0 {
            xs.push(big);
            ys.push(small);
        } else {
            xs.push(small);
            ys.push(big);
        }
    }
    (xs, ys)
}

#[test]
fn hypot_is_within_the_derived_bound_on_the_gpu_and_the_twin() {
    let device = device();
    let (xs, ys) = hypot_inputs();
    for i in 0..COUNT {
        let m = xs[i].abs().max(ys[i].abs());
        assert!(
            m == 0.0 || (2f32.powi(-126)..=2f32.powi(126)).contains(&m),
            "row {i} leaves the domain of the bound: {} {}",
            xs[i],
            ys[i]
        );
    }
    let (x_buf, y_buf) = (buffer(&device, "xs", &xs), buffer(&device, "ys", &ys));
    let out = device.create_typed_buffer("out", &vec![0f32; COUNT], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &math3::hypot_rows::DESCRIPTOR,
            COUNT as u32,
            &[
                BufferBinding::read_only(&x_buf),
                BufferBinding::read_only(&y_buf),
                BufferBinding::read_write(&out),
            ],
        )
        .unwrap();
    let gpu = device.read_typed_buffer(&out).unwrap();
    let cpu = CpuDevice::new();
    let (cx, cy) = (cpu.buffer(&xs), cpu.buffer(&ys));
    let cout = cpu.buffer(&vec![0f32; COUNT]);
    math3::hypot_rows::dispatch(&cpu, COUNT as u32, &cx, &cy, &cout).unwrap();
    let twin = cout.read();

    // `x * x + y * y` in `f64` has relative error 2^-52 at worst, and the `f32` operands are far from its range limits.
    const BOUND_ULPS: f64 = 22.0;
    let (mut gpu_worst, mut twin_worst, mut gpu_exact, mut twin_exact) = (0.0f64, 0.0f64, 0, 0);
    let mut over = Vec::new();
    for i in 0..COUNT {
        let (x, y) = (f64::from(xs[i]), f64::from(ys[i]));
        let exact = (x * x + y * y).sqrt();
        assert!(exact < f64::from(f32::MAX), "row {i} overflows f32");
        for (label, got, worst, hits) in [
            ("gpu", gpu[i], &mut gpu_worst, &mut gpu_exact),
            ("twin", twin[i], &mut twin_worst, &mut twin_exact),
        ] {
            let error = (f64::from(got) - exact).abs() / spacing(exact, got);
            *worst = worst.max(error);
            *hits += usize::from(got == exact as f32);
            if error > BOUND_ULPS || !got.is_finite() {
                over.push(format!(
                    "{label} hypot[{i}]: hypot({x}, {y}) = {got}, exact {exact}: {error} ULP"
                ));
            }
        }
    }
    eprintln!(
        "math3 hypot: rows={COUNT} bound_ulps={BOUND_ULPS} gpu_worst_ulps={gpu_worst:.3} twin_worst_ulps={twin_worst:.3} gpu_equal_to_rounded_exact={gpu_exact} twin_equal_to_rounded_exact={twin_exact}"
    );
    assert!(
        over.is_empty(),
        "rows over the derived bound:\n{}",
        over.join("\n")
    );
}

/// The `mul_add` operands: normal operands, a normal product, and a share of rows where `c` cancels the product.
fn mul_add_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut a: Vec<f32> = vec![
        1.0,
        1.5,
        -1.5,
        3.0,
        0.1,
        1.0e10,
        1.0e-10,
        16_777_217.0,
        0.333_333_34,
    ];
    let mut b: Vec<f32> = vec![1.0, 2.0, 2.0, 5.0, 10.0, 1.0e10, 1.0e-10, 1.0, 3.0];
    let mut c: Vec<f32> = vec![
        1.0,
        -3.0,
        3.0,
        -15.0,
        -1.0,
        -1.0e20,
        1.0,
        -16_777_216.0,
        -1.0,
    ];
    let mut stream = Stream(0x5eed_0003);
    while a.len() < COUNT {
        let x = stream.normal(-30, 30);
        let y = stream.normal(-30, 30);
        let z = match stream.below(3) {
            // Cancellation: `c` is the negated rounded product, moved by up to three ULP either way.
            0 => {
                let product = x * y;
                let step = (stream.below(7) as i32) - 3;
                let bits = (product.to_bits() as i64 + i64::from(step)) as u32;
                -f32::from_bits(bits)
            }
            // The same scale as the product.
            1 => stream.normal(-60, 60),
            // A far smaller or larger addend.
            _ => stream.normal(-90, 90),
        };
        a.push(x);
        b.push(y);
        c.push(z);
    }
    (a, b, c)
}

/// Whether `s` (the `f64` sum `p + z` of an exact `f64` product and an `f32`) is inexact and sits exactly on a tie of
/// `f32`: rounding it to `f32` then could differ from rounding the exact value once.
fn sum_is_ambiguous(p: f64, z: f64) -> bool {
    let s = p + z;
    let bb = s - p;
    let err = (p - (s - bb)) + (z - bb);
    err != 0.0 && s.to_bits() & ((1u64 << 29) - 1) == 1u64 << 28
}

#[test]
fn mul_add_is_within_the_unfused_bound_on_the_gpu_and_the_twin_is_fused() {
    let device = device();
    let (a, b, c) = mul_add_inputs();
    let (a_buf, b_buf, c_buf) = (
        buffer(&device, "a", &a),
        buffer(&device, "b", &b),
        buffer(&device, "c", &c),
    );
    let out = device.create_typed_buffer("out", &vec![0f32; COUNT], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &math3::mul_add_rows::DESCRIPTOR,
            COUNT as u32,
            &[
                BufferBinding::read_only(&a_buf),
                BufferBinding::read_only(&b_buf),
                BufferBinding::read_only(&c_buf),
                BufferBinding::read_write(&out),
            ],
        )
        .unwrap();
    let gpu = device.read_typed_buffer(&out).unwrap();
    let cpu = CpuDevice::new();
    let (ca, cb, cc) = (cpu.buffer(&a), cpu.buffer(&b), cpu.buffer(&c));
    let cout = cpu.buffer(&vec![0f32; COUNT]);
    math3::mul_add_rows::dispatch(&cpu, COUNT as u32, &ca, &cb, &cc, &cout).unwrap();
    let twin = cout.read();

    let (mut worst, mut worst_ratio, mut equal, mut ambiguous, mut cancelling) =
        (0.0f64, 0.0f64, 0usize, 0usize, 0usize);
    let mut over = Vec::new();
    for i in 0..COUNT {
        // The product of two `f32` is exact in `f64`; its sum with a third `f32` is rounded once to `f64` (relative
        // 2^-53), far inside the `f32` bounds, and a row where that could change the single `f32` rounding is skipped
        // for the exactness check of the twin (counted).
        let p = f64::from(a[i]) * f64::from(b[i]);
        let z = f64::from(c[i]);
        let exact = p + z;
        let product = (p as f32).abs().max(f32::MIN_POSITIVE);
        assert!(
            (p as f32).abs() >= f32::MIN_POSITIVE && (p as f32).is_finite(),
            "row {i}: product not normal"
        );
        if exact.abs() < f64::from(f32::MIN_POSITIVE) && exact != 0.0 {
            panic!("row {i}: result not normal: {exact}");
        }
        cancelling += usize::from(exact.abs() < p.abs() * 2f64.powi(-20));
        // The twin is Rust's fused `mul_add`: the `f64` reference rounded once.
        if sum_is_ambiguous(p, z) {
            ambiguous += 1;
        } else {
            assert_eq!(
                twin[i].to_bits(),
                (exact as f32).to_bits(),
                "twin mul_add[{i}]: {} * {} + {}",
                a[i],
                b[i],
                c[i]
            );
        }
        // The GPU: one ULP of the product plus one ULP at the larger of the exact and the actual result.
        let allowed = ulp(product) + spacing(exact, gpu[i]);
        let error = (f64::from(gpu[i]) - exact).abs();
        worst = worst.max(error / spacing(exact, gpu[i]));
        worst_ratio = worst_ratio.max(error / allowed);
        equal += usize::from(gpu[i].to_bits() == twin[i].to_bits());
        if error > allowed || !gpu[i].is_finite() {
            over.push(format!(
                "gpu mul_add[{i}]: {} * {} + {} = {}, exact {exact}: error {error}, allowed {allowed}",
                a[i], b[i], c[i], gpu[i]
            ));
        }
    }
    eprintln!(
        "math3 mul_add: rows={COUNT} cancelling_rows={cancelling} gpu_bit_equal_to_fused={equal} gpu_worst_ulps_at_result={worst:.3} gpu_worst_share_of_allowed={worst_ratio:.3} tie_rows_skipped_for_twin={ambiguous}"
    );
    assert!(
        over.is_empty(),
        "rows over the unfused bound:\n{}",
        over.join("\n")
    );
}
