//! Float math in the kernel language (slice m11.math), executed on the GPU.
//!
//! Every test compares the GPU with an independent host reference (`f64` arithmetic or exact integer arithmetic, never
//! the shader's shadow body) and asserts the CPU twin against the same reference. The twin is never GPU evidence.
//! Each test prints the adapter it ran on; the test code does not assert a name.
//!
//! The bounds of the transcendental methods are the WGSL accuracy table's (15.7.4.1, the f32 rows of the editor's
//! draft), and the inherited ones (`sqrt`, `tan`, `powf`) are derived from the rows the spec names, never a value
//! fitted to an adapter.
//! The inputs stay inside the domain each bound states; NaN, infinity and subnormal inputs are not fed.
include!("../../../tests/fixtures/float_math.rs");

use gust::cpu::CpuDevice;
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

const COUNT: usize = 5000;

fn device() -> HeadlessDevice {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let info = device.adapter_info();
    println!("adapter: {} backend: {:?}", info.name, info.backend);
    device
}

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

    /// A float in `[lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next() >> 8) as f32 / (1u32 << 24) as f32;
        lo + (hi - lo) * unit
    }
}

/// `COUNT` floats: the edges first, then random values in `[-span, span]`.
fn floats_with_edges(seed: u32, span: f32, edges: &[f32]) -> Vec<f32> {
    let mut stream = Stream(seed);
    let mut values = edges.to_vec();
    while values.len() < COUNT {
        values.push(stream.range(-span, span));
    }
    values
}

/// `2^e` for the exponent of `x`, the unit in the last place of a normal `f32`.
fn ulp(x: f32) -> f64 {
    let magnitude = x.abs().max(f32::MIN_POSITIVE);
    let exponent = ((magnitude.to_bits() >> 23) & 0xff) as i32 - 127;
    2f64.powi(exponent - 23)
}

fn buffer<T: gust::GpuPod>(
    device: &HeadlessDevice,
    name: &str,
    data: &[T],
) -> gust_wgpu::GpuBuffer<T> {
    device.create_typed_buffer(name, data, GpuBufferAccess::ReadOnly)
}

#[test]
fn the_exact_group_matches_the_host_on_the_gpu_and_the_twin() {
    let device = device();
    let edges = [
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.0,
        -1.0,
        2.5,
        -2.5,
        0.999_999_94,
        -0.999_999_94,
        1.000_000_1,
        8_388_608.0,
        -8_388_608.0,
        16_777_216.0,
        33_554_432.0,
        1.0e30,
        -1.0e30,
        1.0e-30,
        -1.0e-30,
        123_456.79,
        -123_456.79,
        3.0,
        -3.0,
        0.25,
        7.0,
    ];
    let xs = floats_with_edges(11, 100.0, &edges);
    // The second operand revisits the edges in a different order, and a pair of equal bounds is included.
    let mut ys = floats_with_edges(23, 50.0, &edges);
    ys.rotate_left(7);
    let mut zs = floats_with_edges(37, 50.0, &edges);
    zs.rotate_left(3);
    zs[100] = ys[100];
    let blank = float_math::Exact {
        f_abs: 0.0,
        f_min: 0.0,
        f_max: 0.0,
        f_clamp: 0.0,
        f_floor: 0.0,
        f_ceil: 0.0,
        f_trunc: 0.0,
        f_fract: 0.0,
    };
    let (x_buf, y_buf, z_buf) = (
        buffer(&device, "x", &xs),
        buffer(&device, "y", &ys),
        buffer(&device, "z", &zs),
    );
    let out = device.create_typed_buffer("out", &vec![blank; COUNT], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &float_math::exact::DESCRIPTOR,
            COUNT as u32,
            &[
                BufferBinding::read_only(&x_buf),
                BufferBinding::read_only(&y_buf),
                BufferBinding::read_only(&z_buf),
                BufferBinding::read_write(&out),
            ],
        )
        .unwrap();
    let gpu = device.read_typed_buffer(&out).unwrap();

    let cpu = CpuDevice::new();
    let (cx, cy, cz) = (cpu.buffer(&xs), cpu.buffer(&ys), cpu.buffer(&zs));
    let cout = cpu.buffer(&vec![blank; COUNT]);
    float_math::exact::dispatch(&cpu, COUNT as u32, &cx, &cy, &cz, &cout).unwrap();
    let twin = cout.read();

    for i in 0..COUNT {
        let (x, y, z) = (f64::from(xs[i]), f64::from(ys[i]), f64::from(zs[i]));
        let (lo, hi) = (y.min(z), y.max(z));
        // Independent of the methods under test: each value is an exact operation on `f64` (every `f32` is one).
        let clamp = if x < lo {
            lo
        } else if x > hi {
            hi
        } else {
            x
        };
        let expected = [
            x.abs(),
            if x < y { x } else { y },
            if x > y { x } else { y },
            clamp,
            x.floor(),
            x.ceil(),
            x.trunc(),
            x - x.trunc(),
        ];
        let names = [
            "abs", "min", "max", "clamp", "floor", "ceil", "trunc", "fract",
        ];
        for (label, rows) in [("gpu", &gpu), ("twin", &twin)] {
            let row = &rows[i];
            let actual = [
                row.f_abs,
                row.f_min,
                row.f_max,
                row.f_clamp,
                row.f_floor,
                row.f_ceil,
                row.f_trunc,
                row.f_fract,
            ];
            for (k, name) in names.iter().enumerate() {
                // `==` on values: the sign of a zero is not part of the claim (WGSL 15.7.2 lets it differ).
                assert!(
                    f64::from(actual[k]) == expected[k],
                    "{label} {name}[{i}]: x = {x}, y = {y}, z = {z}: got {}, expected {}",
                    actual[k],
                    expected[k]
                );
            }
        }
    }
}

#[test]
fn the_integer_methods_match_the_host_on_the_gpu_and_the_twin() {
    let device = device();
    let signed_edges = [
        0,
        1,
        -1,
        2,
        -2,
        i32::MAX,
        i32::MIN + 1,
        i32::MIN,
        1 << 24,
        -(1 << 24),
    ];
    let unsigned_edges = [
        0,
        1,
        2,
        u32::MAX,
        u32::MAX - 1,
        1 << 31,
        (1 << 31) - 1,
        1 << 24,
    ];
    let mut stream = Stream(0x1234_5678);
    let mut signed = |count: usize, avoid_min: bool| -> Vec<i32> {
        let mut values: Vec<i32> = signed_edges
            .iter()
            .copied()
            .filter(|v| !(avoid_min && *v == i32::MIN))
            .collect();
        while values.len() < count {
            // Small and full-range values alternate, so both the dense middle and the extremes are covered.
            let raw = stream.next() as i32;
            let value = if values.len().is_multiple_of(2) {
                raw
            } else {
                raw % 1000
            };
            if !(avoid_min && value == i32::MIN) {
                values.push(value);
            }
        }
        values
    };
    // `abs(i32::MIN)` panics in a debug twin and is unspecified on the GPU, so the first operand never is it.
    let ia = signed(COUNT, true);
    let ib = signed(COUNT, false);
    let ic = signed(COUNT, false);
    let mut unsigned = |count: usize| -> Vec<u32> {
        let mut values = unsigned_edges.to_vec();
        while values.len() < count {
            let raw = stream.next();
            values.push(if values.len().is_multiple_of(2) {
                raw
            } else {
                raw % 1000
            });
        }
        values
    };
    let ua = unsigned(COUNT);
    let ub = unsigned(COUNT);
    let uc = unsigned(COUNT);
    let blank = float_math::Ints {
        i_abs: 0,
        i_min: 0,
        i_max: 0,
        i_clamp: 0,
        u_min: 0,
        u_max: 0,
        u_clamp: 0,
    };
    let bufs = (
        buffer(&device, "ia", &ia),
        buffer(&device, "ib", &ib),
        buffer(&device, "ic", &ic),
        buffer(&device, "ua", &ua),
        buffer(&device, "ub", &ub),
        buffer(&device, "uc", &uc),
    );
    let out = device.create_typed_buffer("out", &vec![blank; COUNT], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &float_math::ints::DESCRIPTOR,
            COUNT as u32,
            &[
                BufferBinding::read_only(&bufs.0),
                BufferBinding::read_only(&bufs.1),
                BufferBinding::read_only(&bufs.2),
                BufferBinding::read_only(&bufs.3),
                BufferBinding::read_only(&bufs.4),
                BufferBinding::read_only(&bufs.5),
                BufferBinding::read_write(&out),
            ],
        )
        .unwrap();
    let gpu = device.read_typed_buffer(&out).unwrap();

    let cpu = CpuDevice::new();
    let twins = (
        cpu.buffer(&ia),
        cpu.buffer(&ib),
        cpu.buffer(&ic),
        cpu.buffer(&ua),
        cpu.buffer(&ub),
        cpu.buffer(&uc),
    );
    let cout = cpu.buffer(&vec![blank; COUNT]);
    float_math::ints::dispatch(
        &cpu,
        COUNT as u32,
        &twins.0,
        &twins.1,
        &twins.2,
        &twins.3,
        &twins.4,
        &twins.5,
        &cout,
    )
    .unwrap();
    let twin = cout.read();

    for i in 0..COUNT {
        // Independent reference: `i64` and `u64` arithmetic and comparisons.
        let (a, b, c) = (i64::from(ia[i]), i64::from(ib[i]), i64::from(ic[i]));
        let pick = |p: i64, q: i64| if p < q { p } else { q };
        let pick_max = |p: i64, q: i64| if p > q { p } else { q };
        let (lo, hi) = (pick(b, c), pick_max(b, c));
        let signed_clamp = if a < lo {
            lo
        } else if a > hi {
            hi
        } else {
            a
        };
        let (ua_, ub_, uc_) = (u64::from(ua[i]), u64::from(ub[i]), u64::from(uc[i]));
        let (ulo, uhi) = (ub_.min(uc_), ub_.max(uc_));
        let unsigned_clamp = if ua_ < ulo {
            ulo
        } else if ua_ > uhi {
            uhi
        } else {
            ua_
        };
        let expected = (
            if a < 0 { -a } else { a },
            pick(a, b),
            pick_max(a, b),
            signed_clamp,
            if ua_ < ub_ { ua_ } else { ub_ },
            if ua_ > ub_ { ua_ } else { ub_ },
            unsigned_clamp,
        );
        for (label, rows) in [("gpu", &gpu), ("twin", &twin)] {
            let row = &rows[i];
            let actual = (
                i64::from(row.i_abs),
                i64::from(row.i_min),
                i64::from(row.i_max),
                i64::from(row.i_clamp),
                u64::from(row.u_min),
                u64::from(row.u_max),
                u64::from(row.u_clamp),
            );
            assert_eq!(
                actual, expected,
                "{label}[{i}]: ia {a} ib {b} ic {c}, ua {ua_} ub {ub_} uc {uc_}"
            );
        }
    }
}

#[test]
fn the_constants_are_the_f32_constants_on_the_gpu_and_the_twin() {
    let device = device();
    let blank = float_math::Consts {
        pi: 0.0,
        tau: 0.0,
        e: 0.0,
        user_sqrt: 0.0,
        intrinsic_sqrt: 0.0,
    };
    let out = device.create_typed_buffer("out", &vec![blank; 64], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &float_math::constants::DESCRIPTOR,
            64,
            &[BufferBinding::read_write(&out)],
        )
        .unwrap();
    let gpu = device.read_typed_buffer(&out).unwrap();
    let cpu = CpuDevice::new();
    let cout = cpu.buffer(&vec![blank; 64]);
    float_math::constants::dispatch(&cpu, 64, &cout).unwrap();
    let twin = cout.read();
    // The decimal digits of the three values, written out here and not read from `std`, so a wrong constant in the
    // emitter and in the twin cannot agree by sharing a source.
    let expected = [
        0x4049_0fdbu32, // 3.14159274101257324...
        0x40c9_0fdb,    // 6.28318548202514648...
        0x402d_f854,    // 2.71828174591064453...
    ];
    for (label, rows) in [("gpu", &gpu), ("twin", &twin)] {
        for row in rows {
            let bits = [row.pi.to_bits(), row.tau.to_bits(), row.e.to_bits()];
            assert_eq!(bits, expected, "{label} constants");
            // The helper named `sqrt` is the user's (9 + 100); the method call is the intrinsic (sqrt of 16).
            assert_eq!(row.user_sqrt, 109.0, "{label} user helper `sqrt`");
            assert_eq!(row.intrinsic_sqrt, 4.0, "{label} intrinsic sqrt");
        }
    }
}

/// The error bound of a transcendental method for one input, in the units the WGSL table states.
enum Bound {
    /// Absolute error.
    Absolute(f64),
    /// A number of units in the last place of the exact result.
    Ulps(f64),
}

fn within(actual: f32, exact: f64, bound: &Bound) -> (bool, f64) {
    let error = (f64::from(actual) - exact).abs();
    match bound {
        Bound::Absolute(limit) => (error <= *limit, error),
        Bound::Ulps(limit) => {
            let measure = error / ulp(exact as f32);
            (measure <= *limit, measure)
        }
    }
}

/// The bound of `x.powf(y)` in ULP, from the rule the table gives: `pow(x, y)` is inherited from `exp2(y * log2(x))`.
/// With `L = abs(y * log2(x))`: `log2` has an absolute error of `2^-21` on `[0.5, 2]` and 3 ULP elsewhere (at most
/// `3 * 2^-23 * abs(log2(x))`); the product is correctly rounded (at most `2^-23 * L` more); so the exponent has an
/// absolute error `d` of `abs(y) * 2^-21 + 2^-23 * L` inside `[0.5, 2]` and `4 * 2^-23 * L` outside. That is a relative
/// error `ln(2) * d` of the result, at most `2^24` times as many ULPs (a ULP is at least `2^-24` of the value), and
/// `exp2` adds `3 + 2 * L` ULP.
fn powf_bound_ulps(x: f64, y: f64) -> f64 {
    let l = (y * x.log2()).abs();
    let d = if (0.5..=2.0).contains(&x) {
        y.abs() * 2f64.powi(-21) + 2f64.powi(-23) * l
    } else {
        4.0 * 2f64.powi(-23) * l
    };
    3.0 + 2.0 * l + std::f64::consts::LN_2 * d * 2f64.powi(24)
}

#[test]
fn the_transcendentals_are_within_the_wgsl_bounds_on_the_gpu_and_the_twin() {
    let device = device();
    let mut stream = Stream(0x9e37_79b9);
    // The `f32` constant `PI` is above the real pi, and the table states `sin` and `cos` on `[-pi, pi]`: stay inside.
    let pi = f32::from_bits(std::f32::consts::PI.to_bits() - 1);
    let mut rows = Vec::with_capacity(COUNT);
    for i in 0..COUNT {
        let mut row = float_math::TransIn {
            angle: stream.range(-pi, pi),
            tan_arg: stream.range(-1.2, 1.2),
            wide: stream.range(-50.0, 50.0),
            exp_arg: stream.range(-10.0, 10.0),
            exp2_arg: stream.range(-14.0, 14.0),
            pos: stream.range(0.001, 10_000.0),
            base: stream.range(0.25, 8.0),
            expo: stream.range(-3.0, 3.0),
            y: stream.range(-10.0, 10.0),
            x: stream.range(-10.0, 10.0),
        };
        // `atan2` is discontinuous across the negative x axis, so the sign of y is never near zero there; the table
        // states its 4096 ULP for abs(x) in [2^-126, 2^126], so x is never zero either.
        if row.y.abs() < 0.01 {
            row.y = if row.y < 0.0 { -0.01 } else { 0.01 };
        }
        if row.x.abs() < 0.01 {
            row.x = if row.x < 0.0 { -0.01 } else { 0.01 };
        }
        // A sweep through `[0.5, 2]`, where `ln` and `log2` have the absolute bound, in 300 even steps.
        if (5..305).contains(&i) {
            row.pos = 0.5 + 1.5 * ((i - 5) as f32 / 299.0);
        }
        // Edges inside every domain: the ends of the ranges, zero where it is allowed, and the log bound's corners.
        match i {
            0 => {
                row.angle = 0.0;
                row.tan_arg = 0.0;
                row.wide = 0.0;
                row.exp_arg = 0.0;
                row.exp2_arg = 0.0;
                row.pos = 1.0;
                row.x = 0.01;
            }
            1 => {
                row.angle = pi;
                row.tan_arg = 1.2;
                row.exp_arg = 10.0;
                row.exp2_arg = 14.0;
                row.pos = 0.5;
            }
            2 => {
                row.angle = -pi;
                row.tan_arg = -1.2;
                row.exp_arg = -10.0;
                row.exp2_arg = -14.0;
                row.pos = 2.0;
            }
            3 => {
                row.pos = 0.001;
                row.base = 0.25;
                row.expo = -3.0;
            }
            4 => {
                row.pos = 10_000.0;
                row.base = 8.0;
                row.expo = 3.0;
            }
            _ => {}
        }
        rows.push(row);
    }
    let blank = float_math::TransOut {
        t_sin: 0.0,
        t_cos: 0.0,
        t_tan: 0.0,
        t_atan: 0.0,
        t_exp: 0.0,
        t_exp2: 0.0,
        t_ln: 0.0,
        t_log2: 0.0,
        t_sqrt: 0.0,
        t_powf: 0.0,
        t_atan2: 0.0,
    };
    let input = buffer(&device, "input", &rows);
    let out = device.create_typed_buffer("out", &vec![blank; COUNT], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &float_math::transcendental::DESCRIPTOR,
            COUNT as u32,
            &[
                BufferBinding::read_only(&input),
                BufferBinding::read_write(&out),
            ],
        )
        .unwrap();
    let gpu = device.read_typed_buffer(&out).unwrap();
    let cpu = CpuDevice::new();
    let cin = cpu.buffer(&rows);
    let cout = cpu.buffer(&vec![blank; COUNT]);
    float_math::transcendental::dispatch(&cpu, COUNT as u32, &cin, &cout).unwrap();
    let twin = cout.read();

    const NAMES: [&str; 11] = [
        "sin", "cos", "tan", "atan", "exp", "exp2", "ln", "log2", "sqrt", "powf", "atan2",
    ];
    let mut worst = [[0f64; 2]; 11];
    for i in 0..COUNT {
        let v = &rows[i];
        let f = |x: f32| f64::from(x);
        // Independent references: `f64` library functions on the `f32` inputs.
        let exact = [
            f(v.angle).sin(),
            f(v.angle).cos(),
            f(v.tan_arg).tan(),
            f(v.wide).atan(),
            f(v.exp_arg).exp(),
            f(v.exp2_arg).exp2(),
            f(v.pos).ln(),
            f(v.pos).log2(),
            f(v.pos).sqrt(),
            f(v.base).powf(f(v.expo)),
            f(v.y).atan2(f(v.x)),
        ];
        let in_unit_log_range = (0.5..=2.0).contains(&v.pos);
        let log_bound = if in_unit_log_range {
            Bound::Absolute(2f64.powi(-21))
        } else {
            Bound::Ulps(3.0)
        };
        let tan_cos = f(v.tan_arg).cos().abs();
        let tan_value = exact[2].abs();
        let bounds = [
            // sin, cos: absolute error 2^-11 on [-pi, pi].
            Bound::Absolute(2f64.powi(-11)),
            Bound::Absolute(2f64.powi(-11)),
            // tan: inherited from sin(x) / cos(x), propagated through the quotient, plus 3 ULP for the division.
            Bound::Absolute(
                2f64.powi(-11) * (1.0 + tan_value) / tan_cos + 3.0 * ulp(exact[2] as f32),
            ),
            // atan: 4096 ULP.
            Bound::Ulps(4096.0),
            // exp, exp2: 3 + 2 * abs(x) ULP.
            Bound::Ulps(3.0 + 2.0 * f(v.exp_arg).abs()),
            Bound::Ulps(3.0 + 2.0 * f(v.exp2_arg).abs()),
            // log, log2: absolute 2^-21 on [0.5, 2], else 3 ULP.
            log_bound,
            if in_unit_log_range {
                Bound::Absolute(2f64.powi(-21))
            } else {
                Bound::Ulps(3.0)
            },
            // sqrt: inherited from 1.0 / inverseSqrt(x). inverseSqrt is 2 ULP of its result y = 1/sqrt(x), a relative
            // error of at most 2 * 2^-23; as ULPs of the result that is at most twice as many (a ULP is between 2^-24
            // and 2^-23 of the value), so 4; the division adds 2.5 ULP: 6.5 ULP.
            Bound::Ulps(6.5),
            // powf: inherited from exp2(y * log2(x)); see `powf_bound_ulps`.
            Bound::Ulps(powf_bound_ulps(f(v.base), f(v.expo))),
            // atan2: 4096 ULP.
            Bound::Ulps(4096.0),
        ];
        let gpu_row = &gpu[i];
        let twin_row = &twin[i];
        let gpu_values = [
            gpu_row.t_sin,
            gpu_row.t_cos,
            gpu_row.t_tan,
            gpu_row.t_atan,
            gpu_row.t_exp,
            gpu_row.t_exp2,
            gpu_row.t_ln,
            gpu_row.t_log2,
            gpu_row.t_sqrt,
            gpu_row.t_powf,
            gpu_row.t_atan2,
        ];
        let twin_values = [
            twin_row.t_sin,
            twin_row.t_cos,
            twin_row.t_tan,
            twin_row.t_atan,
            twin_row.t_exp,
            twin_row.t_exp2,
            twin_row.t_ln,
            twin_row.t_log2,
            twin_row.t_sqrt,
            twin_row.t_powf,
            twin_row.t_atan2,
        ];
        // The twin calls the `std` method unchanged, so it is that method's value.
        let host = [
            v.angle.sin(),
            v.angle.cos(),
            v.tan_arg.tan(),
            v.wide.atan(),
            v.exp_arg.exp(),
            v.exp2_arg.exp2(),
            v.pos.ln(),
            v.pos.log2(),
            v.pos.sqrt(),
            v.base.powf(v.expo),
            v.y.atan2(v.x),
        ];
        for k in 0..11 {
            assert_eq!(
                twin_values[k].to_bits(),
                host[k].to_bits(),
                "twin {}[{i}]",
                NAMES[k]
            );
            let (twin_ok, _) = within(twin_values[k], exact[k], &bounds[k]);
            assert!(twin_ok, "twin {}[{i}] outside the bound", NAMES[k]);
            let (ok, measure) = within(gpu_values[k], exact[k], &bounds[k]);
            let unit = usize::from(matches!(bounds[k], Bound::Ulps(_)));
            worst[k][unit] = worst[k][unit].max(measure);
            assert!(
                ok,
                "gpu {}[{i}]: got {}, exact {}, error {measure} (input row {i})",
                NAMES[k], gpu_values[k], exact[k]
            );
        }
    }
    // Observation, not a bound: the largest error per method, in ULP of the result or absolute where the row says so.
    for k in 0..11 {
        println!(
            "largest error {}: absolute {:e}, ULP {:e} (measured on this adapter only)",
            NAMES[k], worst[k][0], worst[k][1]
        );
    }
}

// Slice m11.math2 (more math methods for compute kernels).

/// The inputs of the float group: the edges of `round`, `signum` and the angle methods, every half-way case in
/// `[-1024, 1024)` at both signs, then random values; and for `rem_euclid` a divisor that is never zero, with the first
/// rows exact multiples of the divisor (the quotient is whole there, where a division-based remainder is at risk).
fn math2_float_inputs() -> (Vec<f32>, Vec<f32>) {
    let edges = [
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        0.499_999_97,
        -0.499_999_97,
        0.500_000_06,
        -0.500_000_06,
        0.999_999_94,
        -0.999_999_94,
        1.0e-30,
        -1.0e-30,
        4_194_303.5,
        -4_194_303.5,
        8_388_607.5,
        -8_388_607.5,
        8_388_608.0,
        -8_388_608.0,
        16_777_216.0,
        1.0e30,
        -1.0e30,
        123_456.79,
        -0.3,
        0.3,
    ];
    let mut xs = edges.to_vec();
    for k in -1024..1024 {
        xs.push(k as f32 + 0.5);
    }
    let mut stream = Stream(0x00c0_ffee);
    while xs.len() < COUNT {
        xs.push(stream.range(-100.0, 100.0));
    }
    let mut ds = Vec::with_capacity(COUNT);
    for _ in 0..COUNT {
        let magnitude = stream.range(0.01, 50.0);
        ds.push(if stream.next() & 1 == 0 {
            magnitude
        } else {
            -magnitude
        });
    }
    // Exact multiples: `x = k * y` for simple divisors, at both signs.
    let divisors = [0.5f32, 0.25, 1.0, 2.0, 3.0, 7.0, 0.1, 1.5, -3.0, -0.5];
    let mut at = edges.len() + 2048;
    for y in &divisors {
        for k in -20..20 {
            xs[at] = k as f32 * y;
            ds[at] = *y;
            at += 1;
        }
    }
    // A huge quotient, a dividend below the divisor at both signs, and a whole multiple that is not a small one.
    for (x, y) in [
        (1.0e30f32, 0.7f32),
        (-1.0e30, 7.0),
        (16_777_216.0, 3.0),
        (-16_777_216.0, 0.3),
        (123_456.79, 0.01),
        (1.0e-30, 1.0e-20),
        (-1.0e-30, 1.0e-20),
        (1.0e-30, -1.0e-20),
        (3.0e38, 1.5e-30),
    ] {
        xs[at] = x;
        ds[at] = y;
        at += 1;
    }
    (xs, ds)
}

#[test]
fn round_signum_and_the_angle_methods_are_bit_equal_to_rust_on_the_gpu_and_the_twin() {
    let device = device();
    let (xs, ds) = math2_float_inputs();
    let blank = float_math::Math2Float {
        m_round: 0.0,
        m_signum: 0.0,
        m_radians: 0.0,
        m_degrees: 0.0,
        m_rem_euclid: 0.0,
    };
    let (x_buf, d_buf) = (buffer(&device, "x", &xs), buffer(&device, "d", &ds));
    let out = device.create_typed_buffer("out", &vec![blank; COUNT], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &float_math::math2_float::DESCRIPTOR,
            COUNT as u32,
            &[
                BufferBinding::read_only(&x_buf),
                BufferBinding::read_only(&d_buf),
                BufferBinding::read_write(&out),
            ],
        )
        .unwrap();
    let gpu = device.read_typed_buffer(&out).unwrap();
    let cpu = CpuDevice::new();
    let (cx, cd) = (cpu.buffer(&xs), cpu.buffer(&ds));
    let cout = cpu.buffer(&vec![blank; COUNT]);
    float_math::math2_float::dispatch(&cpu, COUNT as u32, &cx, &cd, &cout).unwrap();
    let twin = cout.read();

    // The constants of `std` (`library/core/src/num/f32.rs`): `consts::PI / 180.0` and
    // `57.2957795130823208767981548141051703_f32`. A product of two `f32` is exact in `f64`, and the cast to `f32`
    // rounds it once, which is the correctly rounded `f32` product.
    let radians = f64::from(std::f32::consts::PI / 180.0);
    let degrees = f64::from(
        "57.2957795130823208767981548141051703"
            .parse::<f32>()
            .unwrap(),
    );
    let mut signed_zero_rows = 0;
    for i in 0..COUNT {
        let x = f64::from(xs[i]);
        // Half-way cases away from zero, from `f64` arithmetic (the nearest integer, ties away from zero).
        let round = x.round();
        let signum = if xs[i].is_sign_negative() { -1.0 } else { 1.0 };
        let expected_radians = (x * radians) as f32;
        let expected_degrees = (x * degrees) as f32;
        for (label, rows) in [("gpu", &gpu), ("twin", &twin)] {
            let row = &rows[i];
            // Bit equality, so the sign of a zero is part of the claim for these methods.
            assert_eq!(
                row.m_round.to_bits(),
                (round as f32).to_bits(),
                "{label} round[{i}]: x = {x}: got {}, expected {}",
                row.m_round,
                round
            );
            assert_eq!(
                row.m_signum.to_bits(),
                (signum as f32).to_bits(),
                "{label} signum[{i}]: x = {x}"
            );
            assert_eq!(
                row.m_radians.to_bits(),
                expected_radians.to_bits(),
                "{label} to_radians[{i}]: x = {x}"
            );
            assert_eq!(
                row.m_degrees.to_bits(),
                expected_degrees.to_bits(),
                "{label} to_degrees[{i}]: x = {x}"
            );
        }
        signed_zero_rows += usize::from(xs[i] == 0.0 && xs[i].is_sign_negative());
    }
    assert!(signed_zero_rows >= 1, "the inputs include -0.0");
}

#[test]
fn rem_euclid_on_f32_is_exact_on_the_gpu_and_the_twin() {
    let device = device();
    let (xs, ds) = math2_float_inputs();
    let blank = float_math::Math2Float {
        m_round: 0.0,
        m_signum: 0.0,
        m_radians: 0.0,
        m_degrees: 0.0,
        m_rem_euclid: 0.0,
    };
    let (x_buf, d_buf) = (buffer(&device, "x", &xs), buffer(&device, "d", &ds));
    let out = device.create_typed_buffer("out", &vec![blank; COUNT], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &float_math::math2_float::DESCRIPTOR,
            COUNT as u32,
            &[
                BufferBinding::read_only(&x_buf),
                BufferBinding::read_only(&d_buf),
                BufferBinding::read_write(&out),
            ],
        )
        .unwrap();
    let gpu = device.read_typed_buffer(&out).unwrap();
    let cpu = CpuDevice::new();
    let (cx, cd) = (cpu.buffer(&xs), cpu.buffer(&ds));
    let cout = cpu.buffer(&vec![blank; COUNT]);
    float_math::math2_float::dispatch(&cpu, COUNT as u32, &cx, &cd, &cout).unwrap();
    let twin = cout.read();
    let (mut worst, mut whole, mut negative_dividends) = (0.0f64, 0usize, 0usize);
    for i in 0..COUNT {
        let (x, y) = (f64::from(xs[i]), f64::from(ds[i]));
        // `fmod` of two `f32` values is exact in `f64`; the sum with `abs(y)` is exact in `f64` for the ratios used
        // here, and the cast to `f32` rounds it once, as Rust's `f32` addition does.
        let r = x % y;
        let expected = if r < 0.0 { r + y.abs() } else { r } as f32;
        negative_dividends += usize::from(x < 0.0);
        whole += usize::from(x % y == 0.0);
        for (label, rows) in [("gpu", &gpu), ("twin", &twin)] {
            // Value equality: the sign of a zero is not part of the claim for this method.
            assert!(
                rows[i].m_rem_euclid == expected,
                "{label} f32 rem_euclid[{i}]: x = {x}, y = {y}: got {}, expected {expected}",
                rows[i].m_rem_euclid
            );
            if label == "gpu" {
                let error = (f64::from(rows[i].m_rem_euclid) - f64::from(expected)).abs();
                worst = worst.max(error);
            }
        }
    }
    assert!(
        negative_dividends > 1000 && whole >= 100,
        "the inputs cover both signs and whole multiples"
    );
    println!(
        "rem_euclid f32: largest error observed {worst} over {COUNT} rows ({whole} whole multiples)"
    );
}

#[test]
fn integer_signum_and_rem_euclid_match_the_host_on_the_gpu_and_the_twin() {
    let device = device();
    let signed_edges = [
        0,
        1,
        -1,
        2,
        -2,
        3,
        -3,
        7,
        -7,
        i32::MAX,
        i32::MIN + 1,
        i32::MIN,
        1 << 24,
        -(1 << 24),
    ];
    let unsigned_edges = [0, 1, 2, 3, u32::MAX, u32::MAX - 1, 1 << 31, (1 << 31) - 1];
    let mut stream = Stream(0x0bad_cafe);
    let mut ia: Vec<i32> = signed_edges.to_vec();
    let mut ib: Vec<i32> = signed_edges
        .iter()
        .map(|v| if *v == 0 { 5 } else { *v })
        .collect();
    // The product of every pair of edges (a divisor of zero and `i32::MIN / -1` are never fed to the twin).
    for a in signed_edges {
        for b in signed_edges {
            if b != 0 && !(a == i32::MIN && b == -1) {
                ia.push(a);
                ib.push(b);
            }
        }
    }
    while ia.len() < COUNT {
        let raw = stream.next() as i32;
        let small = (stream.next() as i32) % 1000;
        let (a, b) = if ia.len().is_multiple_of(2) {
            (raw, small)
        } else {
            (small, (stream.next() as i32) % 100)
        };
        if b != 0 && !(a == i32::MIN && b == -1) {
            ia.push(a);
            ib.push(b);
        }
    }
    let mut ua: Vec<u32> = unsigned_edges.to_vec();
    let mut ub: Vec<u32> = unsigned_edges
        .iter()
        .map(|v| if *v == 0 { 5 } else { *v })
        .collect();
    for a in unsigned_edges {
        for b in unsigned_edges {
            if b != 0 {
                ua.push(a);
                ub.push(b);
            }
        }
    }
    while ua.len() < COUNT {
        let raw = stream.next();
        let b = if ua.len().is_multiple_of(2) {
            stream.next() % 1000
        } else {
            stream.next()
        };
        if b != 0 {
            ua.push(raw);
            ub.push(b);
        }
    }
    ia.truncate(COUNT);
    ib.truncate(COUNT);
    ua.truncate(COUNT);
    ub.truncate(COUNT);
    let blank = float_math::Math2Int {
        i_signum: 0,
        i_rem_euclid: 0,
        u_rem_euclid: 0,
    };
    let bufs = (
        buffer(&device, "ia", &ia),
        buffer(&device, "ib", &ib),
        buffer(&device, "ua", &ua),
        buffer(&device, "ub", &ub),
    );
    let out = device.create_typed_buffer("out", &vec![blank; COUNT], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &float_math::math2_int::DESCRIPTOR,
            COUNT as u32,
            &[
                BufferBinding::read_only(&bufs.0),
                BufferBinding::read_only(&bufs.1),
                BufferBinding::read_only(&bufs.2),
                BufferBinding::read_only(&bufs.3),
                BufferBinding::read_write(&out),
            ],
        )
        .unwrap();
    let gpu = device.read_typed_buffer(&out).unwrap();
    let cpu = CpuDevice::new();
    let twins = (
        cpu.buffer(&ia),
        cpu.buffer(&ib),
        cpu.buffer(&ua),
        cpu.buffer(&ub),
    );
    let cout = cpu.buffer(&vec![blank; COUNT]);
    float_math::math2_int::dispatch(
        &cpu,
        COUNT as u32,
        &twins.0,
        &twins.1,
        &twins.2,
        &twins.3,
        &cout,
    )
    .unwrap();
    let twin = cout.read();
    let mut negative_remainders = 0;
    for i in 0..COUNT {
        // `i64` and `u64` arithmetic: the Euclidean remainder is in `[0, abs(b))`.
        let (a, b) = (i64::from(ia[i]), i64::from(ib[i]));
        let expected_signum = a.signum() as i32;
        let expected_rem = a.rem_euclid(b) as i32;
        let expected_urem = (u64::from(ua[i]) % u64::from(ub[i])) as u32;
        negative_remainders += usize::from(a < 0 && expected_rem != 0);
        for (label, rows) in [("gpu", &gpu), ("twin", &twin)] {
            let row = &rows[i];
            assert_eq!(
                row.i_signum, expected_signum,
                "{label} i32 signum[{i}]: a = {a}"
            );
            assert_eq!(
                row.i_rem_euclid, expected_rem,
                "{label} i32 rem_euclid[{i}]: a = {a}, b = {b}"
            );
            assert_eq!(
                row.u_rem_euclid, expected_urem,
                "{label} u32 rem_euclid[{i}]: a = {}, b = {}",
                ua[i], ub[i]
            );
        }
    }
    assert!(
        negative_remainders > 100,
        "the inputs cover negative dividends"
    );
}

#[test]
fn asin_and_acos_are_within_the_atan2_derived_bound_on_the_gpu_and_the_twin() {
    let device = device();
    let mut stream = Stream(0x5eed_1234);
    let mut xs = vec![
        0.0f32, 0.9, -0.9, 0.5, -0.5, 1.0e-3, -1.0e-3, 0.25, -0.25, 0.75, -0.75,
    ];
    while xs.len() < COUNT {
        xs.push(stream.range(-0.9, 0.9));
    }
    let blank = float_math::Math2Trig {
        t_asin: 0.0,
        t_acos: 0.0,
    };
    let x_buf = buffer(&device, "x", &xs);
    let out = device.create_typed_buffer("out", &vec![blank; COUNT], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &float_math::math2_trig::DESCRIPTOR,
            COUNT as u32,
            &[
                BufferBinding::read_only(&x_buf),
                BufferBinding::read_write(&out),
            ],
        )
        .unwrap();
    let gpu = device.read_typed_buffer(&out).unwrap();
    let cpu = CpuDevice::new();
    let cx = cpu.buffer(&xs);
    let cout = cpu.buffer(&vec![blank; COUNT]);
    float_math::math2_trig::dispatch(&cpu, COUNT as u32, &cx, &cout).unwrap();
    let twin = cout.read();
    let bound = Bound::Ulps(ASIN_ACOS_BOUND_ULPS);
    let (mut worst_asin, mut worst_acos) = (0.0f64, 0.0f64);
    for i in 0..COUNT {
        let x = f64::from(xs[i]);
        let (asin, acos) = (x.asin(), x.acos());
        for (label, rows) in [("gpu", &gpu), ("twin", &twin)] {
            let (ok, measure) = within(rows[i].t_asin, asin, &bound);
            assert!(
                ok,
                "{label} asin[{i}]: x = {x}: got {}, exact {asin}, {measure} ULP",
                rows[i].t_asin
            );
            let (ok_cos, measure_cos) = within(rows[i].t_acos, acos, &bound);
            assert!(
                ok_cos,
                "{label} acos[{i}]: x = {x}: got {}, exact {acos}, {measure_cos} ULP",
                rows[i].t_acos
            );
            if label == "gpu" {
                worst_asin = worst_asin.max(measure);
                worst_acos = worst_acos.max(measure_cos);
            }
        }
    }
    println!(
        "asin worst gpu error {worst_asin} ULP, acos {worst_acos} ULP (bound {ASIN_ACOS_BOUND_ULPS})"
    );
}

/// The bound of `asin` and `acos` on `abs(x) <= 0.9`, in ULP of the exact result. WGSL 15.7.4.1 gives `asin` and `acos`
/// as inherited from `atan2(x, sqrt(1.0 - x * x))` and `atan2(sqrt(1.0 - x * x), x)` (quoted from memory of the spec,
/// UNVERIFIED: the table could not be read in this session). `atan2` is 4096 ULP; on `abs(x) <= 0.9` the argument
/// `sqrt(1 - x * x)` is at least 0.43 and has a relative error of a few ULP (the product, the difference, the square
/// root), which `atan2` carries into the result at a factor below 1 (at most `abs(x) / (1 - x^2)`, about 4.7 at 0.9
/// times a relative error of about `2^-22`, i.e. a handful of ULP), so the allowance of 32 on top is generous.
const ASIN_ACOS_BOUND_ULPS: f64 = 4096.0 + 32.0;
