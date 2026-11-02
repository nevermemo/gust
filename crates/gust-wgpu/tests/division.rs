//! `/`, `%`, `/=` and `%=` through the helpers `__gust_div` and `__gust_rem` on the GPU (slice m6.dv, D74's cure).
//!
//! slangc folds integer division and refuses a divisor that folds to zero; the macro now lowers every `/` and `%`
//! whose divisor is not a literal to an overloaded call that slangc leaves alone (`slang/div_table.rs`). The lowering
//! must not change a result, so the grid runs the same kernel over `i32` and `u32` pairs that include 0, -1, `MIN`,
//! `MAX` and `u32::MAX`, and compares every slot with a host reference. The reference is Rust's own `/` and `%` inside
//! the domain, and the ledger rows KM-01 to KM-03 of `docs/SEMANTICS.md` outside it (`x / 0 == x`, `x % 0 == 0`,
//! `MIN / -1 == MIN`, `MIN % -1 == 0`). The CPU twin must agree with the GPU on the rows inside the domain (it panics
//! outside it, KM-01 to KM-03). The vector test runs the vector and float overloads on values whose results are exact.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use gust::{
    Independent,
    cpu::CpuDevice,
    math::{IVec4, UVec4, Vec4},
};
use gust_wgpu::{BufferBinding, GpuBuffer, GpuBufferAccess, HeadlessDevice};

#[gust::gpu]
mod divide {
    // Six slots per element, for the `int` pair and the `uint` pair: `/`, `%`, `/=` on a local, `%=` on a buffer element,
    // a literal dividend, and literal divisors (which stay plain operators).
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn grid(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<int>,
        b: StructuredBuffer<int>,
        ua: StructuredBuffer<uint>,
        ub: StructuredBuffer<uint>,
        mut iout: RWStructuredBuffer<int>,
        mut uout: RWStructuredBuffer<uint>,
    ) {
        let i: uint = id.x;
        if i < a.len() {
            let x = a[i];
            let y = b[i];
            iout[i * 6] = x / y;
            iout[i * 6 + 1] = x % y;
            let mut t = x;
            t /= y;
            iout[i * 6 + 2] = t;
            iout[i * 6 + 3] = x;
            iout[i * 6 + 3] %= y;
            iout[i * 6 + 4] = 7 / y;
            iout[i * 6 + 5] = x / 3 + x % 7;
            let ux = ua[i];
            let uy = ub[i];
            uout[i * 6] = ux / uy;
            uout[i * 6 + 1] = ux % uy;
            let mut u = ux;
            u /= uy;
            uout[i * 6 + 2] = u;
            uout[i * 6 + 3] = ux;
            uout[i * 6 + 3] %= uy;
            uout[i * 6 + 4] = 7 / uy;
            uout[i * 6 + 5] = ux / 3 + ux % 7;
        }
    }

    // Integer vectors: vector by vector, vector by scalar, and `/=`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn lanes(
        id: SV_DispatchThreadID,
        ia: StructuredBuffer<IVec4>,
        ib: StructuredBuffer<IVec4>,
        ua: StructuredBuffer<UVec4>,
        ub: StructuredBuffer<UVec4>,
        mut iout: RWStructuredBuffer<IVec4>,
        mut uout: RWStructuredBuffer<UVec4>,
    ) {
        let i: uint = id.x;
        if i < ia.len() {
            let x = ia[i];
            let y = ib[i];
            iout[i * 3] = x / y;
            iout[i * 3 + 1] = x / y.x;
            let mut t = x;
            t /= y;
            iout[i * 3 + 2] = t;
            let ux = ua[i];
            let uy = ub[i];
            uout[i * 3] = ux / uy;
            uout[i * 3 + 1] = ux / uy.y;
            uout[i * 3 + 2] = ux;
            uout[i * 3 + 2] /= uy;
        }
    }

    // Floats: `Vec4` by `Vec4` and by `f32`, and `/`, `%`, `/=`, `%=` on the lanes of a `Vec4`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn floats(
        id: SV_DispatchThreadID,
        fa: StructuredBuffer<Vec4>,
        fb: StructuredBuffer<Vec4>,
        mut fout: RWStructuredBuffer<Vec4>,
    ) {
        let i: uint = id.x;
        if i < fa.len() {
            let fx = fa[i];
            let fy = fb[i];
            fout[i * 3] = fx / fy;
            fout[i * 3 + 1] = fx / fy.z;
            let mut p = fx.z;
            p /= fy.z;
            let mut q = fx.w;
            q %= fy.w;
            fout[i * 3 + 2] = Vec4::new(fx.x / fy.x, fx.y % fy.y, p, q);
            fout[i * 3 + 2].x /= fy.y;
        }
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

fn device() -> HeadlessDevice {
    HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip")
}

fn gpu_in<T: gust::GpuPod>(device: &HeadlessDevice, label: &str, data: &[T]) -> GpuBuffer<T> {
    device.create_typed_buffer(label, data, GpuBufferAccess::ReadOnly)
}

fn gpu_out<T: gust::GpuPod + Clone>(
    device: &HeadlessDevice,
    label: &str,
    length: usize,
    zero: T,
) -> GpuBuffer<T> {
    device.create_typed_buffer(label, &vec![zero; length], GpuBufferAccess::ReadWrite)
}

const I32_VALUES: [i32; 14] = [
    0,
    1,
    -1,
    2,
    -2,
    7,
    -7,
    100,
    -100,
    65_535,
    i32::MAX,
    i32::MAX - 1,
    i32::MIN,
    i32::MIN + 1,
];

const U32_VALUES: [u32; 12] = [
    0,
    1,
    2,
    3,
    7,
    100,
    65_535,
    0x8000_0000,
    0x8000_0001,
    u32::MAX - 1,
    u32::MAX,
    1_000_000_007,
];

/// `(x / y, x % y, inside the domain)`: Rust's operators inside it, KM-01 to KM-03 outside it.
fn reference_i32(x: i32, y: i32) -> (i32, i32, bool) {
    if y == 0 {
        (x, 0, false)
    } else if x == i32::MIN && y == -1 {
        (i32::MIN, 0, false)
    } else {
        (x / y, x % y, true)
    }
}

fn reference_u32(x: u32, y: u32) -> (u32, u32, bool) {
    match x.checked_div(y) {
        Some(q) => (q, x % y, true),
        None => (x, 0, false),
    }
}

/// The six slots of the kernel for one `int` pair.
fn expected_i32(x: i32, y: i32) -> [i32; 6] {
    let (quotient, remainder, _) = reference_i32(x, y);
    [
        quotient,
        remainder,
        quotient,
        remainder,
        reference_i32(7, y).0,
        x / 3 + x % 7,
    ]
}

fn expected_u32(x: u32, y: u32) -> [u32; 6] {
    let (quotient, remainder, _) = reference_u32(x, y);
    [
        quotient,
        remainder,
        quotient,
        remainder,
        reference_u32(7, y).0,
        x / 3 + x % 7,
    ]
}

/// Every pair of the value lists, then 256 pairs from a fixed generator; both lists have the same length.
fn pairs() -> (Vec<i32>, Vec<i32>, Vec<u32>, Vec<u32>) {
    let (mut a, mut b, mut ua, mut ub) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for k in 0..I32_VALUES.len() * I32_VALUES.len() {
        a.push(I32_VALUES[k / I32_VALUES.len()]);
        b.push(I32_VALUES[k % I32_VALUES.len()]);
    }
    for k in 0..U32_VALUES.len() * U32_VALUES.len() {
        ua.push(U32_VALUES[k / U32_VALUES.len()]);
        ub.push(U32_VALUES[k % U32_VALUES.len()]);
    }
    let mut rng = Lcg(7);
    for _ in 0..256 {
        a.push((rng.next() ^ (rng.next() << 1)) as i32);
        b.push((rng.next() >> (rng.next() % 31)) as i32 - 1000);
        ua.push(rng.next() ^ (rng.next() << 1));
        ub.push(rng.next() >> (rng.next() % 32));
    }
    // The lists have different lengths (196 and 144 pairs, then 256 each): cycle the shorter one.
    let n = a.len().max(ua.len());
    let cycle = |v: Vec<i32>| (0..n).map(|k| v[k % v.len()]).collect::<Vec<_>>();
    let ucycle = |v: Vec<u32>| (0..n).map(|k| v[k % v.len()]).collect::<Vec<_>>();
    (cycle(a), cycle(b), ucycle(ua), ucycle(ub))
}

#[test]
fn the_integer_grid_matches_the_host_reference_and_the_ledger_rows() {
    let gpu = device();
    let (a, b, ua, ub) = pairs();
    let n = a.len();
    assert_eq!(n, 452, "196 + 256 pairs");

    let (ga, gb, gua, gub) = (
        gpu_in(&gpu, "a", &a),
        gpu_in(&gpu, "b", &b),
        gpu_in(&gpu, "ua", &ua),
        gpu_in(&gpu, "ub", &ub),
    );
    let giout = gpu_out(&gpu, "iout", n * 6, 0_i32);
    let guout = gpu_out(&gpu, "uout", n * 6, 0_u32);
    gpu.dispatch_buffers(
        &divide::grid::DESCRIPTOR,
        n as u32,
        &[
            BufferBinding::read_only(&ga),
            BufferBinding::read_only(&gb),
            BufferBinding::read_only(&gua),
            BufferBinding::read_only(&gub),
            BufferBinding::read_write(&giout).independent_length(),
            BufferBinding::read_write(&guout).independent_length(),
        ],
    )
    .unwrap();
    let iout = gpu.read_typed_buffer(&giout).unwrap();
    let uout = gpu.read_typed_buffer(&guout).unwrap();

    let (mut inside, mut outside) = (0_usize, 0_usize);
    for k in 0..n {
        let want = expected_i32(a[k], b[k]);
        assert_eq!(
            iout[k * 6..k * 6 + 6],
            want,
            "i32 pair {k}: x = {}, y = {}",
            a[k],
            b[k]
        );
        let want = expected_u32(ua[k], ub[k]);
        assert_eq!(
            uout[k * 6..k * 6 + 6],
            want,
            "u32 pair {k}: x = {}, y = {}",
            ua[k],
            ub[k]
        );
        for domain in [reference_i32(a[k], b[k]).2, reference_u32(ua[k], ub[k]).2] {
            if domain {
                inside += 1;
            } else {
                outside += 1;
            }
        }
    }
    // The grid reaches both sides of the domain: `y == 0` for both types, and `MIN / -1`.
    assert!(outside >= 27, "{outside} rows outside the domain");
    assert!(inside > outside, "{inside} rows inside the domain");
    // KM-01 to KM-03 as the reference states them.
    assert_eq!(reference_i32(7, 0), (7, 0, false));
    assert_eq!(reference_i32(-7, 0), (-7, 0, false));
    assert_eq!(reference_i32(i32::MIN, -1), (i32::MIN, 0, false));
    assert_eq!(reference_u32(u32::MAX, 0), (u32::MAX, 0, false));

    // The CPU twin agrees with the GPU where it is defined: inside the domain, for both pairs of an element.
    let keep: Vec<usize> = (0..n)
        .filter(|&k| reference_i32(a[k], b[k]).2 && reference_u32(ua[k], ub[k]).2)
        .collect();
    let pick = |v: &[i32]| keep.iter().map(|&k| v[k]).collect::<Vec<_>>();
    let upick = |v: &[u32]| keep.iter().map(|&k| v[k]).collect::<Vec<_>>();
    let cpu = CpuDevice::new();
    let (ta, tb, tua, tub) = (
        cpu.buffer(&pick(&a)),
        cpu.buffer(&pick(&b)),
        cpu.buffer(&upick(&ua)),
        cpu.buffer(&upick(&ub)),
    );
    let tiout = cpu.buffer(&vec![0_i32; keep.len() * 6]);
    let tuout = cpu.buffer(&vec![0_u32; keep.len() * 6]);
    divide::grid::dispatch(
        &cpu,
        keep.len() as u32,
        &ta,
        &tb,
        &tua,
        &tub,
        Independent(&tiout),
        Independent(&tuout),
    )
    .unwrap();
    let (tiout, tuout) = (tiout.read(), tuout.read());
    assert!(keep.len() > 100, "{} twin rows", keep.len());
    for (row, &k) in keep.iter().enumerate() {
        assert_eq!(
            tiout[row * 6..row * 6 + 6],
            iout[k * 6..k * 6 + 6],
            "twin i32 pair {k}"
        );
        assert_eq!(
            tuout[row * 6..row * 6 + 6],
            uout[k * 6..k * 6 + 6],
            "twin u32 pair {k}"
        );
    }
}

#[test]
fn vector_and_float_division_match_the_host_reference() {
    let gpu = device();
    let n = 256;
    let mut rng = Lcg(3);
    // Nonzero divisors inside the domain, and `f32` values with exact results: dividends are small integers or
    // halves, divisors are powers of two (the quotient and the remainder are exact in `f32`).
    let mut small = |range: u32| (rng.next() % (2 * range + 1)) as i32 - range as i32;
    let nonzero = |value: i32| if value == 0 { 1 } else { value };
    let ia: Vec<IVec4> = (0..n)
        .map(|_| IVec4::new(small(1000), small(1000), small(1000), small(1000)))
        .collect();
    let ib: Vec<IVec4> = (0..n)
        .map(|_| {
            IVec4::new(
                nonzero(small(9)),
                nonzero(small(9)),
                nonzero(small(9)),
                nonzero(small(9)),
            )
        })
        .collect();
    let ua: Vec<UVec4> = ia
        .iter()
        .map(|v| {
            UVec4::new(
                v.x.unsigned_abs(),
                v.y.unsigned_abs(),
                v.z.unsigned_abs(),
                v.w.unsigned_abs(),
            )
        })
        .collect();
    let ub: Vec<UVec4> = ib
        .iter()
        .map(|v| {
            UVec4::new(
                v.x.unsigned_abs(),
                v.y.unsigned_abs(),
                v.z.unsigned_abs(),
                v.w.unsigned_abs(),
            )
        })
        .collect();
    let power = |k: i32| [0.25_f32, 0.5, 1.0, 2.0, 4.0][(k.unsigned_abs() % 5) as usize];
    let fa: Vec<Vec4> = ia
        .iter()
        .map(|v| {
            Vec4::new(
                v.x as f32 / 2.0,
                v.y as f32 / 2.0,
                v.z as f32 / 2.0,
                v.w as f32 / 2.0,
            )
        })
        .collect();
    let fb: Vec<Vec4> = ib
        .iter()
        .map(|v| Vec4::new(power(v.x), -power(v.y), power(v.z), -power(v.w)))
        .collect();

    let (g1, g2, g3, g4, g5, g6) = (
        gpu_in(&gpu, "ia", &ia),
        gpu_in(&gpu, "ib", &ib),
        gpu_in(&gpu, "ua", &ua),
        gpu_in(&gpu, "ub", &ub),
        gpu_in(&gpu, "fa", &fa),
        gpu_in(&gpu, "fb", &fb),
    );
    let giout = gpu_out(&gpu, "iout", n * 3, IVec4::default());
    let guout = gpu_out(&gpu, "uout", n * 3, UVec4::default());
    let gfout = gpu_out(&gpu, "fout", n * 3, Vec4::default());
    gpu.dispatch_buffers(
        &divide::lanes::DESCRIPTOR,
        n as u32,
        &[
            BufferBinding::read_only(&g1),
            BufferBinding::read_only(&g2),
            BufferBinding::read_only(&g3),
            BufferBinding::read_only(&g4),
            BufferBinding::read_write(&giout).independent_length(),
            BufferBinding::read_write(&guout).independent_length(),
        ],
    )
    .unwrap();
    gpu.dispatch_buffers(
        &divide::floats::DESCRIPTOR,
        n as u32,
        &[
            BufferBinding::read_only(&g5),
            BufferBinding::read_only(&g6),
            BufferBinding::read_write(&gfout).independent_length(),
        ],
    )
    .unwrap();
    let iout = gpu.read_typed_buffer(&giout).unwrap();
    let uout = gpu.read_typed_buffer(&guout).unwrap();
    let fout = gpu.read_typed_buffer(&gfout).unwrap();

    let lanes = |v: IVec4| [v.x, v.y, v.z, v.w];
    let ulanes = |v: UVec4| [v.x, v.y, v.z, v.w];
    // Compared by value: `-4.0 % 4.0` is `-0.0` for Rust and `+0.0` for WGSL's `x - y * trunc(x / y)`, an edge-case
    // difference of zeros that D39 leaves alone.
    let flanes = |v: Vec4| [v.x, v.y, v.z, v.w];
    for k in 0..n {
        let (x, y) = (lanes(ia[k]), lanes(ib[k]));
        let by_vector: [i32; 4] = std::array::from_fn(|l| x[l] / y[l]);
        let by_scalar: [i32; 4] = std::array::from_fn(|l| x[l] / y[0]);
        assert_eq!(lanes(iout[k * 3]), by_vector, "ivec / ivec {k}");
        assert_eq!(lanes(iout[k * 3 + 1]), by_scalar, "ivec / i32 {k}");
        assert_eq!(lanes(iout[k * 3 + 2]), by_vector, "ivec /= ivec {k}");

        let (x, y) = (ulanes(ua[k]), ulanes(ub[k]));
        let by_vector: [u32; 4] = std::array::from_fn(|l| x[l] / y[l]);
        let by_scalar: [u32; 4] = std::array::from_fn(|l| x[l] / y[1]);
        assert_eq!(ulanes(uout[k * 3]), by_vector, "uvec / uvec {k}");
        assert_eq!(ulanes(uout[k * 3 + 1]), by_scalar, "uvec / u32 {k}");
        assert_eq!(ulanes(uout[k * 3 + 2]), by_vector, "uvec /= uvec {k}");

        let (x, y) = (
            [fa[k].x, fa[k].y, fa[k].z, fa[k].w],
            [fb[k].x, fb[k].y, fb[k].z, fb[k].w],
        );
        let by_vector: [f32; 4] = std::array::from_fn(|l| x[l] / y[l]);
        let by_scalar: [f32; 4] = std::array::from_fn(|l| x[l] / y[2]);
        assert_eq!(flanes(fout[k * 3]), by_vector, "vec4 / vec4 {k}");
        assert_eq!(flanes(fout[k * 3 + 1]), by_scalar, "vec4 / f32 {k}");
        // The first lane is divided again in the buffer: `fout[..].x /= fy.y`, a place that is a field of an element.
        let scalars = [x[0] / y[0] / y[1], x[1] % y[1], x[2] / y[2], x[3] % y[3]];
        assert_eq!(flanes(fout[k * 3 + 2]), scalars, "f32 / % /= %= {k}");
    }

    // The twin runs the same kernel on the CPU with Rust's own operators.
    let cpu = CpuDevice::new();
    let (t1, t2, t3, t4, t5, t6) = (
        cpu.buffer(&ia),
        cpu.buffer(&ib),
        cpu.buffer(&ua),
        cpu.buffer(&ub),
        cpu.buffer(&fa),
        cpu.buffer(&fb),
    );
    let tiout = cpu.buffer(&vec![IVec4::default(); n * 3]);
    let tuout = cpu.buffer(&vec![UVec4::default(); n * 3]);
    let tfout = cpu.buffer(&vec![Vec4::default(); n * 3]);
    divide::lanes::dispatch(
        &cpu,
        n as u32,
        &t1,
        &t2,
        &t3,
        &t4,
        Independent(&tiout),
        Independent(&tuout),
    )
    .unwrap();
    divide::floats::dispatch(&cpu, n as u32, &t5, &t6, Independent(&tfout)).unwrap();
    let (tiout, tuout, tfout) = (tiout.read(), tuout.read(), tfout.read());
    for k in 0..n * 3 {
        assert_eq!(lanes(tiout[k]), lanes(iout[k]), "twin ivec {k}");
        assert_eq!(ulanes(tuout[k]), ulanes(uout[k]), "twin uvec {k}");
        assert_eq!(flanes(tfout[k]), flanes(fout[k]), "twin vec4 {k}");
    }
}
