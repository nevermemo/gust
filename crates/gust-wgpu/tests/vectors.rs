//! Vectors and matrices on the GPU (roadmap M6, slice m6.g1; Gpu tier).
//!
//! Every kernel below runs three ways: on the GPU (`HeadlessDevice`, the WGSL route, with the native reflection helper
//! checking the descriptor first, D17), as the CPU twin (the same typed call through `CpuDevice`, D22), and against an
//! independent host reference written in this file with plain `f32`/`f64`/integer arithmetic (it does not call
//! `gust::math`). Integer results, swizzles, transposes, constructors and negations are compared exactly. Float
//! results are compared with the bounds derived below, by the method of `docs/SEMANTICS.md` section 4: the GPU and the
//! twin are each checked against the exact value (the sum of the exact products, computed in `f64`) and against each
//! other (the sum of the two bounds). Observed maxima are printed with `--nocapture` as observations; they are never
//! tolerances.
//!
//! # Derived bounds
//!
//! `u = 2^-24`. One rounding of `+ - *` is at most `u |r|` on the twin (0.5 ULP) and `2u |r|` on the GPU (1 ULP); one
//! division is `u |r|` and `5u |r|` (2.5 ULP). `|r|` is the largest magnitude of the result over the input domain.
//!
//! - Component-wise kernel (`a` in [-1, 1], `b` and `k` of magnitude in [0.5, 1.5]): `a + b` and `a - b` have
//!   `|r| <= 2.5`, so the twin 2.5u and the GPU 5u; `a * b` and `a * k` have `|r| <= 1.5`, so 1.5u and 3u; `a / b` and
//!   `a / k` have `|r| <= 2`, so 2u and 10u. Negation is exact. A compound assignment (`t += y`) is the one operation.
//! - A dot product or one element of a matrix product is a sum of N products of operands of magnitude at most 1. Twin,
//!   in the written order: N products at u, then additions whose partial sums are at most 2, 3, 4 in magnitude, so
//!   u (2 + 3 + 4) for N = 4. GPU, worst grouping: 2u per product and 2u times the partial-sum magnitudes. N = 4: twin
//!   4u + 9u = 13u, GPU 8u + 18u = 26u. N = 3: twin 3u + 5u = 8u, GPU 6u + 10u = 16u. N = 2: twin 2u + 2u = 4u, GPU
//!   4u + 4u = 8u. Each asserted bound is the derived number plus 1u for second-order terms: N = 4 14u and 27u, N = 3
//!   9u and 17u, N = 2 5u and 9u. A product of two matrices is checked element by element with the same numbers (each
//!   element is a dot of one row and one column); a chain such as `(m * m) * e0` is read through a unit vector, whose
//!   products and sums are exact, so the element keeps its bound.
//! - `s * m`, `m * s`, `m *= s` and the struct-field products are single roundings with `|r| <= 1.5`; `m + n` and
//!   `m - n` have `|r| <= 2`.
//!
//! The D39 policy applies: a CPU/GPU difference outside the domain (NaN, infinity, overflow, subnormals) is not chased
//! here; the inputs stay inside the domain.

// Kernel signatures list one parameter per buffer, and the host loops index several arrays in step.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use std::collections::BTreeMap;

use gust::GpuPod;
use gust::Independent;
use gust::cpu::CpuDevice;
use gust::math::{IVec2, IVec4, Mat2, Mat4, UVec4, Vec2, Vec4};
use gust_wgpu::{BufferBinding, Error, GpuBuffer, GpuBufferAccess, HeadlessDevice};

/// The element count of every dispatch: not a multiple of the workgroup width, so the guard matters.
const N: usize = 131;

#[gust::gpu]
mod vectors {
    // A struct of vector and matrix fields. Offsets (StorageV2): color 0, xf 16, frame 80, mass 96, tag 100,
    // extra 104, size 112 (alignment 16), with no padding anywhere.
    pub struct Frame {
        pub origin: Vec2,
        pub scale: Vec2,
    }

    pub struct Body {
        pub color: Vec4,
        pub xf: Mat4,
        pub frame: Frame,
        pub mass: f32,
        pub tag: u32,
        pub extra: Vec2,
    }

    // Component-wise operators on Vec4 and Vec2, with a scalar on either side, compound forms from a fresh copy each
    // (so each slot is one rounding), a negation and swizzled operands.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn componentwise(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<Vec4>,
        b: StructuredBuffer<Vec4>,
        s: StructuredBuffer<f32>,
        mut wide: RWStructuredBuffer<Vec4>,
        mut narrow: RWStructuredBuffer<Vec2>,
    ) {
        let i: uint = id.x;
        if i < a.len() {
            let x = a[i];
            let y = b[i];
            let k = s[i];
            wide[i * 14] = x + y;
            wide[i * 14 + 1] = x - y;
            wide[i * 14 + 2] = x * y;
            wide[i * 14 + 3] = x / y;
            wide[i * 14 + 4] = x * k;
            wide[i * 14 + 5] = k * x;
            wide[i * 14 + 6] = x / k;
            wide[i * 14 + 7] = -x;
            let mut t = x;
            t += y;
            wide[i * 14 + 8] = t;
            t = x;
            t -= y;
            wide[i * 14 + 9] = t;
            t = x;
            t *= y;
            wide[i * 14 + 10] = t;
            t = x;
            t *= k;
            wide[i * 14 + 11] = t;
            t = x;
            t /= y;
            wide[i * 14 + 12] = t;
            t = x;
            t /= k;
            wide[i * 14 + 13] = t;
            let p = x.xy();
            let q = y.xy();
            narrow[i * 6] = p + q;
            narrow[i * 6 + 1] = p - q;
            narrow[i * 6 + 2] = p * q;
            narrow[i * 6 + 3] = p / q;
            narrow[i * 6 + 4] = p * k;
            narrow[i * 6 + 5] = q.yx() - p.xx();
        }
    }

    // Integer vectors: vector operands, literal operands on either side (positive, negative, and above i32::MAX for
    // the unsigned ones), compound chains, integer `dot`, constructors and swizzles, and the invocation id times a
    // literal (`UVec3`).
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn integers(
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
            iout[i * 12] = x + y;
            iout[i * 12 + 1] = x - y;
            iout[i * 12 + 2] = x * y;
            iout[i * 12 + 3] = x / y;
            iout[i * 12 + 4] = x * 3;
            iout[i * 12 + 5] = 3 * x;
            iout[i * 12 + 6] = x * -3;
            iout[i * 12 + 7] = -3 * x;
            iout[i * 12 + 8] = x / 3;
            iout[i * 12 + 9] = x / -3;
            iout[i * 12 + 10] = IVec4::new(
                x.dot(y),
                x.xy().dot(y.xy()),
                x.dot(IVec4::splat(1)),
                IVec2::new(x.w, y.x).dot(IVec2::new(2, -3)),
            );
            let mut t = x;
            t += y;
            t *= 3;
            t -= y;
            t /= y;
            iout[i * 12 + 11] = t;

            let u = ua[i];
            let v = ub[i];
            uout[i * 15] = u + v;
            uout[i * 15 + 1] = u - v;
            uout[i * 15 + 2] = u * v;
            uout[i * 15 + 3] = u / v;
            uout[i * 15 + 4] = u * 3;
            uout[i * 15 + 5] = 3 * u;
            uout[i * 15 + 6] = u / 3;
            uout[i * 15 + 7] = u / 3000000000;
            let big = UVec4::new(4000000000, 3000000000, 2147483648, 4294967295);
            uout[i * 15 + 8] = big / v;
            let small_u = u / 256;
            let small_v = v / 256;
            uout[i * 15 + 9] = UVec4::new(
                small_u.dot(small_v),
                small_u.xy().dot(small_v.xy()),
                small_u.dot(UVec4::splat(1)),
                UVec2::new(u.z, v.w).dot(UVec2::new(2, 3)),
            );
            let mut w = u;
            w += v;
            w *= 2;
            w -= v;
            w /= v;
            uout[i * 15 + 10] = w;
            let scaled = id * 3;
            uout[i * 15 + 11] =
                UVec4::new(scaled.x, scaled.y, scaled.z, UVec2::new(v.w, u.w).yx().x);
            let wide = UVec3::new(u.x, u.y, u.z).zyx();
            uout[i * 15 + 12] =
                UVec4::new(wide.x, wide.y, wide.z, UVec3::new(v.x, v.y, v.z).zyxz().w);
            uout[i * 15 + 13] = u.wzyx();
            uout[i * 15 + 14] = v.xxyy();
        }
    }

    // `m * v`, `m * m` (operators and methods), the transpose, a scalar on either side of a matrix, compound
    // scaling, matrix sum and difference, and the float `dot` at three widths.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn products(
        id: SV_DispatchThreadID,
        m: StructuredBuffer<Mat4>,
        n: StructuredBuffer<Mat4>,
        v: StructuredBuffer<Vec4>,
        w: StructuredBuffer<Vec4>,
        s: StructuredBuffer<f32>,
        mut pv: RWStructuredBuffer<Vec4>,
        mut pm: RWStructuredBuffer<Mat4>,
    ) {
        let i: uint = id.x;
        if i < m.len() {
            let a = m[i];
            let b = n[i];
            let x = v[i];
            let y = w[i];
            let k = s[i];
            pv[i * 4] = a * x;
            pv[i * 4 + 1] = a.mul_vec4(x);
            pv[i * 4 + 2] = a.transpose() * x;
            pv[i * 4 + 3] = Vec4::new(x.dot(y), x.xy().dot(y.xy()), x.xyz().dot(y.xyz()), 0.0);
            pm[i * 9] = a * b;
            pm[i * 9 + 1] = a.mul_mat4(b);
            pm[i * 9 + 2] = a.transpose();
            pm[i * 9 + 3] = k * a;
            pm[i * 9 + 4] = a * k;
            let mut c = a;
            c *= k;
            pm[i * 9 + 5] = c;
            pm[i * 9 + 6] = a + b;
            pm[i * 9 + 7] = a - b;
            pm[i * 9 + 8] = Mat4::from_cols(x, y, x, y);
        }
    }

    // Two- and three-wide matrices, the constant identities (WGSL route only, docs/SEMANTICS.md section 4), `Vec3`
    // as a local and a helper parameter, and the row and column constructors.
    fn shear(c0: Vec3, c1: Vec3, c2: Vec3) -> Mat3 {
        Mat3::from_cols(c0, c1, c2)
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn small(
        id: SV_DispatchThreadID,
        m2: StructuredBuffer<Mat2>,
        mm: StructuredBuffer<Mat2>,
        p2: StructuredBuffer<Vec2>,
        q: StructuredBuffer<Vec4>,
        mut o2: RWStructuredBuffer<Vec2>,
        mut o4: RWStructuredBuffer<Vec4>,
        mut om: RWStructuredBuffer<Mat2>,
    ) {
        let i: uint = id.x;
        if i < m2.len() {
            let a = m2[i];
            let b = mm[i];
            let p = p2[i];
            o2[i * 5] = a * p;
            o2[i * 5 + 1] = a.mul_vec2(p);
            o2[i * 5 + 2] = Mat2::IDENTITY * p;
            o2[i * 5 + 3] = a.transpose() * p;
            o2[i * 5 + 4] = Vec2::new(p.dot(p.yx()), p.dot(p));
            om[i * 4] = a * b;
            om[i * 4 + 1] = Mat2::IDENTITY.mul_mat2(a);
            om[i * 4 + 2] = a.transpose();
            om[i * 4 + 3] = Mat2::from_rows(p, p.yx());

            let r = q[i];
            let c0 = r.xyz();
            let c1 = r.zyx();
            let c2 = r.yzx();
            let m3 = shear(c0, c1, c2);
            let product = m3 * m3;
            let r0 = m3 * c0;
            o4[i * 6] = Vec4::new(r0.x, r0.y, r0.z, 0.0);
            let r1 = Mat3::IDENTITY * c1;
            o4[i * 6 + 1] = Vec4::new(r1.x, r1.y, r1.z, 0.0);
            let r2 = m3.transpose() * c2;
            o4[i * 6 + 2] = Vec4::new(r2.x, r2.y, r2.z, 0.0);
            let r3 = product * Vec3::new(1.0, 0.0, 0.0);
            o4[i * 6 + 3] = Vec4::new(r3.x, r3.y, r3.z, 0.0);
            let r4 = product * Vec3::new(0.0, 0.0, 1.0);
            o4[i * 6 + 4] = Vec4::new(r4.x, r4.y, r4.z, 0.0);
            let r5 = Mat3::from_rows(c0, c1, c2) * c0;
            o4[i * 6 + 5] = Vec4::new(r5.x, r5.y, r5.z, 0.0);
        }
    }

    // Vector and matrix fields of a struct read from a buffer, assigned through nested places, rebuilt with a struct
    // literal, and written back.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn bodies(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<Body>,
        mut moved: RWStructuredBuffer<Body>,
        mut rebuilt: RWStructuredBuffer<Body>,
    ) {
        let i: uint = id.x;
        if i < input.len() {
            let b = input[i];
            let mut r = b;
            r.color = b.xf.mul_vec4(b.color);
            r.xf = b.xf.transpose();
            r.frame.origin = b.frame.origin + b.extra;
            r.frame.scale = b.frame.scale * b.mass;
            r.extra.x = b.color.dot(b.color);
            r.mass = b.mass * 2.0;
            r.tag = b.tag + 1;
            moved[i] = r;
            let swapped = Frame {
                origin: b.extra.yx(),
                scale: Vec2::new(b.color.w, b.color.x),
            };
            rebuilt[i] = Body {
                color: Vec4::splat(b.mass),
                xf: Mat4::IDENTITY,
                frame: swapped,
                mass: b.frame.scale.x,
                tag: b.tag,
                extra: b.frame.origin,
            };
        }
    }

    // Class-2 capture cases (plan 2.5 (iii)): user locals named like the Slang names GUST emits, next to the
    // constructs that emit them. A capture would fail to compile or change a result. All float values are
    // multiples of 0.25, so every result is exact. The struct-position name `Optional` (a user `struct Optional` next
    // to an `Option<u32>` local) has no GPU case here: the collision probes of gust-macros compile it (Slangc tier).
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn captures(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<Vec4>,
        mut out: RWStructuredBuffer<Vec4>,
        mut ints: RWStructuredBuffer<IVec4>,
    ) {
        let i: uint = id.x;
        if i < input.len() {
            let p = input[i];
            let mul = p.x * 2.0;
            let transpose = p.y + 1.0;
            let dot = p.z - 1.0;
            let float2 = p.w * 4.0;
            let float3 = p.x + p.y;
            let float4 = p.z * p.w;
            let float2x2 = p.x - p.w;
            let float3x3 = p.y * 2.0;
            let float4x4 = p.w + 3.0;
            let vector = p.x * 0.5;
            let matrix = p.y - p.z;
            let m = Mat4::from_cols(
                Vec4::new(mul, 0.0, 0.0, 0.0),
                Vec4::new(0.0, transpose, 0.0, 0.0),
                Vec4::new(0.0, 0.0, dot, 0.0),
                Vec4::new(float2, float3, float4, 1.0),
            );
            out[i * 8] = m.mul_vec4(Vec4::new(1.0, 2.0, 3.0, 1.0));
            out[i * 8 + 1] = m.transpose().mul_vec4(Vec4::splat(1.0));
            out[i * 8 + 2] = Vec4::new(
                p.dot(p),
                p.xy().dot(p.zw()),
                Vec2::new(float2, float3).dot(Vec2::splat(1.0)),
                Vec3::new(mul, transpose, dot).dot(Vec3::splat(2.0)),
            );
            let turn = Mat2::from_cols(Vec2::new(vector, matrix), Vec2::new(1.0, float2x2));
            let turned = turn.mul_vec2(Vec2::new(2.0, 3.0));
            let wide = Mat3::from_cols(
                Vec3::new(float3x3, 0.0, 0.0),
                Vec3::new(0.0, float3x3, 0.0),
                Vec3::new(0.0, 0.0, float3x3),
            )
            .mul_vec3(Vec3::new(1.0, 1.0, 1.0));
            out[i * 8 + 3] = Vec4::new(turned.x, turned.y, wide.x, float4x4);
            out[i * 8 + 4] = Mat4::IDENTITY.mul_vec4(Vec4::new(mul, transpose, dot, float4));
            let flat = Mat2::IDENTITY.mul_vec2(Vec2::new(float2, float3));
            let deep = Mat3::IDENTITY.mul_vec3(Vec3::new(mul, transpose, dot));
            out[i * 8 + 5] = Vec4::new(flat.x, flat.y, deep.z, deep.x);
            out[i * 8 + 6] = (m * Vec4::new(0.0, 0.0, 0.0, 1.0)) * 2.0;
            out[i * 8 + 7] = m.mul_mat4(m).mul_vec4(Vec4::new(1.0, 0.0, 0.0, 0.0));

            let int2: int = (p.x * 4.0) as int;
            let int4: int = (p.y * 4.0) as int;
            let uint2: uint = i + 2;
            let uint3: uint = i + 3;
            let uint4: uint = i + 4;
            let int: int = int4 + 1;
            let uint: uint = uint4 + 1;
            // `as uint` emits a `uint(..)` conversion and a `bool`-typed declaration emits the type name, so the
            // locals `uint` and `bool` must be renamed in the emitted Slang or these two lines do not compile.
            let wrapped = (int2 + 8) as uint;
            let bool = (i & 1) == 0;
            let other: bool = (i & 1) == 1;
            let a = IVec2::new(int2, int4);
            let b = IVec4::new(int4, int2, a.y, a.x);
            let c = UVec2::new(uint2, uint3);
            let d = UVec3::new(uint2, uint3, uint4);
            let e = UVec4::new(uint4, uint3, uint2, i);
            let mut flag: int = 0;
            if bool {
                flag = 1;
            }
            if other {
                flag += 2;
            }
            ints[i * 3] = IVec4::new(a.x + int, a.y, b.z, (e.x + uint) as int);
            ints[i * 3 + 1] = IVec4::new((c.x + d.z) as int, (c.y + e.w) as int, flag, b.w);
            ints[i * 3 + 2] =
                IVec4::new(d.x as int, d.y as int, d.z as int, (wrapped + e.w) as int);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Host side: inputs, readback helpers, the bound checker and the independent references.
// ---------------------------------------------------------------------------------------------------------------

/// `u = 2^-24`, the relative error of one correctly rounded `f32` operation on the twin.
const U: f64 = 1.0 / 16_777_216.0;

/// Absolute bounds in units of `u`: the twin against the exact value, and the GPU against the exact value. The
/// twin-versus-GPU bound is their sum (docs/SEMANTICS.md section 4).
#[derive(Clone, Copy)]
struct Bound {
    twin: f64,
    gpu: f64,
}

const EXACT: Bound = Bound {
    twin: 0.0,
    gpu: 0.0,
};
/// `a + b` and `a - b` with `|r| <= 2.5`.
const SUM_25: Bound = Bound {
    twin: 2.5,
    gpu: 5.0,
};
/// A sum or difference with `|r| <= 2`.
const SUM_2: Bound = Bound {
    twin: 2.0,
    gpu: 4.0,
};
/// A product with `|r| <= 1.5`.
const PRODUCT_15: Bound = Bound {
    twin: 1.5,
    gpu: 3.0,
};
/// A quotient with `|r| <= 2`.
const QUOTIENT_2: Bound = Bound {
    twin: 2.0,
    gpu: 10.0,
};
/// A dot product (or product element) of 4, 3 and 2 terms of magnitude at most 1 (module documentation).
const DOT4: Bound = Bound {
    twin: 14.0,
    gpu: 27.0,
};
const DOT3: Bound = Bound {
    twin: 9.0,
    gpu: 17.0,
};
const DOT2: Bound = Bound {
    twin: 5.0,
    gpu: 9.0,
};

/// The largest observed deviations of one case, in units of `u`. Printed, never asserted.
#[derive(Default)]
struct Seen {
    gpu_vs_exact: f64,
    twin_vs_exact: f64,
    gpu_vs_twin: f64,
}

#[derive(Default)]
struct Checker {
    seen: BTreeMap<String, Seen>,
}

impl Checker {
    /// One float lane: GPU and twin within their bounds of `want`, and of each other.
    fn float(
        &mut self,
        case: &str,
        i: usize,
        lane: usize,
        gpu: f32,
        twin: f32,
        want: f64,
        bound: Bound,
    ) {
        let gpu_error = (f64::from(gpu) - want).abs();
        let twin_error = (f64::from(twin) - want).abs();
        let pair_error = (f64::from(gpu) - f64::from(twin)).abs();
        let seen = self.seen.entry(case.to_owned()).or_default();
        seen.gpu_vs_exact = seen.gpu_vs_exact.max(gpu_error / U);
        seen.twin_vs_exact = seen.twin_vs_exact.max(twin_error / U);
        seen.gpu_vs_twin = seen.gpu_vs_twin.max(pair_error / U);
        assert!(
            gpu_error <= bound.gpu * U,
            "{case}[{i}].{lane}: GPU {gpu} against exact {want}: {} u, bound {} u",
            gpu_error / U,
            bound.gpu
        );
        assert!(
            twin_error <= bound.twin * U,
            "{case}[{i}].{lane}: twin {twin} against exact {want}: {} u, bound {} u",
            twin_error / U,
            bound.twin
        );
        assert!(
            pair_error <= (bound.twin + bound.gpu) * U,
            "{case}[{i}].{lane}: GPU {gpu} against twin {twin}: {} u, bound {} u",
            pair_error / U,
            bound.twin + bound.gpu
        );
    }

    /// All lanes of one element (a vector or a column-major matrix).
    fn lanes<T: Lanes>(
        &mut self,
        case: &str,
        i: usize,
        gpu: T,
        twin: T,
        want: &[f64],
        bound: Bound,
    ) {
        let (gpu, twin) = (gpu.lanes(), twin.lanes());
        assert_eq!(gpu.len(), want.len(), "{case}[{i}]: lane count");
        for lane in 0..want.len() {
            self.float(case, i, lane, gpu[lane], twin[lane], want[lane], bound);
        }
    }

    fn report(&self, label: &str) {
        for (case, seen) in &self.seen {
            eprintln!(
                "{label} {case}: observed gpu-vs-exact {:.2} u, twin-vs-exact {:.2} u, gpu-vs-twin {:.2} u",
                seen.gpu_vs_exact, seen.twin_vs_exact, seen.gpu_vs_twin
            );
        }
    }
}

/// The `f32` lanes of a vector or matrix read back from a buffer, matrices column-major.
trait Lanes: Copy {
    fn lanes(self) -> Vec<f32>;
}

impl Lanes for Vec2 {
    fn lanes(self) -> Vec<f32> {
        vec![self.x, self.y]
    }
}

impl Lanes for Vec4 {
    fn lanes(self) -> Vec<f32> {
        vec![self.x, self.y, self.z, self.w]
    }
}

impl Lanes for Mat2 {
    fn lanes(self) -> Vec<f32> {
        // SAFETY: `Mat2` is `repr(C)` over two `Vec2` (four `f32`, no padding); every bit pattern is a valid `f32`.
        unsafe { std::mem::transmute::<Mat2, [f32; 4]>(self) }.to_vec()
    }
}

impl Lanes for Mat4 {
    fn lanes(self) -> Vec<f32> {
        // SAFETY: `Mat4` is `repr(C)` over four `Vec4` (sixteen `f32`, no padding); every bit pattern is valid.
        unsafe { std::mem::transmute::<Mat4, [f32; 16]>(self) }.to_vec()
    }
}

fn vec4(lanes: &[f32]) -> Vec4 {
    Vec4::new(lanes[0], lanes[1], lanes[2], lanes[3])
}

fn vec2(lanes: &[f32]) -> Vec2 {
    Vec2::new(lanes[0], lanes[1])
}

/// A matrix from sixteen column-major lanes.
fn mat4(lanes: &[f32]) -> Mat4 {
    Mat4::from_cols(
        vec4(&lanes[0..4]),
        vec4(&lanes[4..8]),
        vec4(&lanes[8..12]),
        vec4(&lanes[12..16]),
    )
}

fn mat2(lanes: &[f32]) -> Mat2 {
    Mat2::from_cols(vec2(&lanes[0..2]), vec2(&lanes[2..4]))
}

/// A deterministic generator (a 64-bit LCG); the inputs are the same on every run.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    /// A value in `[-1, 1)` on a 2^-23 grid.
    fn unit(&mut self) -> f32 {
        ((self.next() >> 7) as f32) / 8_388_608.0 - 1.0
    }

    /// A value of magnitude in `[lo, hi]`, either sign.
    fn magnitude(&mut self, lo: f32, hi: f32) -> f32 {
        let m = lo + (hi - lo) * (((self.next() >> 7) as f32) / 16_777_216.0);
        if self.next() & 1 == 1 { -m } else { m }
    }

    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + self.next() % (hi - lo + 1)
    }

    fn units(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.unit()).collect()
    }
}

fn gpu_in<T: GpuPod>(device: &HeadlessDevice, label: &str, data: &[T]) -> GpuBuffer<T> {
    device.create_typed_buffer(label, data, GpuBufferAccess::ReadOnly)
}

fn gpu_out<T: GpuPod + Clone>(
    device: &HeadlessDevice,
    label: &str,
    length: usize,
    zero: T,
) -> GpuBuffer<T> {
    device.create_typed_buffer(label, &vec![zero; length], GpuBufferAccess::ReadWrite)
}

fn device() -> HeadlessDevice {
    HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip")
}

/// Independent host references (column-major, `f64`; no `gust::math`).
mod host {
    /// `m * v` for an `n x n` column-major `m`: `result[r] = sum over c of m[c * n + r] * v[c]`.
    pub fn mat_vec(m: &[f32], v: &[f32], n: usize) -> Vec<f64> {
        (0..n)
            .map(|r| {
                (0..n)
                    .map(|c| f64::from(m[c * n + r]) * f64::from(v[c]))
                    .sum()
            })
            .collect()
    }

    /// `a * b` for `n x n` column-major matrices: element `(r, c)` is the row `r` of `a` times the column `c` of `b`.
    pub fn mat_mat(a: &[f32], b: &[f32], n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n * n];
        for c in 0..n {
            for r in 0..n {
                out[c * n + r] = (0..n)
                    .map(|k| f64::from(a[k * n + r]) * f64::from(b[c * n + k]))
                    .sum();
            }
        }
        out
    }

    pub fn transpose(m: &[f32], n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n * n];
        for c in 0..n {
            for r in 0..n {
                out[r * n + c] = f64::from(m[c * n + r]);
            }
        }
        out
    }

    pub fn dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum()
    }

    pub fn widen(v: &[f32]) -> Vec<f64> {
        v.iter().map(|x| f64::from(*x)).collect()
    }
}

/// Run `body` on the GPU and the twin, with the shared inputs, and compare. A helper to keep each test to its inputs
/// and references; the closure receives the device pair.
struct Rig {
    gpu: HeadlessDevice,
    cpu: CpuDevice,
}

fn rig() -> Rig {
    let gpu = device();
    eprintln!("vectors adapter: {:?}", gpu.adapter_info());
    Rig {
        gpu,
        cpu: CpuDevice::new(),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------------------------------------------

fn f(x: f32) -> f64 {
    f64::from(x)
}

#[test]
fn component_wise_operators_execute_on_gpu() {
    let Rig { gpu, cpu } = rig();
    let mut rng = Lcg(1);
    let a: Vec<Vec4> = (0..N)
        .map(|_| Vec4::new(rng.unit(), rng.unit(), rng.unit(), rng.unit()))
        .collect();
    let b: Vec<Vec4> = (0..N)
        .map(|_| {
            Vec4::new(
                rng.magnitude(0.5, 1.5),
                rng.magnitude(0.5, 1.5),
                rng.magnitude(0.5, 1.5),
                rng.magnitude(0.5, 1.5),
            )
        })
        .collect();
    let s: Vec<f32> = (0..N).map(|_| rng.magnitude(0.5, 1.5)).collect();

    let (ga, gb, gs) = (
        gpu_in(&gpu, "a", &a),
        gpu_in(&gpu, "b", &b),
        gpu_in(&gpu, "s", &s),
    );
    let gwide = gpu_out(&gpu, "wide", N * 14, Vec4::default());
    let gnarrow = gpu_out(&gpu, "narrow", N * 6, Vec2::default());
    gpu.dispatch_buffers(
        &vectors::componentwise::DESCRIPTOR,
        N as u32,
        &[
            BufferBinding::read_only(&ga),
            BufferBinding::read_only(&gb),
            BufferBinding::read_only(&gs),
            BufferBinding::read_write(&gwide).independent_length(),
            BufferBinding::read_write(&gnarrow).independent_length(),
        ],
    )
    .unwrap();
    let gwide = gpu.read_typed_buffer(&gwide).unwrap();
    let gnarrow = gpu.read_typed_buffer(&gnarrow).unwrap();

    let (ta, tb, ts) = (cpu.buffer(&a), cpu.buffer(&b), cpu.buffer(&s));
    let twide = cpu.buffer(&vec![Vec4::default(); N * 14]);
    let tnarrow = cpu.buffer(&vec![Vec2::default(); N * 6]);
    vectors::componentwise::dispatch(
        &cpu,
        N as u32,
        &ta,
        &tb,
        &ts,
        Independent(&twide),
        Independent(&tnarrow),
    )
    .unwrap();
    let (twide, tnarrow) = (twide.read(), tnarrow.read());

    type Op = fn(f64, f64, f64) -> f64;
    let slots: [(&str, Bound, Op); 14] = [
        ("add", SUM_25, |x, y, _| x + y),
        ("sub", SUM_25, |x, y, _| x - y),
        ("mul", PRODUCT_15, |x, y, _| x * y),
        ("div", QUOTIENT_2, |x, y, _| x / y),
        ("mul_scalar", PRODUCT_15, |x, _, k| x * k),
        ("scalar_mul", PRODUCT_15, |x, _, k| k * x),
        ("div_scalar", QUOTIENT_2, |x, _, k| x / k),
        ("neg", EXACT, |x, _, _| -x),
        ("add_assign", SUM_25, |x, y, _| x + y),
        ("sub_assign", SUM_25, |x, y, _| x - y),
        ("mul_assign", PRODUCT_15, |x, y, _| x * y),
        ("mul_assign_scalar", PRODUCT_15, |x, _, k| x * k),
        ("div_assign", QUOTIENT_2, |x, y, _| x / y),
        ("div_assign_scalar", QUOTIENT_2, |x, _, k| x / k),
    ];
    let mut check = Checker::default();
    for i in 0..N {
        let (x, y, k) = (a[i].lanes(), b[i].lanes(), f(s[i]));
        for (slot, (name, bound, op)) in slots.iter().enumerate() {
            let want: Vec<f64> = (0..4).map(|l| op(f(x[l]), f(y[l]), k)).collect();
            let at = i * 14 + slot;
            let case = format!("wide.{name}");
            check.lanes(&case, i, gwide[at], twide[at], &want, *bound);
        }
        // Vec2: the first two lanes, and a swizzled difference (`q.yx() - p.xx()`).
        let (p, q) = ([f(x[0]), f(x[1])], [f(y[0]), f(y[1])]);
        let narrow: [(&str, Bound, [f64; 2]); 6] = [
            ("add", SUM_25, [p[0] + q[0], p[1] + q[1]]),
            ("sub", SUM_25, [p[0] - q[0], p[1] - q[1]]),
            ("mul", PRODUCT_15, [p[0] * q[0], p[1] * q[1]]),
            ("div", QUOTIENT_2, [p[0] / q[0], p[1] / q[1]]),
            ("mul_scalar", PRODUCT_15, [p[0] * k, p[1] * k]),
            ("swizzled_sub", SUM_25, [q[1] - p[0], q[0] - p[0]]),
        ];
        for (slot, (name, bound, want)) in narrow.iter().enumerate() {
            let at = i * 6 + slot;
            let case = format!("narrow.{name}");
            check.lanes(&case, i, gnarrow[at], tnarrow[at], want, *bound);
        }
    }
    check.report("component_wise");
}

/// An exact integer comparison of the GPU, the twin and the host values for one element.
fn exact_ints(case: &str, i: usize, gpu: [i64; 4], twin: [i64; 4], want: [i64; 4]) {
    assert_eq!(gpu, want, "{case}[{i}]: GPU against the host reference");
    assert_eq!(twin, want, "{case}[{i}]: twin against the host reference");
}

fn ivec4_lanes(v: IVec4) -> [i64; 4] {
    [v.x, v.y, v.z, v.w].map(i64::from)
}

fn uvec4_lanes(v: UVec4) -> [i64; 4] {
    [v.x, v.y, v.z, v.w].map(i64::from)
}

#[test]
fn integer_vectors_with_literal_operands_execute_on_gpu() {
    let Rig { gpu, cpu } = rig();
    let mut rng = Lcg(2);
    let signed = |rng: &mut Lcg, lo: u32| -> i32 {
        let m = rng.range(lo, 1000) as i32;
        if rng.next() & 1 == 1 { -m } else { m }
    };
    let ia: Vec<IVec4> = (0..N)
        .map(|_| {
            IVec4::new(
                signed(&mut rng, 0),
                signed(&mut rng, 0),
                signed(&mut rng, 0),
                signed(&mut rng, 0),
            )
        })
        .collect();
    let ib: Vec<IVec4> = (0..N)
        .map(|_| {
            IVec4::new(
                signed(&mut rng, 1),
                signed(&mut rng, 1),
                signed(&mut rng, 1),
                signed(&mut rng, 1),
            )
        })
        .collect();
    let ua: Vec<UVec4> = (0..N)
        .map(|_| {
            UVec4::new(
                rng.range(1, 60_000),
                rng.range(1, 60_000),
                rng.range(1, 60_000),
                rng.range(1, 60_000),
            )
        })
        .collect();
    let ub: Vec<UVec4> = ua
        .iter()
        .map(|u| {
            UVec4::new(
                rng.range(1, u.x),
                rng.range(1, u.y),
                rng.range(1, u.z),
                rng.range(1, u.w),
            )
        })
        .collect();

    let (gia, gib) = (gpu_in(&gpu, "ia", &ia), gpu_in(&gpu, "ib", &ib));
    let (gua, gub) = (gpu_in(&gpu, "ua", &ua), gpu_in(&gpu, "ub", &ub));
    let giout = gpu_out(&gpu, "iout", N * 12, IVec4::default());
    let guout = gpu_out(&gpu, "uout", N * 15, UVec4::default());
    gpu.dispatch_buffers(
        &vectors::integers::DESCRIPTOR,
        N as u32,
        &[
            BufferBinding::read_only(&gia),
            BufferBinding::read_only(&gib),
            BufferBinding::read_only(&gua),
            BufferBinding::read_only(&gub),
            BufferBinding::read_write(&giout).independent_length(),
            BufferBinding::read_write(&guout).independent_length(),
        ],
    )
    .unwrap();
    let giout = gpu.read_typed_buffer(&giout).unwrap();
    let guout = gpu.read_typed_buffer(&guout).unwrap();

    let (tia, tib) = (cpu.buffer(&ia), cpu.buffer(&ib));
    let (tua, tub) = (cpu.buffer(&ua), cpu.buffer(&ub));
    let tiout = cpu.buffer(&vec![IVec4::default(); N * 12]);
    let tuout = cpu.buffer(&vec![UVec4::default(); N * 15]);
    vectors::integers::dispatch(
        &cpu,
        N as u32,
        &tia,
        &tib,
        &tua,
        &tub,
        Independent(&tiout),
        Independent(&tuout),
    )
    .unwrap();
    let (tiout, tuout) = (tiout.read(), tuout.read());

    for i in 0..N {
        let x = ivec4_lanes(ia[i]);
        let y = ivec4_lanes(ib[i]);
        let each =
            |op: &dyn Fn(i64, i64) -> i64| -> [i64; 4] { [0, 1, 2, 3].map(|l| op(x[l], y[l])) };
        let by = |op: &dyn Fn(i64) -> i64| -> [i64; 4] { x.map(op) };
        let dot4: i64 = (0..4).map(|l| x[l] * y[l]).sum();
        let compound = [0, 1, 2, 3].map(|l| ((x[l] + y[l]) * 3 - y[l]) / y[l]);
        let expected: [(&str, [i64; 4]); 12] = [
            ("add", each(&|p, q| p + q)),
            ("sub", each(&|p, q| p - q)),
            ("mul", each(&|p, q| p * q)),
            ("div", each(&|p, q| p / q)),
            ("mul_3", by(&|p| p * 3)),
            ("3_mul", by(&|p| 3 * p)),
            ("mul_neg3", by(&|p| p * -3)),
            ("neg3_mul", by(&|p| -3 * p)),
            ("div_3", by(&|p| p / 3)),
            ("div_neg3", by(&|p| p / -3)),
            (
                "dots",
                [
                    dot4,
                    x[0] * y[0] + x[1] * y[1],
                    x.iter().sum(),
                    x[3] * 2 + y[0] * -3,
                ],
            ),
            ("compound", compound),
        ];
        for (slot, (name, want)) in expected.iter().enumerate() {
            let at = i * 12 + slot;
            exact_ints(
                &format!("iout.{name}"),
                i,
                ivec4_lanes(giout[at]),
                ivec4_lanes(tiout[at]),
                *want,
            );
        }

        let u = uvec4_lanes(ua[i]);
        let v = uvec4_lanes(ub[i]);
        let ueach =
            |op: &dyn Fn(i64, i64) -> i64| -> [i64; 4] { [0, 1, 2, 3].map(|l| op(u[l], v[l])) };
        let uby = |op: &dyn Fn(i64) -> i64| -> [i64; 4] { u.map(op) };
        let big = [
            4_000_000_000_i64,
            3_000_000_000,
            2_147_483_648,
            4_294_967_295,
        ];
        let (su, sv) = (u.map(|p| p / 256), v.map(|p| p / 256));
        let udot4: i64 = (0..4).map(|l| su[l] * sv[l]).sum();
        let ucompound = [0, 1, 2, 3].map(|l| ((u[l] + v[l]) * 2 - v[l]) / v[l]);
        let expected: [(&str, [i64; 4]); 15] = [
            ("add", ueach(&|p, q| p + q)),
            ("sub", ueach(&|p, q| p - q)),
            ("mul", ueach(&|p, q| p * q)),
            ("div", ueach(&|p, q| p / q)),
            ("mul_3", uby(&|p| p * 3)),
            ("3_mul", uby(&|p| 3 * p)),
            ("div_3", uby(&|p| p / 3)),
            ("div_3e9", uby(&|p| p / 3_000_000_000)),
            ("big_div", [0, 1, 2, 3].map(|l| big[l] / v[l])),
            (
                "dots",
                [
                    udot4,
                    su[0] * sv[0] + su[1] * sv[1],
                    su.iter().sum(),
                    u[2] * 2 + v[3] * 3,
                ],
            ),
            ("compound", ucompound),
            // `id * 3` (a `UVec3`; the y and z lanes of the invocation id are 0) and a `yx` swizzle of a `UVec2`.
            ("id_times_3_and_yx", [i as i64 * 3, 0, 0, u[3]]),
            // `zyx` of a `UVec3` (all three lanes) and the fourth lane of a four-wide `zyxz` of a `UVec3`.
            ("zyx", [u[2], u[1], u[0], v[2]]),
            // The four-wide swizzles `wzyx` and `xxyy`, every lane.
            ("wzyx", [u[3], u[2], u[1], u[0]]),
            ("xxyy", [v[0], v[0], v[1], v[1]]),
        ];
        for (slot, (name, want)) in expected.iter().enumerate() {
            let at = i * 15 + slot;
            exact_ints(
                &format!("uout.{name}"),
                i,
                uvec4_lanes(guout[at]),
                uvec4_lanes(tuout[at]),
                *want,
            );
        }
    }
}

#[test]
fn matrix_products_execute_on_gpu() {
    let Rig { gpu, cpu } = rig();
    let mut rng = Lcg(3);
    let m_lanes: Vec<Vec<f32>> = (0..N).map(|_| rng.units(16)).collect();
    let n_lanes: Vec<Vec<f32>> = (0..N).map(|_| rng.units(16)).collect();
    let m: Vec<Mat4> = m_lanes.iter().map(|l| mat4(l)).collect();
    let n: Vec<Mat4> = n_lanes.iter().map(|l| mat4(l)).collect();
    let v: Vec<Vec4> = (0..N).map(|_| vec4(&rng.units(4))).collect();
    let w: Vec<Vec4> = (0..N).map(|_| vec4(&rng.units(4))).collect();
    let s: Vec<f32> = (0..N).map(|_| rng.magnitude(0.5, 1.5)).collect();

    let (gm, gn) = (gpu_in(&gpu, "m", &m), gpu_in(&gpu, "n", &n));
    let (gv, gw, gs) = (
        gpu_in(&gpu, "v", &v),
        gpu_in(&gpu, "w", &w),
        gpu_in(&gpu, "s", &s),
    );
    let gpv = gpu_out(&gpu, "pv", N * 4, Vec4::default());
    let gpm = gpu_out(&gpu, "pm", N * 9, Mat4::IDENTITY);
    gpu.dispatch_buffers(
        &vectors::products::DESCRIPTOR,
        N as u32,
        &[
            BufferBinding::read_only(&gm),
            BufferBinding::read_only(&gn),
            BufferBinding::read_only(&gv),
            BufferBinding::read_only(&gw),
            BufferBinding::read_only(&gs),
            BufferBinding::read_write(&gpv).independent_length(),
            BufferBinding::read_write(&gpm).independent_length(),
        ],
    )
    .unwrap();
    let gpv = gpu.read_typed_buffer(&gpv).unwrap();
    let gpm = gpu.read_typed_buffer(&gpm).unwrap();

    let (tm, tn) = (cpu.buffer(&m), cpu.buffer(&n));
    let (tv, tw, ts) = (cpu.buffer(&v), cpu.buffer(&w), cpu.buffer(&s));
    let tpv = cpu.buffer(&vec![Vec4::default(); N * 4]);
    let tpm = cpu.buffer(&vec![Mat4::IDENTITY; N * 9]);
    vectors::products::dispatch(
        &cpu,
        N as u32,
        &tm,
        &tn,
        &tv,
        &tw,
        &ts,
        Independent(&tpv),
        Independent(&tpm),
    )
    .unwrap();
    let (tpv, tpm) = (tpv.read(), tpm.read());

    let mut check = Checker::default();
    for i in 0..N {
        let (a, b) = (&m_lanes[i], &n_lanes[i]);
        let (x, y) = (v[i].lanes(), w[i].lanes());
        let k = f(s[i]);
        // m * v, mul_vec4 and the transposed product. The index order is the point: column-major, so the element
        // `(r, c)` of `m` is `m[c * 4 + r]`.
        let product = host::mat_vec(a, &x, 4);
        let (gv0, tv0) = (gpv[i * 4], tpv[i * 4]);
        check.lanes("products.m_times_v", i, gv0, tv0, &product, DOT4);
        let (gv1, tv1) = (gpv[i * 4 + 1], tpv[i * 4 + 1]);
        check.lanes("products.mul_vec4", i, gv1, tv1, &product, DOT4);
        let transposed: Vec<f32> = host::transpose(a, 4).iter().map(|e| *e as f32).collect();
        let want = host::mat_vec(&transposed, &x, 4);
        let (gv2, tv2) = (gpv[i * 4 + 2], tpv[i * 4 + 2]);
        check.lanes("products.transpose_times_v", i, gv2, tv2, &want, DOT4);
        // `dot` at the three widths (lane 3 is a literal zero).
        let dots = [
            host::dot(&x, &y),
            host::dot(&x[..2], &y[..2]),
            host::dot(&x[..3], &y[..3]),
        ];
        let bounds = [DOT4, DOT2, DOT3];
        let (got, twin) = (gpv[i * 4 + 3].lanes(), tpv[i * 4 + 3].lanes());
        for lane in 0..3 {
            let (g, t) = (got[lane], twin[lane]);
            check.float("products.dot", i, lane, g, t, dots[lane], bounds[lane]);
        }
        check.float("products.dot", i, 3, got[3], twin[3], 0.0, EXACT);

        let matrix_product = host::mat_mat(a, b, 4);
        let mut slot = |name: &str, at: usize, want: &[f64], bound: Bound| {
            check.lanes(
                &format!("products.{name}"),
                i,
                gpm[i * 9 + at],
                tpm[i * 9 + at],
                want,
                bound,
            );
        };
        slot("m_times_n", 0, &matrix_product, DOT4);
        slot("mul_mat4", 1, &matrix_product, DOT4);
        slot("transpose", 2, &host::transpose(a, 4), EXACT);
        let scaled: Vec<f64> = a.iter().map(|e| f(*e) * k).collect();
        slot("scalar_times_m", 3, &scaled, PRODUCT_15);
        slot("m_times_scalar", 4, &scaled, PRODUCT_15);
        slot("m_mul_assign_scalar", 5, &scaled, PRODUCT_15);
        let sum: Vec<f64> = a.iter().zip(b).map(|(p, q)| f(*p) + f(*q)).collect();
        slot("m_plus_n", 6, &sum, SUM_2);
        let difference: Vec<f64> = a.iter().zip(b).map(|(p, q)| f(*p) - f(*q)).collect();
        slot("m_minus_n", 7, &difference, SUM_2);
        // Columns built from vectors: (v, w, v, w).
        let columns: Vec<f64> = [&x, &y, &x, &y]
            .iter()
            .flat_map(|c| host::widen(c))
            .collect();
        slot("from_cols", 8, &columns, EXACT);
    }
    check.report("matrix_products");
}

#[test]
fn small_matrices_and_identities_execute_on_gpu() {
    let Rig { gpu, cpu } = rig();
    let mut rng = Lcg(4);
    let a_lanes: Vec<Vec<f32>> = (0..N).map(|_| rng.units(4)).collect();
    let b_lanes: Vec<Vec<f32>> = (0..N).map(|_| rng.units(4)).collect();
    let m2: Vec<Mat2> = a_lanes.iter().map(|l| mat2(l)).collect();
    let mm: Vec<Mat2> = b_lanes.iter().map(|l| mat2(l)).collect();
    let p2: Vec<Vec2> = (0..N).map(|_| vec2(&rng.units(2))).collect();
    let q: Vec<Vec4> = (0..N).map(|_| vec4(&rng.units(4))).collect();

    let (gm2, gmm) = (gpu_in(&gpu, "m2", &m2), gpu_in(&gpu, "mm", &mm));
    let (gp2, gq) = (gpu_in(&gpu, "p2", &p2), gpu_in(&gpu, "q", &q));
    let go2 = gpu_out(&gpu, "o2", N * 5, Vec2::default());
    let go4 = gpu_out(&gpu, "o4", N * 6, Vec4::default());
    let gom = gpu_out(&gpu, "om", N * 4, Mat2::IDENTITY);
    gpu.dispatch_buffers(
        &vectors::small::DESCRIPTOR,
        N as u32,
        &[
            BufferBinding::read_only(&gm2),
            BufferBinding::read_only(&gmm),
            BufferBinding::read_only(&gp2),
            BufferBinding::read_only(&gq),
            BufferBinding::read_write(&go2).independent_length(),
            BufferBinding::read_write(&go4).independent_length(),
            BufferBinding::read_write(&gom).independent_length(),
        ],
    )
    .unwrap();
    let go2 = gpu.read_typed_buffer(&go2).unwrap();
    let go4 = gpu.read_typed_buffer(&go4).unwrap();
    let gom = gpu.read_typed_buffer(&gom).unwrap();

    let (tm2, tmm) = (cpu.buffer(&m2), cpu.buffer(&mm));
    let (tp2, tq) = (cpu.buffer(&p2), cpu.buffer(&q));
    let to2 = cpu.buffer(&vec![Vec2::default(); N * 5]);
    let to4 = cpu.buffer(&vec![Vec4::default(); N * 6]);
    let tom = cpu.buffer(&vec![Mat2::IDENTITY; N * 4]);
    vectors::small::dispatch(
        &cpu,
        N as u32,
        &tm2,
        &tmm,
        &tp2,
        &tq,
        Independent(&to2),
        Independent(&to4),
        Independent(&tom),
    )
    .unwrap();
    let (to2, to4, tom) = (to2.read(), to4.read(), tom.read());

    let mut check = Checker::default();
    for i in 0..N {
        let (a, b, p) = (&a_lanes[i], &b_lanes[i], p2[i].lanes());
        let product = host::mat_vec(a, &p, 2);
        let at: Vec<f32> = host::transpose(a, 2).iter().map(|e| *e as f32).collect();
        let dots = [
            f(p[0]) * f(p[1]) + f(p[1]) * f(p[0]),
            f(p[0]) * f(p[0]) + f(p[1]) * f(p[1]),
        ];
        let vec2_cases: [(&str, Vec<f64>, Bound); 5] = [
            ("m2_times_p", product.clone(), DOT2),
            ("mul_vec2", product, DOT2),
            ("identity2_times_p", host::widen(&p), EXACT),
            ("m2_transpose_times_p", host::mat_vec(&at, &p, 2), DOT2),
            ("vec2_dot", dots.to_vec(), DOT2),
        ];
        for (slot, (name, want, bound)) in vec2_cases.iter().enumerate() {
            let at = i * 5 + slot;
            let case = format!("small.{name}");
            check.lanes(&case, i, go2[at], to2[at], want, *bound);
        }

        // `from_rows(p, p.yx())`: row 0 is (p.x, p.y), row 1 is (p.y, p.x), so the columns are (p.x, p.y), (p.y, p.x).
        let rows = vec![f(p[0]), f(p[1]), f(p[1]), f(p[0])];
        let mat2_cases: [(&str, Vec<f64>, Bound); 4] = [
            ("m2_times_m2", host::mat_mat(a, b, 2), DOT2),
            ("identity2_mul_mat2", host::widen(a), EXACT),
            ("m2_transpose", host::transpose(a, 2), EXACT),
            ("from_rows2", rows, EXACT),
        ];
        for (slot, (name, want, bound)) in mat2_cases.iter().enumerate() {
            let at = i * 4 + slot;
            let case = format!("small.{name}");
            check.lanes(&case, i, gom[at], tom[at], want, *bound);
        }

        // The three-wide matrix of the kernel: columns c0 = q.xyz, c1 = q.zyx, c2 = q.yzx.
        let r = q[i].lanes();
        let (c0, c1, c2) = ([r[0], r[1], r[2]], [r[2], r[1], r[0]], [r[1], r[2], r[0]]);
        let m3: Vec<f32> = [c0, c1, c2].concat();
        let pad = |v: Vec<f64>| -> Vec<f64> { [v, vec![0.0]].concat() };
        let m3t: Vec<f32> = host::transpose(&m3, 3).iter().map(|e| *e as f32).collect();
        let square = host::mat_mat(&m3, &m3, 3);
        let cases: [(&str, Vec<f64>, Bound); 6] = [
            ("m3_times_c0", pad(host::mat_vec(&m3, &c0, 3)), DOT3),
            ("identity3_times_c1", pad(host::widen(&c1)), EXACT),
            (
                "m3_transpose_times_c2",
                pad(host::mat_vec(&m3t, &c2, 3)),
                DOT3,
            ),
            ("m3_squared_column_0", pad(square[0..3].to_vec()), DOT3),
            ("m3_squared_column_2", pad(square[6..9].to_vec()), DOT3),
            // `from_rows(c0, c1, c2)` is the transpose of `from_cols(c0, c1, c2)`.
            (
                "from_rows3_times_c0",
                pad(host::mat_vec(&m3t, &c0, 3)),
                DOT3,
            ),
        ];
        for (slot, (name, want, bound)) in cases.iter().enumerate() {
            let at = i * 6 + slot;
            let case = format!("small.{name}");
            let (got, twin) = (go4[at].lanes(), to4[at].lanes());
            for lane in 0..4 {
                let b = if lane == 3 { EXACT } else { *bound };
                check.float(&case, i, lane, got[lane], twin[lane], want[lane], b);
            }
        }
    }
    check.report("small_matrices");
}

fn body_lanes(body: &vectors::Body) -> Vec<f32> {
    [
        body.color.lanes(),
        body.xf.lanes(),
        body.frame.origin.lanes(),
        body.frame.scale.lanes(),
        vec![body.mass],
        body.extra.lanes(),
    ]
    .concat()
}

#[test]
fn struct_fields_of_vectors_and_matrices_execute_on_gpu() {
    let Rig { gpu, cpu } = rig();
    let mut rng = Lcg(5);
    let bodies: Vec<vectors::Body> = (0..N)
        .map(|i| vectors::Body {
            color: vec4(&rng.units(4)),
            xf: mat4(&rng.units(16)),
            frame: vectors::Frame {
                origin: vec2(&rng.units(2)),
                scale: vec2(&rng.units(2)),
            },
            mass: rng.magnitude(0.5, 1.5),
            tag: 1000 + i as u32,
            extra: vec2(&rng.units(2)),
        })
        .collect();
    let zero = vectors::Body {
        color: Vec4::default(),
        xf: Mat4::IDENTITY,
        frame: vectors::Frame {
            origin: Vec2::default(),
            scale: Vec2::default(),
        },
        mass: 0.0,
        tag: 0,
        extra: Vec2::default(),
    };

    let ginput = gpu_in(&gpu, "bodies", &bodies);
    let gmoved = gpu_out(&gpu, "moved", N, zero);
    let grebuilt = gpu_out(&gpu, "rebuilt", N, zero);
    gpu.dispatch_buffers(
        &vectors::bodies::DESCRIPTOR,
        N as u32,
        &[
            BufferBinding::read_only(&ginput),
            BufferBinding::read_write(&gmoved),
            BufferBinding::read_write(&grebuilt),
        ],
    )
    .unwrap();
    let gmoved = gpu.read_typed_buffer(&gmoved).unwrap();
    let grebuilt = gpu.read_typed_buffer(&grebuilt).unwrap();

    let tinput = cpu.buffer(&bodies);
    let tmoved = cpu.buffer(&vec![zero; N]);
    let trebuilt = cpu.buffer(&vec![zero; N]);
    vectors::bodies::dispatch(&cpu, N as u32, &tinput, &tmoved, &trebuilt).unwrap();
    let (tmoved, trebuilt) = (tmoved.read(), trebuilt.read());

    let mut check = Checker::default();
    for i in 0..N {
        let b = &bodies[i];
        let (color, xf) = (b.color.lanes(), b.xf.lanes());
        let (origin, scale, extra) = (
            b.frame.origin.lanes(),
            b.frame.scale.lanes(),
            b.extra.lanes(),
        );
        // moved: color = xf * color, xf transposed, origin += extra, scale *= mass, extra.x = color . color,
        // mass doubled, tag + 1. Bounds per field, in the layout order of `body_lanes`.
        let mut want = host::mat_vec(&xf, &color, 4);
        let mut bounds = vec![DOT4; 4];
        want.extend(host::transpose(&xf, 4));
        bounds.extend(vec![EXACT; 16]);
        want.extend([f(origin[0]) + f(extra[0]), f(origin[1]) + f(extra[1])]);
        bounds.extend(vec![SUM_2; 2]);
        want.extend([f(scale[0]) * f(b.mass), f(scale[1]) * f(b.mass)]);
        bounds.extend(vec![PRODUCT_15; 2]);
        want.push(f(b.mass) * 2.0);
        bounds.push(EXACT);
        want.extend([host::dot(&color, &color), f(extra[1])]);
        bounds.extend([DOT4, EXACT]);
        let (got, twin) = (body_lanes(&gmoved[i]), body_lanes(&tmoved[i]));
        for lane in 0..want.len() {
            let (g, t) = (got[lane], twin[lane]);
            check.float("bodies.moved", i, lane, g, t, want[lane], bounds[lane]);
        }
        assert_eq!(gmoved[i].tag, b.tag + 1, "moved[{i}].tag on the GPU");
        assert_eq!(tmoved[i].tag, b.tag + 1, "moved[{i}].tag on the twin");

        // rebuilt: a struct literal of splat, the identity, swizzled and constructed vectors, and copied scalars.
        let mut want = vec![f(b.mass); 4];
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        want.extend(host::widen(&identity));
        want.extend([f(extra[1]), f(extra[0])]);
        want.extend([f(color[3]), f(color[0])]);
        want.push(f(scale[0]));
        want.extend(host::widen(&origin));
        let (got, twin) = (body_lanes(&grebuilt[i]), body_lanes(&trebuilt[i]));
        for lane in 0..want.len() {
            let (g, t) = (got[lane], twin[lane]);
            check.float("bodies.rebuilt", i, lane, g, t, want[lane], EXACT);
        }
        assert_eq!(grebuilt[i].tag, b.tag, "rebuilt[{i}].tag on the GPU");
        assert_eq!(trebuilt[i].tag, b.tag, "rebuilt[{i}].tag on the twin");
    }
    check.report("bodies");
}

#[test]
fn locals_named_like_emitted_slang_names_do_not_capture_on_gpu() {
    let Rig { gpu, cpu } = rig();
    let mut rng = Lcg(6);
    // Multiples of 0.25 in [-2, 2]: every product and sum below is exact in `f32`.
    let input: Vec<Vec4> = (0..N)
        .map(|_| {
            let mut q = || (rng.range(0, 16) as f32 - 8.0) * 0.25;
            Vec4::new(q(), q(), q(), q())
        })
        .collect();

    let ginput = gpu_in(&gpu, "captures", &input);
    let gout = gpu_out(&gpu, "out", N * 8, Vec4::default());
    let gints = gpu_out(&gpu, "ints", N * 3, IVec4::default());
    gpu.dispatch_buffers(
        &vectors::captures::DESCRIPTOR,
        N as u32,
        &[
            BufferBinding::read_only(&ginput),
            BufferBinding::read_write(&gout).independent_length(),
            BufferBinding::read_write(&gints).independent_length(),
        ],
    )
    .unwrap();
    let gout = gpu.read_typed_buffer(&gout).unwrap();
    let gints = gpu.read_typed_buffer(&gints).unwrap();

    let tinput = cpu.buffer(&input);
    let tout = cpu.buffer(&vec![Vec4::default(); N * 8]);
    let tints = cpu.buffer(&vec![IVec4::default(); N * 3]);
    vectors::captures::dispatch(
        &cpu,
        N as u32,
        &tinput,
        Independent(&tout),
        Independent(&tints),
    )
    .unwrap();
    let (tout, tints) = (tout.read(), tints.read());

    let mut check = Checker::default();
    for i in 0..N {
        let p = input[i].lanes();
        let (mul, transpose, dot) = (p[0] * 2.0, p[1] + 1.0, p[2] - 1.0);
        let (float2, float3, float4) = (p[3] * 4.0, p[0] + p[1], p[2] * p[3]);
        let (float2x2, float3x3, float4x4) = (p[0] - p[3], p[1] * 2.0, p[3] + 3.0);
        let (vector, matrix) = (p[0] * 0.5, p[1] - p[2]);
        // m has the diagonal (mul, transpose, dot, 1) and the last column (float2, float3, float4, 1).
        let m: Vec<f32> = vec![
            mul, 0.0, 0.0, 0.0, 0.0, transpose, 0.0, 0.0, 0.0, 0.0, dot, 0.0, float2, float3,
            float4, 1.0,
        ];
        let by = |v: [f32; 4]| host::mat_vec(&m, &v, 4);
        let by_transpose = |v: [f32; 4]| {
            let t: Vec<f32> = host::transpose(&m, 4).iter().map(|e| *e as f32).collect();
            host::mat_vec(&t, &v, 4)
        };
        let squared = host::mat_mat(&m, &m, 4);
        let widen4 = |a: f32, b: f32, c: f32, d: f32| vec![f(a), f(b), f(c), f(d)];
        let expected: [Vec<f64>; 8] = [
            by([1.0, 2.0, 3.0, 1.0]),
            by_transpose([1.0; 4]),
            vec![
                host::dot(&p, &p),
                f(p[0]) * f(p[2]) + f(p[1]) * f(p[3]),
                f(float2) + f(float3),
                2.0 * (f(mul) + f(transpose) + f(dot)),
            ],
            // turn = from_cols((vector, matrix), (1, float2x2)) applied to (2, 3); wide is a diagonal float3x3.
            widen4(
                2.0 * vector + 3.0,
                2.0 * matrix + 3.0 * float2x2,
                float3x3,
                float4x4,
            ),
            widen4(mul, transpose, dot, float4),
            widen4(float2, float3, dot, mul),
            by([0.0, 0.0, 0.0, 1.0]).iter().map(|e| e * 2.0).collect(),
            // (m * m) * e0 is the first column of the square.
            squared[0..4].to_vec(),
        ];
        for (slot, want) in expected.iter().enumerate() {
            let at = i * 8 + slot;
            let case = format!("captures.out{slot}");
            check.lanes(&case, i, gout[at], tout[at], want, EXACT);
        }

        // The integer names: int2, int4, uint2, uint3, uint4, int, uint and bool as locals.
        let (int2, int4) = ((p[0] * 4.0) as i32, (p[1] * 4.0) as i32);
        let (uint2, uint3, uint4) = (i as i32 + 2, i as i32 + 3, i as i32 + 4);
        let (int, uint) = (int4 + 1, uint4 + 1);
        let flag = if i & 1 == 0 { 1 } else { 2 };
        let expected: [[i64; 4]; 3] = [
            [int2 + int, int4, int4, uint4 + uint].map(i64::from),
            [uint2 + uint4, uint3 + i as i32, flag, int2].map(i64::from),
            [uint2, uint3, uint4, int2 + 8 + i as i32].map(i64::from),
        ];
        for (slot, want) in expected.iter().enumerate() {
            let at = i * 3 + slot;
            let case = format!("captures.ints{slot}");
            exact_ints(
                &case,
                i,
                ivec4_lanes(gints[at]),
                ivec4_lanes(tints[at]),
                *want,
            );
        }
    }
    check.report("captures");
}

/// A descriptor that lies about one parameter's element layout, with a buffer of the type it claims, so the host
/// checks pass and only the reflection check of D17 (Slang's own layout, through the native helper) can refuse.
fn lying_descriptor(
    kernel: &gust::KernelDescriptor,
    parameter: &str,
    layout: gust::TypeLayout,
) -> gust::KernelDescriptor {
    let mut parameters = kernel.parameters.to_vec();
    let target = parameters
        .iter_mut()
        .find(|p| p.name == parameter)
        .expect("parameter");
    target.resource_layout = Some(gust::ResourceLayout::storage_v2(layout));
    gust::KernelDescriptor {
        parameters: Box::leak(parameters.into_boxed_slice()),
        ..*kernel
    }
}

fn reflection_message(error: Error) -> String {
    match error {
        Error::Reflection(gust::reflect::Error::Mismatch(message)) => message,
        other => panic!("expected a reflection mismatch, got {other}"),
    }
}

/// D17 on the new layouts: the native helper reflects vectors and matrices, and a descriptor whose element layout
/// differs from Slang's is refused before any pipeline exists. The kernels themselves passed the same check in the
/// tests above (every dispatch goes through it).
#[test]
fn reflection_refuses_vector_and_matrix_descriptors_that_lie() {
    let device = device();

    // `a` is a `float4` buffer in Slang; the descriptor and the host buffer both say `Vec2`.
    let descriptor = lying_descriptor(
        &vectors::componentwise::DESCRIPTOR,
        "a",
        <Vec2 as GpuPod>::LAYOUT,
    );
    let a = gpu_in(&device, "a", &vec![Vec2::default(); N]);
    let b = gpu_in(&device, "b", &vec![Vec4::default(); N]);
    let s = gpu_in(&device, "s", &vec![0.0f32; N]);
    let wide = gpu_out(&device, "wide", N * 14, Vec4::default());
    let narrow = gpu_out(&device, "narrow", N * 6, Vec2::default());
    let error = device
        .dispatch_buffers(
            &descriptor,
            N as u32,
            &[
                BufferBinding::read_only(&a),
                BufferBinding::read_only(&b),
                BufferBinding::read_only(&s),
                BufferBinding::read_write(&wide).independent_length(),
                BufferBinding::read_write(&narrow).independent_length(),
            ],
        )
        .expect_err("a Vec2 element must not pass for a float4 buffer");
    assert_eq!(reflection_message(error), "a.size: host 8, compiler 16");

    // The same size, a different scalar: `narrow` is a `float2` buffer; the descriptor and the buffer say `IVec2`.
    let descriptor = lying_descriptor(
        &vectors::componentwise::DESCRIPTOR,
        "narrow",
        <IVec2 as GpuPod>::LAYOUT,
    );
    let a = gpu_in(&device, "a", &vec![Vec4::default(); N]);
    let narrow = gpu_out(&device, "narrow", N * 6, IVec2::default());
    let error = device
        .dispatch_buffers(
            &descriptor,
            N as u32,
            &[
                BufferBinding::read_only(&a),
                BufferBinding::read_only(&b),
                BufferBinding::read_only(&s),
                BufferBinding::read_write(&wide).independent_length(),
                BufferBinding::read_write(&narrow).independent_length(),
            ],
        )
        .expect_err("an IVec2 element must not pass for a float2 buffer");
    assert_eq!(
        reflection_message(error),
        "narrow: type or field count differs"
    );

    // `m` is a `float4x4` buffer in Slang; the descriptor and the host buffer both say `Mat2`.
    let descriptor = lying_descriptor(
        &vectors::products::DESCRIPTOR,
        "m",
        <Mat2 as GpuPod>::LAYOUT,
    );
    let m = gpu_in(&device, "m", &vec![Mat2::IDENTITY; N]);
    let n = gpu_in(&device, "n", &vec![Mat4::IDENTITY; N]);
    let v = gpu_in(&device, "v", &vec![Vec4::default(); N]);
    let w = gpu_in(&device, "w", &vec![Vec4::default(); N]);
    let k = gpu_in(&device, "k", &vec![0.0f32; N]);
    let pv = gpu_out(&device, "pv", N * 4, Vec4::default());
    let pm = gpu_out(&device, "pm", N * 9, Mat4::IDENTITY);
    let error = device
        .dispatch_buffers(
            &descriptor,
            N as u32,
            &[
                BufferBinding::read_only(&m),
                BufferBinding::read_only(&n),
                BufferBinding::read_only(&v),
                BufferBinding::read_only(&w),
                BufferBinding::read_only(&k),
                BufferBinding::read_write(&pv).independent_length(),
                BufferBinding::read_write(&pm).independent_length(),
            ],
        )
        .expect_err("a Mat2 element must not pass for a float4x4 buffer");
    assert_eq!(reflection_message(error), "m.size: host 16, compiler 64");

    // The same size, a vector against a matrix: `q` of `small` is a `float4` buffer; the descriptor says `Mat2`.
    let descriptor = lying_descriptor(&vectors::small::DESCRIPTOR, "q", <Mat2 as GpuPod>::LAYOUT);
    let m2 = gpu_in(&device, "m2", &vec![Mat2::IDENTITY; N]);
    let p2 = gpu_in(&device, "p2", &vec![Vec2::default(); N]);
    let q = gpu_in(&device, "q", &vec![Mat2::IDENTITY; N]);
    let o2 = gpu_out(&device, "o2", N * 5, Vec2::default());
    let o4 = gpu_out(&device, "o4", N * 6, Vec4::default());
    let om = gpu_out(&device, "om", N * 4, Mat2::IDENTITY);
    let error = device
        .dispatch_buffers(
            &descriptor,
            N as u32,
            &[
                BufferBinding::read_only(&m2),
                BufferBinding::read_only(&m2),
                BufferBinding::read_only(&p2),
                BufferBinding::read_only(&q),
                BufferBinding::read_write(&o2).independent_length(),
                BufferBinding::read_write(&o4).independent_length(),
                BufferBinding::read_write(&om).independent_length(),
            ],
        )
        .expect_err("a Mat2 element must not pass for a float4 buffer");
    assert_eq!(
        reflection_message(error),
        "q.alignment: host 8, compiler 16"
    );
}
