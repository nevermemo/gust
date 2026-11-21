// Float math in the kernel language (slice m11.math): the exact methods, the integer methods, the constants and the
// transcendentals, each in a kernel of its own so a failure names the group.
#[gust::gpu]
mod float_math {
    pub struct Exact {
        pub f_abs: float,
        pub f_min: float,
        pub f_max: float,
        pub f_clamp: float,
        pub f_floor: float,
        pub f_ceil: float,
        pub f_trunc: float,
        pub f_fract: float,
    }

    pub struct Ints {
        pub i_abs: int,
        pub i_min: int,
        pub i_max: int,
        pub i_clamp: int,
        pub u_min: uint,
        pub u_max: uint,
        pub u_clamp: uint,
    }

    pub struct Consts {
        pub pi: float,
        pub tau: float,
        pub e: float,
    }

    /// One row of inputs per element, each field inside the domain of the methods that read it.
    pub struct TransIn {
        pub angle: float,
        pub tan_arg: float,
        pub wide: float,
        pub exp_arg: float,
        pub exp2_arg: float,
        pub pos: float,
        pub base: float,
        pub expo: float,
        pub y: float,
        pub x: float,
    }

    pub struct TransOut {
        pub t_sin: float,
        pub t_cos: float,
        pub t_tan: float,
        pub t_atan: float,
        pub t_exp: float,
        pub t_exp2: float,
        pub t_ln: float,
        pub t_log2: float,
        pub t_sqrt: float,
        pub t_powf: float,
        pub t_atan2: float,
    }

    // `lo` and `hi` are ordered by `min` and `max`, so `clamp` never sees `lo > hi`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn exact(
        id: SV_DispatchThreadID,
        xs: StructuredBuffer<float>,
        ys: StructuredBuffer<float>,
        zs: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<Exact>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            let x = xs[i];
            let lo = ys[i].min(zs[i]);
            let hi = ys[i].max(zs[i]);
            out[i] = Exact {
                f_abs: x.abs(),
                f_min: x.min(ys[i]),
                f_max: x.max(ys[i]),
                f_clamp: x.clamp(lo, hi),
                f_floor: x.floor(),
                f_ceil: x.ceil(),
                f_trunc: x.trunc(),
                f_fract: x.fract(),
            };
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn ints(
        id: SV_DispatchThreadID,
        ia: StructuredBuffer<int>,
        ib: StructuredBuffer<int>,
        ic: StructuredBuffer<int>,
        ua: StructuredBuffer<uint>,
        ub: StructuredBuffer<uint>,
        uc: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<Ints>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            out[i] = Ints {
                i_abs: ia[i].abs(),
                i_min: ia[i].min(ib[i]),
                i_max: ia[i].max(ib[i]),
                i_clamp: ia[i].clamp(ib[i].min(ic[i]), ib[i].max(ic[i])),
                u_min: ua[i].min(ub[i]),
                u_max: ua[i].max(ub[i]),
                u_clamp: ua[i].clamp(ub[i].min(uc[i]), ub[i].max(uc[i])),
            };
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn constants(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<Consts>) {
        let i: uint = id.x;
        if i < out.len() {
            out[i] = Consts {
                pi: std::f32::consts::PI,
                tau: core::f32::consts::TAU,
                e: std::f32::consts::E,
            };
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn transcendental(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<TransIn>,
        mut out: RWStructuredBuffer<TransOut>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            let v = input[i];
            out[i] = TransOut {
                t_sin: v.angle.sin(),
                t_cos: v.angle.cos(),
                t_tan: v.tan_arg.tan(),
                t_atan: v.wide.atan(),
                t_exp: v.exp_arg.exp(),
                t_exp2: v.exp2_arg.exp2(),
                t_ln: v.pos.ln(),
                t_log2: v.pos.log2(),
                t_sqrt: v.pos.sqrt(),
                t_powf: v.base.powf(v.expo),
                t_atan2: v.y.atan2(v.x),
            };
        }
    }
}
