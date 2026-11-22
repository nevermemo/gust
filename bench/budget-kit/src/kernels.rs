//! The `#[gpu]` module of the kit: every kernel, in one place.

use gust::gpu;

#[gpu]
pub(crate) mod kit {
    pub struct Params {
        pub n: u32,
        pub side: u32,
        pub pad0: u32,
        pub pad1: u32,
    }

    // The width of the invocation grid the kernels index through (`id.y * ROW + id.x`); the host reads the same
    // constant (m11.g3) to build `Extent::new(ROW, rows, 1)`.
    pub const ROW: u32 = 8192;

    // The flat index of an invocation in a grid of rows of `ROW`.
    fn linear(x: u32, y: u32) -> u32 {
        y * ROW + x
    }

    // `a` wrapped once into [0, side) (a is below 2 * side).
    fn wrap_index(a: u32, side: u32) -> u32 {
        if a >= side { a - side } else { a }
    }

    // The cell of a position in [0, side], clamped to the last cell.
    fn cell_of(v: f32, side: u32) -> u32 {
        let c = v as u32;
        if c >= side { side - 1u32 } else { c }
    }

    // `v` wrapped once into [0, side] (v is in [-side, 2 * side)).
    fn wrap_pos(v: f32, side: f32) -> f32 {
        let mut w = v;
        if w >= side {
            w -= side;
        }
        if w < 0.0f32 {
            w += side;
        }
        w
    }

    // W1: read one array, write another.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w1_copy(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        input: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            out[i] = input[i];
        }
    }

    // W2: one diffusion step of a `side x side` grid of four channels, wrapping at the edges. Each cell keeps half of
    // itself, takes an eighth of each of its four neighbours, and decays by 0.2 percent.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w2_diffuse(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        input: StructuredBuffer<Vec4>,
        mut out: RWStructuredBuffer<Vec4>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let x = i % p.side;
            let y = i / p.side;
            let xl = wrap_index(x + p.side - 1u32, p.side);
            let xr = wrap_index(x + 1u32, p.side);
            let yu = wrap_index(y + p.side - 1u32, p.side);
            let yd = wrap_index(y + 1u32, p.side);
            let c = input[i];
            let l = input[y * p.side + xl];
            let r = input[y * p.side + xr];
            let u = input[yu * p.side + x];
            let d = input[yd * p.side + x];
            out[i] = (c * 0.5f32 + (l + r + u + d) * 0.125f32) * 0.998f32;
        }
    }

    // W3: one step of a particle (x, y, vx, vy in the four lanes): sample the field at the particle's cell, accelerate,
    // move, wrap into [0, side). The field is `side x side`.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w3_step(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        field: StructuredBuffer<Vec4>,
        mut parts: RWStructuredBuffer<Vec4>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let q = parts[i];
            let side = p.side as f32;
            let cx = cell_of(q.x, p.side);
            let cy = cell_of(q.y, p.side);
            let f = field[cy * p.side + cx];
            let vx = q.z + (f.x - 0.5f32) * 0.05f32;
            let vy = q.w + (f.y - 0.5f32) * 0.05f32;
            let x = wrap_pos(q.x + vx, side);
            let y = wrap_pos(q.y + vy, side);
            parts[i] = Vec4::new(x, y, vx, vy);
        }
    }

    // W4: zero the per-cell counts (`n` is the number of cells).
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w4_clear(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        mut counts: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            counts[i] = 0u32;
        }
    }

    // W4: count the particles of each cell of a `side x side` grid with an atomic add. Positions are in [0, side).
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w4_count(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        parts: StructuredBuffer<Vec4>,
        mut counts: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let q = parts[i];
            let cx = cell_of(q.x, p.side);
            let cy = cell_of(q.y, p.side);
            atomic_add(&mut counts[cy * p.side + cx], 1u32);
        }
    }

    // W4: the scatter. Each particle takes the next free slot of its cell (`old` is the cell's cursor before the add,
    // which starts at zero) and writes its own index at `starts[cell] + old`. `starts` is the exclusive scan of the
    // counts. Which particle gets which slot is schedule-dependent: the order inside a cell is unspecified.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w4_scatter(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        parts: StructuredBuffer<Vec4>,
        starts: StructuredBuffer<u32>,
        mut cursor: RWStructuredBuffer<u32>,
        mut order: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let q = parts[i];
            let cx = cell_of(q.x, p.side);
            let cy = cell_of(q.y, p.side);
            let c = cy * p.side + cx;
            let old = atomic_add(&mut cursor[c], 1u32);
            order[starts[c] + old] = i;
        }
    }

    // G4 probe: a linear launch, `i = id.x`; one invocation per element, so the workgroup count on x is n / 256.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn g4_linear(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < p.n {
            out[i] = i.wrapping_mul(3u32).wrapping_add(1u32);
        }
    }

    // G4 remedy: the same elements through a two-dimensional launch of rows of `p.side` invocations, with the tail
    // guard the kernel owns (the extent is rounded up to whole workgroups in x and y).
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn g4_grid(id: SV_DispatchThreadID, p: Uniform<Params>, mut out: RWStructuredBuffer<u32>) {
        let i = id.y * p.side + id.x;
        if i >= p.n {
            return;
        }
        out[i] = i.wrapping_mul(3u32).wrapping_add(1u32);
    }

    // W6: the small kernel whose result the host waits for.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w6_plus_one(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        input: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            out[i] = input[i].wrapping_add(1u32);
        }
    }

    // ---- m11.kit4: W8, a full step with a lifecycle ----

    // The parameters of a lifecycle kernel, eight `u32` (32 bytes): the capacity of the particle buffers, the live
    // count before the step, the step number, the survivors of the step and the particles that spawn (both known to the
    // host from its model), the field side, and two pad fields.
    pub struct LifeParams {
        pub cap: u32,
        pub live: u32,
        pub step: u32,
        pub surv: u32,
        pub spawn: u32,
        pub side: u32,
        pub pad0: u32,
        pub pad1: u32,
    }

    // The amount every live particle adds to its cell of the deposit grid.
    pub const DEPOSIT: u32 = 8;
    // A particle retires when the low 16 bits of its hash draw are below this (over 2^16): one in sixteen.
    pub const RETIRE: u32 = 4096;
    // Seeds of the retire draw and of the spawn draw.
    pub const RETIRE_SEED: u32 = 3735928559;
    pub const SPAWN_SEED: u32 = 2654435769;

    // The "lowbias32" finaliser (Chris Wellons' hash-prospector, public domain), as in `examples/walker-life`.
    fn lb_mix(x: u32) -> u32 {
        let mut h = x ^ (x >> 16u32);
        h = h.wrapping_mul(2146121005u32);
        h = h ^ (h >> 15u32);
        h = h.wrapping_mul(2221713035u32);
        h ^ (h >> 16u32)
    }

    // A hash of a seed and an index, as in `examples/walker-life`.
    fn lb_hash(seed: u32, index: u32) -> u32 {
        lb_mix(lb_mix(seed) ^ index)
    }

    // A value in [0, 1), exact in `f32` (the high 24 bits of the hash over 2^24).
    fn unit_of(h: u32) -> f32 {
        ((h >> 8u32) as f32) / 16777216.0f32
    }

    // W8: the deposit. Every particle adds `DEPOSIT` to its cell of a `side x side` grid of `u32` with an atomic add.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w8_deposit(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        parts: StructuredBuffer<Vec4>,
        mut dep: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let q = parts[i];
            let cx = cell_of(q.x, p.side);
            let cy = cell_of(q.y, p.side);
            atomic_add(&mut dep[cy * p.side + cx], DEPOSIT);
        }
    }

    // W8: folds the deposit grid into the field: lane w of each cell gains its deposit over 1024.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w8_fold(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        dep: StructuredBuffer<u32>,
        mut field: RWStructuredBuffer<Vec4>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let f = field[i];
            field[i] = Vec4::new(f.x, f.y, f.z, f.w + (dep[i] as f32) / 1024.0f32);
        }
    }

    // W8: the alive flags, one invocation per slot of the capacity. A live slot (`i < live`) is alive unless the
    // hash of (step, slot) says it retires; a slot past `live` is not alive, so the scan over the capacity counts
    // the live particles only.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w8_retire(
        id: SV_DispatchThreadID,
        p: Uniform<LifeParams>,
        mut alive: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.cap {
            let mut a = 0u32;
            if i < p.live {
                let r = lb_hash(lb_hash(RETIRE_SEED, p.step), i);
                if (r & 65535u32) >= RETIRE {
                    a = 1u32;
                }
            }
            alive[i] = a;
        }
    }

    // W8: the compaction. Each live slot that is alive writes its particle to the slot `starts[i]` of the second
    // buffer, `starts` being the exclusive scan of the alive flags: the survivors stay in order and dense.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w8_compact(
        id: SV_DispatchThreadID,
        p: Uniform<LifeParams>,
        parts: StructuredBuffer<Vec4>,
        alive: StructuredBuffer<u32>,
        starts: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<Vec4>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.live && alive[i] == 1u32 {
            out[starts[i]] = parts[i];
        }
    }

    // W8: the spawn. Particle `k` of the step takes the slot `surv + k` after the survivors, if that is inside the
    // capacity (the rest are dropped, so the population is clamped at the capacity), with a position in [0, side)
    // and a velocity in [-1, 1) drawn from the hash of (step, k).
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w8_spawn(
        id: SV_DispatchThreadID,
        p: Uniform<LifeParams>,
        mut out: RWStructuredBuffer<Vec4>,
    ) {
        let k = linear(id.x, id.y);
        let t = p.surv + k;
        if k < p.spawn && t < p.cap {
            let h = lb_hash(lb_hash(SPAWN_SEED, p.step), k);
            let side = p.side as f32;
            out[t] = Vec4::new(
                unit_of(lb_hash(h, 0u32)) * side,
                unit_of(lb_hash(h, 1u32)) * side,
                unit_of(lb_hash(h, 2u32)) * 2.0f32 - 1.0f32,
                unit_of(lb_hash(h, 3u32)) * 2.0f32 - 1.0f32,
            );
        }
    }

    // ---- m11.cost: kernels of the W8 variants ----

    // The parameters of a bin kernel on a grid coarser than the field: the particle count, the bins per side and the
    // field side (the positions are in [0, field]).
    pub struct BinParams {
        pub n: u32,
        pub bins: u32,
        pub field: u32,
        pub pad0: u32,
    }

    // The bin of a position in [0, field]: the position scaled to a `bins` wide grid, clamped to the last bin. The scale
    // is exact when both sides are powers of two, so the host computes the same bin from the same position.
    fn bin_of(v: f32, bins: u32, field: u32) -> u32 {
        cell_of(v * ((bins as f32) / (field as f32)), bins)
    }

    // W8b: `w4_count` on a `bins x bins` grid over the field.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w8_bin_count(
        id: SV_DispatchThreadID,
        p: Uniform<BinParams>,
        parts: StructuredBuffer<Vec4>,
        mut counts: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let q = parts[i];
            let cx = bin_of(q.x, p.bins, p.field);
            let cy = bin_of(q.y, p.bins, p.field);
            atomic_add(&mut counts[cy * p.bins + cx], 1u32);
        }
    }

    // W8b: `w4_scatter` on a `bins x bins` grid over the field.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w8_bin_scatter(
        id: SV_DispatchThreadID,
        p: Uniform<BinParams>,
        parts: StructuredBuffer<Vec4>,
        starts: StructuredBuffer<u32>,
        mut cursor: RWStructuredBuffer<u32>,
        mut order: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let q = parts[i];
            let cx = bin_of(q.x, p.bins, p.field);
            let cy = bin_of(q.y, p.bins, p.field);
            let c = cy * p.bins + cx;
            let old = atomic_add(&mut cursor[c], 1u32);
            order[starts[c] + old] = i;
        }
    }

    // W8c: the particle step of `w3_step`, then the count of `w4_count` and the deposit of `w8_deposit` at the new
    // position, in one kernel.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w8_step_count_deposit(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        field: StructuredBuffer<Vec4>,
        mut parts: RWStructuredBuffer<Vec4>,
        mut counts: RWStructuredBuffer<u32>,
        mut dep: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let q = parts[i];
            let side = p.side as f32;
            let cx = cell_of(q.x, p.side);
            let cy = cell_of(q.y, p.side);
            let f = field[cy * p.side + cx];
            let vx = q.z + (f.x - 0.5f32) * 0.05f32;
            let vy = q.w + (f.y - 0.5f32) * 0.05f32;
            let x = wrap_pos(q.x + vx, side);
            let y = wrap_pos(q.y + vy, side);
            parts[i] = Vec4::new(x, y, vx, vy);
            let nx = cell_of(x, p.side);
            let ny = cell_of(y, p.side);
            atomic_add(&mut counts[ny * p.side + nx], 1u32);
            atomic_add(&mut dep[ny * p.side + nx], DEPOSIT);
        }
    }
}
