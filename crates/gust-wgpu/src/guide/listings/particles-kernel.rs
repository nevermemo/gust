#[gpu]
mod particles {
    /// A particle: arrays inside a struct. The layout is the one Rust has (`repr(C)`, 20 bytes), so the host fills
    /// the buffer with this very type.
    pub struct Particle {
        pub pos: [f32; 2],
        pub vel: [f32; 2],
        pub bounces: u32,
    }

    /// One time step of 1/8 under gravity 8 (so a step lowers the vertical speed by exactly 1), with a floor at
    /// `pos[1] = 0` that bounces the particle and halves its vertical speed. Every constant is a power of two, and
    /// the host fills in small integers, so each result is exactly representable in `f32`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn step(id: SV_DispatchThreadID, mut ps: RWStructuredBuffer<Particle>) {
        let i = id.x;
        if i < ps.len() {
            let mut p = ps[i];
            p.vel[1] -= 1.0;
            p.pos[0] += p.vel[0] * 0.125;
            p.pos[1] += p.vel[1] * 0.125;
            if p.pos[1] < 0.0 {
                p.pos[1] = -p.pos[1];
                p.vel[1] = -p.vel[1] * 0.5;
                p.bounces += 1u32;
            }
            ps[i] = p;
        }
    }
}
