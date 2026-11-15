/// The host reference: its own particle type and plain Rust in `f64`, which never goes through the kernel (D26,
/// D42). Gravity 8 and a step of 1/8 are written as the numbers they are.
#[derive(Clone, Copy)]
struct Host {
    pos: [f64; 2],
    vel: [f64; 2],
    bounces: u32,
}

fn host_start(i: usize) -> Host {
    Host {
        pos: [(i % 64) as f64, (i % 9) as f64 + 1.0],
        vel: [(i % 5) as f64 - 2.0, (i % 7) as f64],
        bounces: 0,
    }
}

fn host_steps(mut p: Host, steps: usize) -> Host {
    for _ in 0..steps {
        p.vel[1] -= 8.0 * 0.125;
        p.pos[0] += p.vel[0] * 0.125;
        p.pos[1] += p.vel[1] * 0.125;
        if p.pos[1] < 0.0 {
            p.pos[1] = -p.pos[1];
            p.vel[1] = -p.vel[1] / 2.0;
            p.bounces += 1;
        }
    }
    p
}

