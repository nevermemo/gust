// ANCHOR: imports
use gust::gpu_module;
// The GPU side is only built outside the tests: the tests run the loop on the CPU twin (see `Dev` below).
#[cfg(not(test))]
use gust_wgpu::{BufferDispatch, GpuBufferAccess, HeadlessDevice};
// ANCHOR_END: imports

// ANCHOR: kernel
#[gpu_module]
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
// ANCHOR_END: kernel

// ANCHOR: setup
/// The device type `energies` runs on: the GPU in the program, the CPU twin in the tests. `par_for!` writes its body
/// as a kernel at the call site and emits that kernel's twin only in test builds, so a program that must also run
/// `par_for!` in `cargo test` names its device through this alias (and `main`, which needs the GPU, is not built in
/// tests). A crate that only runs on the GPU needs none of this: it passes the `HeadlessDevice` directly.
#[cfg(not(test))]
type Dev = HeadlessDevice;
#[cfg(test)]
type Dev = gust::cpu::CpuDevice;

/// The number of steps one batch runs.
const STEPS: usize = 16;

/// The starting state of particle `i`: small integers, so every later value is a multiple of 2^-7 and exact.
fn start(i: usize) -> particles::Particle {
    particles::Particle {
        pos: [(i % 64) as f32, (i % 9) as f32 + 1.0],
        vel: [(i % 5) as f32 - 2.0, (i % 7) as f32],
        bounces: 0,
    }
}
// ANCHOR_END: setup

// ANCHOR: energy
/// The mechanical energy of every particle (unit mass, gravity 8): `par_for!` writes the loop at the call site, over
/// flat arrays that the host split out of the particles.
fn energies(y: &[f32], vy: &[f32], device: &Dev) -> Result<Vec<f32>, gust::map::Error> {
    let mut out = vec![0.0f32; y.len()];
    let n = u32::try_from(y.len()).expect("fewer than 2^32 particles");
    gust::par_for!(on(device); i in 0..n, [y: &[f32], vy: &[f32], out: &mut [f32]] {
        out[i] = 0.5 * vy[i] * vy[i] + 8.0 * y[i];
    })?;
    Ok(out)
}
// ANCHOR_END: energy

// ANCHOR: total
/// The energy of the particles through `par_for!` on `device`, summed on the host in `f64`.
fn total_energy(ps: &[particles::Particle], device: &Dev) -> f64 {
    let y: Vec<f32> = ps.iter().map(|p| p.pos[1]).collect();
    let vy: Vec<f32> = ps.iter().map(|p| p.vel[1]).collect();
    let energy = energies(&y, &vy, device).expect("par_for runs");
    energy.iter().map(|e| f64::from(*e)).sum()
}
// ANCHOR_END: total

// ANCHOR: host
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

// ANCHOR_END: host

// ANCHOR: main
/// The program; it runs on the GPU, so it is not part of the test build (which has the CPU twin for `Dev`).
#[cfg(not(test))]
fn main() {
    let n: usize = 1000;
    let count = u32::try_from(n).expect("fewer than 2^32 particles");
    let initial: Vec<_> = (0..n).map(start).collect();
    let device = HeadlessDevice::new().expect("a Vulkan compute adapter");
    let buffer = device.create_typed_buffer("particles", &initial, GpuBufferAccess::ReadWrite);

    // The same call sixteen times in one batch: one command buffer, one pass per step, each pass sees the last one's
    // writes. The call is built once and borrowed by every entry.
    let call = particles::step::call(count, &buffer);
    let batch: Vec<BufferDispatch<'_>> = (0..STEPS).map(|_| (&call).into()).collect();
    device.dispatch_batch(&batch).expect("dispatch the batch");

    let after = device.read_typed_buffer(&buffer).expect("read back");
    for (i, p) in after.iter().enumerate() {
        let want = host_steps(host_start(i), STEPS);
        assert_eq!(p.pos.map(f64::from), want.pos, "pos of particle {i}");
        assert_eq!(p.vel.map(f64::from), want.vel, "vel of particle {i}");
        assert_eq!(p.bounces, want.bounces, "bounces of particle {i}");
    }
    let bounces: u32 = after.iter().map(|p| p.bounces).sum();
    let energy = total_energy(&after, &device);
    println!("particles: {n} particles, {STEPS} steps, {bounces} bounces, energy {energy}");
    println!("adapter: {}", device.adapter_info().name);
}
// ANCHOR_END: main

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use gust::cpu::CpuDevice;

    use super::{STEPS, host_start, host_steps, particles, start, total_energy};

    /// `steps` steps on the CPU twin; the twin has no batch, so each step is one `dispatch`.
    fn on_twin(n: usize, steps: usize) -> Vec<particles::Particle> {
        let device = CpuDevice::new();
        let buffer = device.buffer(&(0..n).map(start).collect::<Vec<_>>());
        for _ in 0..steps {
            particles::step::dispatch(&device, u32::try_from(n).expect("small"), &buffer)
                .expect("the twin dispatches");
        }
        buffer.read()
    }

    #[test]
    fn the_twin_matches_hand_computed_values() {
        // Particle 0 starts at (0, 1) with velocity (-2, 0). Step 1: vy = -1, x = -0.25, y = 1 - 0.125 = 0.875.
        let one = on_twin(1, 1);
        assert_eq!(
            (one[0].pos, one[0].vel, one[0].bounces),
            ([-0.25, 0.875], [-2.0, -1.0], 0)
        );
    }

    /// Exact comparison: every constant is a power of two and every value a small multiple of one, so the `f64`
    /// reference and the `f32` kernel hold the same number at every step; there is no rounding to tolerate.
    #[test]
    fn the_twin_matches_the_host_reference_exactly() {
        // Sizes below, at and above the workgroup size of 64, ones that are not a multiple of it, and 1000.
        for n in [1usize, 63, 64, 65, 1000] {
            for steps in [1, 3, STEPS] {
                let got = on_twin(n, steps);
                for (i, p) in got.iter().enumerate() {
                    let want = host_steps(host_start(i), steps);
                    let case = format!("n = {n}, steps = {steps}, i = {i}");
                    assert_eq!(p.pos.map(f64::from), want.pos, "{case}");
                    assert_eq!(p.vel.map(f64::from), want.vel, "{case}");
                    assert_eq!(p.bounces, want.bounces, "{case}");
                }
            }
        }
    }

    #[test]
    fn some_particles_bounce() {
        // A guard on the example: the floor branch runs.
        let bounces: u32 = on_twin(1000, STEPS).iter().map(|p| p.bounces).sum();
        assert!(bounces > 0);
    }

    /// `par_for!` on the twin against the host: exact for the same reason as above (sums of exact `f32` values in `f64`).
    #[test]
    fn par_for_energy_matches_the_host_reference() {
        for n in [1usize, 63, 64, 65, 1000] {
            let after = on_twin(n, STEPS);
            let want: f64 = (0..n)
                .map(|i| {
                    let p = host_steps(host_start(i), STEPS);
                    0.5 * p.vel[1] * p.vel[1] + 8.0 * p.pos[1]
                })
                .sum();
            assert_eq!(total_energy(&after, &CpuDevice::new()), want, "n = {n}");
        }
    }
}
// ANCHOR_END: tests
