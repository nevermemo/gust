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
