//! The CPU app of the routing fixture (m8.r2): `run_scale` dispatches `physics::scale` on `CpuDevice`, and the test checks
//! both the target the build gave the kernel and the result.

use gust::cpu::CpuDevice;
use route_kernels::physics;

/// Doubles four numbers with the routed kernel on `CpuDevice`.
pub fn run_scale() -> Vec<f32> {
    let device = CpuDevice::new();
    let input = device.buffer(&[1.0f32, 2.0, 3.0, 4.0]);
    let out = device.buffer(&[0.0f32; 4]);
    physics::scale::dispatch(&device, 4, &input, &out).expect("dispatch on CpuDevice");
    out.read()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Passes under `cargo gust test` with `physics = "cpu"` in this package's `gust.toml`; plain `cargo test` leaves the
    /// route at its default (gpu) and the first assertion fails.
    #[test]
    fn the_cpu_route_reached_the_build() {
        assert_eq!(
            physics::scale::DESCRIPTOR.target,
            gust::target::Target::Cpu,
            "the build did not carry the route variable"
        );
        assert_eq!(run_scale(), [2.0, 4.0, 6.0, 8.0]);
    }
}
