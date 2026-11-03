// A 3-wide vector has stride 16 in a buffer but size 12 in Rust, so a `par_for!` buffer cannot hold it (GUST-E0204,
// docs/DIAGNOSTICS.md). The map generator refuses the same type in its own words (map-three-wide). expected.txt pins
// the code, the line of the binding and the start of the message.

use gust::cpu::CpuDevice;
use gust::math::Vec3;

fn run(points: &mut [Vec3], device: &CpuDevice) -> Result<(), gust::map::Error> {
    let n = points.len() as u32;
    gust::par_for!(on(device); i in 0..n, [points: &mut [Vec3]] {
        points[i] = points[i];
    })
}

fn main() {
    let _ = run(&mut [Vec3::new(0.0, 0.0, 0.0); 2], &CpuDevice::new());
}
