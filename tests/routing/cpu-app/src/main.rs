// Runs the CPU-routed `physics::scale` on `CpuDevice` and prints one line, which `cargo xtask check-routing` looks for.

use cpu_app::run_scale;

fn main() {
    let result = run_scale();
    assert_eq!(result, [2.0, 4.0, 6.0, 8.0], "scale doubles its input");
    println!("routing: physics::scale ran on CpuDevice {result:?}");
}
