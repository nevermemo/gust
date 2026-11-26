// A `par_for!` binding is a slice of a buffer element type or a scalar, and an enum is neither (GUST-E0720): the
// kernel of a hand-written loop takes no enum, as the buffer of a hand-written module takes none (GUST-E0211, D244, D270).
// expected.txt pins the code, the line of the binding and the start of the message.

use gust::cpu::CpuDevice;

#[allow(dead_code)]
enum Level {
    Low,
    High,
}

fn run(levels: &[Level], device: &CpuDevice) -> Result<(), gust::map::Error> {
    let n = levels.len() as u32;
    gust::par_for!(on(device); i in 0..n, [levels: &[Level]] {
        let _ = i;
    })
}

fn main() {
    let _ = run(&[Level::Low, Level::High], &CpuDevice::new());
}
