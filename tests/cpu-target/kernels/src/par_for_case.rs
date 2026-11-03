// `gust::par_for!` (M7, slice m7.p) in the production CPU target: a one-dimensional and a two-dimensional loop run on
// `gust::cpu::CpuDevice` with no GPU API and are compared with host loops. Built under `GUST_PROFILE=cpu` only (see
// `build.rs`); the macro's default-device form is used too, which under this profile is the CPU device.

use gust::cpu::CpuDevice;

/// More than one workgroup (256 wide) and not a multiple of it.
const ELEMENTS: usize = 1000;
/// A two-dimensional extent that is not a multiple of the 16 x 16 workgroup on either axis.
const WIDTH: u32 = 37;
const HEIGHT: u32 = 21;

fn product_1d(a: &[f32], b: &[f32], out: &mut [f32], device: &CpuDevice) -> Result<(), gust::map::Error> {
    let n = out.len() as u32;
    gust::par_for!(on(device); i in 0..n, [a: &[f32], b: &[f32], out: &mut [f32]] {
        out[i] = a[i] * b[i] + 1.0;
    })
}

fn tag_2d(out: &mut [u32], w: u32, h: u32, device: &CpuDevice) -> Result<(), gust::map::Error> {
    gust::par_for!(on(device); (x, y) in (0..w, 0..h), [out: &mut [u32], w: u32] {
        out[y * w + x] = y * 1000u32 + x + 1u32;
    })
}

/// The same loop without `on(..)`: the default device of the profile.
fn bump_default(out: &mut [u32]) -> Result<(), gust::map::Error> {
    let n = out.len() as u32;
    gust::par_for!(i in 0..n, [out: &mut [u32]] {
        out[i] += 1u32;
    })
}

/// Runs the loops and compares them with host references. Small integers, so every float result is exact.
pub fn run() -> Result<(), String> {
    let device = CpuDevice::new();
    let a: Vec<f32> = (0..ELEMENTS).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..ELEMENTS).map(|i| (3 * i % 17) as f32 - 8.0).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(a, b)| a * b + 1.0).collect();
    let mut out = vec![0.0_f32; ELEMENTS];
    product_1d(&a, &b, &mut out, &device).map_err(|error| format!("par_for! (1D): {error}"))?;
    if out != expected {
        return Err("par_for! (1D) differs from the host loop".to_owned());
    }

    let mut cells = vec![0_u32; (WIDTH * HEIGHT) as usize];
    tag_2d(&mut cells, WIDTH, HEIGHT, &device).map_err(|error| format!("par_for! (2D): {error}"))?;
    let mut expected_cells = vec![0_u32; (WIDTH * HEIGHT) as usize];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            expected_cells[(y * WIDTH + x) as usize] = y * 1000 + x + 1;
        }
    }
    if cells != expected_cells {
        return Err("par_for! (2D) differs from the host loop".to_owned());
    }

    // Each element exactly once, whatever the workgroup shape: a loop that ran an element twice would show a 2.
    let mut counts = vec![0_u32; ELEMENTS];
    bump_default(&mut counts).map_err(|error| format!("par_for! (default device): {error}"))?;
    if counts != vec![1_u32; ELEMENTS] {
        return Err("par_for! (default device) did not run each element exactly once".to_owned());
    }
    // An empty extent is a no-op, not an error.
    tag_2d(&mut cells, 0, HEIGHT, &device).map_err(|error| format!("par_for! (empty extent): {error}"))?;
    if cells != expected_cells {
        return Err("par_for! with an empty extent changed its buffer".to_owned());
    }
    // Not a `matches` line: `check-cpu-target` pairs those with its list of kernels. Its final line and the exit code
    // of `app` carry this case.
    println!("CPU target: par_for calls match (1D {ELEMENTS} elements, 2D {WIDTH} x {HEIGHT} cells)");
    Ok(())
}
