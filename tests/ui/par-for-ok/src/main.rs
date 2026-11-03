// Control case for `par_for!` (M7 plan 3.1, slice m7.p): the accepted forms expand and compile. Built under
// `GUST_PROFILE=cpu` (env.txt), where the loop runs sequentially on `gust::cpu::CpuDevice` and nothing names a GPU API.

use gust::cpu::CpuDevice;

fn scale(values: &[f32], out: &mut [f32], by: f32, device: &CpuDevice) -> Result<(), gust::map::Error> {
    let n = out.len() as u32;
    gust::par_for!(on(device); i in 0..n, [values: &[f32], out: &mut [f32], by: f32] {
        out[i] = values[i] * by;
    })
}

fn tile(out: &mut [u32], w: u32, h: u32, device: &CpuDevice) -> Result<(), gust::map::Error> {
    gust::par_for!(workgroup_size(8, 8, 1); on(device); (x, y) in (0..w, 0..h), [out: &mut [u32], w: u32] {
        out[y * w + x] = y * 100u32 + x;
    })
}

fn main() {
    let device = CpuDevice::new();
    let mut out = [0.0_f32; 3];
    scale(&[1.0, 2.0, 3.0], &mut out, 2.0, &device).unwrap();
    assert_eq!(out, [2.0, 4.0, 6.0]);
    let mut cells = [0_u32; 6];
    tile(&mut cells, 3, 2, &device).unwrap();
    assert_eq!(cells, [0, 1, 2, 100, 101, 102]);
}
