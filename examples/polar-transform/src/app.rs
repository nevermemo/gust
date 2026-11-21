//! Polar coordinates and rotation with the float math methods of the kernel language (slice m11.math).

use gust::gpu;
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

const POINTS: usize = 4096;

#[gpu]
mod polar {
    pub struct Polar {
        pub radius: f32,
        pub angle: f32,
    }

    pub struct Point {
        pub x: f32,
        pub y: f32,
    }

    /// `(x, y)` as a radius and an angle in `(-pi, pi]`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn to_polar(
        id: SV_DispatchThreadID,
        points: StructuredBuffer<Point>,
        mut out: RWStructuredBuffer<Polar>,
    ) {
        let i = id.x;
        if i < out.len() {
            let p = points[i];
            out[i] = Polar {
                radius: (p.x * p.x + p.y * p.y).sqrt(),
                angle: p.y.atan2(p.x),
            };
        }
    }

    /// Each point turned by `turn` radians, with the angle wrapped by `fract` into `[-pi, pi)` first (turns are not negative).
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn rotate(
        id: SV_DispatchThreadID,
        points: StructuredBuffer<Point>,
        turns: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<Point>,
    ) {
        let i = id.x;
        if i < out.len() {
            let p = points[i];
            let wrapped =
                ((turns[i] / std::f32::consts::TAU + 0.5).fract() - 0.5) * std::f32::consts::TAU;
            let turn = wrapped.clamp(-std::f32::consts::PI, std::f32::consts::PI);
            out[i] = Point {
                x: p.x * turn.cos() - p.y * turn.sin(),
                y: p.x * turn.sin() + p.y * turn.cos(),
            };
        }
    }
}

fn points() -> Vec<polar::Point> {
    (0..POINTS)
        .map(|i| polar::Point {
            x: ((i * 37) % 201) as f32 * 0.05 - 5.0 + 0.013,
            y: ((i * 91) % 199) as f32 * 0.05 - 5.0 + 0.013,
        })
        .collect()
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = HeadlessDevice::new()?;
    let input = points();
    let points = device.create_typed_buffer("points", &input, GpuBufferAccess::ReadOnly);
    let blank = polar::Polar {
        radius: 0.0,
        angle: 0.0,
    };
    let out = device.create_typed_buffer("polar", &vec![blank; POINTS], GpuBufferAccess::ReadWrite);
    device.dispatch_buffers(
        &polar::to_polar::DESCRIPTOR,
        POINTS as u32,
        &[
            BufferBinding::read_only(&points),
            BufferBinding::read_write(&out),
        ],
    )?;
    let result = device.read_typed_buffer(&out)?;
    println!(
        "polar-transform on {}: {} points",
        device.adapter_info().name,
        result.len()
    );
    println!(
        "first point: radius {} angle {}",
        result[0].radius, result[0].angle
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gust::cpu::CpuDevice;

    fn device() -> HeadlessDevice {
        let device = HeadlessDevice::new()
            .expect("Vulkan adapter required; this test must not silently skip");
        let info = device.adapter_info();
        println!("adapter: {} backend: {:?}", info.name, info.backend);
        device
    }

    /// The bound of each output, from the WGSL accuracy table (WGSL 15.7.4.1, see docs/SEMANTICS.md): `radius` is
    /// `sqrt` (inherited, 6.5 ULP) of a sum of two squares (two correctly rounded products and a sum: a relative error of at
    /// most `3 * 2^-23`, half of it after the root, at most 3 ULP of the result): 9.5 ULP, rounded up to 10. `angle` is `atan2`, with `abs(x) >= 0.013`.
    /// `atan2` has 4096 ULP. A rotated coordinate has `sin` and `cos` absolute errors of `2^-11` each times
    /// the point's coordinates, plus 4 `f32` roundings of its largest term.
    #[test]
    fn polar_and_rotation_match_the_host_on_the_gpu_and_the_twin() {
        let device = device();
        let input = points();
        let gpu_points = device.create_typed_buffer("points", &input, GpuBufferAccess::ReadOnly);
        let turns: Vec<f32> = (0..POINTS).map(|i| (i % 97) as f32 * 0.31 + 0.1).collect();
        let gpu_turns = device.create_typed_buffer("turns", &turns, GpuBufferAccess::ReadOnly);
        let blank = polar::Polar {
            radius: 0.0,
            angle: 0.0,
        };
        let gpu_polar =
            device.create_typed_buffer("polar", &vec![blank; POINTS], GpuBufferAccess::ReadWrite);
        let origin = polar::Point { x: 0.0, y: 0.0 };
        let gpu_rotated = device.create_typed_buffer(
            "rotated",
            &vec![origin; POINTS],
            GpuBufferAccess::ReadWrite,
        );
        device
            .dispatch_buffers(
                &polar::to_polar::DESCRIPTOR,
                POINTS as u32,
                &[
                    BufferBinding::read_only(&gpu_points),
                    BufferBinding::read_write(&gpu_polar),
                ],
            )
            .unwrap();
        device
            .dispatch_buffers(
                &polar::rotate::DESCRIPTOR,
                POINTS as u32,
                &[
                    BufferBinding::read_only(&gpu_points),
                    BufferBinding::read_only(&gpu_turns),
                    BufferBinding::read_write(&gpu_rotated),
                ],
            )
            .unwrap();
        let gpu_polar = device.read_typed_buffer(&gpu_polar).unwrap();
        let gpu_rotated = device.read_typed_buffer(&gpu_rotated).unwrap();

        let cpu = CpuDevice::new();
        let (cpu_points, cpu_turns) = (cpu.buffer(&input), cpu.buffer(&turns));
        let (cpu_polar, cpu_rotated) = (
            cpu.buffer(&vec![blank; POINTS]),
            cpu.buffer(&vec![origin; POINTS]),
        );
        polar::to_polar::dispatch(&cpu, POINTS as u32, &cpu_points, &cpu_polar).unwrap();
        polar::rotate::dispatch(&cpu, POINTS as u32, &cpu_points, &cpu_turns, &cpu_rotated)
            .unwrap();
        let (twin_polar, twin_rotated) = (cpu_polar.read(), cpu_rotated.read());

        let ulp = |x: f64| {
            2f64.powi(((x.abs() as f32).max(f32::MIN_POSITIVE).to_bits() >> 23) as i32 - 127 - 23)
        };
        for i in 0..POINTS {
            let (x, y) = (f64::from(input[i].x), f64::from(input[i].y));
            let radius = x.hypot(y);
            let angle = y.atan2(x);
            // The host wraps the turn as the kernel does, in `f64`, from the same `f32` input.
            let t = f64::from(turns[i]) / std::f64::consts::TAU + 0.5;
            let turn = (((t - t.trunc()) - 0.5) * std::f64::consts::TAU)
                .clamp(-std::f64::consts::PI, std::f64::consts::PI);
            let (rx, ry) = (
                x * turn.cos() - y * turn.sin(),
                x * turn.sin() + y * turn.cos(),
            );
            // The wrap itself is a few `f32` operations on a value up to 2^3: allow 8 ULP of the turn on the angle.
            let turn_error = 8.0 * ulp(turn).max(ulp(6.3));
            let rotation_bound =
                (x.abs() + y.abs()) * (2f64.powi(-11) + 4.0 * 2f64.powi(-24) + turn_error);
            for (label, polar, rotated) in [
                ("gpu", &gpu_polar[i], &gpu_rotated[i]),
                ("twin", &twin_polar[i], &twin_rotated[i]),
            ] {
                assert!(
                    (f64::from(polar.radius) - radius).abs() <= 10.0 * ulp(radius),
                    "{label} radius[{i}]"
                );
                assert!(
                    (f64::from(polar.angle) - angle).abs() <= 4096.0 * ulp(angle),
                    "{label} angle[{i}]"
                );
                assert!(
                    (f64::from(rotated.x) - rx).abs() <= rotation_bound,
                    "{label} x[{i}]"
                );
                assert!(
                    (f64::from(rotated.y) - ry).abs() <= rotation_bound,
                    "{label} y[{i}]"
                );
            }
        }
    }
}
