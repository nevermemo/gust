use std::{path::PathBuf, time::Instant};

use gust::{GpuPod, gpu};
use gust_wgpu::{BufferDispatch, GpuBufferAccess, HeadlessDevice, render_wgpu_source};

/// Particles per demo run: 16 384 workgroups of 64 invocations, within the 65 535 workgroups per
/// dimension of the default limits.
const PARTICLE_COUNT: usize = 1 << 20;

#[gpu]
mod particles {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn scalar_order(
        id: SV_DispatchThreadID,
        mut out: RWStructuredBuffer<float>,
        mut echo: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = 1.0;
            out[i] = 3.0;
            echo[i] = out[i];
        }
    }
    #[derive(Debug, PartialEq)]
    pub struct Pair {
        pub x: f32,
        pub y: f32,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn field_order(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<Particle>,
        mut out: RWStructuredBuffer<Particle>,
        mut echo: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            let before = input[i];
            out[i] = before;
            out[i].velocity.y = before.velocity.y + 3.0;
            echo[i] = out[i].velocity.y;
        }
    }

    #[derive(Debug, PartialEq)]
    pub struct Particle {
        pub position: Pair,
        pub velocity: Pair,
        pub mass: f32,
        pub tag: u32,
        pub charge: i32,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn step(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<Particle>,
        acceleration: StructuredBuffer<Pair>,
        mut out: RWStructuredBuffer<Particle>,
        mut energy: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            let particle = input[i];
            let force = acceleration[i];
            let vx = particle.velocity.x + force.x * 0.016;
            let vy = particle.velocity.y + force.y * 0.016;
            // Preserve mass and integer metadata with a whole-struct copy.
            out[i] = particle;
            out[i].velocity.x = vx;
            out[i].velocity.y = vy;
            out[i].position.x = particle.position.x + vx * 0.016;
            out[i].position.y = particle.position.y + vy * 0.016;
            // Read updated fields back to exercise memory ordering within an invocation.
            let speed_squared =
                out[i].velocity.x * out[i].velocity.x + out[i].velocity.y * out[i].velocity.y;
            energy[i] = speed_squared * particle.mass * 0.5;
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn snapshot(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<Particle>,
        mut out: RWStructuredBuffer<Particle>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = input[i];
        }
    }
}

fn data(count: usize) -> (Vec<particles::Particle>, Vec<particles::Pair>) {
    let input = (0..count)
        .map(|i| particles::Particle {
            position: particles::Pair {
                x: (i % 257) as f32 * 0.125,
                y: -2.0,
            },
            velocity: particles::Pair {
                x: 0.5,
                y: (i % 31) as f32 * -0.0625,
            },
            mass: 1.0 + (i % 7) as f32 * 0.25,
            tag: i as u32 ^ 0xdead_beef,
            charge: (i % 5) as i32 - 2,
        })
        .collect();
    let acceleration = (0..count)
        .map(|i| particles::Pair {
            x: (i % 13) as f32 * 0.25,
            y: -9.81,
        })
        .collect();
    (input, acceleration)
}

fn cpu_step(
    input: &[particles::Particle],
    acceleration: &[particles::Pair],
) -> (Vec<particles::Particle>, Vec<f32>) {
    let mut output = input.to_vec();
    let mut energy = vec![0.0; input.len()];
    for (index, (particle, force)) in input.iter().zip(acceleration).enumerate() {
        let vx = particle.velocity.x + force.x * 0.016;
        let vy = particle.velocity.y + force.y * 0.016;
        output[index].velocity.x = vx;
        output[index].velocity.y = vy;
        output[index].position.x = particle.position.x + vx * 0.016;
        output[index].position.y = particle.position.y + vy * 0.016;
        energy[index] = (vx * vx + vy * vy) * particle.mass * 0.5;
    }
    (output, energy)
}

fn assert_matches(
    actual: &[particles::Particle],
    energy: &[f32],
    expected: &[particles::Particle],
    expected_energy: &[f32],
) {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(energy.len(), expected_energy.len());
    for ((actual, energy), (expected, expected_energy)) in actual
        .iter()
        .zip(energy)
        .zip(expected.iter().zip(expected_energy))
    {
        for (actual, expected) in [
            (actual.position.x, expected.position.x),
            (actual.position.y, expected.position.y),
            (actual.velocity.x, expected.velocity.x),
            (actual.velocity.y, expected.velocity.y),
            (actual.mass, expected.mass),
            (*energy, *expected_energy),
        ] {
            assert!(
                (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0),
                "GPU value {actual} differs from CPU value {expected}"
            );
        }
        assert_eq!(actual.tag, expected.tag);
        assert_eq!(actual.charge, expected.charge);
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let count = PARTICLE_COUNT;
    let (input, acceleration) = data(count);
    let device = HeadlessDevice::new()?;
    let cpu_started = Instant::now();
    let (expected, expected_energy) = cpu_step(&input, &acceleration);
    let cpu_elapsed = cpu_started.elapsed();
    let gpu_input = device.create_typed_buffer("particles", &input, GpuBufferAccess::ReadOnly);
    let gpu_acceleration =
        device.create_typed_buffer("acceleration", &acceleration, GpuBufferAccess::ReadOnly);
    let gpu_output =
        device.create_typed_buffer("updated particles", &input, GpuBufferAccess::ReadWrite);
    let gpu_snapshot = device.create_typed_buffer("snapshot", &input, GpuBufferAccess::ReadWrite);
    let gpu_energy =
        device.create_f32_buffer("energy", &vec![0.0; count], GpuBufferAccess::ReadWrite);
    // Typed calls are locals; the batch entries borrow them. `snapshot` reads what `step` wrote, so the
    // batch order carries the dependency.
    let step = particles::step::call(
        count as u32,
        &gpu_input,
        &gpu_acceleration,
        &gpu_output,
        &gpu_energy,
    );
    let snapshot = particles::snapshot::call(count as u32, &gpu_output, &gpu_snapshot);
    let batch = [BufferDispatch::from(&step), BufferDispatch::from(&snapshot)];
    device.dispatch_batch(&batch)?; // Warm the compiled-kernel cache and complete initial uploads.
    let timing = device.dispatch_batch_timed(&batch)?;
    let actual = device.read_typed_buffer(&gpu_snapshot)?;
    let energy = device.read_f32_buffer(&gpu_energy)?;
    assert_matches(&actual, &energy, &expected, &expected_energy);
    // The trait method submits a whole batch (a generated `submit` covers one kernel); this is the
    // example that runs a two-kernel batch through the backend-neutral `gust::Device`.
    let job = gust::Device::submit(&device, &batch)?;
    job.wait()?;
    println!(
        "Particle step: {count} particles on {}",
        device.adapter_info().name
    );
    println!(
        "Particle storage ABI: {} bytes, alignment {}, stride {}",
        <particles::Particle as GpuPod>::LAYOUT.size,
        <particles::Particle as GpuPod>::LAYOUT.alignment,
        <particles::Particle as GpuPod>::LAYOUT.stride
    );
    println!("CPU step including output allocation: {cpu_elapsed:?}");
    println!(
        "GPU resident step + snapshot, host time: {:?}",
        timing.host_elapsed
    );
    println!(
        "GPU resident step + snapshot, hardware time: {:?}",
        timing.gpu_elapsed
    );
    println!(
        "Nested fields, integer metadata, energy output, and dependent snapshot match the CPU reference."
    );
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../generated-wgpu");
    std::fs::create_dir_all(&directory)?;
    for descriptor in [
        &particles::step::DESCRIPTOR,
        &particles::snapshot::DESCRIPTOR,
    ] {
        let stem = format!("{}__{}", descriptor.module, descriptor.name);
        let path = directory.join(format!("{stem}.rs"));
        std::fs::write(&path, render_wgpu_source(descriptor)?)?;
        std::fs::write(
            directory.join(format!("{stem}.slang")),
            descriptor.slang_source,
        )?;
        std::fs::write(
            directory.join(format!("{stem}.wgsl")),
            gust::slang::compile_wgsl(descriptor)?,
        )?;
        std::fs::write(
            directory.join(format!("{stem}.spv")),
            gust::spirv::words_as_le_bytes(&gust::slang::compile_spirv(descriptor)?),
        )?;
        println!(
            "generated shader artifacts: {}",
            path.canonicalize()?.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gust::TypeLayoutKind;
    use gust::cpu::{CpuDevice, Error as CpuError, ulps};

    fn device() -> HeadlessDevice {
        HeadlessDevice::new().unwrap_or_else(|error| {
            panic!(
                "Vulkan adapter required; this test must not silently skip: could not initialize headless wgpu: {error}"
            )
        })
    }

    fn blank() -> particles::Particle {
        particles::Particle {
            position: particles::Pair { x: 0.0, y: 0.0 },
            velocity: particles::Pair { x: 0.0, y: 0.0 },
            mass: 0.0,
            tag: 0,
            charge: 0,
        }
    }

    /// The twin-versus-GPU bounds for `step`, absolute, per output, in units of `U = 2^-24`.
    ///
    /// Derived from the kernel text (docs/SEMANTICS.md section 4, "Deriving a tolerance"), not from a measurement.
    /// With the domain asserted by `assert_derivation_domain`, `c = 0.016` rounded up to 0.0161, one rounding of at
    /// most `u` times the result's magnitude on the twin and `2u` on the GPU for `+ - *`, first-order propagation,
    /// magnitudes `M`, twin error `T` and GPU error `G` in units of `u`, the GPU side without fusion:
    ///
    /// | step | operation | M | T | G |
    /// | --- | --- | --- | --- | --- |
    /// | `vx` | `velocity.x + force.x * c`, `abs(velocity.x) <= 0.5`, `0 <= force.x <= 3` | 0.55 | 0.60 | 1.19 |
    /// | `vy` | `velocity.y + force.y * c`, `abs(velocity.y) <= 1.875`, `abs(force.y) <= 9.82` | 2.04 | 2.19 | 4.38 |
    /// | `position.x` | `position.x + vx * c`, `0 <= position.x <= 32` | 32.01 | 32.03 | 64.05 |
    /// | `position.y` | `position.y + vy * c`, `abs(position.y) <= 2` | 2.04 | 2.10 | 4.20 |
    /// | `vx * vx` | error `2 M T` plus the rounding | 0.31 | 0.95 | 1.91 |
    /// | `vy * vy` | error `2 M T` plus the rounding | 4.14 | 13.04 | 26.09 |
    /// | `speed_squared` | sum of the two squares | 4.44 | 18.43 | 36.86 |
    /// | `energy` | `speed_squared * mass * 0.5`, `1 <= mass <= 2.5` (the 0.5 is exact) | 5.55 | 28.58 | 57.17 |
    ///
    /// Twin plus GPU, rounded up to a whole `u` plus 1u for the second-order terms (below 0.001u here): `velocity.x`
    /// 1.79u gives 3u, `velocity.y` 6.57u gives 8u, `position.x` 96.08u gives 98u, `position.y` 6.30u gives 8u and
    /// `energy` 85.75u gives 87u. Fusion only removes a rounding; reassociating `speed_squared * mass * 0.5` leaves
    /// one rounding of the same relative size. `mass`, `tag` and `charge` are copied and must be equal.
    ///
    /// Observed maxima are printed separately by the ignored test and never become the tolerance.
    const STEP_BOUNDS_IN_U: [f64; 5] = [3.0, 8.0, 98.0, 8.0, 87.0];

    /// The names of the five bounded outputs, in the order of `STEP_BOUNDS_IN_U`.
    const STEP_OUTPUTS: [&str; 5] = [
        "velocity.x",
        "velocity.y",
        "position.x",
        "position.y",
        "energy",
    ];

    /// `2^-24`.
    const U: f64 = 1.0 / 16_777_216.0;

    /// The domain the bounds are derived for. `data` is periodic with periods of at most 257, so 4096 particles
    /// reach every extreme.
    fn assert_derivation_domain(input: &[particles::Particle], acceleration: &[particles::Pair]) {
        for particle in input {
            assert!((0.0..=32.0).contains(&particle.position.x), "position.x");
            assert!(particle.position.y.abs() <= 2.0, "position.y");
            assert!(particle.velocity.x.abs() <= 0.5, "velocity.x");
            assert!(particle.velocity.y.abs() <= 1.875, "velocity.y");
            assert!((1.0..=2.5).contains(&particle.mass), "mass");
        }
        for force in acceleration {
            assert!((0.0..=3.0).contains(&force.x), "force.x");
            assert!(force.y.abs() <= 9.82, "force.y");
        }
    }

    fn scalar_order_on_cpu(count: usize) -> Result<(Vec<f32>, Vec<f32>), CpuError> {
        let device = CpuDevice::new();
        let out = device.buffer(&vec![-1.0f32; count]);
        let echo = device.buffer(&vec![-1.0f32; count]);
        particles::scalar_order::dispatch(&device, count as u32, &out, &echo)?;
        Ok((out.read(), echo.read()))
    }

    fn field_order_on_cpu(
        input: &[particles::Particle],
    ) -> Result<(Vec<particles::Particle>, Vec<f32>), CpuError> {
        let device = CpuDevice::new();
        let source = device.buffer(input);
        let out = device.buffer(&vec![blank(); input.len()]);
        let echo = device.buffer(&vec![-1.0f32; input.len()]);
        particles::field_order::dispatch(&device, input.len() as u32, &source, &out, &echo)?;
        Ok((out.read(), echo.read()))
    }

    fn step_on_cpu(
        input: &[particles::Particle],
        acceleration: &[particles::Pair],
    ) -> Result<(Vec<particles::Particle>, Vec<f32>), CpuError> {
        let device = CpuDevice::new();
        let source = device.buffer(input);
        let force = device.buffer(acceleration);
        let out = device.buffer(&vec![blank(); input.len()]);
        let energy = device.buffer(&vec![-1.0f32; input.len()]);
        particles::step::dispatch(&device, input.len() as u32, &source, &force, &out, &energy)?;
        Ok((out.read(), energy.read()))
    }

    fn snapshot_on_cpu(
        input: &[particles::Particle],
    ) -> Result<Vec<particles::Particle>, CpuError> {
        let device = CpuDevice::new();
        let source = device.buffer(input);
        let out = device.buffer(&vec![blank(); input.len()]);
        particles::snapshot::dispatch(&device, input.len() as u32, &source, &out)?;
        Ok(out.read())
    }

    /// What `field_order` computes: the input with `velocity.y + 3.0` (and that sum echoed). The values are
    /// multiples of 1/16, so the sum is exact in `f32` on both targets.
    fn field_order_reference(
        input: &[particles::Particle],
    ) -> (Vec<particles::Particle>, Vec<f32>) {
        let mut expected = input.to_vec();
        let mut echo = Vec::new();
        for particle in &mut expected {
            particle.velocity.y += 3.0;
            echo.push(particle.velocity.y);
        }
        (expected, echo)
    }

    /// The largest distance between two results in ULPs and in absolute terms (computed in `f64`).
    fn largest_difference(left: &[f32], right: &[f32]) -> (u64, f64) {
        assert_eq!(left.len(), right.len());
        left.iter().zip(right).fold((0, 0.0), |(ulp, abs), (l, r)| {
            (
                ulp.max(ulps(*l, *r)),
                abs.max((f64::from(*l) - f64::from(*r)).abs()),
            )
        })
    }

    /// Per bounded output of `step` (the order of `STEP_OUTPUTS`): the largest (ULPs, absolute difference) between
    /// two results, each a particle buffer and an energy buffer. The mass and the integer fields are asserted
    /// equal here.
    fn step_differences(
        left: (&[particles::Particle], &[f32]),
        right: (&[particles::Particle], &[f32]),
    ) -> [(u64, f64); 5] {
        let field = |get: fn(&particles::Particle) -> f32| {
            let l: Vec<f32> = left.0.iter().map(get).collect();
            let r: Vec<f32> = right.0.iter().map(get).collect();
            largest_difference(&l, &r)
        };
        for (l, r) in left.0.iter().zip(right.0) {
            assert_eq!((l.tag, l.charge, l.mass), (r.tag, r.charge, r.mass));
        }
        [
            field(|p| p.velocity.x),
            field(|p| p.velocity.y),
            field(|p| p.position.x),
            field(|p| p.position.y),
            largest_difference(left.1, right.1),
        ]
    }

    #[test]
    fn inputs_stay_inside_the_derivation_domain() {
        let (input, acceleration) = data(4096);
        assert_derivation_domain(&input, &acceleration);
        let (input, acceleration) = data(PARTICLE_COUNT);
        assert_derivation_domain(&input, &acceleration);
    }

    #[test]
    fn twin_matches_host_references_for_every_kernel() {
        // Zero, one, a partial workgroup, a whole one (64) and counts that end mid-workgroup.
        for count in [0, 1, 63, 64, 65, 129, 4096] {
            let (input, acceleration) = data(count);

            let (out, echo) = scalar_order_on_cpu(count).expect("scalar_order twin");
            assert_eq!(out, vec![3.0; count], "scalar_order out, {count}");
            assert_eq!(echo, vec![3.0; count], "scalar_order echo, {count}");

            let (out, echo) = field_order_on_cpu(&input).expect("field_order twin");
            let (expected, expected_echo) = field_order_reference(&input);
            assert_eq!(out, expected, "field_order out, {count}");
            assert_eq!(echo, expected_echo, "field_order echo, {count}");

            // `step` performs the operations of `cpu_step` in the same order, so the two agree to the last bit.
            let (out, energy) = step_on_cpu(&input, &acceleration).expect("step twin");
            let (expected, expected_energy) = cpu_step(&input, &acceleration);
            assert_eq!(out, expected, "step out, {count}");
            assert_eq!(energy, expected_energy, "step energy, {count}");
            let differences = step_differences((&out, &energy), (&expected, &expected_energy));
            for (max_ulps, _) in differences {
                assert_eq!(max_ulps, 0, "step, {count}");
            }

            assert_eq!(
                snapshot_on_cpu(&input).expect("snapshot twin"),
                input,
                "snapshot, {count}"
            );
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn twin_matches_reference_and_gpu_for_the_exact_kernels() {
        let device = device();
        for count in [1, 63, 64, 65, 257] {
            let (input, _) = data(count);
            let zeros = vec![0.0f32; count];

            let out = device.create_f32_buffer("out", &zeros, GpuBufferAccess::ReadWrite);
            let echo = device.create_f32_buffer("echo", &zeros, GpuBufferAccess::ReadWrite);
            particles::scalar_order::dispatch(&device, count as u32, &out, &echo).unwrap();
            let gpu = (
                device.read_f32_buffer(&out).unwrap(),
                device.read_f32_buffer(&echo).unwrap(),
            );
            let host = (vec![3.0f32; count], vec![3.0f32; count]);
            assert_eq!(gpu, host, "scalar_order GPU versus reference, {count}");
            assert_eq!(
                scalar_order_on_cpu(count).unwrap(),
                host,
                "scalar_order twin versus reference, {count}"
            );

            let source = device.create_typed_buffer("input", &input, GpuBufferAccess::ReadOnly);
            let out = device.create_typed_buffer(
                "out",
                &vec![blank(); count],
                GpuBufferAccess::ReadWrite,
            );
            let echo = device.create_f32_buffer("echo", &zeros, GpuBufferAccess::ReadWrite);
            particles::field_order::dispatch(&device, count as u32, &source, &out, &echo).unwrap();
            let gpu = (
                device.read_typed_buffer(&out).unwrap(),
                device.read_f32_buffer(&echo).unwrap(),
            );
            let host = field_order_reference(&input);
            assert_eq!(gpu, host, "field_order GPU versus reference, {count}");
            assert_eq!(
                field_order_on_cpu(&input).unwrap(),
                host,
                "field_order twin versus reference, {count}"
            );

            let out = device.create_typed_buffer(
                "out",
                &vec![blank(); count],
                GpuBufferAccess::ReadWrite,
            );
            particles::snapshot::dispatch(&device, count as u32, &source, &out).unwrap();
            assert_eq!(
                device.read_typed_buffer(&out).unwrap(),
                input,
                "snapshot GPU versus reference, {count}"
            );
            assert_eq!(
                snapshot_on_cpu(&input).unwrap(),
                input,
                "snapshot twin versus reference, {count}"
            );
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn step_twin_matches_reference_and_gpu_within_the_derived_bounds() {
        let device = device();
        let count = 1 << 14;
        let (input, acceleration) = data(count);
        assert_derivation_domain(&input, &acceleration);
        let (host, host_energy) = cpu_step(&input, &acceleration);

        let source = device.create_typed_buffer("input", &input, GpuBufferAccess::ReadOnly);
        let force = device.create_typed_buffer("force", &acceleration, GpuBufferAccess::ReadOnly);
        let out = device.create_typed_buffer("out", &input, GpuBufferAccess::ReadWrite);
        let energy =
            device.create_f32_buffer("energy", &vec![0.0; count], GpuBufferAccess::ReadWrite);
        particles::step::dispatch(&device, count as u32, &source, &force, &out, &energy).unwrap();
        let gpu = device.read_typed_buffer(&out).unwrap();
        let gpu_energy = device.read_f32_buffer(&energy).unwrap();
        let (twin, twin_energy) = step_on_cpu(&input, &acceleration).expect("step twin");

        let gpu_host = step_differences((&gpu, &gpu_energy), (&host, &host_energy));
        let twin_host = step_differences((&twin, &twin_energy), (&host, &host_energy));
        let twin_gpu = step_differences((&twin, &twin_energy), (&gpu, &gpu_energy));
        // Observations on this adapter, recorded in the slice log (probe/s4-e1.log); not tolerances.
        println!("S4 adapter: {}", device.adapter_info().name);
        for (index, name) in STEP_OUTPUTS.iter().enumerate() {
            println!(
                "S4 step {name}: GPU versus cpu_step max {} ULP, abs {:e}; twin versus cpu_step max {} ULP, abs {:e}; twin versus GPU max {} ULP, abs {:e}; derived bound {:e}",
                gpu_host[index].0,
                gpu_host[index].1,
                twin_host[index].0,
                twin_host[index].1,
                twin_gpu[index].0,
                twin_gpu[index].1,
                STEP_BOUNDS_IN_U[index] * U,
            );
        }

        // The GPU against the host reference comes first so that a failing twin cannot hide it.
        for (index, name) in STEP_OUTPUTS.iter().enumerate() {
            assert!(
                gpu_host[index].1 <= STEP_BOUNDS_IN_U[index] * U,
                "step {name}: GPU versus cpu_step abs error {:e} exceeds the derived bound {:e}",
                gpu_host[index].1,
                STEP_BOUNDS_IN_U[index] * U
            );
        }
        assert_eq!(twin, host, "step twin versus cpu_step");
        assert_eq!(twin_energy, host_energy, "step twin energy versus cpu_step");
        for (index, name) in STEP_OUTPUTS.iter().enumerate() {
            assert!(
                twin_gpu[index].1 <= STEP_BOUNDS_IN_U[index] * U,
                "step {name}: twin versus GPU abs error {:e} exceeds the derived bound {:e}",
                twin_gpu[index].1,
                STEP_BOUNDS_IN_U[index] * U
            );
        }
    }

    #[test]
    fn demo_size_stays_within_the_default_workgroup_limit() {
        for descriptor in [
            &particles::step::DESCRIPTOR,
            &particles::snapshot::DESCRIPTOR,
        ] {
            let groups = u32::try_from(PARTICLE_COUNT)
                .unwrap()
                .div_ceil(descriptor.workgroup_size[0]);
            assert!(
                groups <= gust_wgpu::DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                "{}: {PARTICLE_COUNT} particles need {groups} workgroups",
                descriptor.entry_point
            );
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn nested_layout_matches_executable_artifacts() {
        let layout = <particles::Particle as GpuPod>::LAYOUT;
        assert_eq!((layout.size, layout.alignment, layout.stride), (28, 4, 28));
        let TypeLayoutKind::Struct(fields) = layout.kind else {
            panic!("expected struct")
        };
        assert_eq!(
            fields.iter().map(|field| field.offset).collect::<Vec<_>>(),
            [0, 8, 16, 20, 24]
        );
        for descriptor in [
            &particles::step::DESCRIPTOR,
            &particles::snapshot::DESCRIPTOR,
        ] {
            let source = render_wgpu_source(descriptor).unwrap();
            assert!(source.contains("Particle [storage-v1; size=28, align=4, stride=28]"));
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn scalar_read_after_write() {
        let device = device();
        let out = device.create_f32_buffer("out", &[0.0], GpuBufferAccess::ReadWrite);
        let echo = device.create_f32_buffer("echo", &[0.0], GpuBufferAccess::ReadWrite);
        particles::scalar_order::dispatch(&device, 1, &out, &echo).unwrap();
        assert_eq!(device.read_f32_buffer(&out).unwrap(), vec![3.0]);
        assert_eq!(device.read_f32_buffer(&echo).unwrap(), vec![3.0]);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn field_read_after_write() {
        let device = device();
        let out = device.create_typed_buffer("particle", &data(1).0, GpuBufferAccess::ReadWrite);
        let input = device.create_typed_buffer("input", &data(1).0, GpuBufferAccess::ReadOnly);
        let echo = device.create_f32_buffer("echo", &[0.0], GpuBufferAccess::ReadWrite);
        particles::field_order::dispatch(&device, 1, &input, &out, &echo).unwrap();
        assert_eq!(device.read_typed_buffer(&out).unwrap()[0].velocity.y, 3.0);
        assert_eq!(device.read_f32_buffer(&echo).unwrap(), vec![3.0]);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn cpu_step_has_expected_physics_and_preserves_metadata() {
        let (input, acceleration) = data(1);
        let (output, energy) = cpu_step(&input, &acceleration);
        assert_eq!(output[0].position.x, 0.008);
        assert!((output[0].velocity.y + 0.15696).abs() < 1e-6);
        assert!((energy[0] - 0.13731822).abs() < 1e-6);
        assert_eq!(output[0].tag, input[0].tag);
        assert_eq!(output[0].charge, -2);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_nested_structs_and_dependent_batches_match_cpu() {
        let device = device();
        for count in [0, 1, 63, 64, 65, 129] {
            let (input, acceleration) = data(count);
            let (expected, expected_energy) = cpu_step(&input, &acceleration);
            let source = device.create_typed_buffer("input", &input, GpuBufferAccess::ReadOnly);
            let force =
                device.create_typed_buffer("force", &acceleration, GpuBufferAccess::ReadOnly);
            let out = device.create_typed_buffer("out", &input, GpuBufferAccess::ReadWrite);
            let snapshot =
                device.create_typed_buffer("snapshot", &input, GpuBufferAccess::ReadWrite);
            let energy =
                device.create_f32_buffer("energy", &vec![0.0; count], GpuBufferAccess::ReadWrite);
            let step = particles::step::call(count as u32, &source, &force, &out, &energy);
            let snapshot_call = particles::snapshot::call(count as u32, &out, &snapshot);
            let batch = [
                BufferDispatch::from(&step),
                BufferDispatch::from(&snapshot_call),
            ];
            gust::Device::submit(&device, &batch)
                .unwrap()
                .wait()
                .unwrap();
            assert_matches(
                &device.read_typed_buffer(&snapshot).unwrap(),
                &device.read_f32_buffer(&energy).unwrap(),
                &expected,
                &expected_energy,
            );
            let timing = device.dispatch_batch_timed(&batch).unwrap();
            assert_eq!(
                timing.gpu_elapsed.is_some(),
                count > 0 && device.supports_gpu_timestamps()
            );
            assert_matches(
                &device.read_typed_buffer(&snapshot).unwrap(),
                &device.read_f32_buffer(&energy).unwrap(),
                &expected,
                &expected_energy,
            );
        }
        assert_eq!(device.pipeline_cache_stats().entries, 2);
    }
}
