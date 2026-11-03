//! The user-kernel gather for a struct, the pattern `compact`'s documentation points at (plan section 5.4): there is no
//! generic `gather<T>` before M9, so the program writes the gather kernel for its own struct against the `indices`
//! and `count` of `compact::indices`. The kernel below is the fixture's gather for a struct; the twin test
//! and the adapter test (`gpu::gpu_a_user_kernel_gathers_structs_by_the_compaction_indices`) run it on the indices of a
//! real compaction.

use gust::{Independent, cpu::CpuDevice};

use crate::{cpu, schedule::Pass};

#[gust::gpu]
pub mod particles {
    #[derive(Debug, PartialEq)]
    pub struct Particle {
        pub id: uint,
        pub energy: float,
    }

    /// `out[j] = values[indices[j]]` for `j < count[0]`; the rest of `out` keeps its zero. It is the gather of
    /// `tests/fixtures/compact.rs` with the element type changed: block-strided (launch `Pass::over(n).partials()`
    /// invocations, not one per element, which the dispatch limit refuses past 16 776 960 elements), and it checks
    /// `source < values.len()` before it reads.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn gather_particles(
        id: SV_DispatchThreadID,
        values: StructuredBuffer<Particle>,
        indices: StructuredBuffer<uint>,
        count: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<Particle>,
    ) {
        let i = id.x;
        let n = out.len();
        let live = count[0u32];
        let first = (i / 256u32) * 65536u32 + i % 256u32;
        for k in 0u32..256u32 {
            let j = first + k * 256u32;
            if j >= n || j >= live {
                break;
            }
            let source = indices[j];
            if source < values.len() {
                out[j] = values[source];
            }
        }
    }
}

use particles::Particle;

/// Particles `0..n`, alive when their id is a multiple of three or five: the flags and the structs they select.
pub(super) fn population(n: usize) -> (Vec<u32>, Vec<Particle>) {
    let flags = (0..n as u32)
        .map(|id| u32::from(id % 3 == 0 || id % 5 == 0))
        .collect();
    let particles = (0..n as u32)
        .map(|id| Particle {
            id,
            energy: id as f32 * 0.25 + 1.0,
        })
        .collect();
    (flags, particles)
}

/// The independent reference: the live particles in order, then the zero particles the output is created with.
pub(super) fn reference(flags: &[u32], particles: &[Particle]) -> Vec<Particle> {
    let mut expected: Vec<Particle> = particles
        .iter()
        .zip(flags)
        .filter(|(_, flag)| **flag != 0)
        .map(|(particle, _)| Particle {
            id: particle.id,
            energy: particle.energy,
        })
        .collect();
    expected.resize_with(particles.len(), || Particle { id: 0, energy: 0.0 });
    expected
}

#[test]
fn twin_a_user_kernel_gathers_structs_by_the_compaction_indices() {
    for n in [1usize, 255, 257, 1_000] {
        let (flags, particles) = population(n);
        let (indices, count) = cpu::compact_indices(&flags).unwrap();
        let device = CpuDevice::new();
        let values = device.buffer(&particles);
        let indices = device.buffer(&indices);
        let count = device.buffer(&[count]);
        let out = device.buffer(&vec![Particle { id: 0, energy: 0.0 }; n]);
        particles::gather_particles::dispatch(
            &device,
            Pass::over(u32::try_from(n).unwrap()).partials(),
            Independent(&values),
            Independent(&indices),
            Independent(&count),
            Independent(&out),
        )
        .unwrap();
        assert_eq!(out.read(), reference(&flags, &particles), "n = {n}");
    }
}
