// A module the macro refuses for a construct of its own (here a `while` loop, GUST-E0402, found by the validator
// before any expansion) still gets the items its expansion would have written: the module's struct is a buffer
// element that host code uploads, and the only error is the macro's. expected.txt pins the one error.

fn upload<T: gust::GpuPod>(_values: &[T]) {}

#[gust::gpu_module]
mod stepping {
    pub struct Particle {
        pub position: f32,
        pub speed: f32,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn step(id: SV_DispatchThreadID, mut particles: RWStructuredBuffer<Particle>) {
        let i = id.x;
        let mut count = 0u32;
        while count < 4u32 {
            count += 1u32;
        }
        if i < particles.len() {
            particles[i].position = particles[i].position + particles[i].speed * count as f32;
        }
    }
}

fn main() {
    upload(&[stepping::Particle { position: 0.0, speed: 1.0 }]);
}
