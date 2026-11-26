// An enum is never stored, and a uniform is storage: a kernel parameter `Uniform<Level>` is GUST-E0211 in a hand-written
// module, as for a buffer (D244, D270). Only the module a batch stage generates holds an enum's word, as a `u32` that
// the kernel turns into the enum. expected.txt pins the code and the line of the parameter type.

#[gust::gpu_module]
mod block {
    #[derive(PartialEq, Eq)]
    pub enum Level {
        Low,
        High,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, level: Uniform<Level>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = id.x;
    }
}

fn main() {}
