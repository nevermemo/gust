// A `match` that gives a value needs a type its arms show, or a typed `let`, GUST-E0419: here no arm shows one
// (unsuffixed literals). expected.txt pins the code and the line of the `let`.

#[gust::gpu_module]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let x = match input[id.x] {
            0 => 1,
            _ => 2,
        };
        out[id.x] = x as u32;
    }
}

fn main() {}
