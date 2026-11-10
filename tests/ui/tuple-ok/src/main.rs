// Control case for tuples (m9.t): a helper that returns a tuple, a destructuring (plain and annotated), `t.0`, a tuple
// argument, a tuple of tuples and an element assignment. This crate must compile.

#[gust::gpu]
mod block {
    fn classify(v: u32) -> (u32, f32) {
        (v * 2u32 + 1u32, v as f32 * 0.5f32)
    }

    fn total(t: (u32, f32)) -> f32 {
        t.0 as f32 + t.1
    }

    fn nest(v: u32) -> ((u32, u32), u32) {
        ((v, v + 1u32), v + 2u32)
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<f32>) {
        let v = input[id.x];
        let (a, b) = classify(v);
        let (c, d): (u32, f32) = classify(v + 1u32);
        let mut t = classify(v);
        t.0 = t.0 * 2u32 + 1u32;
        let (inner, z) = nest(v);
        out[id.x] = a as f32 + b + c as f32 + d + t.0 as f32 + total((v, 1.5f32)) + (inner.0 + inner.1 + z) as f32;
    }
}

fn main() {}
