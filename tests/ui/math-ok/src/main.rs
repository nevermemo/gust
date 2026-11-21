// Float math methods are in the dialect (slice m11.math): the validator, the lowering and the preserved Rust (which
// calls the std methods on `f32`, `u32` and `i32`) must all accept this kernel. A literal operand of `min`, `max` or
// `clamp` carries a suffix, and the constants are the full `std::f32::consts` paths.

#[gust::gpu]
mod shaping {
    #[kernel]
    pub fn shape(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        counts: StructuredBuffer<uint>,
        mut output: RWStructuredBuffer<float>,
        mut index: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < input.len() && i < counts.len() && i < output.len() && i < index.len() {
            let x = input[i];
            let r = x.abs().sqrt().min(4.0).max(0.5).clamp(0.25, 3.0);
            let wave = (x * std::f32::consts::TAU).sin() + (x * core::f32::consts::PI).cos();
            let angle = x.atan2(r) + x.atan() + x.tan();
            let grow = x.clamp(-1.0, 1.0).exp() + x.exp2() + r.ln() + r.log2() + r.powf(1.5);
            let parts = x.floor() + x.ceil() + x.trunc() + x.fract();
            output[i] = wave + angle + grow + parts + std::f32::consts::E;
            index[i] = counts[i].min(9u32).max(1u32).clamp(2u32, 8u32);
        }
    }
}

fn main() {}
