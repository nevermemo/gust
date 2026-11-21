// `powi` is not in the dialect (`hypot`, `mul_add` and integer `pow` were added by m11.math3): a call with arguments
// is GUST-E0325 and the message lists the accepted methods, among them the float math ones. expected.txt pins the code.

#[gust::gpu]
mod shaping {
    #[kernel]
    pub fn shape(id: SV_DispatchThreadID, input: StructuredBuffer<float>, mut output: RWStructuredBuffer<float>) {
        let i = id.x;
        if i < input.len() && i < output.len() {
            output[i] = input[i].powi(2);
        }
    }
}

fn main() {}
