// `hypot`, `mul_add` and integer `pow` are not in the dialect (slices m11.math and m11.math2): a call with arguments
// is GUST-E0325 and the message lists the accepted methods, among them the float math ones. expected.txt pins the code.

#[gust::gpu]
mod shaping {
    #[kernel]
    pub fn shape(id: SV_DispatchThreadID, input: StructuredBuffer<float>, mut output: RWStructuredBuffer<float>) {
        let i = id.x;
        if i < input.len() && i < output.len() {
            output[i] = input[i].hypot(2.0);
        }
    }
}

fn main() {}
