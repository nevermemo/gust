// An unannotated `let` whose initializer is an integer literal without a suffix is GUST-E0326: Slang would type it
// `int` while rustc infers the type from the later uses (`u32` here, from `acc += input[i]`), and the two differ for
// a division, a comparison or a shift. The help text asks for a suffix or an annotation. expected.txt pins the code
// and the line of the literal, which is where the error points.

#[gust::gpu]
mod sums {
    #[kernel]
    pub fn total(id: SV_DispatchThreadID, input: StructuredBuffer<uint>, mut output: RWStructuredBuffer<uint>) {
        let i = id.x;
        if i < input.len() && i < output.len() {
            let mut acc = 0;
            acc += input[i];
            output[i] = acc;
        }
    }
}

fn main() {}
