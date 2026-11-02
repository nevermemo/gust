// `wrapping_div` and `wrapping_rem` are not in the dialect: a zero divisor has no wrapped result. A method with
// arguments that is not one of the accepted ones is E0325, and the message lists the accepted methods; it used to be
// E0304 with the text "GPU built-ins do not accept arguments or type arguments". expected.txt pins the code.

#[gust::gpu]
mod hashing {
    #[kernel]
    pub fn divide(id: SV_DispatchThreadID, input: StructuredBuffer<uint>, mut output: RWStructuredBuffer<uint>) {
        let i = id.x;
        if i < input.len() && i < output.len() {
            output[i] = input[i].wrapping_div(3u32);
        }
    }
}

fn main() {}
