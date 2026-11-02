// The wrapping integer methods are in the dialect (slice m7.w): a multiplicative hash that overflows for most inputs
// and is defined modulo 2^32 for all of them. This crate must compile, so the validator, the lowering and the
// preserved Rust (which calls the std methods on `u32` and `i32`) all accept it.

#[gust::gpu]
mod hashing {
    fn scramble(x: uint) -> uint {
        x.wrapping_mul(2654435761u32).wrapping_add(2654435769u32)
    }

    #[kernel]
    pub fn hash(id: SV_DispatchThreadID, input: StructuredBuffer<uint>, mut output: RWStructuredBuffer<uint>) {
        let i = id.x;
        if i < input.len() && i < output.len() {
            output[i] = scramble(input[i]).wrapping_sub(i).wrapping_neg();
        }
    }
}

fn main() {}
