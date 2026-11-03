// The accepted forms next to the E0326 one (literal-let-rejected): a suffix, a type annotation, an annotated
// literal-only expression and a typed operand. This crate must compile, so the validator, the lowering and the
// preserved Rust all accept it.

#[gust::gpu]
mod sums {
    #[kernel]
    pub fn total(id: SV_DispatchThreadID, input: StructuredBuffer<uint>, mut output: RWStructuredBuffer<uint>) {
        let i = id.x;
        if i < input.len() && i < output.len() {
            let mut suffixed = 0u32;
            suffixed += input[i];
            let mut annotated: u32 = 0;
            annotated += input[i];
            let typed = 3 * input[i];
            let negative: i32 = -(1 + 2);
            output[i] = suffixed + annotated + typed + negative as u32;
        }
    }
}

fn main() {}
