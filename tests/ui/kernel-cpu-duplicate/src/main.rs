// The `cpu` option is stated once; a repeat is the macro's error E0710. A macro error carries no rustc code,
// so expected.txt says `code: none` and pins the macro's own message text.

#[gust::gpu]
mod kernels {
    #[kernel(cpu = "unsupported", cpu = "unsupported")]
    pub fn copy(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut output: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < input.len() && i < output.len() {
            output[i] = input[i];
        }
    }
}

fn main() {}
