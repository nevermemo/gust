// The only value of the `cpu` option is `"unsupported"` (it opts a kernel out of its CPU twin). Any other
// string is the macro's error E0709. A macro error carries no rustc code, so expected.txt says `code: none`
// and pins the macro's own message text.

#[gust::gpu]
mod kernels {
    #[kernel(cpu = "off")]
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
