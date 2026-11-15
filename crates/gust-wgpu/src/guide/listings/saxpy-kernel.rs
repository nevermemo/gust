#[gpu]
mod saxpy {
    /// The uniform: one value in a constant buffer, read by every invocation. A uniform is a struct of the module
    /// whose layout has no implicit padding and whose size is a multiple of 16 bytes, so the three `pad` fields are
    /// written out to round the 4-byte `a` up to 16.
    pub struct Params {
        pub a: f32,
        pub pad0: f32,
        pub pad1: f32,
        pub pad2: f32,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn apply(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        x: StructuredBuffer<float>,
        y: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = p.a * x[i] + y[i];
        }
    }
}
