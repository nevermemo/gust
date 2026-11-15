//! Compiles the four forms the facade-path option covers, with `gust` reachable only as `::inner::gust`:
//! the module form, a `route` module, a map function and `par_for!`. Nothing here is run.

/// The module form.
#[inner::gust::gpu(crate = ::inner::gust)]
pub mod plain {
    #[kernel]
    pub fn scale(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() && i < input.len() {
            out[i] = input[i] * 2.0;
        }
    }
}

/// A routed module, with the two arguments in the other order than the map function below.
#[inner::gust::gpu(route = "physics", crate = ::inner::gust)]
pub mod routed {
    #[kernel]
    pub fn scale(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() && i < input.len() {
            out[i] = input[i] * 2.0;
        }
    }
}

/// A map function.
#[inner::gust::gpu(crate = ::inner::gust)]
pub fn saxpy(a: f32, x: f32, y: f32) -> f32 {
    a * x + y
}

/// `par_for!` with the clause.
pub fn scale(values: &[f32], out: &mut [f32], by: f32) -> Result<(), inner::gust::map::Error> {
    let n = out.len() as u32;
    inner::gust::par_for!(crate(::inner::gust); i in 0..n, [values: &[f32], out: &mut [f32], by: f32] {
        out[i] = values[i] * by;
    })
}

/// The typed calls of the generated items are reachable too.
pub fn descriptors() -> [&'static str; 2] {
    [plain::scale::DESCRIPTOR.name, routed::scale::DESCRIPTOR.name]
}
