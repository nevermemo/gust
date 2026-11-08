// A route tag is segments of `[a-z0-9_]+` joined by single `.`, with no `__` (the variable's separator) and no segment
// that begins or ends with `_` (m8.r1, D66): `a__b` and `a.b` would name one route variable. GUST-E0748, reported at the
// string literal (line 5 of this file).

#[gust::gpu(route = "physics__broadphase")]
mod kernels {
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

fn main() {}
