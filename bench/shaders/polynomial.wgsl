// Hand-written W3 (polynomial, Horner) for plain wgpu. Same bindings, guards, workgroup size and loop shape as the GUST
// kernel (bench/gust-bench/src/workloads/polynomial.rs): the coefficients are a read-only storage buffer (binding 0,
// the degree is its length minus one), x is binding 1, the output binding 2. The loop is bounded by 64 and leaves at
// the degree, so the one shader serves degree 4, 16 and 64.
@group(0) @binding(0) var<storage, read> coeffs: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&out) && i < arrayLength(&x) && 1u < arrayLength(&coeffs)) {
        let degree = arrayLength(&coeffs) - 1u;
        let v = x[i];
        var acc = coeffs[degree];
        for (var k = 0u; k < 64u; k++) {
            if (k >= degree) {
                break;
            }
            acc = acc * v + coeffs[degree - 1u - k];
        }
        out[i] = acc;
    }
}
