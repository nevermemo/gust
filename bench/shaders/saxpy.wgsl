// Hand-written W2 (saxpy) for plain wgpu. The scalar is a one-element read-only storage buffer (binding 0), the same
// stand-in the GUST kernel uses until uniform kernels get a typed call (D40), so neither side gets a cheaper scalar
// path. Out of place: out = scale * x + y.
@group(0) @binding(0) var<storage, read> scale: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&out) && 0u < arrayLength(&scale)) {
        out[i] = scale[0u] * x[i] + y[i];
    }
}
