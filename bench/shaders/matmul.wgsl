// Hand-written W9 (naive matmul) for plain wgpu. Same bindings, guards and loop as the GUST kernel
// (bench/gust-bench/src/workloads/matmul.rs): dims (one word, the side N) binding 0, A binding 1, B binding 2, C binding 3,
// one output entry per invocation, a 16 by 16 workgroup, a 2-D dispatch of N by N invocations, k ascending.
@group(0) @binding(0) var<storage, read> dims: array<u32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (0u < arrayLength(&dims)) {
        let n = dims[0u];
        let row = id.y;
        let col = id.x;
        if (row < n && col < n && row * n + col < arrayLength(&c)) {
            var acc = 0.0;
            for (var k = 0u; k < n; k++) {
                acc = acc + a[row * n + k] * b[k * n + col];
            }
            c[row * n + col] = acc;
        }
    }
}
