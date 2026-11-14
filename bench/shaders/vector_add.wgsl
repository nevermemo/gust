// Hand-written W1 (vector-add) for plain wgpu. Same bindings, guard and workgroup size as the GUST kernel
// (bench/gust-bench/src/workloads/vector_add.rs): bindings 0 and 1 read-only, binding 2 read-write, one invocation per
// element, the bound taken from the output's own length.
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&out)) {
        out[i] = a[i] + b[i];
    }
}
