// Hand-written W6 (staged pipeline) for plain wgpu: the six stages of `examples/signal-pipeline`'s `transform` as six
// entry points of one module, the same kernels as `bench/gust-bench/src/workloads/signal_pipeline.rs` (same guard, same
// 256-wide workgroup, same operation per stage). Every entry point has the one three-binding layout the host side builds:
// bindings 0 and 1 read-only inputs, binding 2 the read-write output of the stage. A stage with one input (energy) binds
// the same buffer to bindings 0 and 1; the host side binds each stage's own buffers in its own bind group and runs the six
// as six compute passes in the order below, so the passes' storage writes are ordered by wgpu between passes.
//
//   center   : out = a - b          (signal, baseline)
//   amplify  : out = a * b          (centered, gain)
//   shift    : out = a + b          (amplified, bias)
//   rescale  : out = a / b          (biased, scale)
//   energy   : out = a * a          (normalized)
//   combine  : out = a + 0.25 * b   (energy, normalized)
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn center(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&out)) {
        out[i] = a[i] - b[i];
    }
}

@compute @workgroup_size(256, 1, 1)
fn amplify(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&out)) {
        out[i] = a[i] * b[i];
    }
}

@compute @workgroup_size(256, 1, 1)
fn shift(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&out)) {
        out[i] = a[i] + b[i];
    }
}

@compute @workgroup_size(256, 1, 1)
fn rescale(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&out)) {
        out[i] = a[i] / b[i];
    }
}

@compute @workgroup_size(256, 1, 1)
fn energy(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&out)) {
        out[i] = a[i] * a[i];
    }
}

@compute @workgroup_size(256, 1, 1)
fn combine(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&out)) {
        out[i] = a[i] + (0.25 * b[i]);
    }
}
