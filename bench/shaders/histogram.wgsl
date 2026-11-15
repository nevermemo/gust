// Hand-written W4 (histogram) for plain wgpu. Same bindings, guards and key assignment as the GUST kernel
// (bench/gust-bench/src/workloads/histogram.rs): the keys are binding 0, the bins binding 1 (u32 atomics). A workgroup
// owns a block of 256 * 4 contiguous keys; each lane takes four of them, strided by 256.
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&keys);
    let first = (i / 256u) * 1024u + i % 256u;
    for (var k = 0u; k < 4u; k++) {
        let index = first + k * 256u;
        if (index >= n) {
            break;
        }
        let bin = keys[index];
        if (bin < arrayLength(&bins)) {
            atomicAdd(&bins[bin], 1u);
        }
    }
}
