// Hand-written W7 (f32 sum) for plain wgpu: a two-pass workgroup-tree reduction, workgroup size 256. Pass one
// (reduce_blocks, one workgroup per 2048 values, capped by the device limit) takes a grid-stride share of the input per
// invocation, then tree-reduces the workgroup's lanes in shared memory into one partial; pass two (reduce_total, one
// workgroup) has each lane add its share of the partials, then tree-reduces the lanes into the single output word.
// Bindings: x (read), partials (read_write), out (read_write).
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

var<workgroup> tile: array<f32, 256>;

fn tree(lane: u32) {
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            tile[lane] = tile[lane] + tile[lane + s];
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256, 1, 1)
fn reduce_blocks(
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let n = arrayLength(&x);
    let stride = groups.x * 256u;
    var acc = f32(0);
    for (var i = group.x * 256u + local.x; i < n; i = i + stride) {
        acc = acc + x[i];
    }
    tile[local.x] = acc;
    workgroupBarrier();
    tree(local.x);
    if (local.x == 0u && group.x < arrayLength(&partials)) {
        partials[group.x] = tile[0];
    }
}

@compute @workgroup_size(256, 1, 1)
fn reduce_total(@builtin(local_invocation_id) local: vec3<u32>) {
    let count = arrayLength(&partials);
    var acc = f32(0);
    for (var i = local.x; i < count; i = i + 256u) {
        acc = acc + partials[i];
    }
    tile[local.x] = acc;
    workgroupBarrier();
    tree(local.x);
    if (local.x == 0u && 0u < arrayLength(&out)) {
        out[0u] = tile[0];
    }
}
