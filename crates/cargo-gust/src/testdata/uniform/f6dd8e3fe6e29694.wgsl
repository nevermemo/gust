@binding(3) @group(0) var<storage, read_write> out_0 : array<f32>;

@binding(2) @group(0) var<storage, read> src_0 : array<f32>;

struct Inner_std140_0
{
    @align(16) count_0 : u32,
    @align(4) bias_0 : i32,
    @align(8) pad0_0 : f32,
    @align(4) pad1_0 : f32,
};

struct Outer_std140_0
{
    @align(16) scale_0 : f32,
    @align(4) pad0_1 : f32,
    @align(8) pad1_1 : f32,
    @align(4) pad2_0 : f32,
    @align(16) inner_0 : Inner_std140_0,
};

@binding(0) @group(0) var<uniform> o_0 : Outer_std140_0;
struct Gain_std140_0
{
    @align(16) gain_0 : f32,
    @align(4) pad0_2 : f32,
    @align(8) pad1_2 : f32,
    @align(4) pad2_1 : f32,
};

@binding(1) @group(0) var<uniform> g_0 : Gain_std140_0;
@compute
@workgroup_size(64, 1, 1)
fn gpu_probes_shifted(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 4);
    var i_0 : u32 = id_0.x;
    if((i_0 < (_S1.x)))
    {
        out_0[i_0] = src_0[i_0] * o_0.scale_0 * g_0.gain_0 + f32(o_0.inner_0.count_0) + f32(o_0.inner_0.bias_0);
    }
    return;
}

