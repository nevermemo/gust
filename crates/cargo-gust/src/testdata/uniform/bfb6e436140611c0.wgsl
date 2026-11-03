@binding(2) @group(0) var<storage, read_write> out_0 : array<f32>;

@binding(1) @group(0) var<storage, read> src_0 : array<f32>;

struct Gain_std140_0
{
    @align(16) gain_0 : f32,
    @align(4) pad0_0 : f32,
    @align(8) pad1_0 : f32,
    @align(4) pad2_0 : f32,
};

@binding(0) @group(0) var<uniform> g_0 : Gain_std140_0;
@compute
@workgroup_size(64, 1, 1)
fn gpu_probes_scale(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 4);
    var i_0 : u32 = id_0.x;
    if((i_0 < (_S1.x)))
    {
        out_0[i_0] = src_0[i_0] * g_0.gain_0;
    }
    return;
}

