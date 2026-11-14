@binding(3) @group(0) var<storage, read_write> out_0 : array<f32>;

struct Params_std140_0
{
    @align(16) a_0 : f32,
    @align(4) pad0_0 : f32,
    @align(8) pad1_0 : f32,
    @align(4) pad2_0 : f32,
};

@binding(0) @group(0) var<uniform> p_0 : Params_std140_0;
@binding(1) @group(0) var<storage, read> x_0 : array<f32>;

@binding(2) @group(0) var<storage, read> y_0 : array<f32>;

@compute
@workgroup_size(64, 1, 1)
fn gpu_saxpy_apply(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 4);
    var i_0 : u32 = id_0.x;
    if((i_0 < (_S1.x)))
    {
        out_0[i_0] = p_0.a_0 * x_0[i_0] + y_0[i_0];
    }
    return;
}

