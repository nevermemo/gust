struct _GustLaunchBase_std140_0
{
    @align(16) x_0 : u32,
    @align(4) y_0 : u32,
    @align(8) z_0 : u32,
    @align(4) pad_0 : u32,
};

@binding(3) @group(0) var<uniform> _gust_launch_base_0 : _GustLaunchBase_std140_0;
@binding(2) @group(0) var<storage, read_write> out_0 : array<f32>;

@binding(0) @group(0) var<storage, read> amplified_0 : array<f32>;

@binding(1) @group(0) var<storage, read> bias_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn gpu_kernel_shift(@builtin(global_invocation_id) _gust_launch_id_0 : vec3<u32>)
{
    var id_0 : vec3<u32> = _gust_launch_id_0 + vec3<u32>(_gust_launch_base_0.x_0, _gust_launch_base_0.y_0, _gust_launch_base_0.z_0);
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 4);
    var i_0 : u32 = id_0.x;
    if((i_0 < (_S1.x)))
    {
        out_0[i_0] = amplified_0[i_0] + bias_0[i_0];
    }
    return;
}

