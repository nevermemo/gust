struct _GustLaunchBase_std140_0
{
    @align(16) x_0 : u32,
    @align(4) y_0 : u32,
    @align(8) z_0 : u32,
    @align(4) pad_0 : u32,
};

@binding(3) @group(0) var<uniform> _gust_launch_base_0 : _GustLaunchBase_std140_0;
@binding(2) @group(0) var<storage, read_write> _gust_task_out_0 : array<f32>;

struct _GustExtent_std140_0
{
    @align(16) _gust_ex_0 : u32,
    @align(4) _gust_ey_0 : u32,
    @align(8) _gust_ez_0 : u32,
    @align(4) _gust_pad0_0 : u32,
};

@binding(0) @group(0) var<uniform> _gust_ext_0 : _GustExtent_std140_0;
@binding(1) @group(0) var<storage, read> normalized_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn gpu___gust_pf_2cf42057d6726b19_run(@builtin(global_invocation_id) _gust_launch_id_0 : vec3<u32>)
{
    var _gust_gid_0 : vec3<u32> = _gust_launch_id_0 + vec3<u32>(_gust_launch_base_0.x_0, _gust_launch_base_0.y_0, _gust_launch_base_0.z_0);
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&_gust_task_out_0), 4);
    var _S2 : u32 = _gust_gid_0.x;
    var _S3 : bool;
    if((_S2 < (_gust_ext_0._gust_ex_0)))
    {
        _S3 = ((_gust_gid_0.y) < (_gust_ext_0._gust_ey_0));
    }
    else
    {
        _S3 = false;
    }
    if(_S3)
    {
        _S3 = ((_gust_gid_0.z) < (_gust_ext_0._gust_ez_0));
    }
    else
    {
        _S3 = false;
    }
    if(_S3)
    {
        _gust_task_out_0[_S2] = normalized_0[_S2] * normalized_0[_S2];
    }
    return;
}

