struct _GustLaunchBase_std140_0
{
    @align(16) x_0 : u32,
    @align(4) y_0 : u32,
    @align(8) z_0 : u32,
    @align(4) pad_0 : u32,
};

@binding(3) @group(0) var<uniform> _gust_launch_base_0 : _GustLaunchBase_std140_0;
@binding(2) @group(0) var<storage, read_write> _gust_args_0 : array<u32>;

@binding(1) @group(0) var<storage, read> count_0 : array<u32>;

struct _GustExtent_std140_0
{
    @align(16) _gust_ex_0 : u32,
    @align(4) _gust_ey_0 : u32,
    @align(8) _gust_ez_0 : u32,
    @align(4) _gust_pad0_0 : u32,
};

@binding(0) @group(0) var<uniform> _gust_ext_0 : _GustExtent_std140_0;
@compute
@workgroup_size(1, 1, 1)
fn gpu___gust_pf_c413d752e1648189_run_args(@builtin(global_invocation_id) _gust_launch_id_0 : vec3<u32>)
{
    var _gust_gid_0 : vec3<u32> = _gust_launch_id_0 + vec3<u32>(_gust_launch_base_0.x_0, _gust_launch_base_0.y_0, _gust_launch_base_0.z_0);
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&_gust_args_0), 4);
    if((_gust_gid_0.x) == u32(0))
    {
        var _S2 : u32 = min(count_0[u32(0)], _gust_ext_0._gust_ex_0);
        var groups_0 : u32 = _S2 / u32(256) + (_S2 % u32(256) + u32(255)) / u32(256);
        _gust_args_0[u32(0)] = min(groups_0, u32(65535));
        _gust_args_0[u32(1)] = groups_0 / u32(65535) + (groups_0 % u32(65535) + u32(65534)) / u32(65535);
        _gust_args_0[u32(2)] = u32(1);
    }
    return;
}

