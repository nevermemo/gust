@binding(3) @group(0) var<storage, read_write> _gust_task_out_0 : array<f32>;

struct _GustExtent_std140_0
{
    @align(16) _gust_ex_0 : u32,
    @align(4) _gust_ey_0 : u32,
    @align(8) _gust_ez_0 : u32,
    @align(4) _gust_pad0_0 : u32,
};

@binding(0) @group(0) var<uniform> _gust_ext_0 : _GustExtent_std140_0;
@binding(1) @group(0) var<storage, read> amplified_0 : array<f32>;

@binding(2) @group(0) var<storage, read> bias_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn gpu___gust_pf_26c739b3b029a8d1_run(@builtin(global_invocation_id) _gust_gid_0 : vec3<u32>)
{
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
        _gust_task_out_0[_S2] = amplified_0[_S2] + bias_0[_S2];
    }
    return;
}

