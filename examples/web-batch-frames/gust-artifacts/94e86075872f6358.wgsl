@binding(10) @group(0) var<storage, read_write> _gust_task_out_0 : array<u32>;

struct _GustExtent_std140_0
{
    @align(16) _gust_ex_0 : u32,
    @align(4) _gust_ey_0 : u32,
    @align(8) _gust_ez_0 : u32,
    @align(4) _gust_pad0_0 : u32,
};

@binding(0) @group(0) var<uniform> _gust_ext_0 : _GustExtent_std140_0;
@binding(1) @group(0) var<storage, read> a0_0 : array<u32>;

@binding(2) @group(0) var<storage, read> a1_0 : array<u32>;

@binding(3) @group(0) var<storage, read> a2_0 : array<u32>;

@binding(4) @group(0) var<storage, read> a3_0 : array<u32>;

@binding(5) @group(0) var<storage, read> a4_0 : array<u32>;

@binding(6) @group(0) var<storage, read> a5_0 : array<u32>;

@binding(7) @group(0) var<storage, read> a6_0 : array<u32>;

@binding(8) @group(0) var<storage, read> a7_0 : array<u32>;

@binding(9) @group(0) var<storage, read> a8_0 : array<u32>;

@compute
@workgroup_size(256, 1, 1)
fn gpu___gust_pf_f44aeb04f0d7ea25_run(@builtin(global_invocation_id) _gust_gid_0 : vec3<u32>)
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
        _gust_task_out_0[_S2] = a0_0[_S2] + a1_0[_S2] * u32(2) + a2_0[_S2] * u32(3) + a3_0[_S2] * u32(4) + a4_0[_S2] * u32(5) + a5_0[_S2] * u32(6) + a6_0[_S2] * u32(7) + a7_0[_S2] * u32(8) + a8_0[_S2] * u32(9);
    }
    return;
}

