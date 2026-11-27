struct Row_std430_0
{
    @align(4) a_0 : u32,
    @align(4) b_0 : u32,
    @align(4) c_0 : u32,
};

@binding(4) @group(0) var<storage, read_write> _gust_task_out_0 : array<Row_std430_0>;

@binding(1) @group(0) var<storage, read> count_0 : array<u32>;

struct _GustExtent_std140_0
{
    @align(16) _gust_ex_0 : u32,
    @align(4) _gust_ey_0 : u32,
    @align(8) _gust_ez_0 : u32,
    @align(4) _gust_pad0_0 : u32,
};

@binding(0) @group(0) var<uniform> _gust_ext_0 : _GustExtent_std140_0;
@binding(3) @group(0) var<storage, read> rev_0 : array<u32>;

@binding(2) @group(0) var<storage, read> rows_0 : array<Row_std430_0>;

@compute
@workgroup_size(256, 1, 1)
fn gpu___gust_pf_adb60329cb1e015e_run(@builtin(global_invocation_id) _gust_gid_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&_gust_task_out_0), 12);
    var _S2 : u32 = _gust_gid_0.y * u32(16776960) + _gust_gid_0.x;
    if((_S2 < (min(count_0[u32(0)], _gust_ext_0._gust_ex_0))))
    {
        _gust_task_out_0[_S2] = rows_0[rev_0[_S2]];
    }
    return;
}

