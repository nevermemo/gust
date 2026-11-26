struct Row_std430_0
{
    @align(4) a_0 : u32,
    @align(4) b_0 : u32,
    @align(4) c_0 : u32,
};

@binding(2) @group(0) var<storage, read_write> _gust_task_out_0 : array<Row_std430_0>;

struct _GustExtent_std140_0
{
    @align(16) _gust_ex_0 : u32,
    @align(4) _gust_ey_0 : u32,
    @align(8) _gust_ez_0 : u32,
    @align(4) _gust_pad0_0 : u32,
};

@binding(0) @group(0) var<uniform> _gust_ext_0 : _GustExtent_std140_0;
@binding(1) @group(0) var<storage, read> rows_0 : array<Row_std430_0>;

struct Row_0
{
     a_0 : u32,
     b_0 : u32,
     c_0 : u32,
};

@compute
@workgroup_size(256, 1, 1)
fn gpu___gust_pf_7b67cf42df546847_run(@builtin(global_invocation_id) _gust_gid_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&_gust_task_out_0), 12);
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
        var _gust_struct_1_0 : Row_0;
        _gust_struct_1_0.a_0 = rows_0[_S2].c_0;
        _gust_struct_1_0.b_0 = rows_0[_S2].a_0;
        _gust_struct_1_0.c_0 = rows_0[_S2].b_0;
        _gust_task_out_0[_S2].a_0 = _gust_struct_1_0.a_0;
        _gust_task_out_0[_S2].b_0 = _gust_struct_1_0.b_0;
        _gust_task_out_0[_S2].c_0 = _gust_struct_1_0.c_0;
    }
    return;
}

