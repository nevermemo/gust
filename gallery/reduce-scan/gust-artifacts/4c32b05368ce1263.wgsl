struct _GustLaunchBase_std140_0
{
    @align(16) x_0 : u32,
    @align(4) y_0 : u32,
    @align(8) z_0 : u32,
    @align(4) pad_0 : u32,
};

@binding(2) @group(0) var<uniform> _gust_launch_base_0 : _GustLaunchBase_std140_0;
@binding(0) @group(0) var<storage, read> input_0 : array<u32>;

@binding(1) @group(0) var<storage, read_write> sums_0 : array<u32>;

var<workgroup> scratch_0 : array<u32, i32(64)>;

@compute
@workgroup_size(64, 1, 1)
fn gpu_blocks_tile_sum(@builtin(global_invocation_id) _gust_launch_id_0 : vec3<u32>)
{
    var _id_0 : vec3<u32> = _gust_launch_id_0 + vec3<u32>(_gust_launch_base_0.x_0, _gust_launch_base_0.y_0, _gust_launch_base_0.z_0);
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&input_0), 4);
    var _gpu_len_input_0 : u32 = _S1.x;
    var _S2 : vec2<u32> = vec2<u32>(arrayLength(&sums_0), 4);
    var _S3 : u32 = _id_0.x;
    var _S4 : u32 = _S3 % u32(64);
    var _S5 : u32 = _S3 / u32(64);
    var i_0 : u32 = _S5 * u32(64) + _S4;
    var value_0 : u32;
    if((i_0 < _gpu_len_input_0))
    {
        value_0 = input_0[i_0];
    }
    else
    {
        value_0 = u32(0);
    }
    scratch_0[_S4] = value_0;
    workgroupBarrier();
    var round_0 : u32 = u32(0);
    loop
    {
        if((round_0 < u32(6)))
        {
        }
        else
        {
            break;
        }
        var stride_0 : u32 = (u32(32) >> (round_0));
        if((_S4 < stride_0))
        {
            scratch_0[_S4] = scratch_0[_S4] + scratch_0[_S4 + stride_0];
        }
        workgroupBarrier();
        round_0 = round_0 + u32(1);
    }
    if(_S4 == u32(0))
    {
        sums_0[_S5] = scratch_0[u32(0)];
    }
    return;
}

