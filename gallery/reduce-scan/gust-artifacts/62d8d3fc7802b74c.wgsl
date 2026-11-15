@binding(0) @group(0) var<storage, read> input_0 : array<u32>;

@binding(1) @group(0) var<storage, read_write> out_0 : array<u32>;

@binding(2) @group(0) var<storage, read_write> totals_0 : array<u32>;

var<workgroup> scratch_0 : array<u32, i32(64)>;

@compute
@workgroup_size(64, 1, 1)
fn gpu_blocks_tile_scan(@builtin(global_invocation_id) _id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&input_0), 4);
    var _gpu_len_input_0 : u32 = _S1.x;
    var _S2 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 4);
    var _gpu_len_out_0 : u32 = _S2.x;
    var _S3 : vec2<u32> = vec2<u32>(arrayLength(&totals_0), 4);
    var _S4 : u32 = _id_0.x;
    var _S5 : u32 = _S4 % u32(64);
    var _S6 : u32 = _S4 / u32(64);
    var i_0 : u32 = _S6 * u32(64) + _S5;
    var value_0 : u32;
    if((i_0 < _gpu_len_input_0))
    {
        value_0 = input_0[i_0];
    }
    else
    {
        value_0 = u32(0);
    }
    scratch_0[_S5] = value_0;
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
        var offset_0 : u32 = (u32(1) << (round_0));
        var left_0 : u32;
        if((_S5 >= offset_0))
        {
            left_0 = scratch_0[_S5 - offset_0];
        }
        else
        {
            left_0 = u32(0);
        }
        workgroupBarrier();
        scratch_0[_S5] = scratch_0[_S5] + left_0;
        workgroupBarrier();
        round_0 = round_0 + u32(1);
    }
    if((i_0 < _gpu_len_out_0))
    {
        out_0[i_0] = scratch_0[_S5];
    }
    if(_S5 == u32(63))
    {
        totals_0[_S6] = scratch_0[u32(63)];
    }
    return;
}

