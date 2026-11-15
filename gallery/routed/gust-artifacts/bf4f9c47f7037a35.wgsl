@binding(0) @group(0) var<storage, read> raw_0 : array<u32>;

@binding(1) @group(0) var<storage, read_write> out_0 : array<u32>;

@compute
@workgroup_size(64, 1, 1)
fn gpu_tidy_saturate(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&raw_0), 4);
    var _gpu_len_raw_0 : u32 = _S1.x;
    var _S2 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 4);
    var _gpu_len_out_0 : u32 = _S2.x;
    var i_0 : u32 = id_0.x;
    var _S3 : bool;
    if((i_0 < _gpu_len_raw_0))
    {
        _S3 = (i_0 < _gpu_len_out_0);
    }
    else
    {
        _S3 = false;
    }
    if(_S3)
    {
        var v_0 : u32 = raw_0[i_0];
        var v_1 : u32;
        if((v_0 > u32(255)))
        {
            v_1 = u32(255);
        }
        else
        {
            v_1 = v_0;
        }
        out_0[i_0] = v_1;
    }
    return;
}

