@binding(0) @group(0) var<storage, read> input_0 : array<u32>;

@binding(1) @group(0) var<storage, read_write> out_0 : array<u32>;

@compute
@workgroup_size(64, 1, 1)
fn gpu_smooth_blur(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&input_0), 4);
    var _gpu_len_input_0 : u32 = _S1.x;
    var _S2 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 4);
    var _gpu_len_out_0 : u32 = _S2.x;
    var i_0 : u32 = id_0.x;
    var _S3 : bool;
    if((i_0 < _gpu_len_input_0))
    {
        _S3 = (i_0 < _gpu_len_out_0);
    }
    else
    {
        _S3 = false;
    }
    if(_S3)
    {
        var left_0 : u32;
        if((i_0 > u32(0)))
        {
            left_0 = i_0 - u32(1);
        }
        else
        {
            left_0 = i_0;
        }
        var _S4 : u32 = i_0 + u32(1);
        var right_0 : u32;
        if((_S4 < _gpu_len_input_0))
        {
            right_0 = _S4;
        }
        else
        {
            right_0 = i_0;
        }
        out_0[i_0] = input_0[left_0] + u32(2) * input_0[i_0] + input_0[right_0];
    }
    return;
}

