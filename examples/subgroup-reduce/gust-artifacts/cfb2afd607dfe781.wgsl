@binding(0) @group(0) var<storage, read> data_0 : array<u32>;

@binding(1) @group(0) var<storage, read_write> out_0 : array<u32>;

@compute
@workgroup_size(64, 1, 1)
fn gpu_lanes_survey(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&data_0), 4);
    var _gpu_len_data_0 : u32 = _S1.x;
    var _S2 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 4);
    var _S3 : u32 = id_0.x;
    if((_S3 < _gpu_len_data_0))
    {
        var word_0 : u32 = u32(0);
        loop
        {
            if((word_0 < u32(4)))
            {
            }
            else
            {
                break;
            }
            out_0[_S3 * u32(4) + word_0] = u32(0);
            word_0 = word_0 + u32(1);
        }
    }
    return;
}

