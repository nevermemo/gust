@binding(1) @group(0) var<storage, read_write> out_0 : array<u32>;

@binding(0) @group(0) var<storage, read> offsets_0 : array<u32>;

@compute
@workgroup_size(64, 1, 1)
fn gpu_blocks_add_offsets(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 4);
    var i_0 : u32 = id_0.x;
    if((i_0 < (_S1.x)))
    {
        var tile_0 : u32 = i_0 / u32(64);
        if((tile_0 > u32(0)))
        {
            out_0[i_0] = out_0[i_0] + offsets_0[tile_0 - u32(1)];
        }
    }
    return;
}

