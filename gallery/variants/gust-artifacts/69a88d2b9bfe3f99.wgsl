@binding(0) @group(0) var<storage, read> data_0 : array<u32>;

@binding(1) @group(0) var<storage, read_write> cells_0 : array<atomic<u32>>;

@compute
@workgroup_size(64, 1, 1)
fn gpu_stats_tally(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&data_0), 4);
    var _gpu_len_data_0 : u32 = _S1.x;
    var _S2 : vec2<u32> = vec2<u32>(arrayLength(&cells_0), 4);
    var _S3 : u32 = id_0.x;
    if((_S3 < _gpu_len_data_0))
    {
        var x_0 : u32 = data_0[_S3];
        var _S4 : u32 = atomicAdd(&(cells_0[u32(0)]), x_0);
        if((x_0 > u32(100)))
        {
            var _S5 : u32 = atomicAdd(&(cells_0[u32(1)]), u32(1));
        }
        var _S6 : u32 = atomicMax(&(cells_0[u32(2)]), x_0);
    }
    return;
}

