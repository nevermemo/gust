@binding(0) @group(0) var<storage, read> keys_0 : array<u32>;

@binding(1) @group(0) var<storage, read_write> bins_0 : array<atomic<u32>>;

fn _gust_rem_0( a_0 : u32,  b_0 : u32) -> u32
{
    var _S1 : u32 = a_0 % b_0;
    return _S1;
}

@compute
@workgroup_size(64, 1, 1)
fn gpu_histogram_count(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S2 : vec2<u32> = vec2<u32>(arrayLength(&keys_0), 4);
    var _gpu_len_keys_0 : u32 = _S2.x;
    var _S3 : vec2<u32> = vec2<u32>(arrayLength(&bins_0), 4);
    var _gpu_len_bins_0 : u32 = _S3.x;
    var i_0 : u32 = id_0.x;
    if((i_0 < _gpu_len_keys_0))
    {
        var bin_0 : u32 = _gust_rem_0(keys_0[i_0], _gpu_len_bins_0);
        var _S4 : u32 = atomicAdd(&(bins_0[bin_0]), u32(1));
    }
    return;
}

