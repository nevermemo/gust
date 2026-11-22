struct _GustLaunchBase_std140_0
{
    @align(16) x_0 : u32,
    @align(4) y_0 : u32,
    @align(8) z_0 : u32,
    @align(4) pad_0 : u32,
};

@binding(2) @group(0) var<uniform> _gust_launch_base_0 : _GustLaunchBase_std140_0;
@binding(0) @group(0) var<storage, read> keys_0 : array<u32>;

@binding(1) @group(0) var<storage, read_write> bins_0 : array<atomic<u32>>;

@compute
@workgroup_size(256, 1, 1)
fn gpu_kernel_histogram(@builtin(global_invocation_id) _gust_launch_id_0 : vec3<u32>)
{
    var id_0 : vec3<u32> = _gust_launch_id_0 + vec3<u32>(_gust_launch_base_0.x_0, _gust_launch_base_0.y_0, _gust_launch_base_0.z_0);
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&keys_0), 4);
    var _gpu_len_keys_0 : u32 = _S1.x;
    var _S2 : vec2<u32> = vec2<u32>(arrayLength(&bins_0), 4);
    var _S3 : u32 = _S2.x;
    var i_0 : u32 = id_0.x;
    var _S4 : u32 = i_0 / u32(256) * u32(1024) + i_0 % u32(256);
    var k_0 : u32 = u32(0);
    loop
    {
        if((k_0 < u32(4)))
        {
        }
        else
        {
            break;
        }
        var index_0 : u32 = _S4 + k_0 * u32(256);
        if((index_0 >= _gpu_len_keys_0))
        {
            break;
        }
        var bin_0 : u32 = keys_0[index_0];
        if((bin_0 < _S3))
        {
            var _S5 : u32 = atomicAdd(&(bins_0[bin_0]), u32(1));
        }
        k_0 = k_0 + u32(1);
    }
    return;
}

