struct _GustLaunchBase_std140_0
{
    @align(16) x_0 : u32,
    @align(4) y_0 : u32,
    @align(8) z_0 : u32,
    @align(4) pad_0 : u32,
};

@binding(3) @group(0) var<uniform> _gust_launch_base_0 : _GustLaunchBase_std140_0;
@binding(0) @group(0) var<storage, read> a_0 : array<f32>;

@binding(1) @group(0) var<storage, read> b_0 : array<f32>;

@binding(2) @group(0) var<storage, read_write> _gust_out_0 : array<f32>;

fn add_0( a_1 : f32,  b_1 : f32) -> f32
{
    return a_1 + b_1;
}

@compute
@workgroup_size(256, 1, 1)
fn gpu_add_map_kernel(@builtin(global_invocation_id) _gust_launch_id_0 : vec3<u32>)
{
    var _S1 : vec3<u32> = _gust_launch_id_0 + vec3<u32>(_gust_launch_base_0.x_0, _gust_launch_base_0.y_0, _gust_launch_base_0.z_0);
    var _S2 : vec2<u32> = vec2<u32>(arrayLength(&a_0), 4);
    var _S3 : u32 = _S2.x;
    var _S4 : vec2<u32> = vec2<u32>(arrayLength(&b_0), 4);
    var _S5 : u32 = _S4.x;
    var _S6 : vec2<u32> = vec2<u32>(arrayLength(&_gust_out_0), 4);
    var _gpu_len_gust_out_0 : u32 = _S6.x;
    var _gust_rounds_0 : u32 = _gpu_len_gust_out_0 / u32(16776960) + select(u32(0), u32(1), (_gpu_len_gust_out_0 % u32(16776960)) != u32(0));
    var _gust_round_0 : u32 = u32(0);
    loop
    {
        if((_gust_round_0 < _gust_rounds_0))
        {
        }
        else
        {
            break;
        }
        var _gust_j_0 : u32 = _S1.x + _gust_round_0 * u32(16776960);
        if((_gust_j_0 < _gpu_len_gust_out_0))
        {
            var _gust_i0_0 : u32;
            if(_S3 == u32(1))
            {
                _gust_i0_0 = u32(0);
            }
            else
            {
                _gust_i0_0 = _gust_j_0;
            }
            var _gust_i1_0 : u32;
            if(_S5 == u32(1))
            {
                _gust_i1_0 = u32(0);
            }
            else
            {
                _gust_i1_0 = _gust_j_0;
            }
            _gust_out_0[_gust_j_0] = add_0(a_0[_gust_i0_0], b_0[_gust_i1_0]);
        }
        _gust_round_0 = _gust_round_0 + u32(1);
    }
    return;
}

