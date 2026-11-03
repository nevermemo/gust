@binding(0) @group(0) var<storage, read> x_0 : array<f32>;

@binding(1) @group(0) var<storage, read> k_0 : array<f32>;

@binding(2) @group(0) var<storage, read_write> _gust_out_0 : array<f32>;

fn curve_0( x_1 : f32,  k_1 : f32) -> f32
{
    return x_1 * x_1 * k_1 + x_1;
}

@compute
@workgroup_size(256, 1, 1)
fn gpu_curve_map_kernel(@builtin(global_invocation_id) _gust_gid_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&x_0), 4);
    var _S2 : u32 = _S1.x;
    var _S3 : vec2<u32> = vec2<u32>(arrayLength(&k_0), 4);
    var _S4 : u32 = _S3.x;
    var _S5 : vec2<u32> = vec2<u32>(arrayLength(&_gust_out_0), 4);
    var _gpu_len_gust_out_0 : u32 = _S5.x;
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
        var _gust_j_0 : u32 = _gust_gid_0.x + _gust_round_0 * u32(16776960);
        if((_gust_j_0 < _gpu_len_gust_out_0))
        {
            var _gust_i0_0 : u32;
            if(_S2 == u32(1))
            {
                _gust_i0_0 = u32(0);
            }
            else
            {
                _gust_i0_0 = _gust_j_0;
            }
            var _gust_i1_0 : u32;
            if(_S4 == u32(1))
            {
                _gust_i1_0 = u32(0);
            }
            else
            {
                _gust_i1_0 = _gust_j_0;
            }
            _gust_out_0[_gust_j_0] = curve_0(x_0[_gust_i0_0], k_0[_gust_i1_0]);
        }
        _gust_round_0 = _gust_round_0 + u32(1);
    }
    return;
}

