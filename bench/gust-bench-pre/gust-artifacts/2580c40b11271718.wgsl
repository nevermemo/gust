@binding(0) @group(0) var<storage, read> coeffs_0 : array<f32>;

@binding(1) @group(0) var<storage, read> x_0 : array<f32>;

@binding(2) @group(0) var<storage, read_write> out_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn gpu_kernel_horner(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&coeffs_0), 4);
    var _gpu_len_coeffs_0 : u32 = _S1.x;
    var _S2 : vec2<u32> = vec2<u32>(arrayLength(&x_0), 4);
    var _gpu_len_x_0 : u32 = _S2.x;
    var _S3 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 4);
    var i_0 : u32 = id_0.x;
    var _S4 : bool;
    if((i_0 < (_S3.x)))
    {
        _S4 = (i_0 < _gpu_len_x_0);
    }
    else
    {
        _S4 = false;
    }
    if(_S4)
    {
        _S4 = (u32(1) < _gpu_len_coeffs_0);
    }
    else
    {
        _S4 = false;
    }
    if(_S4)
    {
        var degree_0 : u32 = _gpu_len_coeffs_0 - u32(1);
        var _S5 : f32 = x_0[i_0];
        var acc_0 : f32 = coeffs_0[degree_0];
        var k_0 : u32 = u32(0);
        loop
        {
            if((k_0 < u32(64)))
            {
            }
            else
            {
                break;
            }
            if((k_0 >= degree_0))
            {
                break;
            }
            var _S6 : f32 = acc_0 * _S5 + coeffs_0[degree_0 - u32(1) - k_0];
            var _S7 : u32 = k_0 + u32(1);
            acc_0 = _S6;
            k_0 = _S7;
        }
        out_0[i_0] = acc_0;
    }
    return;
}

