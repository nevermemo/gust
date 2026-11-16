@binding(0) @group(0) var<storage, read> dims_0 : array<u32>;

@binding(3) @group(0) var<storage, read_write> c_0 : array<f32>;

@binding(1) @group(0) var<storage, read> a_0 : array<f32>;

@binding(2) @group(0) var<storage, read> b_0 : array<f32>;

@compute
@workgroup_size(16, 16, 1)
fn gpu_kernel_matmul(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&dims_0), 4);
    var _gpu_len_dims_0 : u32 = _S1.x;
    var _S2 : vec2<u32> = vec2<u32>(arrayLength(&c_0), 4);
    var _gpu_len_c_0 : u32 = _S2.x;
    if((u32(0) < _gpu_len_dims_0))
    {
        var n_0 : u32 = dims_0[u32(0)];
        var row_0 : u32 = id_0.y;
        var col_0 : u32 = id_0.x;
        var _S3 : bool;
        if((row_0 < n_0))
        {
            _S3 = (col_0 < n_0);
        }
        else
        {
            _S3 = false;
        }
        if(_S3)
        {
            _S3 = ((row_0 * n_0 + col_0) < _gpu_len_c_0);
        }
        else
        {
            _S3 = false;
        }
        if(_S3)
        {
            var acc_0 : f32 = 0.0f;
            var k_0 : u32 = u32(0);
            loop
            {
                if((k_0 < n_0))
                {
                }
                else
                {
                    break;
                }
                var _S4 : f32 = acc_0 + a_0[row_0 * n_0 + k_0] * b_0[k_0 * n_0 + col_0];
                var _S5 : u32 = k_0 + u32(1);
                acc_0 = _S4;
                k_0 = _S5;
            }
            c_0[row_0 * n_0 + col_0] = acc_0;
        }
    }
    return;
}

