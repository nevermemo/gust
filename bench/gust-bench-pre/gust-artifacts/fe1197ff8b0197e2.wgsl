@binding(0) @group(0) var<storage, read> scale_0 : array<f32>;

@binding(3) @group(0) var<storage, read_write> out_0 : array<f32>;

@binding(1) @group(0) var<storage, read> x_0 : array<f32>;

@binding(2) @group(0) var<storage, read> y_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn gpu_kernel_saxpy(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&scale_0), 4);
    var _gpu_len_scale_0 : u32 = _S1.x;
    var _S2 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 4);
    var i_0 : u32 = id_0.x;
    var _S3 : bool;
    if((i_0 < (_S2.x)))
    {
        _S3 = (u32(0) < _gpu_len_scale_0);
    }
    else
    {
        _S3 = false;
    }
    if(_S3)
    {
        out_0[i_0] = scale_0[u32(0)] * x_0[i_0] + y_0[i_0];
    }
    return;
}

