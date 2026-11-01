@binding(2) @group(0) var<storage, read_write> out_0 : array<f32>;

@binding(0) @group(0) var<storage, read> a_0 : array<f32>;

@binding(1) @group(0) var<storage, read> b_0 : array<f32>;

@compute
@workgroup_size(64, 1, 1)
fn gpu_vector_add_add(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 4);
    var i_0 : u32 = id_0.x;
    if(i_0 < (_S1.x))
    {
        out_0[i_0] = a_0[i_0] + b_0[i_0];
    }
    return;
}

