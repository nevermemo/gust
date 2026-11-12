@binding(0) @group(0) var<storage, read_write> quads_0 : array<array<u32, i32(4)>>;

@compute
@workgroup_size(64, 1, 1)
fn gpu_pre_arrays_element(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&quads_0), 16);
    var i_0 : u32 = id_0.x;
    if((i_0 < (_S1.x)))
    {
        var q_0 : array<u32, i32(4)> = quads_0[i_0];
        quads_0[i_0][i32(0)] = quads_0[i_0][i32(3)] + u32(100);
        quads_0[i_0][i32(3)] = q_0[i32(0)] + q_0[i32(1)] + q_0[i32(2)];
    }
    return;
}

