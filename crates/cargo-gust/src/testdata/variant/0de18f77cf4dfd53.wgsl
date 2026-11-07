@binding(1) @group(0) var<storage, read_write> out_0 : array<u32>;

@binding(0) @group(0) var<storage, read> data_0 : array<u32>;

@compute
@workgroup_size(64, 1, 1)
fn gpu_gated_probe_total(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 4);
    var _S2 : u32 = id_0.x;
    if((_S2 < (_S1.x)))
    {
        out_0[_S2] = data_0[_S2] + u32(1);
    }
    return;
}

