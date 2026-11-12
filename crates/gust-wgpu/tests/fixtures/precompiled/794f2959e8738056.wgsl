struct Bag_std430_0
{
    @align(4) items_0 : array<u32, i32(4)>,
    @align(4) count_0 : u32,
};

@binding(0) @group(0) var<storage, read_write> bags_0 : array<Bag_std430_0>;

@compute
@workgroup_size(64, 1, 1)
fn gpu_pre_arrays_field(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&bags_0), 20);
    var i_0 : u32 = id_0.x;
    if((i_0 < (_S1.x)))
    {
        bags_0[i_0].items_0[i32(1)] = bags_0[i_0].items_0[i32(1)] + u32(7);
        bags_0[i_0].count_0 = u32(4) + bags_0[i_0].items_0[i32(0)];
    }
    return;
}

