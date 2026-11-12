@binding(0) @group(0) var<storage, read_write> pairs_0 : array<array<vec2<f32>, i32(2)>>;

struct Pt_std430_0
{
    @align(4) x_0 : u32,
    @align(4) y_0 : f32,
};

struct _Array_std430_Pt2_0
{
    @align(4) data_0 : array<Pt_std430_0, i32(2)>,
};

struct Poly_std430_0
{
    @align(8) pts_0 : _Array_std430_Pt2_0,
    @align(8) tip_0 : array<vec2<f32>, i32(2)>,
    @align(8) n_0 : u32,
    @align(4) pad_0 : u32,
};

@binding(1) @group(0) var<storage, read_write> polys_0 : array<Poly_std430_0>;

@compute
@workgroup_size(64, 1, 1)
fn gpu_pre_arrays_shapes(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&pairs_0), 16);
    var _gpu_len_pairs_0 : u32 = _S1.x;
    var _S2 : vec2<u32> = vec2<u32>(arrayLength(&polys_0), 40);
    var i_0 : u32 = id_0.x;
    if((i_0 < _gpu_len_pairs_0))
    {
        var p_0 : array<vec2<f32>, i32(2)> = pairs_0[i_0];
        pairs_0[i_0][i32(0)] = pairs_0[i_0][i32(1)];
        pairs_0[i_0][i32(1)] = p_0[i32(0)];
        polys_0[i_0].pts_0.data_0[i32(1)].x_0 = polys_0[i_0].pts_0.data_0[i32(0)].x_0 + u32(1);
        polys_0[i_0].tip_0[i32(0)] = polys_0[i_0].tip_0[i32(1)];
        polys_0[i_0].n_0 = polys_0[i_0].pts_0.data_0[i32(1)].x_0;
    }
    return;
}

