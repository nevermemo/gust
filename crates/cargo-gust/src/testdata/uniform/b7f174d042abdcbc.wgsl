@binding(4) @group(0) var<storage, read_write> out_0 : array<vec4<f32>>;

@binding(5) @group(0) var<storage, read_write> flat_0 : array<vec2<f32>>;

struct _MatrixStorage_float4x4_ColMajorstd430_0
{
    @align(16) data_0 : array<vec4<f32>, i32(4)>,
};

@binding(0) @group(0) var<storage, read> mats_0 : array<_MatrixStorage_float4x4_ColMajorstd430_0>;

@binding(3) @group(0) var<storage, read> pts_0 : array<vec4<f32>>;

struct Xform_std430_0
{
    @align(16) m_0 : _MatrixStorage_float4x4_ColMajorstd430_0,
    @align(16) tint_0 : vec4<f32>,
};

@binding(2) @group(0) var<storage, read> xs_0 : array<Xform_std430_0>;

struct _MatrixStorage_float2x2_ColMajorstd430_0
{
    @align(8) data_1 : array<vec2<f32>, i32(2)>,
};

@binding(1) @group(0) var<storage, read> small_0 : array<_MatrixStorage_float2x2_ColMajorstd430_0>;

@compute
@workgroup_size(64, 1, 1)
fn gpu_matrices_run(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 16);
    var _gpu_len_out_0 : u32 = _S1.x;
    var _S2 : vec2<u32> = vec2<u32>(arrayLength(&flat_0), 8);
    var _gpu_len_flat_0 : u32 = _S2.x;
    var i_0 : u32 = id_0.x;
    if((i_0 < _gpu_len_out_0))
    {
        var _S3 : Xform_std430_0 = xs_0[i32(0)];
        out_0[i_0] = (((pts_0[i_0]) * (mat4x4<f32>(mats_0[i32(0)].data_0[i32(0)][i32(0)], mats_0[i32(0)].data_0[i32(1)][i32(0)], mats_0[i32(0)].data_0[i32(2)][i32(0)], mats_0[i32(0)].data_0[i32(3)][i32(0)], mats_0[i32(0)].data_0[i32(0)][i32(1)], mats_0[i32(0)].data_0[i32(1)][i32(1)], mats_0[i32(0)].data_0[i32(2)][i32(1)], mats_0[i32(0)].data_0[i32(3)][i32(1)], mats_0[i32(0)].data_0[i32(0)][i32(2)], mats_0[i32(0)].data_0[i32(1)][i32(2)], mats_0[i32(0)].data_0[i32(2)][i32(2)], mats_0[i32(0)].data_0[i32(3)][i32(2)], mats_0[i32(0)].data_0[i32(0)][i32(3)], mats_0[i32(0)].data_0[i32(1)][i32(3)], mats_0[i32(0)].data_0[i32(2)][i32(3)], mats_0[i32(0)].data_0[i32(3)][i32(3)])))) + (((xs_0[i32(0)].tint_0) * (mat4x4<f32>(_S3.m_0.data_0[i32(0)][i32(0)], _S3.m_0.data_0[i32(1)][i32(0)], _S3.m_0.data_0[i32(2)][i32(0)], _S3.m_0.data_0[i32(3)][i32(0)], _S3.m_0.data_0[i32(0)][i32(1)], _S3.m_0.data_0[i32(1)][i32(1)], _S3.m_0.data_0[i32(2)][i32(1)], _S3.m_0.data_0[i32(3)][i32(1)], _S3.m_0.data_0[i32(0)][i32(2)], _S3.m_0.data_0[i32(1)][i32(2)], _S3.m_0.data_0[i32(2)][i32(2)], _S3.m_0.data_0[i32(3)][i32(2)], _S3.m_0.data_0[i32(0)][i32(3)], _S3.m_0.data_0[i32(1)][i32(3)], _S3.m_0.data_0[i32(2)][i32(3)], _S3.m_0.data_0[i32(3)][i32(3)]))));
    }
    if((i_0 < _gpu_len_flat_0))
    {
        flat_0[i_0] = (((vec2<f32>(1.0f, 2.0f)) * (mat2x2<f32>(small_0[i32(0)].data_1[i32(0)][i32(0)], small_0[i32(0)].data_1[i32(1)][i32(0)], small_0[i32(0)].data_1[i32(0)][i32(1)], small_0[i32(0)].data_1[i32(1)][i32(1)]))));
    }
    return;
}

