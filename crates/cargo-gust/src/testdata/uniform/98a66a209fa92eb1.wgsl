@binding(2) @group(0) var<storage, read_write> out_0 : array<vec4<f32>>;

struct _MatrixStorage_float4x4_ColMajorstd140_0
{
    @align(16) data_0 : array<vec4<f32>, i32(4)>,
};

struct Light_std140_0
{
    @align(16) color_0 : vec4<f32>,
    @align(16) intensity_0 : f32,
    @align(4) falloff_0 : f32,
    @align(8) pad0_0 : f32,
    @align(4) pad1_0 : f32,
};

struct Frame_std140_0
{
    @align(16) view_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) tint_0 : vec4<f32>,
    @align(16) light_0 : Light_std140_0,
    @align(16) offset_0 : vec2<f32>,
    @align(8) scale_0 : f32,
    @align(4) count_0 : u32,
};

@binding(0) @group(0) var<uniform> frame_0 : Frame_std140_0;
@binding(1) @group(0) var<storage, read> points_0 : array<vec4<f32>>;

fn _gust_mul_0( a_0 : vec4<f32>,  b_0 : f32) -> vec4<f32>
{
    return a_0 * vec4<f32>(b_0);
}

fn shade_0( base_0 : vec4<f32>,  light_1 : ptr<function, Light_std140_0>) -> vec4<f32>
{
    return _gust_mul_0(base_0, (*light_1).intensity_0) + _gust_mul_0((*light_1).color_0, (*light_1).falloff_0);
}

fn _gust_mul_1( a_1 : f32,  b_1 : f32) -> f32
{
    return a_1 * b_1;
}

@compute
@workgroup_size(64, 1, 1)
fn gpu_uniforms_run(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 16);
    var i_0 : u32 = id_0.x;
    if((i_0 < (_S1.x)))
    {
        var moved_0 : vec4<f32> = (((points_0[i_0]) * (mat4x4<f32>(frame_0.view_0.data_0[i32(0)][i32(0)], frame_0.view_0.data_0[i32(1)][i32(0)], frame_0.view_0.data_0[i32(2)][i32(0)], frame_0.view_0.data_0[i32(3)][i32(0)], frame_0.view_0.data_0[i32(0)][i32(1)], frame_0.view_0.data_0[i32(1)][i32(1)], frame_0.view_0.data_0[i32(2)][i32(1)], frame_0.view_0.data_0[i32(3)][i32(1)], frame_0.view_0.data_0[i32(0)][i32(2)], frame_0.view_0.data_0[i32(1)][i32(2)], frame_0.view_0.data_0[i32(2)][i32(2)], frame_0.view_0.data_0[i32(3)][i32(2)], frame_0.view_0.data_0[i32(0)][i32(3)], frame_0.view_0.data_0[i32(1)][i32(3)], frame_0.view_0.data_0[i32(2)][i32(3)], frame_0.view_0.data_0[i32(3)][i32(3)])))) + frame_0.tint_0;
        var _S2 : Light_std140_0 = frame_0.light_0;
        var _S3 : vec4<f32> = shade_0(moved_0, &(_S2));
        out_0[i_0] = vec4<f32>(_S3.x + frame_0.offset_0.x, _S3.y + frame_0.offset_0.y, _gust_mul_1(_S3.z, frame_0.scale_0), _S3.w + f32(frame_0.count_0));
    }
    return;
}

