@binding(5) @group(0) var<storage, read_write> moved_0 : array<vec4<f32>>;

@binding(6) @group(0) var<storage, read_write> sums_0 : array<f32>;

@binding(7) @group(0) var<storage, read_write> cells_0 : array<vec2<u32>>;

@binding(8) @group(0) var<storage, read_write> velocities_0 : array<vec2<f32>>;

@binding(9) @group(0) var<storage, read_write> signed_out_0 : array<vec4<i32>>;

@binding(0) @group(0) var<storage, read> points_0 : array<vec4<f32>>;

@binding(2) @group(0) var<storage, read> counts_0 : array<vec2<u32>>;

struct Particle_std430_0
{
    @align(8) position_0 : vec2<f32>,
    @align(8) velocity_0 : vec2<f32>,
    @align(8) mass_0 : f32,
    @align(4) tag_0 : u32,
};

@binding(1) @group(0) var<storage, read> particles_0 : array<Particle_std430_0>;

@binding(3) @group(0) var<storage, read> signed_in_0 : array<vec4<i32>>;

@binding(4) @group(0) var<storage, read> wide_in_0 : array<vec4<u32>>;

fn tint_0( color_0 : vec3<f32>,  amount_0 : f32) -> vec3<f32>
{
    return color_0 + vec3<f32>(amount_0);
}

fn _gust_dot_0( a_0 : vec4<f32>,  b_0 : vec4<f32>) -> f32
{
    return dot(a_0, b_0);
}

fn _gust_dot_1( a_1 : vec2<f32>,  b_1 : vec2<f32>) -> f32
{
    return dot(a_1, b_1);
}

fn tilt_0( direction_0 : vec3<f32>) -> vec3<f32>
{
    return (((direction_0) * (mat3x3<f32>(vec3<f32>(1.0f, 0.0f, 0.0f), vec3<f32>(0.0f, 1.0f, 0.0f), vec3<f32>(0.0f, 0.0f, 1.0f)))));
}

fn _gust_mul_0( a_2 : f32,  b_2 : mat4x4<f32>) -> mat4x4<f32>
{
    var _S1 : mat4x4<f32> = mat4x4<f32>(a_2, a_2, a_2, a_2, a_2, a_2, a_2, a_2, a_2, a_2, a_2, a_2, a_2, a_2, a_2, a_2);
    return mat4x4<f32>(_S1[0] * b_2[0], _S1[1] * b_2[1], _S1[2] * b_2[2], _S1[3] * b_2[3]);
}

fn _gust_mul_1( a_3 : mat4x4<f32>,  b_3 : mat4x4<f32>) -> mat4x4<f32>
{
    return (((b_3) * (a_3)));
}

fn _gust_mul_2( a_4 : mat4x4<f32>,  b_4 : vec4<f32>) -> vec4<f32>
{
    return (((b_4) * (a_4)));
}

fn _gust_mul_3( a_5 : vec4<f32>,  b_5 : f32) -> vec4<f32>
{
    return a_5 * vec4<f32>(b_5);
}

fn _gust_mul_4( a_6 : vec4<f32>,  b_6 : vec4<f32>) -> vec4<f32>
{
    return a_6 * b_6;
}

fn _gust_mul_5( a_7 : mat4x4<f32>,  b_7 : f32) -> mat4x4<f32>
{
    var _S2 : mat4x4<f32> = mat4x4<f32>(b_7, b_7, b_7, b_7, b_7, b_7, b_7, b_7, b_7, b_7, b_7, b_7, b_7, b_7, b_7, b_7);
    return mat4x4<f32>(a_7[0] * _S2[0], a_7[1] * _S2[1], a_7[2] * _S2[2], a_7[3] * _S2[3]);
}

fn products_0( m_0 : mat4x4<f32>,  v_0 : vec4<f32>,  s_0 : f32) -> vec4<f32>
{
    return _gust_mul_4(_gust_mul_3(_gust_mul_2(_gust_mul_1(_gust_mul_0(s_0, m_0), m_0), v_0), s_0), v_0) + _gust_mul_2(_gust_mul_5(m_0, s_0), v_0);
}

fn _gust_mul_6( a_8 : vec2<u32>,  b_8 : u32) -> vec2<u32>
{
    return a_8 * vec2<u32>(b_8);
}

fn _gust_neg_0( a_9 : f32) -> f32
{
    return - a_9;
}

fn spin_0( angle_0 : vec2<f32>,  point_0 : vec2<f32>) -> vec2<f32>
{
    var _S3 : f32 = angle_0.x;
    var _S4 : f32 = angle_0.y;
    return (((point_0) * ((((transpose(mat2x2<f32>(vec2<f32>(_S3, _S4), vec2<f32>(_gust_neg_0(_S4), _S3)))) * (mat2x2<f32>(vec2<f32>(1.0f, 0.0f), vec2<f32>(0.0f, 1.0f))))))));
}

fn integer_mix_0( wide_0 : vec3<u32>,  signed_0 : vec4<i32>) -> vec4<i32>
{
    return vec4<i32>(i32(wide_0.z), i32(wide_0.x), signed_0.wzyx.x, signed_0.xy.y);
}

fn _gust_dot_2( a_10 : vec2<i32>,  b_9 : vec2<i32>) -> i32
{
    return a_10.x * b_9.x + a_10.y * b_9.y;
}

fn _gust_dot_3( a_11 : vec4<i32>,  b_10 : vec4<i32>) -> i32
{
    return a_11.x * b_10.x + a_11.y * b_10.y + a_11.z * b_10.z + a_11.w * b_10.w;
}

fn _gust_dot_4( a_12 : vec2<u32>,  b_11 : vec2<u32>) -> u32
{
    return a_12.x * b_11.x + a_12.y * b_11.y;
}

fn _gust_dot_5( a_13 : vec4<u32>,  b_12 : vec4<u32>) -> u32
{
    return a_13.x * b_12.x + a_13.y * b_12.y + a_13.z * b_12.z + a_13.w * b_12.w;
}

fn integer_dots_0( a_14 : vec2<i32>,  b_13 : vec4<i32>,  c_0 : vec2<u32>,  d_0 : vec4<u32>) -> i32
{
    return _gust_dot_2(a_14, vec2<i32>(i32(3), i32(-2))) + _gust_dot_3(b_13, vec4<i32>(i32(1))) + i32(_gust_dot_4(c_0, vec2<u32>(vec2<i32>(i32(2)))) + _gust_dot_5(d_0, vec4<u32>(vec4<i32>(i32(1)))));
}

fn matrices_0( v2_0 : vec2<f32>,  v3_0 : vec3<f32>) -> vec3<f32>
{
    var rows2_0 : mat2x2<f32> = mat2x2<f32>(v2_0, v2_0.yx);
    var _S5 : vec3<f32> = v3_0.yzx;
    return ((((((v2_0) * ((((rows2_0) * (transpose(rows2_0))))))).xyx) * ((((mat3x3<f32>(v3_0, _S5, v3_0.zxy)) * (transpose(transpose(mat3x3<f32>(v3_0, v3_0.zyx, _S5)))))))));
}

@compute
@workgroup_size(64, 1, 1)
fn gpu_vectors_run(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S6 : vec2<u32> = vec2<u32>(arrayLength(&moved_0), 16);
    var _gpu_len_moved_0 : u32 = _S6.x;
    var _S7 : vec2<u32> = vec2<u32>(arrayLength(&sums_0), 4);
    var _S8 : vec2<u32> = vec2<u32>(arrayLength(&cells_0), 8);
    var _S9 : vec2<u32> = vec2<u32>(arrayLength(&velocities_0), 8);
    var _S10 : vec2<u32> = vec2<u32>(arrayLength(&signed_out_0), 16);
    var i_0 : u32 = id_0.x;
    if((i_0 < _gpu_len_moved_0))
    {
        var p_0 : vec4<f32> = points_0[i_0];
        var rgb_0 : vec3<f32> = tint_0(p_0.xyz, 0.5f);
        var _S11 : f32 = rgb_0.x;
        var _S12 : f32 = rgb_0.y;
        var widened_0 : vec4<f32> = vec4<f32>(_S11, _S12, rgb_0.z, p_0.wzyx.x);
        sums_0[i_0] = _gust_dot_0(p_0, vec4<f32>(p_0.w, p_0.z, p_0.y, p_0.x)) + _gust_dot_1(vec2<f32>(_S11, _S12), vec2<f32>(1.0f)) + tilt_0(rgb_0).z;
        const _S13 : vec4<f32> = vec4<f32>(1.0f, 0.0f, 0.0f, 0.0f);
        const _S14 : vec4<f32> = vec4<f32>(0.0f, 1.0f, 0.0f, 0.0f);
        const _S15 : vec4<f32> = vec4<f32>(0.0f, 0.0f, 1.0f, 0.0f);
        const _S16 : vec4<f32> = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
        var both_0 : mat4x4<f32> = (((mat4x4<f32>(vec4<f32>(2.0f, 0.0f, 0.0f, 0.0f), vec4<f32>(0.0f, 2.0f, 0.0f, 0.0f), vec4<f32>(0.0f, 0.0f, 2.0f, 0.0f), _S16)) * (transpose(transpose(mat4x4<f32>(_S13, _S14, _S15, vec4<f32>(3.0f, 4.0f, 5.0f, 1.0f)))))));
        moved_0[i_0] = products_0(both_0, ((((((widened_0) * (mat4x4<f32>(_S13, _S14, _S15, _S16))))) * (both_0))), 0.5f);
        var n_0 : vec2<u32> = counts_0[i_0];
        var _gust_struct_10_0 : vec2<u32>;
        _gust_struct_10_0[i32(0)] = n_0.x + i_0;
        _gust_struct_10_0[i32(1)] = n_0.y;
        cells_0[i_0] = _gust_mul_6(_gust_struct_10_0, u32(2)).yx;
        var part_0 : Particle_std430_0 = particles_0[i_0];
        velocities_0[i_0] = spin_0(part_0.velocity_0, part_0.position_0);
        var signed_1 : vec4<i32> = signed_in_0[i_0];
        var wide_1 : vec4<u32> = wide_in_0[i_0];
        signed_out_0[i_0] = integer_mix_0(wide_1.xyz, signed_1);
        sums_0[i_0] = sums_0[i_0] + f32(integer_dots_0(signed_1.xy, signed_1, wide_1.xy, wide_1)) + matrices_0(part_0.position_0, rgb_0).x;
    }
    return;
}

