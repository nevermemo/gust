@binding(2) @group(0) var<storage, read_write> out_0 : array<f32>;

@binding(0) @group(0) var<storage, read> a_0 : array<f32>;

@binding(1) @group(0) var<storage, read> b_0 : array<f32>;

@fragment
fn gpu_vector_add_add_pixel(@builtin(position) _gust_pos_0 : vec4<f32>)
{
    var _gust_linear_0 : u32 = u32(_gust_pos_0.y) * u32(8192) + u32(_gust_pos_0.x);
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 4);
    out_0[_gust_linear_0] = a_0[_gust_linear_0] + b_0[_gust_linear_0];
    return;
}

