@binding(2) @group(0) var<storage, read_write> flags_0 : array<u32>;

@binding(0) @group(0) var image_0 : texture_2d<f32>;

@binding(1) @group(0) var nearest_0 : sampler;

fn _gust_rem_0( a_0 : u32,  b_0 : u32) -> u32
{
    var _S1 : u32 = a_0 % b_0;
    return _S1;
}

fn _gust_div_0( a_1 : f32,  b_1 : f32) -> f32
{
    return a_1 / b_1;
}

fn _gust_div_1( a_2 : u32,  b_2 : u32) -> u32
{
    var _S2 : u32 = a_2 / b_2;
    return _S2;
}

@fragment
fn gpu_threshold_flag(@builtin(position) _gust_pos_0 : vec4<f32>)
{
    var _gust_linear_0 : u32 = u32(_gust_pos_0.y) * u32(8192) + u32(_gust_pos_0.x);
    var _S3 : vec2<u32> = vec2<u32>(arrayLength(&flags_0), 4);
    var _gpu_width_image_0 : u32;
    var _gpu_height_image_0 : u32;
    {var dim = textureDimensions((image_0));((_gpu_width_image_0)) = dim.x;((_gpu_height_image_0)) = dim.y;};
    var _S4 : u32 = _gust_rem_0(_gust_linear_0, _gpu_width_image_0);
    var u_0 : f32 = _gust_div_0(f32(_S4) + 0.5f, f32(_gpu_width_image_0));
    var _S5 : u32 = _gust_div_1(_gust_linear_0, _gpu_width_image_0);
    if(((textureSampleLevel((image_0), (nearest_0), (vec2<f32>(u_0, _gust_div_0(f32(_S5) + 0.5f, f32(_gpu_height_image_0)))), (0.0f)).x) > 0.5f))
    {
        flags_0[_gust_linear_0] = u32(1);
    }
    else
    {
        flags_0[_gust_linear_0] = u32(0);
    }
    return;
}

