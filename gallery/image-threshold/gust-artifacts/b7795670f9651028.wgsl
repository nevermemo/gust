struct _GustLaunchBase_std140_0
{
    @align(16) x_0 : u32,
    @align(4) y_0 : u32,
    @align(8) z_0 : u32,
    @align(4) pad_0 : u32,
};

@binding(1) @group(0) var<uniform> _gust_launch_base_0 : _GustLaunchBase_std140_0;
@binding(0) @group(0) var image_0 : texture_storage_2d<r32float, read_write>;

@compute
@workgroup_size(8, 8, 1)
fn gpu_threshold_binarize(@builtin(global_invocation_id) _gust_launch_id_0 : vec3<u32>)
{
    var id_0 : vec3<u32> = _gust_launch_id_0 + vec3<u32>(_gust_launch_base_0.x_0, _gust_launch_base_0.y_0, _gust_launch_base_0.z_0);
    var _gpu_width_image_0 : u32;
    var _gpu_height_image_0 : u32;
    {var dim = textureDimensions((image_0));((_gpu_width_image_0)) = dim.x;((_gpu_height_image_0)) = dim.y;};
    var _S1 : u32 = id_0.x;
    var _S2 : bool;
    if((_S1 < _gpu_width_image_0))
    {
        _S2 = ((id_0.y) < _gpu_height_image_0);
    }
    else
    {
        _S2 = false;
    }
    if(_S2)
    {
        var _S3 : vec2<u32> = vec2<u32>(_S1, id_0.y);
        var _S4 : f32 = (textureLoad((image_0), (vec2<i32>(_S3))).x);
        if((_S4 > 0.5f))
        {
            textureStore((image_0), (_S3), vec4<f32>((1.0f), 0, 0, 1));
        }
        else
        {
            textureStore((image_0), (_S3), vec4<f32>((0.0f), 0, 0, 1));
        }
    }
    return;
}

