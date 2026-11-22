struct _GustLaunchBase_std140_0
{
    @align(16) x_0 : u32,
    @align(4) y_0 : u32,
    @align(8) z_0 : u32,
    @align(4) pad_0 : u32,
};

@binding(3) @group(0) var<uniform> _gust_launch_base_0 : _GustLaunchBase_std140_0;
struct Pair_std430_0
{
    @align(4) x_1 : f32,
    @align(4) y_1 : f32,
};

struct Particle_std430_0
{
    @align(4) position_0 : Pair_std430_0,
    @align(4) velocity_0 : Pair_std430_0,
    @align(4) mass_0 : f32,
    @align(4) tag_0 : u32,
    @align(4) charge_0 : i32,
};

@binding(0) @group(0) var<storage, read> input_0 : array<Particle_std430_0>;

@binding(1) @group(0) var<storage, read> acceleration_0 : array<Pair_std430_0>;

@binding(2) @group(0) var<storage, read_write> out_0 : array<Particle_std430_0>;

@compute
@workgroup_size(256, 1, 1)
fn gpu_kernel_step(@builtin(global_invocation_id) _gust_launch_id_0 : vec3<u32>)
{
    var id_0 : vec3<u32> = _gust_launch_id_0 + vec3<u32>(_gust_launch_base_0.x_0, _gust_launch_base_0.y_0, _gust_launch_base_0.z_0);
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&input_0), 28);
    var _gpu_len_input_0 : u32 = _S1.x;
    var _S2 : vec2<u32> = vec2<u32>(arrayLength(&acceleration_0), 8);
    var _gpu_len_acceleration_0 : u32 = _S2.x;
    var _S3 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 28);
    var i_0 : u32 = id_0.x;
    var _S4 : bool;
    if((i_0 < (_S3.x)))
    {
        _S4 = (i_0 < _gpu_len_input_0);
    }
    else
    {
        _S4 = false;
    }
    if(_S4)
    {
        _S4 = (i_0 < _gpu_len_acceleration_0);
    }
    else
    {
        _S4 = false;
    }
    if(_S4)
    {
        var particle_0 : Particle_std430_0 = input_0[i_0];
        var _S5 : Pair_std430_0 = acceleration_0[i_0];
        var _step_0 : u32 = u32(0);
        var vx_0 : f32 = particle_0.velocity_0.x_1;
        var vy_0 : f32 = particle_0.velocity_0.y_1;
        var px_0 : f32 = particle_0.position_0.x_1;
        var py_0 : f32 = particle_0.position_0.y_1;
        loop
        {
            if((_step_0 < u32(16)))
            {
            }
            else
            {
                break;
            }
            var vx_1 : f32 = vx_0 + _S5.x_1 * 0.016000000759959221f;
            var vy_1 : f32 = vy_0 + _S5.y_1 * 0.016000000759959221f;
            var px_1 : f32 = px_0 + vx_1 * 0.016000000759959221f;
            var py_1 : f32 = py_0 + vy_1 * 0.016000000759959221f;
            _step_0 = _step_0 + u32(1);
            vx_0 = vx_1;
            vy_0 = vy_1;
            px_0 = px_1;
            py_0 = py_1;
        }
        out_0[i_0] = particle_0;
        out_0[i_0].velocity_0.x_1 = vx_0;
        out_0[i_0].velocity_0.y_1 = vy_0;
        out_0[i_0].position_0.x_1 = px_0;
        out_0[i_0].position_0.y_1 = py_0;
    }
    return;
}

