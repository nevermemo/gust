struct Particle_std430_0
{
    @align(4) pos_0 : array<f32, i32(2)>,
    @align(4) vel_0 : array<f32, i32(2)>,
    @align(4) bounces_0 : u32,
};

@binding(0) @group(0) var<storage, read_write> ps_0 : array<Particle_std430_0>;

struct Particle_0
{
     pos_0 : array<f32, i32(2)>,
     vel_0 : array<f32, i32(2)>,
     bounces_0 : u32,
};

@compute
@workgroup_size(64, 1, 1)
fn gpu_particles_step(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&ps_0), 20);
    var i_0 : u32 = id_0.x;
    if((i_0 < (_S1.x)))
    {
        var p_0 : Particle_0;
        p_0.pos_0 = ps_0[i_0].pos_0;
        p_0.vel_0 = ps_0[i_0].vel_0;
        p_0.bounces_0 = ps_0[i_0].bounces_0;
        var _S2 : f32 = p_0.vel_0[i32(1)] - 1.0f;
        p_0.vel_0[i32(1)] = _S2;
        p_0.pos_0[i32(0)] = p_0.pos_0[i32(0)] + p_0.vel_0[i32(0)] * 0.125f;
        var _S3 : f32 = p_0.pos_0[i32(1)] + _S2 * 0.125f;
        p_0.pos_0[i32(1)] = _S3;
        if((_S3 < 0.0f))
        {
            p_0.pos_0[i32(1)] = - p_0.pos_0[i32(1)];
            p_0.vel_0[i32(1)] = - p_0.vel_0[i32(1)] * 0.5f;
            p_0.bounces_0 = p_0.bounces_0 + u32(1);
        }
        ps_0[i_0].pos_0 = p_0.pos_0;
        ps_0[i_0].vel_0 = p_0.vel_0;
        ps_0[i_0].bounces_0 = p_0.bounces_0;
    }
    return;
}

