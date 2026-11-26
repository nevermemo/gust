@binding(5) @group(0) var<storage, read_write> out_0 : array<f32>;

@binding(0) @group(0) var<storage, read> signal_0 : array<f32>;

@binding(1) @group(0) var<storage, read> baseline_0 : array<f32>;

@binding(2) @group(0) var<storage, read> gain_0 : array<f32>;

@binding(3) @group(0) var<storage, read> bias_0 : array<f32>;

@binding(4) @group(0) var<storage, read> scale_0 : array<f32>;

fn _gust_div_0( a_0 : f32,  b_0 : f32) -> f32
{
    return a_0 / b_0;
}

@compute
@workgroup_size(256, 1, 1)
fn gpu_fused_transform(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 4);
    var i_0 : u32 = id_0.x;
    if((i_0 < (_S1.x)))
    {
        var normalized_0 : f32 = _gust_div_0((signal_0[i_0] - baseline_0[i_0]) * gain_0[i_0] + bias_0[i_0], scale_0[i_0]);
        out_0[i_0] = normalized_0 * normalized_0 + 0.25f * normalized_0;
    }
    return;
}

