[[vk::binding(0, 0)]] Texture2D<float4> tex;
[[vk::binding(1, 0)]] SamplerState samp;

[shader("fragment")]
float4 gpu_sampler_fragment(float4 pos : SV_Position) : SV_Target0
{
    return tex.SampleLevel(samp, float2(pos.x / 64.0, pos.y / 64.0), 0.0);
}
