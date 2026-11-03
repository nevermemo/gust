[[vk::binding(0, 0)]] Texture2D<float> tex;
[[vk::binding(1, 0)]] SamplerComparisonState samp;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> out;

[shader("compute")]
[numthreads(8, 8, 1)]
void gpu_refuse_comparison_sampler(uint3 id : SV_DispatchThreadID)
{
    out[id.x] = tex.SampleCmpLevelZero(samp, float2(0.5, 0.5), 0.5);
}
