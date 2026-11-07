// The order matters: the sampler is checked before the texture, and the texture must be a DepthTexture2D, because Slang 2026.21 itself refuses SampleCmpLevelZero on a Texture2D<float> (#13907) before GUST's check runs.
[[vk::binding(1, 0)]] SamplerComparisonState samp;
[[vk::binding(0, 0)]] DepthTexture2D tex;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> out;

[shader("compute")]
[numthreads(8, 8, 1)]
void gpu_refuse_comparison_sampler(uint3 id : SV_DispatchThreadID)
{
    out[id.x] = tex.SampleCmpLevelZero(samp, float2(0.5, 0.5), 0.5);
}
