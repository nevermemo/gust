[[vk::binding(0, 0)]] Texture2D<float4> tex;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> out;

[shader("compute")]
[numthreads(8, 8, 1)]
void gpu_sampled_float4(uint3 id : SV_DispatchThreadID)
{
    out[id.x] = tex.Load(int3(int(id.x), int(id.y), 0)).x;
}
