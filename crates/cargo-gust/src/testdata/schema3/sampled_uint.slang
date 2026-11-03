[[vk::binding(0, 0)]] Texture2D<uint> tex;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> out;

[shader("compute")]
[numthreads(8, 8, 1)]
void gpu_sampled_uint(uint3 id : SV_DispatchThreadID)
{
    out[id.x] = tex.Load(int3(int(id.x), int(id.y), 0));
}
