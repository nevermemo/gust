[[vk::binding(0, 0)]] Texture3D<float4> tex;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> out;

[shader("compute")]
[numthreads(8, 8, 1)]
void gpu_refuse_texture3d(uint3 id : SV_DispatchThreadID)
{
    out[id.x] = tex.Load(int4(int(id.x), 0, 0, 0)).x;
}
