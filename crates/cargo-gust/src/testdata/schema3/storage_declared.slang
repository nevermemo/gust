[[vk::binding(0, 0)]] [format("rgba32f")] RWTexture2D<float4> rw;
[[vk::binding(1, 0)]] [format("rg32ui")] WTexture2D<uint2> wo;

[shader("compute")]
[numthreads(8, 8, 1)]
void gpu_storage_declared(uint3 id : SV_DispatchThreadID)
{
    uint2 p = uint2(id.x, id.y);
    rw[p] = rw[p] + float4(1.0, 0.0, 0.0, 0.0);
    wo.Store(p, uint2(id.x, id.y));
}
