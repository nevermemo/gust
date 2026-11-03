[[vk::binding(0, 0)]] RWTexture2D<float4> rw;

[shader("compute")]
[numthreads(8, 8, 1)]
void gpu_refuse_storage_noformat(uint3 id : SV_DispatchThreadID)
{
    rw[uint2(id.x, id.y)] = float4(1.0);
}
