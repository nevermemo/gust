[[vk::binding(0, 0)]] RWTexture2D<uint> rw;

[shader("compute")]
[numthreads(8, 8, 1)]
void gpu_storage_inferred(uint3 id : SV_DispatchThreadID)
{
    uint2 p = uint2(id.x, id.y);
    rw[p] = rw[p] + 1u;
}
