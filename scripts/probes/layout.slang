// Hand-authored investigation fixture, not newly supported Rust GPU syntax.
struct Inner { float value; uint tag; };
struct Pod { Inner inner; int delta; };
// Contrasts the existing four-byte-aligned, padding-free host contract: float3 pads to 16 bytes.
struct Padded { float head; float3 direction; float tail; };
// A vector member between two scalars: a@0 pos@8 b@16, size 24 in a storage buffer.
struct Lit { float a; float2 pos; float b; };
// A constant buffer: scale@0 tint@16 view@32 inner@96 offset@112 bias@120, size 128.
struct Frame { float scale; float4 tint; float4x4 view; Inner inner; float2 offset; float bias; };

[[vk::binding(0, 0)]] StructuredBuffer<float> scalars;
[[vk::binding(1, 0)]] StructuredBuffer<Pod> pods;
[[vk::binding(2, 0)]] StructuredBuffer<Padded> padded;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> output;
[[vk::binding(4, 0)]] StructuredBuffer<float4x4> matrices;
[[vk::binding(5, 0)]] StructuredBuffer<Lit> lits;
[[vk::binding(6, 0)]] ConstantBuffer<Frame> frame;
// Not square, so the column stride (8) differs from the row stride (16) and from the stride of the whole matrix (32).
[[vk::binding(7, 0)]] StructuredBuffer<float2x4> wide;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint count, stride;
    output.GetDimensions(count, stride);
    if (id.x < count)
        output[id.x] = scalars[id.x] + pods[id.x].inner.value
            + float(pods[id.x].inner.tag) + float(pods[id.x].delta)
            + padded[id.x].head + padded[id.x].direction.x + padded[id.x].tail
            + mul(matrices[id.x], frame.tint).x + lits[id.x].a + lits[id.x].pos.y + lits[id.x].b + wide[id.x][1][2]
            + frame.scale * frame.inner.value + frame.offset.x + frame.bias;
}
