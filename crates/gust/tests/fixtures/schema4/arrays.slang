// Hand-authored fixture for the reflection helper's array arm (schema 4, m9.r2h); not Rust GPU syntax.
// P is the struct of the m9.s0 spike (P4b/P10); Q and R and S are the further layouts of that table and a nested array.
struct P { float a[4]; float2 b[2]; float3 c[2]; float4 d[2]; float2x2 e[2]; float4x4 f[2]; float g; };
struct Q { float x; float y; };
struct R { Q q[3]; uint n; };
struct S { float h[2][3]; uint k; };

[[vk::binding(0, 0)]] StructuredBuffer<P> ps;
[[vk::binding(1, 0)]] StructuredBuffer<float[4]> quads;
[[vk::binding(2, 0)]] RWStructuredBuffer<float2[2]> pairs;
[[vk::binding(3, 0)]] StructuredBuffer<R> rs;
[[vk::binding(4, 0)]] StructuredBuffer<S> ss;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> output;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint count, stride;
    output.GetDimensions(count, stride);
    if (id.x < count)
    {
        P p = ps[id.x];
        pairs[id.x][1] = p.b[1] + p.c[1].xy;
        output[id.x] = p.a[3] + p.d[1].w + p.e[1][1][1] + p.f[1][3][3] + p.g + quads[id.x][2] + float(rs[id.x].n)
            + rs[id.x].q[2].y + ss[id.x].h[1][2] + float(ss[id.x].k);
    }
}
