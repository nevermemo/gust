struct Out
{
    uint u : SV_Target0;
    float4 v : SV_Target2;
};

[shader("fragment")]
Out gpu_fragment_gap(float4 pos : SV_Position)
{
    Out o;
    o.u = uint(pos.x);
    o.v = pos;
    return o;
}
