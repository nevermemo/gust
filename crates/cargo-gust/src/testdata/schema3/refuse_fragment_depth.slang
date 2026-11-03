struct Out
{
    float4 c : SV_Target0;
    float d : SV_Depth;
};

[shader("fragment")]
Out gpu_refuse_fragment_depth(float4 pos : SV_Position)
{
    Out o;
    o.c = pos;
    o.d = 0.5;
    return o;
}
