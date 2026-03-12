Texture2D<float4> src : register(t0);
SamplerState samp : register(s0);
float4 PS(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_Target {
    return src.Sample(samp, uv);
}
