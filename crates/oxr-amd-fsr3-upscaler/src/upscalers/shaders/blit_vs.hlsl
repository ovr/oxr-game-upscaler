cbuffer Params : register(b0) { float2 uvScale; };
struct VSOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };
VSOut VS(uint id : SV_VertexID) {
    VSOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.pos = float4(uv * 2.0 - 1.0, 0, 1);
    o.pos.y = -o.pos.y;
    o.uv = uv * uvScale;
    return o;
}
