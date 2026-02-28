// Debug visualization pixel shader.
// Root constants (b0): uvScale (slots 0-1), vizMode (slot 2 reinterpreted as uint).
// vizMode: 0=RGB passthrough, 1=depth (grayscale from R), 2=motion vectors (RG), 3=single-channel mask (grayscale from R)
cbuffer Params : register(b0) { float2 uvScale; float vizModeFloat; };
Texture2D<float4> src : register(t0);
SamplerState samp : register(s0);

float4 PS(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_Target {
    float4 c = src.Sample(samp, uv);
    uint vizMode = asuint(vizModeFloat);

    if (vizMode == 1) {
        // Depth: show R channel as grayscale
        return float4(c.r, c.r, c.r, 1.0);
    }
    if (vizMode == 2) {
        // Motion vectors: show RG, blue=0
        return float4(abs(c.r), abs(c.g), 0.0, 1.0);
    }
    if (vizMode == 3) {
        // Single-channel mask: grayscale from R
        return float4(c.r, c.r, c.r, 1.0);
    }
    // vizMode == 0: RGB passthrough
    return float4(c.rgb, 1.0);
}
