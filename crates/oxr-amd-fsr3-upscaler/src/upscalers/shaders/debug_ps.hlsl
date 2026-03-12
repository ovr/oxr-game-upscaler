// Debug visualization pixel shader.
// Root constants (b0): uvScale (slots 0-1), vizMode (slot 2 reinterpreted as uint).
// vizMode: 0=RGB passthrough, 1=depth grayscale, 2=motion vectors (RG),
//          3=single-channel mask, 4=depth colorized (red-blue), 5=integer texture (point sampled, grayscale)
cbuffer Params : register(b0) { float2 uvScale; float vizModeFloat; };
Texture2D<float4> src : register(t0);
SamplerState samp : register(s0);
SamplerState pointSamp : register(s1);

float4 PS(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_Target {
    uint vizMode = asuint(vizModeFloat);

    // Integer textures must use point sampler
    if (vizMode == 5) {
        float4 c = src.Sample(pointSamp, uv);
        float v = c.r / 255.0; // R8_UINT range [0,255] → [0,1]
        return float4(v, v, v, 1.0);
    }

    float4 c = src.Sample(samp, uv);

    if (vizMode == 1) {
        // Depth: show R channel as grayscale
        return float4(c.r, c.r, c.r, 1.0);
    }
    if (vizMode == 4) {
        // Depth colorized: reverse-Z linearized, red=near, blue=far
        float z = max(c.r, 1e-6);
        float lin_depth = 1.0 / z;
        float norm = saturate(lin_depth / 1000.0);
        norm = pow(norm, 0.15);
        return float4(1.0 - norm, 0.0, norm, 1.0);
    }
    if (vizMode == 2) {
        // Motion vectors: amplify small values, RG → red/green
        float2 mv = abs(c.rg) * 20.0;
        return float4(saturate(mv.x), saturate(mv.y), 0.0, 1.0);
    }
    if (vizMode == 3) {
        // Single-channel mask: grayscale from R
        return float4(c.r, c.r, c.r, 1.0);
    }
    // vizMode == 0: RGB passthrough
    return float4(c.rgb, 1.0);
}
