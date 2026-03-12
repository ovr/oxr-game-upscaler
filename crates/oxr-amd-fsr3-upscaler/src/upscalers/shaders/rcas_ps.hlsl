// AMD FidelityFX RCAS (Robust Contrast-Adaptive Sharpening)
// Ported from vendor/FidelityFX-SDK-v1/sdk/include/FidelityFX/gpu/fsr1/ffx_fsr1.h:684-772

Texture2D<float4> src : register(t0);

cbuffer Params : register(b0) { float sharpness; };

#define FSR_RCAS_LIMIT (0.25 - 1.0/16.0)

float4 PS(float4 sv_pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_Target {
    int2 sp = int2(sv_pos.xy);

    // 3x3 cross neighborhood
    //    b
    //  d e f
    //    h
    float3 b = src.Load(int3(sp + int2( 0,-1), 0)).rgb;
    float3 d = src.Load(int3(sp + int2(-1, 0), 0)).rgb;
    float3 e = src.Load(int3(sp,                0)).rgb;
    float3 f = src.Load(int3(sp + int2( 1, 0), 0)).rgb;
    float3 h = src.Load(int3(sp + int2( 0, 1), 0)).rgb;

    float bR = b.r, bG = b.g, bB = b.b;
    float dR = d.r, dG = d.g, dB = d.b;
    float eR = e.r, eG = e.g, eB = e.b;
    float fR = f.r, fG = f.g, fB = f.b;
    float hR = h.r, hG = h.g, hB = h.b;

    // Luma times 2
    float bL = bB * 0.5 + (bR * 0.5 + bG);
    float dL = dB * 0.5 + (dR * 0.5 + dG);
    float eL = eB * 0.5 + (eR * 0.5 + eG);
    float fL = fB * 0.5 + (fR * 0.5 + fG);
    float hL = hB * 0.5 + (hR * 0.5 + hG);

    // Noise detection (FSR_RCAS_DENOISE)
    float nz = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;
    float range = max(max(max(bL, dL), max(eL, fL)), hL) - min(min(min(bL, dL), min(eL, fL)), hL);
    nz = saturate(abs(nz) * rcp(range));
    nz = -0.5 * nz + 1.0;

    // Min and max of ring
    float mn4R = min(min(min(bR, dR), fR), hR);
    float mn4G = min(min(min(bG, dG), fG), hG);
    float mn4B = min(min(min(bB, dB), fB), hB);
    float mx4R = max(max(max(bR, dR), fR), hR);
    float mx4G = max(max(max(bG, dG), fG), hG);
    float mx4B = max(max(max(bB, dB), fB), hB);

    // Limiters
    float hitMinR = mn4R * rcp(4.0 * mx4R);
    float hitMinG = mn4G * rcp(4.0 * mx4G);
    float hitMinB = mn4B * rcp(4.0 * mx4B);
    float hitMaxR = (1.0 - mx4R) * rcp(4.0 * mn4R - 4.0);
    float hitMaxG = (1.0 - mx4G) * rcp(4.0 * mn4G - 4.0);
    float hitMaxB = (1.0 - mx4B) * rcp(4.0 * mn4B - 4.0);
    float lobeR = max(-hitMinR, hitMaxR);
    float lobeG = max(-hitMinG, hitMaxG);
    float lobeB = max(-hitMinB, hitMaxB);
    float lobe = max(-FSR_RCAS_LIMIT, min(max(max(lobeR, lobeG), lobeB), 0.0)) * sharpness;

    // Apply noise removal
    lobe *= nz;

    // Resolve
    float rcpL = rcp(4.0 * lobe + 1.0);
    float pixR = (lobe * bR + lobe * dR + lobe * hR + lobe * fR + eR) * rcpL;
    float pixG = (lobe * bG + lobe * dG + lobe * hG + lobe * fG + eG) * rcpL;
    float pixB = (lobe * bB + lobe * dB + lobe * hB + lobe * fB + eB) * rcpL;

    return float4(pixR, pixG, pixB, 1.0);
}
