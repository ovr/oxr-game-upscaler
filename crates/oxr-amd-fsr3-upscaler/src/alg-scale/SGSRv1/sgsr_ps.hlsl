// Snapdragon Game Super Resolution v1 — pixel shader
// Ported from vendor/snapdragon-gsr/ (Qualcomm, BSD-3-Clause)
// Self-contained: all SGSR logic inlined (D3DCompile has no include handler).
// OperationMode=1 (RGBA), UseEdgeDirection enabled.

Texture2D<float4> src : register(t0);
SamplerState samp : register(s0);

cbuffer Params : register(b0) {
    float2 uvScale;    // constants 0,1
    float2 inputSize;  // constants 2,3 — color texture dimensions (width, height)
};

// --- SGSR helpers (float instead of half for desktop SM 5.0) ---

static const float EdgeThreshold = 8.0 / 255.0;
static const float EdgeSharpness = 2.0;

float4 SGSRRGBH(float2 p) {
    return src.SampleLevel(samp, p, 0);
}

float4 SGSRRH(float2 p) { return src.GatherRed(samp, p); }
float4 SGSRGH(float2 p) { return src.GatherGreen(samp, p); }
float4 SGSRBH(float2 p) { return src.GatherBlue(samp, p); }

float4 SGSRH(float2 p, uint channel) {
    if (channel == 0) return SGSRRH(p);
    if (channel == 1) return SGSRGH(p);
    return SGSRBH(p);
}

// --- Core SGSR algorithm (from sgsr1_mobile.h, OperationMode=1, UseEdgeDirection) ---

float fastLanczos2(float x) {
    float wA = x - 4.0;
    float wB = x * wA - wA;
    wA *= wA;
    return wB * wA;
}

float2 edgeDirection(float4 left, float4 right) {
    float RxLz = right.x + (-left.z);
    float RwLy = right.w + (-left.y);
    float2 delta;
    delta.x = RxLz + RwLy;
    delta.y = RxLz + (-RwLy);
    float lengthInv = rsqrt(delta.x * delta.x + 3.075740e-05 + delta.y * delta.y);
    return float2(delta.x * lengthInv, delta.y * lengthInv);
}

float2 weightY(float dx, float dy, float c, float3 data) {
    float std = data.x;
    float2 dir = data.yz;
    float edgeDis = dx * dir.y + dy * dir.x;
    float x = (dx * dx + dy * dy) + edgeDis * edgeDis * (clamp(c * c * std, 0.0, 1.0) * 0.7 - 1.0);
    float w = fastLanczos2(x);
    return float2(w, w * c);
}

void SgsrYuvH(out float4 pix, float2 uv, float4 con1) {
    // OperationMode=1 (RGBA): sample base color
    pix.xyz = SGSRRGBH(uv).xyz;

    float2 imgCoord = uv.xy * con1.zw + float2(-0.5, 0.5);
    float2 imgCoordPixel = floor(imgCoord);
    float2 coord = imgCoordPixel * con1.xy;
    float2 pl = imgCoord - imgCoordPixel;
    float4 left = SGSRH(coord, 1); // mode=1 → green channel

    float edgeVote = abs(left.z - left.y) + abs(pix[1] - left.y) + abs(pix[1] - left.z);
    if (edgeVote > EdgeThreshold) {
        coord.x += con1.x;

        float4 right  = SGSRH(coord + float2(con1.x, 0.0), 1);
        float4 upDown;
        upDown.xy = SGSRH(coord + float2(0.0, -con1.y), 1).wz;
        upDown.zw = SGSRH(coord + float2(0.0,  con1.y), 1).yx;

        float mean = (left.y + left.z + right.x + right.w) * 0.25;
        left   = left   - float4(mean, mean, mean, mean);
        right  = right  - float4(mean, mean, mean, mean);
        upDown = upDown - float4(mean, mean, mean, mean);
        pix.w  = pix[1] - mean;

        float sum = (abs(left.x) + abs(left.y) + abs(left.z) + abs(left.w))
                  + (abs(right.x) + abs(right.y) + abs(right.z) + abs(right.w))
                  + (abs(upDown.x) + abs(upDown.y) + abs(upDown.z) + abs(upDown.w));
        float sumMean = 10.14185 / sum;
        float std = sumMean * sumMean;
        float3 data = float3(std, edgeDirection(left, right));

        float2 aWY  = weightY(pl.x,       pl.y + 1.0, upDown.x, data);
        aWY         += weightY(pl.x - 1.0, pl.y + 1.0, upDown.y, data);
        aWY         += weightY(pl.x - 1.0, pl.y - 2.0, upDown.z, data);
        aWY         += weightY(pl.x,       pl.y - 2.0, upDown.w, data);
        aWY         += weightY(pl.x + 1.0, pl.y - 1.0, left.x,   data);
        aWY         += weightY(pl.x,       pl.y - 1.0, left.y,   data);
        aWY         += weightY(pl.x,       pl.y,       left.z,   data);
        aWY         += weightY(pl.x + 1.0, pl.y,       left.w,   data);
        aWY         += weightY(pl.x - 1.0, pl.y - 1.0, right.x,  data);
        aWY         += weightY(pl.x - 2.0, pl.y - 1.0, right.y,  data);
        aWY         += weightY(pl.x - 2.0, pl.y,       right.z,  data);
        aWY         += weightY(pl.x - 1.0, pl.y,       right.w,  data);

        float finalY = aWY.y / aWY.x;

        float max4 = max(max(left.y, left.z), max(right.x, right.w));
        float min4 = min(min(left.y, left.z), min(right.x, right.w));
        finalY = clamp(EdgeSharpness * finalY, min4, max4);

        float deltaY = finalY - pix.w;

        pix.x = saturate(pix.x + deltaY);
        pix.y = saturate(pix.y + deltaY);
        pix.z = saturate(pix.z + deltaY);
    }
    pix.w = 1.0;
}

// --- Entry point ---

float4 PS(float4 sv_pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_Target {
    // ViewportInfo = float4(1/inputSize.x, 1/inputSize.y, inputSize.x, inputSize.y)
    float4 viewportInfo = float4(1.0 / inputSize.x, 1.0 / inputSize.y, inputSize.x, inputSize.y);

    float4 outColor = float4(0, 0, 0, 1);
    SgsrYuvH(outColor, uv, viewportInfo);
    return outColor;
}
