//============================================================================================================
//
//  SGSRv2 3-Pass Upscale — ported from 3Pass/sgsr_upscale.h
//
//  Copyright (c) 2024-2025, Qualcomm Innovation Center, Inc. All rights reserved.
//  SPDX-License-Identifier: BSD-3-Clause
//
//  Adapted for HLSL SM 6.0 pixel shader. MRT architecture: SV_Target0 = tonemapped
//  RGB + Wfactor (history buffer), SV_Target1 = inverse-tonemapped HDR RGB (scene output).
//
//============================================================================================================

cbuffer Params : register(b0)
{
    float4 clipToPrevClip[4];   // 16 floats
    float2 renderSize;          // 2
    float2 outputSize;          // 2
    float2 renderSizeRcp;       // 2
    float2 outputSizeRcp;       // 2
    float2 jitterOffset;        // 2
    float2 scaleRatio;          // 2
    float  cameraFovAngleHor;   // 1
    float  minLerpContribution; // 1
    float  reset;               // 1  (ValidReset)
    float  preExposure;         // 1 = 32 DWORDs total
};

Texture2D<float4> PrevHistoryOutput            : register(t0);
Texture2D<float4> MotionDepthClipAlphaBuffer   : register(t1);
Texture2D<uint>   YCoCgColor                   : register(t2);
SamplerState      samp                         : register(s0);   // linear clamp
SamplerState      pointSamp                    : register(s1);   // point clamp

struct VSOut
{
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD0;
};

half FastLanczos(half base)
{
    half y = base - 1.0h;
    half y2 = y * y;
    half y_temp = 0.75h * y + y2;
    return y_temp * y2;
}

// Decode packed 11-11-10 YCoCg uint → tonemapped RGB [0,1]
half3 DecodeColorRGB(uint sample32)
{
    uint x11 = sample32 >> 21;
    uint y11 = sample32 & (2047 << 10);
    uint z10 = sample32 & 1023;
    half Y  = (half)x11 * (1.0h / 2047.5h);
    half Co = (half)y11 * 4.76953602e-7h - 0.5h;
    half Cg = (half)z10 * (1.0h / 1023.5h) - 0.5h;
    // YCoCg → RGB (saturate: tonemapped values should be in [0,1])
    half tmp = Y - Cg;
    return saturate(half3(tmp + Co, Y + Cg, tmp - Co));
}

struct PSOut {
    float4 history : SV_Target0;   // tonemapped RGB + Wfactor (for history buffer)
    float4 scene   : SV_Target1;   // inverse-tonemapped HDR RGB + 1.0 (for game output)
};

PSOut PS(VSOut input)
{
    // Compute derived params from scale ratio
    float Biasmax_viewportXScale = min(scaleRatio.x, 1.99);
    float Scalefactor = min(20.0, pow(scaleRatio.x * scaleRatio.y, 3.0));
    float Exposure_co_rcp = preExposure;
    float ValidReset = reset;

    float2 Hruv = input.uv;
    float2 Jitteruv;
    Jitteruv.x = clamp(Hruv.x + jitterOffset.x * renderSizeRcp.x, 0.0, 1.0);
    Jitteruv.y = clamp(Hruv.y + jitterOffset.y * renderSizeRcp.y, 0.0, 1.0);
    int2 InputPos = int2(Jitteruv * renderSize);

    float4 mda = MotionDepthClipAlphaBuffer.SampleLevel(samp, Jitteruv, 0);
    float2 Motion = mda.xy;

    float2 PrevUV;
    PrevUV.x = clamp(-0.5 * Motion.x + Hruv.x, 0.0, 1.0);
    PrevUV.y = clamp( 0.5 * Motion.y + Hruv.y, 0.0, 1.0);

    float depthfactor = frac(mda.z);
    float bright = (mda.z - depthfactor) * 1000.0;
    float history_value = frac(mda.w);
    float alphamask = (mda.w - history_value) * 0.001;
    history_value *= 2.0;

    // Read history (already tonemapped RGB + Wfactor from previous frame's SV_Target0)
    float4 History = PrevHistoryOutput.SampleLevel(samp, PrevUV, 0);
    float Wfactor = max(saturate(abs(History.w)), alphamask);
    half3 HistoryColor = (half3)History.xyz;  // already tonemapped [0,1]

    // Upsample and compute bounding box (all in tonemapped-RGB space)
    half4 Upsampledcw = half4(0.0h, 0.0h, 0.0h, 0.0h);
    float kernelfactor = saturate(Wfactor + ValidReset);
    float biasmax = Biasmax_viewportXScale - Biasmax_viewportXScale * kernelfactor;
    float biasmin = max(1.0, 0.3 + 0.3 * biasmax);
    float biasfactor = max(0.25 * depthfactor, kernelfactor);
    float kernelbias = lerp(biasmax, biasmin, biasfactor);
    float motion_viewport_len = length(Motion * outputSize);
    float curvebias = lerp(-2.0, -3.0, saturate(motion_viewport_len * 0.02));

    half3 rectboxcenter = half3(0.0h, 0.0h, 0.0h);
    half3 rectboxvar = half3(0.0h, 0.0h, 0.0h);
    half rectboxweight = 0.0h;

    float2 srcpos = float2(InputPos) + float2(0.5, 0.5) - jitterOffset;
    float2 srcOutputPos = Hruv * renderSize;

    kernelbias *= 0.5;
    float kernelbias2 = kernelbias * kernelbias;
    float2 srcpos_srcOutputPos = srcpos - srcOutputPos;

    int2 InputPosBtmRight = int2(1, 1) + InputPos;
    float2 gatherCoord = float2(InputPos) * renderSizeRcp;
    uint4 topleft = YCoCgColor.GatherRed(pointSamp, gatherCoord);
    uint2 topRight;
    uint2 bottomLeft;

    topRight = YCoCgColor.GatherRed(pointSamp, gatherCoord + float2(renderSizeRcp.x, 0.0)).yz;
    bottomLeft = YCoCgColor.GatherRed(pointSamp, gatherCoord + float2(0.0, renderSizeRcp.y)).xy;

    half3 rectboxmin;
    half3 rectboxmax;
    half3 centerSample; // save for NaN fallback

    // Sample 0: (0, +1)
    {
        half3 samplecolor = DecodeColorRGB(bottomLeft.y);
        float2 baseoffset = srcpos_srcOutputPos + float2(0.0, 1.0);
        float baseoffset_dot = dot(baseoffset, baseoffset);
        half base = (half)clamp(baseoffset_dot * kernelbias2, 0.0, 1.0);
        half weight = FastLanczos(base);
        Upsampledcw = half4(samplecolor * weight, weight);
        half boxweight = (half)exp(baseoffset_dot * curvebias);
        rectboxmin = samplecolor;
        rectboxmax = samplecolor;
        half3 wsample = samplecolor * boxweight;
        rectboxcenter = wsample;
        rectboxvar = samplecolor * wsample;
        rectboxweight = boxweight;
    }
    // Sample 1: (+1, 0)
    {
        half3 samplecolor = DecodeColorRGB(topRight.x);
        float2 baseoffset = srcpos_srcOutputPos + float2(1.0, 0.0);
        float baseoffset_dot = dot(baseoffset, baseoffset);
        half base = (half)clamp(baseoffset_dot * kernelbias2, 0.0, 1.0);
        half weight = FastLanczos(base);
        Upsampledcw += half4(samplecolor * weight, weight);
        half boxweight = (half)exp(baseoffset_dot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        half3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += samplecolor * wsample;
        rectboxweight += boxweight;
    }
    // Sample 2: (-1, 0)
    {
        half3 samplecolor = DecodeColorRGB(topleft.x);
        float2 baseoffset = srcpos_srcOutputPos + float2(-1.0, 0.0);
        float baseoffset_dot = dot(baseoffset, baseoffset);
        half base = (half)clamp(baseoffset_dot * kernelbias2, 0.0, 1.0);
        half weight = FastLanczos(base);
        Upsampledcw += half4(samplecolor * weight, weight);
        half boxweight = (half)exp(baseoffset_dot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        half3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += samplecolor * wsample;
        rectboxweight += boxweight;
    }
    // Sample 3: center (0, 0)
    {
        half3 samplecolor = DecodeColorRGB(topleft.y);
        centerSample = samplecolor;
        float2 baseoffset = srcpos_srcOutputPos;
        float baseoffset_dot = dot(baseoffset, baseoffset);
        half base = (half)clamp(baseoffset_dot * kernelbias2, 0.0, 1.0);
        half weight = FastLanczos(base);
        Upsampledcw += half4(samplecolor * weight, weight);
        half boxweight = (half)exp(baseoffset_dot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        half3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += samplecolor * wsample;
        rectboxweight += boxweight;
    }
    // Sample 4: (0, -1)
    {
        half3 samplecolor = DecodeColorRGB(topleft.z);
        float2 baseoffset = srcpos_srcOutputPos + float2(0.0, -1.0);
        float baseoffset_dot = dot(baseoffset, baseoffset);
        half base = (half)clamp(baseoffset_dot * kernelbias2, 0.0, 1.0);
        half weight = FastLanczos(base);
        Upsampledcw += half4(samplecolor * weight, weight);
        half boxweight = (half)exp(baseoffset_dot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        half3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += samplecolor * wsample;
        rectboxweight += boxweight;
    }
    // Sample 5: (+1, +1)
    {
        uint btmRight = YCoCgColor[InputPosBtmRight].x;
        half3 samplecolor = DecodeColorRGB(btmRight);
        float2 baseoffset = srcpos_srcOutputPos + float2(1.0, 1.0);
        float baseoffset_dot = dot(baseoffset, baseoffset);
        half base = (half)clamp(baseoffset_dot * kernelbias2, 0.0, 1.0);
        half weight = FastLanczos(base);
        Upsampledcw += half4(samplecolor * weight, weight);
        half boxweight = (half)exp(baseoffset_dot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        half3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += samplecolor * wsample;
        rectboxweight += boxweight;
    }
    // Sample 6: (-1, +1)
    {
        half3 samplecolor = DecodeColorRGB(bottomLeft.x);
        float2 baseoffset = srcpos_srcOutputPos + float2(-1.0, 1.0);
        float baseoffset_dot = dot(baseoffset, baseoffset);
        half base = (half)clamp(baseoffset_dot * kernelbias2, 0.0, 1.0);
        half weight = FastLanczos(base);
        Upsampledcw += half4(samplecolor * weight, weight);
        half boxweight = (half)exp(baseoffset_dot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        half3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += samplecolor * wsample;
        rectboxweight += boxweight;
    }
    // Sample 7: (+1, -1)
    {
        half3 samplecolor = DecodeColorRGB(topRight.y);
        float2 baseoffset = srcpos_srcOutputPos + float2(1.0, -1.0);
        float baseoffset_dot = dot(baseoffset, baseoffset);
        half base = (half)clamp(baseoffset_dot * kernelbias2, 0.0, 1.0);
        half weight = FastLanczos(base);
        Upsampledcw += half4(samplecolor * weight, weight);
        half boxweight = (half)exp(baseoffset_dot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        half3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += samplecolor * wsample;
        rectboxweight += boxweight;
    }
    // Sample 8: (-1, -1)
    {
        half3 samplecolor = DecodeColorRGB(topleft.w);
        float2 baseoffset = srcpos_srcOutputPos + float2(-1.0, -1.0);
        float baseoffset_dot = dot(baseoffset, baseoffset);
        half base = (half)clamp(baseoffset_dot * kernelbias2, 0.0, 1.0);
        half weight = FastLanczos(base);
        Upsampledcw += half4(samplecolor * weight, weight);
        half boxweight = (half)exp(baseoffset_dot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        half3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += samplecolor * wsample;
        rectboxweight += boxweight;
    }

    // Normalize bounding box
    rectboxweight = max(rectboxweight, 1e-6h);
    rectboxweight = 1.0h / rectboxweight;
    rectboxcenter *= rectboxweight;
    rectboxvar *= rectboxweight;
    rectboxvar = sqrt(abs(rectboxvar - rectboxcenter * rectboxcenter));

    half3 bias = half3(0.075h, 0.075h, 0.075h);
    half div_w = Upsampledcw.w;
    if (abs(div_w) < 0.001h) div_w = 0.001h;
    Upsampledcw.xyz = Upsampledcw.xyz / div_w;
    // Guard NaN/Inf from division
    if (any(isnan(Upsampledcw.xyz)) || any(isinf(Upsampledcw.xyz)))
        Upsampledcw.xyz = centerSample;
    Upsampledcw.xyz = clamp(Upsampledcw.xyz, rectboxmin - bias, rectboxmax + bias);
    // Guard NaN from rectbox clamp (NaN bounds propagate on some GPUs)
    if (any(isnan(Upsampledcw.xyz)))
        Upsampledcw.xyz = centerSample;
    Upsampledcw.w = max(abs(Upsampledcw.w), 0.001h) * (1.0h / 3.0h);

    half tcontribute = (half)history_value * saturate(rectboxvar.x * 10.0h);
    half OneMinusWfactor = 1.0h - (half)Wfactor;
    tcontribute = tcontribute * OneMinusWfactor;

    half baseupdate = OneMinusWfactor - OneMinusWfactor * (half)depthfactor;
    baseupdate = min(baseupdate, lerp(baseupdate, Upsampledcw.w * 10.0h, saturate((half)(10.0 * motion_viewport_len))));
    baseupdate = min(baseupdate, lerp(baseupdate, Upsampledcw.w, saturate((half)(motion_viewport_len * 0.05))));
    half basealpha = baseupdate;

    const half EPS = 1.192e-07h;
    half boxscale = max((half)depthfactor, saturate((half)(motion_viewport_len * 0.05)));
    half boxsize = lerp((half)Scalefactor, 1.0h, boxscale);

    half3 sboxvar = rectboxvar * boxsize;
    half3 boxmin = rectboxcenter - sboxvar;
    half3 boxmax = rectboxcenter + sboxvar;
    rectboxmax = min(rectboxmax, boxmax);
    rectboxmin = max(rectboxmin, boxmin);

    // Clamp tonemapped history against tonemapped-RGB rectbox
    half3 clampedcolor = clamp(HistoryColor, rectboxmin, rectboxmax);
    half lerpcontribution = (any(rectboxmin > HistoryColor) || any(HistoryColor > rectboxmax)) ? tcontribute : 1.0h;
    lerpcontribution = lerpcontribution - lerpcontribution * (half)sqrt(alphamask);

    HistoryColor = lerp(clampedcolor, HistoryColor, saturate(lerpcontribution));
    half basemin = min(basealpha, 0.1h);
    basealpha = lerp(basemin, basealpha, saturate(lerpcontribution));

    // Blend current frame with history (both in tonemapped-RGB space)
    half alphasum = max(EPS, basealpha + Upsampledcw.w);
    half alpha = saturate(Upsampledcw.w / alphasum + (half)ValidReset);
    half3 blended = lerp(HistoryColor, Upsampledcw.xyz, alpha);

    // NaN guard on blended
    if (any(isnan(blended)) || any(isinf(blended)))
        blended = centerSample;

    // SV_Target0: tonemapped for history (stable, bounded [0,1])
    PSOut o;
    o.history = float4((float3)blended, Wfactor);

    // SV_Target1: inverse tonemap for scene output (HDR)
    float3 blended_f = (float3)blended;
    float compMax = max(blended_f.x, max(blended_f.y, blended_f.z));
    float scale;
    if (bright > 1000.0)
    {
        compMax = clamp(compMax, 0.0, 1.0);
        scale = bright > 4000.0 ? bright : min(Exposure_co_rcp / ((1.0 + 1.0 / 65504.0) - compMax), bright);
    }
    else
    {
        compMax = clamp(compMax, 0.0, 254.0 / 255.0);
        scale = Exposure_co_rcp / ((1.0 + 1.0 / 65504.0) - compMax);
    }
    float3 rgb = blended_f * scale;
    if (any(isnan(rgb)) || any(isinf(rgb)))
        rgb = float3(0, 0, 0);
    o.scene = float4(rgb, 1.0);
    return o;
}
