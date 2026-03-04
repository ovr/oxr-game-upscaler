//============================================================================================================
//
//  SGSRv2 Upscale Pass — ported from glsl_2_pass_fs/sgsr2_upscale.fs
//
//  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//  SPDX-License-Identifier: BSD-3-Clause
//
//  Adapted for HLSL SM 5.0, DirectX NDC (Y-up flipped in VS).
//
//============================================================================================================

float FastLanczos(float base)
{
    float y = base - 1.0;
    float y2 = y * y;
    float y_temp = 0.75 * y + y2;
    return y_temp * y2;
}

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
    float  reset;               // 1
    uint   bSameCamera;         // 1 = 32 DWORDs total
};

Texture2D<float4> PrevOutput                  : register(t0);
Texture2D<float4> MotionDepthClipAlphaBuffer  : register(t1);
Texture2D<float4> InputColor                  : register(t2);
SamplerState      samp                        : register(s0);

struct VSOut
{
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD0;
};

float4 PS(VSOut input) : SV_Target
{
    float Biasmax_viewportXScale = scaleRatio.x;
    float scalefactor = scaleRatio.y;

    float2 Hruv = input.uv;

    float2 Jitteruv;
    Jitteruv.x = clamp(Hruv.x + jitterOffset.x * renderSizeRcp.x, 0.0, 1.0);
    Jitteruv.y = clamp(Hruv.y + jitterOffset.y * renderSizeRcp.y, 0.0, 1.0);

    int2 InputPos = int2(Jitteruv * renderSize);

    float3 mda = MotionDepthClipAlphaBuffer.SampleLevel(samp, Jitteruv, 0).xyz;
    float2 Motion = mda.xy;

    float2 PrevUV;
    PrevUV.x = clamp(-0.5 * Motion.x + Hruv.x, 0.0, 1.0);
    // DirectX Y convention (Y-up in NDC, flipped by VS)
    PrevUV.y = clamp(0.5 * Motion.y + Hruv.y, 0.0, 1.0);

    float depthfactor = mda.z;

    float3 HistoryColor = PrevOutput.SampleLevel(samp, PrevUV, 0).xyz;

    // Upsample and compute bounding box
    float4 Upsampledcw = float4(0, 0, 0, 0);
    float biasmax = Biasmax_viewportXScale;
    float biasmin = max(1.0, 0.3 + 0.3 * biasmax);
    float biasfactor = 0.25 * depthfactor;
    float kernelbias = lerp(biasmax, biasmin, biasfactor);
    float motion_viewport_len = length(Motion * outputSize);
    float curvebias = lerp(-2.0, -3.0, clamp(motion_viewport_len * 0.02, 0.0, 1.0));

    float3 rectboxcenter = float3(0, 0, 0);
    float3 rectboxvar = float3(0, 0, 0);
    float rectboxweight = 0.0;
    float2 srcpos = float2(InputPos) + float2(0.5, 0.5) - jitterOffset;

    kernelbias *= 0.5;
    float kernelbias2 = kernelbias * kernelbias;
    float2 srcpos_srcOutputPos = srcpos - Hruv * renderSize;
    float3 rectboxmin;
    float3 rectboxmax;

    // Sample 5 cross-pattern pixels
    float3 topMid = InputColor.Load(int3(InputPos + int2(0, 1), 0)).xyz;
    {
        float3 samplecolor = topMid;
        float2 baseoffset = srcpos_srcOutputPos + float2(0.0, 1.0);
        float baseoffset_dot = dot(baseoffset, baseoffset);
        float base = clamp(baseoffset_dot * kernelbias2, 0.0, 1.0);
        float weight = FastLanczos(base);
        Upsampledcw += float4(samplecolor * weight, weight);
        float boxweight = exp(baseoffset_dot * curvebias);
        rectboxmin = samplecolor;
        rectboxmax = samplecolor;
        float3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += samplecolor * wsample;
        rectboxweight += boxweight;
    }

    float3 rightMid = InputColor.Load(int3(InputPos + int2(1, 0), 0)).xyz;
    {
        float3 samplecolor = rightMid;
        float2 baseoffset = srcpos_srcOutputPos + float2(1.0, 0.0);
        float baseoffset_dot = dot(baseoffset, baseoffset);
        float base = clamp(baseoffset_dot * kernelbias2, 0.0, 1.0);
        float weight = FastLanczos(base);
        Upsampledcw += float4(samplecolor * weight, weight);
        float boxweight = exp(baseoffset_dot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        float3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += samplecolor * wsample;
        rectboxweight += boxweight;
    }

    float3 leftMid = InputColor.Load(int3(InputPos + int2(-1, 0), 0)).xyz;
    {
        float3 samplecolor = leftMid;
        float2 baseoffset = srcpos_srcOutputPos + float2(-1.0, 0.0);
        float baseoffset_dot = dot(baseoffset, baseoffset);
        float base = clamp(baseoffset_dot * kernelbias2, 0.0, 1.0);
        float weight = FastLanczos(base);
        Upsampledcw += float4(samplecolor * weight, weight);
        float boxweight = exp(baseoffset_dot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        float3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += samplecolor * wsample;
        rectboxweight += boxweight;
    }

    float3 centerMid = InputColor.Load(int3(InputPos, 0)).xyz;
    {
        float3 samplecolor = centerMid;
        float2 baseoffset = srcpos_srcOutputPos;
        float baseoffset_dot = dot(baseoffset, baseoffset);
        float base = clamp(baseoffset_dot * kernelbias2, 0.0, 1.0);
        float weight = FastLanczos(base);
        Upsampledcw += float4(samplecolor * weight, weight);
        float boxweight = exp(baseoffset_dot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        float3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += samplecolor * wsample;
        rectboxweight += boxweight;
    }

    float3 btmMid = InputColor.Load(int3(InputPos + int2(0, -1), 0)).xyz;
    {
        float3 samplecolor = btmMid;
        float2 baseoffset = srcpos_srcOutputPos + float2(0.0, -1.0);
        float baseoffset_dot = dot(baseoffset, baseoffset);
        float base = clamp(baseoffset_dot * kernelbias2, 0.0, 1.0);
        float weight = FastLanczos(base);
        Upsampledcw += float4(samplecolor * weight, weight);
        float boxweight = exp(baseoffset_dot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        float3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += samplecolor * wsample;
        rectboxweight += boxweight;
    }

    // FastLanczos() produces negative weights for base > 0.25 (negative lobes).
    // When all 5 samples have negative weights, total w is negative but xyz/w is
    // still correct (neg/neg = pos). We must preserve sign for the division but
    // use |w| as the temporal confidence weight.
    float div_w = Upsampledcw.w;
    if (abs(div_w) < 0.001) div_w = 0.001; // prevent div-by-zero

    // Variance-based color clamping
    rectboxweight = max(rectboxweight, 1e-6); // guard against zero before reciprocal
    rectboxweight = 1.0 / rectboxweight;
    rectboxcenter *= rectboxweight;
    rectboxvar *= rectboxweight;
    rectboxvar = sqrt(abs(rectboxvar - rectboxcenter * rectboxcenter));

    Upsampledcw.xyz = Upsampledcw.xyz / div_w;
    // Guard against NaN/Inf from Lanczos division — fallback to center sample
    if (any(isnan(Upsampledcw.xyz)) || any(isinf(Upsampledcw.xyz)))
        Upsampledcw.xyz = centerMid;
    Upsampledcw.xyz = clamp(Upsampledcw.xyz,
                            rectboxmin - float3(0.075, 0.075, 0.075),
                            rectboxmax + float3(0.075, 0.075, 0.075));
    // Guard rectbox clamp (NaN bounds propagate NaN through clamp on some GPUs)
    if (any(isnan(Upsampledcw.xyz)))
        Upsampledcw.xyz = centerMid;
    // Temporal confidence = absolute weight magnitude (negative lobes still carry info)
    Upsampledcw.w = max(abs(Upsampledcw.w), 0.001) * (1.0 / 3.0);

    float baseupdate = 1.0 - depthfactor;
    baseupdate = min(baseupdate, lerp(baseupdate, Upsampledcw.w * 10.0, clamp(10.0 * motion_viewport_len, 0.0, 1.0)));
    baseupdate = min(baseupdate, lerp(baseupdate, Upsampledcw.w, clamp(motion_viewport_len * 0.05, 0.0, 1.0)));
    float basealpha = baseupdate;

    const float EPSILON = 1.192e-07;
    float boxscale = max(depthfactor, clamp(motion_viewport_len * 0.05, 0.0, 1.0));
    float boxsize = lerp(scalefactor, 1.0, boxscale);
    float3 sboxvar = rectboxvar * boxsize;
    float3 boxmin = rectboxcenter - sboxvar;
    float3 boxmax = rectboxcenter + sboxvar;
    rectboxmax = min(rectboxmax, boxmax);
    rectboxmin = max(rectboxmin, boxmin);

    float3 clampedcolor = clamp(HistoryColor, rectboxmin, rectboxmax);
    float startLerpValue = minLerpContribution;
    if ((abs(mda.x) + abs(mda.y)) > 0.000001) startLerpValue = 0.0;
    float lerpcontribution = (any(rectboxmin > HistoryColor) || any(HistoryColor > rectboxmax)) ? startLerpValue : 1.0;

    HistoryColor = lerp(clampedcolor, HistoryColor, clamp(lerpcontribution, 0.0, 1.0));
    float basemin = min(basealpha, 0.1);
    basealpha = lerp(basemin, basealpha, clamp(lerpcontribution, 0.0, 1.0));

    // Blend current frame with history
    float alphasum = max(EPSILON, basealpha + Upsampledcw.w);
    float alpha = clamp(Upsampledcw.w / alphasum + reset, 0.0, 1.0);

    Upsampledcw.xyz = lerp(HistoryColor, Upsampledcw.xyz, alpha);

    // Final NaN guard — never write NaN to history buffer (prevents temporal spread)
    if (any(isnan(Upsampledcw.xyz)) || any(isinf(Upsampledcw.xyz)))
        Upsampledcw.xyz = centerMid;

    return float4(Upsampledcw.xyz, 1.0);
}
