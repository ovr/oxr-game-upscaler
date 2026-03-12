//============================================================================================================
//
//  SGSRv2 3-Pass Activate — ported from 3Pass/sgsr_activate.h
//
//  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
//  SPDX-License-Identifier: BSD-3-Clause
//
//  Adapted for HLSL SM 6.0 pixel shader. Temporal depth clipping + luma history tracking.
//  Outputs MRT: RT0 = R16G16B16A16_FLOAT (motion, depthclip, alpha),
//               RT1 = R32_UINT (packed luma history)
//
//============================================================================================================

#define EPSILON 1.19e-07f

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

Texture2D<uint>   YCoCgColor              : register(t0);
Texture2D<float4> MotionDepthAlphaBuffer  : register(t1);
Texture2D<uint>   PrevLumaHistory         : register(t2);
SamplerState      samp                    : register(s0);   // linear clamp
SamplerState      pointSamp               : register(s1);   // point clamp

struct VSOut
{
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD0;
};

struct PSOut
{
    float4 mdca : SV_Target0;   // R16G16B16A16_FLOAT — motion.xy, depthclip, alpha_mask
    uint   luma : SV_Target1;   // R32_UINT — packed f16 luma history
};

float DecodeColorY(uint sample32)
{
    uint x11 = sample32 >> 21;
    return ((float)x11 * (1.0 / 2047.5));
}

PSOut PS(VSOut input)
{
    float2 texCoord = input.uv;
    uint2 DisThreadID = uint2(texCoord * renderSize);

    float2 ViewportUV = (float2(DisThreadID) + 0.5) * renderSizeRcp;
    float2 gatherCoord = ViewportUV + 0.5 * renderSizeRcp;
    uint luma_reference32 = YCoCgColor.GatherRed(pointSamp, gatherCoord).w;
    float luma_reference = DecodeColorY(luma_reference32);

    float4 mda = MotionDepthAlphaBuffer.Load(int3(DisThreadID, 0));
    float depth = frac(mda.z);
    float depth_base = mda.z - depth;
    float alphamask = mda.w;
    float2 motion = mda.xy;

    float2 PrevUV;
    PrevUV.x = -0.5 * motion.x + ViewportUV.x;
    PrevUV.y =  0.5 * motion.y + ViewportUV.y;

    float depthclip = 0.0;

    if (depth > 1.0e-05)
    {
        float2 Prevf_sample = PrevUV * renderSize - 0.5;
        float2 Prevfrac = Prevf_sample - floor(Prevf_sample);

        float OneMinusPrevfacx = (1.0 - Prevfrac.x);
        float Bilinweights[4] = {
            OneMinusPrevfacx - OneMinusPrevfacx * Prevfrac.y,
            (Prevfrac.x - Prevfrac.x * Prevfrac.y),
            OneMinusPrevfacx * (Prevfrac.y),
            (Prevfrac.x) * (Prevfrac.y)
        };

        float Wdepth = 0.0;
        float Ksep = 1.37e-05;
        float Kfov = cameraFovAngleHor;
        float diagonal_length = length(renderSize);
        float Ksep_Kfov_diagonal = Ksep * Kfov * diagonal_length;

        // Sample offsets for GatherBlue
        static const int2 sampleOffset[4] = {
            int2(0, 0),
            int2(0, 1),
            int2(1, 0),
            int2(1, 1),
        };

        for (int index = 0; index < 4; index += 2)
        {
            float4 gPrevdepth = MotionDepthAlphaBuffer.GatherBlue(pointSamp, PrevUV, sampleOffset[index]);
            float tdepth1 = max(frac(gPrevdepth.x), frac(gPrevdepth.y));
            float tdepth2 = max(frac(gPrevdepth.z), frac(gPrevdepth.w));
            float fPrevdepth = max(tdepth1, tdepth2);

            float Depthsep = Ksep_Kfov_diagonal * max(fPrevdepth, depth);
            float weight = Bilinweights[index];
            Wdepth += saturate(Depthsep / (abs(fPrevdepth - depth) + EPSILON)) * weight;

            float2 gPrevdepth2 = MotionDepthAlphaBuffer.GatherBlue(pointSamp, PrevUV, sampleOffset[index + 1]).xy;
            fPrevdepth = max(max(frac(gPrevdepth2.x), frac(gPrevdepth2.y)), tdepth1);
            Depthsep = Ksep_Kfov_diagonal * max(fPrevdepth, depth);
            weight = Bilinweights[index + 1];
            Wdepth += saturate(Depthsep / (abs(fPrevdepth - depth) + EPSILON)) * weight;
        }
        depthclip = saturate(1.0 - Wdepth);
    }

    // Luma history tracking
    float2 current_luma_diff;
    uint prev_lumadiff_pack = PrevLumaHistory.GatherRed(pointSamp, PrevUV).w;
    float2 prev_luma_diff;
    prev_luma_diff.x = f16tof32(prev_lumadiff_pack >> 16);
    prev_luma_diff.y = f16tof32(prev_lumadiff_pack & 0xffff);

    bool enable = false;
    if (depthclip + reset < 0.1)
    {
        enable = all(PrevUV >= 0.0) && all(PrevUV <= 1.0);
    }

    float luma_diff = luma_reference - prev_luma_diff.x;
    if (!enable)
    {
        current_luma_diff.x = 0.0;
        current_luma_diff.y = 0.0;
    }
    else
    {
        current_luma_diff.x = luma_reference;
        current_luma_diff.y = prev_luma_diff.y != 0.0
            ? (sign(luma_diff) == sign(prev_luma_diff.y)
                ? sign(luma_diff) * min(abs(prev_luma_diff.y), abs(luma_diff))
                : prev_luma_diff.y)
            : luma_diff;
    }

    alphamask = floor(alphamask)
        + 0.5 * (float)((current_luma_diff.x != 0.0) && (abs(current_luma_diff.y) != abs(luma_diff)));

    uint pack = (f32tof16(current_luma_diff.x) << 16) | f32tof16(current_luma_diff.y);
    depthclip = depthclip + depth_base;

    PSOut o;
    o.mdca = (half4)float4(motion, depthclip, alphamask);
    o.luma = pack;
    return o;
}
