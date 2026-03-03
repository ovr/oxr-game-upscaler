//============================================================================================================
//
//  SGSRv2 Convert Pass — ported from glsl_2_pass_fs/sgsr2_convert.fs
//
//  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//  SPDX-License-Identifier: BSD-3-Clause
//
//  Adapted for HLSL SM 5.0, Cyberpunk 2077 reverse-Z depth, UV-space motion vectors.
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
    float  reset;               // 1
    uint   bSameCamera;         // 1 = 32 DWORDs total
};

Texture2D<float>  InputDepth    : register(t0);
Texture2D<float4> InputVelocity : register(t1);
SamplerState      samp          : register(s0);

struct VSOut
{
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD0;
};

float4 PS(VSOut input) : SV_Target
{
    float2 texCoord = input.uv;
    uint2 InputPos = uint2(texCoord * renderSize);
    float2 gatherCoord = texCoord - float2(0.5, 0.5) * renderSizeRcp;

    // Texture gather to find nearest depth (4x4 neighbourhood via 4 gathers).
    // Cyberpunk uses reverse-Z: near=1, far=0. Nearest = max depth value.
    float4 btmLeft  = InputDepth.Gather(samp, gatherCoord);
    float2 v10      = float2(renderSizeRcp.x * 2.0, 0.0);
    float4 btmRight = InputDepth.Gather(samp, gatherCoord + v10);
    float2 v12      = float2(0.0, renderSizeRcp.y * 2.0);
    float4 topLeft  = InputDepth.Gather(samp, gatherCoord + v12);
    float2 v14      = float2(renderSizeRcp.x * 2.0, renderSizeRcp.y * 2.0);
    float4 topRight = InputDepth.Gather(samp, gatherCoord + v14);

    // Reverse-Z: use max() instead of min() to find nearest surface
    float maxC = max(max(max(btmLeft.z, btmRight.w), topLeft.y), topRight.x);
    float btmLeft4 = max(max(max(btmLeft.y, btmLeft.x), btmLeft.z), btmLeft.w);
    float btmLeftMax9 = max(topLeft.x, max(max(maxC, btmLeft4), btmRight.x));

    float depthclip = 0.0;
    // Reverse-Z: valid depth is > epsilon (far plane is 0)
    if (maxC > 1.0e-05)
    {
        float btmRight4 = max(max(max(btmRight.y, btmRight.x), btmRight.z), btmRight.w);
        float topLeft4  = max(max(max(topLeft.y, topLeft.x), topLeft.z), topLeft.w);
        float topRight4 = max(max(max(topRight.y, topRight.x), topRight.z), topRight.w);

        float Wdepth = 0.0;
        float Ksep = 1.37e-05;
        float Kfov = cameraFovAngleHor;
        float diagonal_length = length(renderSize);
        float Ksep_Kfov_diagonal = Ksep * Kfov * diagonal_length;

        // Reverse-Z: depth separation uses maxC (nearest) instead of (1-maxC)
        float Depthsep = Ksep_Kfov_diagonal * maxC;
        float EPSILON = 1.19e-07;
        Wdepth += clamp(Depthsep / (abs(maxC - btmLeft4)  + EPSILON), 0.0, 1.0);
        Wdepth += clamp(Depthsep / (abs(maxC - btmRight4) + EPSILON), 0.0, 1.0);
        Wdepth += clamp(Depthsep / (abs(maxC - topLeft4)  + EPSILON), 0.0, 1.0);
        Wdepth += clamp(Depthsep / (abs(maxC - topRight4) + EPSILON), 0.0, 1.0);
        depthclip = clamp(1.0 - Wdepth * 0.25, 0.0, 1.0);
    }

    // Motion vectors: Cyberpunk provides UV-space MVs scaled by motionVectorScale.
    // The scale is already baked into scaleRatio for the convert pass — we read raw
    // and multiply by 2 to convert UV-space → NDC-space motion.
    float4 EncodedVelocity = InputVelocity.Load(int3(InputPos, 0));

    float2 motion;
    float2 rawMV = EncodedVelocity.xy;
    if (dot(abs(rawMV), float2(1, 1)) > 0.0)
    {
        // Valid motion vector: convert UV→NDC (multiply by 2)
        motion = rawMV * 2.0;
    }
    else
    {
        // Zero/invalid MV: derive motion from clipToPrevClip matrix
        float2 ScreenPos = float2(2.0 * texCoord.x - 1.0, 1.0 - 2.0 * texCoord.y);
        float3 Position = float3(ScreenPos, btmLeftMax9);
        float4 PreClip = clipToPrevClip[3]
            + clipToPrevClip[2] * Position.z
            + clipToPrevClip[1] * ScreenPos.y
            + clipToPrevClip[0] * ScreenPos.x;
        float2 PreScreen = PreClip.xy / PreClip.w;
        motion = Position.xy - PreScreen;
    }

    return float4(motion, depthclip, 0.0);
}
