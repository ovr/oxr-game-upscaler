//============================================================================================================
//
//  SGSRv2 3-Pass Convert — ported from 3Pass/sgsr_convert.h
//
//  Copyright (c) 2024-2025, Qualcomm Innovation Center, Inc. All rights reserved.
//  SPDX-License-Identifier: BSD-3-Clause
//
//  Adapted for HLSL SM 6.0 pixel shader, Cyberpunk 2077 reverse-Z depth, UV-space motion vectors.
//  Outputs MRT: RT0 = R32_UINT (packed YCoCg), RT1 = R16G16B16A16_FLOAT (motion, depth_bright, alpha)
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
    float  preExposure;         // 1 = 32 DWORDs total
};

Texture2D<float>  InputDepth    : register(t0);
Texture2D<float4> InputVelocity : register(t1);
Texture2D<float4> InputColor    : register(t2);
SamplerState      samp          : register(s0);   // linear clamp
SamplerState      pointSamp     : register(s1);   // point clamp

struct VSOut
{
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD0;
};

struct PSOut
{
    uint   ycocg : SV_Target0;   // R32_UINT — packed 11-11-10 YCoCg
    float4 mda   : SV_Target1;   // R16G16B16A16_FLOAT — motion.xy, depth_bright, alpha_mask
};

PSOut PS(VSOut input)
{
    float2 texCoord = input.uv;
    uint2 InputPos = uint2(texCoord * renderSize);
    float2 ViewportUV = texCoord;

    // Gather depth to find nearest (reverse-Z: near=1, far=0 → use max)
    float2 gatherCoord = float2(InputPos) * renderSizeRcp;

    int2 InputPosBtmRight = int2(1, 1) + int2(InputPos);
    float NearestZ = InputDepth[InputPosBtmRight].x;
    float4 topleft = InputDepth.GatherRed(pointSamp, gatherCoord);
    NearestZ = max(topleft.x, NearestZ);
    NearestZ = max(topleft.y, NearestZ);
    NearestZ = max(topleft.z, NearestZ);
    NearestZ = max(topleft.w, NearestZ);

    float2 topRight = InputDepth.GatherRed(pointSamp, gatherCoord + float2(renderSizeRcp.x, 0.0)).yz;
    NearestZ = max(topRight.x, NearestZ);
    NearestZ = max(topRight.y, NearestZ);

    float2 bottomLeft = InputDepth.GatherRed(pointSamp, gatherCoord + float2(0.0, renderSizeRcp.y)).xy;
    NearestZ = max(bottomLeft.x, NearestZ);
    NearestZ = max(bottomLeft.y, NearestZ);

    // Motion vectors: Cyberpunk provides UV-space MVs, multiply by 2 for NDC-space
    float4 EncodedVelocity = InputVelocity.Load(int3(InputPos, 0));

    float2 motion;
    float2 rawMV = EncodedVelocity.xy;
    if (dot(abs(rawMV), float2(1, 1)) > 0.0)
    {
        motion = rawMV * 2.0;
    }
    else
    {
        float2 ScreenPos = float2(2.0 * ViewportUV.x - 1.0, 1.0 - 2.0 * ViewportUV.y);
        float3 Position = float3(ScreenPos, NearestZ);
        float4 PreClip = clipToPrevClip[3]
            + clipToPrevClip[2] * Position.z
            + clipToPrevClip[1] * ScreenPos.y
            + clipToPrevClip[0] * ScreenPos.x;
        float2 PreScreen = PreClip.xy / PreClip.w;
        motion = Position.xy - PreScreen;
    }

    // Read scene color and tonemap
    float3 Colorrgb = InputColor.Load(int3(InputPos, 0)).xyz;

    // Simple tonemap: divide by max component + exposure reciprocal
    float Exposure_co_rcp = preExposure;
    float ColorMax = max(max(Colorrgb.x, Colorrgb.y), Colorrgb.z) + Exposure_co_rcp;
    Colorrgb /= ColorMax;

    // Encode brightness into depth (integer part = brightness * 0.001, fractional = depth)
    float depth_bright = floor(ColorMax * 0.001) + NearestZ;

    // RGB → YCoCg conversion (all values in [0,1] range)
    float3 Colorycocg;
    Colorycocg.x = 0.25 * (Colorrgb.x + 2.0 * Colorrgb.y + Colorrgb.z);
    Colorycocg.y = saturate(0.5 * Colorrgb.x + 0.5 - 0.5 * Colorrgb.z);
    Colorycocg.z = saturate(Colorycocg.x + Colorycocg.y - Colorrgb.x);

    // Pack YCoCg into 11-11-10 uint
    uint x11 = (uint)(Colorycocg.x * 2047.5);
    uint y11 = (uint)(Colorycocg.y * 2047.5);
    uint z10 = (uint)(Colorycocg.z * 1023.5);

    // No InputOpaqueColor available from Cyberpunk — set alpha_mask = 0
    float alpha_mask = 0.0;

    PSOut o;
    o.ycocg = (x11 << 21) | (y11 << 10) | z10;
    o.mda = float4(motion, depth_bright, alpha_mask);
    return o;
}
