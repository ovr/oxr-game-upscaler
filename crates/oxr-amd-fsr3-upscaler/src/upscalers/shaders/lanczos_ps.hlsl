// Lanczos6 upscaler pixel shader
// Ported from Magpie's Lanczos.hlsl, which ports from:
// https://github.com/libretro/common-shaders/blob/master/windowed/shaders/lanczos6.cg

Texture2D<float4> src : register(t0);
SamplerState samp : register(s0);

cbuffer Params : register(b0) {
    float2 uvScale;    // constants 0,1 — consumed by VS
    float2 inputSize;  // constants 2,3 — color texture dimensions (width, height)
};

static const float ARStrength = 0.5;

#define FIX(c) max(abs(c), 1e-5)
#define PI 3.14159265359
#define min4(a, b, c, d) min(min(a, b), min(c, d))
#define max4(a, b, c, d) max(max(a, b), max(c, d))

float3 weight3(float x) {
    const float rcpRadius = 1.0f / 3.0f;
    float3 s = FIX(2.0 * PI * float3(x - 1.5, x - 0.5, x + 0.5));
    return sin(s) * sin(s * rcpRadius) * rcp(s * s);
}

float4 PS(float4 sv_pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_Target {
    // uv is in [0, uvScale] range; multiply by inputSize to get render-region pixel coords.
    float2 inputPt = 1.0 / inputSize;
    float2 pos = uv * inputSize;

    uint i, j;

    float2 f = frac(pos.xy + 0.5f);
    float3 linetaps1 = weight3(0.5f - f.x * 0.5f);
    float3 linetaps2 = weight3(1.0f - f.x * 0.5f);
    float3 columntaps1 = weight3(0.5f - f.y * 0.5f);
    float3 columntaps2 = weight3(1.0f - f.y * 0.5f);

    // Normalize taps so they sum to exactly 1.0.
    float suml = dot(linetaps1, float3(1, 1, 1)) + dot(linetaps2, float3(1, 1, 1));
    float sumc = dot(columntaps1, float3(1, 1, 1)) + dot(columntaps2, float3(1, 1, 1));
    linetaps1 /= suml;
    linetaps2 /= suml;
    columntaps1 /= sumc;
    columntaps2 /= sumc;

    pos -= f + 1.5f;

    float3 tap[6][6];

    [unroll]
    for (i = 0; i <= 4; i += 2) {
        [unroll]
        for (j = 0; j <= 4; j += 2) {
            // Convert pixel coords to UV in the full color texture.
            float2 tpos = (pos + uint2(i, j)) * inputPt;
            const float4 sr = src.GatherRed(samp,   tpos);
            const float4 sg = src.GatherGreen(samp, tpos);
            const float4 sb = src.GatherBlue(samp,  tpos);

            // GatherRed/Green/Blue return a 2x2 footprint in (w,z,x,y) = (TL,TR,BL,BR) order:
            //   w z
            //   x y
            tap[i][j]         = float3(sr.w, sg.w, sb.w);
            tap[i][j + 1]     = float3(sr.x, sg.x, sb.x);
            tap[i + 1][j]     = float3(sr.z, sg.z, sb.z);
            tap[i + 1][j + 1] = float3(sr.y, sg.y, sb.y);
        }
    }

    float3 color = float3(0, 0, 0);
    [unroll]
    for (i = 0; i <= 4; i += 2) {
        color +=
            (mul(linetaps1, float3x3(tap[0][i],     tap[2][i],     tap[4][i]))
           + mul(linetaps2, float3x3(tap[1][i],     tap[3][i],     tap[5][i]))) * columntaps1[i / 2]
          + (mul(linetaps1, float3x3(tap[0][i + 1], tap[2][i + 1], tap[4][i + 1]))
           + mul(linetaps2, float3x3(tap[1][i + 1], tap[3][i + 1], tap[5][i + 1]))) * columntaps2[i / 2];
    }

    // Anti-ringing: clamp toward the center 2x2 neighborhood to suppress ringing artifacts.
    float3 min_sample = min4(tap[2][2], tap[3][2], tap[2][3], tap[3][3]);
    float3 max_sample = max4(tap[2][2], tap[3][2], tap[2][3], tap[3][3]);
    color = lerp(color, clamp(color, min_sample, max_sample), ARStrength);

    return float4(color, 1);
}
