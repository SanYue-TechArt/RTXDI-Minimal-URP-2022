#ifndef RTXDI_MINIMAL_SHADER_PARAMETERS
#define RTXDI_MINIMAL_SHADER_PARAMETERS

struct TriangleLightVertex
{
    float3 position;
    float3 normal;
    float4 tangent;
    float2 uv;
};

struct PrepareLightsTask
{
    float3  emissiveColor;
    uint    triangleCount;
    uint    lightBufferOffset;
};

/**
 * RAB: RTXDI Application Bridge
 */
struct RAB_LightInfo
{
    // uint4[0]
    float3 center;
    uint scalars; // 2x float16
    
    // uint4[1]
    uint2 radiance; // fp16x4
    uint direction1; // oct-encoded
    uint direction2; // oct-encoded

    float4 debug;
};

#endif