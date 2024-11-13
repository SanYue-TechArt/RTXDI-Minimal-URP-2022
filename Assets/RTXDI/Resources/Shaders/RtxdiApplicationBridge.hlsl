#ifndef RTXDI_MINIMAL_RAB
#define RTXDI_MINIMAL_RAB

#include "Packages/com.unity.render-pipelines.universal@14.0.9/ShaderLibrary/Core.hlsl"
#include "Packages/com.unity.render-pipelines.universal/Shaders/Utils/Deferred.hlsl"
#include "ShaderParameters.hlsl"

// 作为ConstantBuffer使用，只有1个element
RWStructuredBuffer<ResamplingConstants> ResampleConstants;
#define g_Const ResampleConstants[0]

// 光照结果
RWTexture2D<float4> ShadingOutput;

RWStructuredBuffer<RTXDI_PackedDIReservoir> LightReservoirs;
#define RTXDI_LIGHT_RESERVOIR_BUFFER LightReservoirs

StructuredBuffer<RAB_LightInfo> LightDataBuffer;

RaytracingAccelerationStructure PolymorphicLightTLAS;
RaytracingAccelerationStructure SceneTLAS;

StructuredBuffer<uint> GeometryInstanceToLight;

// TODO:填充Buffer
Buffer<float2> NeighborOffsets;
#define RTXDI_NEIGHBOR_OFFSETS_BUFFER NeighborOffsets

TEXTURE2D_X(_CameraDepthTexture);
TEXTURE2D_X(_GBuffer0);
TEXTURE2D_X(_GBuffer1);
TEXTURE2D_X(_GBuffer2);

// TODO:填充Prev Info
TEXTURE2D_X(_PreviousCameraDepthTexture);
TEXTURE2D_X(_PreviousGBuffer0);
TEXTURE2D_X(_PreviousGBuffer1);
TEXTURE2D_X(_PreviousGBuffer2);

#include "TriangleLight.hlsl"

static const float kMinRoughness = 0.05f;

#define BACKGROUND_DEPTH 65504.f

// -------------------------------------
//               BSDF
// -------------------------------------
float Schlick_Fresnel(float F0, float VdotH)
{
    return F0 + (1 - F0) * pow(max(1 - VdotH, 0), 5);
}

float G_Smith_over_NdotV(float roughness, float NdotV, float NdotL)
{
    float alpha = square(roughness);
    float g1 = NdotV * sqrt(square(alpha) + (1.0 - square(alpha)) * square(NdotL));
    float g2 = NdotL * sqrt(square(alpha) + (1.0 - square(alpha)) * square(NdotV));
    return 2.0 * NdotL / (g1 + g2);
}

float3 GGX_times_NdotL(float3 V, float3 L, float3 N, float roughness, float3 F0)
{
    float3 H = normalize(L + V);

    float NoL = saturate(dot(N, L));
    float VoH = saturate(dot(V, H));
    float NoV = saturate(dot(N, V));
    float NoH = saturate(dot(N, H));

    if (NoL > 0)
    {
        float G = G_Smith_over_NdotV(roughness, NoV, NoL);
        float alpha = square(roughness);
        float D = square(alpha) / (PI * square(square(NoH) * square(alpha) + (1 - square(NoH))));

        float3 F = Schlick_Fresnel(F0, VoH);

        return F * (D * G / 4);
    }
    return 0;
}

float ImportanceSampleGGX_VNDF_PDF(float roughness, float3 N, float3 V, float3 L)
{
    float3 H = normalize(L + V);
    float NoH = saturate(dot(N, H));
    float VoH = saturate(dot(V, H));

    float alpha = square(roughness);
    float D = square(alpha) / (PI * square(square(NoH) * square(alpha) + (1 - square(NoH))));
    return (VoH > 0.0) ? D / (4.0 * VoH) : 0.0;
}

float2 SampleDisk(float2 random)
{
    float angle = 2 * PI * random.x;
    return float2(cos(angle), sin(angle)) * sqrt(random.y);
}

float3 SampleCosHemisphere(float2 random, out float solidAnglePdf)
{
    float2 tangential = SampleDisk(random);
    float elevation = sqrt(saturate(1.0 - random.y));

    solidAnglePdf = elevation / PI;

    return float3(tangential.xy, elevation);
}
// ----------------- End BSDF -----------------

// Constructs an orthonormal basis based on the provided normal.
// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
void ConstructONB(float3 normal, out float3 tangent, out float3 bitangent)
{
    float sign = (normal.z >= 0) ? 1 : -1;
    float a = -1.0 / (sign + normal.z);
    float b = normal.x * normal.y * a;
    tangent = float3(1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    bitangent = float3(b, sign + normal.y * normal.y * a, -normal.y);
}

// Returns the sampled H vector in tangent space, assuming N = (0, 0, 1).
float3 RTXDI_ImportanceSampleGGX(float2 random, float roughness)
{
    float alpha = square(roughness);

    float phi = 2 * PI * random.x;
    float cosTheta = sqrt((1 - random.y) / (1 + (square(alpha) - 1) * random.y));
    float sinTheta = sqrt(1 - cosTheta * cosTheta);

    float3 H;
    H.x = sinTheta * cos(phi);
    H.y = sinTheta * sin(phi);
    H.z = cosTheta;

    return H;
}

// A surface with enough information to evaluate BRDFs
struct RAB_Surface
{
    float3 worldPos;
    float3 viewDir;
    float viewDepth;
    float3 normal;
    float3 geoNormal;
    float3 diffuseAlbedo;
    float3 specularF0;
    float roughness;
    float diffuseProbability;
};

typedef RandomSamplerState RAB_RandomSamplerState;

float2 RAB_GetEnvironmentMapRandXYFromDir(float3 worldDir)
{
    return 0;
}

float RAB_EvaluateEnvironmentMapSamplingPdf(float3 L)
{
    // No Environment sampling
    return 0;
}

float RAB_EvaluateLocalLightSourcePdf(uint lightIndex)
{
    // Uniform pdf
    return 1.0 / g_Const.lightBufferParams.localLightBufferRegion.numLights;
}

float getSurfaceDiffuseProbability(RAB_Surface surface)
{
    float diffuseWeight = calcLuminance(surface.diffuseAlbedo);
    float specularWeight = calcLuminance(Schlick_Fresnel(surface.specularF0, dot(surface.viewDir, surface.normal)));
    float sumWeights = diffuseWeight + specularWeight;
    return sumWeights < 1e-7f ? 1.f : (diffuseWeight / sumWeights);
}

float3 worldToTangent(RAB_Surface surface, float3 w)
{
    // reconstruct tangent frame based off worldspace normal
    // this is ok for isotropic BRDFs
    // for anisotropic BRDFs, we need a user defined tangent
    float3 tangent;
    float3 bitangent;
    ConstructONB(surface.normal, tangent, bitangent);

    return float3(dot(bitangent, w), dot(tangent, w), dot(surface.normal, w));
}

float3 tangentToWorld(RAB_Surface surface, float3 h)
{
    // reconstruct tangent frame based off worldspace normal
    // this is ok for isotropic BRDFs
    // for anisotropic BRDFs, we need a user defined tangent
    float3 tangent;
    float3 bitangent;
    ConstructONB(surface.normal, tangent, bitangent);

    return bitangent * h.x + tangent * h.y + surface.normal * h.z;
}

RAB_RandomSamplerState RAB_InitRandomSampler(uint2 index, uint pass)
{
    return initRandomSampler(index, g_Const.frameIndex + pass * 13);
}

float RAB_GetNextRandom(inout RAB_RandomSamplerState rng)
{
    return sampleUniformRng(rng);
}

// Output an importanced sampled reflection direction from the BRDF given the view
// Return true if the returned direction is above the surface
bool RAB_GetSurfaceBrdfSample(RAB_Surface surface, inout RAB_RandomSamplerState rng, out float3 dir)
{
    float3 rand;
    rand.x = RAB_GetNextRandom(rng);
    rand.y = RAB_GetNextRandom(rng);
    rand.z = RAB_GetNextRandom(rng);
    if (rand.x < surface.diffuseProbability)
    {
        float pdf;
        float3 h = SampleCosHemisphere(rand.yz, pdf);
        dir = tangentToWorld(surface, h);
    }
    else
    {
        float3 h = RTXDI_ImportanceSampleGGX(rand.yz, max(surface.roughness, kMinRoughness));
        dir = reflect(-surface.viewDir, tangentToWorld(surface, h));
    }

    return dot(surface.normal, dir) > 0.f;
}

// Load a sample from the previous G-buffer.
RAB_Surface RAB_GetGBufferSurface(int2 pixelPosition, bool previousFrame)
{
    RAB_Surface surface = (RAB_Surface)0;

    // We do not have access to the current G-buffer in this sample because it's using
    // a single render pass with a fused resampling kernel, so just return an invalid surface.
    // This should never happen though, as the fused kernel doesn't call RAB_GetGBufferSurface(..., false)
    if (!previousFrame)
        return surface;

    if (any(pixelPosition >= _ScreenParams.xy))
        return surface;

    float d             = LOAD_TEXTURE2D_X_LOD(_PreviousCameraDepthTexture, pixelPosition, 0).x;
    surface.viewDepth   = LinearEyeDepth(d, _ZBufferParams);

    if(surface.viewDepth == BACKGROUND_DEPTH)
        return surface;

    float4 gbuffer0 = LOAD_TEXTURE2D_X_LOD(_PreviousGBuffer0, pixelPosition, 0);
    float4 gbuffer1 = LOAD_TEXTURE2D_X_LOD(_PreviousGBuffer1, pixelPosition, 0);
    float4 gbuffer2 = LOAD_TEXTURE2D_X_LOD(_PreviousGBuffer2, pixelPosition, 0);

    float2 positionNDC = float2((float)pixelPosition.x / _ScreenParams.x, (float)pixelPosition.y / _ScreenParams.y); 

    // TODO: 区分GeoNormal和PixelNormal？
    surface.normal              = normalize(UnpackNormal(gbuffer2.xyz));
    surface.geoNormal           = surface.normal;
    surface.diffuseAlbedo       = gbuffer0.rgb;
    surface.specularF0          = gbuffer1.rgb;
    surface.roughness           = 1.0f - gbuffer2.a;
    surface.worldPos            = ComputeWorldSpacePosition(positionNDC, d, UNITY_MATRIX_I_VP);
    surface.viewDir             = GetWorldSpaceNormalizeViewDir(surface.worldPos);
    surface.diffuseProbability  = getSurfaceDiffuseProbability(surface);

    return surface;
}

RAB_LightInfo RAB_EmptyLightInfo()
{
    return (RAB_LightInfo)0;
}

RAB_LightSample RAB_EmptyLightSample()
{
    return (RAB_LightSample)0;
}

RAB_LightSample RAB_SamplePolymorphicLight(RAB_LightInfo lightInfo, RAB_Surface surface, float2 uv)
{
    return TriangleLight::Create(lightInfo).calcSample(uv, surface.worldPos);
}

// Translate the light index between the current and previous frame.
// Do nothing as our lights are static in this sample.
int RAB_TranslateLightIndex(uint lightIndex, bool currentToPrevious)
{
    return int(lightIndex);
}

float RAB_LightSampleSolidAnglePdf(RAB_LightSample lightSample)
{
    return lightSample.solidAnglePdf;
}

bool RAB_IsAnalyticLightSample(RAB_LightSample lightSample)
{
    return false;
}

bool RAB_IsSurfaceValid(RAB_Surface surface)
{
    return surface.viewDepth != BACKGROUND_DEPTH;
}

void RAB_GetLightDirDistance(RAB_Surface surface, RAB_LightSample lightSample,
    out float3 o_lightDir,
    out float o_lightDistance)
{
    float3 toLight = lightSample.position - surface.worldPos;
    o_lightDistance = length(toLight);
    o_lightDir = toLight / o_lightDistance;
}

// Return PDF wrt solid angle for the BRDF in the given dir
float RAB_GetSurfaceBrdfPdf(RAB_Surface surface, float3 dir)
{
    float cosTheta = saturate(dot(surface.normal, dir));
    float diffusePdf = cosTheta / PI;
    float specularPdf = ImportanceSampleGGX_VNDF_PDF(max(surface.roughness, kMinRoughness), surface.normal, surface.viewDir, dir);
    float pdf = cosTheta > 0.f ? lerp(specularPdf, diffusePdf, surface.diffuseProbability) : 0.f;
    return pdf;
}

// Evaluate the surface BRDF and compute the weighted reflected radiance for the given light sample
float3 ShadeSurfaceWithLightSample(RAB_LightSample lightSample, RAB_Surface surface)
{
    // Ignore invalid light samples
    if (lightSample.solidAnglePdf <= 0)
        return 0;

    float3 L = normalize(lightSample.position - surface.worldPos);

    // Ignore light samples that are below the geometric surface (but above the normal mapped surface)
    if (dot(L, surface.geoNormal) <= 0)
        return 0;


    float3 V = surface.viewDir;
    
    // Evaluate the BRDF
    float diffuse = max(0, -dot(surface.normal, -L)) / PI;
    float3 specular = GGX_times_NdotL(V, L, surface.normal, max(surface.roughness, kMinRoughness), surface.specularF0);

    float3 reflectedRadiance = lightSample.radiance * (diffuse * surface.diffuseAlbedo + specular);

    return reflectedRadiance / lightSample.solidAnglePdf;
}

// Compute the target PDF (p-hat) for the given light sample relative to a surface
float RAB_GetLightSampleTargetPdfForSurface(RAB_LightSample lightSample, RAB_Surface surface)
{
    // Second-best implementation: the PDF is proportional to the reflected radiance.
    // The best implementation would be taking visibility into account,
    // but that would be prohibitively expensive.
    return calcLuminance(ShadeSurfaceWithLightSample(lightSample, surface));
}

// Load the packed light information from the buffer.
// Ignore the previousFrame parameter as our lights are static in this sample.
RAB_LightInfo RAB_LoadLightInfo(uint index, bool previousFrame)
{
    return LightDataBuffer[index];
}

struct PolymorphicLightRayPayload
{
    bool    hitLight;
    uint    lightIndex;
    float2  hitUV;
};

// Return true if anything was hit. If false, RTXDI will do environment map sampling
// o_lightIndex: If hit, must be a valid light index for RAB_LoadLightInfo, if no local light was hit, must be RTXDI_InvalidLightIndex
// randXY: The randXY that corresponds to the hit location and is the same used for RAB_SamplePolymorphicLight
bool RAB_TraceRayForLocalLight(float3 origin, float3 direction, float tMin, float tMax,
    out uint o_lightIndex, out float2 o_randXY)
{
    o_lightIndex    = RTXDI_InvalidLightIndex;
    o_randXY        = 0;

    RayDesc rayDesc;
    rayDesc.Origin      = origin;
    rayDesc.Direction   = direction;
    rayDesc.TMin        = tMin;
    rayDesc.TMax        = tMax;

    PolymorphicLightRayPayload payload;
    payload.hitLight    = false;
    payload.lightIndex  = RTXDI_InvalidLightIndex;

    TraceRay(PolymorphicLightTLAS, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 1, 0, rayDesc, payload);
    
    if (payload.hitLight)
    {
        o_lightIndex = payload.lightIndex;
        if (o_lightIndex != RTXDI_InvalidLightIndex)
        {
            float2 hitUV = payload.hitUV;
            o_randXY = randomFromBarycentric(hitUVToBarycentric(hitUV));
        }
    }

    return payload.hitLight;
}

struct VisibilityRayPayload
{
    bool isHit;
};

RayDesc setupVisibilityRay(RAB_Surface surface, RAB_LightSample lightSample, float offset = 0.001)
{
    float3 L = lightSample.position - surface.worldPos;

    RayDesc ray;
    ray.TMin        = offset;
    ray.TMax        = length(L) - offset;
    ray.Direction   = normalize(L);
    ray.Origin      = surface.worldPos;

    return ray;
}

// Tests the visibility between a surface and a light sample.
// Returns true if there is nothing between them.
bool RAB_GetConservativeVisibility(RAB_Surface surface, RAB_LightSample lightSample)
{
    RayDesc rayDesc = setupVisibilityRay(surface, lightSample);

    VisibilityRayPayload payload;
    payload.isHit = false;

    TraceRay(SceneTLAS, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 1, 0, rayDesc, payload);
    
    return !payload.isHit;
}

// Tests the visibility between a surface and a light sample on the previous frame.
// Since the scene is static in this sample app, it's equivalent to RAB_GetConservativeVisibility.
bool RAB_GetTemporalConservativeVisibility(RAB_Surface currentSurface, RAB_Surface previousSurface,
    RAB_LightSample lightSample)
{
    return RAB_GetConservativeVisibility(currentSurface, lightSample);
}

#endif