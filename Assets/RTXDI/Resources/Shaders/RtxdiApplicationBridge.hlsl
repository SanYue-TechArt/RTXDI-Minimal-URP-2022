#ifndef RTXDI_MINIMAL_RAB
#define RTXDI_MINIMAL_RAB

static const float kMinRoughness = 0.05f;

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

#endif