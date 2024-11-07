#ifndef RTXDI_MINIMAL_INITIAL_SAMPLING_FUNCTIONS
#define RTXDI_MINIMAL_INITIAL_SAMPLING_FUNCTIONS

#include "DIReservoir.hlsl"
#include "LocalLightSelection.hlsl" 

struct RTXDI_SampleParameters
{    
    uint numLocalLightSamples;
    uint numInfiniteLightSamples;
    uint numEnvironmentMapSamples;
    uint numBrdfSamples;

    uint numMisSamples;
    float localLightMisWeight;
    float environmentMapMisWeight;
    float brdfMisWeight;
    float brdfCutoff;
    float brdfRayMinT;
};

// Sample parameters struct
// Defined so that so these can be compile time constants as defined by the user
// brdfCutoff Value in range [0,1] to determine how much to shorten BRDF rays. 0 to disable shortening
RTXDI_SampleParameters RTXDI_InitSampleParameters(
    uint numLocalLightSamples,
    uint numInfiniteLightSamples,
    uint numEnvironmentMapSamples,
    uint numBrdfSamples,
    float brdfCutoff RTXDI_DEFAULT(0.0),
    float brdfRayMinT RTXDI_DEFAULT(0.001f))
{
    RTXDI_SampleParameters result;
    result.numLocalLightSamples         = numLocalLightSamples;
    result.numInfiniteLightSamples      = numInfiniteLightSamples;
    result.numEnvironmentMapSamples     = numEnvironmentMapSamples;
    result.numBrdfSamples               = numBrdfSamples;

    result.numMisSamples            = numLocalLightSamples + numEnvironmentMapSamples + numBrdfSamples;
    result.localLightMisWeight      = float(numLocalLightSamples) / result.numMisSamples;
    result.environmentMapMisWeight  = float(numEnvironmentMapSamples) / result.numMisSamples;
    result.brdfMisWeight            = float(numBrdfSamples) / result.numMisSamples;
    result.brdfCutoff               = brdfCutoff;
    result.brdfRayMinT              = brdfRayMinT;

    return result;
}

// Heuristic to determine a max visibility ray length from a PDF wrt. solid angle.
float RTXDI_BrdfMaxDistanceFromPdf(float brdfCutoff, float pdf)
{
    const float kRayTMax = 3.402823466e+38F; // FLT_MAX
    return brdfCutoff > 0.f ? sqrt((1.f / brdfCutoff - 1.f) * pdf) : kRayTMax;
}

// Computes the multi importance sampling pdf for brdf and light sample.
// For light and BRDF PDFs wrt solid angle, blend between the two.
//      lightSelectionPdf is a dimensionless selection pdf
float RTXDI_LightBrdfMisWeight(RAB_Surface surface, RAB_LightSample lightSample,
    float lightSelectionPdf, float lightMisWeight, bool isEnvironmentMap,
    RTXDI_SampleParameters sampleParams)
{
    float lightSolidAnglePdf = RAB_LightSampleSolidAnglePdf(lightSample);
    if (sampleParams.brdfMisWeight == 0 || RAB_IsAnalyticLightSample(lightSample) ||
        lightSolidAnglePdf <= 0 || isinf(lightSolidAnglePdf) || isnan(lightSolidAnglePdf))
    {
        // BRDF samples disabled or we can't trace BRDF rays MIS with analytical lights
        return lightMisWeight * lightSelectionPdf;
    }

    float3 lightDir;
    float lightDistance;
    RAB_GetLightDirDistance(surface, lightSample, lightDir, lightDistance);

    // Compensate for ray shortening due to brdf cutoff, does not apply to environment map sampling
    float brdfPdf = RAB_GetSurfaceBrdfPdf(surface, lightDir);
    float maxDistance = RTXDI_BrdfMaxDistanceFromPdf(sampleParams.brdfCutoff, brdfPdf);
    if (!isEnvironmentMap && lightDistance > maxDistance)
        brdfPdf = 0.f;

    // Convert light selection pdf (unitless) to a solid angle measurement
    float sourcePdfWrtSolidAngle = lightSelectionPdf * lightSolidAnglePdf;

    // MIS blending against solid angle pdfs.
    float blendedPdfWrtSolidangle = lightMisWeight * sourcePdfWrtSolidAngle + sampleParams.brdfMisWeight * brdfPdf;

    // Convert back, RTXDI divides shading again by this term later
    return blendedPdfWrtSolidangle / lightSolidAnglePdf;
}

float2 RTXDI_RandomlySelectLocalLightUV(inout RAB_RandomSamplerState rng)
{
    float2 uv;
    uv.x = RAB_GetNextRandom(rng);
    uv.y = RAB_GetNextRandom(rng);
    return uv;
}

bool RTXDI_StreamLocalLightAtUVIntoReservoir(
    inout RAB_RandomSamplerState rng,
    RTXDI_SampleParameters sampleParams,
    RAB_Surface surface,
    uint lightIndex,
    float2 uv,
    float invSourcePdf,
    RAB_LightInfo lightInfo,
    inout RTXDI_DIReservoir state,
    inout RAB_LightSample o_selectedSample)
{
    RAB_LightSample candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, uv);
    float blendedSourcePdf = RTXDI_LightBrdfMisWeight(surface, candidateSample, 1.0 / invSourcePdf,
        sampleParams.localLightMisWeight, false, sampleParams);
    // 这里可以看到，目标PDF（也就是光源的PDF），本质上是计算光源相对该点的照明功率，计算好targetPDF后，targetPdf/proposalPdf将作为重采样样本的权值
    float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
    float risRnd = RAB_GetNextRandom(rng);

    if (blendedSourcePdf == 0)
    {
        return false;
    }
    bool selected = RTXDI_StreamSample(state, lightIndex, uv, risRnd, targetPdf, 1.0 / blendedSourcePdf);

    if (selected) {
        o_selectedSample = candidateSample;
    }
    return true;
}

RTXDI_LocalLightSelectionContext RTXDI_InitializeLocalLightSelectionContext(
    inout RAB_RandomSamplerState coherentRng,
    ReSTIRDI_LocalLightSamplingMode localLightSamplingMode,
    RTXDI_LightBufferRegion localLightBufferRegion
#if RTXDI_ENABLE_PRESAMPLING
    ,RTXDI_RISBufferSegmentParameters localLightRISBufferSegmentParams
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    ,ReGIR_Parameters regirParams
    ,RAB_Surface surface
#endif
#endif
    )
{
    RTXDI_LocalLightSelectionContext ctx;
    #if RTXDI_ENABLE_PRESAMPLING
    #if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    if (localLightSamplingMode == ReSTIRDI_LocalLightSamplingMode_REGIR_RIS)
    {
        ctx = RTXDI_InitializeLocalLightSelectionContextReGIRRIS(coherentRng, localLightBufferRegion, localLightRISBufferSegmentParams, regirParams, surface);
    }
    else
        #endif // RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
        if (localLightSamplingMode == ReSTIRDI_LocalLightSamplingMode_POWER_RIS)
        {
            ctx = RTXDI_InitializeLocalLightSelectionContextRIS(coherentRng, localLightRISBufferSegmentParams);
        }
        else
            #endif // RTXDI_ENABLE_PRESAMPLING
        {
            ctx = RTXDI_InitializeLocalLightSelectionContextUniform(localLightBufferRegion);
        }
    return ctx;
}

RTXDI_DIReservoir RTXDI_SampleLocalLightsInternal(
    inout RAB_RandomSamplerState rng,
    inout RAB_RandomSamplerState coherentRng,
    RAB_Surface surface,
    RTXDI_SampleParameters sampleParams,
    ReSTIRDI_LocalLightSamplingMode localLightSamplingMode,
    RTXDI_LightBufferRegion localLightBufferRegion,
#if RTXDI_ENABLE_PRESAMPLING
    RTXDI_RISBufferSegmentParameters localLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    ReGIR_Parameters regirParams,
#endif
#endif
    out RAB_LightSample o_selectedSample)
{
    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();

    // 在mini sample中，因为RTXDI_ENABLE_PRESAMPLING被禁用，因此ctx实际只是做了个赋值操作
    // ctx包含采样模式（localLightSamplingMode）以及灯光数据buffer（localLightBufferRegion）
    RTXDI_LocalLightSelectionContext lightSelectionContext = RTXDI_InitializeLocalLightSelectionContext(coherentRng, localLightSamplingMode, localLightBufferRegion
#if RTXDI_ENABLE_PRESAMPLING
    ,localLightRISBufferSegmentParams
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    ,regirParams
    ,surface
#endif
#endif
    );

    for (uint i = 0; i < sampleParams.numLocalLightSamples; i++)
    {
        uint lightIndex;
        RAB_LightInfo lightInfo;
        float invSourcePdf;

        // 首先，对于每一个像素点位置P来说，场景灯光按照其对点P的辐照贡献可以生成一个PDF
        // 如果我们不使用RIS，为了让样本更倾向贡献较大的灯光，那么我们就只能尝试按照灯光辐照贡献PDF进行采样，但问题是这个PDF的CDF反函数很难求得（因为PDF本身就是结合光源和BRDF的复杂函数）
        // 所以我们使用RIS，转而在一个更容易的分布中采样，然后对样本加权，这个“更容易的分布”在实践中可以是一个很辣鸡的分布，也没有问题
        
        // 可以看到这里只是均匀的生成随机数，然后随机选择一个灯光，然后得到invSourcePdf用于后续RIS加权
        RTXDI_SelectNextLocalLight(lightSelectionContext, rng, lightInfo, lightIndex, invSourcePdf);
        float2 uv = RTXDI_RandomlySelectLocalLightUV(rng);

        // 将选中的样本流入储层当中并立即执行RIS，这就是储层的流特性的好处，我们可以选择一个光后直接流到储层，而非像常规RIS那样必须要先选择若干个灯光，然后放一块执行RIS
        bool zeroPdf = RTXDI_StreamLocalLightAtUVIntoReservoir(rng, sampleParams, surface, lightIndex, uv, invSourcePdf, lightInfo, state, o_selectedSample);

        if (zeroPdf)
            continue;
    }

    RTXDI_FinalizeResampling(state, 1.0, sampleParams.numMisSamples);
    state.M = 1;

    return state;
}

RTXDI_DIReservoir RTXDI_SampleLocalLights(
    inout RAB_RandomSamplerState rng,
    inout RAB_RandomSamplerState coherentRng,
    RAB_Surface surface,
    RTXDI_SampleParameters sampleParams,
    ReSTIRDI_LocalLightSamplingMode localLightSamplingMode,
    RTXDI_LightBufferRegion localLightBufferRegion,
#if RTXDI_ENABLE_PRESAMPLING
    RTXDI_RISBufferSegmentParameters localLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    ReGIR_Parameters regirParams,
#endif
#endif
    out RAB_LightSample o_selectedSample)
{
    o_selectedSample = RAB_EmptyLightSample();

    if (localLightBufferRegion.numLights == 0)
        return RTXDI_EmptyDIReservoir();

    if (sampleParams.numLocalLightSamples == 0)
        return RTXDI_EmptyDIReservoir();

    return RTXDI_SampleLocalLightsInternal(rng, coherentRng, surface, sampleParams, localLightSamplingMode, localLightBufferRegion,
#if RTXDI_ENABLE_PRESAMPLING
    localLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    regirParams,
#endif
#endif
    o_selectedSample);
}

#endif