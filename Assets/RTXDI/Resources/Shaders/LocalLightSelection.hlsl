#ifndef RTXDI_MINIMAL_LOCAL_LIGHT_SELECTION
#define RTXDI_MINIMAL_LOCAL_LIGHT_SELECTION

#define RTXDI_LocalLightContextSamplingMode uint
#define RTXDI_LocalLightContextSamplingMode_UNIFORM 0
#if RTXDI_ENABLE_PRESAMPLING
#define RTXDI_LocalLightContextSamplingMode_RIS 1
#endif

void RTXDI_RandomlySelectLightUniformly(
    inout RAB_RandomSamplerState rng,
    RTXDI_LightBufferRegion region,
    out RAB_LightInfo lightInfo,
    out uint lightIndex,
    out float invSourcePdf)
{
    float rnd = RAB_GetNextRandom(rng);
    invSourcePdf = float(region.numLights);
    lightIndex = region.firstLightIndex + min(uint(floor(rnd * region.numLights)), region.numLights - 1);
    lightInfo = RAB_LoadLightInfo(lightIndex, false);
}

struct RTXDI_LocalLightSelectionContext
{
    RTXDI_LocalLightContextSamplingMode mode;

    #if RTXDI_ENABLE_PRESAMPLING
    RTXDI_RISTileInfo risTileInfo;
    #endif // RTXDI_ENABLE_PRESAMPLING
    RTXDI_LightBufferRegion lightBufferRegion;
};

RTXDI_LocalLightSelectionContext RTXDI_InitializeLocalLightSelectionContextUniform(RTXDI_LightBufferRegion lightBufferRegion)
{
    RTXDI_LocalLightSelectionContext ctx;
    ctx.mode                = RTXDI_LocalLightContextSamplingMode_UNIFORM;
    ctx.lightBufferRegion   = lightBufferRegion;
    return ctx;
}

void RTXDI_SelectNextLocalLight(
    RTXDI_LocalLightSelectionContext ctx,
    inout RAB_RandomSamplerState rng,
    out RAB_LightInfo lightInfo,
    out uint lightIndex,
    out float invSourcePdf)
{
    switch (ctx.mode)
    {
        #if RTXDI_ENABLE_PRESAMPLING
    case RTXDI_LocalLightContextSamplingMode_RIS:
        RTXDI_RandomlySelectLocalLightFromRISTile(rng, ctx.risTileInfo, lightInfo, lightIndex, invSourcePdf);
        break;
        #endif // RTXDI_ENABLE_PRESAMPLING
    default:
    case RTXDI_LocalLightContextSamplingMode_UNIFORM:
        RTXDI_RandomlySelectLightUniformly(rng, ctx.lightBufferRegion, lightInfo, lightIndex, invSourcePdf);
        break;
    }
}

#endif