#ifndef RTXDI_MINIMAL_DI_RENDER_RAYGEN
#define RTXDI_MINIMAL_DI_RENDER_RAYGEN

[shader("raygeneration")]
void RtxdiRayGen()
{
    const RTXDI_LightBufferParameters lightBufferParams = g_Const.lightBufferParams;
    const uint2 pixelPosition = uint2(DispatchRaysIndex().x, DispatchRaysIndex().y);

    RAB_Surface primarySurface = (RAB_Surface)0;
    // TODO: Fill primary surface data.
    // worldPos: 从深度图重建，参考StencilDeferred.shader
    // viewDir: UnityGBuffer.hlsl -> InputDataFromGbufferAndWorldPosition
    // viewDepth: LinearEyeDepth
    // normal: 采GBuffer
    // geonormal: 暂时没有直接获取方法，可能需要额外渲一张geoNormal Buffer
    // diffuseAlbedo: 采颜色
    // specularF0: 采GBuffer
    // roughness: 采GBuffer
    // diffuseProbability: getSurfaceDiffuseProbability方法获取

    RTXDI_DIReservoir reservoir = RTXDI_EmptyDIReservoir();

    RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 1);

    RTXDI_SampleParameters sampleParams = RTXDI_InitSampleParameters(
        g_Const.numInitialSamples, // local light samples 
        0, // infinite light samples
        0, // environment map samples
        g_Const.numInitialBRDFSamples,
        g_Const.brdfCutoff,
        0.001f);

    // Generate the initial sample
    RAB_LightSample lightSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir localReservoir = RTXDI_SampleLocalLights(rng, rng, primarySurface,
        sampleParams, ReSTIRDI_LocalLightSamplingMode_UNIFORM, lightBufferParams.localLightBufferRegion, lightSample);
    RTXDI_CombineDIReservoirs(reservoir, localReservoir, 0.5, localReservoir.targetPdf);
}

#endif