#ifndef RTXDI_MINIMAL_DI_RENDER_RAYGEN
#define RTXDI_MINIMAL_DI_RENDER_RAYGEN

[shader("raygeneration")]
void RtxdiRayGen()
{
    const RTXDI_LightBufferParameters lightBufferParams = g_Const.lightBufferParams;
    const uint2 pixelPosition   = uint2(DispatchRaysIndex().x, DispatchRaysIndex().y);
    const float2 positionNDC    = float2(pixelPosition.x / DispatchRaysDimensions().x, pixelPosition.y / DispatchRaysDimensions().y);
    const float2 screen_uv      = positionNDC;

    float d         = SAMPLE_TEXTURE2D_X_LOD(_CameraDepthTexture, sampler_PointClamp, screen_uv, 0).x;
    float4 gbuffer0 = SAMPLE_TEXTURE2D_X_LOD(_GBuffer0, sampler_PointClamp, screen_uv, 0);
    float4 gbuffer1 = SAMPLE_TEXTURE2D_X_LOD(_GBuffer1, sampler_PointClamp, screen_uv, 0);
    float4 gbuffer2 = SAMPLE_TEXTURE2D_X_LOD(_GBuffer2, sampler_PointClamp, screen_uv, 0);
    
    RAB_Surface primarySurface          = (RAB_Surface)0;
    primarySurface.worldPos             = ComputeWorldSpacePosition(positionNDC, d, UNITY_MATRIX_I_VP);
    primarySurface.viewDir              = GetWorldSpaceNormalizeViewDir(primarySurface.worldPos);
    primarySurface.viewDepth            = LinearEyeDepth(d, _ZBufferParams);
    primarySurface.normal               = normalize(UnpackNormal(gbuffer2.xyz));
    primarySurface.geoNormal            = primarySurface.normal;
    primarySurface.diffuseAlbedo        = gbuffer0.rgb;
    primarySurface.specularF0           = gbuffer1.rgb;
    primarySurface.roughness            = 1.0f - gbuffer2.a;
    primarySurface.diffuseProbability   = getSurfaceDiffuseProbability(primarySurface);

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

    // Resample BRDF samples.
    RAB_LightSample brdfSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir brdfReservoir = RTXDI_SampleBrdf(rng, primarySurface, sampleParams, lightBufferParams, brdfSample);
    bool selectBrdf = RTXDI_CombineDIReservoirs(reservoir, brdfReservoir, RAB_GetNextRandom(rng), brdfReservoir.targetPdf);
    if (selectBrdf)
    {
        lightSample = brdfSample;
    }

    ShadingOutput[pixelPosition] = float4(1,0,0,0);
    return;
}

#endif