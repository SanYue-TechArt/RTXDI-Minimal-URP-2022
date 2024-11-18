#ifndef RTXDI_MINIMAL_DI_RENDER_RAYGEN
#define RTXDI_MINIMAL_DI_RENDER_RAYGEN

[shader("raygeneration")]
void RtxdiRayGen()
{
    const RTXDI_LightBufferParameters lightBufferParams = g_Const.lightBufferParams;
    const uint2 pixelPosition   = uint2(DispatchRaysIndex().x, DispatchRaysIndex().y);
    const float2 positionNDC    = float2((float)pixelPosition.x / DispatchRaysDimensions().x, (float)pixelPosition.y / DispatchRaysDimensions().y);
    const float2 screen_uv      = positionNDC;

    float d         = SAMPLE_TEXTURE2D_X_LOD(_CameraDepthTexture, sampler_PointClamp, screen_uv, 0).x;
    float4 gbuffer0 = SAMPLE_TEXTURE2D_X_LOD(_GBuffer0, sampler_PointClamp, screen_uv, 0);
    float4 gbuffer1 = SAMPLE_TEXTURE2D_X_LOD(_GBuffer1, sampler_PointClamp, screen_uv, 0);
    float4 gbuffer2 = SAMPLE_TEXTURE2D_X_LOD(_GBuffer2, sampler_PointClamp, screen_uv, 0);

    // TODO: 区分GeoNormal和PixelNormal？
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
    
    if (RAB_IsSurfaceValid(primarySurface))
    {
        RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 1);

        RTXDI_SampleParameters sampleParams = RTXDI_InitSampleParameters(
            g_Const.numInitialSamples,  // local light samples 
            0,                          // infinite light samples
            0,                          // environment map samples
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

        RTXDI_FinalizeResampling(reservoir, 1.0, 1.0);
        reservoir.M = 1;

        // BRDF was generated with a trace so no need to trace visibility again
        if (RTXDI_IsValidDIReservoir(reservoir) && !selectBrdf)
        {
            // See if the initial sample is visible from the surface
            if (!RAB_GetConservativeVisibility(primarySurface, lightSample))
            {
                // If not visible, discard the sample (but keep the M)
                RTXDI_StoreVisibilityInDIReservoir(reservoir, 0, true);
            }
        }
        
        if(true)
        {
            float2 mv           = LOAD_TEXTURE2D_X_LOD(_MotionVectorTexture, pixelPosition, 0).xy;
            mv                  *= DispatchRaysDimensions();
            float depthDiff     = LinearEyeDepth(LOAD_TEXTURE2D_X_LOD(_PreviousCameraDepthTexture, pixelPosition, 0).r, _ZBufferParams) - primarySurface.viewDepth;
            float3 motionVector = float3(-mv, depthDiff);
            
            // TODO: 填充参数
            RTXDI_DISpatioTemporalResamplingParameters stparams;
            stparams.screenSpaceMotion              = motionVector;
            stparams.sourceBufferIndex              = g_Const.inputBufferIndex;
            stparams.maxHistoryLength               = 20;
            stparams.biasCorrectionMode             = g_Const.unbiasedMode ? RTXDI_BIAS_CORRECTION_RAY_TRACED : RTXDI_BIAS_CORRECTION_BASIC;
            stparams.depthThreshold                 = 0.1;
            stparams.normalThreshold                = 0.5;
            stparams.numSamples                     = g_Const.numSpatialSamples + 1;
            stparams.numDisocclusionBoostSamples    = 0;
            stparams.samplingRadius                 = 32;
            stparams.enableVisibilityShortcut       = true;
            stparams.enablePermutationSampling      = true;
            stparams.discountNaiveSamples           = false;

            // This variable will receive the position of the sample reused from the previous frame.
            // It's only needed for gradient evaluation, ignore it here.
            int2 temporalSamplePixelPos = -1;

            // Call the resampling function, update the reservoir and lightSample variables
            reservoir = RTXDI_DISpatioTemporalResampling(pixelPosition, primarySurface, reservoir,
                    rng, g_Const.runtimeParams, g_Const.restirDIReservoirBufferParams, stparams, temporalSamplePixelPos, lightSample);
        }

        float3 shadingOutput = 0;

        // Shade the surface with the selected light sample
        if (RTXDI_IsValidDIReservoir(reservoir))
        {
            // Compute the correctly weighted reflected radiance
            shadingOutput = ShadeSurfaceWithLightSample(lightSample, primarySurface)
                          * RTXDI_GetDIReservoirInvPdf(reservoir);

            // Test if the selected light is visible from the surface
            bool visibility = RAB_GetConservativeVisibility(primarySurface, lightSample);

            // If not visible, discard the shading output and the light sample
            if (!visibility)
            {
                shadingOutput = 0;
                RTXDI_StoreVisibilityInDIReservoir(reservoir, 0, true);
            }
        }

        ShadingOutput[pixelPosition] = shadingOutput;   
    }
    else
    {
        ShadingOutput[pixelPosition] = 0.0f;
    }

    RTXDI_StoreDIReservoir(reservoir, g_Const.restirDIReservoirBufferParams, pixelPosition, g_Const.outputBufferIndex);
}

[shader("miss")]
void MissShader(inout RayPayload rayIntersection : SV_RayPayload)
{

}

#endif