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

    // Early Out: Not valid reservoir
    if (!RAB_IsSurfaceValid(primarySurface))
    {
        ShadingOutput[pixelPosition] = float4(0, 0, 0, 1);
        return;
    }

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

    // 测试：localReservoir为何无效？
    {
        RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();

        // 在mini sample中，因为RTXDI_ENABLE_PRESAMPLING被禁用，因此ctx实际只是做了个赋值操作
        // ctx包含采样模式（localLightSamplingMode）以及灯光数据buffer（localLightBufferRegion）
        RTXDI_LocalLightSelectionContext lightSelectionContext = RTXDI_InitializeLocalLightSelectionContext(rng, ReSTIRDI_LocalLightSamplingMode_UNIFORM, lightBufferParams.localLightBufferRegion
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

            if(i == 0)
            {
                lightInfo = LightDataBuffer[1];
                ShadingOutput[pixelPosition] = float4(Unpack_R16G16B16A16_FLOAT(lightInfo.radiance).rgb, 1);
                break;
            }

            // 将选中的样本流入储层当中并立即执行RIS，这就是储层的流特性的好处，我们可以选择一个光后直接流到储层，而非像常规RIS那样必须要先选择若干个灯光，然后放一块执行RIS
            bool zeroPdf = RTXDI_StreamLocalLightAtUVIntoReservoir(rng, sampleParams, primarySurface, lightIndex, uv, invSourcePdf, lightInfo, state, lightSample);

            if (zeroPdf)
                continue;
        }

        RTXDI_FinalizeResampling(state, 1.0, sampleParams.numMisSamples);
        state.M = 1;

        return;
    }

    // Resample BRDF samples.
    RAB_LightSample brdfSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir brdfReservoir = RTXDI_SampleBrdf(rng, primarySurface, sampleParams, lightBufferParams, brdfSample);
    bool selectBrdf = RTXDI_CombineDIReservoirs(reservoir, brdfReservoir, RAB_GetNextRandom(rng), brdfReservoir.targetPdf);
    /*if (selectBrdf)
    {
        lightSample = brdfSample;
    }*/

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

    ShadingOutput[pixelPosition] = float4(sampleParams.numLocalLightSamples.xxx, 1);

    //RTXDI_StoreDIReservoir(reservoir, g_Const.restirDIReservoirBufferParams, pixelPosition, g_Const.outputBufferIndex);
}

#endif