#ifndef RTXDI_MINIMAL_DI_RESAMPLING_FUCTIONS
#define RTXDI_MINIMAL_DI_RESAMPLING_FUCTIONS

#include "DIReservoir.hlsl"

// This macro can be defined in the including shader file to reduce code bloat
// and/or remove ray tracing calls from temporal and spatial resampling shaders
// if bias correction is not necessary.
#ifndef RTXDI_ALLOWED_BIAS_CORRECTION
#define RTXDI_ALLOWED_BIAS_CORRECTION RTXDI_BIAS_CORRECTION_RAY_TRACED
#endif

#ifndef RTXDI_NEIGHBOR_OFFSETS_BUFFER
#error "RTXDI_NEIGHBOR_OFFSETS_BUFFER must be defined to point to a Buffer<float2> type resource"
#endif

#define RTXDI_NAIVE_SAMPLING_M_THRESHOLD 2

// A structure that groups the application-provided settings for temporal resampling.
struct RTXDI_DITemporalResamplingParameters
{
    // Screen-space motion vector, computed as (previousPosition - currentPosition).
    // The X and Y components are measured in pixels.
    // The Z component is in linear depth units.
    float3 screenSpaceMotion;

    // The index of the reservoir buffer to pull the temporal samples from.
    uint sourceBufferIndex;

    // Maximum history length for temporal reuse, measured in frames.
    // Higher values result in more stable and high quality sampling, at the cost of slow reaction to changes.
    uint maxHistoryLength;

    // Controls the bias correction math for temporal reuse. Depending on the setting, it can add
    // some shader cost and one approximate shadow ray per pixel (or per two pixels if checkerboard sampling is enabled).
    // Ideally, these rays should be traced through the previous frame's BVH to get fully unbiased results.
    uint biasCorrectionMode;

    // Surface depth similarity threshold for temporal reuse.
    // If the previous frame surface's depth is within this threshold from the current frame surface's depth,
    // the surfaces are considered similar. The threshold is relative, i.e. 0.1 means 10% of the current depth.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float depthThreshold;

    // Surface normal similarity threshold for temporal reuse.
    // If the dot product of two surfaces' normals is higher than this threshold, the surfaces are considered similar.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float normalThreshold;

    // Allows the temporal resampling logic to skip the bias correction ray trace for light samples
    // reused from the previous frame. Only safe to use when invisible light samples are discarded
    // on the previous frame, then any sample coming from the previous frame can be assumed visible.
    bool enableVisibilityShortcut;

    // Enables permuting the pixels sampled from the previous frame in order to add temporal
    // variation to the output signal and make it more denoiser friendly.
    bool enablePermutationSampling;

    // Random number for permutation sampling that is the same for all pixels in the frame
    uint uniformRandomNumber;
};

RTXDI_DIReservoir RTXDI_DITemporalResampling(
    uint2 pixelPosition,
    RAB_Surface surface,
    RTXDI_DIReservoir curSample,
    inout RAB_RandomSamplerState rng,
    RTXDI_RuntimeParameters params,
    RTXDI_ReservoirBufferParameters reservoirParams,
    RTXDI_DITemporalResamplingParameters tparams,
    out int2 temporalSamplePixelPos,
    inout RAB_LightSample selectedLightSample)
{
    // For temporal reuse, there's only a pair of samples; pairwise and basic MIS are essentially identical
    if (tparams.biasCorrectionMode == RTXDI_BIAS_CORRECTION_PAIRWISE)
    {
        tparams.biasCorrectionMode = RTXDI_BIAS_CORRECTION_BASIC;
    }

    uint historyLimit = min(RTXDI_PackedDIReservoir_MaxM, uint(tparams.maxHistoryLength * curSample.M));

    int selectedLightPrevID = -1;

    if (RTXDI_IsValidDIReservoir(curSample))
    {
        selectedLightPrevID = RAB_TranslateLightIndex(RTXDI_GetDIReservoirLightIndex(curSample), true);
    }

    temporalSamplePixelPos = int2(-1, -1);

    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();
    RTXDI_CombineDIReservoirs(state, curSample, /* random = */ 0.5, curSample.targetPdf);

    // Backproject this pixel to last frame
    float3 motion = tparams.screenSpaceMotion;
    
    if (!tparams.enablePermutationSampling)
    {
        motion.xy += float2(RAB_GetNextRandom(rng), RAB_GetNextRandom(rng)) - 0.5;
    }

    float2 reprojectedSamplePosition = float2(pixelPosition) + motion.xy;
    int2 prevPos = int2(round(reprojectedSamplePosition));

    float expectedPrevLinearDepth = surface.viewDepth + motion.z;

    RAB_Surface temporalSurface = (RAB_Surface)0;
    bool foundNeighbor = false;
    const float radius = (params.activeCheckerboardField == 0) ? 4 : 8;
    int2 spatialOffset = int2(0, 0);

    // Try to find a matching surface in the neighborhood of the reprojected pixel
    for(int i = 0; i < 9; i++)
    {
        int2 offset = int2(0, 0);
        if(i > 0)
        {
            offset.x = int((RAB_GetNextRandom(rng) - 0.5) * radius);
            offset.y = int((RAB_GetNextRandom(rng) - 0.5) * radius);
        }

        int2 idx = prevPos + offset;
        if (tparams.enablePermutationSampling && i == 0)
        {
            RTXDI_ApplyPermutationSampling(idx, tparams.uniformRandomNumber);
        }

        RTXDI_ActivateCheckerboardPixel(idx, true, params.activeCheckerboardField);

        // Grab shading / g-buffer data from last frame
        temporalSurface = RAB_GetGBufferSurface(idx, true);
        if (!RAB_IsSurfaceValid(temporalSurface))
            continue;
        
        // Test surface similarity, discard the sample if the surface is too different.
        if (!RTXDI_IsValidNeighbor(
            surface.normal, temporalSurface.normal, 
            expectedPrevLinearDepth, temporalSurface.viewDepth, 
            tparams.normalThreshold, tparams.depthThreshold))
            continue;

        spatialOffset = idx - prevPos;
        prevPos = idx;
        foundNeighbor = true;

        break;
    }

    bool selectedPreviousSample = false;
    float previousM = 0;

    if (foundNeighbor)
    {
        // Resample the previous frame sample into the current reservoir, but reduce the light's weight
        // according to the bilinear weight of the current pixel
        uint2 prevReservoirPos = RTXDI_PixelPosToReservoirPos(prevPos, params.activeCheckerboardField);
        RTXDI_DIReservoir prevSample = RTXDI_LoadDIReservoir(reservoirParams,
            prevReservoirPos, tparams.sourceBufferIndex);
        prevSample.M = min(prevSample.M, historyLimit);
        prevSample.spatialDistance += spatialOffset;
        prevSample.age += 1;

        uint originalPrevLightID = RTXDI_GetDIReservoirLightIndex(prevSample);

        // Map the light ID from the previous frame into the current frame, if it still exists
        if (RTXDI_IsValidDIReservoir(prevSample))
        {
            if (prevSample.age <= 1)
            {
                temporalSamplePixelPos = prevPos;
            }

            int mappedLightID = RAB_TranslateLightIndex(RTXDI_GetDIReservoirLightIndex(prevSample), false);

            if (mappedLightID < 0)
            {
                // Kill the reservoir
                prevSample.weightSum = 0;
                prevSample.lightData = 0;
            }
            else
            {
                // Sample is valid - modify the light ID stored
                prevSample.lightData = mappedLightID | RTXDI_DIReservoir_LightValidBit;
            }
        }

        previousM = prevSample.M;

        float weightAtCurrent = 0;
        RAB_LightSample candidateLightSample = RAB_EmptyLightSample();
        if (RTXDI_IsValidDIReservoir(prevSample))
        {
            const RAB_LightInfo candidateLight = RAB_LoadLightInfo(RTXDI_GetDIReservoirLightIndex(prevSample), false);

            candidateLightSample = RAB_SamplePolymorphicLight(
                candidateLight, surface, RTXDI_GetDIReservoirSampleUV(prevSample));

            weightAtCurrent = RAB_GetLightSampleTargetPdfForSurface(candidateLightSample, surface);
        }

        bool sampleSelected = RTXDI_CombineDIReservoirs(state, prevSample, RAB_GetNextRandom(rng), weightAtCurrent);
        if(sampleSelected)
        {
            selectedPreviousSample = true;
            selectedLightPrevID = int(originalPrevLightID);
            selectedLightSample = candidateLightSample;
        }
    }

#if RTXDI_ALLOWED_BIAS_CORRECTION >= RTXDI_BIAS_CORRECTION_BASIC
    if (tparams.biasCorrectionMode >= RTXDI_BIAS_CORRECTION_BASIC)
    {
        // Compute the unbiased normalization term (instead of using 1/M)
        float pi = state.targetPdf;
        float piSum = state.targetPdf * curSample.M;
        
        if (RTXDI_IsValidDIReservoir(state) && selectedLightPrevID >= 0 && previousM > 0)
        {
            float temporalP = 0;

            const RAB_LightInfo selectedLightPrev = RAB_LoadLightInfo(selectedLightPrevID, true);

            // Get the PDF of the sample RIS selected in the first loop, above, *at this neighbor* 
            const RAB_LightSample selectedSampleAtTemporal = RAB_SamplePolymorphicLight(
                selectedLightPrev, temporalSurface, RTXDI_GetDIReservoirSampleUV(state));
        
            temporalP = RAB_GetLightSampleTargetPdfForSurface(selectedSampleAtTemporal, temporalSurface);

#if RTXDI_ALLOWED_BIAS_CORRECTION >= RTXDI_BIAS_CORRECTION_RAY_TRACED
            if (tparams.biasCorrectionMode == RTXDI_BIAS_CORRECTION_RAY_TRACED && temporalP > 0 && (!selectedPreviousSample || !tparams.enableVisibilityShortcut))
            {
                if (!RAB_GetTemporalConservativeVisibility(surface, temporalSurface, selectedSampleAtTemporal))
                {
                    temporalP = 0;
                }
            }
#endif

            pi = selectedPreviousSample ? temporalP : pi;
            piSum += temporalP * previousM;
        }

        RTXDI_FinalizeResampling(state, pi, piSum);
    }
    else
#endif
    {
        RTXDI_FinalizeResampling(state, 1.0, state.M);
    }

    return state;
}

// A structure that groups the application-provided settings for spatio-temporal resampling.
struct RTXDI_DISpatioTemporalResamplingParameters
{
    // Screen-space motion vector, computed as (previousPosition - currentPosition).
    // The X and Y components are measured in pixels.
    // The Z component is in linear depth units.
    float3 screenSpaceMotion;

    // The index of the reservoir buffer to pull the temporal samples from.
    uint sourceBufferIndex;

    // Maximum history length for temporal reuse, measured in frames.
    // Higher values result in more stable and high quality sampling, at the cost of slow reaction to changes.
    uint maxHistoryLength;

    // Controls the bias correction math for temporal reuse. Depending on the setting, it can add
    // some shader cost and one approximate shadow ray per pixel (or per two pixels if checkerboard sampling is enabled).
    // Ideally, these rays should be traced through the previous frame's BVH to get fully unbiased results.
    uint biasCorrectionMode;

    // Surface depth similarity threshold for temporal reuse.
    // If the previous frame surface's depth is within this threshold from the current frame surface's depth,
    // the surfaces are considered similar. The threshold is relative, i.e. 0.1 means 10% of the current depth.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float depthThreshold;

    // Surface normal similarity threshold for temporal reuse.
    // If the dot product of two surfaces' normals is higher than this threshold, the surfaces are considered similar.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float normalThreshold;

    // Number of neighbor pixels considered for resampling (1-32)
    // Some of the may be skipped if they fail the surface similarity test.
    uint numSamples;

    // Number of neighbor pixels considered when there is no temporal surface (1-32)
    // Setting this parameter equal or lower than `numSpatialSamples` effectively
    // disables the disocclusion boost.
    uint numDisocclusionBoostSamples;

    // Screen-space radius for spatial resampling, measured in pixels.
    float samplingRadius;

    // Allows the temporal resampling logic to skip the bias correction ray trace for light samples
    // reused from the previous frame. Only safe to use when invisible light samples are discarded
    // on the previous frame, then any sample coming from the previous frame can be assumed visible.
    bool enableVisibilityShortcut;

    // Enables permuting the pixels sampled from the previous frame in order to add temporal
    // variation to the output signal and make it more denoiser friendly.
    bool enablePermutationSampling;

    // Enables the comparison of surface materials before taking a surface into resampling.
    bool enableMaterialSimilarityTest;

    // Prevents samples which are from the current frame or have no reasonable temporal history merged being spread to neighbors
    bool discountNaiveSamples;

    // Random number for permutation sampling that is the same for all pixels in the frame
    uint uniformRandomNumber;
};

RTXDI_DIReservoir RTXDI_DISpatioTemporalResampling(
    uint2 pixelPosition,
    RAB_Surface surface,
    RTXDI_DIReservoir curSample,
    inout RAB_RandomSamplerState rng,
    RTXDI_RuntimeParameters params,
    RTXDI_ReservoirBufferParameters reservoirParams,
    RTXDI_DISpatioTemporalResamplingParameters stparams,
    out int2 temporalSamplePixelPos,
    inout RAB_LightSample selectedLightSample)
{
    // TODO: Pairwise?
    /*if (stparams.biasCorrectionMode == RTXDI_BIAS_CORRECTION_PAIRWISE)
    {
        return RTXDI_DISpatioTemporalResamplingWithPairwiseMIS(pixelPosition, surface,
            curSample, rng, params, reservoirParams, stparams, temporalSamplePixelPos, selectedLightSample);
    }*/

    uint historyLimit = min(RTXDI_PackedDIReservoir_MaxM, uint(stparams.maxHistoryLength * curSample.M));

    int selectedLightPrevID = -1;

    if (RTXDI_IsValidDIReservoir(curSample))
    {
        selectedLightPrevID = RAB_TranslateLightIndex(RTXDI_GetDIReservoirLightIndex(curSample), true);
    }

    temporalSamplePixelPos = int2(-1, -1);

    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();
    RTXDI_CombineDIReservoirs(state, curSample, /* random = */ 0.5, curSample.targetPdf);

    uint startIdx = uint(RAB_GetNextRandom(rng) * params.neighborOffsetMask);

    // Backproject this pixel to last frame
    float3 motion = stparams.screenSpaceMotion;

    if (!stparams.enablePermutationSampling)
    {
        motion.xy += float2(RAB_GetNextRandom(rng), RAB_GetNextRandom(rng)) - 0.5;
    }

    float2 reprojectedSamplePosition = float2(pixelPosition) + motion.xy;
    int2 prevPos = int2(round(reprojectedSamplePosition));

    float expectedPrevLinearDepth = surface.viewDepth + motion.z;

    int i;

    RAB_Surface temporalSurface = (RAB_Surface)0;
    bool foundTemporalSurface = false;
    const float temporalSearchRadius = (params.activeCheckerboardField == 0) ? 4 : 8;
    int2 temporalSpatialOffset = int2(0, 0);

    // Try to find a matching surface in the neighborhood of the reprojected pixel
    for (i = 0; i < 9; i++)
    {
        int2 offset = int2(0, 0);
        if (i > 0)
        {
            offset.x = int((RAB_GetNextRandom(rng) - 0.5) * temporalSearchRadius);
            offset.y = int((RAB_GetNextRandom(rng) - 0.5) * temporalSearchRadius);
        }

        int2 idx = prevPos + offset;

        if (stparams.enablePermutationSampling && i == 0)
        {
            RTXDI_ApplyPermutationSampling(idx, stparams.uniformRandomNumber);
        }

        RTXDI_ActivateCheckerboardPixel(idx, true, params.activeCheckerboardField);

        // Grab shading / g-buffer data from last frame
        temporalSurface = RAB_GetGBufferSurface(idx, true);
        if (!RAB_IsSurfaceValid(temporalSurface))
            continue;
        
        // Test surface similarity, discard the sample if the surface is too different.
        if (!RTXDI_IsValidNeighbor(
            surface.normal, temporalSurface.normal, 
            expectedPrevLinearDepth, temporalSurface.viewDepth, 
            stparams.normalThreshold, stparams.depthThreshold))
            continue;

        temporalSpatialOffset = idx - prevPos;
        foundTemporalSurface = true;
        break;
    }

    // Clamp the sample count at 32 to make sure we can keep the neighbor mask in an uint (cachedResult)
    uint numSamples = clamp(stparams.numSamples, 1, 32);

    // Apply disocclusion boost if there is no temporal surface
    if (!foundTemporalSurface)
        numSamples = clamp(stparams.numDisocclusionBoostSamples, numSamples, 32);

    // We loop through neighbors twice.  Cache the validity / edge-stopping function
    //   results for the 2nd time through.
    uint cachedResult = 0;

    // Since we're using our bias correction scheme, we need to remember which light selection we made
    int selected = -1;

    // Walk the specified number of neighbors, resampling using RIS
    for (i = 0; i < numSamples; ++i)
    {
        int2 spatialOffset, idx;

        // Get screen-space location of neighbor
        if (i == 0 && foundTemporalSurface)
        {
            spatialOffset = temporalSpatialOffset;
            idx = prevPos + spatialOffset;
        }
        else
        {
            uint sampleIdx = (startIdx + i) & params.neighborOffsetMask;
            spatialOffset = int2(float2(RTXDI_NEIGHBOR_OFFSETS_BUFFER[sampleIdx].xy) * stparams.samplingRadius);

            idx = prevPos + spatialOffset;

            idx = RAB_ClampSamplePositionIntoView(idx, true);

            RTXDI_ActivateCheckerboardPixel(idx, true, params.activeCheckerboardField);

            temporalSurface = RAB_GetGBufferSurface(idx, true);

            if (!RAB_IsSurfaceValid(temporalSurface))
                continue;

            if (!RTXDI_IsValidNeighbor(surface.normal, temporalSurface.normal, 
                surface.viewDepth, temporalSurface.viewDepth, 
                stparams.normalThreshold, stparams.depthThreshold))
                continue;

            if (stparams.enableMaterialSimilarityTest && !RAB_AreMaterialsSimilar(surface, temporalSurface))
                continue;
        }
        
        cachedResult |= (1u << uint(i));

        uint2 neighborReservoirPos = RTXDI_PixelPosToReservoirPos(idx, params.activeCheckerboardField);

        RTXDI_DIReservoir prevSample = RTXDI_LoadDIReservoir(reservoirParams,
            neighborReservoirPos, stparams.sourceBufferIndex);

        if (RTXDI_IsValidDIReservoir(prevSample))
        {
            if (stparams.discountNaiveSamples && prevSample.M <= RTXDI_NAIVE_SAMPLING_M_THRESHOLD)
                continue;
        }

        prevSample.M = min(prevSample.M, historyLimit);
        prevSample.spatialDistance += spatialOffset;
        prevSample.age += 1;

        uint originalPrevLightID = RTXDI_GetDIReservoirLightIndex(prevSample);

        // Map the light ID from the previous frame into the current frame, if it still exists
        if (RTXDI_IsValidDIReservoir(prevSample))
        {   
            if (i == 0 && foundTemporalSurface && prevSample.age <= 1)
            {
                temporalSamplePixelPos = idx;
            }

            int mappedLightID = RAB_TranslateLightIndex(RTXDI_GetDIReservoirLightIndex(prevSample), false);

            if (mappedLightID < 0)
            {
                // Kill the reservoir
                prevSample.weightSum = 0;
                prevSample.lightData = 0;
            }
            else
            {
                // Sample is valid - modify the light ID stored
                prevSample.lightData = mappedLightID | RTXDI_DIReservoir_LightValidBit;
            }
        }

        RAB_LightInfo candidateLight;

        // Load that neighbor's RIS state, do resampling
        float neighborWeight = 0;
        RAB_LightSample candidateLightSample = RAB_EmptyLightSample();
        if (RTXDI_IsValidDIReservoir(prevSample))
        {   
            candidateLight = RAB_LoadLightInfo(RTXDI_GetDIReservoirLightIndex(prevSample), false);
            
            candidateLightSample = RAB_SamplePolymorphicLight(
                candidateLight, surface, RTXDI_GetDIReservoirSampleUV(prevSample));
            
            neighborWeight = RAB_GetLightSampleTargetPdfForSurface(candidateLightSample, surface);
        }

        if (RTXDI_CombineDIReservoirs(state, prevSample, RAB_GetNextRandom(rng), neighborWeight))
        {
            selected = i;
            selectedLightPrevID = int(originalPrevLightID);
            selectedLightSample = candidateLightSample;
        }
    }

    if (RTXDI_IsValidDIReservoir(state))
    {
#if RTXDI_ALLOWED_BIAS_CORRECTION >= RTXDI_BIAS_CORRECTION_BASIC
        if (stparams.biasCorrectionMode >= RTXDI_BIAS_CORRECTION_BASIC)
        {
            // Compute the unbiased normalization term (instead of using 1/M)
            float pi = state.targetPdf;
            float piSum = state.targetPdf * curSample.M;

            if (selectedLightPrevID >= 0)
            {
                const RAB_LightInfo selectedLightPrev = RAB_LoadLightInfo(selectedLightPrevID, true);

                // To do this, we need to walk our neighbors again
                for (i = 0; i < numSamples; ++i)
                {
                    // If we skipped this neighbor above, do so again.
                    if ((cachedResult & (1u << uint(i))) == 0) continue;

                    uint sampleIdx = (startIdx + i) & params.neighborOffsetMask;

                    // Get the screen-space location of our neighbor
                    int2 spatialOffset = (i == 0 && foundTemporalSurface) 
                        ? temporalSpatialOffset 
                        : int2(float2(RTXDI_NEIGHBOR_OFFSETS_BUFFER[sampleIdx].xy) * stparams.samplingRadius);
                    int2 idx = prevPos + spatialOffset;

                    if (!(i == 0 && foundTemporalSurface))
                    {
                        idx = RAB_ClampSamplePositionIntoView(idx, true);
                    }

                    RTXDI_ActivateCheckerboardPixel(idx, true, params.activeCheckerboardField);

                    // Load our neighbor's G-buffer
                    RAB_Surface neighborSurface = RAB_GetGBufferSurface(idx, true);
                    
                    // Get the PDF of the sample RIS selected in the first loop, above, *at this neighbor* 
                    const RAB_LightSample selectedSampleAtNeighbor = RAB_SamplePolymorphicLight(
                        selectedLightPrev, neighborSurface, RTXDI_GetDIReservoirSampleUV(state));

                    float ps = RAB_GetLightSampleTargetPdfForSurface(selectedSampleAtNeighbor, neighborSurface);

#if RTXDI_ALLOWED_BIAS_CORRECTION >= RTXDI_BIAS_CORRECTION_RAY_TRACED
                                                                                                              // TODO:  WHY?
                    if (stparams.biasCorrectionMode == RTXDI_BIAS_CORRECTION_RAY_TRACED && ps > 0 && (selected != i || i != 0 || !stparams.enableVisibilityShortcut))
                    {
                        RAB_Surface fallbackSurface;
                        if (i == 0 && foundTemporalSurface)
                            fallbackSurface = surface;
                        else
                            fallbackSurface = neighborSurface;

                        if (!RAB_GetTemporalConservativeVisibility(fallbackSurface, neighborSurface, selectedSampleAtNeighbor))
                        {
                            ps = 0;
                        }
                    }
#endif

                    uint2 neighborReservoirPos = RTXDI_PixelPosToReservoirPos(idx, params.activeCheckerboardField);

                    RTXDI_DIReservoir prevSample = RTXDI_LoadDIReservoir(reservoirParams,
                        neighborReservoirPos, stparams.sourceBufferIndex);
                    prevSample.M = min(prevSample.M, historyLimit);

                    // Select this sample for the (normalization) numerator if this particular neighbor pixel
                    //     was the one we selected via RIS in the first loop, above.
                    pi = selected == i ? ps : pi;

                    // Add to the sums of weights for the (normalization) denominator
                    piSum += ps * prevSample.M;
                }
            }

            // Use "MIS-like" normalization
            RTXDI_FinalizeResampling(state, pi, piSum);
        }
        else
#endif
        {
            RTXDI_FinalizeResampling(state, 1.0, state.M);
        }
    }

    return state;
}

#endif