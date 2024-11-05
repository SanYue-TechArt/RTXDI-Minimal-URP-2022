#ifndef RTXDI_MINIMAL_INITIAL_SAMPLING_FUNCTIONS
#define RTXDI_MINIMAL_INITIAL_SAMPLING_FUNCTIONS

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

#endif