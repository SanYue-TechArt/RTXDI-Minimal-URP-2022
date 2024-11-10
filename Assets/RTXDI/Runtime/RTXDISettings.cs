using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

[Serializable, VolumeComponentMenuForRenderPipeline("Post-processing/RTXDI", typeof(UniversalRenderPipeline))]
public sealed class RTXDISettings : VolumeComponent, IPostProcessComponent
{
    public BoolParameter enable = new BoolParameter(false);

    public BoolParameter enableResampling = new BoolParameter(true);

    public BoolParameter unbiasedMode = new BoolParameter(false);

    public ClampedIntParameter numInitialSamples = new ClampedIntParameter(8, 0, 16);
    
    public ClampedIntParameter numInitialBrdfSamples = new ClampedIntParameter(1, 0, 16);

    public ClampedFloatParameter brdfCutoff = new ClampedFloatParameter(0.0f, 0.0f, 1.0f);

    public ClampedIntParameter numSpatialSamples = new ClampedIntParameter(1, 0, 16);
    
    public bool IsActive() => enable.value;
    
    public bool IsTileCompatible() => false;
}
