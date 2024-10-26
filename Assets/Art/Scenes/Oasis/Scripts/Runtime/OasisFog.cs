using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class OasisFog : FullscreenEffectBase<OasisFogPass>
{
}

public class OasisFogPass : FullscreenPassBase
{
    public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
    {
        var volumeComponent = VolumeManager.instance.stack.GetComponent<OasisFogVolumeComponent>();

        float fogDensity = volumeComponent.Density.value;
        if (fogDensity < Mathf.Epsilon) return;

        float fogStartDistance = volumeComponent.StartDistance.value;
        Color fogTint = volumeComponent.Tint.value;
        float fogSunScatteringIntensity = volumeComponent.SunScatteringIntensity.value;
        Vector2 fogHeightRange = volumeComponent.HeightRange.value;

        material.SetColor("_Tint", fogTint);
        material.SetFloat("_Density", fogDensity);
        material.SetFloat("_StartDistance", fogStartDistance);
        material.SetFloat("_SunScatteringIntensity", fogSunScatteringIntensity);
        material.SetVector("_Height_Range", fogHeightRange);

        base.Execute(context, ref renderingData);
    }
}
