using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class CockpitOutline : FullscreenEffectBase<CockpitOutlinePass>
{
}

public class CockpitOutlinePass : FullscreenPassBase
{
    public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
    {
        var volumeComponent = VolumeManager.instance.stack.GetComponent<OutlineVolumeComponent>();

        bool enabled = volumeComponent.Enabled.value;

        if (!enabled) return;
        base.Execute(context, ref renderingData);
    }
}