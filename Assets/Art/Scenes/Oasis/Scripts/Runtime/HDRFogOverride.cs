using UnityEngine;
using UnityEngine.Rendering;

[ExecuteAlways]
public class HDRFogOverride : MonoBehaviour
{
    // private var for previous fog color
    private Color _previousFogColor;
    // public var for fog color
    [ColorUsage(false, true)]public Color FogColor = Color.white;
    
    private void OnEnable()
    {
        // subscribe to RenderPipelineManager.beginFrameRendering
        RenderPipelineManager.beginFrameRendering += BeginFrame;
        // subscribe to RenderPipelineManager.endFrameRendering
        RenderPipelineManager.endFrameRendering += EndFrame;
    }
    
    private void OnDisable()
    {
        RenderPipelineManager.beginFrameRendering -= BeginFrame;
        RenderPipelineManager.endFrameRendering -= EndFrame;
    }

    private void BeginFrame(ScriptableRenderContext arg1, Camera[] arg2)
    {
        // get current fog color
        _previousFogColor = RenderSettings.fogColor;
        // set fog color to our override color
        RenderSettings.fogColor = FogColor;
    }
    
    private void EndFrame(ScriptableRenderContext arg1, Camera[] arg2)
    {
        // revert fog color to previous color
        RenderSettings.fogColor = _previousFogColor;
    }
}
