using UnityEngine;
using UnityEngine.Rendering;

[ExecuteAlways]
public class SimpleUpvector : MonoBehaviour
{
    private Transform parent;
    public Transform upVectorGoal;

    private Quaternion cachedRotation;

    // Start is called before the first frame update
    private void OnEnable()
    {
        parent = transform.parent;
        
        if (parent == null || upVectorGoal == null) return;
        
        RenderPipelineManager.beginFrameRendering += BeginFrame;
        RenderPipelineManager.endFrameRendering += EndFrame;
    }

    private void OnDisable()
    {
        RenderPipelineManager.beginFrameRendering -= BeginFrame;
        RenderPipelineManager.endFrameRendering -= EndFrame;
    }
    
    private void BeginFrame(ScriptableRenderContext context, Camera[] cams)
    {
        cachedRotation = transform.rotation;
        
        transform.rotation = Quaternion.LookRotation(parent.forward, upVectorGoal.position - transform.position);
    }
    
    private void EndFrame(ScriptableRenderContext context, Camera[] cams)
    {
        if (Application.isPlaying) return;
        
        transform.rotation = cachedRotation;
    }
}
