using UnityEngine;
using UnityEngine.Rendering;

[ExecuteAlways]
public class WormLazerBeam : MonoBehaviour
{
    public bool hitting = false;

    private bool cachedActive;
    
    public GameObject beamHitObj;
    public GameObject beamMissObj;

    private Vector3 cachedBeamScale;
    private Vector3 beamScale;
    public Transform beamPivot;

    private void OnEnable()
    {
        RenderPipelineManager.beginFrameRendering += OnBeginCamera;
        RenderPipelineManager.endFrameRendering += EndCamera;
    }
    
    private void OnDisable()
    {
        RenderPipelineManager.beginFrameRendering -= OnBeginCamera;
        RenderPipelineManager.endFrameRendering -= EndCamera;
    }
    
    private void OnBeginCamera(ScriptableRenderContext arg1, Camera[] arg2)
    {
        if(beamPivot == null || beamHitObj == null || beamMissObj == null) return;
        
        cachedBeamScale = beamPivot.localScale;
        cachedActive = beamHitObj.activeSelf;

        beamScale = cachedBeamScale;
        beamScale.y = -beamHitObj.transform.localPosition.z;
        beamPivot.localScale = beamScale;
        beamHitObj.SetActive(hitting);
        beamMissObj.SetActive(!hitting);
    }
    
    private void EndCamera(ScriptableRenderContext arg1, Camera[] arg2)
    {
        if(beamPivot == null || beamHitObj == null || beamMissObj == null) return;
        
        beamPivot.localScale = cachedBeamScale;
        beamHitObj.SetActive(cachedActive);
        beamMissObj.SetActive(!cachedActive);
    }
}
