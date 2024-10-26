using UnityEngine;
using UnityEngine.Rendering;

[ExecuteAlways]
public class Wobble : MonoBehaviour
{
    private Vector3 cachedPosition;
    private Vector3 randOffset;
    private Vector3 posOffset;
    private Vector3 offsetVel;
    
    private Vector3 cachedScale;
    private Vector3 randScale;
    private Vector3 scaleOffset;
    private Vector3 scaleVel;

    public bool positionJitter = true;
    public bool scaleJitter = false;
    
    public float scale = 5;
    public float speed = 1;
    
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
        if (!gameObject.activeInHierarchy) return;
        
        cachedPosition = transform.localPosition;
        cachedScale = transform.localScale;

        if (Vector3.Distance(posOffset, randOffset) < 0.1f)
            randOffset = Random.insideUnitSphere * scale;
        
        if (Vector3.Distance(scaleOffset, randScale) < 0.03f)
            randScale = cachedScale * Random.Range(-0.25f, 0.25f);
        
        var t = Time.time * speed;
        
        if (positionJitter)
        {
            posOffset = Vector3.SmoothDamp(posOffset, randOffset, ref offsetVel, Time.deltaTime, speed);
            transform.position += posOffset;
        }

        if (scaleJitter)
        {
            scaleOffset = Vector3.SmoothDamp(scaleOffset, randScale, ref scaleVel, Time.deltaTime, speed * 10f);
            transform.localScale += scaleOffset;
        }
    }
    
    private void EndCamera(ScriptableRenderContext arg1, Camera[] arg2)
    {
        if (!gameObject.activeInHierarchy) return;
        
        transform.localPosition = cachedPosition;
        transform.localScale = cachedScale;
    }
}
