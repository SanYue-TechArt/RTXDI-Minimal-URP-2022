using System;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Rendering;
using UnityEngine.Splines;

[ExecuteAlways]
public class WormSplineFollower : MonoBehaviour
{
    public SplineContainer Spline;
    public bool flipDirection;
    public float Distance;
    public float PositionOffset;
    public float ScaleOffset;
    
    private float m_SplineLength;

    // Start is called before the first frame update
    private void OnEnable()
    {
        m_SplineLength = Spline.CalculateLength();
        RenderPipelineManager.beginFrameRendering += BeginFrame;
        RenderPipelineManager.endFrameRendering += EndFrame;
    }

    private void OnDisable()
    {
        RenderPipelineManager.beginFrameRendering -= BeginFrame;
        RenderPipelineManager.endFrameRendering -= EndFrame;
    }
    
    private void BeginFrame(ScriptableRenderContext arg1, Camera[] arg2)
    {
        if (Spline == null) return;
        
        foreach (Transform child in transform)
        {
            var scale = 1 - ScaleOffset * child.GetSiblingIndex();
            child.localScale = Vector3.one * scale;

            var offset = flipDirection
                ? Distance + PositionOffset * child.GetSiblingIndex()
                : Distance - PositionOffset * child.GetSiblingIndex();
            offset = Mathf.Clamp(offset, 0, m_SplineLength - float.MinValue);
            var (pos, rot) = GotPositionAndRotation(offset);
            child.SetPositionAndRotation(pos, rot);
        }
    }
    
    private void EndFrame(ScriptableRenderContext arg1, Camera[] arg2)
    {
        // We dont revert the transform in playmode since this is not serialized, and we would like the playmode logic to see the moved transform.
        if (Application.isPlaying) return;
        
        foreach (Transform child in transform)
        {
            child.SetPositionAndRotation(Vector3.zero, quaternion.identity);
            child.localScale = Vector3.one;
        }
    }

    private (Vector3, Quaternion) GotPositionAndRotation(float distance)
    {
        Profiler.BeginSample("GotPositionAndRotation");
        float splineTime = GetSplineTime(distance);
        
        Profiler.BeginSample("EvaluateSpline");
        Spline.Evaluate(splineTime, out var position, out var forward, out var up);
        Profiler.EndSample();

        Quaternion rotation = Quaternion.LookRotation(flipDirection ? -forward : forward, up);
        
        Profiler.EndSample();
        return (position, rotation);
    }

    private float GetSplineTime(float distance)
    {
        return SplineUtility.GetNormalizedInterpolation(Spline.Spline, distance, PathIndexUnit.Distance);
    }
}
