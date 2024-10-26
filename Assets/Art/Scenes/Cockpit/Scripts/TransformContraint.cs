using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

[ExecuteAlways]
public class TransformContraint : MonoBehaviour
{
    public Transform pivot;
    [Header("Constraints")]
    public Transform constraintA;
    public Transform constraintB;

    [Range(0f, 1f)]
    public float blendContraints = 0f;
    [Header("Blend Controls")]
    [Range(0f, 1f)]
    public float blendTotal = 0f;
    [Range(0f, 1f)]
    public float blendPosition = 0f;
    [Range(0f, 1f)]
    public float blendRotation = 0f;
    [Range(0f, 1f)]
    public float blendScale = 0f;
    
    private Vector3 positionWS;
    private Quaternion rotation;
    private Vector3 scale = Vector3.one;

    public float dampening = 0.1f;
    private Vector3 posVel;
    private Quaternion rotVel;
    
    private void OnEnable()
    {
        RenderPipelineManager.beginContextRendering += BeginFrame;
        RenderPipelineManager.endContextRendering += EndFrame;
    }

    private void OnDisable()
    {
        RenderPipelineManager.beginContextRendering -= BeginFrame;
        RenderPipelineManager.endContextRendering -= EndFrame;
    }
    
    private void Update()
    {
        // If we are in playmode we want to update the constraint every frame
        //if(Application.isPlaying)
            //UpdateConstraint();
    }

    /// <summary>
    /// When out of playmode we want ot still update the position of the Transform but not commit to it since
    /// it will dirty the scene with changes to the transforms position.
    /// </summary>
    /// <param name="context">Rendering Context, this is unused in this context</param>
    /// <param name="cameras">List of cameras, this is unused in this context</param>
    private void BeginFrame(ScriptableRenderContext context, List<Camera> cameras)
    {
        if (!Application.isPlaying)
        {
            var transform1 = transform;
            positionWS = transform1.position;
            rotation = transform1.rotation;
            scale = transform1.localScale;
        }
        UpdateConstraint();
    }
    
    /// <summary>
    /// At the end of the frame, when not in Playmode we want to revert the transform back to its original state,
    /// doing this will not dirty the scene with changes to the transforms position.
    /// </summary>
    /// <param name="context"></param>
    /// <param name="cameras"></param>
    private void EndFrame(ScriptableRenderContext context, List<Camera> cameras)
    {
        if (!Application.isPlaying)
        {
            pivot.SetPositionAndRotation(positionWS, rotation);
            pivot.localScale = scale;
        }
    }

    /// <summary>
    /// Updates the constraint transform based on the blend values between the two constraints.
    /// </summary>
    private void UpdateConstraint()
    {
        if (!constraintA || !constraintB) return;

        var targetPosition = Vector3.Lerp(constraintA.position, constraintB.position, blendContraints);
        var targetRotation = Quaternion.Lerp(constraintA.rotation, constraintB.rotation, blendContraints);
        var targetScale = Vector3.Lerp(constraintA.lossyScale, constraintB.lossyScale, blendContraints);

        var pos = Vector3.Lerp(transform.position, targetPosition, blendTotal * blendPosition);
        var rot = Quaternion.Lerp(transform.rotation, targetRotation, blendTotal * blendRotation);
        
        pivot.SetPositionAndRotation(
            pos,
            rot
        );
        pivot.localScale = Vector3.Lerp(transform.localScale, targetScale, blendTotal * blendScale);
    }
}
