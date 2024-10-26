using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Splines;

[ExecuteAlways]
public class NpcSplineFollower : MonoBehaviour
{
    public Transform HandAnimatedTransform;
    [Range(0,1)]
    public float HandAnimationWeight;
    public SplineContainer Spline;
    public float Distance;


    public float m_SplineLength;
    private Quaternion RotVel;
    
    // Start is called before the first frame update
    void Start()
    {
        m_SplineLength = Spline.CalculateLength();
    }

    // Update is called once per frame
    void Update()
    {
        float splineTime = GetSplineTime(Distance);
        Vector3 position = Spline.EvaluatePosition(splineTime);
        Vector3 forward = Vector3.Normalize(Spline.EvaluateTangent(splineTime));
        Vector3 up = Spline.EvaluateUpVector(GetSplineTime(Distance));
        
        Debug.DrawRay(transform.position, up * 5, Color.magenta);

        Quaternion rotation = SplineFollower.SmoothDamp(transform.rotation, Quaternion.LookRotation(forward, up), ref RotVel, 0.3f);


        if (HandAnimatedTransform != null)
        {
            transform.position = Vector3.Lerp(position, HandAnimatedTransform.position, HandAnimationWeight);
            transform.rotation = Quaternion.Lerp(rotation, HandAnimatedTransform.rotation, HandAnimationWeight);
        }
        else
        {
            transform.position = position;
            transform.rotation = rotation;
        }

    }

    public float GetSplineTime(float distance)
    {
        return Mathf.Repeat(distance / m_SplineLength, 1);
    }
}
