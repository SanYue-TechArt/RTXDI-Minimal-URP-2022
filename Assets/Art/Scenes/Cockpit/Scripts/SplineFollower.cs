using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Splines;

[ExecuteAlways]
public class SplineFollower : MonoBehaviour
{
    public SplineContainer splineContainer;
    public Transform handAnimatedTransform;

    [Range(0,1)]
    public float animationLerp;

    [Range(0, 1)] public float BankContribution;
    
    public float Distance;
    public float Prediction;

    public float SmoothTime;

    public float BankSampleDistance;

    //Is public so I can see it (readonly)
    public float splineLength;
    
    //Smoothdamp variables
    private Quaternion rotVel;
    private float rollVel;
    private float currentRotationSpeed;
    
    // Start is called before the first frame update
    void Start()
    {
        splineLength = splineContainer.CalculateLength();
    }

    // Update is called once per frame
    void Update()
    {
        //Distance = Time.time * Speed;
        
        Vector3 prediction = splineContainer.EvaluatePosition(GetSplineTime(Distance + Prediction));
        Vector3 trailingSample = splineContainer.EvaluatePosition(GetSplineTime(Distance + Prediction - BankSampleDistance));
        Vector3 leadingSample = splineContainer.EvaluatePosition(GetSplineTime(Distance+ Prediction + BankSampleDistance));
        
        Vector3 angle1 = (trailingSample - prediction).normalized;
        Vector3 angle2 = (leadingSample - prediction).normalized;
        
        Debug.DrawRay(prediction, angle1 * BankSampleDistance, Color.cyan);
        Debug.DrawRay(prediction, angle2 * BankSampleDistance, Color.magenta);

        float BankAmount = Vector3.Dot(angle1, angle2);
        
        BankAmount = Mathf.Clamp01((BankAmount + 1) * 4);// 0: straight, 1: More that 90 degrees
        
        float t = GetSplineTime(Distance);
        
        Vector3 sample = splineContainer.EvaluatePosition(t);
        Vector3 forwardVector = (prediction - sample).normalized; //Vector3.Normalize(splineContainer.EvaluateTangent(t));
        
        Vector3 bankVector = Vector3.Cross(angle1, angle2).normalized;
        bankVector = Vector3.Cross(forwardVector, bankVector);
        
        Debug.DrawRay(prediction, bankVector * 5, Color.blue);

        
        
        Quaternion splineTargetRotation;
        splineTargetRotation = Quaternion.Lerp(Quaternion.LookRotation(forwardVector, Vector3.up), Quaternion.LookRotation(forwardVector, bankVector), BankContribution);

        splineTargetRotation = SmoothDamp(transform.rotation, splineTargetRotation,ref rotVel, SmoothTime);

        Vector3 splineTargetPosition = sample;

        transform.rotation = Quaternion.Lerp(splineTargetRotation, handAnimatedTransform.rotation, animationLerp);


        transform.position = Vector3.Lerp(splineTargetPosition, handAnimatedTransform.position, animationLerp);
    }
    
    /*
    Copyright 2016 Max Kaufmann (max.kaufmann@gmail.com)
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    */
    public static Quaternion SmoothDamp(Quaternion rot, Quaternion target, ref Quaternion deriv, float time) {
        if (Time.deltaTime < Mathf.Epsilon) return rot;
        // account for double-cover
        var Dot = Quaternion.Dot(rot, target);
        var Multi = Dot > 0f ? 1f : -1f;
        target.x *= Multi;
        target.y *= Multi;
        target.z *= Multi;
        target.w *= Multi;
        // smooth damp (nlerp approx)
        var Result = new Vector4(
            Mathf.SmoothDamp(rot.x, target.x, ref deriv.x, time),
            Mathf.SmoothDamp(rot.y, target.y, ref deriv.y, time),
            Mathf.SmoothDamp(rot.z, target.z, ref deriv.z, time),
            Mathf.SmoothDamp(rot.w, target.w, ref deriv.w, time)
        ).normalized;
		
        // ensure deriv is tangent
        var derivError = Vector4.Project(new Vector4(deriv.x, deriv.y, deriv.z, deriv.w), Result);
        deriv.x -= derivError.x;
        deriv.y -= derivError.y;
        deriv.z -= derivError.z;
        deriv.w -= derivError.w;		
		
        return new Quaternion(Result.x, Result.y, Result.z, Result.w);
    }
    
    

    private float GetSplineTime(float distance)
    {
        return Mathf.Repeat(distance / splineLength, 1);
        //return Mathf.Clamp01(distance / splineLength);
    }
}
