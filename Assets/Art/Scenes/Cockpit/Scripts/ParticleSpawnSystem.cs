using UnityEngine;

public class ParticleSpawnSystem : MonoBehaviour
{
    public ParticleSystem PS;
    public float speed = 10f;
    
    public void Spawn(Vector3 positionWS, Vector3 directionWS, Vector3 endPoint)
    {
        var param = new ParticleSystem.EmitParams();
        param.position = positionWS;
        param.rotation3D = Quaternion.LookRotation(directionWS).eulerAngles;
        var endDist = (endPoint - positionWS).magnitude;
        param.velocity = directionWS * speed;
        param.startLifetime = endDist / speed;
        PS.Emit(param, 1);
    }
}
