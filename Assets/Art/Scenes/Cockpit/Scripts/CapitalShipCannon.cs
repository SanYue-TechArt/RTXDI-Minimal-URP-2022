using UnityEngine;
using Random = UnityEngine.Random;

public class CapitalShipCannon : MonoBehaviour
{
    [Header("Transform Refs")]
    public Transform CannonYaw;
    public Transform CannonPitch;
    public Transform[] Barrels;

    [Header("Shooting")] 
    public int teamIndex;
    public Transform ManualAim;
    public bool autoFire = true;
    public float fireRate = 100;
    public float range = 350;

    // Shooting
    private ParticleSpawnSystem partSpawner;
    private bool firing;
    private int barrelIndex;
    
    // auto aiming
    public Vector3 targetPosition = Vector3.zero;
    public Vector3 aimingPosition;
    private Boids.Boid targetBoid;
    private bool newBoid = true;
    private float cooldown;
    // Yaw
    private Vector3 yawValue;
    private Vector3 manualYawValue;
    //private Vector3 yawVel;
    // Pitch
    private Quaternion pitchValue;
    private Quaternion manualPitchValue;
    private float pitchSmooth;
    private float pitchVel;
    
    
    [Range(0, 1)]
    public float manualBlend = 1f;


    private void Start()
    {
        if (partSpawner == null)
        {
            var systems = FindObjectsByType<ParticleSpawnSystem>(FindObjectsSortMode.None);
            foreach (var system in systems)
            {
                if (system.CompareTag(gameObject.tag))
                {
                    partSpawner = system;
                }
            }
        }
        manualYawValue = CannonYaw.forward;
        manualPitchValue = CannonPitch.localRotation;
    }

    // Update is called once per frame
    void Update()
    {
        if (!CannonYaw || !CannonPitch) return;
        
        // do auto aim
        if (Boids.Instance && manualBlend < 1)
        {
            if (newBoid || targetBoid.id < 0 || targetBoid.health <= 0f || Vector3.Distance(targetBoid.position, CannonYaw.position) > (350f + 200f))
            {
                targetBoid = Boids.Instance.GetClosestBoid(ManualAim.position, teamIndex);
                newBoid = false;
            }
            else
            {
                targetBoid = Boids.Instance.GetBoid(targetBoid.id);
            }
        }
        
        if(targetBoid.id == -1)
            return;

        targetPosition = targetBoid.position + targetBoid.velocity * targetBoid.speed;
        
        // manual aim
        aimingPosition = Vector3.Lerp(targetPosition, ManualAim.position, manualBlend);
        
        // Yaw
        var yawDest = transform.InverseTransformPoint(aimingPosition);
        yawDest.y = 0f;
        var yawDestWorld = transform.TransformDirection(yawDest.normalized);//.normalized;


        var debugfwd = yawDestWorld * 10;
        Debug.DrawLine(CannonYaw.position,  CannonYaw.position + debugfwd, Color.blue);

        manualYawValue = Vector3.RotateTowards(manualYawValue, yawDestWorld, 70f * Mathf.Deg2Rad * Time.deltaTime, Time.deltaTime * 0.1f);

        // Pitch
        var localPitch = CannonYaw.InverseTransformPoint(aimingPosition);
        localPitch.x = 0;
        localPitch.y -= CannonPitch.localPosition.y;
        var pitch = Vector3.Angle(Vector3.forward, localPitch.normalized);
        pitch = localPitch.y > 0f ? -pitch : pitch;
        pitch = Mathf.Clamp(pitch, -60f, 5f);

        pitchSmooth = Mathf.SmoothDamp(pitchSmooth, pitch, ref pitchVel, 0.2f, 25f);
        
        manualPitchValue = Quaternion.AngleAxis(pitchSmooth, Vector3.right);

        // blend
        
        CannonYaw.rotation = Quaternion.LookRotation(manualYawValue, transform.up);
        CannonPitch.localRotation = manualPitchValue;

        if (autoFire == false) return;
        
        // Shooting
        if (Vector3.Distance(CannonPitch.position, aimingPosition) < range) // in range
        {
            firing = Vector3.Dot(CannonPitch.forward, (aimingPosition - CannonPitch.position).normalized) > 0.95f; // aiming at target
        }
        else
        {
            firing = false;
        }

        if (!firing) return;
        
        if (cooldown < 0f)
        {
            // shoot
            var barrel = Barrels[barrelIndex];
            partSpawner.Spawn(barrel.position, barrel.forward, aimingPosition);
            if (Boids.Instance && manualBlend <= float.Epsilon)
            {
                Boids.Instance.DamageBoid(targetBoid.id, 10f);
            }

            barrelIndex = (int)Mathf.Repeat(barrelIndex + 1, Barrels.Length);
            cooldown += (Random.Range(0.9f, 1.1f) / fireRate) * 60f;
        }
        else
        {
            cooldown -= Time.deltaTime;
        }
    }

    private void OnDrawGizmos()
    {
        // Manual
        var c = Color.blue;
        c.a = 0.1f;
        Gizmos.color = c;
        var manualAimPosition = ManualAim.position;
        Gizmos.DrawSphere(manualAimPosition, 2f);
        Gizmos.DrawLine(manualAimPosition, CannonPitch.position);
        // Auto
        c = Color.green;
        c.a = 0.1f;
        Gizmos.color = c;
        Gizmos.DrawSphere(targetPosition, 2f);
        Gizmos.DrawLine(targetPosition, CannonPitch.position);
        // Actual
        c = Color.red;
        c.a = 0.25f;
        Gizmos.color = c;
        //Gizmos.DrawSphere(aimingPosition, 2f);
        Gizmos.DrawRay(CannonPitch.position, CannonPitch.forward * 250);
    }
}
