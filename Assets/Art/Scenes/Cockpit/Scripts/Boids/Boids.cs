using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEditor;

public class Boids : MonoBehaviour
{
    private static Boids _instance;
    
    public static Boids Instance
    {
        get
        {
            if (_instance == null)
            {
                _instance = FindObjectOfType<Boids>();
            }
            return _instance;
        }
        //set => _instance = value;
    }

    [Header("System Settings")]
    public int boidsPerTeam = 256;
    int maxBoids;
    public ParticleSystem system;
    private ParticleSystemRenderer systemRenderer;
    public ParticleSystem explosionSystem;
    public ParticleSystem.EmitParams explosionSystem_ep;
    public ParticleSystem laserSystem;
    public ParticleSystem.EmitParams laserSystem_ep;

    public float spawnRate = 2.0f;

    public Team[] teams;
    
    [Header("Boid Settings")]
    public float boidSpeed = 10f;
    public float boidSpeedVariation = 2f;
    public float boidTurnSpeed = 0.5f;
    public float boidTurnSpeedVariation = 0.1f;

    public Transform randomBoid;
    private Boid nullBoid;
    private int boidIndex = -1;
    private int boidSpawnIndex;
    
    private NativeArray<Boid> _boids;
    [NonSerialized] public NativeArray<Boid> _boidsAlt;
    private NativeArray<ParticleSystem.Particle> _particles;
    private JobHandle _boidMovementHandle;
    private JobHandle _boidCopyHandle;

    private float[] DamageList;

    // TODO replace with basic state machine
    private bool startFrame = true;
    private bool resetNextFrame = false;

    private float spawnTicker = 1.0f;
    

    private void OnEnable()
    {
        nullBoid = new Boid
        {
            active = false,
            id = -1
        };

        maxBoids = boidsPerTeam * teams.Length;
        
        system.AllocateMeshIndexAttribute();
        var main = system.main;
        main.maxParticles = maxBoids;
        
        _particles = new NativeArray<ParticleSystem.Particle>(maxBoids, Allocator.Persistent);
        
        system.GetParticles(_particles);
        _boids = new NativeArray<Boid>(maxBoids, Allocator.Persistent);
        DamageList = new float[_boids.Length];

        systemRenderer = system.GetComponent<ParticleSystemRenderer>();

        var meshes = new Mesh[teams.Length];
        for (int t = 0; t < teams.Length; t++)
        {
            meshes[t] = teams[t].ship.mesh;
            
            for (int i = 0; i < boidsPerTeam; i++)
            {
                var index = t * boidsPerTeam + i;
                
                // setup boids
                var boid = _boids[index];
                boid.Init(index % teams.Length, index);
                
                boid.ResetBoid();
                _boids[index] = boid;
                
                // setup particles
                var p = _particles[index];
                p.SetMeshIndex((int)boid.team);
                p.startLifetime = 6f;
                p.startColor = teams[boid.team].laserColor;
                _particles[index] = p;
            }
        }
        systemRenderer.SetMeshes(meshes);

        system.SetParticles(_particles);

        _boidsAlt = new NativeArray<Boid>(maxBoids, Allocator.Persistent);
    }
    
    private void OnDisable()
    {
        _boidCopyHandle.Complete();
        
        if (_boids.IsCreated)
            _boids.Dispose();
        if (_particles.IsCreated)
            _particles.Dispose();
        if (_boidsAlt.IsCreated)
            _boidsAlt.Dispose();
    }

    private void Update()
    {
        SpawnTicker();

        if (startFrame)
            return;

        _boidCopyHandle.Complete();

        if (resetNextFrame)
        {
            for (var index = 0; index < _boids.Length; index++)
            {
                var boid = _boids[index];
                boid.ResetBoid();
                _boids[index] = boid;
            }
            boidSpawnIndex = 0;
            // reset damage list
            for (int i = 0; i < DamageList.Length; i++)
            {
                DamageList[i] = 0;
            }
            resetNextFrame = false;
        }
        
        SpawnNewShips();
        SpawnExplosions();
        SpawnLasers();

        if (randomBoid)
        {
            if (boidIndex >= 0 && _boids[boidIndex].active && _boids[boidIndex].team == 0)
            {
                randomBoid.forward = _boids[boidIndex].velocity;
                randomBoid.position = _boids[boidIndex].position;// Vector3.Lerp(randomBoid.position, _boids[boidIndex].position, Time.deltaTime);
            }
            else
            {
                boidIndex = UnityEngine.Random.Range(0, _boids.Length);
            }
        }

        system.SetParticles(_particles);
        _boidsAlt.CopyFrom(_boids);
        // read frames
    }
    
    private void LateUpdate()
    {
        // kick off jobs

        var movement = new BoidMovementJob()
        {
            _boids = _boids,
            _boidsAlt = _boidsAlt,
            deltaTime = Time.deltaTime,
            playerBoid = boidIndex,
        };

        var handle = new JobHandle();
        _boidMovementHandle = movement.Schedule(_boids.Length, handle);
        
        var copy = new BoidCopyJob()
        {
            _particles = _particles,
            _boids = _boids,
            playerBoid = boidIndex,
        };

        _boidCopyHandle = copy.Schedule(_boids.Length, 32, _boidMovementHandle);
        
        JobHandle.ScheduleBatchedJobs();

        if (startFrame)
            startFrame = false;
    }
    
    public Boid GetBoid(int id)
    {
        foreach (var boid in _boidsAlt)
        {
            if (boid.id == id)
            {
                return boid;
            }
        }

        return nullBoid;
    }

    public Boid GetClosestBoid(Vector3 positionWS)
    {
        return GetClosestBoid(positionWS, -1);
    }
    
    public Boid GetClosestBoid(Vector3 positionWS, int team)
    {
        var i = -1;
        var dist = float.PositiveInfinity;
        for (var index = 0; index < _boidsAlt.Length; index++)
        {
            var boid = _boidsAlt[index];
            if(!boid.active || boid.health <= 0 || team == boid.team) continue;
            var boidDist = Vector3.Distance(boid.position, positionWS);
            if (!(boidDist < dist)) continue;
            dist = boidDist;
            i = index;
        }

        if (i >= 0 && i < _boidsAlt.Length)
        {
            return _boidsAlt[i];
        }
        else
        {
            return nullBoid;
        }
    }

    public void ResetBoids()
    {
        resetNextFrame = true;
    }

    public void DamageBoid(int id, float damage)
    {
        if(DamageList.Length > id && id >= 0)
        {
            DamageList[id] += damage;
        }
    }

    [BurstCompile]
    struct BoidMovementJob : IJobFor
    {
        public NativeArray<Boid> _boids;
        [ReadOnly]
        public NativeArray<Boid> _boidsAlt;

        [ReadOnly] public float deltaTime;

        [ReadOnly] public int playerBoid;

        public void Execute(int index)
        {
            var b = _boids[index];

            if (!b.active) return;

            float turningSpeed = b.turningSpeed;
            if (index == playerBoid)
                turningSpeed *= 1.5f;
            
            // target
            float3 targetDir;
            float targetFacing = 0f;

            b.rateOfFire -= deltaTime;
            
            if(b.targetID == -1 || _boidsAlt[b.targetID].active == false)
            {
                b.targetID = -1;
                b.targetAge = 0;
                targetDir = math.normalize(b.destination - b.position);
            }
            else
            {
                targetDir = math.normalize(_boidsAlt[b.targetID].position - b.position);
                targetFacing = math.dot(targetDir, math.normalize(b.velocity));
                targetDir *= 2f;
            }
            
            // general boids
            float3 cohesionVect = 0;
            float3 separationVect = 0;
            float3 alignmentVect = 0;

            // counts
            int separationCount = 0;
            int cohesionCount = 0;

            for(int i = 0; i < _boidsAlt.Length; i++)
            {
                if(b.id != _boidsAlt[i].id && _boidsAlt[i].active)
                {
                    float3 distVect = _boidsAlt[i].position - b.position;
                    float dist = math.length(distVect);

                    if (b.team == _boidsAlt[i].team) // on team
                    {
                        //cohesion
                        if (dist < 100f)
                        {
                            cohesionVect += distVect * 0.1f;
                            cohesionCount++;
                            //alignment
                            alignmentVect += _boidsAlt[i].velocity;
                        }
                    }
                    else
                    {// if not on team
                        if(dist < 350f)
                        {
                            if((b.targetID == -1 || b.targetAge >= 4200f) && i != playerBoid)
                            {
                                b.targetID = i;
                                b.targetAge = 0;
                            }
                            else
                            {
                                if (i == b.targetID)
                                {
                                    var targetBoid = _boids[i];
                                    if (!targetBoid.active)
                                    {
                                        b.targetID = -1;
                                        b.targetAge = 0;
                                        continue;
                                    }
                                    
                                    turningSpeed *= 2;
                                    if (targetFacing > 0.85f && b.rateOfFire <= 0)
                                    {
                                        targetBoid.health -= b.damage;
                                        b.shooting = true;
                                        b.rateOfFire = 0.8f;
                                    }
                                    if (targetBoid.health <= 0f)
                                    {
                                        targetBoid.active = false;
                                    }

                                    _boids[i] = targetBoid;
                                }
                            }
                        }
                    }
                    //separation
                    if(dist <= 50f && dist > 0){
                        separationVect -= distVect / (dist * 0.1f);
                        separationCount++;
                    }
                }
            }

            if (cohesionCount > 0)
            {
                cohesionVect /= cohesionCount;
                alignmentVect /= cohesionCount;
            }

            if(separationCount > 0)
                separationVect /= separationCount;
            

            var vec = float3.zero;
            vec += cohesionVect * 0.01f;
            vec += separationVect * 0.5f;
            vec += alignmentVect * 0.1f;
            vec += targetDir;
            vec += b.velocity;

            var vel = math.normalizesafe(vec + math.EPSILON);
            
            vel = RotateTowards(b.velocity, vel, math.radians(100f * deltaTime), 0f); // clamp the velocity
            
            b.velocity = math.lerp(b.velocity, math.lerp(b.smoothVel, vel, 0.5f), 0.5f); // smooth the velocity
            
            b.position += b.velocity * b.speed * deltaTime; // scale to speed

            b.smoothVel = math.lerp(b.smoothVel, vel, 0.1f);
            
            b.targetAge++;
            _boids[index] = b;
        }

        float3 RotateTowards(float3 start, float3 end, float maxAngle, float maxMagnitude)
        {
            var startMag = math.length(start);
            var endMag = math.length(end);

            var dot = math.dot(start, end);
            
            if (dot > 1f - math.EPSILON) // direction almost the same
            {
                return end;
            }
            else // normal case
            {
                var angle = math.acos(dot);
                var axis = math.normalize(math.cross(start, end));
                var matrix = float3x3.AxisAngle(axis, math.min(angle, maxAngle));
                float3 rotated = math.mul(matrix, start);
                rotated *= ClampedMove(startMag, endMag, maxMagnitude);
                return math.normalize(rotated);
            }
        }
        
        static float ClampedMove(float start, float end, float clampedDelta)
        {
            var delta = end - start;
            if (delta > 0.0F)
                return start + math.min(delta, clampedDelta);
            else
                return start - math.min(-delta, clampedDelta);
        }
        
        static float3 Slerp(float3 start, float3 end, float percent)
        {
            float dotP = math.dot(start, end);
            dotP = math.clamp(dotP, -1.0f, 1.0f);
            float theta = math.acos(dotP)*percent;
            float3 RelativeVec = math.normalizesafe(end - start*dotP);
            return ((start*math.cos(theta)) + (RelativeVec*math.sin(theta)));
        }
    }
    
    [BurstCompile]
    struct BoidCopyJob : IJobParallelFor
    {
        public NativeArray<ParticleSystem.Particle> _particles;
        [ReadOnly]
        public NativeArray<Boid> _boids;

        [ReadOnly] public int playerBoid;

        public void Execute(int index)
        {
            var p = _particles[index];
            var b = _boids[index];

            if (b.active)
            {
                //setup alive boid
                p.velocity =  math.normalizesafe(b.position - (float3)p.position);
                p.position = b.position;
                p.remainingLifetime = math.clamp(p.remainingLifetime + 1f, 0.1f, 10f);
                p.rotation = 0f;
                p.startSize3D = index != playerBoid ? Vector3.one : Vector3.zero;
                p.startSize = math.clamp(p.startSize + 0.1f, 0.0f, 1f);
            }
            else
            {
                //null boid
                p.position = Vector3.zero;
                p.remainingLifetime = -10f;
                p.rotation += 0.1f;
                p.startSize3D = Vector3.zero;
                p.startSize = 0f;
            }
            
            _particles[index] = p;
        }
    }
    
    private void SpawnLasers()
    {
        // do the lasers
        for (int i = 0; i < _boids.Length; i++)
        {
            var b = _boids[i];

            if (b.shooting && b.targetID != -1 && _boids[b.targetID].active)
            {
                laserSystem_ep.position = b.position;

                var vector = _boids[b.targetID].position - b.position;
                var distance = math.length(vector);
                var dir = math.normalize(vector);
                laserSystem_ep.velocity = dir * 1200f;
                laserSystem_ep.startLifetime = distance / 1200f;
                laserSystem_ep.startColor = teams[b.team].laserColor;
                laserSystem.Emit(laserSystem_ep, 1);
                b.shooting = false;
            }

            _boids[i] = b;
        }
    }

    private void SpawnExplosions()
    {
        // spawn explosions at each death
        for (var i = 0; i < _boids.Length; i++)
        {
            var boid = _boids[i];

            boid.health -= DamageList[boid.id];
            DamageList[boid.id] = 0f;
            
            if (boid.health <= 0f)
            {
                explosionSystem_ep.position = boid.position;
                explosionSystem_ep.velocity = boid.velocity * 0.1f;
                explosionSystem.Emit(explosionSystem_ep, 1);
                boid.health = 100f;
                boid.position = teams[boid.team].spawnPoints[UnityEngine.Random.Range(0, teams[boid.team].spawnPoints.Length)].position;
            }
            _boids[i] = boid;
        }
    }
    
    private void SpawnNewShips()
    {
        if (spawnTicker > 0f) return;

        while (_boids[boidSpawnIndex].active)
        {
            boidSpawnIndex = (int)Mathf.Repeat(boidSpawnIndex + 1, _boids.Length);
        }
        
        var boid = _boids[boidSpawnIndex];

        var point = teams[boid.team].spawnPoints[UnityEngine.Random.Range(0, teams[boid.team].spawnPoints.Length)];

        if (point.gameObject.activeInHierarchy)
        {
            boid.Spawn(point.position, point.forward);
            _boids[boidSpawnIndex] = boid;
            boidSpawnIndex = 0;
        }

        ResetTicker();
    }

    private void SpawnTicker()
    {
        if (spawnTicker > 0f)
            spawnTicker -= Time.deltaTime;
        else
        {
            ResetTicker();
        }
    }

    private void ResetTicker()
    {
        spawnTicker = Mathf.Max(0.05f, spawnRate);
    }

    public struct Boid
    {
        public float3 position;
        public float3 velocity;
        public float3 smoothVel;
        public float3 destination;

        // states
        public bool active;
        public bool shooting;
        
        public float health;
        public float damage;
        public float rateOfFire;
        public float speed;
        public float turningSpeed;

        public int targetID;
        public int targetAge;

        public int team;
        public int id;
        
        public void Init(int team, int id)
        {
            position = float3.zero;
            velocity = float3.zero;
            smoothVel = float3.zero;
            destination = float3.zero;
            active = false;
            shooting = false;
            health = 100f;
            damage = 22f;
            rateOfFire = 0.5f;
            speed = 10f;
            turningSpeed = 0.5f;
            targetID = -1;
            targetAge = 1;
            this.team = team;
            this.id = id;
        }

        public void ResetBoid()
        {
            Init(team, id);
        }
        
        public void Spawn(Vector3 position, Vector3 direction)
        {
            this.position = position + UnityEngine.Random.insideUnitSphere;
            var dir = direction;
            velocity = dir * speed;
            active = true;
        }
    }

    [Serializable]
    public class Team
    {
        public Color shipColor;
        public Color laserColor;
        public Transform[] spawnPoints;
        public Ship ship;
    }

    [Serializable]
    public class Ship
    {
        public Mesh mesh;
    }

    #if UNITY_EDITOR

    private GUIStyle style;
    
    private void OnDrawGizmos()
    {
        foreach (var boid in _boidsAlt)
        {
            Gizmos.color = teams[boid.team].shipColor * (boid.active ? 1f : 0.1f);
            Gizmos.DrawWireSphere(boid.position, boid.id == boidIndex ? 20f : 10f);
            if (boid.active)
            {
                Gizmos.color = Color.gray * 0.25f;
                Gizmos.DrawLine(boid.position, boid.destination);
                if (boid.targetID >= 0)
                {
                    var target = _boidsAlt[boid.targetID];
                    Gizmos.color = Color.red * 0.25f;
                    Gizmos.DrawLine(boid.position, target.position);
                    Gizmos.color = Color.red;
                    Gizmos.DrawLine(boid.position,
                        (Vector3) boid.position + Vector3.Normalize(target.position - boid.position) * 25f);
                }

                var text = $"boid:{boid.id}\n" +
                           $"HP:{boid.health}";
                style ??= new GUIStyle(); 
                style.normal.textColor = teams[boid.team].shipColor;
                Handles.Label(boid.position, text, style);
            }
        }
    }
    #endif
}
