using UnityEngine;

public class GetClosestBoidTest : MonoBehaviour
{
    private Boids.Boid boid;


    private void Update()
    {
        if (Boids.Instance)
        {
            boid = Boids.Instance.GetClosestBoid(transform.position);
        }
    }

    private void OnDrawGizmos()
    {
        Gizmos.DrawLine(transform.position, boid.position);
    }
}
