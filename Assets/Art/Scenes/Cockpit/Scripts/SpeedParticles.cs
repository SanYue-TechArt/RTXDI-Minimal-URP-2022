using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteAlways]
public class SpeedParticles : MonoBehaviour
{
    public Transform Ship;
    
    private ParticleSystem m_SpeedParticles;

    private Vector3 m_LastShipPos;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (m_SpeedParticles == null)
        {
            m_SpeedParticles = GetComponent<ParticleSystem>();
        }

        float speed = (Ship.position - m_LastShipPos).magnitude/Time.deltaTime;

        var main = m_SpeedParticles.main;
        main.startSpeed = speed * 2;

        m_LastShipPos = Ship.position;
    }
    
    
}
