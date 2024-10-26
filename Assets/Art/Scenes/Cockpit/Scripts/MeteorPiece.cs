using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MeteorPiece : MonoBehaviour
{ 
    public Transform explosionOrigin;

    [Range(0,1)]
    public float T;
    
    private Vector3 m_RelativePosition;
    // Start is called before the first frame update
    void Start()
    {
        m_RelativePosition = transform.position - explosionOrigin.position;
    }

    // Update is called once per frame
    void Update()
    {
        transform.position = explosionOrigin.position + m_RelativePosition +
                             m_RelativePosition * (1 / m_RelativePosition.magnitude);
    }
}
