using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Playables;

[RequireComponent(typeof(PlayableDirector))]
public class TimelineLooper : MonoBehaviour
{
    public float LoopPoint;
    
    private PlayableDirector m_PlayableDirector;
    
    void Start()
    {
        m_PlayableDirector = GetComponent<PlayableDirector>();
    }

    
    void Update()
    {
        if (m_PlayableDirector.state == PlayState.Paused)
        {
            m_PlayableDirector.time = LoopPoint;
            m_PlayableDirector.Play();
        }
    }
}
