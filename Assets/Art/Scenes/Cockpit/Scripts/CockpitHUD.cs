using UnityEngine;

public class CockpitHUD : MonoBehaviour
{
    
    [Header("Fighter Ship")]
    public Transform fighterShip;
    
    // Update is called once per frame
    void Update()
    {
        fighterShip.forward = Vector3.forward;
    }
}
