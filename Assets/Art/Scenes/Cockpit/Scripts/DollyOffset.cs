using Cinemachine;
using UnityEngine;

public class DollyOffset : MonoBehaviour
{
    public CinemachineDollyCart cart;
    public CinemachineDollyCart targetCart;
    public float offset;

    // Update is called once per frame
    void LateUpdate()
    {
        if (cart == null || targetCart == null) return;

        cart.m_Position = targetCart.m_Position + offset;
    }
}
