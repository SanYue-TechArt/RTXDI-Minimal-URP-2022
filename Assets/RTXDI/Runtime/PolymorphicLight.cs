using System.Collections.Generic;
using System.Runtime.InteropServices;
using JetBrains.Annotations;
using UnityEngine;
using UnityEngine.Rendering;

[ExecuteAlways]
public class PolymorphicLight : MonoBehaviour
{
    public bool IsValid() => _mesh_filter != null && _mesh_filter.sharedMesh != null && _mesh_renderer != null && _material != null;

    [CanBeNull]
    public Mesh GetMesh() => IsValid() ? _mesh_filter.sharedMesh : null;

    public Vector3 GetLightColor() => new Vector3(_light_color.r, _light_color.g, _light_color.b);

    [CanBeNull]
    public Texture2D GetEmissiveTexture() => IsValid() ? _material.GetTexture(MaterialParam._BaseMap) as Texture2D : null;

    [CanBeNull]
    public MeshRenderer GetMeshRenderer() => _mesh_renderer;

    public bool IsVertexAttributeAvailable()
    {
        if (!IsValid()) return false;

        var result = true;
        var mesh = _mesh_filter.sharedMesh;
        var attributes = mesh.GetVertexAttributes();
        result &= attributes.Length == 4;
        result &= mesh.HasVertexAttribute(VertexAttribute.Position);
        result &= mesh.HasVertexAttribute(VertexAttribute.Normal);
        result &= mesh.HasVertexAttribute(VertexAttribute.Tangent);
        result &= mesh.HasVertexAttribute(VertexAttribute.TexCoord0);

        return result;
    }
    
    // 结构体使用顺序布局，严格按照先后顺序在内存中记录
    [StructLayout(LayoutKind.Sequential)]
    public struct TriangleLightVertex
    {
        public Vector3 position;
        public Vector3 normal;
        public Vector4 tangent;
        public Vector2 uv;
    }

    #region Buffer Data Gather

    [CanBeNull]
    public TriangleLightVertex[] GetCpuVertexBuffer()
    {
        if (!IsValid() || !IsVertexAttributeAvailable()) return null;

        var mesh = _mesh_filter.sharedMesh;
        var buffer = new TriangleLightVertex[mesh.vertexCount];
        // 下列代码可能造成内存泄漏，最好用临时变量接管Vertex Buffer并手动释放
        // mesh.GetVertexBuffer(0).GetData(buffer);
        var gfxBuffer = mesh.GetVertexBuffer(0);
        gfxBuffer.GetData(buffer);
        gfxBuffer.Release();
        gfxBuffer = null;

        return buffer;
    }
    
    [CanBeNull]
    public GraphicsBuffer GetGpuVertexBuffer() => IsValid() ? _mesh_filter.sharedMesh.GetVertexBuffer(0) : null;

    [CanBeNull]
    public int[] GetCpuIndexBuffer()
    {
        if (!IsValid() || !IsVertexAttributeAvailable()) return null;
        
        var mesh = _mesh_filter.sharedMesh;
        return mesh.GetIndices(0);
    }
    
    [CanBeNull]
    public GraphicsBuffer GetGpuIndexBuffer() => IsValid() ? _mesh_filter.sharedMesh.GetIndexBuffer() : null;

    #endregion

    #region Debug

    public static void OutputDebugVerticesStr(TriangleLightVertex[] vertices, string title = "")
    {
        var msg = string.IsNullOrEmpty(title) ? title : (title + "\n\n");
        var number = 1;
        foreach (var vertex in vertices)
        {
            msg += $"{number}. Position = ({vertex.position}); Normal = ({vertex.normal}); Tangent = ({vertex.tangent}); UV0 = ({vertex.uv}) \n\n";
            number += 1;
        }
        Debug.Log(msg); 
    }

    public static void OutputDebugIndicesStr(int[] indices, string title = "")
    {
        var msg = string.IsNullOrEmpty(title) ? title : (title + "\n\n");
        var number = 1;
        for (int i = 0; i < indices.Length / 3; ++i)
        {
            msg += $"triangle{number}. [{indices[3 * i]}, {indices[3 * i + 1]}, {indices[3 * i + 2]}]\n\n";
            number += 1;
        }
        Debug.Log(msg);
    }

    public void DebugCpuVertexBuffer()
    {
        var vertices = GetCpuVertexBuffer();
        if (vertices == null)
        {
            Debug.Log("Cpu Vertex Buffer is null");
            return;
        }

        OutputDebugVerticesStr(vertices, $"{gameObject.name} 的vertex buffer数据如下：");
    }

    public void DebugCpuIndexBuffer()
    {
        var indices = GetCpuIndexBuffer();
        if (indices == null)
        {
            Debug.Log("Cpu Index Buffer is null");
            return;
        }
        
        OutputDebugIndicesStr(indices, $"{gameObject.name} 的index buffer数据如下：");
    }

    #endregion

    private Color _light_color = Color.black;

    private MeshFilter _mesh_filter;
    private MeshRenderer _mesh_renderer;
    private Material _material;

    private static class MaterialParam
    {
        public static readonly int _BaseColor = Shader.PropertyToID("_BaseColor");
        public static readonly int _BaseMap = Shader.PropertyToID("_BaseMap");
    }

    void OnEnable()
    {
        /*DebugCpuVertexBuffer();
        DebugCpuIndexBuffer();*/
    }

    void Update()
    {
        RecollectAllComponents();
        if (!IsValid()) return;
        
        if (_material.HasProperty(MaterialParam._BaseColor))
            _light_color = _material.GetColor(MaterialParam._BaseColor);
        else
            _light_color = Color.black;
    }
    
    public void RecollectAllComponents()
    {
        _mesh_filter = gameObject.GetComponent<MeshFilter>();
        _mesh_renderer = gameObject.GetComponent<MeshRenderer>();
        if (_mesh_renderer != null) _material = _mesh_renderer.sharedMaterial;
    }
}
