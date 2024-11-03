using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.InteropServices;
using JetBrains.Annotations;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering.Universal.Internal;

public class RTXDIMinimalFeature : ScriptableRendererFeature
{
    private class RTXDIPrepareLightPass : ScriptableRenderPass
    {
        public bool DebugLightDataBuffer = false;
        
        private ComputeShader _prepare_light_cs = null;
        private readonly int _prepare_light_kernel;
        
        private GraphicsBuffer _merged_vertex_buffer_gpu = null;
        private GraphicsBuffer _merged_index_buffer_gpu = null;
        private GraphicsBuffer _light_task_buffer_gpu = null;
        private GraphicsBuffer _light_data_buffer_gpu = null;

        [StructLayout(LayoutKind.Sequential)]
        public struct PrepareLightsTask
        {
            public Vector3 emissiveColor;
            public uint triangleCount;
            public uint lightBufferOffset;
        }
        
        [StructLayout(LayoutKind.Sequential)]
        public struct RAB_LightInfo
        {
            // uint4[0]
            public Vector3 center;
            public uint scalars; // 2x float16
    
            // uint4[1]
            public uint2 radiance; // fp16x4
            public uint direction1; // oct-encoded
            public uint direction2; // oct-encoded
        };
        
        private Vector3 octToNdirSigned(Vector2 p)
        {
            // https://twitter.com/Stubbesaurus/status/937994790553227264
            Vector3 n = new Vector3(p.x, p.y, 1.0f - Mathf.Abs(p.x) - Mathf.Abs(p.y));
            float t = Mathf.Max(0, -n.z);
            n.x += n.x >= 0.0 ? -t : t;
            n.y += n.y >= 0.0 ? -t : t;
            return Vector3.Normalize(n);
        }
        
        private Vector3 octToNdirUnorm32(uint pUnorm)
        {
            Vector2 p;
            p.x = Mathf.Clamp01((pUnorm & 0xffff) / 0xfffe);
            p.y = Mathf.Clamp01((pUnorm >> 16) / 0xfffe);
            p.x = p.x * 2.0f - 1.0f;
            p.y = p.y * 2.0f - 1.0f;
            return octToNdirSigned(p);
        }
        
        private Vector2 Unpack_R16G16_FLOAT(uint rg)
        {
            uint2 d = new uint2(rg, (rg >> 16));
            return math.f16tof32(d);
        }

        private Vector4 Unpack_R16G16B16A16_FLOAT(uint2 rgba)
        {
            return new Vector4(Unpack_R16G16_FLOAT(rgba.x).x, Unpack_R16G16_FLOAT(rgba.x).y, Unpack_R16G16_FLOAT(rgba.y).x,
                Unpack_R16G16_FLOAT(rgba.y).y);
        }

        private static class GpuParams
        {
            public static readonly int HAS_POLYMORPHIC_LIGHTS = Shader.PropertyToID("HAS_POLYMORPHIC_LIGHTS");
            
            public static readonly int numTasks = Shader.PropertyToID("numTasks");
            public static readonly int TaskBuffer = Shader.PropertyToID("TaskBuffer");
            public static readonly int LightVertexBuffer = Shader.PropertyToID("LightVertexBuffer");
            public static readonly int LightIndexBuffer = Shader.PropertyToID("LightIndexBuffer");
            public static readonly int LightDataBuffer = Shader.PropertyToID("LightDataBuffer");
        }
        
        public RTXDIPrepareLightPass()
        {
            _prepare_light_cs = Resources.Load<ComputeShader>("Shaders/PrepareLights");
            if (_prepare_light_cs != null) _prepare_light_kernel = _prepare_light_cs.FindKernel("PrepareLights");
        }

        public override unsafe void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            var cmd = CommandBufferPool.Get("RTXDI: Prepare Lights");

            var tasks = new List<PrepareLightsTask>();
            
            bool prepare_polymorphic_light = false;
            uint light_buffer_offset = 0;
            
            // Generate cpu-gpu side vertex buffer.
            {
                var polyLights = FindObjectsByType<PolymorphicLight>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);

                // 收集每一盏灯光的cpu端顶点数据
                var local_vertex_buffers_cpu = new List<PolymorphicLight.TriangleLightVertex[]>(polyLights.Length);
                var local_index_buffers_cpu = new List<int[]>(polyLights.Length);
                var total_vertex_count = 0;
                var total_index_count = 0;
                foreach (var polyLight in polyLights)
                {
                    var vertex_buffer = polyLight.GetCpuVertexBuffer();
                    var index_buffer = polyLight.GetCpuIndexBuffer();

                    if (vertex_buffer != null && index_buffer != null)
                    {
                        local_vertex_buffers_cpu.Add(vertex_buffer);
                        local_index_buffers_cpu.Add(index_buffer);
                        total_vertex_count += vertex_buffer.Length;
                        total_index_count += index_buffer.Length;

                        PrepareLightsTask task;
                        task.emissiveColor = polyLight.GetLightColor();
                        task.lightBufferOffset = light_buffer_offset;
                        task.triangleCount = (uint)(index_buffer.Length / 3);

                        light_buffer_offset += task.triangleCount;
                        
                        tasks.Add(task);
                    }
                }

                if (total_vertex_count > 0 && total_index_count > 0)
                {
                    // 将所有灯光的cpu顶点数据合并到一起
                    var merged_vertex_buffer_cpu = new PolymorphicLight.TriangleLightVertex[total_vertex_count];
                    var merged_index_buffer_cpu = new int[total_index_count];
                    var destination_index = 0;
                    foreach (var local_vertex_buffer in local_vertex_buffers_cpu)
                    {
                        Array.Copy(local_vertex_buffer, 0, merged_vertex_buffer_cpu, destination_index,
                            local_vertex_buffer.Length);
                        destination_index += local_vertex_buffer.Length;
                    }

                    destination_index = 0;
                    foreach (var local_index_buffer in local_index_buffers_cpu)
                    {
                        Array.Copy(local_index_buffer, 0, merged_index_buffer_cpu, destination_index,
                            local_index_buffer.Length);
                        destination_index += local_index_buffer.Length;
                    }
         
                    // 将合并后的cpu灯光数据上载到gpu
                    _merged_vertex_buffer_gpu?.Release();
                    _merged_vertex_buffer_gpu = new GraphicsBuffer(GraphicsBuffer.Target.Structured, total_vertex_count,
                        sizeof(PolymorphicLight.TriangleLightVertex));
                    _merged_vertex_buffer_gpu.SetData(merged_vertex_buffer_cpu);
                    
                    _merged_index_buffer_gpu?.Release();
                    _merged_index_buffer_gpu = new GraphicsBuffer(GraphicsBuffer.Target.Structured, total_index_count,
                        sizeof(int));
                    _merged_index_buffer_gpu.SetData(merged_index_buffer_cpu);
                    
                    _light_task_buffer_gpu?.Release();
                    _light_task_buffer_gpu = new GraphicsBuffer(GraphicsBuffer.Target.Structured, tasks.Count,
                        sizeof(PrepareLightsTask));
                    _light_task_buffer_gpu.SetData(tasks);
                    
                    _light_data_buffer_gpu?.Release();
                    _light_data_buffer_gpu = new GraphicsBuffer(GraphicsBuffer.Target.Structured, total_index_count / 3,
                        sizeof(RAB_LightInfo));

                    prepare_polymorphic_light = true;
                }
                else
                {
                    prepare_polymorphic_light = false;
                }
            }
            
            // Enable/Disable light prepare computation.
            cmd.SetGlobalInt(GpuParams.HAS_POLYMORPHIC_LIGHTS, prepare_polymorphic_light ? 1 : 0);
            
            // Dispatch Light Prepare Tasks.
            if(prepare_polymorphic_light)
            {
                cmd.SetComputeIntParam(_prepare_light_cs, GpuParams.numTasks, _light_task_buffer_gpu.count);
                cmd.SetComputeBufferParam(_prepare_light_cs, _prepare_light_kernel, GpuParams.LightVertexBuffer, _merged_vertex_buffer_gpu);
                cmd.SetComputeBufferParam(_prepare_light_cs, _prepare_light_kernel, GpuParams.LightIndexBuffer, _merged_index_buffer_gpu);
                cmd.SetComputeBufferParam(_prepare_light_cs, _prepare_light_kernel, GpuParams.TaskBuffer, _light_task_buffer_gpu);
                cmd.SetComputeBufferParam(_prepare_light_cs, _prepare_light_kernel, GpuParams.LightDataBuffer, _light_data_buffer_gpu);
                cmd.DispatchCompute(_prepare_light_cs, _prepare_light_kernel,
                    Mathf.CeilToInt((float)light_buffer_offset / 256), 1, 1);
            }
            
            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }

        public override void FrameCleanup(CommandBuffer cmd)
        {
            if (DebugLightDataBuffer)
            {
                if (_light_data_buffer_gpu == null)
                {
                    Debug.Log("light data buffer gpu is null");
                    return;
                }
                
                var light_data_cpu = new RAB_LightInfo[_light_data_buffer_gpu.count];
                _light_data_buffer_gpu.GetData(light_data_cpu);

                foreach (var data in light_data_cpu)
                {
                    var msg = $"center={data.center}, direction1={octToNdirUnorm32(data.direction1)}, radiance={Unpack_R16G16B16A16_FLOAT(data.radiance)}";
                    Debug.Log(msg);
                }
            }
        }

        public void Release()
        {
            
        }
    }
    
    private class RTXDIPass : ScriptableRenderPass
    {
        public override void Configure(CommandBuffer cmd, RenderTextureDescriptor cameraTextureDescriptor)
        {
            base.Configure(cmd, cameraTextureDescriptor);
        }

        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            
        }
    }

    private class HistoryGBufferPass : ScriptableRenderPass
    {
        private RTHandle _prev_normal;
        private RTHandle _prev_depth;
        
        private static class GpuParams
        {
            public static readonly int _PrevNormalBuffer = Shader.PropertyToID("_PrevNormalBuffer");
            public static readonly int _PrevDepthBuffer = Shader.PropertyToID("_PrevDepthBuffer");
        }

        public override void Configure(CommandBuffer cmd, RenderTextureDescriptor cameraTextureDescriptor)
        {
            var renderer_data = GetRendererDataWithinFeature();
            var accurate_normal = false;
            if (renderer_data == null)
                Debug.LogWarning($"{nameof(HistoryGBufferPass)}: 获取Universal Renderer Data失败，法线回退到一般精度");
            else
                accurate_normal = renderer_data.accurateGbufferNormals;
            
            var normal_descriptor = cameraTextureDescriptor;
            // Reference: DeferredLights.cs - GetGBufferFormat
            normal_descriptor.graphicsFormat = accurate_normal ? GraphicsFormat.R8G8B8A8_UNorm : DepthNormalOnlyPass.GetGraphicsFormat();
            RenderingUtils.ReAllocateIfNeeded(ref _prev_normal, normal_descriptor);

            var depth_descriptor = cameraTextureDescriptor;
            depth_descriptor.depthBufferBits = 32;
            depth_descriptor.graphicsFormat = GraphicsFormat.R32_SFloat;
            RenderingUtils.ReAllocateIfNeeded(ref _prev_depth, depth_descriptor);
        }

        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            var cmd = CommandBufferPool.Get("History GBuffer Copy Pass");
            {
                cmd.CopyTexture(_prev_depth, renderingData.cameraData.renderer.cameraDepthTargetHandle);
            }
            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }

        public void Release() { }
    }

    public bool DebugLightData = false;

    private RTXDIPrepareLightPass _prepare_light_pass;
    private HistoryGBufferPass _history_gbuffer_pass;
    
    public override void Create()
    {
        if (!isActive)
        {
            _prepare_light_pass?.Release();
            _history_gbuffer_pass?.Release();
            return;
        }
        
        _prepare_light_pass ??= new RTXDIPrepareLightPass();
        _history_gbuffer_pass ??= new HistoryGBufferPass();
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        _prepare_light_pass.renderPassEvent = RenderPassEvent.AfterRenderingGbuffer;
        _prepare_light_pass.DebugLightDataBuffer = DebugLightData;
        renderer.EnqueuePass(_prepare_light_pass);
        
        /*_history_gbuffer_pass.renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;
        renderer.EnqueuePass(_history_gbuffer_pass);*/
    }

    [CanBeNull]
    private static UniversalRendererData GetRendererDataWithinFeature()
    {
        // 获取当前使用的rp asset
        var rp_asset = GraphicsSettings.renderPipelineAsset;
        
        // 尝试转换为URP Asset，不成功则返回null
        if (rp_asset == null || !(rp_asset is UniversalRenderPipelineAsset)) return null;
        
        // 尝试在URP Renderer Data中寻找当前Feature，找到的话就返回对应的Renderer Data，反之返回null
        Type urp_asset_type = rp_asset.GetType();
        FieldInfo renderer_data_list_field = urp_asset_type.GetField("m_RendererDataList", BindingFlags.Instance | BindingFlags.NonPublic);
        if (renderer_data_list_field != null)
        {
            ScriptableRendererData[] renderer_data_list = renderer_data_list_field.GetValue(rp_asset) as ScriptableRendererData[];

            foreach (var renderer_data in renderer_data_list)
            {
                var feature = (RTXDIMinimalFeature)renderer_data.rendererFeatures.Find(x =>
                        x.GetType() == typeof(RTXDIMinimalFeature));
                if (feature != null && renderer_data is UniversalRendererData) return (UniversalRendererData)renderer_data;
            }
        }

        return null;
    }
}
