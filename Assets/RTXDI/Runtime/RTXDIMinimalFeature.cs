using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using JetBrains.Annotations;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering.Universal.Internal;

public sealed class RTXDIMinimalFeature : ScriptableRendererFeature
{
    private sealed class RTXDIPass : ScriptableRenderPass
    {
        #region [Structure]
        
        [StructLayout(LayoutKind.Sequential)]
        private struct PrepareLightsTask
        {
            public Vector3 emissiveColor;
            public uint triangleCount;
            
            public uint lightBufferOffset;
            public uint vertexOffset;
            public int emissiveTextureIndex;
            public uint padding;
            
            public float4x4 localToWorld;
        }
        
        [StructLayout(LayoutKind.Sequential)]
        private struct RAB_LightInfo
        {
            // uint4[0]
            public Vector3 center;
            public uint scalars; // 2x float16
    
            // uint4[1]
            public uint2 radiance; // fp16x4
            public uint direction1; // oct-encoded
            public uint direction2; // oct-encoded
            
            public Vector4 debug;
        };
        
        [StructLayout(LayoutKind.Sequential)]
        private struct TriangleLightDebug
        {
            public Vector3 basePoint;
            public Vector3 v1;
            public Vector3 v2;
            public Vector3 edge1;
            public Vector3 edge2;
            public Vector3 radiance;
        }
        
        [StructLayout(LayoutKind.Sequential)]
        private struct RTXDI_LightBufferRegion
        {
            public uint firstLightIndex;
            public uint numLights;
            public uint pad1;
            public uint pad2;
        };

        [StructLayout(LayoutKind.Sequential)]
        private struct RTXDI_EnvironmentLightBufferParameters
        {
            public uint lightPresent;
            public uint lightIndex;
            public uint pad1;
            public uint pad2;
        };

        [StructLayout(LayoutKind.Sequential)]
        private struct RTXDI_LightBufferParameters
        {
            public RTXDI_LightBufferRegion localLightBufferRegion;
            public RTXDI_LightBufferRegion infiniteLightBufferRegion;
            public RTXDI_EnvironmentLightBufferParameters environmentLightParams;
        };
        
        [StructLayout(LayoutKind.Sequential)]
        private struct RTXDI_RuntimeParameters
        {
            public uint neighborOffsetMask; // Spatial
            public uint activeCheckerboardField; // 0 - no checkerboard, 1 - odd pixels, 2 - even pixels
            public uint pad1;
            public uint pad2;
        };
        
        [StructLayout(LayoutKind.Sequential)]
        private struct RTXDI_ReservoirBufferParameters
        {
            public uint reservoirBlockRowPitch;
            public uint reservoirArrayPitch;
            public uint pad1;
            public uint pad2;
        };
        
        [StructLayout(LayoutKind.Sequential)]
        private struct ResamplingConstants
        {
            public RTXDI_RuntimeParameters runtimeParams;
            public RTXDI_LightBufferParameters lightBufferParams;
            public RTXDI_ReservoirBufferParameters restirDIReservoirBufferParams;

            public uint frameIndex;
            public uint numInitialSamples;
            public uint numSpatialSamples;
            public uint pad1;

            public uint numInitialBRDFSamples;
            public float brdfCutoff;
            public uint2 pad2;

            public uint enableResampling;
            public uint unbiasedMode;
            public uint inputBufferIndex;
            public uint outputBufferIndex;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct RTXDI_PackedDIReservoir
        {
            uint lightData;
            uint uvData;
            uint mVisibility;
            uint distanceAge;
            float targetPdf;
            float weight;
        }

        #endregion

        #region [Prepare Polymorphic Lights]

        public bool DebugLightDataBuffer = false;

        private ComputeShader _prepare_light_cs = null;
        private readonly int _prepare_light_kernel;
        
        private GraphicsBuffer _merged_vertex_buffer_gpu = null;
        private GraphicsBuffer _merged_index_buffer_gpu = null;
        private GraphicsBuffer _light_task_buffer_gpu = null;
        private GraphicsBuffer _light_data_buffer_gpu = null;
        private GraphicsBuffer _geometry_instance_to_light_gpu = null;
        
        private RTXDI_LightBufferParameters _light_buffer_parameters = new RTXDI_LightBufferParameters();

        private class PrepareLightContext
        {
            public int _last_total_vertex_count = 0;
            public int _last_total_index_count = 0;
            public int _last_light_task_count = 0;
            public int _last_light_reservoir_count = 0;
            public List<Texture2D> _last_emissive_textures_cpu = new List<Texture2D>();
            
            public Texture2DArray _polymorphic_light_texture_array;
            public int _task_count = 0;
            public uint _total_triangle_count = 0;
        }
        private PrepareLightContext _prepare_light_context = new PrepareLightContext();

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

        #endregion

        #region [ReSTIR Lighting]

        private const int RTXDI_RESERVOIR_BLOCK_SIZE = 16;
        
        private const int NEIGHBOR_OFFSET_COUNT = 8192;
        private ComputeBuffer _neighbor_offset_buffer = null;
        
        private ResamplingConstants _resampling_constants;
        private ComputeBuffer _resampling_constants_buffer_gpu = null;
        private RayTracingShader _rtxdi_raytracing_shader = null;
        private RayTracingAccelerationStructure _scene_tlas = null;

        public RTHandle GetShadingOutput() => _shading_output;
        private RTHandle _shading_output;

        private const int NUM_RESTIR_DI_RESERVOIR_BUFFERS = 3;
        private GraphicsBuffer _light_reservoir_buffer_gpu = null;

        #endregion
        
        #region [Previous GBuffer]

        private RTHandle _input_prev_gbuffer0 = null;
        private RTHandle _input_prev_gbuffer1 = null;
        private RTHandle _input_prev_gbuffer2 = null;
        private RTHandle _input_prev_depth = null;

        private bool IsHistoryGBufferReady() => _input_prev_gbuffer0 != null && _input_prev_gbuffer1 != null &&
                                                _input_prev_gbuffer2 != null && _input_prev_depth != null;

        #endregion

        public bool Executable() => _is_pass_executable;
        
        private bool _is_pass_executable = true;

        private static class GpuParams
        {
            // Buffer & Textures & TLAS
            public static readonly int GeometryInstanceToLight = Shader.PropertyToID("GeometryInstanceToLight");
            public static readonly int ShadingOutput = Shader.PropertyToID("ShadingOutput");
            public static readonly int ResampleConstants = Shader.PropertyToID("ResampleConstants");
            public static readonly int LightDataBuffer = Shader.PropertyToID("LightDataBuffer");
            public static readonly int SceneTLAS = Shader.PropertyToID("SceneTLAS");
            public static readonly int LightReservoirs = Shader.PropertyToID("LightReservoirs");
            public static readonly int NeighborOffsets = Shader.PropertyToID("NeighborOffsets");

            // Prepare Lights
            public static readonly int HAS_POLYMORPHIC_LIGHTS = Shader.PropertyToID("HAS_POLYMORPHIC_LIGHTS");
            public static readonly int numTasks = Shader.PropertyToID("numTasks");
            public static readonly int TaskBuffer = Shader.PropertyToID("TaskBuffer");
            public static readonly int LightVertexBuffer = Shader.PropertyToID("LightVertexBuffer");
            public static readonly int LightIndexBuffer = Shader.PropertyToID("LightIndexBuffer");
            public static readonly int EmissiveTextureArray = Shader.PropertyToID("EmissiveTextureArray");
            
            // Previous GBuffer
            public static readonly int _PreviousCameraDepthTexture = Shader.PropertyToID("_PreviousCameraDepthTexture");
            public static readonly int _PreviousGBuffer0 = Shader.PropertyToID("_PreviousGBuffer0");
            public static readonly int _PreviousGBuffer1 = Shader.PropertyToID("_PreviousGBuffer1");
            public static readonly int _PreviousGBuffer2 = Shader.PropertyToID("_PreviousGBuffer2");
        }

        public RTXDIPass()
        {
            // Shaders
            _rtxdi_raytracing_shader = Resources.Load<RayTracingShader>("Shaders/DiRender");
            _prepare_light_cs = Resources.Load<ComputeShader>("Shaders/PrepareLights");
            if (_prepare_light_cs != null) _prepare_light_kernel = _prepare_light_cs.FindKernel("PrepareLights");

            // TLAS
            RayTracingAccelerationStructure.RASSettings setting = new RayTracingAccelerationStructure.RASSettings
                (RayTracingAccelerationStructure.ManagementMode.Automatic, RayTracingAccelerationStructure.RayTracingModeMask.Everything,  255);
            _scene_tlas = new RayTracingAccelerationStructure(setting);
            
            // Low Diff Sequence
            InitializeNeighborOffsets();
        }

        public override unsafe void Configure(CommandBuffer cmd, RenderTextureDescriptor cameraTextureDescriptor)
        {
            var rtxdiSettings = VolumeManager.instance.stack.GetComponent<RTXDISettings>();
            var polyLights = FindObjectsByType<PolymorphicLight>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
            
            _is_pass_executable = rtxdiSettings != null && rtxdiSettings.IsActive();
            _is_pass_executable &= _rtxdi_raytracing_shader != null;
            _is_pass_executable &= _prepare_light_cs != null;
            _is_pass_executable &= polyLights.Length > 0; // TODO: 为减少复杂性，目前只支持Polymorphic Light，Infinite Light以及Environment Light不受支持
            if (!_is_pass_executable) return;

            #region 创建Prepare Lights Task并提供一些数量信息

            var tasks = new List<PrepareLightsTask>();
            uint light_buffer_offset = 0;
            
            if (DebugLightDataBuffer) OutputLightDataStr();

            // 收集每一盏灯光的cpu端顶点数据
            var local_vertex_buffers_cpu = new List<PolymorphicLight.TriangleLightVertex[]>(polyLights.Length);
            var local_index_buffers_cpu = new List<int[]>(polyLights.Length);
            var geometry_instance_to_light = new List<uint>(polyLights.Length); // 记录每一个polymorphic light相对前一个polymorphic light的三角形偏移量
            var local_emissive_textures_cpu = new List<Texture2D>();
            var total_vertex_count = 0;
            var total_index_count = 0;
            foreach (var polyLight in polyLights)
            {
                if (!polyLight.IsValid()) continue;
                
                var vertex_buffer = polyLight.GetCpuVertexBuffer();
                var index_buffer = polyLight.GetCpuIndexBuffer();
                var mesh_renderer = polyLight.GetMeshRenderer(); 

                if (vertex_buffer != null && index_buffer != null && mesh_renderer != null)
                {
                    local_vertex_buffers_cpu.Add(vertex_buffer);
                    local_index_buffers_cpu.Add(index_buffer);

                    var emissive_texture = polyLight.GetEmissiveTexture();
                    if (emissive_texture != null) local_emissive_textures_cpu.Add(emissive_texture);

                    PrepareLightsTask task;
                    task.emissiveColor = polyLight.GetLightColor();
                    task.triangleCount = (uint)(index_buffer.Length / 3);
                    task.lightBufferOffset = light_buffer_offset;
                    task.vertexOffset = (uint)total_vertex_count;
                    task.emissiveTextureIndex = emissive_texture == null ? -1 : (local_emissive_textures_cpu.Count - 1);
                    task.padding = 0u;
                    task.localToWorld = polyLight.transform.localToWorldMatrix;

                    geometry_instance_to_light.Add(light_buffer_offset);

                    // light_buffer_offset同时也是total_triangle_count
                    light_buffer_offset += task.triangleCount;
                    total_vertex_count += vertex_buffer.Length;
                    total_index_count += index_buffer.Length;

                    tasks.Add(task);
                }
            }

            #endregion
            
            // 更新prepare light上下文信息
            _prepare_light_context._task_count = tasks.Count;
            _prepare_light_context._total_triangle_count = light_buffer_offset;

            // 推送GPU常量
            FillResamplingConstants(rtxdiSettings, cameraTextureDescriptor.width, cameraTextureDescriptor.height);

            #region 重新分配GPU Resource
            
            // 按需分配ReSTIR Lighting Texture
            var shadingOutputDesc = cameraTextureDescriptor;
            shadingOutputDesc.depthBufferBits = 0;
            shadingOutputDesc.graphicsFormat = GraphicsFormat.B10G11R11_UFloatPack32;
            shadingOutputDesc.autoGenerateMips = false;
            shadingOutputDesc.useMipMap = false;
            shadingOutputDesc.enableRandomWrite = true;
            RenderingUtils.ReAllocateIfNeeded(ref _shading_output, shadingOutputDesc);

            // 按需分配Light Task Buffer以及Geometry Instance to light buffer
            if (_prepare_light_context._last_light_task_count != tasks.Count && tasks.Count > 0)
            {
                _prepare_light_context._last_light_task_count = tasks.Count;
                _light_task_buffer_gpu?.Release();
                _light_task_buffer_gpu = new GraphicsBuffer(GraphicsBuffer.Target.Structured, tasks.Count, sizeof(PrepareLightsTask));
                
                // Geometry Instance to light的数组大小是跟随Task一起变化的，所以无需额外的变量来跟踪
                _geometry_instance_to_light_gpu?.Release();
                _geometry_instance_to_light_gpu = new GraphicsBuffer(GraphicsBuffer.Target.Structured, geometry_instance_to_light.Count, sizeof(uint));
            }
            _light_task_buffer_gpu?.SetData(tasks);
            _geometry_instance_to_light_gpu?.SetData(geometry_instance_to_light);

            // 按需分配Light Reservoir Buffer
            var light_reservoir_count = (int)_resampling_constants.restirDIReservoirBufferParams.reservoirArrayPitch * NUM_RESTIR_DI_RESERVOIR_BUFFERS;
            if (_prepare_light_context._last_light_reservoir_count != light_reservoir_count)
            {
                _light_reservoir_buffer_gpu?.Release();
                _light_reservoir_buffer_gpu = new GraphicsBuffer(GraphicsBuffer.Target.Structured, light_reservoir_count, sizeof(RTXDI_PackedDIReservoir));
            }

            // 按需分配场景网格灯的Vertex Buffer和Index Buffer
            if (total_vertex_count > 0 && total_index_count > 0 &&
                _prepare_light_context._last_total_vertex_count != total_vertex_count &&
                _prepare_light_context._last_total_index_count != total_index_count)
            {
                _prepare_light_context._last_total_vertex_count = total_vertex_count;
                _prepare_light_context._last_total_index_count = total_index_count;
                
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
                _merged_vertex_buffer_gpu = new GraphicsBuffer(GraphicsBuffer.Target.Structured, total_vertex_count, sizeof(PolymorphicLight.TriangleLightVertex));
                _merged_vertex_buffer_gpu.SetData(merged_vertex_buffer_cpu);

                _merged_index_buffer_gpu?.Release();
                _merged_index_buffer_gpu = new GraphicsBuffer(GraphicsBuffer.Target.Structured, total_index_count, sizeof(int));
                _merged_index_buffer_gpu.SetData(merged_index_buffer_cpu);

                _light_data_buffer_gpu?.Release();
                _light_data_buffer_gpu = new GraphicsBuffer(GraphicsBuffer.Target.Structured, total_index_count / 3, sizeof(RAB_LightInfo));
            } 

            // 按需分配网格灯用到的纹理
            if (local_emissive_textures_cpu.Count > 0 && !_prepare_light_context._last_emissive_textures_cpu.SequenceEqual(local_emissive_textures_cpu))
            {
                _prepare_light_context._last_emissive_textures_cpu = local_emissive_textures_cpu;
                
                var template = local_emissive_textures_cpu[0];
                _prepare_light_context._polymorphic_light_texture_array = new Texture2DArray(template.width, template.height,
                    local_emissive_textures_cpu.Count, TextureFormat.DXT1, true); 

                for (int j = 0; j < local_emissive_textures_cpu.Count; ++j)
                {
                    var texture = local_emissive_textures_cpu[j];
                    Graphics.CopyTexture(texture, 0, _prepare_light_context._polymorphic_light_texture_array, j);
                }
            }

            #endregion
        }

        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            if (!_is_pass_executable) return;

            // -----------------------------------------------------------
            //                          命令录制
            // -----------------------------------------------------------
            var cmd = CommandBufferPool.Get("RTXDI Pass");

            using (new ProfilingScope(cmd, new ProfilingSampler("RTXDI: Prepare Polymorphic Lights")))
            {
                cmd.SetComputeIntParam(_prepare_light_cs, GpuParams.numTasks, _prepare_light_context._task_count);
                cmd.SetComputeBufferParam(_prepare_light_cs, _prepare_light_kernel, GpuParams.LightVertexBuffer, _merged_vertex_buffer_gpu);
                cmd.SetComputeBufferParam(_prepare_light_cs, _prepare_light_kernel, GpuParams.LightIndexBuffer, _merged_index_buffer_gpu);
                cmd.SetComputeBufferParam(_prepare_light_cs, _prepare_light_kernel, GpuParams.TaskBuffer, _light_task_buffer_gpu);
                cmd.SetComputeBufferParam(_prepare_light_cs, _prepare_light_kernel, GpuParams.LightDataBuffer, _light_data_buffer_gpu);
                cmd.SetComputeTextureParam(_prepare_light_cs, _prepare_light_kernel, GpuParams.EmissiveTextureArray, _prepare_light_context._polymorphic_light_texture_array);
                cmd.DispatchCompute(_prepare_light_cs, _prepare_light_kernel,
                    Mathf.CeilToInt((float)_prepare_light_context._total_triangle_count / 256), 1, 1);
            }

            using (new ProfilingScope(cmd, new ProfilingSampler("RTXDI: ReSTIR Lighting")))
            {
                // Ray Tracing Stuff.
                cmd.BuildRayTracingAccelerationStructure(_scene_tlas);
                cmd.SetRayTracingAccelerationStructure(_rtxdi_raytracing_shader, GpuParams.SceneTLAS, _scene_tlas);
                cmd.SetRayTracingShaderPass(_rtxdi_raytracing_shader, "RTXDIVisibilityTracing");

                // Buffer and Texture Resource.
                if (IsHistoryGBufferReady())
                {
                    cmd.SetRayTracingTextureParam(_rtxdi_raytracing_shader, GpuParams._PreviousGBuffer0, _input_prev_gbuffer0);
                    cmd.SetRayTracingTextureParam(_rtxdi_raytracing_shader, GpuParams._PreviousGBuffer1, _input_prev_gbuffer1);
                    cmd.SetRayTracingTextureParam(_rtxdi_raytracing_shader, GpuParams._PreviousGBuffer2, _input_prev_gbuffer2);
                    cmd.SetRayTracingTextureParam(_rtxdi_raytracing_shader, GpuParams._PreviousCameraDepthTexture, _input_prev_depth);
                }
                cmd.SetRayTracingBufferParam(_rtxdi_raytracing_shader, GpuParams.NeighborOffsets, _neighbor_offset_buffer);
                cmd.SetGlobalBuffer(GpuParams.LightReservoirs, _light_reservoir_buffer_gpu);
                cmd.SetGlobalBuffer(GpuParams.GeometryInstanceToLight, _geometry_instance_to_light_gpu);
                cmd.SetGlobalBuffer(GpuParams.ResampleConstants, _resampling_constants_buffer_gpu);
                cmd.SetGlobalBuffer(GpuParams.LightDataBuffer, _light_data_buffer_gpu);
                cmd.SetRayTracingTextureParam(_rtxdi_raytracing_shader, GpuParams.ShadingOutput, _shading_output);

                var cameraDescriptor = renderingData.cameraData.cameraTargetDescriptor;
                cmd.DispatchRays(_rtxdi_raytracing_shader, "RtxdiRayGen", (uint)cameraDescriptor.width, (uint)cameraDescriptor.height, 1);
            }

            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }

        public void Setup(bool debugLightData, RTHandle prevGBuffer0, RTHandle prevGBuffer1, RTHandle prevGBuffer2, RTHandle prevDepth)
        {
            DebugLightDataBuffer = debugLightData;
            _input_prev_gbuffer0 = prevGBuffer0;
            _input_prev_gbuffer1 = prevGBuffer1;
            _input_prev_gbuffer2 = prevGBuffer2;
            _input_prev_depth = prevDepth;
        }

        public void Release() 
        {
            _merged_vertex_buffer_gpu?.Release(); _merged_vertex_buffer_gpu = null;
            _merged_index_buffer_gpu?.Release(); _merged_index_buffer_gpu = null;
            _light_task_buffer_gpu?.Release(); _light_task_buffer_gpu = null;
            _light_data_buffer_gpu?.Release(); _light_data_buffer_gpu = null;
            _geometry_instance_to_light_gpu?.Release(); _geometry_instance_to_light_gpu = null;
            _resampling_constants_buffer_gpu?.Release(); _resampling_constants_buffer_gpu = null;
            _neighbor_offset_buffer?.Release(); _neighbor_offset_buffer = null;
            _light_reservoir_buffer_gpu?.Release(); _light_reservoir_buffer_gpu = null;
            _scene_tlas?.Release(); _scene_tlas = null; 
            _shading_output?.Release(); _shading_output = null;
        }

        private unsafe void FillResamplingConstants(RTXDISettings rtxdiSettings, int renderWidth, int renderHeight)
        {
            // 更新Light Buffer Parameters
            _light_buffer_parameters.localLightBufferRegion.firstLightIndex = 0;
            _light_buffer_parameters.localLightBufferRegion.numLights = _prepare_light_context._total_triangle_count;
            _light_buffer_parameters.infiniteLightBufferRegion.firstLightIndex = 0;
            _light_buffer_parameters.infiniteLightBufferRegion.numLights = 0;
            _light_buffer_parameters.environmentLightParams.lightIndex = 0xffffffffu; // INVALID
            _light_buffer_parameters.environmentLightParams.lightPresent = 0;

            // 更新Resampling Constants（整个RTXDI过程用到的唯一GPU全局Constant Buffer）
            _resampling_constants.runtimeParams.activeCheckerboardField = 0;
            _resampling_constants.runtimeParams.neighborOffsetMask = NEIGHBOR_OFFSET_COUNT - 1; // NeighborOffsetCount = 8192
            _resampling_constants.runtimeParams.pad1 = 0;
            _resampling_constants.runtimeParams.pad2 = 0;

            _resampling_constants.lightBufferParams = _light_buffer_parameters;
            
            int renderWidthBlocks = (renderWidth + RTXDI_RESERVOIR_BLOCK_SIZE - 1) / RTXDI_RESERVOIR_BLOCK_SIZE;
            int renderHeightBlocks = (renderHeight + RTXDI_RESERVOIR_BLOCK_SIZE - 1) / RTXDI_RESERVOIR_BLOCK_SIZE;
            _resampling_constants.restirDIReservoirBufferParams.reservoirBlockRowPitch =
                (uint)renderWidthBlocks * (RTXDI_RESERVOIR_BLOCK_SIZE * RTXDI_RESERVOIR_BLOCK_SIZE);
            _resampling_constants.restirDIReservoirBufferParams.reservoirArrayPitch =
                _resampling_constants.restirDIReservoirBufferParams.reservoirBlockRowPitch * (uint)renderHeightBlocks;
            _resampling_constants.restirDIReservoirBufferParams.pad1 = 0;
            _resampling_constants.restirDIReservoirBufferParams.pad2 = 0;

            _resampling_constants.frameIndex = (uint)Time.frameCount;
            _resampling_constants.numInitialSamples = (uint)rtxdiSettings.numInitialSamples.value;
            _resampling_constants.numSpatialSamples = (uint)rtxdiSettings.numSpatialSamples.value;
            _resampling_constants.pad1 = 0;

            _resampling_constants.numInitialBRDFSamples = (uint)rtxdiSettings.numInitialBrdfSamples.value;
            _resampling_constants.brdfCutoff = rtxdiSettings.brdfCutoff.value;
            _resampling_constants.pad2 = new uint2(0, 0);

            _resampling_constants.enableResampling =
                IsHistoryGBufferReady() ? (rtxdiSettings.enableResampling.value ? 1u : 0u) : 0u;
            _resampling_constants.unbiasedMode = rtxdiSettings.unbiasedMode.value ? 1u : 0u;
            _resampling_constants.inputBufferIndex = ~(_resampling_constants.frameIndex & 1u);
            _resampling_constants.outputBufferIndex = _resampling_constants.frameIndex & 1u;
            
            // 将数据上载到GPU
            _resampling_constants_buffer_gpu ??=
                new ComputeBuffer( 1, sizeof(ResamplingConstants), ComputeBufferType.Default);
            _resampling_constants_buffer_gpu.SetData(new[] { _resampling_constants });
        }

        private void OutputLightDataStr()
        {
            if (_light_data_buffer_gpu == null)
            {
                Debug.Log("light data buffer gpu is null");
                return;
            }

            {
                var light_data_buffer_cpu = new RAB_LightInfo[_light_data_buffer_gpu.count];
                _light_data_buffer_gpu.GetData(light_data_buffer_cpu);
                
                var msg = "";
                var idx = 1;
                foreach (var triangle in light_data_buffer_cpu)
                {
                    msg += $"triangle{idx}: base={triangle.center} radiance={Unpack_R16G16B16A16_FLOAT(triangle.radiance)}\n\n";
                    idx += 1;
                }
                Debug.Log(msg);
            }
        }

        private unsafe void InitializeNeighborOffsets()
        {
            var offsets = new Vector2[NEIGHBOR_OFFSET_COUNT];
            Array.Fill(offsets, Vector2.zero);
            {
                int R = 250;
                const float phi2 = 1.0f / 1.3247179572447f;
                uint num = 0;
                float u = 0.5f;
                float v = 0.5f;
                while (num < NEIGHBOR_OFFSET_COUNT) 
                {
                    u += phi2;
                    v += phi2 * phi2;
                    if (u >= 1.0f) u -= 1.0f;
                    if (v >= 1.0f) v -= 1.0f;

                    float rSq = (u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f);
                    if (rSq > 0.25f)
                        continue;

                    offsets[num++] = new Vector2((u - 0.5f) * R / 128.0f, (v - 0.5f) * R / 128.0f);
                }
            }

            _neighbor_offset_buffer?.Release();
            _neighbor_offset_buffer = new ComputeBuffer( NEIGHBOR_OFFSET_COUNT, sizeof(Vector2), ComputeBufferType.Default);
            _neighbor_offset_buffer.SetData(offsets);
        }
    }

    private sealed class HistoryGBufferPass : ScriptableRenderPass
    {
        private RTHandle _prev_gBuffer0 = null; // Albedo
        private RTHandle _prev_gBuffer1 = null; // Specular and Metallic
        private RTHandle _prev_gBuffer2 = null; // Normal and Smoothness
        private RTHandle _prev_depth = null;

        private Shader _copy_gbuffer_ps;
        private Material _copy_gbuffer_mat;

        [CanBeNull]
        public RTHandle GetPrevGBuffer0() => _prev_gBuffer0;
        [CanBeNull]
        public RTHandle GetPrevGBuffer1() => _prev_gBuffer1;
        [CanBeNull]
        public RTHandle GetPrevGBuffer2() => _prev_gBuffer2;
        [CanBeNull]
        public RTHandle GetPrevDepth() => _prev_depth;

        private bool _is_pass_executable = true;

        public HistoryGBufferPass()
        {
            _copy_gbuffer_ps = Resources.Load<Shader>("Shaders/CopyGBuffer");
            if (_copy_gbuffer_ps != null) _copy_gbuffer_mat = CoreUtils.CreateEngineMaterial(_copy_gbuffer_ps);
        }

        private static class GpuParams
        {
            public static readonly int _BlitScaleBias = Shader.PropertyToID("_BlitScaleBias");
        }

        public override void Configure(CommandBuffer cmd, RenderTextureDescriptor cameraTextureDescriptor)
        {
            _is_pass_executable &= _copy_gbuffer_ps != null;
            _is_pass_executable &= _copy_gbuffer_mat != null;
            if (!_is_pass_executable)
            {
                ReleaseAllHistoryBuffers();
                return;
            }
            
            // --------------------------------------
            // Previous GBuffers.
            var diffuse_descriptor = cameraTextureDescriptor;
            diffuse_descriptor.graphicsFormat = QualitySettings.activeColorSpace == ColorSpace.Linear ? GraphicsFormat.R8G8B8A8_SRGB : GraphicsFormat.R8G8B8A8_UNorm;
            diffuse_descriptor.depthBufferBits = 0;
            RenderingUtils.ReAllocateIfNeeded(ref _prev_gBuffer0, diffuse_descriptor);

            var specular_descriptor = cameraTextureDescriptor;
            specular_descriptor.graphicsFormat = GraphicsFormat.R8G8B8A8_UNorm;
            specular_descriptor.depthBufferBits = 0;
            RenderingUtils.ReAllocateIfNeeded(ref _prev_gBuffer1, specular_descriptor);
            
            var renderer_data = GetRendererDataWithinFeature();
            var accurate_normal = false;
            if (renderer_data == null) Debug.LogWarning($"{nameof(HistoryGBufferPass)}: 获取Universal Renderer Data失败，法线回退到一般精度");
            else accurate_normal = renderer_data.accurateGbufferNormals;
            
            var normal_descriptor = cameraTextureDescriptor;
            // Reference: DeferredLights.cs - GetGBufferFormat
            normal_descriptor.graphicsFormat = accurate_normal ? GraphicsFormat.R8G8B8A8_UNorm : DepthNormalOnlyPass.GetGraphicsFormat();
            normal_descriptor.depthBufferBits = 0;
            RenderingUtils.ReAllocateIfNeeded(ref _prev_gBuffer2, normal_descriptor);

            var depth_descriptor = cameraTextureDescriptor;
            depth_descriptor.depthBufferBits = 32;
            depth_descriptor.graphicsFormat = GraphicsFormat.R32_SFloat;
            RenderingUtils.ReAllocateIfNeeded(ref _prev_depth, depth_descriptor);
            
            // Blit with MRT
            ConfigureTarget(new[] { _prev_gBuffer0, _prev_gBuffer1, _prev_gBuffer2 });
        }

        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            if (!_is_pass_executable) return;
            
            var cmd = CommandBufferPool.Get("RTXDI Pass"); 
            
            using (new ProfilingScope(cmd, new ProfilingSampler("RTXDI: Previous GBuffer Copy")))
            {
                var block = new MaterialPropertyBlock();
                block.SetVector(GpuParams._BlitScaleBias, new Vector4(1.0f, 1.0f, 0.0f, 0.0f));
                cmd.DrawProcedural(Matrix4x4.identity, _copy_gbuffer_mat, 0, MeshTopology.Triangles, 3, 1, block);
                cmd.CopyTexture(renderingData.cameraData.renderer.cameraDepthTargetHandle, _prev_depth);
            }
            
            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }

        public void Setup(bool executable) => _is_pass_executable = executable;

        public void Release()
        {
            CoreUtils.Destroy(_copy_gbuffer_mat);
            ReleaseAllHistoryBuffers();
        }

        private void ReleaseAllHistoryBuffers()
        {
            _prev_gBuffer0?.Release(); _prev_gBuffer0 = null;
            _prev_gBuffer1?.Release(); _prev_gBuffer1 = null;
            _prev_gBuffer2?.Release(); _prev_gBuffer2 = null;
            _prev_depth?.Release(); _prev_depth = null;
        }
    }

    private sealed class LightingCompositePass : ScriptableRenderPass
    {
        private Shader _lighting_composite_ps;
        private Material _lighting_composite_mat;

        private RTHandle _scene_color_copy;
        private RTHandle _input_rtxdi_lighting;

        public LightingCompositePass()
        {
            _lighting_composite_ps = Resources.Load<Shader>("Shaders/LightingComposite");
            _lighting_composite_mat = CoreUtils.CreateEngineMaterial(_lighting_composite_ps);
        }
        
        private static class GpuParams
        {
            public static readonly int _BlitScaleBias = Shader.PropertyToID("_BlitScaleBias");
            public static readonly int _SceneColorSource = Shader.PropertyToID("_SceneColorSource");
            public static readonly int _LightingTexture = Shader.PropertyToID("_LightingTexture");
        }

        public override void Configure(CommandBuffer cmd, RenderTextureDescriptor cameraTextureDescriptor)
        {
            var scene_color_copy_desc = cameraTextureDescriptor;
            scene_color_copy_desc.depthBufferBits = 0;
            RenderingUtils.ReAllocateIfNeeded(ref _scene_color_copy, scene_color_copy_desc);
        }

        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            if (_input_rtxdi_lighting == null) return;
            
            var cmd = CommandBufferPool.Get("RTXDI Lighting Composite");
            var renderer = renderingData.cameraData.renderer;
            {
                cmd.CopyTexture(renderer.cameraColorTargetHandle, _scene_color_copy);
                
                cmd.SetGlobalTexture(GpuParams._SceneColorSource, _scene_color_copy);
                cmd.SetGlobalTexture(GpuParams._LightingTexture, _input_rtxdi_lighting);
                cmd.SetRenderTarget(renderer.cameraColorTargetHandle, renderer.cameraDepthTargetHandle);
                
                var block = new MaterialPropertyBlock();
                block.SetVector(GpuParams._BlitScaleBias, new Vector4(1.0f, 1.0f, 0.0f, 0.0f));
                cmd.DrawProcedural(Matrix4x4.identity, _lighting_composite_mat, 0, MeshTopology.Triangles, 3, 1, block);
            }
            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }

        public void Setup(RTHandle rtxdiLighting) => _input_rtxdi_lighting = rtxdiLighting;

        public void Release()
        {
            CoreUtils.Destroy(_lighting_composite_mat);
            _scene_color_copy?.Release(); _scene_color_copy = null;
        }
    }

    public bool DebugLightData = false;
    
    private RTXDIPass _rtxdi_pass;
    private HistoryGBufferPass _history_gbuffer_pass;
    private LightingCompositePass _lighting_composite_pass;

    public override void Create()
    {
        if (!isActive)
        {
            SafeReleaseFeatureResources();
            return;
        }

        _rtxdi_pass ??= new RTXDIPass();
        _history_gbuffer_pass ??= new HistoryGBufferPass();
        _lighting_composite_pass ??= new LightingCompositePass();
    }

    protected override void Dispose(bool disposing)
    {
        SafeReleaseFeatureResources();
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        if (renderingData.cameraData.isPreviewCamera) return;
        
        _rtxdi_pass.renderPassEvent = RenderPassEvent.AfterRenderingGbuffer + 1;
        _rtxdi_pass.Setup(DebugLightData, _history_gbuffer_pass.GetPrevGBuffer0(), _history_gbuffer_pass.GetPrevGBuffer1(),
            _history_gbuffer_pass.GetPrevGBuffer2(), _history_gbuffer_pass.GetPrevDepth());
        renderer.EnqueuePass(_rtxdi_pass);

        _history_gbuffer_pass.renderPassEvent = _rtxdi_pass.renderPassEvent + 1;
        _history_gbuffer_pass.Setup(_rtxdi_pass.Executable());
        renderer.EnqueuePass(_history_gbuffer_pass);

        _lighting_composite_pass.renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;
        _lighting_composite_pass.Setup(_rtxdi_pass.GetShadingOutput());
        renderer.EnqueuePass(_lighting_composite_pass);
    }

    private void SafeReleaseFeatureResources()
    {
        _rtxdi_pass?.Release(); _rtxdi_pass = null;
        _history_gbuffer_pass?.Release(); _history_gbuffer_pass = null;
        _lighting_composite_pass?.Release(); _lighting_composite_pass = null;
    }

    [CanBeNull]
    private static UniversalRendererData GetRendererDataWithinFeature()
    {
        // 获取当前使用的rp asset
        var rp_asset = GraphicsSettings.renderPipelineAsset;
        // 有时Graphics面板里可能没有设置rp asset，此时尝试从Quality面板内获取
        if (rp_asset == null) rp_asset = QualitySettings.renderPipeline;

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
