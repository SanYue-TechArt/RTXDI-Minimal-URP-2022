Shader "RTXDI/CopyGBuffer"
{
    Properties
    {

    }
    SubShader
    {
        Tags 
        { 
            "RenderPipeline" = "UniversalPipeline"
        }

        Pass
        {
            Name "Copy GBuffer"
            
            Blend   One Zero
            ZWrite  Off
            ZTest   Always
            
            HLSLPROGRAM

            #pragma vertex Vert
            #pragma fragment Frag_GBufferCopy

            #include "Packages/com.unity.render-pipelines.universal@14.0.9/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

            TEXTURE2D_X(_GBuffer0);
            TEXTURE2D_X(_GBuffer1);
            TEXTURE2D_X(_GBuffer2);

            struct GBufferCopy
            {
                float4 gbuffer0 : SV_Target0; // Albedo
                float4 gbuffer1 : SV_Target1; // Specular and Metallic
                float4 gbuffer2 : SV_Target2; // Normal and Smoothness
            };

            void Frag_GBufferCopy(Varyings pin, out GBufferCopy gBufferCopy)
            {
                gBufferCopy.gbuffer0 = SAMPLE_TEXTURE2D_X_LOD(_GBuffer0, sampler_PointClamp, pin.texcoord, 0);
                gBufferCopy.gbuffer1 = SAMPLE_TEXTURE2D_X_LOD(_GBuffer1, sampler_PointClamp, pin.texcoord, 0);
                gBufferCopy.gbuffer2 = SAMPLE_TEXTURE2D_X_LOD(_GBuffer2, sampler_PointClamp, pin.texcoord, 0);
            }
            
            ENDHLSL
        }
    }
}
