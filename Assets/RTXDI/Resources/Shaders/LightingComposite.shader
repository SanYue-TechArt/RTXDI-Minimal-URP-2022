Shader "RTXDI/LightingComposite"
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
            Name "RTXDI Lighting Composite"
            
            Blend   One Zero
            ZWrite  Off
            ZTest   Always
            
            HLSLPROGRAM

            #pragma vertex Vert
            #pragma fragment Frag_LightingComposite

            #include "Packages/com.unity.render-pipelines.universal@14.0.9/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

            TEXTURE2D_X(_SceneColorSource);
            TEXTURE2D_X(_LightingTexture);

            float4 Frag_LightingComposite(Varyings pin) : SV_Target
            {
                const float3 sceneColor = SAMPLE_TEXTURE2D_X_LOD(_SceneColorSource, sampler_PointClamp, pin.texcoord, 0).rgb;
                const float3 lighting = SAMPLE_TEXTURE2D_X_LOD(_LightingTexture, sampler_PointClamp, pin.texcoord, 0).rgb;
                
                return float4(sceneColor * 0.1f + lighting * 1.5f, 1.0f);
            }
            
            ENDHLSL
        }
    }
}
