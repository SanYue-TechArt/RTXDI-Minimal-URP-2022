#ifndef TOON_LIGHTING

#define TOON_LIGHTING

//#define MAIN_LIGHT_SHADOW 1

void MainLight_float(float3 worldPos, out float3 direction, out float3 color, out float shadow)
{
    #ifdef SHADERGRAPH_PREVIEW
    direction = normalize(float3(-0.5,0.5,-0.5));
    color = float3(1,1,1);
    shadow = 1;
    #else
    
    Light mainLight = GetMainLight(TransformWorldToShadowCoord(worldPos), worldPos, unity_ProbesOcclusion);
    shadow = mainLight.shadowAttenuation;
    direction = mainLight.direction;
    color = mainLight.color;
    
    #endif
}

void MainLight_half(half3 worldPos, out half3 direction, out half3 color, out half shadow)
{
    #ifdef SHADERGRAPH_PREVIEW
    direction = normalize(half3(-0.5,0.5,0.5));
    color = half3(1,1,1);
    shadow = 1;

    #else
    
    Light mainLight = GetMainLight(TransformWorldToShadowCoord(worldPos), worldPos, unity_ProbesOcclusion);
    shadow = mainLight.shadowAttenuation;
    direction = mainLight.direction;
    color = mainLight.color;
    
    #endif
}

void MainLightNoShadow_float(out float3 direction, float3 color)
{
    #ifdef SHADERGRAPH_PREVIEW
    direction = normalize(float3(-0.5,0.5,-0.5));
    color = float3(1,1,1);
    #else

    Light mainLight = GetMainLight();
    direction = mainLight.direction;
    color = mainLight.color;
    
    #endif
}

#endif