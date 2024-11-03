#ifndef RTXDI_MINIMAL_TYPES
#define RTXDI_MINIMAL_TYPES

#define uint32_t uint
#define RTXDI_TEX2D Texture2D
#define RTXDI_TEX2D_LOAD(t,pos,lod) t.Load(int3(pos,lod))
#define RTXDI_DEFAULT(value) = value

#endif