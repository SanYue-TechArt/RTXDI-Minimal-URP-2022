#ifndef RTXDI_MINIMAL_HELPERS
#define RTXDI_MINIMAL_HELPERS

#include "RtxdiMath.hlsl"

bool RTXDI_IsActiveCheckerboardPixel(
    uint2 pixelPosition,
    bool previousFrame,
    uint activeCheckerboardField)
{
    if (activeCheckerboardField == 0)
        return true;

    return ((pixelPosition.x + pixelPosition.y + int(previousFrame)) & 1) == (activeCheckerboardField & 1);
}

void RTXDI_ActivateCheckerboardPixel(inout uint2 pixelPosition, bool previousFrame, uint activeCheckerboardField)
{
    if (RTXDI_IsActiveCheckerboardPixel(pixelPosition, previousFrame, activeCheckerboardField))
        return;
    
    if (previousFrame)
        pixelPosition.x += int(activeCheckerboardField) * 2 - 3;
    else
        pixelPosition.x += (pixelPosition.y & 1) != 0 ? 1 : -1;
}

void RTXDI_ActivateCheckerboardPixel(inout int2 pixelPosition, bool previousFrame, uint activeCheckerboardField)
{
    uint2 uPixelPosition = uint2(pixelPosition);
    RTXDI_ActivateCheckerboardPixel(uPixelPosition, previousFrame, activeCheckerboardField);
    pixelPosition = int2(uPixelPosition);
}

uint2 RTXDI_PixelPosToReservoirPos(uint2 pixelPosition, uint activeCheckerboardField)
{
    if (activeCheckerboardField == 0)
        return pixelPosition;

    return uint2(pixelPosition.x >> 1, pixelPosition.y);
}

uint2 RTXDI_ReservoirPosToPixelPos(uint2 reservoirIndex, uint activeCheckerboardField)
{
    if (activeCheckerboardField == 0)
        return reservoirIndex;

    uint2 pixelPosition = uint2(reservoirIndex.x << 1, reservoirIndex.y);
    pixelPosition.x += ((pixelPosition.y + activeCheckerboardField) & 1);
    return pixelPosition;
}

// Internal SDK function that permutes the pixels sampled from the previous frame.
void RTXDI_ApplyPermutationSampling(inout int2 prevPixelPos, uint uniformRandomNumber)
{
    int2 offset = int2(uniformRandomNumber & 3, (uniformRandomNumber >> 2) & 3);
    prevPixelPos += offset;
 
    prevPixelPos.x ^= 3;
    prevPixelPos.y ^= 3;
    
    prevPixelPos -= offset;
}

uint RTXDI_ReservoirPositionToPointer(
    RTXDI_ReservoirBufferParameters reservoirParams,
    uint2 reservoirPosition,
    uint reservoirArrayIndex)
{
    uint2 blockIdx = reservoirPosition / RTXDI_RESERVOIR_BLOCK_SIZE;
    uint2 positionInBlock = reservoirPosition % RTXDI_RESERVOIR_BLOCK_SIZE;

    return reservoirArrayIndex * reservoirParams.reservoirArrayPitch
        + blockIdx.y * reservoirParams.reservoirBlockRowPitch
        + blockIdx.x * (RTXDI_RESERVOIR_BLOCK_SIZE * RTXDI_RESERVOIR_BLOCK_SIZE)
        + positionInBlock.y * RTXDI_RESERVOIR_BLOCK_SIZE
        + positionInBlock.x;
}

#endif