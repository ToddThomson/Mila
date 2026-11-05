// output = a + b, added in 32 bits and rounded once.

#include <metal_stdlib>

using namespace metal;

template<typename TElement>
kernel void residual(
    device const TElement* a [[buffer(0)]],
    device const TElement* b [[buffer(1)]],
    device TElement* output [[buffer(2)]],
    uint index [[thread_position_in_grid]] )
{
    output[index] = TElement( float( a[index] ) + float( b[index] ) );
}

template [[host_name("residual_half")]] kernel void residual<half>(
    device const half*, device const half*, device half*, uint );
template [[host_name("residual_bfloat")]] kernel void residual<bfloat>(
    device const bfloat*, device const bfloat*, device bfloat*, uint );
