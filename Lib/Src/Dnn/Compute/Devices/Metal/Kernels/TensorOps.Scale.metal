// output = input * scalar, multiplied in 32 bits and rounded once. Input and output may be the same memory.

#include <metal_stdlib>

using namespace metal;

template<typename TElement>
kernel void scale(
    device const TElement* input [[buffer(0)]],
    device TElement* output [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    uint index [[thread_position_in_grid]] )
{
    output[index] = static_cast<TElement>( float( input[index] ) * scalar );
}

template [[host_name("scale_float")]] kernel void scale<float>( device const float*, device float*, constant float&, uint );
template [[host_name("scale_half")]] kernel void scale<half>( device const half*, device half*, constant float&, uint );
template [[host_name("scale_bfloat")]] kernel void scale<bfloat>( device const bfloat*, device bfloat*, constant float&, uint );
