// output = silu( gate ) * up, where each input row is its gate half followed by its up half. All arithmetic in 32 bits.

#include <metal_stdlib>

using namespace metal;

struct SwigluShape
{
    uint half_width;
};

template<typename TElement>
kernel void swiglu(
    device const TElement* input [[buffer(0)]],
    device TElement* output [[buffer(1)]],
    constant SwigluShape& shape [[buffer(2)]],
    uint index [[thread_position_in_grid]] )
{
    const uint row = index / shape.half_width;
    const uint column = index % shape.half_width;
    const uint gate_index = row * 2 * shape.half_width + column;

    const float gate = float( input[gate_index] );
    const float up = float( input[gate_index + shape.half_width] );

    output[index] = TElement( gate / ( 1.0f + exp( -gate ) ) * up );
}

template [[host_name("swiglu_half")]] kernel void swiglu<half>(
    device const half*, device half*, constant SwigluShape&, uint );
template [[host_name("swiglu_bfloat")]] kernel void swiglu<bfloat>(
    device const bfloat*, device bfloat*, constant SwigluShape&, uint );
