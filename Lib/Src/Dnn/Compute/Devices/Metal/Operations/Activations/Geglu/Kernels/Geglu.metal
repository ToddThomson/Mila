// output = gelu( gate ) * up, GELU in its tanh form, where each input row is its gate half followed by its up half. All
// arithmetic in 32 bits, narrowed at the store, as CUDA's GeGLU does.

#include <metal_stdlib>

using namespace metal;

struct GegluShape
{
    uint half_width;
};

template<typename TElement>
kernel void geglu(
    device const TElement* input [[buffer(0)]],
    device TElement* output [[buffer(1)]],
    constant GegluShape& shape [[buffer(2)]],
    uint index [[thread_position_in_grid]] )
{
    const uint row = index / shape.half_width;
    const uint column = index % shape.half_width;
    const uint gate_index = row * 2 * shape.half_width + column;

    const float gate = float( input[gate_index] );
    const float up = float( input[gate_index + shape.half_width] );

    // sqrt( 2 / pi ) and 0.044715, CUDA's GeluTanh. Metal's fast tanh is its own approximation, so the precise one is called.
    const float inner = 0.7978845608f * ( gate + 0.044715f * gate * gate * gate );

    output[index] = TElement( 0.5f * gate * ( 1.0f + precise::tanh( inner ) ) * up );
}

template [[host_name("geglu_half")]] kernel void geglu<half>(
    device const half*, device half*, constant GegluShape&, uint );
template [[host_name("geglu_bfloat")]] kernel void geglu<bfloat>(
    device const bfloat*, device bfloat*, constant GegluShape&, uint );
