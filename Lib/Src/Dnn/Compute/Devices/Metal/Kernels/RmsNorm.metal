// output = input * inverse_deviation * ( weight + unit_offset ) + bias, with inverse_deviation = 1 / sqrt( mean( input^2 ) + epsilon ) over each slice.
// One SIMD-group per slice, whose lanes walk it side by side. The sum of squares is 32 bits: 4096 values of 10 already sum past FP16's largest.

#include <metal_stdlib>

using namespace metal;

struct RmsNormShape
{
    uint elements;
    uint groups;
    uint has_bias;
    float epsilon;
    float unit_offset;
};

// A slice is one row or one group of it, and the weights repeat per group. `bias` stands in for nothing when has_bias is 0, and is not read.
template<typename TElement>
kernel void rms_norm(
    device const TElement* input [[buffer(0)]],
    device const TElement* weight [[buffer(1)]],
    device const TElement* bias [[buffer(2)]],
    device TElement* output [[buffer(3)]],
    constant RmsNormShape& shape [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]] )
{
    const uint slice = group.x;
    const uint first = slice * shape.elements;
    const uint first_weight = ( slice % shape.groups ) * shape.elements;

    float sum = 0.0f;

    for ( uint k = lane; k < shape.elements; k += threads.x )
    {
        const float value = float( input[first + k] );

        sum += value * value;
    }

    const float inverse_deviation = 1.0f / sqrt( simd_sum( sum ) / float( shape.elements ) + shape.epsilon );

    for ( uint k = lane; k < shape.elements; k += threads.x )
    {
        const float scaled = float( input[first + k] ) * inverse_deviation * ( float( weight[first_weight + k] ) + shape.unit_offset );

        output[first + k] = TElement( shape.has_bias != 0 ? scaled + float( bias[first_weight + k] ) : scaled );
    }
}

template [[host_name("rms_norm_half")]] kernel void rms_norm<half>(
    device const half*, device const half*, device const half*, device half*, constant RmsNormShape&, uint3, uint, uint3 );
template [[host_name("rms_norm_bfloat")]] kernel void rms_norm<bfloat>(
    device const bfloat*, device const bfloat*, device const bfloat*, device bfloat*, constant RmsNormShape&, uint3, uint, uint3 );
