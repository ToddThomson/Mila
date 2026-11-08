// output = input * inverse_deviation * ( weight + unit_offset ) + bias, with inverse_deviation = 1 / sqrt( mean( input^2 ) + epsilon ) over each slice.
// One threadgroup of kRmsNormThreads per slice: one SIMD-group alone waits on memory through every step of a 3840-wide row, and a
// decode step runs hundreds of these. The sum of squares is 32 bits: 4096 values of 10 already sum past FP16's largest.

#include <metal_stdlib>

using namespace metal;

// MetalRmsNormOp dispatches this many threads to a threadgroup.
constant constexpr uint kRmsNormThreads = 256;
constant constexpr uint kRmsNormSimdgroups = 8;

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
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]] )
{
    threadgroup float shared[kRmsNormSimdgroups];

    const uint slice = group.x;
    const uint first = slice * shape.elements;
    const uint first_weight = ( slice % shape.groups ) * shape.elements;

    float sum = 0.0f;

    for ( uint k = thread_index; k < shape.elements; k += kRmsNormThreads )
    {
        const float value = float( input[first + k] );

        sum += value * value;
    }

    sum = simd_sum( sum );

    if ( lane == 0 )
    {
        shared[simd] = sum;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    const float total = simd_sum( lane < kRmsNormSimdgroups ? shared[lane] : 0.0f );
    const float inverse_deviation = 1.0f / sqrt( total / float( shape.elements ) + shape.epsilon );

    for ( uint k = thread_index; k < shape.elements; k += kRmsNormThreads )
    {
        const float scaled = float( input[first + k] ) * inverse_deviation * ( float( weight[first_weight + k] ) + shape.unit_offset );

        output[first + k] = TElement( shape.has_bias != 0 ? scaled + float( bias[first_weight + k] ) : scaled );
    }
}

template [[host_name("rms_norm_half")]] kernel void rms_norm<half>(
    device const half*, device const half*, device const half*, device half*, constant RmsNormShape&, uint3, uint, uint, uint );
template [[host_name("rms_norm_bfloat")]] kernel void rms_norm<bfloat>(
    device const bfloat*, device const bfloat*, device const bfloat*, device bfloat*, constant RmsNormShape&, uint3, uint, uint, uint );
