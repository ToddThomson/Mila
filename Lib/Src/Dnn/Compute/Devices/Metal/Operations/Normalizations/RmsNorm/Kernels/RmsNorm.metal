// output = input * inverse_deviation * ( weight + unit_offset ) + bias, with inverse_deviation = 1 / sqrt( mean( input^2 ) + epsilon ) over each slice.
// One threadgroup per slice, of TThreads threads, each reading four values a load: a load waits on memory, so a slice's time is
// the loads one thread makes one after another, and a decode step runs hundreds of these. MetalRmsNormOp gives a slice wider than
// 1024 the threadgroup of 1024. The sum of squares is 32 bits: 4096 values of 10 already sum past FP16's largest.

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
// A slice starts wherever its width puts it, so the four-value loads are packed ones, aligned as one value is.
template<typename TElement, uint TThreads>
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
    constexpr uint kSimdgroupWidth = 32;
    constexpr uint kSimdgroups = TThreads / kSimdgroupWidth;

    threadgroup float shared[kSimdgroups];

    typedef packed_vec<TElement, 4> Packed;

    const uint elements = shape.elements;
    const uint vectors = elements / 4;
    const uint first = group.x * elements;
    const uint first_weight = ( group.x % shape.groups ) * elements;

    device const Packed* x = reinterpret_cast<device const Packed*>( input + first );
    device const Packed* w = reinterpret_cast<device const Packed*>( weight + first_weight );
    device const Packed* b = reinterpret_cast<device const Packed*>( bias + first_weight );
    device Packed* y = reinterpret_cast<device Packed*>( output + first );

    // A thread's first four values stay in registers for the second pass, which is every value of a slice up to 4 * TThreads wide.
    float4 held = float4( 0.0f );
    float4 squares = float4( 0.0f );

    if ( thread_index < vectors )
    {
        held = float4( x[thread_index] );
        squares = held * held;
    }

    for ( uint k = thread_index + TThreads; k < vectors; k += TThreads )
    {
        const float4 value = float4( x[k] );

        squares += value * value;
    }

    float sum = squares.x + squares.y + squares.z + squares.w;

    // The values past the last whole four, one to a thread.
    const uint tail = vectors * 4 + thread_index;
    float tail_value = 0.0f;

    if ( tail < elements )
    {
        tail_value = float( input[first + tail] );
        sum += tail_value * tail_value;
    }

    sum = simd_sum( sum );

    if ( lane == 0 )
    {
        shared[simd] = sum;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    float total = 0.0f;

    for ( uint s = lane; s < kSimdgroups; s += kSimdgroupWidth )
    {
        total += shared[s];
    }

    total = simd_sum( total );

    const float inverse_deviation = 1.0f / sqrt( total / float( elements ) + shape.epsilon );
    const bool has_bias = shape.has_bias != 0;

    if ( thread_index < vectors )
    {
        float4 scaled = held * inverse_deviation * ( float4( w[thread_index] ) + shape.unit_offset );

        if ( has_bias )
        {
            scaled += float4( b[thread_index] );
        }

        y[thread_index] = Packed( vec<TElement, 4>( scaled ) );
    }

    for ( uint k = thread_index + TThreads; k < vectors; k += TThreads )
    {
        float4 scaled = float4( x[k] ) * inverse_deviation * ( float4( w[k] ) + shape.unit_offset );

        if ( has_bias )
        {
            scaled += float4( b[k] );
        }

        y[k] = Packed( vec<TElement, 4>( scaled ) );
    }

    if ( tail < elements )
    {
        const float scaled = tail_value * inverse_deviation * ( float( weight[first_weight + tail] ) + shape.unit_offset );

        output[first + tail] = TElement( has_bias ? scaled + float( bias[first_weight + tail] ) : scaled );
    }
}

#define MILA_RMS_NORM( element, threads ) \
    template [[host_name( "rms_norm_" #element "_" #threads )]] kernel void rms_norm<element, threads>( \
        device const element*, device const element*, device const element*, device element*, constant RmsNormShape&, uint3, uint, uint, uint );

MILA_RMS_NORM( half, 256 )
MILA_RMS_NORM( half, 1024 )
MILA_RMS_NORM( bfloat, 256 )
MILA_RMS_NORM( bfloat, 1024 )
