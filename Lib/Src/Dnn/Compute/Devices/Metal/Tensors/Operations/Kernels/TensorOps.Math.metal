// Elementwise add, subtract, multiply and divide of two tensors, each in 32 bits and rounded once, and the sum of a tensor
// in 32 bits. The result may be the memory of either input.

#include <metal_stdlib>

using namespace metal;

struct AddValues
{
    static float apply( float a, float b ) { return a + b; }
};

struct SubtractValues
{
    static float apply( float a, float b ) { return a - b; }
};

struct MultiplyValues
{
    static float apply( float a, float b ) { return a * b; }
};

struct DivideValues
{
    static float apply( float a, float b ) { return a / b; }
};

template<typename TOperation, typename TElement>
kernel void elementwise_math(
    device const TElement* a [[buffer(0)]],
    device const TElement* b [[buffer(1)]],
    device TElement* result [[buffer(2)]],
    uint index [[thread_position_in_grid]] )
{
    result[index] = TElement( TOperation::apply( float( a[index] ), float( b[index] ) ) );
}

#define MILA_ELEMENTWISE_MATH( operation, name ) \
    template [[host_name( name "_float" )]] kernel void elementwise_math<operation, float>( \
        device const float*, device const float*, device float*, uint ); \
    template [[host_name( name "_half" )]] kernel void elementwise_math<operation, half>( \
        device const half*, device const half*, device half*, uint ); \
    template [[host_name( name "_bfloat" )]] kernel void elementwise_math<operation, bfloat>( \
        device const bfloat*, device const bfloat*, device bfloat*, uint );

MILA_ELEMENTWISE_MATH( AddValues, "add" )
MILA_ELEMENTWISE_MATH( SubtractValues, "subtract" )
MILA_ELEMENTWISE_MATH( MultiplyValues, "multiply" )
MILA_ELEMENTWISE_MATH( DivideValues, "divide" )

// One threadgroup sums every element: each thread a strided share, then the SIMD-groups' sums through threadgroup memory.
template<typename TElement>
kernel void sum_elements(
    device const TElement* input [[buffer(0)]],
    device float* total [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]] )
{
    threadgroup float partial[32];

    float share = 0.0f;

    for ( uint i = thread_index; i < count; i += threads )
    {
        share += float( input[i] );
    }

    share = simd_sum( share );

    if ( lane == 0 )
    {
        partial[simdgroup] = share;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    if ( simdgroup == 0 )
    {
        const float value = simd_sum( lane < simdgroups ? partial[lane] : 0.0f );

        if ( lane == 0 )
        {
            total[0] = value;
        }
    }
}

template [[host_name( "sum_float" )]] kernel void sum_elements<float>(
    device const float*, device float*, constant uint&, uint, uint, uint, uint, uint );
template [[host_name( "sum_half" )]] kernel void sum_elements<half>(
    device const half*, device float*, constant uint&, uint, uint, uint, uint, uint );
template [[host_name( "sum_bfloat" )]] kernel void sum_elements<bfloat>(
    device const bfloat*, device float*, constant uint&, uint, uint, uint, uint, uint );
