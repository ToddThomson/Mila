// The depthwise causal convolution over the sequence axis, as CUDA's (CausalConv1d.cuh): tensors [B, T, C], the filter
// [C, K], and the R = (K - 1) * D rows left of a chunk from the state [B, R, C], or zero where the sequence starts here. Sums
// in 32 bits, rounded once.

#include <metal_stdlib>

using namespace metal;

struct CausalConvolutionShape
{
    uint batch;
    uint steps;
    uint channels;
    uint kernel_width;
    uint dilation;
    uint has_state;
    uint has_bias;
};

// One thread an output element.
template<typename TElement>
kernel void causal_convolution(
    device const TElement* input [[buffer(0)]],
    device const TElement* state [[buffer(1)]],
    device const TElement* weight [[buffer(2)]],
    device const TElement* bias [[buffer(3)]],
    device TElement* output [[buffer(4)]],
    constant CausalConvolutionShape& shape [[buffer(5)]],
    uint index [[thread_position_in_grid]] )
{
    const uint c = index % shape.channels;
    const uint t = ( index / shape.channels ) % shape.steps;
    const uint b = index / ( shape.channels * shape.steps );
    const int state_rows = int( ( shape.kernel_width - 1 ) * shape.dilation );

    float sum = shape.has_bias != 0 ? float( bias[c] ) : 0.0f;

    for ( uint i = 0; i < shape.kernel_width; ++i )
    {
        const int source = int( t ) - state_rows + int( i * shape.dilation );
        float value = 0.0f;

        if ( source >= 0 )
        {
            value = float( input[( b * shape.steps + uint( source ) ) * shape.channels + c] );
        }
        else if ( shape.has_state != 0 )
        {
            value = float( state[( b * uint( state_rows ) + uint( state_rows + source ) ) * shape.channels + c] );
        }

        sum += float( weight[c * shape.kernel_width + i] ) * value;
    }

    output[index] = TElement( sum );
}

// The state becomes the last R rows of [state ; input]. One thread a (batch, channel) reads every row it writes before it
// writes any, which makes shifting the state in place safe.
template<typename TElement>
kernel void causal_convolution_update_state(
    device TElement* state [[buffer(0)]],
    device const TElement* input [[buffer(1)]],
    constant CausalConvolutionShape& shape [[buffer(2)]],
    uint index [[thread_position_in_grid]] )
{
    constexpr uint kMostStateRows = 16;

    const uint c = index % shape.channels;
    const uint b = index / shape.channels;
    const int state_rows = int( ( shape.kernel_width - 1 ) * shape.dilation );

    TElement staged[kMostStateRows];

    for ( int j = 0; j < state_rows; ++j )
    {
        const int source = int( shape.steps ) - state_rows + j;

        staged[j] = source >= 0
            ? input[( b * shape.steps + uint( source ) ) * shape.channels + c]
            : state[( b * uint( state_rows ) + uint( state_rows + source ) ) * shape.channels + c];
    }

    for ( int j = 0; j < state_rows; ++j )
    {
        state[( b * uint( state_rows ) + uint( j ) ) * shape.channels + c] = staged[j];
    }
}

#define MILA_CAUSAL_CONVOLUTION( element ) \
    template [[host_name( "causal_convolution_" #element )]] kernel void causal_convolution<element>( \
        device const element*, device const element*, device const element*, device const element*, device element*, \
        constant CausalConvolutionShape&, uint ); \
    template [[host_name( "causal_convolution_update_state_" #element )]] kernel void causal_convolution_update_state<element>( \
        device element*, device const element*, constant CausalConvolutionShape&, uint );

MILA_CAUSAL_CONVOLUTION( half )
MILA_CAUSAL_CONVOLUTION( bfloat )
