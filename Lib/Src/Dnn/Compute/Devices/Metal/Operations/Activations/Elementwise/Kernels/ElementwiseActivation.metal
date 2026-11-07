// output = f( input ), elementwise, for each functor of Components/Activations/Activation/Kernels/ElementwiseActivation.h,
// which these compute in the same 32-bit arithmetic. The precise exp, tanh and log, since CUDA's are; input and output may
// be the same memory.

#include <metal_stdlib>

using namespace metal;

struct IdentityActivation
{
    static float apply( float x, float ) { return x; }
};

struct GeluTanhActivation
{
    static float apply( float x, float )
    {
        return 0.5f * x * ( 1.0f + precise::tanh( 0.7978845608f * ( x + 0.044715f * x * x * x ) ) );
    }
};

struct SiluActivation
{
    static float apply( float x, float ) { return x * ( 1.0f / ( 1.0f + precise::exp( -x ) ) ); }
};

struct ReluActivation
{
    static float apply( float x, float ) { return x > 0.0f ? x : 0.0f; }
};

struct TanhActivation
{
    static float apply( float x, float ) { return precise::tanh( x ); }
};

struct SigmoidActivation
{
    static float apply( float x, float ) { return 1.0f / ( 1.0f + precise::exp( -x ) ); }
};

struct LeakyReluActivation
{
    static float apply( float x, float alpha ) { return x > 0.0f ? x : alpha * x; }
};

struct MishActivation
{
    static float apply( float x, float )
    {
        // softplus, with log(1 + e^x) taken as x past 20, as the functor does.
        const float softplus = x > 20.0f ? x : precise::log( 1.0f + precise::exp( x ) );

        return x * precise::tanh( softplus );
    }
};

template<typename TActivation, typename TElement>
kernel void elementwise_activation(
    device const TElement* input [[buffer(0)]],
    device TElement* output [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    uint index [[thread_position_in_grid]] )
{
    output[index] = TElement( TActivation::apply( float( input[index] ), alpha ) );
}

#define MILA_ACTIVATION( activation, name ) \
    template [[host_name( "activation_" name "_half" )]] kernel void elementwise_activation<activation, half>( \
        device const half*, device half*, constant float&, uint ); \
    template [[host_name( "activation_" name "_bfloat" )]] kernel void elementwise_activation<activation, bfloat>( \
        device const bfloat*, device bfloat*, constant float&, uint );

MILA_ACTIVATION( IdentityActivation, "identity" )
MILA_ACTIVATION( GeluTanhActivation, "gelu_tanh" )
MILA_ACTIVATION( SiluActivation, "silu" )
MILA_ACTIVATION( ReluActivation, "relu" )
MILA_ACTIVATION( TanhActivation, "tanh" )
MILA_ACTIVATION( SigmoidActivation, "sigmoid" )
MILA_ACTIVATION( LeakyReluActivation, "leaky_relu" )
MILA_ACTIVATION( MishActivation, "mish" )
