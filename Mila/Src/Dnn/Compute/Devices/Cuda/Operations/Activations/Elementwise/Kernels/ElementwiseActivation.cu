/**
 * @file ElementwiseActivation.cu
 * @brief Functor-templated CUDA kernels for the elementwise activation family.
 *
 * One grid-stride kernel pair (forward/backward) serves every elementwise function;
 * nvcc inlines the functor so each thread runs identical, branch-free code. The
 * shared functor library (ElementwiseActivation.h) is the single math source for
 * both this CUDA op and the CPU op. Elementwise activation is memory-bound; all
 * arithmetic is performed in float, with BF16 promoted on load and demoted on store.
 *
 * Explicit instantiations at the bottom register one (native, functor) symbol pair
 * per function for FP32 and BF16; the host-compiled op module unit calls the
 * launchers by name and the linker resolves them here.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "ElementwiseActivation.cuh"
#include "../../../../../../../Components/Activations/Activation/Kernels/ElementwiseActivation.h"

namespace Mila::Dnn::Compute::Cuda::Activation
{
    using namespace Mila::Dnn::Activations;

    // Native <-> float conversion helpers (FP32 promotion for BF16 arithmetic).
    __device__ __forceinline__ float to_float( float x ) { return x; }
    __device__ __forceinline__ float to_float( __nv_bfloat16 x ) { return __bfloat162float( x ); }

    __device__ __forceinline__ void from_float( float& dst, float v ) { dst = v; }
    __device__ __forceinline__ void from_float( __nv_bfloat16& dst, float v ) { dst = __float2bfloat16( v ); }

    template<typename TNative, typename TFunctor>
    __global__ void elementwise_forward_kernel( TNative* Y, const TNative* X, int N, TFunctor functor )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if ( i < N )
        {
            float x = to_float( X[ i ] );
            TNative y;
            from_float( y, functor.fwd( x ) );
            Y[ i ] = y;
        }
    }

    template<typename TNative, typename TFunctor>
    __global__ void elementwise_backward_kernel( TNative* dX, const TNative* X, const TNative* dY, int N, TFunctor functor )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if ( i < N )
        {
            float x = to_float( X[ i ] );
            float dy = to_float( dY[ i ] );
            TNative d;
            from_float( d, functor.df( x ) * dy );
            dX[ i ] = d;
        }
    }

    template<typename TNative, typename TFunctor>
    void launch_elementwise_forward( TNative* Y, const TNative* X, int N, TFunctor functor, cudaStream_t stream )
    {
        const int block_size = 256;
        const int grid_size = ceil_div( N, block_size );

        elementwise_forward_kernel<TNative, TFunctor><<<grid_size, block_size, 0, stream>>>( Y, X, N, functor );

        cudaCheck( cudaGetLastError() );
    }

    template<typename TNative, typename TFunctor>
    void launch_elementwise_backward( TNative* dX, const TNative* X, const TNative* dY, int N, TFunctor functor, cudaStream_t stream )
    {
        const int block_size = 256;
        const int grid_size = ceil_div( N, block_size );

        elementwise_backward_kernel<TNative, TFunctor><<<grid_size, block_size, 0, stream>>>( dX, X, dY, N, functor );

        cudaCheck( cudaGetLastError() );
    }

    // Explicit instantiations: one (native, functor) symbol pair per elementwise
    // function for FP32 and BF16. A local codegen macro keeps the list readable; it
    // is confined to this .cu (not module code) and undefined immediately after.
#define MILA_INSTANTIATE_ELEMENTWISE_ACTIVATION( NATIVE, FUNCTOR ) \
    template void launch_elementwise_forward<NATIVE, FUNCTOR>( NATIVE*, const NATIVE*, int, FUNCTOR, cudaStream_t ); \
    template void launch_elementwise_backward<NATIVE, FUNCTOR>( NATIVE*, const NATIVE*, const NATIVE*, int, FUNCTOR, cudaStream_t );

#define MILA_INSTANTIATE_ELEMENTWISE_ACTIVATION_ALL( FUNCTOR ) \
    MILA_INSTANTIATE_ELEMENTWISE_ACTIVATION( float, FUNCTOR ) \
    MILA_INSTANTIATE_ELEMENTWISE_ACTIVATION( __nv_bfloat16, FUNCTOR )

    MILA_INSTANTIATE_ELEMENTWISE_ACTIVATION_ALL( Identity )
    MILA_INSTANTIATE_ELEMENTWISE_ACTIVATION_ALL( GeluTanh )
    MILA_INSTANTIATE_ELEMENTWISE_ACTIVATION_ALL( Silu )
    MILA_INSTANTIATE_ELEMENTWISE_ACTIVATION_ALL( Relu )
    MILA_INSTANTIATE_ELEMENTWISE_ACTIVATION_ALL( Tanh )
    MILA_INSTANTIATE_ELEMENTWISE_ACTIVATION_ALL( Sigmoid )
    MILA_INSTANTIATE_ELEMENTWISE_ACTIVATION_ALL( LeakyRelu )
    MILA_INSTANTIATE_ELEMENTWISE_ACTIVATION_ALL( Mish )

#undef MILA_INSTANTIATE_ELEMENTWISE_ACTIVATION_ALL
#undef MILA_INSTANTIATE_ELEMENTWISE_ACTIVATION
}
