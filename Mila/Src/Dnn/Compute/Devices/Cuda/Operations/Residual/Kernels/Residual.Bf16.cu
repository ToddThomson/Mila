/**
 * @file Residual.Bf16.cu
 * @brief BF16 CUDA kernels and host launchers for the residual (y = x1 + x2) operation.
 *
 * All kernels require SM >= 8.0 (Ampere) for native BF16 and BF16 atomicAdd support.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute::Cuda::Residual
{
    __global__ void residual_forward_bf16_kernel(
        __nv_bfloat16* __restrict__       out,
        const __nv_bfloat16* __restrict__ input_1,
        const __nv_bfloat16* __restrict__ input_2,
        int N )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < N )
            out[ idx ] = input_1[ idx ] + input_2[ idx ];
    }

    __global__ void residual_forward_bf16_vectorized_kernel(
        __nv_bfloat162* __restrict__       out,
        const __nv_bfloat162* __restrict__ input_1,
        const __nv_bfloat162* __restrict__ input_2,
        int N )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < N )
            out[ idx ] = input_1[ idx ] + input_2[ idx ];
    }

    // Accumulates dY into both gradient buffers via BF16 atomicAdd (SM >= 8.0).
    __global__ void residual_backward_bf16_kernel(
        __nv_bfloat16* __restrict__       grad_input_1,
        __nv_bfloat16* __restrict__       grad_input_2,
        const __nv_bfloat16* __restrict__ grad_output,
        int N )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < N )
        {
            __nv_bfloat16 grad = grad_output[ idx ];
            atomicAdd( &grad_input_1[ idx ], grad );
            atomicAdd( &grad_input_2[ idx ], grad );
        }
    }

    // Vectorized backward: processes two BF16 elements per thread.
    // No atomicAdd for __nv_bfloat162, so splits into two scalar BF16 atomics.
    __global__ void residual_backward_bf16_vectorized_kernel(
        __nv_bfloat162* __restrict__       grad_input_1,
        __nv_bfloat162* __restrict__       grad_input_2,
        const __nv_bfloat162* __restrict__ grad_output,
        int N )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < N )
        {
            __nv_bfloat162 grad = grad_output[ idx ];

            __nv_bfloat16* g1 = reinterpret_cast<__nv_bfloat16*>( &grad_input_1[ idx ] );
            __nv_bfloat16* g2 = reinterpret_cast<__nv_bfloat16*>( &grad_input_2[ idx ] );
            __nv_bfloat16* g = reinterpret_cast<__nv_bfloat16*>( &grad );

            atomicAdd( &g1[ 0 ], g[ 0 ] );
            atomicAdd( &g1[ 1 ], g[ 1 ] );
            atomicAdd( &g2[ 0 ], g[ 0 ] );
            atomicAdd( &g2[ 1 ], g[ 1 ] );
        }
    }

    // =========================================================================
    // Host launchers

    void cuda_residual_forward_bf16(
        __nv_bfloat16* Y,
        const __nv_bfloat16* X1, const __nv_bfloat16* X2,
        float /*scale*/,
        int N,
        cudaStream_t stream )
    {
        constexpr int block_size = 256;

        if ( N % 2 == 0 )
        {
            const int N_bf162 = N / 2;
            const int grid_size = ceil_div( N_bf162, block_size );

            residual_forward_bf16_vectorized_kernel << <grid_size, block_size, 0, stream >> > (
                reinterpret_cast<__nv_bfloat162*>(Y),
                reinterpret_cast<const __nv_bfloat162*>(X1),
                reinterpret_cast<const __nv_bfloat162*>(X2),
                N_bf162);
        }
        else
        {
            const int grid_size = ceil_div( N, block_size );

            residual_forward_bf16_kernel << <grid_size, block_size, 0, stream >> > (
                Y, X1, X2, N);
        }

        cudaCheck( cudaGetLastError() );
    }

    void cuda_residual_backward_bf16(
        __nv_bfloat16* dX1,
        __nv_bfloat16* dX2,
        const __nv_bfloat16* dY,
        int N,
        cudaStream_t stream )
    {
        constexpr int block_size = 256;

        if ( N % 2 == 0 )
        {
            const int N_bf162 = N / 2;
            const int grid_size = ceil_div( N_bf162, block_size );

            residual_backward_bf16_vectorized_kernel << <grid_size, block_size, 0, stream >> > (
                reinterpret_cast<__nv_bfloat162*>(dX1),
                reinterpret_cast<__nv_bfloat162*>(dX2),
                reinterpret_cast<const __nv_bfloat162*>(dY),
                N_bf162);
        }
        else
        {
            const int grid_size = ceil_div( N, block_size );

            residual_backward_bf16_kernel << <grid_size, block_size, 0, stream >> > (
                dX1, dX2, dY, N);
        }

        cudaCheck( cudaGetLastError() );
    }
}