#define _USE_MATH_DEFINES
#include <math.h>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

#ifndef NDEBUG
#  include <cassert>
#  define KERNEL_ASSERT(cond) assert(cond)
#else
#  define KERNEL_ASSERT(cond) ((void)0)
#endif

namespace Mila::Dnn::Compute::Cuda::Residual
{
    // Conservative residual magnitude bound used only for debug assertions.
    // Chosen to catch explosions while avoiding false positives for common models.
    static __device__ __constant__ float kResidualAbsLimit = 100.0f;

    /**
     * @brief CUDA kernel for element-wise addition of two input tensors with FP32 precision
     *
     * This kernel performs residual connection by adding elements from two input tensors.
     * Uses __ldcs for cached loads to optimize memory access patterns.
     *
     * @param out Output tensor where the result is stored
     * @param input_1 First input tensor
     * @param input_2 Second input tensor (residual connection)
     * @param N Total number of elements in the tensors
     */
    __global__ void residual_forward_fp32_kernel(
        float* Y,
        const float* A,
        const float* B,
        int N )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < N )
        {
            Y[ idx ] = __ldg( &A[ idx ] ) + __ldg( &B[ idx ] );

            KERNEL_ASSERT( fabsf( Y[ idx ] ) <= kResidualAbsLimit );
        }
    }

    /**
     * @brief Vectorized FP32 forward kernel using float4 for 4x throughput
     *
     * Processes four float values at once for improved memory bandwidth utilization.
     * This kernel is used when N is divisible by 4 and pointers are aligned.
     *
     * @param out Output tensor as float4 vectors
     * @param input_1 First input tensor as float4 vectors
     * @param input_2 Second input tensor as float4 vectors
     * @param N Total number of float4 elements (1/4 of total float elements)
     */
    __global__ void residual_forward_fp32_vectorized_kernel(
        float4* Y,
        const float4* A,
        const float4* B,
        int N )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < N )
        {
            float4 a = __ldg( &A[ idx ] );
            float4 b = __ldg( &B[ idx ] );

            float4 r;
            r.x = a.x + b.x; KERNEL_ASSERT( fabsf( r.x ) <= kResidualAbsLimit );
            r.y = a.y + b.y; KERNEL_ASSERT( fabsf( r.y ) <= kResidualAbsLimit );
            r.z = a.z + b.z; KERNEL_ASSERT( fabsf( r.z ) <= kResidualAbsLimit );
            r.w = a.w + b.w; KERNEL_ASSERT( fabsf( r.w ) <= kResidualAbsLimit );

            Y[ idx ] = r;
        }
    }

    /**
     * @brief CUDA kernel for backward pass of residual connection with FP32 precision
     *
     * Propagates gradients through the residual connection. Since forward is y = x1 + x2,
     * the gradient flows equally to both inputs: dx1 = dy, dx2 = dy.
     *
     * @param grad_input_1 Gradient w.r.t. first input
     * @param grad_input_2 Gradient w.r.t. second input
     * @param grad_output Gradient from the next layer
     * @param N Total number of elements in the tensors
     */
    __global__ void residual_backward_fp32_kernel(
        float* dA,
        float* dB,
        const float* dY,
        int N )
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < N )
        {
            float grad = __ldg( &dY[ idx ] );

            KERNEL_ASSERT( fabsf( grad ) <= kResidualAbsLimit );
            
            dA[ idx ] = grad;
            dB[ idx ] = grad;
        }
    }

    /**
     * @brief Vectorized FP32 backward kernel using float4 for 4x throughput
     *
     * Processes four float values at once when propagating gradients.
     * Uses direct assignment since residual backward simply broadcasts
     * the output gradient to both input gradients.
     *
     * @param dA Gradient w.r.t. first input as float4 vectors
     * @param dB Gradient w.r.t. second input as float4 vectors
     * @param dY Gradient from next layer as float4 vectors
     * @param N Total number of float4 elements
     */
    __global__ void residual_backward_fp32_vectorized_kernel(
        float4* dA,
        float4* dB,
        const float4* dY,
        int N )
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < N )
        {
            float4 grad = __ldg( &dY[ idx ] );

            KERNEL_ASSERT( fabsf( grad.x ) <= kResidualAbsLimit );
            KERNEL_ASSERT( fabsf( grad.y ) <= kResidualAbsLimit );
            KERNEL_ASSERT( fabsf( grad.z ) <= kResidualAbsLimit );
            KERNEL_ASSERT( fabsf( grad.w ) <= kResidualAbsLimit );

            dA[ idx ] = grad;
            dB[ idx ] = grad;
        }
    }

    // ========================================================================
    // Host functions
    // ========================================================================

    /**
     * @brief Host function to launch residual addition kernel with full precision (FP32)
     *
     * Adds two tensors element-wise, implementing the residual connection commonly
     * used in neural network architectures to improve gradient flow.
     *
     * Formula: out = inp1 + inp2
     *
     * @param out Output tensor to store the result
     * @param inp1 First input tensor
     * @param inp2 Second input tensor
     * @param N Total number of elements in the tensors
     * @param stream CUDA stream for asynchronous execution
     */
     /**
      * @brief Optimized host function for FP32 forward pass
      *
      * Since tensors are guaranteed to be aligned to CUDA_WARP_SIZE * sizeof(float) = 128 bytes,
      * we can always use vectorized kernels when N is divisible by 4.
      */
    void cuda_residual_forward_fp32(
        float* out,
        const float* inp1,
        const float* inp2,
        int N,
        cudaStream_t stream )
    {
        const int block_size = 256;

        if ( N % 4 == 0 )
        {
            const size_t N_float4 = N / 4;
            const int grid_size = ceil_div( static_cast<int>(N_float4), block_size );

            residual_forward_fp32_vectorized_kernel << <grid_size, block_size, 0, stream >> > (
                reinterpret_cast<float4*>(out),
                reinterpret_cast<const float4*>(inp1),
                reinterpret_cast<const float4*>(inp2),
                N_float4
                );
        }
        else
        {
            // Fall back to scalar kernel for non-multiple-of-4 sizes
            const int grid_size = ceil_div( static_cast<int>(N), block_size );

            residual_forward_fp32_kernel << <grid_size, block_size, 0, stream >> > (
                out, inp1, inp2, N
                );
        }

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host function to launch residual backward kernel with full precision (FP32)
     *
     * Propagates gradients through the residual connection. Both inputs receive
     * the same gradient since the forward pass is simple addition.
     *
     * Formula: grad_inp1 = grad_out, grad_inp2 = grad_out
     *
     * @param grad_inp1 Gradient tensor for first input
     * @param grad_inp2 Gradient tensor for second input
     * @param grad_out Gradient from downstream layers
     * @param N Total number of elements in the tensors
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_residual_backward_fp32(
        float* grad_inp1,
        float* grad_inp2,
        const float* grad_out,
        int N,
        cudaStream_t stream )
    {
        const int block_size = 256;

        if ( N % 4 == 0 )
        {
            const size_t N_float4 = N / 4;
            const int grid_size = ceil_div( static_cast<int>(N_float4), block_size );

            residual_backward_fp32_vectorized_kernel << <grid_size, block_size, 0, stream >> > (
                reinterpret_cast<float4*>(grad_inp1),
                reinterpret_cast<float4*>(grad_inp2),
                reinterpret_cast<const float4*>(grad_out),
                N_float4);
        }
        else
        {
            // Fall back to scalar kernel
            const int grid_size = ceil_div( static_cast<int>(N), block_size );

            residual_backward_fp32_kernel << <grid_size, block_size, 0, stream >> > (
                grad_inp1, grad_inp2, grad_out, N);
        }

        cudaCheck( cudaGetLastError() );
    }
}