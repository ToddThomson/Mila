#define _USE_MATH_DEFINES
#include <math.h>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute::Cuda::Residual
{
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
        float* out, 
        const float* input_1, 
        const float* input_2, 
        int N )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < N)
        {
            out[idx] = __ldg( &input_1[idx] ) + __ldg( &input_2[idx] );
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
        float4* out,
        const float4* input_1,
        const float4* input_2,
        int N )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < N)
        {
            float4 a = __ldg( &input_1[idx] );
            float4 b = __ldg( &input_2[idx] );

            // Element-wise addition of float4
            out[idx] = make_float4(
                a.x + b.x,
                a.y + b.y,
                a.z + b.z,
                a.w + b.w
            );
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
        float* grad_input_1,
        float* grad_input_2,
        const float* grad_output,
        int N )
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < N )
        {
            float grad = __ldg( &grad_output[ idx ] );
            grad_input_1[ idx ] = grad;
            grad_input_2[ idx ] = grad;
        }
    }

    /**
 * @brief Vectorized FP32 backward kernel using float4 for 4x throughput
 *
 * Processes four float values at once when propagating gradients.
 * Uses direct assignment since residual backward simply broadcasts
 * the output gradient to both input gradients.
 *
 * @param grad_input_1 Gradient w.r.t. first input as float4 vectors
 * @param grad_input_2 Gradient w.r.t. second input as float4 vectors
 * @param grad_output Gradient from next layer as float4 vectors
 * @param N Total number of float4 elements
 */
    __global__ void residual_backward_fp32_vectorized_kernel(
        float4* grad_input_1,
        float4* grad_input_2,
        const float4* grad_output,
        int N )
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx < N )
        {
            float4 grad = __ldg( &grad_output[ idx ] );
            grad_input_1[ idx ] = grad;
            grad_input_2[ idx ] = grad;
        }
    }

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

        if (N % 4 == 0)
        {
            // Use float4 vectorized kernel
            // Alignment is guaranteed by tensor allocation
            const size_t N_float4 = N / 4;
            const int grid_size = ceil_div( static_cast<int>(N_float4), block_size );

            residual_forward_fp32_vectorized_kernel <<<grid_size, block_size, 0, stream >> > (
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

        if (N % 4 == 0)
        {
            const size_t N_float4 = N / 4;
            const int grid_size = ceil_div( static_cast<int>(N_float4), block_size );

            residual_backward_fp32_vectorized_kernel<<<grid_size, block_size, 0, stream >> > (
                reinterpret_cast<float4*>(grad_inp1),
                reinterpret_cast<float4*>(grad_inp2),
                reinterpret_cast<const float4*>(grad_out),
                N_float4 );
        }
        else
        {
            // Fall back to scalar kernel
            const int grid_size = ceil_div( static_cast<int>(N), block_size );

            residual_backward_fp32_kernel<<<grid_size, block_size, 0, stream >>>(
                grad_inp1, grad_inp2, grad_out, N );
        }

        cudaCheck( cudaGetLastError() );
    }
}