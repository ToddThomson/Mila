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
     * @brief CUDA kernel for element-wise addition of two input tensors with FP16 precision
     *
     * This kernel performs residual connection by adding elements from two input tensors using
     * half-precision floating point arithmetic for improved performance on compatible hardware.
     * Uses vector loads where possible to optimize memory throughput.
     *
     * @param out Output tensor where the result is stored in half precision
     * @param input_1 First input tensor in half precision
     * @param input_2 Second input tensor in half precision (residual connection)
     * @param N Total number of elements in the tensors
     */
    __global__ void residual_forward_fp16_kernel(
        half* out, 
        const half* input_1, 
        const half* input_2,
        size_t N ) 
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < N ) {
            out[ idx ] = __hadd( input_1[ idx ], input_2[ idx ] );
        }
    }

    /**
     * @brief CUDA kernel for backward pass of residual connection with FP16 precision
     *
     * Propagates gradients through the residual connection using half-precision arithmetic.
     * Accumulates gradients to both inputs.
     *
     * @param grad_input_1 Gradient w.r.t. first input in half precision (accumulated)
     * @param grad_input_2 Gradient w.r.t. second input in half precision (accumulated)
     * @param grad_output Gradient from the next layer in half precision
     * @param N Total number of elements in the tensors
     */
    __global__ void residual_backward_fp16_kernel(
        half* grad_input_1,
        half* grad_input_2,
        const half* grad_output,
        int N )
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < N )
        {
            half grad = grad_output[ idx ];
            // Note: atomicAdd for half requires compute capability 7.0+
            atomicAdd( &grad_input_1[ idx ], grad );
            atomicAdd( &grad_input_2[ idx ], grad );
        }
    }

    /**
     * @brief Vectorized half-precision backward kernel using half2
     *
     * Processes two half values at once for improved throughput.
     * Uses half2 atomic operations where supported.
     *
     * @param grad_input_1 Gradient w.r.t. first input as half2 vectors
     * @param grad_input_2 Gradient w.r.t. second input as half2 vectors
     * @param grad_output Gradient from the next layer as half2 vectors
     * @param N Total number of half2 elements
     */
    __global__ void residual_backward_fp16_vectorized_kernel(
        half2* grad_input_1,
        half2* grad_input_2,
        const half2* grad_output,
        int N )
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < N )
        {
            half2 grad = grad_output[ idx ];

            // For half2, we need to handle atomics carefully
            // Option 1: Use half2 atomic if available (compute capability 9.0+)
            // Option 2: Split into two half atomics
            half* grad1_ptr = reinterpret_cast<half*>( &grad_input_1[ idx ] );
            half* grad2_ptr = reinterpret_cast<half*>( &grad_input_2[ idx ] );
            half* grad_ptr = reinterpret_cast<half*>( &grad );

            atomicAdd( &grad1_ptr[ 0 ], grad_ptr[ 0 ] );
            atomicAdd( &grad1_ptr[ 1 ], grad_ptr[ 1 ] );
            atomicAdd( &grad2_ptr[ 0 ], grad_ptr[ 0 ] );
            atomicAdd( &grad2_ptr[ 1 ], grad_ptr[ 1 ] );
        }
    }

    /**
     * @brief Alternative half-precision kernel using half2 for better vectorization
     *
     * This kernel processes two half values at once using half2 vector type,
     * improving throughput on hardware with good FP16 support.
     *
     * @param out Output tensor as half2 vectors
     * @param input_1 First input tensor as half2 vectors
     * @param input_2 Second input tensor as half2 vectors
     * @param N Total number of half2 elements (half the number of total elements)
     */
    __global__ void residual_forward_fp16_vectorized_kernel( 
        half2* out, 
        const half2* input_1, const half2* input_2,
        int N ) 
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < N ) {
            out[ idx ] = __hadd2( input_1[ idx ], input_2[ idx ] );
        }
    }

    /**
     * @brief Host function to launch residual addition kernel with half precision (FP16)
     *
     * Adds two tensors element-wise with half-precision arithmetic,
     * implementing the residual connection commonly used in neural network architectures.
     *
     * Formula: out = inp1 + inp2
     *
     * @param out Output tensor to store the result in half precision
     * @param inp1 First input tensor in half precision
     * @param inp2 Second input tensor in half precision
     * @param N Total number of elements in the tensors
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_residual_forward_fp16(
        half* out,
        const half* inp1,
        const half* inp2,
        int N,
        cudaStream_t stream ) {

        // Check if N is even for potential vectorization
        if ( N % 2 == 0 ) {
            // Use vectorized version for better performance when possible
            const int block_size = 256;
            const int N_half2 = N / 2;
            const int grid_size = ceil_div( N_half2, block_size );

            residual_forward_fp16_vectorized_kernel << <grid_size, block_size, 0, stream >> > (
                reinterpret_cast<half2*>(out),
                reinterpret_cast<const half2*>(inp1),
                reinterpret_cast<const half2*>(inp2),
                N_half2);
        }
        else {
            // Fall back to non-vectorized version for odd N
            const int block_size = 256;
            const int grid_size = ceil_div( N, block_size );

            residual_forward_fp16_kernel << <grid_size, block_size, 0, stream >> > (out, inp1, inp2, N);
        }

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host function to launch residual backward kernel with half precision (FP16)
     *
     * Propagates gradients through the residual connection using half-precision arithmetic.
     * Both inputs receive the same gradient.
     *
     * Formula: grad_inp1 += grad_out, grad_inp2 += grad_out
     *
     * @param grad_inp1 Gradient tensor for first input in half precision (accumulated)
     * @param grad_inp2 Gradient tensor for second input in half precision (accumulated)
     * @param grad_out Gradient from downstream layers in half precision
     * @param N Total number of elements in the tensors
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_residual_backward_fp16(
        half* grad_inp1,
        half* grad_inp2,
        const half* grad_out,
        int N,
        cudaStream_t stream )
    {
        // Check if N is even for potential vectorization
        if ( N % 2 == 0 )
        {
            // Use vectorized version
            const int block_size = 256;
            const size_t N_half2 = N / 2;
            const int grid_size = ceil_div( static_cast<int>(N_half2), block_size );

            residual_backward_fp16_vectorized_kernel << <grid_size, block_size, 0, stream >> > (
                reinterpret_cast<half2*>(grad_inp1),
                reinterpret_cast<half2*>(grad_inp2),
                reinterpret_cast<const half2*>(grad_out),
                N_half2);
        }
        else
        {
            // Fall back to non-vectorized version
            const int block_size = 256;
            const int grid_size = ceil_div( static_cast<int>(N), block_size );

            residual_backward_fp16_kernel << <grid_size, block_size, 0, stream >> > (
                grad_inp1, grad_inp2, grad_out, N);
        }

        cudaCheck( cudaGetLastError() );
    }
}
