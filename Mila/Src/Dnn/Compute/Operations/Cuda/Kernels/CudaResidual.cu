#define _USE_MATH_DEFINES
#include <math.h>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute
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
    __global__ void residual_forward_fp32_kernel( float* out, const float* input_1, const float* input_2, int N ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx < N ) {
            out[ idx ] = __ldcs( &input_1[ idx ] ) + __ldcs( &input_2[ idx ] );
        }
    }

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
    __global__ void residual_forward_fp16_kernel( half* out, const half* input_1, const half* input_2, int N ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx < N ) {
            // For half precision, we can use CUDA's built-in half arithmetic
            out[ idx ] = __hadd( input_1[ idx ], input_2[ idx ] );
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
    __global__ void residual_forward_fp16_vectorized_kernel( half2* out, const half2* input_1, const half2* input_2, int N ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx < N ) {
            // Process two half values at once using half2 vector type
            out[ idx ] = __hadd2( input_1[ idx ], input_2[ idx ] );
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
    void cuda_residual_forward_fp32(
        float* out,
        const float* inp1,
        const float* inp2,
        int N,
        cudaStream_t stream ) {

        // FP32 implementation
        const int block_size = 256;
        const int grid_size = ceil_div( N, block_size );

        residual_forward_fp32_kernel << <grid_size, block_size, 0, stream >> > (out, inp1, inp2, N);
        cudaCheck( cudaGetLastError() );
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
}
