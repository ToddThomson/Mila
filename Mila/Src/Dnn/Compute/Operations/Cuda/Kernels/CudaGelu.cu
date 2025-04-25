#define _USE_MATH_DEFINES
#include <math.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute
{
    #define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

    __global__ void gelu_forward_fp32_kernel( float* Y, const float* X, int N ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i < N ) {
            float xi = X[ i ];
            float cube = 0.044715f * xi * xi * xi;
            Y[ i ] = 0.5f * xi * (1.0f + tanhf( GELU_SCALING_FACTOR * (xi + cube) ));
        }
    }

    __global__ void gelu_backward_fp32_kernel( float* dX, const float* X, const float* dY, const int N ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i < N ) {
            float x = X[ i ];
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = tanhf( tanh_arg );
            float coshf_out = coshf( tanh_arg );
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            dX[ i ] = local_grad * dY[ i ];
        }
    }

    /**
     * @brief CUDA kernel for GELU activation forward pass with FP16 precision
     *
     * Computes the GELU activation function using FP16 arithmetic:
     * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/?) * (x + 0.044715 * x^3)))
     *
     * @param Y Output tensor in half precision
     * @param X Input tensor in half precision
     * @param N Number of elements in the tensors
     */
    __global__ void gelu_forward_fp16_kernel( half* Y, const half* X, int N ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i < N ) {
            // Convert half to float for computation accuracy
            float xi = __half2float( X[ i ] );
            float cube = 0.044715f * xi * xi * xi;
            float result = 0.5f * xi * (1.0f + tanhf( GELU_SCALING_FACTOR * (xi + cube) ));
            // Convert result back to half
            Y[ i ] = __float2half( result );
        }
    }

    /**
     * @brief CUDA kernel for vectorized GELU forward pass with FP16 precision
     *
     * Processes two half values at once using half2 vector type for improved throughput.
     *
     * @param Y Output tensor as half2 vectors
     * @param X Input tensor as half2 vectors
     * @param N Number of half2 elements (half the number of total elements)
     */
    __global__ void gelu_forward_fp16_vectorized_kernel( half2* Y, const half2* X, int N ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i < N ) {
            // Extract the two half values from half2
            float2 vals = __half22float2( X[ i ] );

            // Process first element
            float x1 = vals.x;
            float cube1 = 0.044715f * x1 * x1 * x1;
            float result1 = 0.5f * x1 * (1.0f + tanhf( GELU_SCALING_FACTOR * (x1 + cube1) ));

            // Process second element
            float x2 = vals.y;
            float cube2 = 0.044715f * x2 * x2 * x2;
            float result2 = 0.5f * x2 * (1.0f + tanhf( GELU_SCALING_FACTOR * (x2 + cube2) ));

            // Combine results back into half2
            Y[ i ] = __float22half2_rn( make_float2( result1, result2 ) );
        }
    }

    /**
     * @brief CUDA kernel for GELU activation backward pass with FP16 precision
     *
     * Computes the gradient of the GELU function with respect to its input.
     *
     * @param dX Output gradient with respect to input in half precision
     * @param X Original input values from forward pass in half precision
     * @param dY Gradient from upstream in half precision
     * @param N Number of elements in the tensors
     */
    __global__ void gelu_backward_fp16_kernel( half* dX, const half* X, const half* dY, const int N ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i < N ) {
            // Convert half to float for computation accuracy
            float x = __half2float( X[ i ] );
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = tanhf( tanh_arg );
            float coshf_out = coshf( tanh_arg );
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            float dY_val = __half2float( dY[ i ] );
            // Convert result back to half
            dX[ i ] = __float2half( local_grad * dY_val );
        }
    }

    /**
     * @brief Host function to launch GELU forward pass with full precision (FP32)
     *
     * Computes the GELU activation function on each element of the input tensor.
     *
     * @param Y Output tensor
     * @param X Input tensor
     * @param N Number of elements in the tensors
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_gelu_forward_fp32( float* Y, const float* X, int N, cudaStream_t stream ) {
        const int block_size = 128;
        const int grid_size = ceil_div( N, block_size );

        gelu_forward_fp32_kernel << <grid_size, block_size, 0, stream >> > (Y, X, N);

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host function to launch GELU backward pass with full precision (FP32)
     *
     * Computes the gradient of GELU with respect to its input.
     *
     * @param dX Output gradient with respect to input
     * @param X Original input values from forward pass
     * @param dY Gradient from upstream
     * @param N Number of elements in the tensors
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_gelu_backward_fp32( float* dX, const float* X, const float* dY, const int N, cudaStream_t stream ) {
        const int block_size = 128;
        const int grid_size = ceil_div( N, block_size );
        gelu_backward_fp32_kernel << <grid_size, block_size, 0, stream >> > (dX, X, dY, N);
        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host function to launch GELU forward pass with half precision (FP16)
     *
     * Computes the GELU activation function on each element of the input tensor using FP16.
     *
     * @param Y Output tensor in half precision
     * @param X Input tensor in half precision
     * @param N Number of elements in the tensors
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_gelu_forward_fp16( half* Y, const half* X, int N, cudaStream_t stream ) {
        const int block_size = 128;

        // Check if N is even for potential vectorization
        if ( N % 2 == 0 ) {
            // Use vectorized version for better performance when possible
            const int N_half2 = N / 2;
            const int grid_size = ceil_div( N_half2, block_size );

            gelu_forward_fp16_vectorized_kernel << <grid_size, block_size, 0, stream >> > (
                reinterpret_cast<half2*>(Y),
                reinterpret_cast<const half2*>(X),
                N_half2);
        }
        else {
            // Fall back to non-vectorized version for odd N
            const int grid_size = ceil_div( N, block_size );

            gelu_forward_fp16_kernel << <grid_size, block_size, 0, stream >> > (Y, X, N);
        }

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host function to launch GELU backward pass with half precision (FP16)
     *
     * Computes the gradient of GELU with respect to its input using FP16.
     *
     * @param dX Output gradient with respect to input in half precision
     * @param X Original input values from forward pass in half precision
     * @param dY Gradient from upstream in half precision
     * @param N Number of elements in the tensors
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_gelu_backward_fp16( half* dX, const half* X, const half* dY, const int N, cudaStream_t stream ) {
        const int block_size = 128;
        const int grid_size = ceil_div( N, block_size );

        gelu_backward_fp16_kernel << <grid_size, block_size, 0, stream >> > (dX, X, dY, N);

        cudaCheck( cudaGetLastError() );
    }
}
