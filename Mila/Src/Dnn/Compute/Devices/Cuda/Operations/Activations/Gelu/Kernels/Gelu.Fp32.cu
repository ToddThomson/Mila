/**
 * @file Gelu.Fp32.cu
 * @brief CUDA kernels for GELU activation forward and backward passes.
 *
 * Implements FP32 precision variants with optional vectorization.
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#ifndef NDEBUG
#  include <cassert>
#  define KERNEL_ASSERT(cond) assert(cond)
#else
#  define KERNEL_ASSERT(cond) ((void)0)
#endif


namespace Mila::Dnn::Compute::Cuda::Gelu
{
    constexpr float GELU_CUBIC_COEFF = 0.044715f;
    constexpr float GELU_SCALING_FACTOR = 0.7978845608028654f; // sqrt(2/pi)

    /**
     * @brief CUDA kernel for GELU activation forward pass with FP32 precision
     *
     * Computes the GELU activation function using the tanh approximation:
     * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
     *
     * @param Y Output tensor
     * @param X Input tensor
     * @param N Number of elements in the tensors
     */
    __global__ void gelu_forward_fp32_kernel( float* Y, const float* X, int N )
    {
        constexpr float kGeluInputAbsLimit = 50.0f;   // GELU input should be reasonable was 50.0f, but increased to 75.0f to avoid false positives in some cases
        constexpr float kGeluOutputAbsLimit = 100.0f; // GELU output can be slightly larger

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i < N )
        {
            float xi = X[ i ];

            // Check input
            if ( !isfinite( xi ) || fabsf( xi ) > kGeluInputAbsLimit )
            {
                printf(
                    "GELU DEBUG input: block=%d thread=%d idx=%d input=%f\n",
                    blockIdx.x, threadIdx.x, i, xi
                );
            }
            KERNEL_ASSERT( isfinite( xi ) );
            KERNEL_ASSERT( fabsf( xi ) <= kGeluInputAbsLimit );

            float cube = GELU_CUBIC_COEFF * xi * xi * xi;
            float y = 0.5f * xi * (1.0f + tanhf( GELU_SCALING_FACTOR * (xi + cube) ));

            // Check output
            if ( !isfinite( y ) || fabsf( y ) > kGeluOutputAbsLimit )
            {
                printf(
                    "GELU DEBUG output: block=%d thread=%d idx=%d input=%f output=%f cube=%f\n",
                    blockIdx.x, threadIdx.x, i, xi, y, cube
                );
            }
            KERNEL_ASSERT( isfinite( y ) );
            KERNEL_ASSERT( fabsf( y ) <= kGeluOutputAbsLimit );

            Y[ i ] = y;
        }
    }

    /**
     * @brief CUDA kernel for GELU activation backward pass with FP32 precision
     *
     * Computes the gradient of the GELU function with respect to its input using
     * the derivative: d(GELU)/dx = 0.5 * (1 + tanh(z)) + x * 0.5 * sech^2(z) * dz/dx
     * where z = sqrt(2/pi) * (x + 0.044715 * x^3)
     *
     * @param dX Output gradient with respect to input
     * @param X Original input values from forward pass
     * @param dY Gradient from upstream
     * @param N Number of elements in the tensors
     */
    __global__ void gelu_backward_fp32_kernel( float* dX, const float* X, const float* dY, int N )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if ( i < N )
        {
            float x = X[ i ];
            float cube = GELU_CUBIC_COEFF * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);

            // TODO: Alternative implementation using sech^2
            //float tanh_out = tanhf( tanh_arg );
            //float sech_squared = 1.0f - tanh_out * tanh_out;
            //float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_squared * GELU_SCALING_FACTOR * (1.0f + 3.0f * GELU_CUBIC_COEFF * x * x);

            float tanh_out = tanhf( tanh_arg );
            float coshf_out = coshf( tanh_arg );
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * GELU_CUBIC_COEFF * x * x);
            
            dX[ i ] = local_grad * dY[ i ];
        }
    }

    /**
     * @brief Host function to launch GELU forward pass with FP32 precision
     *
     * Computes the GELU activation function on each element of the input tensor.
     *
     * @param Y Output tensor
     * @param X Input tensor
     * @param N Number of elements in the tensors
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_gelu_forward_fp32( float* Y, const float* X, int N, cudaStream_t stream )
    {
        const int block_size = 128;
        const int grid_size = ceil_div( N, block_size );

        gelu_forward_fp32_kernel << <grid_size, block_size, 0, stream >> > (Y, X, N);

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host function to launch GELU backward pass with FP32 precision
     *
     * Computes the gradient of GELU with respect to its input.
     *
     * @param dX Output gradient with respect to input
     * @param X Original input values from forward pass
     * @param dY Gradient from upstream
     * @param N Number of elements in the tensors
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_gelu_backward_fp32( float* dX, const float* X, const float* dY, int N, cudaStream_t stream )
    {
        const int block_size = 128;
        const int grid_size = ceil_div( N, block_size );

        gelu_backward_fp32_kernel<<< grid_size, block_size, 0, stream >>> (dX, X, dY, N);

        cudaCheck( cudaGetLastError() );
    }
}