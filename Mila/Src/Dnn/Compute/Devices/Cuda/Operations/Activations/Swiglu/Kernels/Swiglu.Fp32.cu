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

namespace Mila::Dnn::Compute::Cuda::Swiglu
{
    /**
     * @brief CUDA kernel for SwiGLU activation forward pass with FP32 precision
     *
     * Computes the SwiGLU activation function:
     * SwiGLU(x, gate) = Swish(gate) * x
     * where Swish(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
     *
     * Input layout: [batch_size, 2 * hidden_dim]
     * - First half: gate values [batch_size, hidden_dim]
     * - Second half: x values [batch_size, hidden_dim]
     * Output layout: [batch_size, hidden_dim]
     *
     * @param Y Output tensor [batch_size * hidden_dim]
     * @param X Input tensor [batch_size * 2 * hidden_dim]
     * @param N Number of output elements (batch_size * hidden_dim)
     */
    __global__ void swiglu_forward_fp32_kernel(
        float* __restrict__ Y,
        const float* __restrict__ X,
        int N )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if ( i < N )
        {
            // Read gate and x values
            // Gate is in first half, x is in second half
            float gate = X[ i ];
            float x = X[ i + N ];

        #ifndef NDEBUG
            constexpr float kSwigluInputAbsLimit = 50.0f;
            constexpr float kSwigluOutputAbsLimit = 100.0f;

            if ( !isfinite( gate ) || fabsf( gate ) > kSwigluInputAbsLimit ||
                !isfinite( x ) || fabsf( x ) > kSwigluInputAbsLimit )
            {
                printf(
                    "SwiGLU FWD input anomaly: idx=%d gate=%f x=%f\n",
                    i, gate, x
                );
                assert( false );
            }
        #endif

            // Compute Swish(gate) = gate * sigmoid(gate)
            // sigmoid(gate) = 1 / (1 + exp(-gate))
            float swish = gate / (1.0f + expf( -gate ));

            // SwiGLU output
            float y = swish * x;

        #ifndef NDEBUG
            if ( !isfinite( y ) || fabsf( y ) > kSwigluOutputAbsLimit )
            {
                printf(
                    "SwiGLU FWD output anomaly: idx=%d gate=%f x=%f swish=%f output=%f\n",
                    i, gate, x, swish, y
                );
                assert( false );
            }
        #endif

            Y[ i ] = y;
        }
    }

    /**
     * @brief CUDA kernel for SwiGLU activation backward pass with FP32 precision
     *
     * Computes the gradient of the SwiGLU function with respect to its inputs.
     *
     * For SwiGLU(x, gate) = Swish(gate) * x where Swish(g) = g * sigmoid(g):
     * - d(SwiGLU)/dx = Swish(gate)
     * - d(SwiGLU)/d(gate) = x * d(Swish)/d(gate)
     *   where d(Swish)/d(gate) = sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate))
     *                           = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
     *
     * @param dX Output gradient with respect to input [batch_size * 2 * hidden_dim]
     * @param X Original input values from forward pass [batch_size * 2 * hidden_dim]
     * @param dY Gradient from upstream [batch_size * hidden_dim]
     * @param N Number of output elements (batch_size * hidden_dim)
     */
    __global__ void swiglu_backward_fp32_kernel(
        float* __restrict__ dX,
        const float* __restrict__ X,
        const float* __restrict__ dY,
        int N )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if ( i < N )
        {
            float gate = X[ i ];
            float x = X[ i + N ];
            float dy = dY[ i ];

            // Compute sigmoid(gate) = 1 / (1 + exp(-gate))
            float sigmoid_gate = 1.0f / (1.0f + expf( -gate ));

            // Swish(gate) = gate * sigmoid(gate)
            float swish = gate * sigmoid_gate;

            // Gradient with respect to x: d(SwiGLU)/dx = Swish(gate)
            float dx = swish * dy;

            // Gradient with respect to gate:
            // d(Swish)/d(gate) = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
            float dswish_dgate = sigmoid_gate * (1.0f + gate * (1.0f - sigmoid_gate));
            float dgate = x * dswish_dgate * dy;

            // Write gradients
            dX[ i ] = dgate;        // Gradient for gate (first half)
            dX[ i + N ] = dx;       // Gradient for x (second half)
        }
    }

    /**
     * @brief Host function to launch SwiGLU forward pass with FP32 precision
     *
     * Computes the SwiGLU activation function on the input tensor.
     *
     * @param Y Output tensor [batch_size * hidden_dim]
     * @param X Input tensor [batch_size * 2 * hidden_dim]
     * @param N Number of output elements (batch_size * hidden_dim)
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_swiglu_forward_fp32(
        float* Y,
        const float* X,
        int N,
        cudaStream_t stream )
    {
        const int block_size = 128;
        const int grid_size = (N + block_size - 1) / block_size;

        swiglu_forward_fp32_kernel << <grid_size, block_size, 0, stream >> > (Y, X, N);

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host function to launch SwiGLU backward pass with FP32 precision
     *
     * Computes the gradient of SwiGLU with respect to its inputs.
     *
     * @param dX Output gradient with respect to input [batch_size * 2 * hidden_dim]
     * @param X Original input values from forward pass [batch_size * 2 * hidden_dim]
     * @param dY Gradient from upstream [batch_size * hidden_dim]
     * @param N Number of output elements (batch_size * hidden_dim)
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_swiglu_backward_fp32(
        float* dX,
        const float* X,
        const float* dY,
        int N,
        cudaStream_t stream )
    {
        const int block_size = 128;
        const int grid_size = (N + block_size - 1) / block_size;

        swiglu_backward_fp32_kernel << <grid_size, block_size, 0, stream >> > (dX, X, dY, N);

        cudaCheck( cudaGetLastError() );
    }
}