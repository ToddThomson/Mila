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
     * @brief CUDA kernel for softmax with arbitrary axis support (FP32 version)
     */
    __global__ void softmax_forward_general_fp32_kernel(
        float* Y, const float* X,
        int outer_size, int dim_size, int inner_size )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < outer_size * inner_size ) {
            int outer_idx = idx / inner_size;
            int inner_idx = idx % inner_size;

            // Find maximum value in this slice for numerical stability
            float max_val = -INFINITY;
            for ( int d = 0; d < dim_size; d++ ) {
                int index = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
                float val = X[ index ];
                if ( val > max_val ) {
                    max_val = val;
                }
            }

            // Compute sum of exp(x - max_val)
            float sum = 0.0f;
            for ( int d = 0; d < dim_size; d++ ) {
                int index = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
                Y[ index ] = expf( X[ index ] - max_val );
                sum += Y[ index ];
            }

            // Normalize by sum
            float inv_sum = 1.0f / sum;
            for ( int d = 0; d < dim_size; d++ ) {
                int index = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
                Y[ index ] *= inv_sum;
            }
        }
    }

    /**
     * @brief CUDA kernel for softmax with arbitrary axis support (FP16 version)
     */
    __global__ void softmax_forward_general_fp16_kernel(
        half* Y, const half* X,
        int outer_size, int dim_size, int inner_size )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < outer_size * inner_size ) {
            int outer_idx = idx / inner_size;
            int inner_idx = idx % inner_size;

            // Find maximum value in this slice for numerical stability
            float max_val = -INFINITY;  // Use float for intermediate computations
            for ( int d = 0; d < dim_size; d++ ) {
                int index = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
                float val = __half2float( X[ index ] );
                if ( val > max_val ) {
                    max_val = val;
                }
            }

            // Compute sum of exp(x - max_val)
            float sum = 0.0f;
            for ( int d = 0; d < dim_size; d++ ) {
                int index = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
                float exp_val = expf( __half2float( X[ index ] ) - max_val );
                Y[ index ] = __float2half( exp_val );
                sum += exp_val;
            }

            // Normalize by sum
            float inv_sum = 1.0f / sum;
            for ( int d = 0; d < dim_size; d++ ) {
                int index = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
                Y[ index ] = __float2half( __half2float( Y[ index ] ) * inv_sum );
            }
        }
    }

    /**
     * @brief CUDA kernel for softmax forward pass with FP32 precision
     *
     * Computes softmax function on each row of the input tensor.
     *
     * @param Y Output tensor (N, C)
     * @param X Input tensor (N, C)
     * @param N Number of rows
     * @param C Number of columns (features)
     */
    __global__ void softmax_forward_fp32_kernel( float* Y, const float* X, int N, int C ) {
        // input is (N, C)
        // output is (N, C), each row of input will get softmaxed
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i < N ) {
            const float* X_row = X + i * C;
            float* Y_row = Y + i * C;

            float maxval = -INFINITY;
            for ( int j = 0; j < C; j++ ) {
                if ( X_row[ j ] > maxval ) {
                    maxval = X_row[ j ];
                }
            }
            double sum = 0.0;
            for ( int j = 0; j < C; j++ ) {
                Y_row[ j ] = expf( X_row[ j ] - maxval );
                sum += Y_row[ j ];
            }
            for ( int j = 0; j < C; j++ ) {
                Y_row[ j ] /= (float)sum;
            }
        }
    }

    /**
     * @brief CUDA kernel for softmax forward pass with FP16 precision
     *
     * Computes softmax function on each row of the input tensor using half precision.
     *
     * @param Y Output tensor (N, C) in half precision
     * @param X Input tensor (N, C) in half precision
     * @param N Number of rows
     * @param C Number of columns (features)
     */
    __global__ void softmax_forward_fp16_kernel( half* Y, const half* X, int N, int C ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i < N ) {
            const half* X_row = X + i * C;
            half* Y_row = Y + i * C;

            // Find max value for numerical stability
            float maxval = -INFINITY;
            for ( int j = 0; j < C; j++ ) {
                float val = __half2float( X_row[ j ] );
                if ( val > maxval ) {
                    maxval = val;
                }
            }

            // Compute exp and sum
            float sum = 0.0f;
            for ( int j = 0; j < C; j++ ) {
                float val = expf( __half2float( X_row[ j ] ) - maxval );
                Y_row[ j ] = __float2half( val );
                sum += val;
            }

            // Normalize
            for ( int j = 0; j < C; j++ ) {
                Y_row[ j ] = __float2half( __half2float( Y_row[ j ] ) / sum );
            }
        }
    }

    /**
     * @brief Host function to launch softmax forward pass (FP32 version)
     */
    void cuda_softmax_forward_fp32(
        float* Y,
        const float* X,
        int N,
        int C,
        cudaStream_t stream )
    {
        const int block_size = 512;
        const int grid_size = ceil_div( N, block_size );

        softmax_forward_fp32_kernel << <grid_size, block_size, 0, stream >> > (Y, X, N, C);
        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host function to launch softmax forward pass (FP16 version)
     */
    void cuda_softmax_forward_fp16(
        half* Y,
        const half* X,
        int N,
        int C,
        cudaStream_t stream )
    {
        const int block_size = 512;
        const int grid_size = ceil_div( N, block_size );

        softmax_forward_fp16_kernel << <grid_size, block_size, 0, stream >> > (Y, X, N, C);
        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host function to launch arbitrary axis softmax (FP32 version)
     */
    void cuda_softmax_forward_general_fp32(
        float* Y,
        const float* X,
        int outer_size,
        int dim_size,
        int inner_size,
        cudaStream_t stream )
    {
        int total_slices = outer_size * inner_size;
        int block_size = 512;
        int grid_size = ceil_div( total_slices, block_size );

        softmax_forward_general_fp32_kernel << <grid_size, block_size, 0, stream >> > (
            Y, X, outer_size, dim_size, inner_size);

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host function to launch arbitrary axis softmax (FP16 version)
     */
    void cuda_softmax_forward_general_fp16(
        half* Y,
        const half* X,
        int outer_size,
        int dim_size,
        int inner_size,
        cudaStream_t stream )
    {
        int total_slices = outer_size * inner_size;
        int block_size = 512;
        int grid_size = ceil_div( total_slices, block_size );

        softmax_forward_general_fp16_kernel << <grid_size, block_size, 0, stream >> > (
            Y, X, outer_size, dim_size, inner_size);

        cudaCheck( cudaGetLastError() );
    }

    // Wrappers for backwards compatibility
    template <typename TPrecision>
    void cuda_softmax_forward(
        TPrecision* Y,
        const TPrecision* X,
        int N,
        int C,
        cudaStream_t stream )
    {
        if constexpr ( std::is_same_v<TPrecision, float> ) {
            cuda_softmax_forward_fp32( Y, X, N, C, stream );
        }
        else if constexpr ( std::is_same_v<TPrecision, half> ) {
            cuda_softmax_forward_fp16( Y, X, N, C, stream );
        }
    }

    template <typename TPrecision>
    void cuda_softmax_forward_general(
        TPrecision* Y,
        const TPrecision* X,
        int outer_size,
        int dim_size,
        int inner_size,
        cudaStream_t stream )
    {
        if constexpr ( std::is_same_v<TPrecision, float> ) {
            cuda_softmax_forward_general_fp32( Y, X, outer_size, dim_size, inner_size, stream );
        }
        else if constexpr ( std::is_same_v<TPrecision, half> ) {
            cuda_softmax_forward_general_fp16( Y, X, outer_size, dim_size, inner_size, stream );
        }
    }

    // Explicit instantiations for float
    template void cuda_softmax_forward<float>(
        float* Y,
        const float* X,
        int N,
        int C,
        cudaStream_t stream );

    template void cuda_softmax_forward_general<float>(
        float* Y,
        const float* X,
        int outer_size,
        int dim_size,
        int inner_size,
        cudaStream_t stream );

    // Explicit instantiations for half
    template void cuda_softmax_forward<half>(
        half* Y,
        const half* X,
        int N,
        int C,
        cudaStream_t stream );

    template void cuda_softmax_forward_general<half>(
        half* Y,
        const half* X,
        int outer_size,
        int dim_size,
        int inner_size,
        cudaStream_t stream );
}
