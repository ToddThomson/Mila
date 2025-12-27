#define _USE_MATH_DEFINES
#include <math.h>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include <cmath>
#include "Softmax.cuh"

namespace Mila::Dnn::Compute::Cuda::Softmax
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

            float max_val = -INFINITY;

            for ( int d = 0; d < dim_size; d++ ) {
                int index = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
                float val = X[ index ];
                if ( val > max_val ) {
                    max_val = val;
                }
            }

            float sum = 0.0f;

            for ( int d = 0; d < dim_size; d++ ) {
                int index = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
                Y[ index ] = expf( X[ index ] - max_val );
                sum += Y[ index ];
            }

            float inv_sum = 1.0f / sum;

            for ( int d = 0; d < dim_size; d++ ) {
                int index = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
                Y[ index ] *= inv_sum;
            }
        }
    }

    /**
     * @brief CUDA kernel for softmax forward pass with FP32 precision
     */
    __global__ void softmax_forward_fp32_kernel( float* Y, const float* X, int N, int C )
    {
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
     * @brief CUDA kernel for backward softmax on contiguous last-axis (FP32)
     *
     * dX_i = Y_i * (dY_i - sum_j dY_j * Y_j)
     */
    __global__ void softmax_backward_fp32_kernel(
        float* dX, const float* dY, const float* Y, int N, int C )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if ( i < N ) {
            const float* y_row = Y + i * C;
            const float* dy_row = dY + i * C;
            float* dx_row = dX + i * C;

            float dot = 0.0f;

            for ( int j = 0; j < C; j++ ) {
                dot += dy_row[ j ] * y_row[ j ];
            }

            for ( int j = 0; j < C; j++ ) {
                dx_row[ j ] = y_row[ j ] * ( dy_row[ j ] - dot );
            }
        }
    }

    /**
     * @brief CUDA kernel for backward softmax with arbitrary axis (FP32)
     */
    __global__ void softmax_backward_general_fp32_kernel(
        float* dX, const float* dY, const float* Y,
        int outer_size, int dim_size, int inner_size )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < outer_size * inner_size ) {
            int outer_idx = idx / inner_size;
            int inner_idx = idx % inner_size;

            float dot = 0.0f;

            for ( int d = 0; d < dim_size; d++ ) {
                int index = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
                dot += dY[ index ] * Y[ index ];
            }

            for ( int d = 0; d < dim_size; d++ ) {
                int index = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
                dX[ index ] = Y[ index ] * ( dY[ index ] - dot );
            }
        }
    }

    void cuda_softmax_forward_fp32(
        float* Y,
        const float* X,
        int N,
        int C,
        cudaStream_t stream )
    {
        const int block_size = 512;
        const int grid_size = ceil_div( N, block_size );

        softmax_forward_fp32_kernel<<<grid_size, block_size, 0, stream>>>(Y, X, N, C);

        cudaCheck( cudaGetLastError() );
    }

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

        softmax_forward_general_fp32_kernel<<<grid_size, block_size, 0, stream>>>(Y, X, outer_size, dim_size, inner_size);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_softmax_backward_fp32(
        float* dX,
        const float* dY,
        const float* Y,
        int N,
        int C,
        cudaStream_t stream )
    {
        const int block_size = 512;
        const int grid_size = ceil_div( N, block_size );

        softmax_backward_fp32_kernel<<<grid_size, block_size, 0, stream>>>(dX, dY, Y, N, C);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_softmax_backward_general_fp32(
        float* dX,
        const float* dY,
        const float* Y,
        int outer_size,
        int dim_size,
        int inner_size,
        cudaStream_t stream )
    {
        int total_slices = outer_size * inner_size;
        int block_size = 512;
        int grid_size = ceil_div( total_slices, block_size );

        softmax_backward_general_fp32_kernel<<<grid_size, block_size, 0, stream>>>(dX, dY, Y, outer_size, dim_size, inner_size);

        cudaCheck( cudaGetLastError() );
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

    // Backward explicit instantiations for float
    template void cuda_softmax_backward<float>(
        float* dX,
        const float* dY,
        const float* Y,
        int N,
        int C,
        cudaStream_t stream );

    template void cuda_softmax_backward_general<float>(
        float* dX,
        const float* dY,
        const float* Y,
        int outer_size,
        int dim_size,
        int inner_size,
        cudaStream_t stream );
}