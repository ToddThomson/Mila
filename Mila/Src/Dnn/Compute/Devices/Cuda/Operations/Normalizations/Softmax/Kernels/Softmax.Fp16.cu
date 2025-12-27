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
     * @brief CUDA kernel for softmax with arbitrary axis support (FP16 version)
     *
     * Uses float for intermediate accumulation for improved precision, converts
     * to/from half when reading/writing global memory.
     */
    __global__ void softmax_forward_general_fp16_kernel(
        half* Y, const half* X,
        int outer_size, int dim_size, int inner_size )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < outer_size * inner_size ) {
            int outer_idx = idx / inner_size;
            int inner_idx = idx % inner_size;

            float max_val = -INFINITY;

            for ( int d = 0; d < dim_size; d++ ) {
                int index = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
                float val = __half2float( X[ index ] );
                if ( val > max_val ) {
                    max_val = val;
                }
            }

            float sum = 0.0f;

            for ( int d = 0; d < dim_size; d++ ) {
                int index = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
                float exp_val = expf( __half2float( X[ index ] ) - max_val );
                Y[ index ] = __float2half( exp_val );
                sum += exp_val;
            }

            float inv_sum = 1.0f / sum;

            for ( int d = 0; d < dim_size; d++ ) {
                int index = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
                Y[ index ] = __float2half( __half2float( Y[ index ] ) * inv_sum );
            }
        }
    }

    /**
     * @brief CUDA kernel for softmax forward pass with FP16 precision (row-contiguous)
     *
     * Computes softmax per row (N x C). Uses float accumulation and converts to half.
     */
    __global__ void softmax_forward_fp16_kernel( half* Y, const half* X, int N, int C )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if ( i < N ) {
            const half* X_row = X + i * C;
            half* Y_row = Y + i * C;

            float maxval = -INFINITY;

            for ( int j = 0; j < C; j++ ) {
                float v = __half2float( X_row[ j ] );
                if ( v > maxval ) {
                    maxval = v;
                }
            }

            float sum = 0.0f;

            for ( int j = 0; j < C; j++ ) {
                float v = expf( __half2float( X_row[ j ] ) - maxval );
                Y_row[ j ] = __float2half( v );
                sum += v;
            }

            float inv_sum = 1.0f / sum;

            for ( int j = 0; j < C; j++ ) {
                Y_row[ j ] = __float2half( __half2float( Y_row[ j ] ) * inv_sum );
            }
        }
    }

    /**
     * @brief CUDA kernel for backward softmax on contiguous last-axis (FP16)
     *
     * Computes dX = Y * (dY - sum_j dY_j * Y_j) using float accumulation,
     * then writes results as half.
     */
    __global__ void softmax_backward_fp16_kernel(
        half* dX, const half* dY, const half* Y, int N, int C )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if ( i < N ) {
            const half* y_row_h = Y + i * C;
            const half* dy_row_h = dY + i * C;
            half* dx_row_h = dX + i * C;

            float dot = 0.0f;

            for ( int j = 0; j < C; j++ ) {
                float dy = __half2float( dy_row_h[ j ] );
                float y  = __half2float( y_row_h[ j ] );
                dot += dy * y;
            }

            for ( int j = 0; j < C; j++ ) {
                float dy = __half2float( dy_row_h[ j ] );
                float y  = __half2float( y_row_h[ j ] );
                float dx = y * ( dy - dot );
                dx_row_h[ j ] = __float2half( dx );
            }
        }
    }

    /**
     * @brief CUDA kernel for backward softmax with arbitrary axis (FP16)
     *
     * Uses float accumulation per slice and converts results to half.
     */
    __global__ void softmax_backward_general_fp16_kernel(
        half* dX, const half* dY, const half* Y,
        int outer_size, int dim_size, int inner_size )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < outer_size * inner_size ) {
            int outer_idx = idx / inner_size;
            int inner_idx = idx % inner_size;

            float dot = 0.0f;

            for ( int d = 0; d < dim_size; d++ ) {
                int index = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
                dot += __half2float( dY[ index ] ) * __half2float( Y[ index ] );
            }

            for ( int d = 0; d < dim_size; d++ ) {
                int index = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
                float y = __half2float( Y[ index ] );
                float dy = __half2float( dY[ index ] );
                float dx = y * ( dy - dot );
                dX[ index ] = __float2half( dx );
            }
        }
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

        softmax_forward_fp16_kernel<<<grid_size, block_size, 0, stream>>>(Y, X, N, C);

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

        softmax_forward_general_fp16_kernel<<<grid_size, block_size, 0, stream>>>(Y, X, outer_size, dim_size, inner_size);

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host function to launch softmax backward pass (FP16, contiguous last-axis)
     */
    void cuda_softmax_backward_fp16(
        half* dX,
        const half* dY,
        const half* Y,
        int N,
        int C,
        cudaStream_t stream )
    {
        const int block_size = 512;
        const int grid_size = ceil_div( N, block_size );

        softmax_backward_fp16_kernel<<<grid_size, block_size, 0, stream>>>(dX, dY, Y, N, C);

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host function to launch softmax backward pass (FP16, arbitrary axis)
     */
    void cuda_softmax_backward_general_fp16(
        half* dX,
        const half* dY,
        const half* Y,
        int outer_size,
        int dim_size,
        int inner_size,
        cudaStream_t stream )
    {
        int total_slices = outer_size * inner_size;
        int block_size = 512;
        int grid_size = ceil_div( total_slices, block_size );

        softmax_backward_general_fp16_kernel<<<grid_size, block_size, 0, stream>>>(dX, dY, Y, outer_size, dim_size, inner_size);

        cudaCheck( cudaGetLastError() );
    }

    // Explicit instantiations for half templated wrappers (optional but keeps linkage consistent)
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

    template void cuda_softmax_backward<half>(
        half* dX,
        const half* dY,
        const half* Y,
        int N,
        int C,
        cudaStream_t stream );

    template void cuda_softmax_backward_general<half>(
        half* dX,
        const half* dY,
        const half* Y,
        int outer_size,
        int dim_size,
        int inner_size,
        cudaStream_t stream );
}