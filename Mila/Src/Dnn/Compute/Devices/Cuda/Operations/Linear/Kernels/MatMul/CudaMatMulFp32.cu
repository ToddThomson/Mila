/**
 * @file CudaMatMulFp32.cu
 * @brief CUDA implementation of optimized matrix multiplication operations.
 *
 * This file contains CUDA kernels and host functions for high-performance matrix
 * multiplication operations, designed specifically for deep learning workloads.
 * The implementation utilizes shared memory and float4 vector operations for
 * improved memory bandwidth and computational efficiency.
 */

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute::Cuda::Linear
{
    /**
     * @brief Scalar CUDA kernel for matrix multiplication with optional bias addition.
     *
     * This kernel computes the matrix multiplication Y = X * weight + bias without using vector operations,
     * making it suitable for matrices with dimensions that aren't multiples of 4. Each thread
     * computes one element of the output matrix.
     *
     * @param[out] Y Pointer to the output matrix of shape [B*T, OC]
     * @param[in] X Pointer to the input matrix of shape [B*T, C]
     * @param[in] weight Pointer to the weight matrix of shape [OC, C] (packed as col-major per forward)
     * @param[in] bias Pointer to the bias vector (can be NULL if no bias is required)
     * @param[in] C Input feature dimension
     * @param[in] OC Output feature dimension
     */
    __global__ void matmul_forward_fp32_scalar_kernel(
        float* Y, const float* X, const float* weight, const float* bias, int C, int OC )
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x; // flattened batch index (B*T)
        int col = blockIdx.y * blockDim.y + threadIdx.y; // output channel index

        if ( row < gridDim.x * blockDim.x && col < OC )
        {
            float sum = 0.0f;

            for ( int i = 0; i < C; ++i )
            {
                // weight layout: weight[col * C + i]
                sum += X[ row * C + i ] * weight[ col * C + i ];
            }

            if ( bias != NULL )
            {
                sum += bias[ col ];
            }

            Y[ row * OC + col ] = sum;
        }
    }

    /**
     * @brief Loads a float4 vector from memory.
     */
    __device__ float4 ld_vec( const float* address )
    {
        return *reinterpret_cast<const float4*>( address );
    }

    /**
     * @brief Stores a float4 vector to memory.
     */
    __device__ void st_vec( float* address, float4 val )
    {
        *reinterpret_cast<float4*>( address ) = val;
    }

    /**
     * @brief Optimized CUDA kernel for matrix multiplication with optional bias addition.
     *
     * Vectorized/shared-memory kernel optimized for large aligned shapes.
     */
    __global__ void __launch_bounds__( 16 * 16 ) matmul_forward_fp32_vectorized_kernel(
        float* Y, const float* X, const float* weight, const float* bias, int C, int OC )
    {
        int oc = 8 * ( blockIdx.y * blockDim.y + threadIdx.y );

        __shared__ float lhs_s[ 128 ][ 32 ];
        __shared__ float rhs_s[ 128 ][ 32 ];

        X += 128 * blockIdx.x * C;
        weight += 128 * blockIdx.y * C;
        Y += 128 * blockIdx.x * OC + 128 * blockIdx.y;

        float vals[ 8 ][ 8 ] = {};
        if ( bias != NULL )
        {
            for ( int i = 0; i < 8; i++ )
            {
                for ( int j = 0; j < 8; j += 4 )
                {
                    float4 b = ld_vec( bias + oc + j );
                    vals[ i ][ j + 0 ] = b.x;
                    vals[ i ][ j + 1 ] = b.y;
                    vals[ i ][ j + 2 ] = b.z;
                    vals[ i ][ j + 3 ] = b.w;
                }
            }
        }

        int si_start = 4 * ( 16 * threadIdx.y + threadIdx.x );

        for ( int so = 0; so < C; so += 32 )
        {
            __syncthreads();

            int xmod8 = threadIdx.x % 8;
            int xby8 = threadIdx.x / 8;
            int xo = 4 * xmod8;

            for ( int y = 2 * threadIdx.y + xby8; y < 128; y += 32 )
            {
                st_vec( &lhs_s[ y ][ xo ], ld_vec( X + y * C + so + xo ) );
                st_vec( &rhs_s[ y ][ xo ], ld_vec( weight + y * C + so + xo ) );
            }

            __syncthreads();

            for ( int si = si_start; si < si_start + 32; si += 4 )
            {
                float4 rhs[ 8 ];
                for ( int u = 0; u < 8; ++u )
                {
                    rhs[ u ] = ld_vec( &rhs_s[ u + 8 * threadIdx.y ][ si % 32 ] );
                }

                for ( int ii = 0; ii < 8; ++ii )
                {
                    float4 lhs = ld_vec( &lhs_s[ ii + 8 * threadIdx.x ][ si % 32 ] );

                    for ( int ji = 0; ji < 8; ++ji )
                    {
                        vals[ ii ][ ji ] += lhs.x * rhs[ ji ].x;
                        vals[ ii ][ ji ] += lhs.y * rhs[ ji ].y;
                        vals[ ii ][ ji ] += lhs.z * rhs[ ji ].z;
                        vals[ ii ][ ji ] += lhs.w * rhs[ ji ].w;
                    }
                }
            }
        }

        for ( int i = 0; i < 8; ++i )
        {
            for ( int j = 0; j < 8; j += 4 )
            {
                float4 result;
                result.x = vals[ i ][ j + 0 ];
                result.y = vals[ i ][ j + 1 ];
                result.z = vals[ i ][ j + 2 ];
                result.w = vals[ i ][ j + 3 ];
                st_vec( Y + ( 8 * threadIdx.x + i ) * OC + 8 * threadIdx.y + j, result );
            }
        }
    };

    // ========================================================================
    // Host functions - forward
    // ========================================================================

    void cuda_matmul_forward_fp32(
        float* Y, const float* X, const float* weight, const float* bias,
        int outer_size, int C, int OC,
        cudaStream_t stream )
    {
        int sqrt_block_size = 16;

        if ( C % 4 != 0 || OC % 4 != 0 || outer_size < 128 || OC < 128 || C < 32 )
        {
            dim3 gridDim( ceil_div( outer_size, sqrt_block_size ), ceil_div( OC, sqrt_block_size ) );
            dim3 blockDim( sqrt_block_size, sqrt_block_size );

            matmul_forward_fp32_scalar_kernel<<< gridDim, blockDim, 0, stream >>>( Y, X, weight, bias, C, OC );
        }
        else
        {
            dim3 gridDim( ceil_div( outer_size, 8 * sqrt_block_size ), ceil_div( OC, 8 * sqrt_block_size ) );
            dim3 blockDim( sqrt_block_size, sqrt_block_size );

            matmul_forward_fp32_vectorized_kernel<<< gridDim, blockDim, 0, stream >>>( Y, X, weight, bias, C, OC );
        }

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Scalar CUDA kernel for backward of matrix multiplication.
     *
     * Computes contributions for:
     *   - dX  (accumulated into `dX`) : dX[row, i] += sum_j dY[row, j] * weight[j, i]
     *   - dW  (accumulated into `dW`) : dW[j, i] += dY[row, j] * X[row, i]
     *   - db  (accumulated into `dbias`) : dbias[j] += dY[row, j]
     *
     * This scalar kernel is a simple, correct fallback for shapes where the
     * vectorized kernel cannot be used. It uses atomicAdd for accumulation to
     * support concurrent updates from multiple threads.
     */
    __global__ void matmul_backward_fp32_scalar_kernel(
        float* dX,
        float* dW,
        float* dbias,
        const float* dY,
        const float* X,
        const float* weight,
        int outer_size,
        int C,
        int OC )
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x; // flattened batch index
        int col = blockIdx.y * blockDim.y + threadIdx.y; // output channel index

        if ( row >= outer_size || col >= OC )
        {
            return;
        }

        float dy = dY[ row * OC + col ];

        if ( dbias != nullptr )
        {
            atomicAdd( &dbias[ col ], dy );
        }

        // Accumulate dW and dX contributions
        for ( int i = 0; i < C; ++i )
        {
            float x = X[ row * C + i ];
            float w = weight[ col * C + i ];

            if ( dW != nullptr )
            {
                atomicAdd( &dW[ col * C + i ], dy * x );
            }

            if ( dX != nullptr )
            {
                atomicAdd( &dX[ row * C + i ], dy * w );
            }
        }
    }

    /**
     * @brief Host launcher for backward pass that selects scalar kernel when needed.
     *
     * Mirrors the forward host selection strategy: when shapes are not
     * aligned for the vectorized implementation, use the scalar fallback.
     */
    void cuda_matmul_backward_fp32(
        float* dX,
        float* dW,
        float* dbias,
        const float* dY,
        const float* X,
        const float* weight,
        int outer_size,
        int C,
        int OC,
        cudaStream_t stream )
    {
        int sqrt_block_size = 16;

        // Use the same safe conditions as forward scalar fallback
        if ( C % 4 != 0 || OC % 4 != 0 || outer_size < 128 || OC < 128 || C < 32 )
        {
            dim3 gridDim( ceil_div( outer_size, sqrt_block_size ), ceil_div( OC, sqrt_block_size ) );
            dim3 blockDim( sqrt_block_size, sqrt_block_size );

            matmul_backward_fp32_scalar_kernel<<< gridDim, blockDim, 0, stream >>>(
                dX, dW, dbias, dY, X, weight, outer_size, C, OC );
        }
        else
        {
            // No optimized backward kernel implemented here yet; use scalar fallback for correctness.
            dim3 gridDim( ceil_div( outer_size, sqrt_block_size ), ceil_div( OC, sqrt_block_size ) );
            dim3 blockDim( sqrt_block_size, sqrt_block_size );

            matmul_backward_fp32_scalar_kernel<<< gridDim, blockDim, 0, stream >>>(
                dX, dW, dbias, dY, X, weight, outer_size, C, OC );
        }

        cudaCheck( cudaGetLastError() );
    }
}