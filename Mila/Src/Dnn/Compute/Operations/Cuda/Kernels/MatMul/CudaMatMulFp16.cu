/**
 * @file CudaMatMulFp16.cu
 * @brief CUDA implementation of optimized matrix multiplication operations in FP16.
 *
 * This file contains CUDA kernels and host functions for high-performance matrix
 * multiplication operations using FP16 precision, designed specifically for deep learning workloads.
 * The implementation utilizes shared memory and half2 vector operations for
 * improved memory bandwidth and computational efficiency.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "../CudaUtils.h"

namespace Mila::Dnn::Compute
{
    /**
     * @brief Scalar CUDA kernel for matrix multiplication with optional bias addition in FP16.
     *
     * This kernel computes the matrix multiplication Y = X * weight + bias without using vector operations,
     * making it suitable for matrices with dimensions that aren't multiples of 2. Each thread
     * computes one element of the output matrix.
     *
     * While less efficient than the vectorized version, this kernel works with any input dimensions
     * and serves as a fallback when the optimized kernel cannot be used.
     *
     * @param[out] Y Pointer to the output matrix of shape [B*T, OC]
     * @param[in] X Pointer to the input matrix of shape [B*T, C]
     * @param[in] weight Pointer to the weight matrix of shape [C, OC]
     * @param[in] bias Pointer to the bias vector (can be NULL if no bias is required)
     * @param[in] C Input feature dimension
     * @param[in] OC Output feature dimension
     */
    __global__ void matmul_forward_fp16_scalar_kernel(
        half* Y, const half* X, const half* weight, const half* bias, int C, int OC ) {

        // Calculate indices
        int row = blockIdx.x * blockDim.x + threadIdx.x; // B*T dimension
        int col = blockIdx.y * blockDim.y + threadIdx.y; // OC dimension

        // Check if we're within bounds
        if ( row < gridDim.x * blockDim.x && col < OC ) {
            float sum = 0.0f; // Use float for accumulation to maintain precision

            // Compute dot product of input row and weight column
            for ( int i = 0; i < C; ++i ) {
                float x_val = __half2float( X[ row * C + i ] );
                float w_val = __half2float( weight[ col * C + i ] );
                sum += x_val * w_val;
            }

            // Add bias if provided
            if ( bias != NULL ) {
                sum += __half2float( bias[ col ] );
            }

            // Write result back to half precision
            Y[ row * OC + col ] = __float2half( sum );
        }
    }

    /**
     * @brief Loads a half2 vector from memory.
     *
     * This device function loads 2 adjacent half values from memory as a single half2 vector,
     * which helps optimize memory bandwidth utilization.
     *
     * @param[in] address Pointer to the memory location to load from (must be aligned properly)
     * @return half2 The loaded vector containing 2 half values
     */
    __device__ half2 ld_vec_half2( const half* address ) {
        return *reinterpret_cast<const half2*>(address);
    }

    /**
     * @brief Stores a half2 vector to memory.
     *
     * This device function stores a half2 vector (2 adjacent half values) to memory,
     * which helps optimize memory bandwidth utilization.
     *
     * @param[out] address Pointer to the memory location to store to (must be aligned properly)
     * @param[in] val The half2 vector to store
     */
    __device__ void st_vec_half2( half* address, half2 val ) {
        *reinterpret_cast<half2*>(address) = val;
    }

    /**
     * @brief Optimized CUDA kernel for matrix multiplication with optional bias addition in FP16.
     *
     * This kernel computes the matrix multiplication Y = X * weight + bias, where:
     * - X is the input matrix of shape [128 * blockIdx.x, C]
     * - weight is the weight matrix of shape [C, 128 * blockIdx.y]
     * - Y is the output matrix of shape [128 * blockIdx.x, 128 * blockIdx.y]
     * - bias is an optional bias vector of length 128 * blockIdx.y
     *
     * The kernel utilizes shared memory to cache chunks of input and weight matrices,
     * and each thread computes an 8x8 block of output elements. The implementation uses
     * half2 vector operations to improve memory bandwidth utilization.
     *
     * @param[out] Y Pointer to the output matrix
     * @param[in] X Pointer to the input matrix
     * @param[in] weight Pointer to the weight matrix
     * @param[in] bias Pointer to the bias vector (can be NULL if no bias is required)
     * @param[in] C Input feature dimension
     * @param[in] OC Output feature dimension
     *
     * @note This kernel requires that C and OC are multiples of 2 for proper alignment.
     * @note The kernel is configured with __launch_bounds__(256) to optimize occupancy.
     */
    __global__ void __launch_bounds__( 16 * 16 ) matmul_forward_fp16_vectorized_kernel(
        half* Y, const half* X, const half* weight, const half* bias, int C, int OC ) {
        // Each thread handles 8x8 elements; each block 128 by 128 elements.
        int oc = 8 * (blockIdx.y * blockDim.y + threadIdx.y);

        // buffers to cache chunks of the input matrices - use half precision for shared memory
        __shared__ half lhs_s[ 128 ][ 32 ];
        __shared__ half rhs_s[ 128 ][ 32 ];

        // adjust our pointers for the current block
        X += 128 * blockIdx.x * C;
        weight += 128 * blockIdx.y * C;
        Y += 128 * blockIdx.x * OC + 128 * blockIdx.y;

        // Initialize output values array, preloading with bias if available
        float vals[ 8 ][ 8 ] = {}; // Use float for intermediate accumulation for better precision

        if ( bias != NULL ) {
            for ( int i = 0; i < 8; i++ ) {
                for ( int j = 0; j < 8; j += 2 ) {
                    half2 b = ld_vec_half2( bias + oc + j );
                    vals[ i ][ j + 0 ] = __half2float( b.x );
                    vals[ i ][ j + 1 ] = __half2float( b.y );
                }
            }
        }

        // Determine start index for shared memory loading
        int si_start = 4 * (16 * threadIdx.y + threadIdx.x);

        // Process the matrices in chunks of 32 columns
        for ( int so = 0; so < C; so += 32 ) {
            __syncthreads();
            // Collaborative loading of input and weight tiles into shared memory
            int xmod8 = threadIdx.x % 8;
            int xby8 = threadIdx.x / 8;
            int xo = 4 * xmod8;
            for ( int y = 2 * threadIdx.y + xby8; y < 128; y += 32 ) {
                // Load 4 half values at a time (2 half2 vectors)
                half2 lhs_val1 = ld_vec_half2( X + y * C + so + xo );
                half2 lhs_val2 = ld_vec_half2( X + y * C + so + xo + 2 );
                half2 rhs_val1 = ld_vec_half2( weight + y * C + so + xo );
                half2 rhs_val2 = ld_vec_half2( weight + y * C + so + xo + 2 );

                st_vec_half2( &lhs_s[ y ][ xo ], lhs_val1 );
                st_vec_half2( &lhs_s[ y ][ xo + 2 ], lhs_val2 );
                st_vec_half2( &rhs_s[ y ][ xo ], rhs_val1 );
                st_vec_half2( &rhs_s[ y ][ xo + 2 ], rhs_val2 );
            }
            __syncthreads();

            // Compute the matrix multiplication for this chunk
            for ( int si = si_start; si < si_start + 32; si += 2 ) {
                half2 rhs[ 8 ];
                for ( int u = 0; u < 8; ++u ) {
                    rhs[ u ] = ld_vec_half2( &rhs_s[ u + 8 * threadIdx.y ][ si % 32 ] );
                }

                for ( int ii = 0; ii < 8; ++ii ) {
                    half2 lhs = ld_vec_half2( &lhs_s[ ii + 8 * threadIdx.x ][ si % 32 ] );
                    for ( int ji = 0; ji < 8; ++ji ) {
                        // Convert to float for computation to avoid precision loss
                        float lhs_x = __half2float( lhs.x );
                        float lhs_y = __half2float( lhs.y );
                        float rhs_x = __half2float( rhs[ ji ].x );
                        float rhs_y = __half2float( rhs[ ji ].y );

                        vals[ ii ][ ji ] += lhs_x * rhs_x;
                        vals[ ii ][ ji ] += lhs_y * rhs_y;
                    }
                }
            }
        }

        // Write results back to global memory
        for ( int i = 0; i < 8; ++i ) {
            for ( int j = 0; j < 8; j += 2 ) {
                half2 result;
                result.x = __float2half( vals[ i ][ j + 0 ] );
                result.y = __float2half( vals[ i ][ j + 1 ] );
                st_vec_half2( Y + (8 * threadIdx.x + i) * OC + 8 * threadIdx.y + j, result );
            }
        }
    };

    /**
     * @brief Host function to perform matrix multiplication on the GPU with automatic kernel selection.
     *
     * This function chooses between the optimized vectorized kernel and the scalar fallback kernel
     * based on the dimensions of the matrices. The vectorized kernel is used when C and OC are both
     * multiples of 2; otherwise, the scalar kernel is used.
     *
     * @param[out] Y Pointer to the output matrix (B, T, OC)
     * @param[in] X Pointer to the input matrix (B, T, C)
     * @param[in] weight Pointer to the weight matrix (C, OC)
     * @param[in] bias Pointer to the bias vector (OC), can be NULL if no bias is applied
     * @param[in] B Batch size
     * @param[in] T Sequence length
     * @param[in] C Number of input channels
     * @param[in] OC Number of output channels
     * @param[in] stream CUDA stream to use for the kernel launch
     */
    void cuda_matmul_forward_fp16(
        half* Y, const half* X, 
        const half* weight, const half* bias,
        int B, int T, int C, int OC,
        cudaStream_t stream ) {

        int sqrt_block_size = 16;

        // Use scalar kernel if dimensions don't meet requirements for optimized kernel
        if ( C % 2 != 0 || OC % 2 != 0 ||
            B * T < 128 || OC < 128 ||  // Size requirements
            C < 32 ) {                   // Minimum size for chunking
            // Use a safe implementation for non-aligned dimensions
            dim3 gridDim( ceil_div( B * T, sqrt_block_size ), ceil_div( OC, sqrt_block_size ) );
            dim3 blockDim( sqrt_block_size, sqrt_block_size );

            // Use the scalar kernel that doesn't rely on half2
            matmul_forward_fp16_scalar_kernel << <gridDim, blockDim, 0, stream >> > (Y, X, weight, bias, C, OC);
        }
        else {
            // Use optimized kernel for aligned dimensions
            dim3 gridDim( ceil_div( B * T, 8 * sqrt_block_size ), ceil_div( OC, 8 * sqrt_block_size ) );
            dim3 blockDim( sqrt_block_size, sqrt_block_size );

            matmul_forward_fp16_vectorized_kernel << <gridDim, blockDim, 0, stream >> > (Y, X, weight, bias, C, OC);
        }

        cudaCheck( cudaGetLastError() );
    }
}