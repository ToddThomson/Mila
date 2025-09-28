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
     * @brief Adds two float4 vectors element-wise
     *
     * @param a First float4 vector
     * @param b Second float4 vector
     * @return float4 Result of component-wise addition
     */
    __device__ inline float4 add_float4( const float4& a, const float4& b ) {
        return make_float4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
    }

    /**
     * @brief CUDA kernel for encoder forward pass using float4 vectorization
     *
     * This kernel uses float4 operations for memory efficiency, which leads to
     * using 128-bit LDG/STG instructions in SASS, very helpful in memory-bound
     * kernels like encoder_forward.
     *
     * @param Y Output tensor [B, T, C]
     * @param X Input indices [B, T]
     * @param Wte Token embedding weights [vocab_size, C]
     * @param Wpe Position embedding weights [max_seq_len, C]
     * @param B Batch size
     * @param T Sequence length
     * @param C Hidden dimension size (must be divisible by 4)
     */
    __global__ void encoder_forward_fp32_kernel( float4* Y,
        const int* X, const float4* Wte, const float4* Wpe,
        int B, int T, int C ) {
        int C4 = C / 4;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int N = B * T * C4;
        if ( idx < N ) {
            int bt = idx / C4;
            int b = bt / T;
            int t = bt % T;
            int c4 = idx % C4;
            int ix = X[ b * T + t ];
            Y[ b * T * C4 + t * C4 + c4 ] = add_float4( Wte[ ix * C4 + c4 ], Wpe[ t * C4 + c4 ] );
        }
    }

    /**
     * @brief Adds two half2 vectors element-wise
     *
     * @param a First half2 vector
     * @param b Second half2 vector
     * @return half2 Result of component-wise addition
     */
    __device__ inline half2 add_half2( const half2& a, const half2& b ) {
        return __hadd2( a, b );
    }

    /**
     * @brief CUDA kernel for encoder forward pass using half-precision (FP16)
     *
     * This kernel uses half2 vector operations for better throughput on
     * hardware with good FP16 support.
     *
     * @param Y Output tensor [B, T, C] in half precision
     * @param X Input indices [B, T]
     * @param Wte Token embedding weights [vocab_size, C] in half precision
     * @param Wpe Position embedding weights [max_seq_len, C] in half precision
     * @param B Batch size
     * @param T Sequence length
     * @param C Hidden dimension size (must be divisible by 2)
     */
    __global__ void encoder_forward_fp16_vectorized_kernel( half2* Y,
        const int* X, const half2* Wte, const half2* Wpe,
        int B, int T, int C ) {

        int C2 = C / 2; // We're using half2 which holds 2 half values
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int N = B * T * C2;

        if ( idx < N ) {
            int bt = idx / C2;
            int b = bt / T;
            int t = bt % T;
            int c2 = idx % C2;
            int ix = X[ b * T + t ];
            Y[ b * T * C2 + t * C2 + c2 ] = add_half2( Wte[ ix * C2 + c2 ], Wpe[ t * C2 + c2 ] );
        }
    }

    /**
     * @brief Host function to launch encoder forward pass with full precision (FP32)
     *
     * Combines token embeddings and positional embeddings for transformer encoder input.
     *
     * @param Y Output tensor [B, T, C]
     * @param X Input token indices [B, T]
     * @param Wte Token embedding weights [vocab_size, C]
     * @param Wpe Position embedding weights [max_seq_len, C]
     * @param B Batch size
     * @param T Sequence length
     * @param C Hidden dimension size (must be divisible by 4)
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_encoder_forward_fp32(
        float* Y,
        const int* X,
        const float* Wte, const float* Wpe,
        int B, int T, int C,
        cudaStream_t stream ) {

        // FP32 implementation
        assert( C % 4 == 0 );
        const int block_size = 512;
        const int N = B * T * C;
        const int grid_size = ceil_div( N / 4, block_size );

        encoder_forward_fp32_kernel << <grid_size, block_size, 0, stream >> > (
            reinterpret_cast<float4*>(Y),
            X,
            reinterpret_cast<const float4*>(Wte),
            reinterpret_cast<const float4*>(Wpe),
            B, T, C);

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host function to launch encoder forward pass with half precision (FP16)
     *
     * Combines token embeddings and positional embeddings for transformer encoder input
     * using FP16 for better performance on compatible hardware.
     *
     * @param Y Output tensor [B, T, C] in half precision
     * @param X Input token indices [B, T]
     * @param Wte Token embedding weights [vocab_size, C] in half precision
     * @param Wpe Position embedding weights [max_seq_len, C] in half precision
     * @param B Batch size
     * @param T Sequence length
     * @param C Hidden dimension size (must be divisible by 2)
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_encoder_forward_fp16(
        half* Y,
        const int* X,
        const half* Wte, const half* Wpe,
        int B, int T, int C,
        cudaStream_t stream ) {

        // FP16 implementation
        assert( C % 2 == 0 );
        const int block_size = 512;
        const int N = B * T * C;
        const int grid_size = ceil_div( N / 2, block_size );

        encoder_forward_fp16_vectorized_kernel << <grid_size, block_size, 0, stream >> > (
            reinterpret_cast<half2*>(Y),
            X,
            reinterpret_cast<const half2*>(Wte),
            reinterpret_cast<const half2*>(Wpe),
            B, T, C);

        cudaCheck( cudaGetLastError() );
    }
}