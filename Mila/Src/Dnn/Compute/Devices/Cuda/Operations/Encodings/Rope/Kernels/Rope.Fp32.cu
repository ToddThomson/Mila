#define _USE_MATH_DEFINES
#include <math.h>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute::Cuda::Rope
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
     * @brief CUDA kernel for RoPE forward pass using float4 vectorization
     *
     * This kernel adds token embeddings and positional embeddings. Kept vectorized
     * layout for memory efficiency. Caller must ensure C is divisible by 4.
     *
     * @param Y Output tensor [B, T, C] (as float4)
     * @param X Input token indices [B, T]
     * @param Wte Token embedding weights [vocab_size, C] (as float4)
     * @param Wpe Position embedding weights [max_seq_len, C] (as float4)
     * @param B Batch size
     * @param T Sequence length
     * @param C Hidden dimension size (must be divisible by 4)
     */
    __global__ void rope_forward_fp32_kernel(
        float4* Y, const int* X,
        const float4* Wte, const float4* Wpe,
        int B, int T, int C )
    {
        int C4 = C / 4;
        int bt = blockIdx.x;      // Token index [0, B*T)
        int c4 = threadIdx.x;     // Channel index [0, C4)

        if ( bt < B * T ) {
            int t = bt % T;
            int ix = X[ bt ];

            // Perfect coalescing: consecutive threads access consecutive memory
            Y[ bt * C4 + c4 ] = add_float4( Wte[ ix * C4 + c4 ], Wpe[ t * C4 + c4 ] );
        }
    }

    __global__ void rope_forward_fp32_kernel_v2(
        float4* __restrict__ Y,
        const int* __restrict__ X,
        const float4* __restrict__ Wte,
        const float4* __restrict__ Wpe,
        int B, int T, int C )
    {
        int C4 = C / 4;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = B * T * C4;

        if ( idx < total ) {
            int bt = idx / C4;
            int c4 = idx % C4;
            int t = bt % T;
            int ix = X[ bt ];

            // Single 128-bit load + 128-bit load + 128-bit store
            float4 a = Wte[ ix * C4 + c4 ];
            float4 b = Wpe[ t * C4 + c4 ];
            Y[ bt * C4 + c4 ] = add_float4( a, b );
        }
    }

    /**
     * @brief CUDA kernel for RoPE backward pass using float4 vectorization
     *
     * Accumulates gradients from output tensor to token and position embedding weights.
     * Uses atomicAdd for thread-safe gradient accumulation since multiple positions
     * may reference the same token embedding.
     *
     * @param dWte Gradient for token embedding weights [vocab_size, C] (as float4)
     * @param dWpe Gradient for position embedding weights [max_seq_len, C] (as float4)
     * @param dY Gradient from output [B, T, C] (as float4)
     * @param X Input token indices [B, T]
     * @param B Batch size
     * @param T Sequence length
     * @param C Hidden dimension size (must be divisible by 4)
     */
    __global__ void rope_backward_fp32_kernel(
        float4* dWte, float4* dWpe,
        const float4* dY, const int* X,
        int B, int T, int C )
    {
        int C4 = C / 4;
        int bt = blockIdx.x;      // Token index [0, B*T)
        int c4 = threadIdx.x;     // Channel index [0, C4)

        if ( bt < B * T ) {
            int t = bt % T;
            int ix = X[ bt ];

            float4 grad = dY[ bt * C4 + c4 ];

            atomicAdd( &dWte[ ix * C4 + c4 ].x, grad.x );
            atomicAdd( &dWte[ ix * C4 + c4 ].y, grad.y );
            atomicAdd( &dWte[ ix * C4 + c4 ].z, grad.z );
            atomicAdd( &dWte[ ix * C4 + c4 ].w, grad.w );

            atomicAdd( &dWpe[ t * C4 + c4 ].x, grad.x );
            atomicAdd( &dWpe[ t * C4 + c4 ].y, grad.y );
            atomicAdd( &dWpe[ t * C4 + c4 ].z, grad.z );
            atomicAdd( &dWpe[ t * C4 + c4 ].w, grad.w );
        }
    }

    // ========================================================================
    // Host functions
    // ========================================================================

    /**
     * @brief Host function to launch RoPE forward pass with full precision (FP32)
     *
     * Combines token embeddings and positional embeddings for transformer input.
     *
     * @param Y Output tensor [B, T, C]
     * @param X Input token indices [B, T]
     * @param wte Token embedding weights [vocab_size, C]
     * @param wpe Position embedding weights [max_seq_len, C]
     * @param B Batch size
     * @param T Sequence length
     * @param C Hidden dimension size (must be divisible by 4)
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_rope_forward_fp32(
        float* Y,
        const int* X,
        const float* wte, const float* wpe,
        int B, int T, int C,
        cudaStream_t stream )
    {
        // C must be divisible by 4 for float4 vectorization
        assert( C % 4 == 0 );

        int C4 = C / 4;

        dim3 grid( B * T );
        dim3 block( C4 );

        rope_forward_fp32_kernel << <grid, block, 0, stream >> > (
            reinterpret_cast<float4*>(Y), X,
            reinterpret_cast<const float4*>(wte),
            reinterpret_cast<const float4*>(wpe),
            B, T, C);

        cudaCheck( cudaGetLastError() );

    }

    /**
     * @brief Host function to launch RoPE backward pass with full precision (FP32)
     *
     * Computes gradients for token and position embeddings from output gradient.
     *
     * @param dWte Gradient for token embedding weights [vocab_size, C]
     * @param dWpe Gradient for position embedding weights [max_seq_len, C]
     * @param dY Gradient from output [B, T, C]
     * @param X Input token indices [B, T]
     * @param B Batch size
     * @param T Sequence length
     * @param C Hidden dimension size (must be divisible by 4)
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_rope_backward_fp32(
        float* wte_grad, float* wpe_grad,
        const float* dY,
        const int* X,
        int B, int T, int C,
        cudaStream_t stream )
    {
        assert( C % 4 == 0 );

        int C4 = C / 4;

        dim3 grid( B * T );
        dim3 block( C4 );

        rope_backward_fp32_kernel << <grid, block, 0, stream >> > (
            reinterpret_cast<float4*>(wte_grad),
            reinterpret_cast<float4*>(wpe_grad),
            reinterpret_cast<const float4*>(dY),
            X,
            B, T, C);

        cudaCheck( cudaGetLastError() );

    }
}