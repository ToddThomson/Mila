#define _USE_MATH_DEFINES
#include <math.h>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute::Cuda::Lpe
{
    /**
     * @brief Element-wise addition of two float4 vectors.
     */
    __device__ inline float4 add_float4( const float4& a, const float4& b ) {
        return make_float4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
    }

    /**
     * @brief Full-sequence encoder forward kernel using float4 vectorization.
     *
     * For each (batch, position) pair: Y[b,t,:] = wte[X[b,t],:] + wpe[t,:].
     * float4 loads/stores emit 128-bit LDG/STG instructions, maximizing
     * memory throughput in this bandwidth-bound kernel.
     *
     * @param Y   Output embeddings [B, T, C/4] (float4 view).
     * @param X   Input token indices [B, T].
     * @param Wte Token embedding table [vocab_size, C/4].
     * @param Wpe Positional embedding table [max_seq_len, C/4].
     * @param B   Batch size.
     * @param T   Sequence length.
     * @param C   Embedding dimension (must be divisible by 4).
     */
    __global__ void encoder_forward_fp32_kernel(
        float4* Y, const int* X,
        const float4* Wte, const float4* Wpe,
        int B, int T, int C )
    {
        int C4 = C / 4;
        int bt = blockIdx.x;
        int c4 = threadIdx.x;

        if ( bt < B * T ) {
            int t  = bt % T;
            int ix = X[ bt ];
            Y[ bt * C4 + c4 ] = add_float4( Wte[ ix * C4 + c4 ], Wpe[ t * C4 + c4 ] );
        }
    }

    __global__ void encoder_forward_fp32_kernel_v2(
        float4* __restrict__ Y,
        const int* __restrict__ X,
        const float4* __restrict__ Wte,
        const float4* __restrict__ Wpe,
        int B, int T, int C )
    {
        int C4  = C / 4;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * T * C4 ) {
            int bt = idx / C4;
            int c4 = idx % C4;
            int t  = bt % T;
            int ix = X[ bt ];
            Y[ bt * C4 + c4 ] = add_float4( Wte[ ix * C4 + c4 ], Wpe[ t * C4 + c4 ] );
        }
    }

    /**
     * @brief Single-token decode kernel using float4 vectorization.
     *
     * Inference-only. Computes Y[b,:] = wte[X[b],:] + wpe[position,:] for each
     * batch element b. Unlike the forward kernel, the wpe row is fixed at `position`
     * for all threads — every batch element receives the same positional embedding,
     * which is correct for KV-cache autoregressive decode where all batch elements
     * share the same current sequence position.
     *
     * The fixed wpe row is broadcast across all b: the Wpe[position * C4 + c4] load
     * is L1-cached after the first access, so no additional memory traffic is incurred
     * relative to a single-token forward pass.
     *
     * @param Y        Output embeddings [B, C/4] (float4 view).
     * @param X        Token indices [B] — one index per batch element.
     * @param Wte      Token embedding table [vocab_size, C/4].
     * @param Wpe      Positional embedding table [max_seq_len, C/4].
     * @param position Absolute sequence position selecting the wpe row (0-based).
     * @param B        Batch size.
     * @param C        Embedding dimension (must be divisible by 4).
     */
    __global__ void encoder_decode_fp32_kernel(
        float4* __restrict__ Y,
        const int* __restrict__ X,
        const float4* __restrict__ Wte,
        const float4* __restrict__ Wpe,
        int position, int B, int C )
    {
        int C4  = C / 4;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * C4 ) {
            int b  = idx / C4;
            int c4 = idx % C4;
            int ix = X[ b ];

            float4 tok = Wte[ ix * C4 + c4 ];
            float4 pos = Wpe[ position * C4 + c4 ];
            Y[ b * C4 + c4 ] = add_float4( tok, pos );
        }
    }

    /**
     * @brief Full-sequence encoder backward kernel using float4 vectorization.
     *
     * Accumulates gradients from the output into wte and wpe. atomicAdd is required
     * because multiple sequence positions can reference the same token embedding row.
     *
     * @param dWte Gradient accumulation buffer for wte [vocab_size, C/4].
     * @param dWpe Gradient accumulation buffer for wpe [max_seq_len, C/4].
     * @param dY   Upstream gradient [B, T, C/4].
     * @param X    Token indices used in the forward pass [B, T].
     * @param B    Batch size.
     * @param T    Sequence length.
     * @param C    Embedding dimension (must be divisible by 4).
     */
    __global__ void encoder_backward_fp32_kernel(
        float4* dWte, float4* dWpe,
        const float4* dY, const int* X,
        int B, int T, int C )
    {
        int C4 = C / 4;
        int bt = blockIdx.x;
        int c4 = threadIdx.x;

        if ( bt < B * T ) {
            int t  = bt % T;
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
    // Host launchers
    // ========================================================================

    /**
     * @brief Launch the full-sequence FP32 encoder forward pass.
     *
     * @param Y      Output embeddings [B, T, C].
     * @param X      Input token indices [B, T].
     * @param wte    Token embedding table [vocab_size, C].
     * @param wpe    Positional embedding table [max_seq_len, C].
     * @param B      Batch size.
     * @param T      Sequence length.
     * @param C      Embedding dimension (must be divisible by 4).
     * @param stream CUDA stream.
     */
    void cuda_encoder_forward_fp32(
        float* Y,
        const int* X,
        const float* wte, const float* wpe,
        int B, int T, int C,
        cudaStream_t stream )
    {
        assert( C % 4 == 0 );

        int C4 = C / 4;

        dim3 grid( B * T );
        dim3 block( C4 );

        encoder_forward_fp32_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<float4*>(Y), X,
            reinterpret_cast<const float4*>(wte),
            reinterpret_cast<const float4*>(wpe),
            B, T, C );

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Launch the inference-only FP32 single-token decode pass.
     *
     * Computes Y[b,:] = wte[X[b],:] + wpe[position,:] for each batch element.
     * Only the wpe row at `position` is read — no causal masking or sequence
     * iteration is performed. Intended exclusively for KV-cache autoregressive
     * generation; use cuda_encoder_forward_fp32 for training and prefill.
     *
     * @param Y        Output embeddings [B, C].
     * @param X        Token indices [B] (one per batch element).
     * @param wte      Token embedding table [vocab_size, C].
     * @param wpe      Positional embedding table [max_seq_len, C].
     * @param B        Batch size.
     * @param position Absolute sequence position for the wpe row lookup (0-based).
     * @param C        Embedding dimension (must be divisible by 4).
     * @param stream   CUDA stream.
     */
    void cuda_encoder_decode_fp32(
        float* Y,
        const int* X,
        const float* wte, const float* wpe,
        int B, int position, int C,
        cudaStream_t stream )
    {
        assert( C % 4 == 0 );

        int C4 = C / 4;

        constexpr int BLOCK_SIZE = 256;
        int grid_size = ( B * C4 + BLOCK_SIZE - 1 ) / BLOCK_SIZE;

        encoder_decode_fp32_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<float4*>(Y), X,
            reinterpret_cast<const float4*>(wte),
            reinterpret_cast<const float4*>(wpe),
            position, B, C );

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Launch the full-sequence FP32 encoder backward pass.
     *
     * @param wte_grad Gradient accumulation buffer for wte [vocab_size, C].
     * @param wpe_grad Gradient accumulation buffer for wpe [max_seq_len, C].
     * @param dY       Upstream gradient [B, T, C].
     * @param X        Token indices used in forward [B, T].
     * @param B        Batch size.
     * @param T        Sequence length.
     * @param C        Embedding dimension (must be divisible by 4).
     * @param stream   CUDA stream.
     */
    void cuda_encoder_backward_fp32(
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

        encoder_backward_fp32_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<float4*>(wte_grad),
            reinterpret_cast<float4*>(wpe_grad),
            reinterpret_cast<const float4*>(dY),
            X,
            B, T, C );

        cudaCheck( cudaGetLastError() );
    }
}