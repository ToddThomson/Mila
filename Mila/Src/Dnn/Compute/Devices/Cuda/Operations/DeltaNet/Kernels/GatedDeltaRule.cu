/**
 * @file GatedDeltaRule.cu
 * @brief Recurrent gated delta rule -- one templated kernel, both precisions.
 *
 * Written as one file rather than the Fp32/Bf16 pair used elsewhere because the kernel is
 * long and identical in both, differing only in the load/store conversion. Duplicating a
 * sequential recurrence to change two casts is how the two copies drift.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute::Cuda::DeltaNet
{
    namespace
    {
        __device__ __forceinline__ float toFloat( float x ) { return x; }
        __device__ __forceinline__ float toFloat( __nv_bfloat16 x ) { return __bfloat162float( x ); }

        __device__ __forceinline__ void fromFloat( float v, float& out ) { out = v; }
        __device__ __forceinline__ void fromFloat( float v, __nv_bfloat16& out ) { out = __float2bfloat16( v ); }

        // Matches torch.nn.functional.softplus, including its beta=1 threshold=20 cutover:
        // past 20 the exp overflows long before log1p could matter, and x IS the answer.
        // Precise expf, not __expf: this kernel is latency-bound on a sequential
        // recurrence, not on transcendental throughput, and the FP32 arm is checked against
        // a float32 oracle at 1e-5. The fast intrinsics buy nothing here and cost parity.
        __device__ __forceinline__ float softplus( float x )
        {
            return x > 20.0f ? x : log1pf( expf( x ) );
        }

        __device__ __forceinline__ float sigmoid( float x )
        {
            return 1.0f / (1.0f + expf( -x ) );
        }
    }

    /**
     * @tparam TMaxHeadK Compile-time bound on head_k_dim. The per-thread state array must
     *         be indexed by constants to live in registers, so the loops unroll to this
     *         bound and mask on the runtime width. Instantiated at the sizes actually used.
     */
    template<typename TElement, int TMaxHeadK>
    __global__ void gated_delta_rule_kernel(
        TElement* __restrict__       out,
        const TElement* __restrict__ q,
        const TElement* __restrict__ k,
        const TElement* __restrict__ v,
        const TElement* __restrict__ a,
        const TElement* __restrict__ b,
        const TElement* __restrict__ A_log,
        const TElement* __restrict__ dt_bias,
        float* __restrict__          state,
        int batch, int steps,
        int num_k_heads, int num_v_heads,
        int head_k_dim, int head_v_dim )
    {
        extern __shared__ float shared[];
        float* q_shared = shared;                 // [head_k_dim]
        float* k_shared = shared + head_k_dim;    // [head_k_dim]

        const int block = blockIdx.x;
        const int batch_index = block / num_v_heads;
        const int head = block % num_v_heads;
        const int group = num_v_heads / num_k_heads;
        const int k_head = head / group;

        const int column = threadIdx.x;
        const bool active = (column < head_v_dim);

        const int q_width = num_k_heads * head_k_dim;
        const int v_width = num_v_heads * head_v_dim;

        // Thread-local state column S[:, column]. Registers, for the whole chunk.
        float column_state[ TMaxHeadK ];

        const int state_base = (batch_index * num_v_heads + head) * head_k_dim * head_v_dim;

        #pragma unroll
        for ( int d = 0; d < TMaxHeadK; ++d )
        {
            if ( d < head_k_dim && active )
            {
                column_state[ d ] = state[ state_base + d * head_v_dim + column ];
            }
            else
            {
                column_state[ d ] = 0.0f;
            }
        }

        const float scale = rsqrtf( static_cast<float>( head_k_dim ) );

        const float head_A_log = toFloat( A_log[ head ] );
        const float head_dt_bias = toFloat( dt_bias[ head ] );

        for ( int t = 0; t < steps; ++t )
        {
            const int qk_offset = (batch_index * steps + t) * q_width + k_head * head_k_dim;

            // Every thread walks the same bound, so the syncs below are block-uniform.
            for ( int d = threadIdx.x; d < head_k_dim; d += blockDim.x )
            {
                q_shared[ d ] = toFloat( q[ qk_offset + d ] );
                k_shared[ d ] = toFloat( k[ qk_offset + d ] );
            }

            __syncthreads();

            // L2 norms, recomputed per thread from shared memory. A block reduction would
            // save the arithmetic and cost a second sync plus a partial-warp hazard; these
            // are broadcast reads and the loop is already unrolled.
            float q_square_sum = 0.0f;
            float k_square_sum = 0.0f;

            #pragma unroll
            for ( int d = 0; d < TMaxHeadK; ++d )
            {
                if ( d < head_k_dim )
                {
                    q_square_sum += q_shared[ d ] * q_shared[ d ];
                    k_square_sum += k_shared[ d ] * k_shared[ d ];
                }
            }

            const float q_inv_norm = rsqrtf( q_square_sum + 1e-6f ) * scale;
            const float k_inv_norm = rsqrtf( k_square_sum + 1e-6f );

            const int scalar_offset = (batch_index * steps + t) * num_v_heads + head;
            const float beta = sigmoid( toFloat( b[ scalar_offset ] ) );
            const float g = -expf( head_A_log )
                * softplus( toFloat( a[ scalar_offset ] ) + head_dt_bias );
            const float decay = expf( g );

            const int v_offset = (batch_index * steps + t) * v_width + head * head_v_dim + column;
            const float v_t = active ? toFloat( v[ v_offset ] ) : 0.0f;

            float kv_memory = 0.0f;

            #pragma unroll
            for ( int d = 0; d < TMaxHeadK; ++d )
            {
                if ( d < head_k_dim )
                {
                    column_state[ d ] *= decay;
                    kv_memory += column_state[ d ] * (k_shared[ d ] * k_inv_norm);
                }
            }

            const float delta = (v_t - kv_memory) * beta;

            float result = 0.0f;

            #pragma unroll
            for ( int d = 0; d < TMaxHeadK; ++d )
            {
                if ( d < head_k_dim )
                {
                    column_state[ d ] += (k_shared[ d ] * k_inv_norm) * delta;
                    result += column_state[ d ] * (q_shared[ d ] * q_inv_norm);
                }
            }

            if ( active )
            {
                fromFloat( result, out[ v_offset ] );
            }

            // The next iteration overwrites q_shared/k_shared.
            __syncthreads();
        }

        #pragma unroll
        for ( int d = 0; d < TMaxHeadK; ++d )
        {
            if ( d < head_k_dim && active )
            {
                state[ state_base + d * head_v_dim + column ] = column_state[ d ];
            }
        }
    }

    namespace
    {
        template<typename TElement>
        void launch(
            TElement* out, const TElement* q, const TElement* k, const TElement* v,
            const TElement* a, const TElement* b,
            const TElement* A_log, const TElement* dt_bias,
            float* state,
            int batch, int steps,
            int num_k_heads, int num_v_heads,
            int head_k_dim, int head_v_dim,
            cudaStream_t stream )
        {
            const int blocks = batch * num_v_heads;
            const int threads = head_v_dim;
            const size_t shared_bytes = static_cast<size_t>( 2 * head_k_dim ) * sizeof( float );

            if ( head_k_dim <= 8 )
            {
                gated_delta_rule_kernel<TElement, 8> << <blocks, threads, shared_bytes, stream >> > (
                    out, q, k, v, a, b, A_log, dt_bias, state,
                    batch, steps, num_k_heads, num_v_heads, head_k_dim, head_v_dim );
            }
            else if ( head_k_dim <= 64 )
            {
                gated_delta_rule_kernel<TElement, 64> << <blocks, threads, shared_bytes, stream >> > (
                    out, q, k, v, a, b, A_log, dt_bias, state,
                    batch, steps, num_k_heads, num_v_heads, head_k_dim, head_v_dim );
            }
            else
            {
                gated_delta_rule_kernel<TElement, 128> << <blocks, threads, shared_bytes, stream >> > (
                    out, q, k, v, a, b, A_log, dt_bias, state,
                    batch, steps, num_k_heads, num_v_heads, head_k_dim, head_v_dim );
            }

            cudaCheck( cudaGetLastError() );
        }
    }

    void cuda_gated_delta_rule_fp32(
        float* out, const float* q, const float* k, const float* v,
        const float* a, const float* b, const float* A_log, const float* dt_bias,
        float* state,
        int batch, int steps, int num_k_heads, int num_v_heads,
        int head_k_dim, int head_v_dim, cudaStream_t stream )
    {
        launch<float>( out, q, k, v, a, b, A_log, dt_bias, state,
            batch, steps, num_k_heads, num_v_heads, head_k_dim, head_v_dim, stream );
    }

    void cuda_gated_delta_rule_bf16(
        __nv_bfloat16* out, const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
        const __nv_bfloat16* a, const __nv_bfloat16* b,
        const __nv_bfloat16* A_log, const __nv_bfloat16* dt_bias,
        float* state,
        int batch, int steps, int num_k_heads, int num_v_heads,
        int head_k_dim, int head_v_dim, cudaStream_t stream )
    {
        launch<__nv_bfloat16>( out, q, k, v, a, b, A_log, dt_bias, state,
            batch, steps, num_k_heads, num_v_heads, head_k_dim, head_v_dim, stream );
    }
}
