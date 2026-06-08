#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "SoftmaxCrossEntropy.cuh"

namespace Mila::Dnn::Compute::Cuda::SoftmaxCrossEntropy
{
    __device__ __forceinline__ float blockReduceMax( float val, float* shared ) {
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;

        // Warp-level reduction
        for ( int offset = 16; offset > 0; offset /= 2 ) {
            val = fmaxf( val, __shfl_down_sync( 0xffffffff, val, offset ) );
        }

        if ( lane == 0 ) shared[ wid ] = val;
        __syncthreads();

        // Final reduction across warps
        val = (threadIdx.x < blockDim.x / 32) ? shared[ lane ] : -INFINITY;
        if ( wid == 0 ) {
            for ( int offset = 16; offset > 0; offset /= 2 ) {
                val = fmaxf( val, __shfl_down_sync( 0xffffffff, val, offset ) );
            }
        }

        __syncthreads();
        return __shfl_sync( 0xffffffff, val, 0 );
    }

    __device__ __forceinline__ float blockReduceSum( float val, float* shared ) {
        // Similar to blockReduceMax but with addition
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;

        for ( int offset = 16; offset > 0; offset /= 2 ) {
            val += __shfl_down_sync( 0xffffffff, val, offset );
        }

        if ( lane == 0 ) shared[ wid ] = val;
        __syncthreads();

        val = (threadIdx.x < blockDim.x / 32) ? shared[ lane ] : 0.0f;
        if ( wid == 0 ) {
            for ( int offset = 16; offset > 0; offset /= 2 ) {
                val += __shfl_down_sync( 0xffffffff, val, offset );
            }
        }

        __syncthreads();
        return __shfl_sync( 0xffffffff, val, 0 );
    }

    /**
     * @brief Block-level softmax cross-entropy forward for large vocabularies (>= 1024)
     * Each block processes one (b,s) position with all threads cooperating
     */
    __global__ void softmax_crossentropy_forward_fp32_block_kernel(
        float* __restrict__ losses,
        const float* __restrict__ logits,
        const int* __restrict__ targets,
        int batch_size,
        int seq_len,
        int vocab_size )
    {
        int bs_idx = blockIdx.x;
        if ( bs_idx >= batch_size * seq_len ) return;

        const float* logits_bs = logits + bs_idx * vocab_size;
        int target_idx = targets[ bs_idx ];

        // Phase 1: Find max (block-wide reduction)
        const int vec_size = vocab_size / 4;
        float max_val = -INFINITY;

        for ( int i = threadIdx.x; i < vec_size; i += blockDim.x ) {
            float4 vals = reinterpret_cast<const float4*>( logits_bs )[ i ];
            max_val = fmaxf( max_val, fmaxf( fmaxf( vals.x, vals.y ), fmaxf( vals.z, vals.w ) ) );
        }

        for ( int v = vec_size * 4 + threadIdx.x; v < vocab_size; v += blockDim.x ) {
            max_val = fmaxf( max_val, logits_bs[ v ] );
        }

        __shared__ float shared_max[ 32 ];
        max_val = blockReduceMax( max_val, shared_max );

        // Phase 2: Compute sum
        float sum = 0.0f;

        for ( int i = threadIdx.x; i < vec_size; i += blockDim.x ) {
            float4 vals = reinterpret_cast<const float4*>( logits_bs )[ i ];
            sum += expf( vals.x - max_val ) + expf( vals.y - max_val ) +
                expf( vals.z - max_val ) + expf( vals.w - max_val );
        }

        for ( int v = vec_size * 4 + threadIdx.x; v < vocab_size; v += blockDim.x ) {
            sum += expf( logits_bs[ v ] - max_val );
        }

        __shared__ float shared_sum[ 32 ];
        sum = blockReduceSum( sum, shared_sum );

        // Phase 3: Compute loss
        if ( threadIdx.x == 0 && target_idx >= 0 && target_idx < vocab_size ) {
            float target_logit = logits_bs[ target_idx ];
            losses[ bs_idx ] = -(target_logit - max_val - logf( sum ));
        }
    }

    /**
     * @brief Block-level softmax cross-entropy backward for large vocabularies
     */
    __global__ void softmax_crossentropy_backward_fp32_block_kernel(
        float* __restrict__ dlogits,
        const float* __restrict__ dlosses,
        const float* __restrict__ logits,
        const int* __restrict__ targets,
        int batch_size,
        int seq_len,
        int vocab_size )
    {
        int bs_idx = blockIdx.x;
        if ( bs_idx >= batch_size * seq_len ) return;

        const float* logits_bs = logits + bs_idx * vocab_size;
        float* dlogits_bs = dlogits + bs_idx * vocab_size;

        int target_idx = targets[ bs_idx ];
        float dloss = dlosses[ bs_idx ];

        if ( target_idx < 0 || target_idx >= vocab_size ) {
            const int vec_size = vocab_size / 4;
            for ( int i = threadIdx.x; i < vec_size; i += blockDim.x ) {
                reinterpret_cast<float4*>( dlogits_bs )[ i ] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
            }
            for ( int v = vec_size * 4 + threadIdx.x; v < vocab_size; v += blockDim.x ) {
                dlogits_bs[ v ] = 0.0f;
            }
            return;
        }

        // Recompute softmax statistics
        const int vec_size = vocab_size / 4;
        float max_val = -INFINITY;

        for ( int i = threadIdx.x; i < vec_size; i += blockDim.x ) {
            float4 vals = reinterpret_cast<const float4*>( logits_bs )[ i ];
            max_val = fmaxf( max_val, fmaxf( fmaxf( vals.x, vals.y ), fmaxf( vals.z, vals.w ) ) );
        }

        for ( int v = vec_size * 4 + threadIdx.x; v < vocab_size; v += blockDim.x ) {
            max_val = fmaxf( max_val, logits_bs[ v ] );
        }

        __shared__ float shared_max[ 32 ];
        max_val = blockReduceMax( max_val, shared_max );

        float sum = 0.0f;
        for ( int i = threadIdx.x; i < vec_size; i += blockDim.x ) {
            float4 vals = reinterpret_cast<const float4*>( logits_bs )[ i ];
            sum += expf( vals.x - max_val ) + expf( vals.y - max_val ) +
                expf( vals.z - max_val ) + expf( vals.w - max_val );
        }

        for ( int v = vec_size * 4 + threadIdx.x; v < vocab_size; v += blockDim.x ) {
            sum += expf( logits_bs[ v ] - max_val );
        }

        __shared__ float shared_sum[ 32 ];
        sum = blockReduceSum( sum, shared_sum );

        // Compute gradients
        float inv_sum = 1.0f / sum;

        for ( int i = threadIdx.x; i < vec_size; i += blockDim.x ) {
            float4 vals = reinterpret_cast<const float4*>( logits_bs )[ i ];
            float4 grad;

            int base_idx = i * 4;
            grad.x = expf( vals.x - max_val ) * inv_sum - ((base_idx + 0) == target_idx ? 1.0f : 0.0f);
            grad.y = expf( vals.y - max_val ) * inv_sum - ((base_idx + 1) == target_idx ? 1.0f : 0.0f);
            grad.z = expf( vals.z - max_val ) * inv_sum - ((base_idx + 2) == target_idx ? 1.0f : 0.0f);
            grad.w = expf( vals.w - max_val ) * inv_sum - ((base_idx + 3) == target_idx ? 1.0f : 0.0f);

            grad.x *= dloss;
            grad.y *= dloss;
            grad.z *= dloss;
            grad.w *= dloss;

            reinterpret_cast<float4*>(dlogits_bs)[ i ] = grad;
        }

        for ( int v = vec_size * 4 + threadIdx.x; v < vocab_size; v += blockDim.x ) {
            float prob = expf( logits_bs[ v ] - max_val ) * inv_sum;
            float indicator = (v == target_idx) ? 1.0f : 0.0f;
            dlogits_bs[ v ] = dloss * (prob - indicator);
        }
    }
}