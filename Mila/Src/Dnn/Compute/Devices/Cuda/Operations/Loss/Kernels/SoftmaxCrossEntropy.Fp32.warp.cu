#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "SoftmaxCrossEntropy.cuh"

namespace Mila::Dnn::Compute::Cuda::SoftmaxCrossEntropy
{
    /**
     * @brief Warp-level softmax cross-entropy forward for small vocabularies (< 1024)
     * Each warp processes one (b,s) position cooperatively across vocabulary
     */
    __global__ void softmax_crossentropy_forward_fp32_warp_kernel(
        float* __restrict__ losses,
        const float* __restrict__ logits,
        const int* __restrict__ targets,
        int batch_size,
        int seq_len,
        int vocab_size )
    {
        int lane_id = threadIdx.x % WARP_SIZE;
        int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
        int num_positions = batch_size * seq_len;

        if ( warp_id >= num_positions ) return;

        const float* logits_bs = logits + warp_id * vocab_size;
        int target_idx = targets[ warp_id ];

        // Phase 1: Find max with float4 vectorization
        const int vec_size = vocab_size / 4;
        float max_val = -INFINITY;

        for ( int i = lane_id; i < vec_size; i += WARP_SIZE ) {
            float4 vals = reinterpret_cast<const float4*>( logits_bs )[ i ];
            max_val = fmaxf( max_val, fmaxf( fmaxf( vals.x, vals.y ), fmaxf( vals.z, vals.w ) ) );
        }

        // Handle remainder
        for ( int v = vec_size * 4 + lane_id; v < vocab_size; v += WARP_SIZE ) {
            max_val = fmaxf( max_val, logits_bs[ v ] );
        }

        // Warp reduce max
        for ( int offset = WARP_SIZE / 2; offset > 0; offset /= 2 ) {
            max_val = fmaxf( max_val, __shfl_down_sync( 0xffffffff, max_val, offset ) );
        }
        max_val = __shfl_sync( 0xffffffff, max_val, 0 );

        // Phase 2: Compute sum of exp(logits - max)
        float sum = 0.0f;

        for ( int i = lane_id; i < vec_size; i += WARP_SIZE ) {
            float4 vals = reinterpret_cast<const float4*>( logits_bs )[ i ];
            sum += expf( vals.x - max_val ) + expf( vals.y - max_val ) +
                expf( vals.z - max_val ) + expf( vals.w - max_val );
        }

        for ( int v = vec_size * 4 + lane_id; v < vocab_size; v += WARP_SIZE ) {
            sum += expf( logits_bs[ v ] - max_val );
        }

        // Warp reduce sum
        for ( int offset = WARP_SIZE / 2; offset > 0; offset /= 2 ) {
            sum += __shfl_down_sync( 0xffffffff, sum, offset );
        }
        sum = __shfl_sync( 0xffffffff, sum, 0 );

        // Phase 3: Compute loss (lane 0 only)
        if ( lane_id == 0 && target_idx >= 0 && target_idx < vocab_size ) {
            float target_logit = logits_bs[ target_idx ];
            losses[ warp_id ] = -(target_logit - max_val - logf( sum ));
        }
    }

    /**
     * @brief Warp-level softmax cross-entropy backward for small vocabularies
     */
    __global__ void softmax_crossentropy_backward_fp32_warp_kernel(
        float* __restrict__ dlogits,
        const float* __restrict__ dlosses,
        const float* __restrict__ logits,
        const int* __restrict__ targets,
        int batch_size,
        int seq_len,
        int vocab_size )
    {
        int lane_id = threadIdx.x % WARP_SIZE;
        int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
        int num_positions = batch_size * seq_len;

        if ( warp_id >= num_positions ) return;

        const float* logits_bs = logits + warp_id * vocab_size;
        float* dlogits_bs = dlogits + warp_id * vocab_size;

        int target_idx = targets[ warp_id ];
        float dloss = dlosses[ warp_id ];

        if ( target_idx < 0 || target_idx >= vocab_size ) {
            // Zero out gradients for invalid targets
            const int vec_size = vocab_size / 4;
            for ( int i = lane_id; i < vec_size; i += WARP_SIZE ) {
                reinterpret_cast<float4*>( dlogits_bs )[ i ] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
            }
            for ( int v = vec_size * 4 + lane_id; v < vocab_size; v += WARP_SIZE ) {
                dlogits_bs[ v ] = 0.0f;
            }
            return;
        }

        // Recompute softmax statistics
        const int vec_size = vocab_size / 4;
        float max_val = -INFINITY;

        for ( int i = lane_id; i < vec_size; i += WARP_SIZE ) {
            float4 vals = reinterpret_cast<const float4*>( logits_bs )[ i ];
            max_val = fmaxf( max_val, fmaxf( fmaxf( vals.x, vals.y ), fmaxf( vals.z, vals.w ) ) );
        }

        for ( int v = vec_size * 4 + lane_id; v < vocab_size; v += WARP_SIZE ) {
            max_val = fmaxf( max_val, logits_bs[ v ] );
        }

        for ( int offset = WARP_SIZE / 2; offset > 0; offset /= 2 ) {
            max_val = fmaxf( max_val, __shfl_down_sync( 0xffffffff, max_val, offset ) );
        }
        max_val = __shfl_sync( 0xffffffff, max_val, 0 );

        float sum = 0.0f;
        for ( int i = lane_id; i < vec_size; i += WARP_SIZE ) {
            float4 vals = reinterpret_cast<const float4*>( logits_bs )[ i ];
            sum += expf( vals.x - max_val ) + expf( vals.y - max_val ) +
                expf( vals.z - max_val ) + expf( vals.w - max_val );
        }

        for ( int v = vec_size * 4 + lane_id; v < vocab_size; v += WARP_SIZE ) {
            sum += expf( logits_bs[ v ] - max_val );
        }

        for ( int offset = WARP_SIZE / 2; offset > 0; offset /= 2 ) {
            sum += __shfl_down_sync( 0xffffffff, sum, offset );
        }
        sum = __shfl_sync( 0xffffffff, sum, 0 );

        // Compute gradients with vectorization
        float inv_sum = 1.0f / sum;

        for ( int i = lane_id; i < vec_size; i += WARP_SIZE ) {
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

        for ( int v = vec_size * 4 + lane_id; v < vocab_size; v += WARP_SIZE ) {
            float prob = expf( logits_bs[ v ] - max_val ) * inv_sum;
            float indicator = (v == target_idx) ? 1.0f : 0.0f;
            dlogits_bs[ v ] = dloss * (prob - indicator);
        }
    }
}