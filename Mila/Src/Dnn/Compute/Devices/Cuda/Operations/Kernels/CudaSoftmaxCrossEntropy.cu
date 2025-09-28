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
     * @brief CUDA kernel for fused softmax and cross-entropy forward pass with FP32 precision
     *
     * This kernel computes both softmax probabilities and cross-entropy loss in a single pass
     * for better computational efficiency.
     *
     * @param losses Output tensor containing cross-entropy losses (B, S)
     * @param probs Output tensor containing softmax probabilities (B, S, V)
     * @param logits Input tensor containing unnormalized logits (B, S, V)
     * @param targets Input tensor containing target class indices (B, S)
     * @param batch_size Batch dimension size (B)
     * @param seq_len Sequence length dimension size (S)
     * @param vocab_size Vocabulary size dimension (V)
     */
    __global__ void softmax_crossentropy_forward_fp32_kernel(
        float* losses,
        float* probs,
        const float* logits,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size )
    {
        // Each thread processes one position (b,s) across vocab dimension
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int b = idx / seq_len;  // Batch index
        int s = idx % seq_len;  // Sequence position

        if ( b < batch_size && s < seq_len ) {
            // Pointer to start of logits for this position
            const float* logits_bs = logits + (b * seq_len * vocab_size) + (s * vocab_size);
            float* probs_bs = probs + (b * seq_len * vocab_size) + (s * vocab_size);

            // Get target index for this position
            int target_idx = targets[ b * seq_len + s ];

            // Find max value for numerical stability
            float max_val = -INFINITY;
            for ( int v = 0; v < vocab_size; v++ ) {
                if ( logits_bs[ v ] > max_val ) {
                    max_val = logits_bs[ v ];
                }
            }

            // Compute exp(logits - max_val) and sum
            float sum = 0.0f;
            for ( int v = 0; v < vocab_size; v++ ) {
                probs_bs[ v ] = expf( logits_bs[ v ] - max_val );
                sum += probs_bs[ v ];
            }

            // Normalize to get probabilities and compute loss
            float inv_sum = 1.0f / sum;
            for ( int v = 0; v < vocab_size; v++ ) {
                probs_bs[ v ] *= inv_sum;
            }

            // Compute cross-entropy loss: -log(prob[target_idx])
            // Check if target_idx is valid
            if ( target_idx >= 0 && target_idx < vocab_size ) {
                losses[ b * seq_len + s ] = -logf( probs_bs[ target_idx ] + 1e-10f ); // Add small epsilon to avoid log(0)
            }
            else {
                losses[ b * seq_len + s ] = 0.0f; // Invalid target index
            }
        }
    }

    /**
     * @brief CUDA kernel for fused softmax and cross-entropy forward pass with FP16 precision
     *
     * This kernel computes both softmax probabilities and cross-entropy loss in a single pass
     * using FP16 precision with intermediate FP32 calculations for numerical stability.
     *
     * @param losses Output tensor containing cross-entropy losses (B, S)
     * @param probs Output tensor containing softmax probabilities (B, S, V)
     * @param logits Input tensor containing unnormalized logits (B, S, V)
     * @param targets Input tensor containing target class indices (B, S)
     * @param batch_size Batch dimension size (B)
     * @param seq_len Sequence length dimension size (S)
     * @param vocab_size Vocabulary size dimension (V)
     */
    __global__ void softmax_crossentropy_forward_fp16_kernel(
        half* losses,
        half* probs,
        const half* logits,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int b = idx / seq_len;
        int s = idx % seq_len;

        if ( b < batch_size && s < seq_len ) {
            // Pointer to start of logits for this position
            const half* logits_bs = logits + (b * seq_len * vocab_size) + (s * vocab_size);
            half* probs_bs = probs + (b * seq_len * vocab_size) + (s * vocab_size);

            // Get target index for this position
            int target_idx = targets[ b * seq_len + s ];

            // Find max value for numerical stability (using float for computation)
            float max_val = -INFINITY;
            for ( int v = 0; v < vocab_size; v++ ) {
                float val = __half2float( logits_bs[ v ] );
                if ( val > max_val ) {
                    max_val = val;
                }
            }

            // Compute exp(logits - max_val) and sum
            float sum = 0.0f;
            for ( int v = 0; v < vocab_size; v++ ) {
                float exp_val = expf( __half2float( logits_bs[ v ] ) - max_val );
                probs_bs[ v ] = __float2half( exp_val );
                sum += exp_val;
            }

            // Normalize to get probabilities
            float inv_sum = 1.0f / sum;
            for ( int v = 0; v < vocab_size; v++ ) {
                float prob = __half2float( probs_bs[ v ] ) * inv_sum;
                probs_bs[ v ] = __float2half( prob );
            }

            // Compute cross-entropy loss: -log(prob[target_idx])
            if ( target_idx >= 0 && target_idx < vocab_size ) {
                float prob = __half2float( probs_bs[ target_idx ] );
                float loss = -logf( prob + 1e-10f );
                losses[ b * seq_len + s ] = __float2half( loss );
            }
            else {
                losses[ b * seq_len + s ] = __float2half( 0.0f ); // Invalid target index
            }
        }
    }

    /**
     * @brief CUDA kernel for fused softmax cross-entropy backward pass with FP32 precision
     *
     * This kernel computes gradients for the fused softmax and cross-entropy operation.
     * For each position, gradient is: prob[i] - one_hot[i]
     *
     * @param dlogits Output tensor for gradients with respect to logits (B, S, V)
     * @param dlosses Input tensor containing gradient of loss with respect to outputs (B, S)
     * @param probs Input tensor containing softmax probabilities from forward pass (B, S, V)
     * @param targets Input tensor containing target class indices (B, S)
     * @param batch_size Batch dimension size (B)
     * @param seq_len Sequence length dimension size (S)
     * @param vocab_size Vocabulary size dimension (V)
     */
    __global__ void softmax_crossentropy_backward_fp32_kernel(
        float* dlogits,
        const float* dlosses,
        const float* probs,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int b = idx / seq_len;
        int s = idx % seq_len;

        if ( b < batch_size && s < seq_len ) {
            float dloss = dlosses[ b * seq_len + s ];
            int target_idx = targets[ b * seq_len + s ];

            // Pointer to probability and gradient arrays for this position
            const float* probs_bs = probs + (b * seq_len * vocab_size) + (s * vocab_size);
            float* dlogits_bs = dlogits + (b * seq_len * vocab_size) + (s * vocab_size);

            // Only process if target_idx is valid
            if ( target_idx >= 0 && target_idx < vocab_size ) {
                // Gradient is (prob[i] - one_hot[i]) * dloss
                for ( int v = 0; v < vocab_size; v++ ) {
                    float indicator = (v == target_idx) ? 1.0f : 0.0f;
                    dlogits_bs[ v ] = dloss * (probs_bs[ v ] - indicator);
                }
            }
            else {
                // If target_idx is invalid, just zero out gradients
                for ( int v = 0; v < vocab_size; v++ ) {
                    dlogits_bs[ v ] = 0.0f;
                }
            }
        }
    }

    /**
     * @brief CUDA kernel for fused softmax cross-entropy backward pass with FP16 precision
     *
     * This kernel computes gradients for the fused softmax and cross-entropy operation
     * using FP16 precision with intermediate FP32 calculations for numerical stability.
     *
     * @param dlogits Output tensor for gradients with respect to logits (B, S, V)
     * @param dlosses Input tensor containing gradient of loss with respect to outputs (B, S)
     * @param probs Input tensor containing softmax probabilities from forward pass (B, S, V)
     * @param targets Input tensor containing target class indices (B, S)
     * @param batch_size Batch dimension size (B)
     * @param seq_len Sequence length dimension size (S)
     * @param vocab_size Vocabulary size dimension (V)
     */
    __global__ void softmax_crossentropy_backward_fp16_kernel(
        half* dlogits,
        const half* dlosses,
        const half* probs,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int b = idx / seq_len;
        int s = idx % seq_len;

        if ( b < batch_size && s < seq_len ) {
            float dloss = __half2float( dlosses[ b * seq_len + s ] );
            int target_idx = targets[ b * seq_len + s ];

            // Pointer to probability and gradient arrays for this position
            const half* probs_bs = probs + (b * seq_len * vocab_size) + (s * vocab_size);
            half* dlogits_bs = dlogits + (b * seq_len * vocab_size) + (s * vocab_size);

            // Only process if target_idx is valid
            if ( target_idx >= 0 && target_idx < vocab_size ) {
                // Gradient is (prob[i] - one_hot[i]) * dloss
                for ( int v = 0; v < vocab_size; v++ ) {
                    float prob = __half2float( probs_bs[ v ] );
                    float indicator = (v == target_idx) ? 1.0f : 0.0f;
                    dlogits_bs[ v ] = __float2half( dloss * (prob - indicator) );
                }
            }
            else {
                // If target_idx is invalid, just zero out gradients
                for ( int v = 0; v < vocab_size; v++ ) {
                    dlogits_bs[ v ] = __float2half( 0.0f );
                }
            }
        }
    }

    /**
     * @brief Host function to launch fused softmax cross-entropy forward pass (FP32 version)
     */
    void cuda_softmax_crossentropy_forward_fp32(
        float* losses,
        float* probs,
        const float* logits,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream )
    {
        int total_elements = batch_size * seq_len;
        int block_size = 512;
        int grid_size = ceil_div( total_elements, block_size );

        softmax_crossentropy_forward_fp32_kernel << <grid_size, block_size, 0, stream >> > (
            losses, probs, logits, targets, batch_size, seq_len, vocab_size);
        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host function to launch fused softmax cross-entropy forward pass (FP16 version)
     */
    void cuda_softmax_crossentropy_forward_fp16(
        half* losses,
        half* probs,
        const half* logits,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream )
    {
        int total_elements = batch_size * seq_len;
        int block_size = 512;
        int grid_size = ceil_div( total_elements, block_size );

        softmax_crossentropy_forward_fp16_kernel << <grid_size, block_size, 0, stream >> > (
            losses, probs, logits, targets, batch_size, seq_len, vocab_size);
        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host function to launch fused softmax cross-entropy backward pass (FP32 version)
     */
    void cuda_softmax_crossentropy_backward_fp32(
        float* dlogits,
        const float* dlosses,
        const float* probs,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream )
    {
        int total_elements = batch_size * seq_len;
        int block_size = 512;
        int grid_size = ceil_div( total_elements, block_size );

        softmax_crossentropy_backward_fp32_kernel << <grid_size, block_size, 0, stream >> > (
            dlogits, dlosses, probs, targets, batch_size, seq_len, vocab_size);
        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host function to launch fused softmax cross-entropy backward pass (FP16 version)
     */
    void cuda_softmax_crossentropy_backward_fp16(
        half* dlogits,
        const half* dlosses,
        const half* probs,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream )
    {
        int total_elements = batch_size * seq_len;
        int block_size = 512;
        int grid_size = ceil_div( total_elements, block_size );

        softmax_crossentropy_backward_fp16_kernel << <grid_size, block_size, 0, stream >> > (
            dlogits, dlosses, probs, targets, batch_size, seq_len, vocab_size);
        cudaCheck( cudaGetLastError() );
    }

    // Template wrapper functions for type-agnostic calling
    template <typename TPrecision>
    void cuda_softmax_crossentropy_forward(
        TPrecision* losses,
        TPrecision* probs,
        const TPrecision* logits,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream )
    {
        if constexpr ( std::is_same_v<TPrecision, float> ) {
            cuda_softmax_crossentropy_forward_fp32(
                losses, probs, logits, targets, batch_size, seq_len, vocab_size, stream );
        }
        else if constexpr ( std::is_same_v<TPrecision, half> ) {
            cuda_softmax_crossentropy_forward_fp16(
                losses, probs, logits, targets, batch_size, seq_len, vocab_size, stream );
        }
    }

    template <typename TPrecision>
    void cuda_softmax_crossentropy_backward(
        TPrecision* dlogits,
        const TPrecision* dlosses,
        const TPrecision* probs,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream )
    {
        if constexpr ( std::is_same_v<TPrecision, float> ) {
            cuda_softmax_crossentropy_backward_fp32(
                dlogits, dlosses, probs, targets, batch_size, seq_len, vocab_size, stream );
        }
        else if constexpr ( std::is_same_v<TPrecision, half> ) {
            cuda_softmax_crossentropy_backward_fp16(
                dlogits, dlosses, probs, targets, batch_size, seq_len, vocab_size, stream );
        }
    }

    // Explicit instantiations for float
    template void cuda_softmax_crossentropy_forward<float>(
        float* losses,
        float* probs,
        const float* logits,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream );

    template void cuda_softmax_crossentropy_backward<float>(
        float* dlogits,
        const float* dlosses,
        const float* probs,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream );

    // Explicit instantiations for half
    template void cuda_softmax_crossentropy_forward<half>(
        half* losses,
        half* probs,
        const half* logits,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream );

    template void cuda_softmax_crossentropy_backward<half>(
        half* dlogits,
        const half* dlosses,
        const half* probs,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream );
}
