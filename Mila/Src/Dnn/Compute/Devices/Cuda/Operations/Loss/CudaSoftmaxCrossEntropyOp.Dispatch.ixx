/**
 * @file CudaSoftmaxCrossEntropyOp.Dispatch.ixx
 * @brief Implementation of the CUDA SoftmaxCrossEntropy kernel dispatch mechanism.
 */

module;
#include <cuda_bf16.h>
#include <type_traits>
#include "Kernels/SoftmaxCrossEntropy.cuh"

export module Compute.CudaSoftmaxCrossEntropyOp:Dispatch;

namespace Mila::Dnn::Compute::Cuda::SoftmaxCrossEntropy
{
    /**
     * @brief Namespace for CUDA fused softmax cross entropy implementation details.
     */
    namespace Detail
    {
        /**
         * @brief CUDA kernel dispatcher for SoftmaxCrossEntropy operations.
         *
         * Specialized for float (FP32), nv_bfloat16 (BF16) CUDA types.
         * Primary template - will cause compile error if no specialization exists.
         */
        template<typename TComputeType>
            requires std::is_same_v<TComputeType, float> || std::is_same_v<TComputeType, nv_bfloat16>
        struct cuda_softmax_crossentropy_impl;

        template<>
        struct cuda_softmax_crossentropy_impl<float>
        {
            static inline void forward(
                float* losses,
                // float* probs,
                const float* logits,
                const int* targets,
                int batch_size,
                int seq_len,
                int vocab_size,
                cudaStream_t stream )
            {
                cuda_softmax_crossentropy_forward_fp32(
                    losses, /* probs, */ logits, targets,
                    batch_size, seq_len, vocab_size, stream );
            }

            static inline void backward(
                float* dlogits,
                const float* dlosses,
                const float* logits,
                const int* targets,
                int batch_size,
                int seq_len,
                int vocab_size,
                cudaStream_t stream )
            {
                cuda_softmax_crossentropy_backward_fp32(
                    dlogits, dlosses, logits, targets,
                    batch_size, seq_len, vocab_size, stream );
            }
        };

        template<>
        struct cuda_softmax_crossentropy_impl<nv_bfloat16>
        {
            static inline void forward(
                half* losses,
                // half* probs,
                const half* logits,
                const int* targets,
                int batch_size,
                int seq_len,
                int vocab_size,
                cudaStream_t stream )
            {
                /*cuda_softmax_crossentropy_forward<half>(
                    losses, probs, logits, targets,
                    batch_size, seq_len, vocab_size, stream );*/
            }

            static inline void backward(
                half* dlogits,
                const half* dlosses,
                // const half* probs,
                const int* targets,
                int batch_size,
                int seq_len,
                int vocab_size,
                cudaStream_t stream )
            {
                /*cuda_softmax_crossentropy_backward<half>(
                    dlogits, dlosses, probs, targets,
                    batch_size, seq_len, vocab_size, stream );*/
            }
        };
    }
}
