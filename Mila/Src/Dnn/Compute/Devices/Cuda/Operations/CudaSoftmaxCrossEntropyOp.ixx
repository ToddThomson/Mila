/**
 * @file CudaSoftmaxCrossEntropyOp.ixx
 * @brief Fused CUDA implementation of Softmax + CrossEntropy loss operation.
 *
 * Combines softmax and cross-entropy into a single numerically stable operation
 * following the ExecutionContext / TensorDataType BinaryOperation interface.
 *
 * Key advantages over separate Softmax + CrossEntropy:
 * - Numerical stability: Uses log-sum-exp trick throughout
 * - Performance: Single GPU kernel pass, no materialized probability distribution
 * - Simplified gradient: dL/dlogits = softmax(logits) - one_hot(targets)
 * - Memory efficiency: No intermediate probability tensor exposed in API
 */

module;
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <type_traits>
#include "Kernels/CudaOps.h"

export module Compute.CudaSoftmaxCrossEntropyOp;

import Dnn.Modules.SoftmaxCrossEntropy;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ConfigurationBase;
import Compute.OperationBase;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.CudaExecutionContext;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief Namespace for CUDA fused softmax cross entropy implementation details.
     */
    namespace Detail
    {
        /**
         * @brief CUDA kernel dispatcher for SoftmaxCrossEntropy operations.
         *
         * Specialized for float and half native CUDA types.
         * Primary template - will cause compile error if no specialization exists.
         */
        template <typename TNative>
        struct cuda_softmax_crossentropy_impl;

        // Specialization for float (FP32)
        template <>
        struct cuda_softmax_crossentropy_impl<float>
        {
            static inline void forward(
                float* losses,
                float* probs,
                const float* logits,
                const int* targets,
                int batch_size,
                int seq_len,
                int vocab_size,
                cudaStream_t stream )
            {
                cuda_softmax_crossentropy_forward<float>(
                    losses, probs, logits, targets,
                    batch_size, seq_len, vocab_size, stream );
            }

            static inline void backward(
                float* dlogits,
                const float* dlosses,
                const float* probs,
                const int* targets,
                int batch_size,
                int seq_len,
                int vocab_size,
                cudaStream_t stream )
            {
                cuda_softmax_crossentropy_backward<float>(
                    dlogits, dlosses, probs, targets,
                    batch_size, seq_len, vocab_size, stream );
            }
        };

        // Specialization for half (FP16)
        template <>
        struct cuda_softmax_crossentropy_impl<half>
        {
            static inline void forward(
                half* losses,
                half* probs,
                const half* logits,
                const int* targets,
                int batch_size,
                int seq_len,
                int vocab_size,
                cudaStream_t stream )
            {
                cuda_softmax_crossentropy_forward<half>(
                    losses, probs, logits, targets,
                    batch_size, seq_len, vocab_size, stream );
            }

            static inline void backward(
                half* dlogits,
                const half* dlosses,
                const half* probs,
                const int* targets,
                int batch_size,
                int seq_len,
                int vocab_size,
                cudaStream_t stream )
            {
                cuda_softmax_crossentropy_backward<half>(
                    dlogits, dlosses, probs, targets,
                    batch_size, seq_len, vocab_size, stream );
            }
        };
    }

    /**
     * @brief Fused CUDA implementation of Softmax + CrossEntropy using abstract TensorDataType API.
     *
     * This operation combines softmax normalization and cross-entropy loss computation
     * into a single numerically stable binary operation (logits + targets ? loss).
     *
     * Key properties:
     * 1. Numerical Stability: Uses log-sum-exp trick to avoid overflow/underflow
     * 2. Performance: Single GPU kernel pass, no intermediate probability tensor in API
     * 3. Simplified Gradient: dL/dlogits = softmax(logits) - one_hot(targets)
     * 4. Memory Efficiency: Probabilities cached internally for backward pass
     *
     * Design philosophy:
     * - Two-phase initialization: build() does all setup, forward()/backward() are pure dispatch
     * - Internal state management: Probabilities cached as mutable private member
     * - Forward computes loss directly from logits
     * - Backward uses cached probabilities from forward pass
     * - All dimension computation happens once in build()
     *
     * @tparam TLogitsPrecision Precision for logits/gradients (FP32, FP16)
     * @tparam TTargets Target indices data type (typically INT32)
     */
    export template<TensorDataType TPrecision, TensorDataType TLogits = TPrecision, TensorDataType TTargets = TensorDataType::INT32>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda> && PrecisionSupportedOnDevice<TLogits, DeviceType::Cuda>
    class CudaSoftmaxCrossEntropyOp : public BinaryOperation<DeviceType::Cuda, TPrecision, TLogits, TTargets>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using BinaryOperationBase = BinaryOperation<DeviceType::Cuda, TPrecision, TLogits, TTargets>;
        using LogitsTensorType = Tensor<TLogits, MR>;
        using TargetsTensorType = Tensor<TTargets, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TLogits>::native_type;
        using TargetsNativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TTargets>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        /**
         * @brief Construct fused Softmax+CrossEntropy operation with execution context.
         *
         * @param context CUDA execution context
         * @param config CrossEntropy configuration (vocab_size required)
         */
        CudaSoftmaxCrossEntropyOp(
            std::shared_ptr<CudaExecutionContext> context,
            const CrossEntropyConfig& config )
            : context_( context ), config_( config )
        {
            if (!context_)
            {
                throw std::runtime_error( "CudaSoftmaxCrossEntropyOp requires a CUDA execution context" );
            }

            config_.validate();
        }

        // ====================================================================
        // Parameters (inherited from OperationBase)
        // ====================================================================

        /**
         * @brief Bind optional class weights parameter.
         *
         * @param class_weights Optional class weights tensor (may be null)
         * @param bias Unused (must be null)
         */
        void setParameters( ITensor* class_weights, ITensor* bias ) override
        {
            if (bias != nullptr)
            {
                throw std::invalid_argument(
                    "CudaSoftmaxCrossEntropyOp::setParameters - bias parameter not supported" );
            }

            if (class_weights != nullptr)
            {
                if (class_weights->getDeviceType() != DeviceType::Cuda)
                {
                    throw std::invalid_argument(
                        "CudaSoftmaxCrossEntropyOp::setParameters - class_weights must be a CUDA tensor" );
                }

                class_weights_ = static_cast<const NativeType*>(class_weights->rawData());
            }
            else
            {
                class_weights_ = nullptr;
            }
        }

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /**
         * @brief Build the operation for a concrete input shape.
         *
         * This is the COLD PATH where all setup, validation, and computation happens ONCE.
         *
         * Expected input shape: [batch_size, seq_length, vocab_size] or [batch_size, vocab_size]
         * Target shape: [batch_size, seq_length] or [batch_size]
         *
         * Responsibilities:
         *  1. Validate input shape (rank >= 2, last dim = vocab_size)
         *  2. Compute and cache dimension sizes (batch, seq, vocab)
         *  3. Cache CUDA stream
         *  4. Allocate internal probability cache for backward pass
         */
        void build( const shape_t& input_shape ) override
        {
            const auto& shape = input_shape;
            const int64_t ndim = static_cast<int64_t>(shape.size());

            if (ndim < 2)
            {
                throw std::invalid_argument(
                    "CudaSoftmaxCrossEntropyOp::build - input must have rank >= 2 (batch x ... x vocab)" );
            }

            // Last dimension is vocabulary size
            cached_vocab_size_ = static_cast<int>( shape.back() );

            if (cached_vocab_size_ != config_.getVocabSize())
            {
                throw std::invalid_argument(
                    "CudaSoftmaxCrossEntropyOp::build - input vocab dimension doesn't match config vocab_size" );
            }

            // For typical shape [B, T, V]: batch_size=B, seq_len=T, vocab_size=V
            if (ndim == 3)
            {
                cached_batch_size_ = static_cast<int>(shape[0]);
                cached_seq_len_ = static_cast<int>(shape[1]);
            }
            // For shape [B, V]: batch_size=B, seq_len=1, vocab_size=V
            else if (ndim == 2)
            {
                cached_batch_size_ = static_cast<int>(shape[0]);
                cached_seq_len_ = 1;
            }
            // For higher rank tensors, flatten all except last dimension
            else
            {
                cached_batch_size_ = 1;
                for (int64_t i = 0; i < ndim - 2; ++i)
                {
                    cached_batch_size_ *= static_cast<int>( shape[i] );
                }
                cached_seq_len_ = static_cast<int>( shape[ndim - 2] );
            }

            cached_stream_ = context_->getStream();

            // Allocate internal probability cache for backward pass
            cached_probs_ = std::make_shared<LogitsTensorType>(
                context_->getDevice(),
                input_shape );

            BinaryOperationBase::is_built_ = true;
        }

        // ====================================================================
        // Forward Pass (BinaryOperation interface)
        // ====================================================================

        /**
         * @brief Forward pass - HOT PATH, computes fused softmax+cross-entropy loss.
         *
         * Computes: loss = -log(softmax(logits)[target])
         *
         * Algorithm (numerically stable):
         *   For each sample:
         *     1. max_logit = max(logits)
         *     2. sum_exp = sum(exp(logits - max_logit))
         *     3. loss = -(logits[target] - max_logit - log(sum_exp))
         *
         * @param inputA Logits tensor [batch, seq, vocab]
         * @param inputB Targets tensor [batch, seq] (integer class indices)
         * @param output Loss tensor (per-sample losses [batch, seq])
         */
        void forward(
            const ITensor& inputA,
            const ITensor& inputB,
            ITensor& output ) const override
        {
            const NativeType* logits_ptr = static_cast<const NativeType*>(inputA.rawData());
            const TargetsNativeType* targets_ptr = static_cast<const TargetsNativeType*>(inputB.rawData());
            NativeType* losses_ptr = static_cast<NativeType*>(output.rawData());

            // Cache probabilities internally for backward pass
            NativeType* probs_ptr = static_cast<NativeType*>(cached_probs_->rawData());

            Detail::cuda_softmax_crossentropy_impl<NativeType>::forward(
                losses_ptr,
                probs_ptr,
                logits_ptr,
                targets_ptr,
                cached_batch_size_,
                cached_seq_len_,
                cached_vocab_size_,
                cached_stream_ );
        }

        // ====================================================================
        // Backward Pass (BinaryOperation interface)
        // ====================================================================

        /**
         * @brief Backward pass - HOT PATH, computes fused gradient.
         *
         * Beautiful property of fused softmax+cross-entropy:
         *   dL/dlogits = softmax(logits) - one_hot(targets)
         *
         * Algorithm:
         *   For each sample:
         *     dL/dlogits[i] = prob[i] - (i == target ? 1 : 0)
         *     Scale by output_gradient
         *
         * @param inputA Logits tensor from forward pass
         * @param inputB Targets tensor from forward pass
         * @param output_gradient Gradient w.r.t. loss (per-sample gradients)
         * @param inputA_gradient Output: gradient w.r.t. logits
         * @param inputB_gradient Unused (targets are integers, not differentiable)
         */
        void backward(
            const ITensor& inputA,
            const ITensor& inputB,
            const ITensor& output_gradient,
            ITensor& inputA_gradient,
            ITensor& inputB_gradient ) const override
        {
            const TargetsNativeType* targets_ptr = static_cast<const TargetsNativeType*>(inputB.rawData());
            const NativeType* dlosses_ptr = static_cast<const NativeType*>(output_gradient.rawData());
            NativeType* dlogits_ptr = static_cast<NativeType*>(inputA_gradient.rawData());

            // Use cached probabilities from forward pass
            const NativeType* probs_ptr = static_cast<const NativeType*>(cached_probs_->rawData());

            // Call CUDA kernel for fused backward pass
            Detail::cuda_softmax_crossentropy_impl<NativeType>::backward(
                dlogits_ptr,
                dlosses_ptr,
                probs_ptr,
                targets_ptr,
                cached_batch_size_,
                cached_seq_len_,
                cached_vocab_size_,
                cached_stream_ );

            // Note: inputB_gradient (targets) is intentionally unused
            // Integer targets are not differentiable
            (void)inputB_gradient;
        }

        // ====================================================================
        // Operation Interface
        // ====================================================================

        OperationType getOperationType() const override
        {
            return OperationType::CrossEntropyOp;
        }

        std::string getName() const override
        {
            return "Cuda::SoftmaxCrossEntropyOp";
        }

        const CrossEntropyConfig& getConfig() const
        {
            return config_;
        }

    private:
        CrossEntropyConfig config_;
        std::shared_ptr<CudaExecutionContext> context_;

        // Parameters bound via setParameters()
        const NativeType* class_weights_{ nullptr };

        // Internal state: Cached probabilities from forward pass for backward
        mutable std::shared_ptr<LogitsTensorType> cached_probs_;

        // Cached dimension values computed once in build() for hot-path dispatch
        int cached_batch_size_{ 0 };
        int cached_seq_len_{ 0 };
        int cached_vocab_size_{ 0 };
        cudaStream_t cached_stream_{ nullptr };
    };

    /**
     * @brief Registrar for fused Softmax+CrossEntropy CUDA operation.
     */
    export class CudaSoftmaxCrossEntropyOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "SoftmaxCrossEntropyOp";

            OperationRegistry::instance().registerBinaryOperation<
                DeviceType::Cuda,
				TensorDataType::FP32,
                TensorDataType::FP32,
				TensorDataType::INT32>(
                    opName,
                    []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                        const ConfigurationBase& config )
                    -> std::shared_ptr<BinaryOperation<DeviceType::Cuda, TensorDataType::FP32, TensorDataType::FP32, TensorDataType::INT32>>
                    {
                        const auto& crossEntropyConfig = static_cast<const CrossEntropyConfig&>(config);
                        return std::make_shared<CudaSoftmaxCrossEntropyOp<TensorDataType::FP32, TensorDataType::FP32, TensorDataType::INT32>>(
                            context, crossEntropyConfig );
                    }
                );

            // Register FP16 variant with INT32 targets
            OperationRegistry::instance().registerBinaryOperation<
                DeviceType::Cuda,
				TensorDataType::FP16,
                TensorDataType::FP16,
				TensorDataType::INT32>(
                    opName,
                    []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                        const ConfigurationBase& config )
                    -> std::shared_ptr<BinaryOperation<DeviceType::Cuda, TensorDataType::FP16, TensorDataType::FP16, TensorDataType::INT32>>
                    {
                        const auto& ceConfig = static_cast<const CrossEntropyConfig&>(config);
                        return std::make_shared<CudaSoftmaxCrossEntropyOp<TensorDataType::FP16, TensorDataType::FP16, TensorDataType::INT32>>(
                            context, ceConfig );
                    }
                );
        }

        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}