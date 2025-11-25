/**
 * @file CpuSoftmaxCrossEntropyOp.ixx
 * @brief Fused CPU implementation of Softmax + CrossEntropy loss operation.
 *
 * Combines softmax and cross-entropy into a single numerically stable operation
 * following the ExecutionContext / TensorDataType BinaryOperation interface.
 *
 * Key advantages over separate Softmax + CrossEntropy:
 * - Numerical stability: Uses log-sum-exp trick throughout
 * - Performance: Single pass, no materialized probability distribution
 * - Simplified gradient: dL/dlogits = softmax(logits) - one_hot(targets)
 * - Memory efficiency: No intermediate probability tensor exposed in API
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <limits>
#ifdef USE_OMP
#include <omp.h>
#endif
#include <cstdint>

export module Compute.CpuSoftmaxCrossEntropyOp;

import Dnn.Components.SoftmaxCrossEntropy;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorHostTypeMap;
import Dnn.ComponentConfig;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.CpuExecutionContext;
import Compute.OperationType;
import Compute.OperationBase;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief Fused CPU implementation of Softmax + CrossEntropy using abstract TensorDataType API.
     *
     * This operation combines softmax normalization and cross-entropy loss computation
     * into a single numerically stable binary operation (logits + targets ? loss).
     *
     * Key properties:
     * 1. Numerical Stability: Uses log-sum-exp trick to avoid overflow/underflow
     * 2. Performance: Single pass, no intermediate probability tensor in API
     * 3. Simplified Gradient: dL/dlogits = softmax(logits) - one_hot(targets)
     * 4. Memory Efficiency: Probabilities cached internally only if needed
     *
     * Design philosophy:
     * - Two-phase initialization: build() does all setup, forward()/backward() are pure dispatch
     * - Internal state management: Probabilities cached as mutable private member
     * - Forward computes loss directly from logits
     * - Backward uses cached probabilities (or recomputes if not cached)
     * - All dimension computation happens once in build()
     *
     * @tparam TLogitsPrecision Precision for logits/gradients (FP32)
     * @tparam TTargets Target indices data type (typically INT32)
     */
    export template<
        TensorDataType TPrecision = TensorDataType::FP32,
        TensorDataType TLogits = TPrecision,
        TensorDataType TTargets = TensorDataType::INT32>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cpu> && PrecisionSupportedOnDevice<TLogits, DeviceType::Cpu>
    class CpuSoftmaxCrossEntropyOp : public BinaryOperation<DeviceType::Cpu, TPrecision, TLogits, TTargets>
    {
    public:
        using MR = CpuMemoryResource;
        using BinaryOperationBase = BinaryOperation<DeviceType::Cpu, TPrecision, TLogits, TTargets>;
        using LogitsTensorType = Tensor<TLogits, MR>;
        using TargetsTensorType = Tensor<TTargets, MR>;
        using LogitsHostType = typename TensorHostTypeMap<TLogits>::host_type;
        using TargetsHostType = typename TensorHostTypeMap<TTargets>::host_type;
        using CpuExecutionContext = ExecutionContext<DeviceType::Cpu>;

        /**
         * @brief Construct fused Softmax+CrossEntropy operation with execution context.
         *
         * @param context CPU execution context
         * @param config CrossEntropy configuration (vocab_size required)
         */
        CpuSoftmaxCrossEntropyOp( std::shared_ptr<CpuExecutionContext> context, const CrossEntropyConfig& config )
            : context_( context ), config_( config )
        {
            if (!context_)
            {
                throw std::runtime_error( "CpuSoftmaxCrossEntropyOp requires a CPU execution context" );
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
                    "CpuSoftmaxCrossEntropyOp::setParameters - bias parameter not supported" );
            }

            class_weights_ = class_weights;
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
         *  2. Compute and cache dimension sizes (outer_size, vocab_size)
         *  3. Cache OMP parallelization threshold
         */
        void build( const shape_t& input_shape ) override
        {
            const auto& shape = input_shape;
            const int64_t ndim = static_cast<int64_t>(shape.size());

            if (ndim < 2)
            {
                throw std::invalid_argument(
                    "CpuSoftmaxCrossEntropyOp::build - input must have rank >= 2 (batch x ... x vocab)" );
            }

            // Last dimension is vocabulary size
            cached_vocab_size_ = shape.back();

            if (cached_vocab_size_ != config_.getVocabSize())
            {
                throw std::invalid_argument(
                    "CpuSoftmaxCrossEntropyOp::build - input vocab dimension doesn't match config vocab_size" );
            }

            // Flatten all dimensions except last into outer_size
            cached_outer_size_ = 1;
            for (int64_t i = 0; i < ndim - 1; ++i)
            {
                cached_outer_size_ *= shape[i];
            }

            // Enable OMP if we have enough elements
            enable_omp_ = (cached_outer_size_ > 100);

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
         * @param inputA Logits tensor [outer_size, vocab_size]
         * @param inputB Targets tensor [outer_size] (integer class indices)
         * @param output Loss tensor (scalar or per-sample)
         */
        void forward(
            const ITensor& inputA,
            const ITensor& inputB,
            ITensor& output ) const override
        {
            const auto& logits = dynamic_cast<const LogitsTensorType&>(inputA);
            const auto& targets = dynamic_cast<const TargetsTensorType&>(inputB);
            auto& losses = dynamic_cast<LogitsTensorType&>(output);

            const LogitsHostType* logits_data = logits.data();
            const TargetsHostType* targets_data = targets.data();
            LogitsHostType* output_data = losses.data();

            const int64_t outer_size = cached_outer_size_;
            const int64_t vocab_size = cached_vocab_size_;

            // Parallel reduction for total loss
            long double total_loss = 0.0L;
            int64_t valid_samples = 0;

#pragma omp parallel for reduction(+:total_loss,valid_samples) if(enable_omp_)
            for (int64_t i = 0; i < outer_size; ++i)
            {
                const LogitsHostType* logits_i = logits_data + i * vocab_size;
                TargetsHostType target = targets_data[i];

                // Validate target index (skip invalid)
                if (target < 0 || target >= vocab_size)
                {
                    continue;
                }

                // ============================================================
                // SOFTMAX: Numerical stability with max subtraction
                // ============================================================

                LogitsHostType max_logit = -std::numeric_limits<LogitsHostType>::infinity();
                for (int64_t v = 0; v < vocab_size; ++v)
                {
                    if (logits_i[v] > max_logit)
                        max_logit = logits_i[v];
                }

                // Compute log-sum-exp denominator
                long double sum_exp = 0.0L;
                for (int64_t v = 0; v < vocab_size; ++v)
                {
                    sum_exp += std::expl( static_cast<long double>( logits_i[v] - max_logit ) );
                }

                // ============================================================
                // CROSS-ENTROPY: Compute -log(softmax(logits)[target])
                // ============================================================

                long double log_sum_exp = std::logl( sum_exp );
                long double target_logit = static_cast<long double>( logits_i[target] );
                long double sample_loss = -(target_logit - static_cast<long double>(max_logit) - log_sum_exp);

                // Apply class weight if provided
                if (class_weights_)
                {
                    const auto& weights_tensor = dynamic_cast<const LogitsTensorType&>(*class_weights_);
                    LogitsHostType weight = weights_tensor.data()[target];
                    sample_loss *= static_cast<long double>(weight);
                }

                total_loss += sample_loss;
                valid_samples++;
            }

            // Mean reduction (always applied for now)
            output_data[0] = valid_samples > 0
                ? static_cast<LogitsHostType>(total_loss / static_cast<long double>(valid_samples))
                : static_cast<LogitsHostType>(0.0L);
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
         *     1. Compute softmax probabilities
         *     2. dL/dlogits[i] = prob[i] - (i == target ? 1 : 0)
         *     3. Scale by output_gradient
         *
         * @param inputA Logits tensor from forward pass
         * @param inputB Targets tensor from forward pass
         * @param output_gradient Gradient w.r.t. loss (typically scalar 1.0)
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
            const auto& logits = dynamic_cast<const LogitsTensorType&>(inputA);
            const auto& targets = dynamic_cast<const TargetsTensorType&>(inputB);
            const auto& grad_output = dynamic_cast<const LogitsTensorType&>(output_gradient);
            auto& grad_logits = dynamic_cast<LogitsTensorType&>(inputA_gradient);

            const LogitsHostType* logits_data = logits.data();
            const TargetsHostType* targets_data = targets.data();
            const LogitsHostType* grad_output_data = grad_output.data();
            LogitsHostType* grad_logits_data = grad_logits.data();

            const int64_t outer_size = cached_outer_size_;
            const int64_t vocab_size = cached_vocab_size_;

            // Scalar gradient from loss (typically 1.0, but could be scaled)
            LogitsHostType loss_scale = grad_output_data[0];

            // Average gradient over samples (matches forward mean reduction)
            loss_scale /= static_cast<LogitsHostType>(outer_size);

#pragma omp parallel for if(enable_omp_)
            for (int64_t i = 0; i < outer_size; ++i)
            {
                const LogitsHostType* logits_i = logits_data + i * vocab_size;
                LogitsHostType* grad_i = grad_logits_data + i * vocab_size;
                TargetsHostType target = targets_data[i];

                // Validate target index
                if (target < 0 || target >= vocab_size)
                {
                    // Zero gradient for invalid targets
                    for (int64_t v = 0; v < vocab_size; ++v)
                    {
                        grad_i[v] = static_cast<LogitsHostType>( 0.0 );
                    }
                    continue;
                }

                // ============================================================
                // Compute softmax probabilities (recompute from logits)
                // ============================================================

                LogitsHostType max_logit = -std::numeric_limits<LogitsHostType>::infinity();
                for (int64_t v = 0; v < vocab_size; ++v)
                {
                    if (logits_i[v] > max_logit)
                        max_logit = logits_i[v];
                }

                long double sum_exp = 0.0L;
                for (int64_t v = 0; v < vocab_size; ++v)
                {
                    sum_exp += std::expl( static_cast<long double>( logits_i[v] - max_logit ) );
                }

                long double inv_sum = 1.0L / sum_exp;

                // ============================================================
                // FUSED GRADIENT: softmax(logits) - one_hot(targets)
                // ============================================================

                for (int64_t v = 0; v < vocab_size; ++v)
                {
                    // Compute softmax probability
                    long double prob = std::expl( static_cast<long double>( logits_i[v] - max_logit ) ) * inv_sum;

                    // Subtract one-hot target
                    long double indicator = (v == target) ? 1.0L : 0.0L;

                    // Accumulate gradient (+=)
                    grad_i[v] += static_cast<LogitsHostType>( (prob - indicator) * static_cast<long double>( loss_scale ) );
                }
            }

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
            return "Cpu::SoftmaxCrossEntropyOp";
        }

        const CrossEntropyConfig& getConfig() const
        {
            return config_;
        }

    private:
        CrossEntropyConfig config_;
        std::shared_ptr<CpuExecutionContext> context_;

        // Parameters bound via setParameters()
        ITensor* class_weights_{ nullptr };

        // Cached dimension values computed once in build() for hot-path dispatch
        int64_t cached_outer_size_{ 0 };    // batch * seq_length
        int64_t cached_vocab_size_{ 0 };    // vocabulary dimension
        bool enable_omp_{ false };
    };

    /**
     * @brief Registrar for fused Softmax+CrossEntropy operation.
     */
    export class CpuSoftmaxCrossEntropyOpRegistrar
    {
    public:
        static void registerOperations()
        {
            // Register op keyed by input types (logits, targets). Return shared_ptr typed to the precision/logits/targets variant.
            OperationRegistry::instance().registerBinaryOperation<
                DeviceType::Cpu,
                TensorDataType::FP32,
				TensorDataType::FP32,
                TensorDataType::INT32>(
                    "SoftmaxCrossEntropyOp",
                    []( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context,
                        const ComponentConfig& config ) -> std::shared_ptr<BinaryOperation<DeviceType::Cpu, TensorDataType::FP32, TensorDataType::FP32, TensorDataType::INT32>>
                    {
                        const auto& crossEntropyConfig = static_cast<const CrossEntropyConfig&>(config);
                        return std::make_shared<
                            CpuSoftmaxCrossEntropyOp<TensorDataType::FP32, TensorDataType::FP32, TensorDataType::INT32>>(
                                context,
                                crossEntropyConfig);
                    }
                );
        }

        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}