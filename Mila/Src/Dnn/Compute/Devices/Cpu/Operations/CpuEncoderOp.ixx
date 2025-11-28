/**
 * @file CpuEncoderOp.ixx
 * @brief CPU backend for the Encoder operation.
 *
 * Provides CPUs kernels that combine token and positional embeddings and
 * accumulate gradients into embedding tables. Registers the operation
 * with the OperationRegistry.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#ifdef USE_OMP
#include <omp.h>
#endif

export module Compute.CpuEncoderOp;

import Dnn.Components.Encoder;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorHostTypeMap;
import Dnn.ComponentConfig;
import Compute.Precision;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.CpuExecutionContext;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CpuDevice;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief CPU implementation of the Encoder operation.
     *
     * Contract and behavior:
     * - Forward: for each batch b and sequence position t, computes
     *     output[b, t, c] = wte[input[b, t], c] + wpe[t, c]
     *   where `input` is a token-id tensor of shape [B, T] (INT32) and
     *   `output` is an embedding tensor of shape [B, T, C] (FP32).
     *
     * - Backward: accumulates gradients into two parameter tensors:
     *     dwte[input[b, t], :] += output_grad[b, t, :]
     *     dwpe[t, :] += output_grad[b, t, :]
     *   Gradients are accumulated (in-place) into `wte_grad` and `wpe_grad`.
     *
     * Threading and safety:
     * - Forward and backward use OpenMP parallelization across batch and
     *   sequence dimensions when available.
     * - Backward requires atomic updates when multiple threads may update
     *   the same embedding row or positional row concurrently. The implementation
     *   uses OpenMP atomics for per-element accumulation.
     *
     * Parameter binding and ownership:
     * - `setParameters(ITensor* wte, ITensor* wpe)` binds raw parameter tensors.
     *   The operation does NOT take ownership; the caller (module) retains ownership
     *   and must ensure the lifetime exceeds operation usage.
     * - Bound parameter shapes must match the EncoderConfig (vocabulary length,
     *   max sequence length and channels). If shapes mismatch, the method throws.
     *
     * Edge-cases:
     * - Out-of-range token indices are ignored (the implementation currently
     *   skips writing that output location).
     * - Token indices are discrete: no input gradient is produced. The `input_grad`
     *   parameter exists to satisfy the UnaryOperation interface but is unused.
     */
    export class CpuEncoderOp : public UnaryOperation<DeviceType::Cpu, TensorDataType::INT32, TensorDataType::FP32>
    {
    public:
        using MR = CpuMemoryResource;
        using OperationBase = UnaryOperation<DeviceType::Cpu, TensorDataType::INT32, TensorDataType::FP32>;
        using CpuExecutionContext = ExecutionContext<DeviceType::Cpu>;
        using TensorType = Tensor<TensorDataType::FP32, MR>;

        /**
         * @brief Construct with execution context and configuration.
         *
         * Preconditions:
         * - `context` must be non-null and refer to a CPU execution context.
         * - `config` must be valid (EncoderConfig::validate()).
         *
         * Ownership:
         * - The operation stores the provided execution context shared_ptr.
         */
        explicit CpuEncoderOp(
            std::shared_ptr<CpuExecutionContext> context, const EncoderConfig& config )
            : context_( context ), config_( config )
        {
            if (!context)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();
        }

        ~CpuEncoderOp() override = default;

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /**
         * @brief Prepare internal caches for a concrete input shape.
         *
         * Validates the input shape and caches B, T and C for hot-path loops.
         * Must be called (via Module::build) before forward/backward.
         */
        void build( const shape_t& input_shape ) override
        {
            if (is_built_)
                return;

            validateInputShape( input_shape );

            cached_batch_size_ = input_shape[0];
            cached_seq_length_ = input_shape[1];
            cached_embedding_dim_ = config_.getChannels();

            is_built_ = true;
        }

        // ====================================================================
        // Parameter binding
        // ====================================================================

        /**
         * @brief Bind parameter tensors for forward execution.
         *
         * Preconditions:
         * - `wte` and `wpe` must be CPU tensors with shapes:
         *     wte: [vocabulary_length, channels]
         *     wpe: [max_sequence_length, channels]
         *
         * Ownership:
         * - The operation stores raw data pointers to the tensor storage but
         *   does not take ownership of the ITensor objects.
         *
         * Throws:
         * - std::invalid_argument for null pointers, device mismatches or shape mismatches.
         */
        void setParameters( ITensor* wte, ITensor* wpe ) override
        {
            if (!wte)
            {
                throw std::invalid_argument( "CpuEncoderOp::setParameters - wte parameter is required" );
            }

            if (!wpe)
            {
                throw std::invalid_argument( "CpuEncoderOp::setParameters - wpe parameter is required" );
            }

            if (wte->getDeviceType() != DeviceType::Cpu || wpe->getDeviceType() != DeviceType::Cpu)
            {
                throw std::invalid_argument( "CpuEncoderOp::setParameters - parameters must be CPU tensors" );
            }

            // Validate shapes immediately and cache both ITensor* and typed data pointer
            const auto& wte_shape = wte->shape();
            if (wte_shape.size() != 2 ||
                wte_shape[0] != config_.getVocabularyLength() ||
                wte_shape[1] != config_.getChannels())
            {
                throw std::invalid_argument( "CpuEncoderOp::setParameters - wte shape mismatch" );
            }

            const auto& wpe_shape = wpe->shape();
            if (wpe_shape.size() != 2 ||
                wpe_shape[0] != config_.getMaxSequenceLength() ||
                wpe_shape[1] != config_.getChannels())
            {
                throw std::invalid_argument( "CpuEncoderOp::setParameters - wpe shape mismatch" );
            }

            wte_ = static_cast<const float*>(wte->rawData());
            wpe_ = static_cast<const float*>(wpe->rawData());
        }

        /**
         * @brief Bind gradient tensors for training.
         *
         * Preconditions:
         * - `wte_grad` and `wpe_grad` must be CPU tensors with shapes matching
         *   the corresponding parameter tensors.
         *
         * Semantics:
         * - Gradients are accumulated into these buffers (in-place).
         * - Caller is responsible for zeroing gradients when appropriate.
         *
         * Throws:
         * - std::invalid_argument for null pointers, device mismatches or shape mismatches.
         */
        void setGradients( ITensor* wte_grad, ITensor* wpe_grad ) override
        {
            // Both gradients are required for encoder training
            if (!wte_grad || !wpe_grad)
            {
                throw std::invalid_argument( "CpuEncoderOp::setParameterGradients - both wte and wpe gradients are required" );
            }

            if (wte_grad->getDeviceType() != DeviceType::Cpu || wpe_grad->getDeviceType() != DeviceType::Cpu)
            {
                throw std::invalid_argument( "CpuEncoderOp::setParameterGradients - gradients must be CPU tensors" );
            }

            const auto& wte_g_shape = wte_grad->shape();
            if (wte_g_shape.size() != 2 ||
                wte_g_shape[0] != config_.getVocabularyLength() ||
                wte_g_shape[1] != config_.getChannels())
            {
                throw std::invalid_argument( "CpuEncoderOp::setParameterGradients - wte_grad shape mismatch" );
            }

            const auto& wpe_g_shape = wpe_grad->shape();
            if (wpe_g_shape.size() != 2 ||
                wpe_g_shape[0] != config_.getMaxSequenceLength() ||
                wpe_g_shape[1] != config_.getChannels())
            {
                throw std::invalid_argument( "CpuEncoderOp::setParameterGradients - wpe_grad shape mismatch" );
            }

            wte_grad_ = static_cast<float*>(wte_grad->rawData());
            wpe_grad_ = static_cast<float*>(wpe_grad->rawData());
        }

        // ====================================================================
        // Forward pass
        // ====================================================================

        /**
         * @brief Forward pass: combines token and positional embeddings.
         *
         * Parameters:
         * - input: INT32 token indices tensor with shape [B, T]
         * - output: FP32 tensor with shape [B, T, C] to receive embeddings
         *
         * Preconditions:
         * - build() and setParameters() must have been called.
         *
         * Behavior:
         * - Writes output in-place. Out-of-range token indices are skipped.
         */
        void forward( const ITensor& input, ITensor& output ) const override
        {
            if (!is_built_)
            {
                throw std::runtime_error( "CpuEncoderOp: forward called before build()" );
            }

            if (!wte_ || !wpe_)
            {
                throw std::runtime_error( "CpuEncoderOp: parameters not set via setParameters()" );
            }

            validateInputShape( input );

            // Get data pointers (input is INT32)
            const int32_t* X = static_cast<const int32_t*>(input.rawData());
            float* Y = static_cast<float*>(output.rawData());

            const int64_t B = cached_batch_size_;
            const int64_t T = cached_seq_length_;
            const int64_t C = cached_embedding_dim_;

            // Parallel over batch and sequence dimensions
#pragma omp parallel for collapse(2)
            for (int64_t b = 0; b < B; b++)
            {
                for (int64_t t = 0; t < T; t++)
                {
                    // Get token index for this position
                    const int32_t token_idx = X[b * T + t];

                    // Bounds check for token index
                    if (token_idx < 0 || token_idx >= config_.getVocabularyLength())
                    {
                        // Invalid token index - could fill with zeros or throw
                        // For now, skip (output will have undefined values)
                        continue;
                    }

                    // Output pointer for this position
                    float* out_bt = Y + b * T * C + t * C;

                    // Token embedding pointer
                    const float* wte_ix = wte_ + token_idx * C;

                    // Position embedding pointer
                    const float* wpe_t = wpe_ + t * C;

                    // Add token and position embeddings
                    for (int64_t c = 0; c < C; c++)
                    {
                        out_bt[c] = wte_ix[c] + wpe_t[c];
                    }
                }
            }
        }

        // ====================================================================
        // Backward pass
        // ====================================================================

        /**
         * @brief Backward pass: accumulates gradients into embedding tables.
         *
         * Parameters:
         * - input: INT32 token indices tensor with shape [B, T]
         * - output_grad: FP32 gradients tensor with shape [B, T, C]
         * - input_grad: unused (token indices are non-differentiable)
         *
         * Semantics:
         * - Accumulates gradients into `wte_grad` and `wpe_grad` in-place.
         * - Uses atomic updates to ensure thread-safety when multiple threads
         *   update the same row.
         */
        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            if (!is_built_)
            {
                throw std::runtime_error( "CpuEncoderOp: backward called before build()" );
            }

            if (!wte_grad_ || !wpe_grad_)
            {
                throw std::runtime_error( "CpuEncoderOp: parameter gradients not set via setParameterGradients()" );
            }

            const float* X = static_cast<const float*>(input.rawData());
            const float* dY = static_cast<const float*>(output_grad.rawData());
            float* dX = static_cast<float*>(input_grad.rawData());

            const int64_t B = cached_batch_size_;
            const int64_t T = cached_seq_length_;
            const int64_t C = cached_embedding_dim_;

            // Accumulate gradients
            // Note: atomic operations needed because multiple positions can reference same token
#pragma omp parallel for collapse(2)
            for (int64_t b = 0; b < B; b++)
            {
                for (int64_t t = 0; t < T; t++)
                {
                    const int32_t token_idx = X[b * T + t];

                    // Bounds check
                    if (token_idx < 0 || token_idx >= config_.getVocabularyLength())
                    {
                        continue;
                    }

                    const float* dout_bt = dY + b * T * C + t * C;
                    float* dwte_ix = wte_grad_ + token_idx * C;
                    float* dwpe_t = wpe_grad_ + t * C;

                    // Accumulate gradients (atomic for thread safety on dwte)
                    for (int64_t c = 0; c < C; c++)
                    {
                        const float grad = dout_bt[c];

                        // Token embedding gradient (needs atomic - multiple tokens can be same)
#pragma omp atomic
                        dwte_ix[c] += grad;

                        // Position embedding gradient (needs atomic - multiple batches same position)
#pragma omp atomic
                        dwpe_t[c] += grad;
                    }
                }
            }

            // input_grad is unused (token indices are discrete)
        }

        // ====================================================================
        // Operation metadata
        // ====================================================================

        OperationType getOperationType() const override
        {
            return OperationType::EncoderOp;
        }

        std::string getName() const override
        {
            return "CpuEncoderOp";
        }

    private:

        std::shared_ptr<CpuExecutionContext> context_;
        EncoderConfig config_;

        bool is_built_{ false };

        // Parameter pointers (bound by module via setParameters)
        const float* wte_{ nullptr };      // Token embeddings (V, C)
        const float* wpe_{ nullptr };      // Position embeddings (maxT, C)

        // Gradient pointers (bound by module via setParameterGradients)
        float* wte_grad_{ nullptr };
        float* wpe_grad_{ nullptr };

        int64_t cached_batch_size_{ 0 };
        int64_t cached_seq_length_{ 0 };
        int64_t cached_embedding_dim_{ 0 };

        void validateInputShape( const ITensor& input ) const
        {
            const auto& input_shape = input.shape();
            validateInputShape( input_shape );
        }

        void validateInputShape( const shape_t& input_shape ) const
        {
            if (input_shape.size() != 2)
            {
                throw std::invalid_argument(
                    "CpuEncoderOp: input must have rank 2 (batch_size, sequence_length)" );
            }

            if (input_shape[1] > config_.getMaxSequenceLength())
            {
                throw std::invalid_argument(
                    "CpuEncoderOp: sequence length exceeds configured maximum" );
            }
        }


    };

    /**
     * @brief Registrar for CpuEncoderOp operation.
     *
     * Registers the operation with the OperationRegistry during static initialization.
     */
    export class CpuEncoderOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "EncoderOp";

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cpu, TensorDataType::INT32, TensorDataType::FP32>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::INT32, TensorDataType::FP32>>
                {
                    const auto& encoder_config = dynamic_cast<const EncoderConfig&>(config);
                    return std::make_shared<CpuEncoderOp>( context, encoder_config );
                } );
        }

        /*static inline bool isRegistered = []()
            {
                registerOperations();
                return true;
            }();*/
    };
}