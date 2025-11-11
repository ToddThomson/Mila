/**
 * @file CpuAttentionOp.ixx
 * @brief CPU implementation of the Multi-Head Attention operation.
 *
 * Implements the forward and backward passes for scaled dot-product attention
 * with multiple heads on CPU devices. This is a stateful operation that caches
 * attention weights for the backward pass.
 */

module;
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <algorithm>
#ifdef USE_OMP
#include <omp.h>
#endif

export module Compute.CpuAttention;

import Dnn.Modules.Attention;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ConfigurationBase;
import Compute.Precision;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.OperationAttributes;
import Compute.OperationType;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.CpuDevice;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief CPU implementation of Multi-Head Attention operation.
     *
     * Performs scaled dot-product attention with multiple heads:
     * - Forward: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
     * - Backward: Computes gradients for Q, K, V
     *
     * The operation is stateful - it caches pre-attention scores and attention
     * weights during forward pass for use in backward pass.
     *
     * @tparam TPrecision Tensor data type (typically FP32 or FP64)
     */
    export class CpuAttentionOp : public UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using MR = CpuMemoryResource;
        using OperationBase = UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>;
        using CpuExecutionContext = ExecutionContext<DeviceType::Cpu>;
        using TensorType = Tensor<TensorDataType::FP32, MR>;

        /**
         * @brief Construct with execution context and configuration.
         *
         * @param exec_context Shared execution context for CPU resources.
         * @param config Multi-head attention configuration.
         */
        explicit CpuAttentionOp( std::shared_ptr<CpuExecutionContext> context, const AttentionConfig& config )
            : context_( context ), config_( config )
        {
            if (!context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();
        }

        ~CpuAttentionOp() override = default;

        // ====================================================================
        // Lifecycle
        // ====================================================================

        void build( const shape_t& input_shape ) override
        {
            if (is_built_)
                return;

            validateInputShape( input_shape );

            cached_batch_size_ = input_shape[0];
            cached_seq_length_ = input_shape[1];
            cached_qkv_dim_ = input_shape[2];  // 3 * embedding_dim

            cached_embedding_dim_ = config_.getEmbeddingDim();
            cached_num_heads_ = config_.getNumHeads();
            cached_head_size_ = cached_embedding_dim_ / cached_num_heads_;

            // Allocate state tensors for forward pass caching
            allocateStateTensors();

            is_built_ = true;
        }

        // ====================================================================
        // Parameter binding (no parameters for attention - it's purely a transformation)
        // ====================================================================

        void setParameters( ITensor* /*unused1*/, ITensor* /*unused2*/ ) override
        {
            // Attention has no learnable parameters
        }

        void setParameterGradients( ITensor* /*unused1*/, ITensor* /*unused2*/ ) override
        {
            // Attention has no learnable parameters
        }

        // ====================================================================
        // Forward pass
        // ====================================================================

        /**
         * @brief Forward pass: scaled dot-product attention with multiple heads.
         *
         * Input shape: [B, T, 3*C] containing concatenated Q, K, V
         * Output shape: [B, T, C]
         *
         * For each head h and position t:
         * 1. Compute attention scores: scores[t, t2] = Q[t] · K[t2] / sqrt(d_k)
         * 2. Apply causal mask (t2 <= t)
         * 3. Apply softmax: att[t, t2] = exp(scores[t, t2]) / sum(exp(scores[t, :]))
         * 4. Compute output: out[t] = sum(att[t, t2] * V[t2])
         *
         * @param input Input tensor [B, T, 3*C] (Q, K, V concatenated)
         * @param output Output tensor [B, T, C]
         */
        void forward( const ITensor& input, ITensor& output ) const override
        {
            if (!is_built_)
            {
                throw std::runtime_error( "CpuAttentionOp: forward called before build()" );
            }

            validateInputShape( input );

            const auto& input_tensor = dynamic_cast<const TensorType&>(input);
            auto& output_tensor = dynamic_cast<TensorType&>(output);

            const float* X = input_tensor.data();
            float* Y = output_tensor.data();

            const int64_t B = cached_batch_size_;
            const int64_t T = cached_seq_length_;
            const int64_t C = cached_embedding_dim_;
            const int64_t C3 = cached_qkv_dim_;
            const int64_t NH = cached_num_heads_;
            const int64_t hs = cached_head_size_;

            const float scale = static_cast<float>(1.0) / std::sqrt( static_cast<float>(hs) );

            float* preatt_data = preatt_cache_->data();
            float* att_data = att_cache_->data();

            // Parallel over batch, sequence, and heads
#pragma omp parallel for collapse(3)
            for (int64_t b = 0; b < B; b++)
            {
                for (int64_t t = 0; t < T; t++)
                {
                    for (int64_t h = 0; h < NH; h++)
                    {
                        // Query vector for this position and head
                        const float* query_t = X + b * T * C3 + t * C3 + h * hs;

                        // Pointers to pre-attention scores and attention weights
                        float* preatt_bth = preatt_data + b * NH * T * T + h * T * T + t * T;
                        float* att_bth = att_data + b * NH * T * T + h * T * T + t * T;

                        // Compute Q·K^T scores with causal masking
                        float maxval = static_cast<float>( -10000.0 );

                        for (int64_t t2 = 0; t2 <= t; t2++)  // Causal mask
                        {
                            const float* key_t2 = X + b * T * C3 + t2 * C3 + h * hs + C;

                            float val = static_cast<float>(0.0);
                            for (int64_t i = 0; i < hs; i++)
                            {
                                val += query_t[i] * key_t2[i];
                            }

                            val *= scale;

                            if (val > maxval)
                            {
                                maxval = val;
                            }

                            preatt_bth[t2] = val;
                        }

                        // Softmax: exp(scores - max) / sum(exp)
                        float expsum = static_cast<float>(0.0);

                        for (int64_t t2 = 0; t2 <= t; t2++)
                        {
                            float expv = std::exp( preatt_bth[t2] - maxval );
                            expsum += expv;
                            att_bth[t2] = expv;
                        }

                        const float expsum_inv = (expsum == static_cast<float>(0.0))
                            ? static_cast<float>(0.0)
                            : static_cast<float>(1.0) / expsum;

                        // Normalize attention weights and zero out future positions
                        for (int64_t t2 = 0; t2 < T; t2++)
                        {
                            if (t2 <= t)
                            {
                                att_bth[t2] *= expsum_inv;
                            }
                            else
                            {
                                att_bth[t2] = static_cast<float>( 0.0 );
                            }
                        }

                        // Compute weighted sum of values
                        float* out_bth = Y + b * T * C + t * C + h * hs;

                        for (int64_t i = 0; i < hs; i++)
                        {
                            out_bth[i] = static_cast<float>( 0.0 );
                        }

                        for (int64_t t2 = 0; t2 <= t; t2++)
                        {
                            const float* value_t2 = X + b * T * C3 + t2 * C3 + h * hs + C * 2;
                            const float att_weight = att_bth[t2];

                            for (int64_t i = 0; i < hs; i++)
                            {
                                out_bth[i] += att_weight * value_t2[i];
                            }
                        }
                    }
                }
            }
        }

        // ====================================================================
        // Backward pass
        // ====================================================================

        /**
         * @brief Backward pass: computes gradients for Q, K, V.
         *
         * Uses cached pre-attention scores and attention weights from forward pass.
         *
         * @param input Input from forward pass [B, T, 3*C]
         * @param output_grad Gradient w.r.t. output [B, T, C]
         * @param input_grad Gradient w.r.t. input [B, T, 3*C]
         */
        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            if (!is_built_)
            {
                throw std::runtime_error( "CpuAttentionOp: backward called before build()" );
            }

            const float* X = static_cast<const float*>(input.rawData());
            const float* dY = static_cast<const float*>(output_grad.rawData());
            float* dX = static_cast<float*>(input_grad.rawData());

            const float* preatt = preatt_cache_->data();
            const float* att = att_cache_->data();

            // Create temporary gradient tensors for attention state
            auto device = context_->getDevice();
            TensorType dpreatt( device, preatt_cache_->shape() );
            TensorType datt( device, att_cache_->shape() );

            // FIXME: zeros( dpreatt );
            // FIXME: zeros( datt );

            float* dpreatt_data = dpreatt.data();
            float* datt_data = datt.data();

            const int64_t B = cached_batch_size_;
            const int64_t T = cached_seq_length_;
            const int64_t C = cached_embedding_dim_;
            const int64_t C3 = cached_qkv_dim_;
            const int64_t NH = cached_num_heads_;
            const int64_t hs = cached_head_size_;

            const float scale = static_cast<float>(1.0) / std::sqrt( static_cast<float>(hs) );

            // Backward through attention
            for (int64_t b = 0; b < B; b++)
            {
                for (int64_t t = 0; t < T; t++)
                {
                    for (int64_t h = 0; h < NH; h++)
                    {
                        const float* att_bth = att + b * NH * T * T + h * T * T + t * T;
                        float* datt_bth = datt_data + b * NH * T * T + h * T * T + t * T;
                        float* dpreatt_bth = dpreatt_data + b * NH * T * T + h * T * T + t * T;

                        float* dquery_t = dX + b * T * C3 + t * C3 + h * hs;
                        const float* query_t = X + b * T * C3 + t * C3 + h * hs;
                        const float* dout_bth = dY + b * T * C + t * C + h * hs;

                        // Backprop through weighted value sum
                        for (int64_t t2 = 0; t2 <= t; t2++)
                        {
                            const float* value_t2 = X + b * T * C3 + t2 * C3 + h * hs + C * 2;
                            float* dvalue_t2 = dX + b * T * C3 + t2 * C3 + h * hs + C * 2;

                            for (int64_t i = 0; i < hs; i++)
                            {
                                // Gradient w.r.t. attention weights
                                datt_bth[t2] += value_t2[i] * dout_bth[i];

                                // Gradient w.r.t. values
                                dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                            }
                        }

                        // Backprop through softmax
                        for (int64_t t2 = 0; t2 <= t; t2++)
                        {
                            for (int64_t t3 = 0; t3 <= t; t3++)
                            {
                                const float indicator = (t2 == t3)
                                    ? static_cast<float>(1.0)
                                    : static_cast<float>(0.0);

                                const float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);

                                dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                            }
                        }

                        // Backprop through Q·K^T
                        for (int64_t t2 = 0; t2 <= t; t2++)
                        {
                            const float* key_t2 = X + b * T * C3 + t2 * C3 + h * hs + C;
                            float* dkey_t2 = dX + b * T * C3 + t2 * C3 + h * hs + C;

                            for (int64_t i = 0; i < hs; i++)
                            {
                                // Gradient w.r.t. queries
                                dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;

                                // Gradient w.r.t. keys
                                dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                            }
                        }
                    }
                }
            }
        }

        // ====================================================================
        // Operation metadata
        // ====================================================================

        OperationType getOperationType() const override
        {
            return OperationType::AttentionOp;
		}

        std::string getName() const override
        {
            return "CpuAttentionOp";
        }

    private:
        std::shared_ptr<CpuExecutionContext> context_;
        AttentionConfig config_;
        bool is_built_{ false };

        // Cached dimensions
        int64_t cached_batch_size_{ 0 };
        int64_t cached_seq_length_{ 0 };
        int64_t cached_qkv_dim_{ 0 };         // 3 * embedding_dim
        int64_t cached_embedding_dim_{ 0 };
        int64_t cached_num_heads_{ 0 };
        int64_t cached_head_size_{ 0 };       // embedding_dim / num_heads

        // State tensors cached during forward for backward pass
        mutable std::shared_ptr<TensorType> preatt_cache_{ nullptr };  // Pre-softmax scores [B, NH, T, T]
        mutable std::shared_ptr<TensorType> att_cache_{ nullptr };     // Attention weights [B, NH, T, T]

        void validateInputShape( const ITensor& input ) const
        {
            const auto& input_shape = input.shape();
            validateInputShape( input_shape );
        }

        void validateInputShape( const shape_t& input_shape ) const
        {
            if (input_shape.size() != 3)
            {
                throw std::invalid_argument(
                    "CpuAttentionOp: input must have rank 3 (batch_size, seq_length, 3*embedding_dim)" );
            }

            const int64_t expected_qkv_dim = 3 * config_.getEmbeddingDim();

            if (input_shape[2] != expected_qkv_dim)
            {
                throw std::invalid_argument(
                    "CpuAttentionOp: input last dimension must be 3*embedding_dim (Q, K, V concatenated)" );
            }
        }

        void allocateStateTensors()
        {
            auto device = context_->getDevice();

            // Pre-attention scores: [B, NH, T, T]
            shape_t preatt_shape = {
                cached_batch_size_,
                cached_num_heads_,
                cached_seq_length_,
                cached_seq_length_
            };

            preatt_cache_ = std::make_shared<TensorType>( device, preatt_shape );
            preatt_cache_->setName( "preatt_cache" );

            // Attention weights: [B, NH, T, T]
            att_cache_ = std::make_shared<TensorType>( device, preatt_shape );
            att_cache_->setName( "att_cache" );
        }
    };

    /**
     * @brief Registrar for CpuAttentionOp operation.
     *
     * Registers the operation with the OperationRegistry during static initialization.
     */
    export class CpuAttentionOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "AttentionOp";

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cpu, TensorDataType::FP32, TensorDataType::FP32>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context,
                    const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>>
                {
                    const auto& attention_config = dynamic_cast<const AttentionConfig&>(config);
                    return std::make_shared<CpuAttentionOp>( context, attention_config );
                }
            );
        }

        static inline bool isRegistered = []()
            {
                registerOperations();
                return true;
            }();
    };
}