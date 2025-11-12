/**
 * @file CpuAttentionOp.ixx
 * @brief CPU implementation of the Multi-Head Attention operation (head-major inputs).
 *
 * Expects Q, K, V tensors laid out as [B, NH, T, hs]. Produces gradients and
 * outputs in the same head-major layout to avoid per-call memory reorganization.
 */

    module;
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstring>
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
import Compute.TernaryOperation;
import Compute.OperationRegistry;
import Compute.OperationAttributes;
import Compute.OperationType;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.CpuExecutionContext;
import Compute.CpuDevice;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief CPU implementation of Multi-Head Attention operation.
     *
     * This variant expects inputs in head-major layout: Q, K, V each are
     * [B, NH, T, hs]. Output and gradients are produced in the same layout.
     */
    export class CpuAttentionOp : public TernaryOperation<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using MR = CpuMemoryResource;
        using CpuExecutionContext = ExecutionContext<DeviceType::Cpu>;
        using TensorType = Tensor<TensorDataType::FP32, MR>;

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

        // Build expects the head-major Q shape: [B, NH, T, hs]
        void build( const shape_t& input_shape ) override
        {
            if (is_built_)
            {
                return;
            }

            validateHeadMajorShape( input_shape );

            cached_batch_size_ = input_shape[0];
            cached_num_heads_ = input_shape[1];
            cached_seq_length_ = input_shape[2];
            cached_head_size_ = input_shape[3];

            cached_embedding_dim_ = cached_num_heads_ * cached_head_size_;
            cached_qkv_dim_ = 3 * cached_embedding_dim_;

            // Allocate state tensors (attention scores / weights)
            allocateStateTensors();

            is_built_ = true;
        }

        void setParameters( ITensor* /*unused1*/, ITensor* /*unused2*/ ) override
        {
            // No learnable parameters
        }

        void setParameterGradients( ITensor* /*unused1*/, ITensor* /*unused2*/ ) override
        {
            // No learnable parameters
        }

        /**
         * Forward pass expects:
         *  - input_q, input_k, input_v: [B, NH, T, hs] (head-major)
         *  - output: [B, NH, T, hs] (head-major)
         *
         * Produces attention result in head-major layout to avoid reorganization.
         */
        void forward(
            const ITensor& input_q,
            const ITensor& input_k,
            const ITensor& input_v,
            ITensor& output ) const override
        {
            if (!is_built_)
            {
                throw std::runtime_error( "CpuAttentionOp: forward called before build()" );
            }

            validateHeadMajorShapes( input_q, input_k, input_v, output );

            const float* q_data = static_cast<const float*>(input_q.rawData());
            const float* k_data = static_cast<const float*>(input_k.rawData());
            const float* v_data = static_cast<const float*>(input_v.rawData());
            float* out_data = static_cast<float*>(output.rawData());

            const int64_t B = cached_batch_size_;
            const int64_t NH = cached_num_heads_;
            const int64_t T = cached_seq_length_;
            const int64_t hs = cached_head_size_;

            const float scale = 1.0f / std::sqrt( static_cast<float>(hs) );

            float* preatt_data = preatt_cache_->data();
            float* att_data = att_cache_->data();

            // Step 1: Compute QK^T for all heads: Scores [B, NH, T, T]
#pragma omp parallel for collapse(2)
            for (int64_t b = 0; b < B; b++)
            {
                for (int64_t h = 0; h < NH; h++)
                {
                    const int64_t head_offset = b * NH * T * hs + h * T * hs;
                    const int64_t score_offset = b * NH * T * T + h * T * T;

                    const float* Q_bh = q_data + head_offset;   // [T, hs]
                    const float* K_bh = k_data + head_offset;   // [T, hs]
                    float* scores_bh = preatt_data + score_offset;  // [T, T]

                    for (int64_t i = 0; i < T; i++)
                    {
                        for (int64_t j = 0; j < T; j++)
                        {
                            float sum = 0.0f;
                            for (int64_t k = 0; k < hs; k++)
                            {
                                sum += Q_bh[i * hs + k] * K_bh[j * hs + k];
                            }
                            scores_bh[i * T + j] = sum * scale;
                        }
                    }
                }
            }

            // Step 2: Apply causal mask and softmax -> att_data
#pragma omp parallel for collapse(3)
            for (int64_t b = 0; b < B; b++)
            {
                for (int64_t h = 0; h < NH; h++)
                {
                    for (int64_t t = 0; t < T; t++)
                    {
                        const int64_t score_offset = b * NH * T * T + h * T * T;
                        float* scores_row = preatt_data + score_offset + t * T;
                        float* att_row = att_data + score_offset + t * T;

                        float maxval = -INFINITY;
                        for (int64_t t2 = 0; t2 <= t; t2++)
                        {
                            if (scores_row[t2] > maxval) maxval = scores_row[t2];
                        }

                        float expsum = 0.0f;
                        for (int64_t t2 = 0; t2 <= t; t2++)
                        {
                            float expv = std::exp( scores_row[t2] - maxval );
                            att_row[t2] = expv;
                            expsum += expv;
                        }

                        const float expsum_inv = (expsum > 0.0f) ? (1.0f / expsum) : 0.0f;
                        for (int64_t t2 = 0; t2 <= t; t2++)
                        {
                            att_row[t2] *= expsum_inv;
                        }

                        for (int64_t t2 = t + 1; t2 < T; t2++)
                        {
                            att_row[t2] = 0.0f;
                        }
                    }
                }
            }

            // Step 3: Out = Att × V  => out_data is [B, NH, T, hs]
#pragma omp parallel for collapse(2)
            for (int64_t b = 0; b < B; b++)
            {
                for (int64_t h = 0; h < NH; h++)
                {
                    const int64_t head_offset = b * NH * T * hs + h * T * hs;
                    const int64_t score_offset = b * NH * T * T + h * T * T;

                    const float* att_bh = att_data + score_offset;  // [T, T]
                    const float* V_bh = v_data + head_offset;       // [T, hs]
                    float* out_bh = out_data + head_offset;         // [T, hs]

                    for (int64_t i = 0; i < T; i++)
                    {
                        for (int64_t k = 0; k < hs; k++)
                        {
                            float sum = 0.0f;
                            for (int64_t j = 0; j < T; j++)
                            {
                                sum += att_bh[i * T + j] * V_bh[j * hs + k];
                            }
                            out_bh[i * hs + k] = sum;
                        }
                    }
                }
            }
        }

        /**
         * Backward: inputs and grads are all head-major:
         *  - input_q, input_k, input_v: [B, NH, T, hs]
         *  - output_grad: [B, NH, T, hs]
         *  - q_grad, k_grad, v_grad: [B, NH, T, hs] (written by this routine)
         */
        void backward(
            const ITensor& input_q,
            const ITensor& input_k,
            const ITensor& input_v,
            const ITensor& output_grad,
            ITensor& q_grad,
            ITensor& k_grad,
            ITensor& v_grad ) const override
        {
            if (!is_built_)
            {
                throw std::runtime_error( "CpuAttentionOp: backward called before build()" );
            }

            validateHeadMajorShapesForBackward( input_q, input_k, input_v, output_grad, q_grad, k_grad, v_grad );

            const float* q_data = static_cast<const float*>(input_q.rawData());
            const float* k_data = static_cast<const float*>(input_k.rawData());
            const float* v_data = static_cast<const float*>(input_v.rawData());
            const float* dY = static_cast<const float*>(output_grad.rawData());

            float* dq_data = static_cast<float*>(q_grad.rawData());
            float* dk_data = static_cast<float*>(k_grad.rawData());
            float* dv_data = static_cast<float*>(v_grad.rawData());

            const int64_t B = cached_batch_size_;
            const int64_t NH = cached_num_heads_;
            const int64_t T = cached_seq_length_;
            const int64_t hs = cached_head_size_;

            const float scale = 1.0f / std::sqrt( static_cast<float>(hs) );

            // Zero gradients
            std::memset( dq_data, 0, static_cast<size_t>(B) * NH * T * hs * sizeof( float ) );
            std::memset( dk_data, 0, static_cast<size_t>(B) * NH * T * hs * sizeof( float ) );
            std::memset( dv_data, 0, static_cast<size_t>(B) * NH * T * hs * sizeof( float ) );

            // Reuse preatt_cache_ and att_cache_ from forward
            float* preatt_data = preatt_cache_->data();
            float* att_data = att_cache_->data();

            // Temporary buffers for intermediate grads dAtt, dPreatt
            auto device = context_->getDevice();
            shape_t score_shape = { B, NH, T, T };

            TensorType dAtt( device, score_shape );
            TensorType dPreatt( device, score_shape );

            float* datt_data = dAtt.data();
            float* dpreatt_data = dPreatt.data();

            std::memset( datt_data, 0, static_cast<size_t>(B) * NH * T * T * sizeof( float ) );
            std::memset( dpreatt_data, 0, static_cast<size_t>(B) * NH * T * T * sizeof( float ) );

            // Backprop through Att × V, softmax, and QK^T
#pragma omp parallel for collapse(2)
            for (int64_t b = 0; b < B; b++)
            {
                for (int64_t h = 0; h < NH; h++)
                {
                    const int64_t head_offset = b * NH * T * hs + h * T * hs;
                    const int64_t score_offset = b * NH * T * T + h * T * T;

                    const float* att_bh = att_data + score_offset;
                    const float* V_bh = v_data + head_offset;
                    const float* Q_bh = q_data + head_offset;
                    const float* K_bh = k_data + head_offset;

                    float* datt_bh = datt_data + score_offset;
                    float* dpreatt_bh = dpreatt_data + score_offset;
                    const float* dout_bh = dY + head_offset;
                    float* dV_bh = dv_data + head_offset;
                    float* dQ_bh = dq_data + head_offset;
                    float* dK_bh = dk_data + head_offset;

                    // dAtt = dOut × V^T ; dV = Att^T × dOut
                    for (int64_t t = 0; t < T; t++)
                    {
                        for (int64_t t2 = 0; t2 < T; t2++)
                        {
                            for (int64_t k = 0; k < hs; k++)
                            {
                                float dout_val = dout_bh[t * hs + k];
                                datt_bh[t * T + t2] += dout_val * V_bh[t2 * hs + k];
                                dV_bh[t2 * hs + k] += att_bh[t * T + t2] * dout_val;
                            }
                        }
                    }

                    // softmax backward (causal)
                    for (int64_t t = 0; t < T; t++)
                    {
                        const float* att_row = att_bh + t * T;
                        const float* datt_row = datt_bh + t * T;
                        float* dpreatt_row = dpreatt_bh + t * T;

                        for (int64_t t2 = 0; t2 <= t; t2++)
                        {
                            for (int64_t t3 = 0; t3 <= t; t3++)
                            {
                                const float indicator = (t2 == t3) ? 1.0f : 0.0f;
                                const float local_derivative = att_row[t2] * (indicator - att_row[t3]);
                                dpreatt_row[t3] += local_derivative * datt_row[t2];
                            }
                        }
                    }

                    // dQ = dPreatt × K × scale
                    // dK = dPreatt^T × Q × scale
                    for (int64_t t = 0; t < T; t++)
                    {
                        for (int64_t t2 = 0; t2 < T; t2++)
                        {
                            const float dpreatt_val = dpreatt_bh[t * T + t2] * scale;
                            for (int64_t k = 0; k < hs; k++)
                            {
                                dQ_bh[t * hs + k] += dpreatt_val * K_bh[t2 * hs + k];
                                dK_bh[t2 * hs + k] += dpreatt_val * Q_bh[t * hs + k];
                            }
                        }
                    }
                }
            }
        }

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

        // Cached dimensions for head-major layout
        int64_t cached_batch_size_{ 0 };
        int64_t cached_seq_length_{ 0 };
        int64_t cached_qkv_dim_{ 0 };
        int64_t cached_embedding_dim_{ 0 };
        int64_t cached_num_heads_{ 0 };
        int64_t cached_head_size_{ 0 };

        // State tensors: attention scores/weights [B, NH, T, T]
        mutable std::shared_ptr<TensorType> preatt_cache_{ nullptr };
        mutable std::shared_ptr<TensorType> att_cache_{ nullptr };

        void validateHeadMajorShape( const shape_t& s ) const
        {
            if (s.size() != 4)
            {
                throw std::invalid_argument( "CpuAttentionOp: expected head-major shape [B, NH, T, hs]" );
            }

            if (s[1] != config_.getNumHeads())
            {
                throw std::invalid_argument( "CpuAttentionOp: NH (shape[1]) must match config.num_heads" );
            }

            if (s[3] <= 0)
            {
                throw std::invalid_argument( "CpuAttentionOp: head size (hs) must be > 0" );
            }

            if ((s[1] * s[3]) != config_.getEmbeddingDim())
            {
                throw std::invalid_argument( "CpuAttentionOp: NH * hs must equal embedding_dim from config" );
            }
        }

        void validateHeadMajorShapes(
            const ITensor& q, const ITensor& k, const ITensor& v, const ITensor& out ) const
        {
            const auto& qshape = q.shape();
            const auto& kshape = k.shape();
            const auto& vshape = v.shape();
            const auto& oshape = out.shape();

            validateHeadMajorShape( qshape );

            if (kshape != qshape || vshape != qshape)
            {
                throw std::invalid_argument( "CpuAttentionOp: Q, K, V must have identical shapes [B, NH, T, hs]" );
            }

            if (oshape != qshape)
            {
                throw std::invalid_argument( "CpuAttentionOp: output must have same head-major shape as inputs" );
            }
        }

        void validateHeadMajorShapesForBackward(
            const ITensor& q, const ITensor& k, const ITensor& v,
            const ITensor& out_grad,
            const ITensor& q_grad, const ITensor& k_grad, const ITensor& v_grad ) const
        {
            validateHeadMajorShape( q.shape() );

            if (k.shape() != q.shape() || v.shape() != q.shape())
            {
                throw std::invalid_argument( "CpuAttentionOp: Q, K, V must have identical shapes [B, NH, T, hs]" );
            }

            if (out_grad.shape() != q.shape())
            {
                throw std::invalid_argument( "CpuAttentionOp: output_grad must have same head-major shape as inputs" );
            }

            if (q_grad.shape() != q.shape() || k_grad.shape() != q.shape() || v_grad.shape() != q.shape())
            {
                throw std::invalid_argument( "CpuAttentionOp: q_grad/k_grad/v_grad must have same head-major shape as inputs" );
            }
        }

        void allocateStateTensors()
        {
            auto device = context_->getDevice();

            const int64_t B = cached_batch_size_;
            const int64_t NH = cached_num_heads_;
            const int64_t T = cached_seq_length_;

            shape_t score_shape = { B, NH, T, T };

            preatt_cache_ = std::make_shared<TensorType>( device, score_shape );
            preatt_cache_->setName( "preatt_cache" );

            att_cache_ = std::make_shared<TensorType>( device, score_shape );
            att_cache_->setName( "att_cache" );
        }
    };

    // Registrar: register as ternary operation
    export class CpuAttentionOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "AttentionOp";

            OperationRegistry::instance().registerTernaryOperation<
                DeviceType::Cpu, TensorDataType::FP32, TensorDataType::FP32, TensorDataType::FP32, TensorDataType::FP32>(
                    opName,
                    []( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context,
                        const ConfigurationBase& config ) -> std::shared_ptr<TernaryOperation<DeviceType::Cpu, TensorDataType::FP32>>
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