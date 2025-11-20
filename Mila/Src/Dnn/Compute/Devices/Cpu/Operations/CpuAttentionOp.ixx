/**
 * @file CpuAttentionOp.ixx
 * @brief CPU implementation of the Multi-Head Attention operation (concatenated QKV input).
 *
 * Accepts a single input tensor in model layout [B, T, 3 * embedding_dim] where
 * the last dimension concatenates Q, K and V (each of size embedding_dim). Internally
 * the operator splits and computes attention per-head and returns output in model
 * layout [B, T, embedding_dim].
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
import Dnn.ModuleConfig;
import Compute.Precision;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
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
     * This variant expects a single concatenated input in model-layout:
     *   input:  [B, T, 3 * embedding_dim]  (Q || K || V)
     *   output: [B, T, embedding_dim]
     *
     * Internally splits input into Q/K/V, computes attention per-head and writes
     * output back in model-layout.
     */
    export class CpuAttentionOp : public UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>
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

        // Build expects model-layout input shape: [B, T, 3 * embedding_dim]
        void build( const shape_t& input_shape ) override
        {
            if (is_built_)
            {
                return;
            }

            validateModelLayoutShape( input_shape );

            cached_batch_size_ = input_shape[0];
            cached_seq_length_ = input_shape[1];

            cached_qkv_dim_ = input_shape[2];
            cached_embedding_dim_ = config_.getEmbeddingDim();

            if (cached_qkv_dim_ != 3 * cached_embedding_dim_)
            {
                throw std::invalid_argument( "CpuAttentionOp: input last-dimension must equal 3 * embedding_dim (Q||K||V)" );
            }

            cached_num_heads_ = config_.getNumHeads();
            if (cached_num_heads_ <= 0) throw std::invalid_argument( "CpuAttentionOp: num_heads must be > 0" );

            cached_head_size_ = static_cast<int64_t>(cached_embedding_dim_ / cached_num_heads_);
            if (cached_head_size_ * cached_num_heads_ != cached_embedding_dim_)
            {
                throw std::invalid_argument( "CpuAttentionOp: embedding_dim must be divisible by num_heads" );
            }

            // Allocate state tensors (attention scores / weights)
            allocateStateTensors();

            is_built_ = true;
        }

        void setParameters( ITensor* /*unused1*/, ITensor* /*unused2*/ ) override
        {
            // No learnable parameters
        }

        void setGradients( ITensor* /*unused1*/, ITensor* /*unused2*/ ) override
        {
            // No learnable parameters
        }

        /**
         * Forward:
         *  - input_qkv: [B, T, 3 * embedding_dim]  (Q || K || V)
         *  - output:    [B, T, embedding_dim]
         */
        void forward( const ITensor& input_qkv, ITensor& output ) const override
        {
            if (!is_built_)
            {
                throw std::runtime_error( "CpuAttentionOp: forward called before build()" );
            }

            validateModelLayoutShapes( input_qkv, output );

            const float* in_data = static_cast<const float*>(static_cast<const ITensor&>(input_qkv).rawData());
            float* out_data = static_cast<float*>(output.rawData());

            const int64_t B = cached_batch_size_;
            const int64_t T = cached_seq_length_;
            const int64_t D = cached_embedding_dim_;
            const int64_t NH = cached_num_heads_;
            const int64_t hs = cached_head_size_;

            const float scale = 1.0f / std::sqrt( static_cast<float>(hs) );

            float* preatt_data = preatt_cache_->data();
            float* att_data = att_cache_->data();

            // Convenience offsets for Q/K/V within the last-dimension
            const int64_t q_offset_in_last = 0;
            const int64_t k_offset_in_last = D;
            const int64_t v_offset_in_last = 2 * D;
            const int64_t last_stride = 3 * D;

            // Step 1: compute scores per head: scores[b, h, i, j] = (Q_bh[i] · K_bh[j]) * scale
#pragma omp parallel for collapse(2)
            for (int64_t b = 0; b < B; b++)
            {
                for (int64_t h = 0; h < NH; h++)
                {
                    const int64_t score_offset = b * NH * T * T + h * T * T;
                    for (int64_t i = 0; i < T; i++)
                    {
                        for (int64_t j = 0; j < T; j++)
                        {
                            float sum = 0.0f;
                            // sum over head-size
                            for (int64_t k = 0; k < hs; k++)
                            {
                                // embedding index within embedding_dim
                                int64_t emb_idx = h * hs + k;

                                // Q at (b, i, emb_idx) -> in_data[ ((b*T)+i) * (3*D) + q_offset + emb_idx ]
                                const int64_t q_index = ((b * T + i) * last_stride) + q_offset_in_last + emb_idx;
                                const int64_t k_index = ((b * T + j) * last_stride) + k_offset_in_last + emb_idx;

                                const float qv = in_data[q_index];
                                const float kv = in_data[k_index];

                                sum += qv * kv;
                            }
                            preatt_data[score_offset + i * T + j] = sum * scale;
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

            // Step 3: Out = Att × V  => write into model-layout output [B, T, D]
#pragma omp parallel for collapse(2)
            for (int64_t b = 0; b < B; b++)
            {
                for (int64_t h = 0; h < NH; h++)
                {
                    const int64_t score_offset = b * NH * T * T + h * T * T;
                    for (int64_t i = 0; i < T; i++)
                    {
                        for (int64_t k = 0; k < hs; k++)
                        {
                            float sum = 0.0f;
                            for (int64_t j = 0; j < T; j++)
                            {
                                // V at (b, j, emb_idx) with emb_idx = h*hs + k
                                int64_t emb_idx = h * hs + k;
                                const int64_t v_index = ((b * T + j) * last_stride) + v_offset_in_last + emb_idx;

                                sum += att_data[score_offset + i * T + j] * in_data[v_index];
                            }

                            // Write to output model-layout: out[(b*T + i)*D + (h*hs + k)]
                            out_data[(b * T + i) * D + (h * hs + k)] = sum;
                        }
                    }
                }
            }
        }

        /**
         * Backward:
         *  - input_qkv: [B, T, 3*D]  (Q||K||V)
         *  - output_grad: [B, T, D]
         *  - input_grad: [B, T, 3*D] (written)
         *
         * Implements gradient propagation for the above forward.
         */
        void backward(
            const ITensor& input_qkv,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            if (!is_built_)
            {
                throw std::runtime_error( "CpuAttentionOp: backward called before build()" );
            }

            validateModelLayoutShapesForBackward( input_qkv, output_grad, input_grad );

            const float* in_data = static_cast<const float*>(static_cast<const ITensor&>(input_qkv).rawData());
            const float* dY = static_cast<const float*>(output_grad.rawData());
            float* dIn = static_cast<float*>(input_grad.rawData());

            const int64_t B = cached_batch_size_;
            const int64_t T = cached_seq_length_;
            const int64_t D = cached_embedding_dim_;
            const int64_t NH = cached_num_heads_;
            const int64_t hs = cached_head_size_;

            const float scale = 1.0f / std::sqrt( static_cast<float>(hs) );

            float* preatt_data = preatt_cache_->data();
            float* att_data = att_cache_->data();

            const int64_t last_stride = 3 * D;
            const int64_t q_offset_in_last = 0;
            const int64_t k_offset_in_last = D;
            const int64_t v_offset_in_last = 2 * D;

            // Zero input_grad
            std::memset( dIn, 0, static_cast<size_t>(B) * T * (3 * D) * sizeof( float ) );

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
                    const int64_t score_offset = b * NH * T * T + h * T * T;

                    // Pointers are computed on-the-fly from concatenated input
                    for (int64_t t = 0; t < T; t++)
                    {
                        for (int64_t t2 = 0; t2 < T; t2++)
                        {
                            for (int64_t k = 0; k < hs; k++)
                            {
                                // indices for V and output_grad
                                int64_t emb_idx = h * hs + k;

                                // dY at (b, t, emb_idx) -> dY[ (b*T + t)*D + emb_idx ]
                                const float dout_val = dY[(b * T + t) * D + emb_idx];

                                // Accumulate dAtt and dV
                                // dAtt[t, t2] += dout[t, emb_idx] * V[t2, emb_idx]
                                // dV[t2, emb_idx] += Att[t, t2] * dout[t, emb_idx]
                                const int64_t v_index = ((b * T + t2) * last_stride) + v_offset_in_last + emb_idx;
                                const float v_val = in_data[v_index];

                                // atomic accumulation is not used here; omp may cause race if parallelized across same locations.
                                // We parallelized outer loops (b,h) so inner loops are safe.
                                datt_data[score_offset + t * T + t2] += dout_val * v_val;
                                // accumulate into dV at (b, t2, emb_idx) -> dIn position for V segment
                                dIn[(b * T + t2) * last_stride + v_offset_in_last + emb_idx] += att_data[score_offset + t * T + t2] * dout_val;
                            }
                        }
                    }

                    // softmax backward (causal): dpreatt += J_softmax^T * dAtt
                    for (int64_t t = 0; t < T; t++)
                    {
                        const float* att_row = att_data + score_offset + t * T;
                        const float* datt_row = datt_data + score_offset + t * T;
                        float* dpreatt_row = dpreatt_data + score_offset + t * T;

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
                            const float coeff = dpreatt_data[score_offset + t * T + t2] * scale;
                            for (int64_t k = 0; k < hs; k++)
                            {
                                int64_t emb_idx = h * hs + k;

                                // Q at (b, t, emb_idx) -> in_data[ ((b*T)+t)*last_stride + q_offset + emb_idx ]
                                const int64_t q_index = ((b * T + t) * last_stride) + q_offset_in_last + emb_idx;
                                const int64_t k_index = ((b * T + t2) * last_stride) + k_offset_in_last + emb_idx;

                                const float q_val = in_data[q_index];
                                const float k_val = in_data[k_index];

                                // accumulate into dQ and dK locations inside dIn (for Q and K segments)
                                dIn[(b * T + t) * last_stride + q_offset_in_last + emb_idx] += coeff * k_val;
                                dIn[(b * T + t2) * last_stride + k_offset_in_last + emb_idx] += coeff * q_val;
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

        // Cached dimensions for model-layout / per-head computations
        int64_t cached_batch_size_{ 0 };
        int64_t cached_seq_length_{ 0 };
        int64_t cached_qkv_dim_{ 0 };
        int64_t cached_embedding_dim_{ 0 };
        int64_t cached_num_heads_{ 0 };
        int64_t cached_head_size_{ 0 };

        // State tensors: attention scores/weights [B, NH, T, T]
        mutable std::shared_ptr<TensorType> preatt_cache_{ nullptr };
        mutable std::shared_ptr<TensorType> att_cache_{ nullptr };

        void validateModelLayoutShape( const shape_t& s ) const
        {
            if (s.size() != 3)
            {
                throw std::invalid_argument( "CpuAttentionOp: expected model-layout shape [B, T, 3*embedding_dim]" );
            }
        }

        void validateModelLayoutShapes( const ITensor& in, const ITensor& out ) const
        {
            const auto& ishape = in.shape();
            const auto& oshape = out.shape();

            validateModelLayoutShape( ishape );

            const int64_t D_expected = config_.getEmbeddingDim();
            if (ishape[2] != 3 * D_expected)
            {
                throw std::invalid_argument( "CpuAttentionOp: input last-dimension must be 3 * embedding_dim" );
            }

            if (oshape.size() != 3 || oshape[2] != D_expected)
            {
                throw std::invalid_argument( "CpuAttentionOp: output must be [B, T, embedding_dim]" );
            }

            if (oshape[0] != ishape[0] || oshape[1] != ishape[1])
            {
                throw std::invalid_argument( "CpuAttentionOp: input and output batch/sequence dimensions must match" );
            }
        }

        void validateModelLayoutShapesForBackward(
            const ITensor& in, const ITensor& out_grad, const ITensor& in_grad ) const
        {
            const auto& ishape = in.shape();

            validateModelLayoutShape( ishape );

            const int64_t D_expected = config_.getEmbeddingDim();
            if (ishape[2] != 3 * D_expected)
            {
                throw std::invalid_argument( "CpuAttentionOp: input last-dimension must be 3 * embedding_dim" );
            }

            if (out_grad.shape().size() != 3 || out_grad.shape()[2] != D_expected)
            {
                throw std::invalid_argument( "CpuAttentionOp: output_grad must be [B, T, embedding_dim]" );
            }

            if (in_grad.shape().size() != 3 || in_grad.shape()[2] != 3 * D_expected)
            {
                throw std::invalid_argument( "CpuAttentionOp: input_grad must be [B, T, 3*embedding_dim]" );
            }

            if (out_grad.shape()[0] != ishape[0] || out_grad.shape()[1] != ishape[1] ||
                in_grad.shape()[0] != ishape[0] || in_grad.shape()[1] != ishape[1])
            {
                throw std::invalid_argument( "CpuAttentionOp: batch/sequence dimensions must match between tensors" );
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

    export class CpuAttentionOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "AttentionOp";

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cpu, TensorDataType::FP32, TensorDataType::FP32>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context,
                    const ModuleConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>>
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