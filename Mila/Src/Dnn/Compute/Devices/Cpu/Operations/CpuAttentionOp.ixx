/**
 * @file CpuAttentionOp.ixx
 * @brief CPU implementation of Multi-Head Attention operation.
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
#include <cassert>
#include <sstream>
#include <iostream>
#ifdef USE_OMP
#include <omp.h>
#endif

export module Compute.CpuAttention;

import Dnn.Components.MultiHeadAttention;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ComponentConfig;
import Compute.Precision;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.OperationType;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.CpuDevice;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Utils.Logger;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief CPU implementation of Multi-Head Attention operation.
     *
     * Design philosophy:
     * - Two-phase initialization: build() creates all required tensors, forward()/backward() use them
     * - All dimension computation and tensor allocation happens once in build()
     * - Forward/backward are hot-path methods with zero setup overhead
     *
     * Forward pass:
     *  1. Permute QKV from [B, T, 3*C] to separate Q, K, V [B, NH, T, HS]
     *  2. Compute attention scores: preatt = Q @ K^T
     *  3. Apply softmax with causal masking: att = softmax(preatt / sqrt(HS))
     *  4. Compute values: v_out = Att @ V
     *  5. Unpermute output from [B, NH, T, HS] to [B, T, C]
     *
     * Backward pass:
     *  1. Unpermute output gradient to [B, NH, T, HS]
     *  2. Compute dV = Att^T @ dVout
     *  3. Compute dAtt = dVout @ V^T
     *  4. Softmax backward: dPreatt = softmax_backward(dAtt, Att) * scale
     *  5. Compute dQ = dPreatt @ K
     *  6. Compute dK = dPreatt^T @ Q
     *  7. Permute gradients back to concatenated QKV format
     */
    export class CpuAttentionOp : public UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using MR = CpuMemoryResource;
        using CpuExecutionContext = ExecutionContext<DeviceType::Cpu>;
        using TensorType = Tensor<TensorDataType::FP32, MR>;

        explicit CpuAttentionOp( IExecutionContext* context, const MultiHeadAttentionConfig& config )
            : context_( context ), config_( config )
        {
            if ( !context_ )
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();
        }

        ~CpuAttentionOp() override = default;

        void build( const shape_t& input_shape ) override
        {
            if ( is_built_ )
            {
                return;
            }

            validateInputShape( input_shape );

            B_ = static_cast<int>(input_shape[ 0 ]);
            T_ = static_cast<int>(input_shape[ 1 ]);
            qkv_dim_ = static_cast<int>(input_shape[ 2 ]);

            embedding_dim_ = qkv_dim_ / 3;
            NH_ = config_.getNumHeads();
            HS_ = embedding_dim_ / NH_;

            if ( embedding_dim_ % NH_ != 0 )
            {
                throw std::invalid_argument( "CpuAttentionOp: embedding_dim must be divisible by num_heads" );
            }

            allocateStateTensors();

            is_built_ = true;
        }

        void setParameters( ITensor* /*unused1*/, ITensor* /*unused2*/ ) override
        {}

        void setGradients( ITensor* /*unused1*/, ITensor* /*unused2*/ ) override
        {}

        void forward( const ITensor& input, ITensor& output ) const override
        {
            assert( is_built_ && "CpuAttentionOp must be built before calling forward()" );

            const float* X = static_cast<const float*>(input.rawData());
            float* Y = static_cast<float*>(output.rawData());

            const float scale = 1.0f / std::sqrt( static_cast<float>(HS_) );

            permuteQKV( X );

            computeAttentionScores( scale );

            applySoftmax();

            computeOutputValues();

            unpermute( Y );
        }

        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            assert( is_built_ && "CpuAttentionOp must be built before calling backward()" );

            const float* dY = static_cast<const float*>(output_grad.rawData());
            float* dX = static_cast<float*>(input_grad.rawData());

            const float scale = 1.0f / std::sqrt( static_cast<float>(HS_) );

            std::memset( dX, 0, static_cast<size_t>(B_) * T_ * qkv_dim_ * sizeof( float ) );

            unpermute_backward( dY );

            computeGradientV();

            computeGradientAtt();

            computeGradientPreatt( scale );

            computeGradientQ();

            computeGradientK();

            permute_backward( dX );
        }

        OperationType getOperationType() const override
        {
            return OperationType::MultiHeadAttentionOp;
        }

        std::string getName() const override
        {
            return "CpuAttentionOp";
        }

    private:
        IExecutionContext* context_;
        MultiHeadAttentionConfig config_;
        bool is_built_{ false };

        int B_{ 0 };
        int T_{ 0 };
        int qkv_dim_{ 0 };
        int embedding_dim_{ 0 };
        int NH_{ 0 };
        int HS_{ 0 };

        std::shared_ptr<TensorType> q_tensor_;
        std::shared_ptr<TensorType> k_tensor_;
        std::shared_ptr<TensorType> v_tensor_;
        std::shared_ptr<TensorType> preatt_tensor_;
        std::shared_ptr<TensorType> att_tensor_;
        std::shared_ptr<TensorType> v_out_tensor_;

        std::shared_ptr<TensorType> dq_tensor_;
        std::shared_ptr<TensorType> dk_tensor_;
        std::shared_ptr<TensorType> dv_tensor_;
        std::shared_ptr<TensorType> dpreatt_tensor_;
        std::shared_ptr<TensorType> datt_tensor_;
        std::shared_ptr<TensorType> dvout_tensor_;

        float* q_{ nullptr };
        float* k_{ nullptr };
        float* v_{ nullptr };
        float* preatt_{ nullptr };
        float* att_{ nullptr };
        float* v_out_{ nullptr };

        float* dq_{ nullptr };
        float* dk_{ nullptr };
        float* dv_{ nullptr };
        float* dpreatt_{ nullptr };
        float* datt_{ nullptr };
        float* dvout_{ nullptr };

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.size() != 3 )
            {
                throw std::invalid_argument(
                    "CpuAttentionOp: input must have rank 3 (batch_size, seq_length, 3*embedding_dim)" );
            }

            const int64_t expected_qkv_dim = 3 * config_.getModelDim();

            if ( input_shape[ 2 ] != expected_qkv_dim )
            {
                throw std::invalid_argument(
                    "CpuAttentionOp: input last dimension must be 3 * model_dim (Q, K, V concatenated)" );
            }
        }

        void allocateStateTensors()
        {
            auto device = context_->getDeviceId();

            shape_t qkv_shape = { B_, NH_, T_, HS_ };

            q_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            q_tensor_->setName( "q_" );
            q_ = q_tensor_->data();

            k_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            k_tensor_->setName( "k_" );
            k_ = k_tensor_->data();

            v_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            v_tensor_->setName( "v_" );
            v_ = v_tensor_->data();

            shape_t att_shape = { B_, NH_, T_, T_ };

            preatt_tensor_ = std::make_shared<TensorType>( device, att_shape );
            preatt_tensor_->setName( "preatt_" );
            preatt_ = preatt_tensor_->data();

            att_tensor_ = std::make_shared<TensorType>( device, att_shape );
            att_tensor_->setName( "att_" );
            att_ = att_tensor_->data();

            shape_t v_out_shape = { B_, NH_, T_, HS_ };

            v_out_tensor_ = std::make_shared<TensorType>( device, v_out_shape );
            v_out_tensor_->setName( "v_out_" );
            v_out_ = v_out_tensor_->data();

            dq_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            dq_tensor_->setName( "dq_" );
            dq_ = dq_tensor_->data();

            dk_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            dk_tensor_->setName( "dk_" );
            dk_ = dk_tensor_->data();

            dv_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            dv_tensor_->setName( "dv_" );
            dv_ = dv_tensor_->data();

            dpreatt_tensor_ = std::make_shared<TensorType>( device, att_shape );
            dpreatt_tensor_->setName( "dpreatt_" );
            dpreatt_ = dpreatt_tensor_->data();

            datt_tensor_ = std::make_shared<TensorType>( device, att_shape );
            datt_tensor_->setName( "datt_" );
            datt_ = datt_tensor_->data();

            dvout_tensor_ = std::make_shared<TensorType>( device, v_out_shape );
            dvout_tensor_->setName( "dvout_" );
            dvout_ = dvout_tensor_->data();
        }

        void permuteQKV( const float* X ) const
        {
        #pragma omp parallel for collapse(2)
            for ( int b = 0; b < B_; b++ )
            {
                for ( int h = 0; h < NH_; h++ )
                {
                    for ( int t = 0; t < T_; t++ )
                    {
                        for ( int k = 0; k < HS_; k++ )
                        {
                            const int emb_idx = h * HS_ + k;
                            const size_t base = static_cast<size_t>( (b * T_ + t) * qkv_dim_ );
                            const size_t idx = static_cast<size_t>( ((b * NH_ + h) * T_ + t) * HS_ + k );

                            q_[ idx ] = X[ base + emb_idx ];
                            k_[ idx ] = X[ base + embedding_dim_ + emb_idx ];
                            v_[ idx ] = X[ base + 2 * embedding_dim_ + emb_idx ];
                        }
                    }
                }
            }
        }

        void computeAttentionScores( float scale ) const
        {
        #pragma omp parallel for collapse(2)
            for ( int b = 0; b < B_; b++ )
            {
                for ( int h = 0; h < NH_; h++ )
                {
                    const size_t score_offset = static_cast<size_t>( (b * NH_ + h) * T_ * T_ );

                    for ( int i = 0; i < T_; i++ )
                    {
                        for ( int j = 0; j < T_; j++ )
                        {
                            float sum = 0.0f;

                            for ( int k = 0; k < HS_; k++ )
                            {
                                const size_t qi = static_cast<size_t>( ((b * NH_ + h) * T_ + i) * HS_ + k );
                                const size_t kj = static_cast<size_t>( ((b * NH_ + h) * T_ + j) * HS_ + k );

                                sum += q_[ qi ] * k_[ kj ];
                            }

                            preatt_[ score_offset + i * T_ + j ] = sum * scale;
                        }
                    }
                }
            }
        }

        void applySoftmax() const
        {
        #pragma omp parallel for collapse(3)
            for ( int b = 0; b < B_; b++ )
            {
                for ( int h = 0; h < NH_; h++ )
                {
                    for ( int t = 0; t < T_; t++ )
                    {
                        const size_t score_offset = static_cast<size_t>( (b * NH_ + h) * T_ * T_ );
                        float* scores_row = preatt_ + score_offset + t * T_;
                        float* att_row = att_ + score_offset + t * T_;

                        float maxval = -INFINITY;
                        for ( int t2 = 0; t2 <= t; t2++ )
                        {
                            if ( scores_row[ t2 ] > maxval ) maxval = scores_row[ t2 ];
                        }

                        float expsum = 0.0f;
                        for ( int t2 = 0; t2 <= t; t2++ )
                        {
                            float expv = std::exp( scores_row[ t2 ] - maxval );
                            att_row[ t2 ] = expv;
                            expsum += expv;
                        }

                        const float expsum_inv = (expsum > 0.0f) ? (1.0f / expsum) : 0.0f;
                        for ( int t2 = 0; t2 <= t; t2++ )
                        {
                            att_row[ t2 ] *= expsum_inv;
                        }

                        for ( int t2 = t + 1; t2 < T_; t2++ )
                        {
                            att_row[ t2 ] = 0.0f;
                        }
                    }
                }
            }
        }

        void computeOutputValues() const
        {
            std::memset( v_out_, 0, static_cast<size_t>(B_) * NH_ * T_ * HS_ * sizeof( float ) );

        #pragma omp parallel for collapse(2)
            for ( int b = 0; b < B_; b++ )
            {
                for ( int h = 0; h < NH_; h++ )
                {
                    const size_t score_offset = static_cast<size_t>( (b * NH_ + h) * T_ * T_ );

                    for ( int i = 0; i < T_; i++ )
                    {
                        for ( int k = 0; k < HS_; k++ )
                        {
                            float sum = 0.0f;

                            for ( int j = 0; j < T_; j++ )
                            {
                                const size_t vidx = static_cast<size_t>( ((b * NH_ + h) * T_ + j) * HS_ + k );
                                sum += att_[ score_offset + i * T_ + j ] * v_[ vidx ];
                            }

                            const size_t outidx = static_cast<size_t>( ((b * NH_ + h) * T_ + i) * HS_ + k );
                            v_out_[ outidx ] = sum;
                        }
                    }
                }
            }
        }

        void unpermute( float* Y ) const
        {
        #pragma omp parallel for collapse(2)
            for ( int b = 0; b < B_; b++ )
            {
                for ( int i = 0; i < T_; i++ )
                {
                    for ( int h = 0; h < NH_; h++ )
                    {
                        for ( int k = 0; k < HS_; k++ )
                        {
                            const int emb_idx = h * HS_ + k;
                            const size_t vidx = static_cast<size_t>( ((b * NH_ + h) * T_ + i) * HS_ + k );
                            Y[ (b * T_ + i) * embedding_dim_ + emb_idx ] = v_out_[ vidx ];
                        }
                    }
                }
            }
        }

        void unpermute_backward( const float* dY ) const
        {
        #pragma omp parallel for collapse(2)
            for ( int b = 0; b < B_; b++ )
            {
                for ( int h = 0; h < NH_; h++ )
                {
                    for ( int t = 0; t < T_; t++ )
                    {
                        for ( int k = 0; k < HS_; k++ )
                        {
                            const int emb_idx = h * HS_ + k;
                            const size_t idx = static_cast<size_t>( ((b * NH_ + h) * T_ + t) * HS_ + k );
                            dvout_[ idx ] = dY[ (b * T_ + t) * embedding_dim_ + emb_idx ];
                        }
                    }
                }
            }
        }

        void computeGradientV() const
        {
            std::memset( dv_, 0, static_cast<size_t>(B_) * NH_ * T_ * HS_ * sizeof( float ) );

        #pragma omp parallel for collapse(2)
            for ( int b = 0; b < B_; b++ )
            {
                for ( int h = 0; h < NH_; h++ )
                {
                    const size_t score_offset = static_cast<size_t>( (b * NH_ + h) * T_ * T_ );

                    for ( int j = 0; j < T_; j++ )
                    {
                        for ( int k = 0; k < HS_; k++ )
                        {
                            float sum = 0.0f;

                            for ( int i = 0; i < T_; i++ )
                            {
                                const size_t dvout_idx = static_cast<size_t>( ((b * NH_ + h) * T_ + i) * HS_ + k );
                                sum += att_[ score_offset + i * T_ + j ] * dvout_[ dvout_idx ];
                            }

                            const size_t dv_idx = static_cast<size_t>( ((b * NH_ + h) * T_ + j) * HS_ + k );
                            dv_[ dv_idx ] = sum;
                        }
                    }
                }
            }
        }

        void computeGradientAtt() const
        {
            std::memset( datt_, 0, static_cast<size_t>(B_) * NH_ * T_ * T_ * sizeof( float ) );

        #pragma omp parallel for collapse(2)
            for ( int b = 0; b < B_; b++ )
            {
                for ( int h = 0; h < NH_; h++ )
                {
                    const size_t score_offset = static_cast<size_t>( (b * NH_ + h) * T_ * T_ );

                    for ( int i = 0; i < T_; i++ )
                    {
                        for ( int j = 0; j < T_; j++ )
                        {
                            float sum = 0.0f;

                            for ( int k = 0; k < HS_; k++ )
                            {
                                const size_t dvout_idx = static_cast<size_t>( ((b * NH_ + h) * T_ + i) * HS_ + k );
                                const size_t v_idx = static_cast<size_t>( ((b * NH_ + h) * T_ + j) * HS_ + k );
                                sum += dvout_[ dvout_idx ] * v_[ v_idx ];
                            }

                            datt_[ score_offset + i * T_ + j ] = sum;
                        }
                    }
                }
            }
        }

        void computeGradientPreatt( float scale ) const
        {
            std::memset( dpreatt_, 0, static_cast<size_t>(B_) * NH_ * T_ * T_ * sizeof( float ) );

        #pragma omp parallel for collapse(2)
            for ( int b = 0; b < B_; b++ )
            {
                for ( int h = 0; h < NH_; h++ )
                {
                    const size_t score_offset = static_cast<size_t>( (b * NH_ + h) * T_ * T_ );

                    for ( int t = 0; t < T_; t++ )
                    {
                        const float* att_row = att_ + score_offset + t * T_;
                        const float* datt_row = datt_ + score_offset + t * T_;
                        float* dpreatt_row = dpreatt_ + score_offset + t * T_;

                        for ( int t2 = 0; t2 <= t; t2++ )
                        {
                            for ( int t3 = 0; t3 <= t; t3++ )
                            {
                                const float indicator = (t2 == t3) ? 1.0f : 0.0f;
                                const float local_derivative = att_row[ t2 ] * (indicator - att_row[ t3 ]);
                                dpreatt_row[ t3 ] += local_derivative * datt_row[ t2 ];
                            }
                        }

                        for ( int t2 = 0; t2 <= t; t2++ )
                        {
                            dpreatt_row[ t2 ] *= scale;
                        }
                    }
                }
            }

            // DEBUG: Print pre-attention scores and dPreatt tensors
            /*std::cout << "Computed gradient dPreatt" << std::endl;
            std::cout << dpreatt_tensor_->toString( true ) << std::endl;*/
        }

        void computeGradientQ() const
        {
            std::memset( dq_, 0, static_cast<size_t>(B_) * NH_ * T_ * HS_ * sizeof( float ) );

        #pragma omp parallel for collapse(2)
            for ( int b = 0; b < B_; b++ )
            {
                for ( int h = 0; h < NH_; h++ )
                {
                    const size_t score_offset = static_cast<size_t>( (b * NH_ + h) * T_ * T_ );

                    for ( int i = 0; i < T_; i++ )
                    {
                        for ( int k = 0; k < HS_; k++ )
                        {
                            float sum = 0.0f;

                            for ( int j = 0; j < T_; j++ )
                            {
                                const size_t k_idx = static_cast<size_t>( ((b * NH_ + h) * T_ + j) * HS_ + k );
                                sum += dpreatt_[ score_offset + i * T_ + j ] * k_[ k_idx ];
                            }

                            const size_t dq_idx = static_cast<size_t>( ((b * NH_ + h) * T_ + i) * HS_ + k );
                            dq_[ dq_idx ] = sum;
                        }
                    }
                }
            }

            // DEBUG: Print K and dQ tensors
            /*std::cout << "Key tensor K" << std::endl;
            std::cout << k_tensor_->toString( true ) << std::endl;

            std::cout << "Computed gradient dQ" << std::endl;
            std::cout << dq_tensor_->toString( true ) << std::endl;*/
        }

        void computeGradientK() const
        {
            std::memset( dk_, 0, static_cast<size_t>(B_) * NH_ * T_ * HS_ * sizeof( float ) );

        #pragma omp parallel for collapse(2)
            for ( int b = 0; b < B_; b++ )
            {
                for ( int h = 0; h < NH_; h++ )
                {
                    const size_t score_offset = static_cast<size_t>( (b * NH_ + h) * T_ * T_ );

                    for ( int j = 0; j < T_; j++ )
                    {
                        for ( int k = 0; k < HS_; k++ )
                        {
                            float sum = 0.0f;

                            for ( int i = 0; i < T_; i++ )
                            {
                                const size_t q_idx = static_cast<size_t>( ((b * NH_ + h) * T_ + i) * HS_ + k );
                                sum += dpreatt_[ score_offset + i * T_ + j ] * q_[ q_idx ];
                            }

                            const size_t dk_idx = static_cast<size_t>( ((b * NH_ + h) * T_ + j) * HS_ + k );
                            dk_[ dk_idx ] = sum;
                        }
                    }
                }
            }
        }

        void permute_backward( float* dX ) const
        {
        #pragma omp parallel for collapse(2)
            for ( int b = 0; b < B_; b++ )
            {
                for ( int h = 0; h < NH_; h++ )
                {
                    for ( int t = 0; t < T_; t++ )
                    {
                        for ( int k = 0; k < HS_; k++ )
                        {
                            const int emb_idx = h * HS_ + k;
                            const size_t idx = static_cast<size_t>( ((b * NH_ + h) * T_ + t) * HS_ + k );
                            const size_t base = static_cast<size_t>( (b * T_ + t) * qkv_dim_ );

                            dX[ base + emb_idx ] = dq_[ idx ];
                            dX[ base + embedding_dim_ + emb_idx ] = dk_[ idx ];
                            dX[ base + 2 * embedding_dim_ + emb_idx ] = dv_[ idx ];
                        }
                    }
                }
            }
        }
    };

    export class CpuAttentionOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string_view opName = Compute::OperationNames::MultiHeadAttention;

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cpu, TensorDataType::FP32, TensorDataType::FP32>(
                opName,
                []( IExecutionContext* context, const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>>
                {
                    const auto& attention_config = dynamic_cast<const MultiHeadAttentionConfig&>(config);

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