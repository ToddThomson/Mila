/**
 * @file CudaAttentionOp.ixx
 * @brief CUDA implementation of Multi-Head Attention cuBLASLt optimization.
 */

module;
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <type_traits>
#include <sstream>
#include <cassert>
#include "Kernels/CudaAttention.cuh"

export module Compute.CudaAttentionOp;
import :Plans;
import :Dispatch;

import Dnn.Components.Attention;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.Precision;
import Compute.OperationRegistry;
import Compute.Device;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.CudaDevice;
import Compute.KVCacheable;
import CublasLt.Error;
import CublasLtHelpers;
import Utils.Logger;

namespace Mila::Dnn::Compute::Cuda::Attention
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute::Cuda::Common::CublasLtHelpers;

    /**
     * @brief CUDA implementation of Multi-Head Attention using column-major cuBLASLt optimization.
     *
     * Design philosophy:
     * - Two-phase initialization: build() creates cuBLASLt plans, forward()/backward() execute them
     * - Column-major layout eliminates most transpose operations in cuBLASLt
     * - All dimension computation and algorithm selection happens once in build()
     * - Forward/backward are hot-path methods with zero setup overhead
     * - Custom CUDA kernels handle permute/unpermute and softmax operations
     *
     * Forward pass:
     *  1. Permute QKV from [B, T, 3*C] to separate Q, K, V [B, NH, HS, T] (column-major)
     *  2. Compute attention scores: preatt = Q^T @ K (exploiting column-major layout)
     *  3. Apply softmax with causal masking: att = softmax(preatt / sqrt(HS))
     *  4. Compute values: vaccum = Att @ V^T
     *  5. Unpermute output from [B, NH, HS, T] to [B, T, C]
     *
     * Backward pass:
     *  1. Unpermute output gradient to [B, NH, HS, T]
     *  2. Compute dV = Att^T @ dvaccum^T
     *  3. Compute dAtt = dvaccum^T @ V
     *  4. Softmax backward: dPreatt = softmax_backward(dAtt, Att)
     *  5. Compute dQ = dPreatt @ K^T
     *  6. Compute dK = dPreatt^T @ Q^T
     *  7. Permute gradients back to concatenated QKV format
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaAttentionOp : public UnaryOperation<DeviceType::Cuda, TPrecision>, public IKVCacheable
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ConfigType = AttentionConfig;

        CudaAttentionOp( IExecutionContext* context, const AttentionConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaAttentionOp" ) ), config_( config )
        {
            config_.validate();
        }

        void setParameters( ITensor* /*unused1*/, ITensor* /*unused2*/ ) override
        {}

        void setGradients( ITensor* /*unused1*/, ITensor* /*unused2*/ ) override
        {}

        void initializeKVCache( int batch_size, int max_seq_length )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "CudaAttentionOp::initializeKVCache requires build() to be called first" );
            }

            if ( batch_size != B_ )
            {
                throw std::invalid_argument( "CudaAttentionOp::initializeKVCache batch size mismatch" );
            }

            if ( max_seq_length <= 0 || max_seq_length > T_ )
            {
                throw std::invalid_argument( "CudaAttentionOp::initializeKVCache max_seq_length out of range" );
            }

            active_max_seq_len_ = max_seq_length;
            cached_seq_len_ = 0;
            kv_cache_enabled_ = true;
        }

        void resetKVCache()
        {
            cached_seq_len_ = 0;
        }

        void forwardPrefill( const ITensor& input, ITensor& output )
        {
            const auto& input_shape = input.shape();

            ensureKVCacheEnabled();

            validatePrefillInputShape( input_shape );

            int actual_seq_len = static_cast<int>(input_shape[ 1 ]);

            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            cudaStream_t stream = context_->getStream();

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const float scale = 1.0f / sqrtf( static_cast<float>(HS_) );

            Detail::cuda_mha_kernels<NativeType>::permute_qkv_padded(
                q_, k_, v_,
                X,
                B_, actual_seq_len, T_, NH_, HS_,
                stream );

            execute_plan<NativeType>(
                cublaslt_handle_,
                qk_score_plan_,
                &scale,
                q_, k_,
                &beta,
                preatt_,
                nullptr,
                stream );

            Detail::cuda_mha_kernels<NativeType>::softmax_padded_forward(
                att_, 1.0f, preatt_,
                B_, NH_, T_, actual_seq_len,
                stream );

            execute_plan<NativeType>(
                cublaslt_handle_,
                att_value_plan_,
                &alpha,
                att_, v_,
                &beta,
                v_out_,
                nullptr,
                stream );

            Detail::cuda_mha_kernels<NativeType>::unpermute_output(
                v_out_, Y,
                B_, T_, NH_, HS_,
                stream );

            cached_seq_len_ = actual_seq_len;
        }

        void forwardDecode( const ITensor& input, ITensor& output, int position )
        {
            const auto& input_shape = input.shape();

            ensureKVCacheEnabled();

            validateDecodeInputShape( input_shape );

            if ( position < 0 || position >= active_max_seq_len_ )
            {
                throw std::invalid_argument( "CudaAttentionOp::forwardDecode position out of range" );
            }

            int actual_len = position + 1;

            const NativeType* X = static_cast<const NativeType*>( input.rawData() );
            NativeType* Y = static_cast<NativeType*>( output.rawData() );

            cudaStream_t stream = context_->getStream();

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const float scale = 1.0f / sqrtf( static_cast<float>(HS_) );

            Detail::cuda_mha_kernels<NativeType>::permute_qkv_decode(
                q_, k_, v_,
                X,
                B_, position, T_, NH_, HS_,
                stream );

            const NativeType* q_decode = q_ + static_cast<int64_t>(position) * HS_;

            execute_plan<NativeType>(
                cublaslt_handle_,
                qk_decode_plan_,
                &scale,
                q_decode, k_,
                &beta,
                preatt_decode_,
                nullptr,
                stream );

            Detail::cuda_mha_kernels<NativeType>::softmax_decode_forward(
                att_decode_, 1.0f, preatt_decode_,
                B_, NH_, T_, actual_len,
                stream );

            execute_plan<NativeType>(
                cublaslt_handle_,
                att_value_decode_plan_,
                &alpha,
                att_decode_, v_,
                &beta,
                v_out_decode_,
                nullptr,
                stream );

            Detail::cuda_mha_kernels<NativeType>::unpermute_output(
                v_out_decode_, Y,
                B_, 1, NH_, HS_,
                stream );

            if ( actual_len > cached_seq_len_ )
            {
                cached_seq_len_ = actual_len;
            }
        }

        void build( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            B_ = static_cast<int>(input_shape[ 0 ]);
            T_ = static_cast<int>(input_shape[ 1 ]);
            qkv_dim_ = static_cast<int>(input_shape[ 2 ]);

            embedding_dim_ = qkv_dim_ / 3;
            NH_ = config_.getNumHeads();
            HS_ = embedding_dim_ / NH_;

            if ( embedding_dim_ % NH_ != 0 )
            {
                throw std::invalid_argument(
                    "CudaAttentionOp: embedding_dim must be divisible by num_heads" );
            }

            active_max_seq_len_ = T_;
            cached_seq_len_ = 0;
            kv_cache_enabled_ = false;

            allocateStateTensors();

            cublaslt_handle_ = context_->getCublasLtHandle();

            if ( cublaslt_handle_ == nullptr )
            {
                throw std::runtime_error(
                    "CudaAttentionOp requires cuBLASLt support. "
                    "Ensure CUDA 10.1 or newer is installed." );
            }

            precision_policy_ = config_.getPrecisionPolicy();

            buildCublasLtPlans();

            UnaryOperationBase::build( input_shape );
        }

        void forward( const ITensor& input, ITensor& output ) const override
        {
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            cudaStream_t stream = context_->getStream();

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const float scale = 1.0f / sqrtf( static_cast<float>(HS_) );

            // Permute QKV from [B, T, 3*C] to separate Q, K, V with shape [B, NH, T, HS]
            Detail::cuda_mha_kernels<NativeType>::permute_qkv(
                q_, k_, v_,
                X,
                B_, T_, NH_, HS_,
                stream );

            context_->synchronize();
            {
                /*shape_t qkv_shape = { B_, NH_, T_, HS_ };
                
                std::string q_dump = dump_tensor<NativeType>(
                    q_, qkv_shape, this->getName() + ".dbg.Q", 4, stream );

                Utils::Logger::info( this->getName() + ": dbg.Q (device dump):\n" + q_dump );

                std::string k_dump = dump_tensor<NativeType>(
                    k_, qkv_shape, this->getName() + ".dbg.K", 4, stream );

                Utils::Logger::info( this->getName() + ": dbg.K (device dump):\n" + k_dump );*/

                /*std::string v_dump = dump_tensor<NativeType>(
                    v_, qkv_shape, this->getName() + ".dbg.V", 4, stream );

                Utils::Logger::info( this->getName() + ": dbg.V (device dump):\n" + v_dump );*/
            }

            // Compute attention scores: preatt = Q @ K^T
            execute_plan<NativeType>(
                cublaslt_handle_,
                qk_score_plan_,
                &scale,
                q_, k_,
                &beta,
                preatt_,
                nullptr,
                stream );

            /*context_->synchronize();
            {
                shape_t preatt_shape = { B_, NH_, T_, T_ };
                std::string preatt_dump = dump_tensor<NativeType>(
                    preatt_, preatt_shape, this->getName() + ".dbg.preatt", 4, stream );

                Utils::Logger::info( this->getName() + ": dbg.preatt (device dump):\n" + preatt_dump );
            }*/

            // att_: [B, NH, T, T] (after softmax)
            Detail::cuda_mha_kernels<NativeType>::softmax_forward(
                att_, 1.0f, preatt_,
                B_, NH_, T_,
                stream );

            /*context_->synchronize();
            {
                shape_t att_shape = { B_, NH_, T_, T_ };
                std::string att_dump = dump_tensor<NativeType>(
                    att_, att_shape, this->getName() + ".dbg.att", 4, stream );

                Utils::Logger::info( this->getName() + ": dbg.att (device dump):\n" + att_dump );
            }*/

            // Compute output values: v_out = Att @ V^T
            execute_plan<NativeType>(
                cublaslt_handle_,
                att_value_plan_,
                &alpha,
                att_, v_,
                &beta,
                v_out_,
                nullptr,
                stream );

            /*context_->synchronize();
            {
                shape_t v_out_shape = { B_, NH_, T_, HS_ };
                std::string v_out_dump = dump_tensor<NativeType>(
                    v_out_, v_out_shape, this->getName() + ".dbg.vaccum", 4, stream );

                Utils::Logger::info( this->getName() + ": dbg.vaccum (device dump):\n" + v_out_dump );
            }*/

            Detail::cuda_mha_kernels<NativeType>::unpermute_output(
                v_out_, Y,
                B_, T_, NH_, HS_,
                stream );
        }

        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            assert( this->isBuilt() && "CudaAttentionOp must be built before calling backward()" );

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "CudaAttentionOp::backward called in inference mode" );
            }

            const NativeType* dY = static_cast<const NativeType*>(output_grad.rawData());
            NativeType* dX = static_cast<NativeType*>(input_grad.rawData());

            cudaStream_t stream = context_->getStream();

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const float scale = 1.0f / sqrtf( static_cast<float>(HS_) );

            Detail::cuda_mha_kernels<NativeType>::unpermute_backward(
                dVout_, dY,
                B_, T_, NH_, HS_,
                stream );

            // Compute dV = Att^T @ dVout^T
            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_v_plan_,
                &alpha,
                att_, dVout_,
                &beta,
                dV_,
                nullptr,
                stream );

            // Compute dAtt = dVout^T @ V
            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_att_plan_,
                &alpha,
                dVout_, v_,
                &beta,
                datt_,
                nullptr,
                stream );

            Detail::cuda_mha_kernels<NativeType>::softmax_backward(
                dpreatt_, datt_, att_,
                1.0f, // TJT: Scale must be the same as used in forward
                B_, NH_, T_,
                stream );

            // Compute dQ = dPreatt @ K^T
            // Note: scale is applied here to match forward scaling
            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_q_plan_,
                &scale,
                dpreatt_, k_,
                &beta,
                dq_,
                nullptr,
                stream );

            // Compute dK = dPreatt^T @ Q^T
            // Note: scale is applied here to match forward scaling
            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_k_plan_,
                &scale,
                dpreatt_, q_,
                &beta,
                dk_,
                nullptr,
                stream );

            // Permute gradients back to concatenated QKV format
            Detail::cuda_mha_kernels<NativeType>::permute_backward(
                dX,
                dq_, dk_, dV_,
                B_, T_, NH_, HS_,
                stream );
        }

        OperationType getOperationType() const override
        {
            return OperationType::AttentionOp;
        }

        std::string getName() const override
        {
            return "Cuda::AttentionOp";
        }

        const AttentionConfig& getConfig() const
        {
            return config_;
        }

    private:
        AttentionConfig config_;
        CudaExecutionContext* context_;

        int B_{ 0 };
        int T_{ 0 };
        int qkv_dim_{ 0 };
        int embedding_dim_{ 0 };
        int NH_{ 0 };
        int HS_{ 0 };

        int active_max_seq_len_{ 0 };
        int cached_seq_len_{ 0 };
        bool kv_cache_enabled_{ false };

        cublasLtHandle_t cublaslt_handle_{ nullptr };
        ComputePrecision::Policy precision_policy_;

        Detail::CublasLtMatMulPlan<NativeType> qk_score_plan_;
        Detail::CublasLtMatMulPlan<NativeType> att_value_plan_;
        Detail::CublasLtMatMulPlan<NativeType> qk_decode_plan_;
        Detail::CublasLtMatMulPlan<NativeType> att_value_decode_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_v_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_att_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_q_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_k_plan_;

        std::shared_ptr<TensorType> q_tensor_;
        std::shared_ptr<TensorType> k_tensor_;
        std::shared_ptr<TensorType> v_tensor_;
        std::shared_ptr<TensorType> preatt_tensor_;
        std::shared_ptr<TensorType> att_tensor_;
        std::shared_ptr<TensorType> v_out_tensor_;

        std::shared_ptr<TensorType> preatt_decode_tensor_;
        std::shared_ptr<TensorType> att_decode_tensor_;
        std::shared_ptr<TensorType> v_out_decode_tensor_;

        std::shared_ptr<TensorType> dq_tensor_;
        std::shared_ptr<TensorType> dk_tensor_;
        std::shared_ptr<TensorType> dV_tensor_;
        std::shared_ptr<TensorType> dpreatt_tensor_;
        std::shared_ptr<TensorType> datt_tensor_;
        std::shared_ptr<TensorType> dVout_tensor_;

        NativeType* q_{ nullptr };
        NativeType* k_{ nullptr };
        NativeType* v_{ nullptr };
        NativeType* preatt_{ nullptr };
        NativeType* att_{ nullptr };
        NativeType* v_out_{ nullptr };

        NativeType* preatt_decode_{ nullptr };
        NativeType* att_decode_{ nullptr };
        NativeType* v_out_decode_{ nullptr };

        NativeType* dq_{ nullptr };
        NativeType* dk_{ nullptr };
        NativeType* dV_{ nullptr };
        NativeType* dpreatt_{ nullptr };
        NativeType* datt_{ nullptr };
        NativeType* dVout_{ nullptr };

        void validatePrefillInputShape( const shape_t& input_shape ) const
        {
            validateInputShape( input_shape );

            if ( input_shape[ 1 ] <= 0 || input_shape[ 1 ] > T_ )
            {
                throw std::invalid_argument( "CudaAttentionOp: prefill sequence length out of range" );
            }
        }

        void validateDecodeInputShape( const shape_t& input_shape ) const
        {
            validateInputShape( input_shape );

            if ( input_shape[ 1 ] != 1 )
            {
                throw std::invalid_argument( "CudaAttentionOp: decode input must have sequence length 1" );
            }
        }

        void ensureKVCacheEnabled() const
        {
            if ( !kv_cache_enabled_ )
            {
                throw std::runtime_error( "CudaAttentionOp: KV cache must be initialized before prefill or decode" );
            }
        }

        void validateInputShape( const shape_t& input_shape ) const
        {
            // REVIEW: Should input shape be just [ B, T ]?
            // We already know embedding_dim from config.
            if ( input_shape.size() != 3 )
            {
                throw std::invalid_argument(
                    "CudaAttentionOp: input must have rank 3 (batch_size, seq_length, 3 * model_dim)" );
            }

            const int64_t expected_qkv_dim = 3 * config_.getModelDim();

            if ( input_shape[ 2 ] != expected_qkv_dim )
            {
                throw std::invalid_argument(
                    "CudaAttentionOp: input last dimension must be 3*embedding_dim (Q, K, V concatenated)" );
            }
        }

        void allocateStateTensors()
        {
            auto device = context_->getDeviceId();

            shape_t qkv_shape = { B_, NH_, T_, HS_ };

            q_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            q_tensor_->setName( "q_" );
            q_ = static_cast<NativeType*>(q_tensor_->rawData());

            k_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            k_tensor_->setName( "k_" );
            k_ = static_cast<NativeType*>(k_tensor_->rawData());

            v_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            v_tensor_->setName( "v_" );
            v_ = static_cast<NativeType*>(v_tensor_->rawData());

            shape_t att_shape = { B_, NH_, T_, T_ };

            preatt_tensor_ = std::make_shared<TensorType>( device, att_shape );
            preatt_tensor_->setName( "preatt_" );
            preatt_ = static_cast<NativeType*>(preatt_tensor_->rawData());

            att_tensor_ = std::make_shared<TensorType>( device, att_shape );
            att_tensor_->setName( "att_" );
            att_ = static_cast<NativeType*>(att_tensor_->rawData());

            shape_t v_out_shape = { B_, NH_, T_, HS_ };

            v_out_tensor_ = std::make_shared<TensorType>( device, v_out_shape );
            v_out_tensor_->setName( "v_out_" );
            v_out_ = static_cast<NativeType*>(v_out_tensor_->rawData());

            shape_t decode_att_shape = { B_, NH_, 1, T_ };
            shape_t decode_v_out_shape = { B_, NH_, 1, HS_ };

            preatt_decode_tensor_ = std::make_shared<TensorType>( device, decode_att_shape );
            preatt_decode_tensor_->setName( "preatt_decode_" );
            preatt_decode_ = static_cast<NativeType*>(preatt_decode_tensor_->rawData());

            att_decode_tensor_ = std::make_shared<TensorType>( device, decode_att_shape );
            att_decode_tensor_->setName( "att_decode_" );
            att_decode_ = static_cast<NativeType*>(att_decode_tensor_->rawData());

            v_out_decode_tensor_ = std::make_shared<TensorType>( device, decode_v_out_shape );
            v_out_decode_tensor_->setName( "v_out_decode_" );
            v_out_decode_ = static_cast<NativeType*>(v_out_decode_tensor_->rawData());

            dq_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            dq_tensor_->setName( "dq_" );
            dq_ = static_cast<NativeType*>(dq_tensor_->rawData());

            dk_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            dk_tensor_->setName( "dk_" );
            dk_ = static_cast<NativeType*>(dk_tensor_->rawData());

            dV_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            dV_tensor_->setName( "dV_" );
            dV_ = static_cast<NativeType*>(dV_tensor_->rawData());

            dpreatt_tensor_ = std::make_shared<TensorType>( device, att_shape );
            dpreatt_tensor_->setName( "dpreatt_" );
            dpreatt_ = static_cast<NativeType*>(dpreatt_tensor_->rawData());

            datt_tensor_ = std::make_shared<TensorType>( device, att_shape );
            datt_tensor_->setName( "datt_" );
            datt_ = static_cast<NativeType*>(datt_tensor_->rawData());

            dVout_tensor_ = std::make_shared<TensorType>( device, v_out_shape );
            dVout_tensor_->setName( "dVout_" );
            dVout_ = static_cast<NativeType*>(dVout_tensor_->rawData());
        }

        void buildCublasLtPlans()
        {
            cudaDataType_t cuda_data_type = getCudaDataType();
            cublasComputeType_t compute_type;
            cudaDataType_t scale_type;

            getComputeTypes( compute_type, scale_type );

            qk_score_plan_ = Detail::build_qk_score_plan<NativeType>(
                cublaslt_handle_,
                B_, NH_, T_, HS_,
                cuda_data_type,
                compute_type,
                scale_type );

            att_value_plan_ = Detail::build_att_value_plan<NativeType>(
                cublaslt_handle_,
                B_, NH_, T_, HS_,
                cuda_data_type,
                compute_type,
                scale_type );

            qk_decode_plan_ = Detail::build_qk_decode_plan<NativeType>(
                cublaslt_handle_,
                B_, NH_, T_, HS_,
                cuda_data_type,
                compute_type,
                scale_type );

            att_value_decode_plan_ = Detail::build_att_value_decode_plan<NativeType>(
                cublaslt_handle_,
                B_, NH_, T_, HS_,
                cuda_data_type,
                compute_type,
                scale_type );

            backward_v_plan_ = Detail::build_backward_v_plan<NativeType>(
                cublaslt_handle_,
                B_, NH_, T_, HS_,
                cuda_data_type,
                compute_type,
                scale_type );

            backward_att_plan_ = Detail::build_backward_att_plan<NativeType>(
                cublaslt_handle_,
                B_, NH_, T_, HS_,
                cuda_data_type,
                compute_type,
                scale_type );

            backward_q_plan_ = Detail::build_backward_q_plan<NativeType>(
                cublaslt_handle_,
                B_, NH_, T_, HS_,
                cuda_data_type,
                compute_type,
                scale_type );

            backward_k_plan_ = Detail::build_backward_k_plan<NativeType>(
                cublaslt_handle_,
                B_, NH_, T_, HS_,
                cuda_data_type,
                compute_type,
                scale_type );
        }

        cudaDataType_t getCudaDataType() const
        {
            if constexpr ( std::is_same_v<NativeType, float> )
            {
                return CUDA_R_32F;
            }
            else if constexpr ( std::is_same_v<NativeType, half> )
            {
                return CUDA_R_16F;
            }
        }

        void getComputeTypes( cublasComputeType_t& compute_type, cudaDataType_t& scale_type ) const
        {
            scale_type = CUDA_R_32F;

            switch ( precision_policy_ )
            {
                case ComputePrecision::Policy::Native:
                case ComputePrecision::Policy::Accuracy:
                    if constexpr ( std::is_same_v<NativeType, half> )
                    {
                        compute_type = CUBLAS_COMPUTE_16F;
                    }
                    else
                    {
                        compute_type = CUBLAS_COMPUTE_32F;
                    }
                    break;

                case ComputePrecision::Policy::Performance:
                case ComputePrecision::Policy::Auto:
                default:
                    if constexpr ( std::is_same_v<NativeType, half> )
                    {
                        compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
                    }
                    else
                    {
                        compute_type = CUBLAS_COMPUTE_32F;
                    }
                    break;
            }
        }
    };

    export class CudaAttentionOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "AttentionOp";

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP32, TensorDataType::FP32>(
                opName,
                []( IExecutionContext* context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP32>>
                {
                    const auto& attentionConfig = static_cast<const AttentionConfig&>(config);
                    return std::make_shared<CudaAttentionOp<TensorDataType::FP32>>( context, attentionConfig );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP16, TensorDataType::FP16>(
                opName,
                []( IExecutionContext* context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP16>>
                {
                    const auto& attentionConfig = static_cast<const AttentionConfig&>(config);
                    return std::make_shared<CudaAttentionOp<TensorDataType::FP16>>( context, attentionConfig );
                }
            );
        }
    };
}