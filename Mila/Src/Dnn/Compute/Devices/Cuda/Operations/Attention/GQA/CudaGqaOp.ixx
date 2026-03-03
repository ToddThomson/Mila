/**
 * @file CudaGqaOp.ixx
 * @brief CUDA implementation of Grouped-Query Attention (GQA) using cuBLASLt.
 *
 * GQA generalises MHA by allowing num_kv_heads < num_heads.  Every group of
 * (num_heads / num_kv_heads) Q heads shares a single K/V head.  This reduces
 * KV cache memory and bandwidth proportionally to the group size while
 * retaining most of the representational power of full MHA.
 *
 * Design overview
 * ───────────────
 * The implementation follows the same cuBLASLt-based approach as
 * CudaMultiHeadAttentionOp, with one systematic change: K and V are stored
 * compactly in [B, NKV, T, HS] layout and are *expanded* to [B, NH, T, HS]
 * before the matmuls by gqa_expand_kv_kernel (in CudaGqa.cuh).  This keeps
 * every cuBLASLt plan at batch_count = B * NH — the same as MHA — so the
 * plan builders in :Plans are structurally identical to their MHA counterparts.
 *
 * The expansion trades a small amount of extra VRAM for simplicity and maximum
 * cuBLASLt utilisation.  A future optimisation could use a custom batched
 * kernel to perform the group broadcast implicitly.
 *
 * Forward pass (prefill / training)
 * ──────────────────────────────────
 *  1. gqa_permute_qkv         : unpack [B,T,(NH+2*NKV)*HS] → Q[B,NH,T,HS]
 *                                                              K[B,NKV,T,HS]
 *                                                              V[B,NKV,T,HS]
 *  2. gqa_expand_kv           : K/V [B,NKV,T,HS] → k_exp/v_exp [B,NH,T,HS]
 *  3. cuBLASLt qk_score_plan  : preatt[B,NH,T,T]  = Q @ k_exp^T
 *  4. softmax_forward         : att  [B,NH,T,T]   = softmax(preatt / √HS)
 *  5. cuBLASLt att_value_plan : v_out[B,NH,T,HS]  = att @ v_exp
 *  6. gqa_unpermute_output    : v_out [B,NH,T,HS] → Y [B,T,NH*HS]
 *
 * Forward pass (decode / KV-cache)
 * ─────────────────────────────────
 *  1. gqa_permute_qkv_decode  : write single Q token; append K/V to cache
 *  2. gqa_expand_kv           : expand K/V cache slice up to current position
 *  3. cuBLASLt qk_decode_plan : preatt_decode [B,NH,1,T]
 *  4. softmax_decode_forward  : att_decode
 *  5. cuBLASLt att_value_decode_plan : v_out_decode [B,NH,1,HS]
 *  6. gqa_unpermute_output    : single-token Y [B,1,C]
 *
 * Backward pass (training only)
 * ──────────────────────────────
 *  1. gqa_unpermute_backward  : dY → dVout [B,NH,T,HS]
 *  2. dV_exp = Att^T @ dVout          (backward_v_plan_,  on expanded layout)
 *  3. dAtt   = dVout @ v_exp^T        (backward_att_plan_)
 *  4. softmax_backward        : dPreatt
 *  5. dQ    = dPreatt @ k_exp         (backward_q_plan_)
 *  6. dK_exp= dPreatt^T @ Q           (backward_k_plan_)
 *  7. gqa_reduce_kv_grad      : dK_exp/dV_exp [B,NH,T,HS]
 *                               → dK/dV      [B,NKV,T,HS]  (sum over group)
 *  8. gqa_permute_backward    : pack dQ[B,NH], dK[B,NKV], dV[B,NKV]
 *                               → dX [B,T,(NH+2*NKV)*HS]
 *
 * KV cache lifecycle
 * ──────────────────
 * Mirrors CudaMultiHeadAttentionOp exactly:
 *   initializeKVCache()  called by owning transformer's generate()
 *   forwardPrefill()     populates cache, sets cached_seq_len_
 *   forwardDecode()      appends one token per call
 *   resetKVCache()       clears cached_seq_len_ for a new session
 */

module;
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <vector>
#include <memory>
#include <string>
#include <format>
#include <stdexcept>
#include <cstdint>
#include <type_traits>
#include <sstream>
#include <cassert>
#include "Kernels/CudaGqa.cuh"

export module Compute.CudaGroupedQueryAttentionOp;
import :Dispatch;
import :Plans;

import Dnn.Components.GroupedQueryAttention;
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
import Compute.CublasLtPlan;
import CublasLt.Error;
import Utils.Logger;

namespace Mila::Dnn::Compute::Cuda::GroupedQueryAttention
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute::Cuda;

    /**
     * @brief CUDA Grouped-Query Attention operation.
     *
     * Template parameter TPrecision drives both the tensor element type and
     * the cuBLASLt data/compute type selection, identical to CudaMhaOp.
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaGroupedQueryAttentionOp
        : public UnaryOperation<DeviceType::Cuda, TPrecision>
        , public IKVCacheable
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ConfigType = GroupedQueryAttentionConfig;

        CudaGroupedQueryAttentionOp(
            IExecutionContext* context,
            const GroupedQueryAttentionConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaGroupedQueryAttentionOp" ) ) , config_( config )
        {
            config_.validate();
        }

        // No learnable parameters — GQA weights live in the projection layers.
        void setParameters( ITensor*, ITensor* ) override
        {}
        void setGradients( ITensor*, ITensor* ) override
        {}

        // ====================================================================
        // IKVCacheable
        // ====================================================================

        void initializeKVCache( int batch_size, int max_seq_length ) override
        {
            if ( !this->isBuilt() )
                throw std::runtime_error(
                    "CudaGroupedQueryAttentionOp::initializeKVCache requires build() first" );

            if ( batch_size != B_ )
                throw std::invalid_argument(
                    "CudaGroupedQueryAttentionOp::initializeKVCache batch size mismatch" );

            if ( max_seq_length <= 0 || max_seq_length > T_ )
                throw std::invalid_argument(
                    "CudaGroupedQueryAttentionOp::initializeKVCache max_seq_length out of range" );

            active_max_seq_len_ = max_seq_length;
            cached_seq_len_ = 0;
            kv_cache_enabled_ = true;
        }

        void resetKVCache() override
        {
            cached_seq_len_ = 0;
        }

        void forwardPrefill( const ITensor& input, ITensor& output ) override
        {
            ensureKVCacheEnabled();
            validatePrefillInputShape( input.shape() );

            const int actual_seq_len = static_cast<int>(input.shape()[ 1 ]);

            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());
            cudaStream_t stream = context_->getStream();

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const float scale = 1.0f / sqrtf( static_cast<float>(HS_) );

            // 1. Unpack packed QKV → Q[B,NH,T,HS], K[B,NKV,T,HS], V[B,NKV,T,HS]
            Detail::cuda_gqa_kernels<NativeType>::permute_qkv_padded(
                q_, k_, v_,
                X,
                B_, actual_seq_len, T_, NH_, NKV_, HS_,
                stream );

            // 2. Expand K/V from [B,NKV,T,HS] → k_exp/v_exp [B,NH,T,HS]
            Detail::cuda_gqa_kernels<NativeType>::expand_kv(
                k_exp_, v_exp_,
                k_, v_,
                B_, actual_seq_len, NH_, NKV_, HS_,
                stream );

            // 3. preatt = Q @ k_exp^T  [B,NH,T,T]
            execute_plan<NativeType>(
                cublaslt_handle_, qk_score_plan_,
                &scale, q_, k_exp_, &beta, preatt_,
                nullptr,
                stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            // 4. att = softmax(preatt / √HS)  with causal mask
            Detail::cuda_gqa_kernels<NativeType>::softmax_padded_forward(
                att_, 1.0f, preatt_,
                B_, NH_, T_, actual_seq_len,
                stream );

            // 5. v_out = att @ v_exp  [B,NH,T,HS]
            execute_plan<NativeType>(
                cublaslt_handle_, att_value_plan_,
                &alpha, att_, v_exp_, &beta, v_out_,
                nullptr,
                stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            context_->synchronize();

            // 6. Unpack v_out [B,NH,T,HS] → Y [B,T,C]
            Detail::cuda_gqa_kernels<NativeType>::unpermute_output_padded(
                v_out_, Y,
                B_, actual_seq_len, T_, NH_, HS_,
                stream );

            context_->synchronize();
            cached_seq_len_ = actual_seq_len;
        }

        void forwardDecode( const ITensor& input, ITensor& output, int position ) override
        {
            ensureKVCacheEnabled();
            validateDecodeInputShape( input.shape() );

            if ( position < 0 || position >= active_max_seq_len_ )
                throw std::invalid_argument(
                    "CudaGroupedQueryAttentionOp::forwardDecode position out of range" );

            const int actual_len = position + 1;

            const NativeType* X = static_cast<const NativeType*>( input.rawData() );
            NativeType* Y = static_cast<NativeType*>( output.rawData() );
            cudaStream_t stream = context_->getStream();

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const float scale = 1.0f / sqrtf( static_cast<float>(HS_) );

            // 1. Write single Q token; append K/V token to compact cache.
            Detail::cuda_gqa_kernels<NativeType>::permute_qkv_decode(
                q_, k_, v_,
                X,
                B_, position, T_, NH_, NKV_, HS_,
                stream );

            // 2. Expand KV cache up to actual_len into k_exp / v_exp.
            Detail::cuda_gqa_kernels<NativeType>::expand_kv(
                k_exp_, v_exp_,
                k_, v_,
                B_, actual_len, NH_, NKV_, HS_,
                stream );

            // 3. Decode QK: q_decode points at the current token's row.
            const NativeType* q_decode = q_ + static_cast<int64_t>(position) * HS_;

            execute_plan<NativeType>(
                cublaslt_handle_, qk_decode_plan_,
                &scale, q_decode, k_exp_, &beta, preatt_decode_,
                nullptr,
                stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            // 4. Softmax over actual_len positions.
            Detail::cuda_gqa_kernels<NativeType>::softmax_decode_forward(
                att_decode_, 1.0f, preatt_decode_,
                B_, NH_, T_, actual_len,
                stream );

            // 5. Weighted sum over V.
            execute_plan<NativeType>(
                cublaslt_handle_, att_value_decode_plan_,
                &alpha, att_decode_, v_exp_, &beta, v_out_decode_,
                nullptr,
                stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            // 6. Unpack single token output.
            Detail::cuda_gqa_kernels<NativeType>::unpermute_output(
                v_out_decode_, Y,
                B_, 1, NH_, HS_,
                stream );

            if ( actual_len > cached_seq_len_ )
                cached_seq_len_ = actual_len;
        }

        // ====================================================================
        // build / forward / backward
        // ====================================================================

        void build( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            B_ = static_cast<int>(input_shape[ 0 ]);
            T_ = static_cast<int>(input_shape[ 1 ]);
            NH_ = static_cast<int>(config_.getNumHeads());
            NKV_ = static_cast<int>(config_.getNumKvHeads());
            HS_ = static_cast<int>(config_.getHeadDim());
            C_ = static_cast<int>(config_.getModelDim());  // = NH_ * HS_
            GS_ = NH_ / NKV_;                               // group size

            active_max_seq_len_ = T_;
            cached_seq_len_ = 0;
            kv_cache_enabled_ = false;

            allocateStateTensors();

            cublaslt_handle_ = context_->getCublasLtHandle();

            if ( !cublaslt_handle_ )
                throw std::runtime_error(
                    "CudaGroupedQueryAttentionOp requires cuBLASLt. "
                    "Ensure CUDA 10.1 or newer." );

            precision_policy_ = config_.getPrecisionPolicy();
            buildCublasLtPlans();

            UnaryOperationBase::build( input_shape );
        }

        /**
         * @brief Standard (non-cached) forward pass used during training.
         */
        void forward( const ITensor& input, ITensor& output ) const override
        {
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());
            cudaStream_t stream = context_->getStream();

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const float scale = 1.0f / sqrtf( static_cast<float>(HS_) );

            // 1. Unpack QKV
            Detail::cuda_gqa_kernels<NativeType>::permute_qkv(
                q_, k_, v_,
                X,
                B_, T_, NH_, NKV_, HS_,
                stream );

            // 2. Expand KV
            Detail::cuda_gqa_kernels<NativeType>::expand_kv(
                k_exp_, v_exp_,
                k_, v_,
                B_, T_, NH_, NKV_, HS_,
                stream );

            // 3. Q @ K^T
            execute_plan<NativeType>(
                cublaslt_handle_, qk_score_plan_,
                &scale, q_, k_exp_, &beta, preatt_,
                nullptr,
                stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            // 4. Causal softmax
            Detail::cuda_gqa_kernels<NativeType>::softmax_forward(
                att_, 1.0f, preatt_,
                B_, NH_, T_,
                stream );

            // 5. att @ V
            execute_plan<NativeType>(
                cublaslt_handle_, att_value_plan_,
                &alpha, att_, v_exp_, &beta, v_out_,
                nullptr,
                stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            // 6. Unpack output
            Detail::cuda_gqa_kernels<NativeType>::unpermute_output(
                v_out_, Y,
                B_, T_, NH_, HS_,
                stream );
        }

        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            assert( this->isBuilt() );

            if ( !this->isTraining() )
            {
                throw std::runtime_error(
                    "CudaGroupedQueryAttentionOp::backward called in inference mode" );
            }

            const NativeType* dY = static_cast<const NativeType*>(output_grad.rawData());
            NativeType* dX = static_cast<NativeType*>(input_grad.rawData());
            cudaStream_t stream = context_->getStream();

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const float scale = 1.0f / sqrtf( static_cast<float>(HS_) );

            // 1. dY [B,T,C] → dVout [B,NH,T,HS]
            Detail::cuda_gqa_kernels<NativeType>::unpermute_backward(
                dVout_, dY,
                B_, T_, NH_, HS_,
                stream );

            // 2. dV_exp = Att^T @ dVout  (expanded, [B,NH,T,HS])
            execute_plan<NativeType>(
                cublaslt_handle_, backward_v_plan_,
                &alpha, att_, dVout_, &beta, dV_exp_,
                nullptr,
                stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            // 3. dAtt = dVout @ v_exp^T
            execute_plan<NativeType>(
                cublaslt_handle_, backward_att_plan_,
                &alpha, dVout_, v_exp_, &beta, datt_,
                nullptr,
                stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            // 4. Softmax backward → dPreatt
            Detail::cuda_gqa_kernels<NativeType>::softmax_backward(
                dpreatt_, datt_, att_,
                1.0f,
                B_, NH_, T_,
                stream );

            // 5. dQ = dPreatt @ k_exp
            execute_plan<NativeType>(
                cublaslt_handle_, backward_q_plan_,
                &scale, dpreatt_, k_exp_, &beta, dq_,
                nullptr,
                stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            // 6. dK_exp = dPreatt^T @ Q  (expanded, [B,NH,T,HS])
            execute_plan<NativeType>(
                cublaslt_handle_, backward_k_plan_,
                &scale, dpreatt_, q_, &beta, dK_exp_,
                nullptr,
                stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            // 7. Reduce expanded gradients → compact KV grad layout [B,NKV,T,HS]
            //    dV_exp  [B,NH,T,HS] → dV_  [B,NKV,T,HS]  (sum over group)
            //    dK_exp_ [B,NH,T,HS] → dK_  [B,NKV,T,HS]  (sum over group)
            Detail::cuda_gqa_kernels<NativeType>::reduce_kv_grad(
                dK_, dV_,
                dK_exp_, dV_exp_,
                B_, T_, NH_, NKV_, HS_,
                stream );

            // 8. Pack dQ[B,NH], dK[B,NKV], dV[B,NKV] → dX [B,T,(NH+2*NKV)*HS]
            Detail::cuda_gqa_kernels<NativeType>::permute_backward(
                dX,
                dq_, dK_, dV_,
                B_, T_, NH_, NKV_, HS_,
                stream );
        }

        // ====================================================================
        // Operation interface
        // ====================================================================

        OperationType getOperationType() const override
        {
            return OperationType::GroupedQueryAttentionOp;
        }

        std::string getName() const override
        {
            return "Cuda::GroupedQueryAttentionOp";
        }

        const GroupedQueryAttentionConfig& getConfig() const
        {
            return config_;
        }

    private:
        GroupedQueryAttentionConfig config_;
        CudaExecutionContext* context_;

        // Cached dimensions (set in build())
        int B_{ 0 };   ///< Batch size
        int T_{ 0 };   ///< Max sequence length
        int C_{ 0 };   ///< Model dim  = NH * HS
        int NH_{ 0 };  ///< Number of Q heads
        int NKV_{ 0 }; ///< Number of KV heads
        int HS_{ 0 };  ///< Head dim   = C / NH
        int GS_{ 0 };  ///< Group size = NH / NKV

        int  active_max_seq_len_{ 0 };
        int  cached_seq_len_{ 0 };
        bool kv_cache_enabled_{ false };

        cublasLtHandle_t        cublaslt_handle_{ nullptr };
        ComputePrecision::Policy precision_policy_;

        // cuBLASLt plans (all operate on expanded [B,NH,...] layout)
        Detail::CublasLtMatMulPlan<NativeType> qk_score_plan_;
        Detail::CublasLtMatMulPlan<NativeType> att_value_plan_;
        Detail::CublasLtMatMulPlan<NativeType> qk_decode_plan_;
        Detail::CublasLtMatMulPlan<NativeType> att_value_decode_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_v_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_att_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_q_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_k_plan_;

        // ── Forward state tensors ──────────────────────────────────────────

        // Compact KV (NKV heads)
        std::shared_ptr<TensorType> q_tensor_;     // [B, NH,  T, HS]
        std::shared_ptr<TensorType> k_tensor_;     // [B, NKV, T, HS]
        std::shared_ptr<TensorType> v_tensor_;     // [B, NKV, T, HS]

        // Expanded KV (NH heads) — input to cuBLASLt
        std::shared_ptr<TensorType> k_exp_tensor_; // [B, NH, T, HS]
        std::shared_ptr<TensorType> v_exp_tensor_; // [B, NH, T, HS]

        std::shared_ptr<TensorType> preatt_tensor_; // [B, NH, T, T]
        std::shared_ptr<TensorType> att_tensor_;    // [B, NH, T, T]
        std::shared_ptr<TensorType> v_out_tensor_;  // [B, NH, T, HS]

        // Decode-path tensors
        std::shared_ptr<TensorType> preatt_decode_tensor_; // [B, NH, 1, T]
        std::shared_ptr<TensorType> att_decode_tensor_;    // [B, NH, 1, T]
        std::shared_ptr<TensorType> v_out_decode_tensor_;  // [B, NH, 1, HS]

        // ── Backward state tensors ─────────────────────────────────────────

        std::shared_ptr<TensorType> dq_tensor_;     // [B, NH,  T, HS]
        std::shared_ptr<TensorType> dK_tensor_;     // [B, NKV, T, HS]  (compact)
        std::shared_ptr<TensorType> dV_tensor_;     // [B, NKV, T, HS]  (compact)
        std::shared_ptr<TensorType> dK_exp_tensor_; // [B, NH,  T, HS]  (expanded)
        std::shared_ptr<TensorType> dV_exp_tensor_; // [B, NH,  T, HS]  (expanded)
        std::shared_ptr<TensorType> dpreatt_tensor_;// [B, NH,  T, T]
        std::shared_ptr<TensorType> datt_tensor_;   // [B, NH,  T, T]
        std::shared_ptr<TensorType> dVout_tensor_;  // [B, NH,  T, HS]

        // Raw device pointers (non-owning aliases into the tensors above)
        NativeType* q_{ nullptr };
        NativeType* k_{ nullptr };
        NativeType* v_{ nullptr };
        NativeType* k_exp_{ nullptr };
        NativeType* v_exp_{ nullptr };
        NativeType* preatt_{ nullptr };
        NativeType* att_{ nullptr };
        NativeType* v_out_{ nullptr };
        NativeType* preatt_decode_{ nullptr };
        NativeType* att_decode_{ nullptr };
        NativeType* v_out_decode_{ nullptr };
        NativeType* dq_{ nullptr };
        NativeType* dK_{ nullptr };
        NativeType* dV_{ nullptr };
        NativeType* dK_exp_{ nullptr };
        NativeType* dV_exp_{ nullptr };
        NativeType* dpreatt_{ nullptr };
        NativeType* datt_{ nullptr };
        NativeType* dVout_{ nullptr };

        // ====================================================================
        // Private helpers
        // ====================================================================

        void validateInputShape( const shape_t& s ) const
        {
            if ( s.size() != 3 )
                throw std::invalid_argument(
                    "CudaGroupedQueryAttentionOp: input must be rank 3 [B, T, QKV_dim]" );

            const int64_t expected =
                (config_.getNumHeads() + 2 * config_.getNumKvHeads()) * config_.getHeadDim();

            if ( s[ 2 ] != expected )
            {
                std::ostringstream oss;
                oss << "CudaGroupedQueryAttentionOp: expected QKV trailing dim "
                    << expected << ", got " << s[ 2 ];
                throw std::invalid_argument( oss.str() );
            }
        }

        void validatePrefillInputShape( const shape_t& s ) const
        {
            validateInputShape( s );
            if ( s[ 1 ] <= 0 || s[ 1 ] > T_ )
                throw std::invalid_argument(
                    "CudaGroupedQueryAttentionOp: prefill sequence length out of range" );
        }

        void validateDecodeInputShape( const shape_t& s ) const
        {
            validateInputShape( s );
            if ( s[ 1 ] != 1 )
                throw std::invalid_argument(
                    "CudaGroupedQueryAttentionOp: decode input must have sequence length 1" );
        }

        void ensureKVCacheEnabled() const
        {
            if ( !kv_cache_enabled_ )
                throw std::runtime_error(
                    "CudaGroupedQueryAttentionOp: KV cache must be initialized before "
                    "prefill or decode" );
        }

        void allocateStateTensors()
        {
            auto device = context_->getDeviceId();

            const shape_t q_shape = { B_, NH_,  T_, HS_ };
            const shape_t kv_shape = { B_, NKV_, T_, HS_ };
            const shape_t kv_exp = { B_, NH_,  T_, HS_ };
            const shape_t att_shape = { B_, NH_,  T_, T_ };
            const shape_t vout_shape = { B_, NH_,  T_, HS_ };

            const shape_t dec_att = { B_, NH_, 1, T_ };
            const shape_t dec_vout = { B_, NH_, 1, HS_ };

            auto make = [&]( const shape_t& sh, const std::string& nm )
                {
                    auto t = std::make_shared<TensorType>( device, sh );
                    t->setName( nm );
                    return t;
                };

            q_tensor_ = make( q_shape, "gqa.q" );  q_ = raw( q_tensor_ );
            k_tensor_ = make( kv_shape, "gqa.k" );  k_ = raw( k_tensor_ );
            v_tensor_ = make( kv_shape, "gqa.v" );  v_ = raw( v_tensor_ );
            k_exp_tensor_ = make( kv_exp, "gqa.k_exp" );  k_exp_ = raw( k_exp_tensor_ );
            v_exp_tensor_ = make( kv_exp, "gqa.v_exp" );  v_exp_ = raw( v_exp_tensor_ );
            preatt_tensor_ = make( att_shape, "gqa.preatt" );  preatt_ = raw( preatt_tensor_ );
            att_tensor_ = make( att_shape, "gqa.att" );  att_ = raw( att_tensor_ );
            v_out_tensor_ = make( vout_shape, "gqa.v_out" );  v_out_ = raw( v_out_tensor_ );

            preatt_decode_tensor_ = make( dec_att, "gqa.preatt_dec" ); preatt_decode_ = raw( preatt_decode_tensor_ );
            att_decode_tensor_ = make( dec_att, "gqa.att_dec" ); att_decode_ = raw( att_decode_tensor_ );
            v_out_decode_tensor_ = make( dec_vout, "gqa.v_out_dec" ); v_out_decode_ = raw( v_out_decode_tensor_ );

            dq_tensor_ = make( q_shape, "gqa.dq" );  dq_ = raw( dq_tensor_ );
            dK_tensor_ = make( kv_shape, "gqa.dK" );  dK_ = raw( dK_tensor_ );
            dV_tensor_ = make( kv_shape, "gqa.dV" );  dV_ = raw( dV_tensor_ );
            dK_exp_tensor_ = make( kv_exp, "gqa.dK_exp" );  dK_exp_ = raw( dK_exp_tensor_ );
            dV_exp_tensor_ = make( kv_exp, "gqa.dV_exp" );  dV_exp_ = raw( dV_exp_tensor_ );
            dpreatt_tensor_ = make( att_shape, "gqa.dpreatt" );  dpreatt_ = raw( dpreatt_tensor_ );
            datt_tensor_ = make( att_shape, "gqa.datt" );  datt_ = raw( datt_tensor_ );
            dVout_tensor_ = make( vout_shape, "gqa.dVout" );  dVout_ = raw( dVout_tensor_ );
        }

        static NativeType* raw( const std::shared_ptr<TensorType>& t )
        {
            return static_cast<NativeType*>(t->rawData());
        }

        void buildCublasLtPlans()
        {
            const cudaDataType_t    cuda_dt = getCudaDataType();
            cublasComputeType_t     compute_type;
            cudaDataType_t          scale_type;
            getComputeTypes( compute_type, scale_type );

            // All plans use batch_count = B * NH because KV is pre-expanded.
            qk_score_plan_ = Detail::build_qk_score_plan<NativeType>(
                cublaslt_handle_, B_, NH_, T_, HS_,
                cuda_dt, compute_type, scale_type );

            att_value_plan_ = Detail::build_att_value_plan<NativeType>(
                cublaslt_handle_, B_, NH_, T_, HS_,
                cuda_dt, compute_type, scale_type );

            qk_decode_plan_ = Detail::build_qk_decode_plan<NativeType>(
                cublaslt_handle_, B_, NH_, T_, HS_,
                cuda_dt, compute_type, scale_type );

            att_value_decode_plan_ = Detail::build_att_value_decode_plan<NativeType>(
                cublaslt_handle_, B_, NH_, T_, HS_,
                cuda_dt, compute_type, scale_type );

            backward_v_plan_ = Detail::build_backward_v_plan<NativeType>(
                cublaslt_handle_, B_, NH_, T_, HS_,
                cuda_dt, compute_type, scale_type );

            backward_att_plan_ = Detail::build_backward_att_plan<NativeType>(
                cublaslt_handle_, B_, NH_, T_, HS_,
                cuda_dt, compute_type, scale_type );

            backward_q_plan_ = Detail::build_backward_q_plan<NativeType>(
                cublaslt_handle_, B_, NH_, T_, HS_,
                cuda_dt, compute_type, scale_type );

            backward_k_plan_ = Detail::build_backward_k_plan<NativeType>(
                cublaslt_handle_, B_, NH_, T_, HS_,
                cuda_dt, compute_type, scale_type );
        }

        cudaDataType_t getCudaDataType() const
        {
            if constexpr ( std::is_same_v<NativeType, float> )
                return CUDA_R_32F;
            else if constexpr ( std::is_same_v<NativeType, half> )
                return CUDA_R_16F;
        }

        void getComputeTypes(
            cublasComputeType_t& compute_type,
            cudaDataType_t& scale_type ) const
        {
            scale_type = CUDA_R_32F;

            switch ( precision_policy_ )
            {
                case ComputePrecision::Policy::Native:
                case ComputePrecision::Policy::Accuracy:
                    if constexpr ( std::is_same_v<NativeType, half> )
                        compute_type = CUBLAS_COMPUTE_16F;
                    else
                        compute_type = CUBLAS_COMPUTE_32F;
                    break;

                case ComputePrecision::Policy::Performance:
                case ComputePrecision::Policy::Auto:
                default:
                    if constexpr ( std::is_same_v<NativeType, half> )
                        compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
                    else
                        compute_type = CUBLAS_COMPUTE_32F;
                    break;
            }
        }
    };

    // ========================================================================
    // Registration
    // ========================================================================

    export class CudaGroupedQueryAttentionOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string_view opName = Compute::OperationNames::GroupedQueryAttention;

            OperationRegistry::instance()
                .registerUnaryOperation<DeviceType::Cuda,
                TensorDataType::FP32,
                TensorDataType::FP32>(
                    opName,
                    []( IExecutionContext* ctx,
                        const ComponentConfig& cfg )
                    -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP32>>
                    {
                        return std::make_shared<CudaGroupedQueryAttentionOp<TensorDataType::FP32>>(
                            ctx,
                            static_cast<const GroupedQueryAttentionConfig&>(cfg) );
                    } );

            OperationRegistry::instance()
                .registerUnaryOperation<DeviceType::Cuda,
                TensorDataType::FP16,
                TensorDataType::FP16>(
                    opName,
                    []( IExecutionContext* ctx,
                        const ComponentConfig& cfg )
                    -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP16>>
                    {
                        return std::make_shared<CudaGroupedQueryAttentionOp<TensorDataType::FP16>>(
                            ctx,
                            static_cast<const GroupedQueryAttentionConfig&>(cfg) );
                    } );
        }
    };
}
