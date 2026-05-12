/**
 * @file CudaGqaOp.ixx
 * @brief CUDA Grouped-Query Attention (GQA) operation using cuBLASLt.
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
#include <unordered_map>
#include "Kernels/CudaGqa.cuh"

// DEBUG:
#include <iostream>

export module Compute.CudaGqaOp;
import :Dispatch;
import :Plans;

import Dnn.Components.GqaConfig;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.UnaryOperation;
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
import Compute.IKvInference;
import Compute.GqaState;
import Compute.CublasLtPlan;
import CublasLt.Error;
import Logging.Logger;

import Cuda.Debug;

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute::Cuda;

    // TODO: Temporary. To be replaced by tuned chunk size based on empirical perf testing across devices and precisions.
    constexpr int64_t kPrefillChunkSize = 64;

    // TEMP: A/B validation gate. Flip to true to activate the optimized NKV-layout path.
    // Remove this constant and all legacy/optimized branching once the optimized path is validated.
    static constexpr bool kUseOptimizedPath = true;

    /**
     * @brief CUDA Grouped-Query Attention operation.
     *
     * GQA generalises MHA by allowing num_kv_heads < num_heads. Every group of
     * (num_heads / num_kv_heads) Q heads shares a single K/V head, reducing
     * KV cache memory and bandwidth proportionally to the group size.
     *
     * The legacy path uses cuBLASLt batched matmuls on an expanded layout:
     * K and V are stored compactly in [B, NKV, T, HS] and expanded to
     * [B, NH, T, HS] before the matmuls so every cuBLASLt plan operates at
     * batch_count = B * NH.
     *
     * The optimized path (kUseOptimizedPath) eliminates the expansion buffers
     * and q_tensor_ by rebuilding cuBLASLt plans against the compact NKV layout
     * with grouped head strides. See GqaMemory.md Phase 1 and Phase 2.
     *
     * Forward pass (training):
     *  1. permute_qkv       -> Q[B,NH,T,HS], K[B,NKV,T,HS], V[B,NKV,T,HS]
     *  2. expand_kv         -> k_exp/v_exp [B,NH,T,HS]
     *  3. qk_score_plan     -> preatt [B,NH,T,T]
     *  4. softmax_forward   -> att [B,NH,T,T]
     *  5. att_value_plan    -> v_out [B,NH,T,HS]
     *  6. unpermute_output  -> Y [B,T,NH*HS]
     *
     * Prefill pass (inference only, with KV cache):
     *  1. prefill_permute_qkv      -> Q[B,NH,chunk,HS], K/V[B,NKV,chunk,HS] (padded to T)
     *  2. prefill_expand_kv        -> k_exp/v_exp [B,NH,chunk,HS] (padded to T)
     *  3. prefill_qk_plan          -> preatt [B,NH,chunk,T]
     *  4. prefill_softmax          -> att [B,NH,chunk,T]
     *  5. prefill_att_value_plan   -> v_out [B,NH,chunk,HS]
     *  6. prefill_unpermute_output -> Y [B,chunk,C]
     *
     * Decode pass (decode / KV-cache):
     *  1. permute_qkv_decode -> single Q token; append K/V to cache
     *  2. expand_kv          -> expand cache slice up to current position
     *  3. qk_decode_plan     -> preatt_decode [B,NH,1,T]
     *  4. softmax_decode     -> att_decode
     *  5. att_value_decode   -> v_out_decode [B,NH,1,HS]
     *  6. unpermute_output   -> Y [B,1,C]
     *
     * Backward pass (training only):
     *  1. unpermute_backward -> dVout [B,NH,T,HS]
     *  2. backward_v_plan    -> dV_exp (expanded)
     *  3. backward_att_plan  -> dAtt
     *  4. softmax_backward   -> dPreatt
     *  5. backward_q_plan    -> dQ
     *  6. backward_k_plan    -> dK_exp (expanded)
     *  7. reduce_kv_grad     -> dK/dV [B,NKV,T,HS] (sum over group)
     *  8. permute_backward   -> dX [B,T,(NH+2*NKV)*HS]
     *
     * @tparam TPrecision Tensor element type and cuBLASLt data/compute type.
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaGqaOp : public Operation<DeviceType::Cuda, TPrecision>, public IKvInference
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ConfigType = GqaConfig;

        CudaGqaOp( IExecutionContext* context, const GqaConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaGqaOp" ) )
            , config_( config )
            , use_optimized_path_( kUseOptimizedPath )
        {
            config_.validate();
        }

        // No learnable parameters — GQA weights live in the projection layers.
        void setParameters( ITensor*, ITensor* ) override
        {
        }

        void setGradients( ITensor*, ITensor* ) override
        {
        }

        /**
         * @brief Wire the shared transient scratch buffers for the optimized inference path.
         *
         * Called once per build by LlamaTransformer after all blocks are built. The tensors
         * are owned by LlamaTransformer and shared across all GQA layers sequentially.
         * Must be called before prefill() or decode() when use_optimized_path_ is true.
         *
         * @param state Non-owning pointers to the shared workspace tensors. All slots must
         *              be non-null for the optimized inference path.
         */
        void setState( const GqaState& state )
        {
            if ( state.q_permute )
                q_permute_opt_ = static_cast<NativeType*>(state.q_permute->rawData());

            if ( state.preatt )
                preatt_opt_ = static_cast<NativeType*>(state.preatt->rawData());

            if ( state.att )
                att_opt_ = static_cast<NativeType*>(state.att->rawData());

            if ( state.v_out )
                v_out_opt_ = static_cast<NativeType*>(state.v_out->rawData());

            if ( state.preatt_decode )
                preatt_decode_opt_ = static_cast<NativeType*>(state.preatt_decode->rawData());

            if ( state.att_decode )
                att_decode_opt_ = static_cast<NativeType*>(state.att_decode->rawData());

            if ( state.v_out_decode )
                v_out_decode_opt_ = static_cast<NativeType*>(state.v_out_decode->rawData());
        }

        void initializeKvCache( int batch_size, int max_seq_length ) override
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

        void resetKvCache() override
        {
            cached_seq_len_ = 0;
        }

        void prefill(
            const ITensor& q, const ITensor& k, const ITensor& v,
            ITensor& output,
            int position_offset ) override
        {
            ensureKVCacheEnabled();

            if ( use_optimized_path_ )
                prefill_optimized( q, k, v, output, position_offset );
            else
                prefillImpl( q, k, v, output, position_offset );
        }

        void decode(
            const ITensor& q, const ITensor& k, const ITensor& v,
            ITensor& output,
            int position ) override
        {
            ensureKVCacheEnabled();

            if ( use_optimized_path_ )
                decode_optimized( q, k, v, output, position );
            else
                decodeImpl( q, k, v, output, position );
        }

        // ====================================================================
        // build / forward / backward
        // ====================================================================

        void build( const BuildContext& context ) override
        {
            const shape_t& input_shape = context.inputShape();

            validateInputShape( input_shape );

            B_ = static_cast<int>(input_shape[ 0 ]);
            T_ = static_cast<int>(input_shape[ 1 ]);

            NH_ = static_cast<int>(config_.getNumHeads());
            NKV_ = static_cast<int>(config_.getNumKvHeads());
            HS_ = static_cast<int>(config_.getHeadDim());
            C_ = static_cast<int>(config_.getModelDim());
            GS_ = NH_ / NKV_;

            prefill_chunk_size_ = static_cast<int>(kPrefillChunkSize);

            active_max_seq_len_ = T_;
            cached_seq_len_ = 0;
            kv_cache_enabled_ = false;

            if ( use_optimized_path_ )
                initializeState_optimized( context );
            else
                initializeState( context );

            cublaslt_handle_ = context_->getCublasLtHandle();

            if ( !cublaslt_handle_ )
                throw std::runtime_error(
                    "CudaGroupedQueryAttentionOp requires cuBLASLt. "
                    "Ensure CUDA 10.1 or newer." );

            if ( use_optimized_path_ )
                buildCublasLtPlans_optimized();
            else
                buildCublasLtPlans();

            Operation<DeviceType::Cuda, TPrecision>::build( context );
        }

        /**
         * @brief Standard (non-cached) forward pass used during training.
         */
        void forward( const ITensor& input, ITensor& output ) const
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

        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) const
        {
            assert( this->isBuilt() );

            if ( !this->isEvalMode() )
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

            // 1. dY [B,T,C] -> dVout [B,NH,T,HS]
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

            // 4. Softmax backward -> dPreatt
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

            // 7. Reduce expanded gradients -> compact KV grad layout [B,NKV,T,HS]
            Detail::cuda_gqa_kernels<NativeType>::reduce_kv_grad(
                dK_, dV_,
                dK_exp_, dV_exp_,
                B_, T_, NH_, NKV_, HS_,
                stream );

            // 8. Pack dQ[B,NH], dK[B,NKV], dV[B,NKV] -> dX [B,T,(NH+2*NKV)*HS]
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
            return "Cuda::GqaOp";
        }

        const GqaConfig& getConfig() const
        {
            return config_;
        }

        std::size_t getStateMemorySize() const override
        {
            return state_memory_size_;
        }

    private:

        GqaConfig config_;
        CudaExecutionContext* context_;

        // TEMP: Mirrors kUseOptimizedPath. Removed with the gate once the optimized path is validated.
        bool use_optimized_path_{ false };

        // -------- Static model topology (set in build(), never change) --------

        int B_{ 0 };   ///< Batch size
        int T_{ 0 };   ///< Max sequence length
        int C_{ 0 };   ///< Model dim  = NH * HS
        int NH_{ 0 };  ///< Number of Q heads
        int NKV_{ 0 }; ///< Number of KV heads
        int HS_{ 0 };  ///< Head dim   = C / NH
        int GS_{ 0 };  ///< Group size = NH / NKV

        int prefill_chunk_size_{ 0 };
        int active_max_seq_len_{ 0 };
        int cached_seq_len_{ 0 };
        bool kv_cache_enabled_{ false };

        cublasLtHandle_t cublaslt_handle_{ nullptr };

        // ====================================================================
        // Legacy path — cuBLASLt plans (expanded [B,NH,...] layout)
        // ====================================================================

        Detail::CublasLtMatMulPlan<NativeType> qk_score_plan_;
        Detail::CublasLtMatMulPlan<NativeType> att_value_plan_;
        Detail::CublasLtMatMulPlan<NativeType> qk_prefill_plan_;
        Detail::CublasLtMatMulPlan<NativeType> att_value_prefill_plan_;
        Detail::CublasLtMatMulPlan<NativeType> qk_decode_plan_;
        Detail::CublasLtMatMulPlan<NativeType> att_value_decode_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_v_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_att_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_q_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_k_plan_;

        std::unordered_map<int, Detail::CublasLtMatMulPlan<NativeType>> qk_partial_prefill_plan_cache_;
        std::unordered_map<int, Detail::CublasLtMatMulPlan<NativeType>> att_value_partial_prefill_plan_cache_;

        // ====================================================================
        // Optimized path — cuBLASLt plans (compact NKV layout, grouped head strides)
        // TEMP: Remove legacy plans and this block once the optimized path is validated.
        // ====================================================================

        Detail::CublasLtMatMulPlan<NativeType> qk_prefill_plan_optimized_;
        Detail::CublasLtMatMulPlan<NativeType> att_value_prefill_plan_optimized_;
        Detail::CublasLtMatMulPlan<NativeType> qk_decode_plan_optimized_;
        Detail::CublasLtMatMulPlan<NativeType> att_value_decode_plan_optimized_;

        std::unordered_map<int, Detail::CublasLtMatMulPlan<NativeType>> qk_partial_prefill_plan_cache_optimized_;
        std::unordered_map<int, Detail::CublasLtMatMulPlan<NativeType>> att_value_partial_prefill_plan_cache_optimized_;

        // ====================================================================
        // Legacy path — forward state tensors
        // ====================================================================

        // Compact KV cache (NKV heads) — shared by both paths
        std::shared_ptr<TensorType> k_tensor_;     // [B, NKV, T, HS]
        std::shared_ptr<TensorType> v_tensor_;     // [B, NKV, T, HS]

        // Legacy-only: persistent Q cache and expanded KV buffers
        std::shared_ptr<TensorType> q_tensor_;     // [B, NH,  T, HS]
        std::shared_ptr<TensorType> k_exp_tensor_; // [B, NH,  T, HS]
        std::shared_ptr<TensorType> v_exp_tensor_; // [B, NH,  T, HS]

        std::shared_ptr<TensorType> preatt_tensor_; // [B, NH, T|chunk, T]
        std::shared_ptr<TensorType> att_tensor_;    // [B, NH, T|chunk, T]
        std::shared_ptr<TensorType> v_out_tensor_;  // [B, NH, T|chunk, HS]

        // Decode path — shared by both paths
        std::shared_ptr<TensorType> preatt_decode_tensor_; // [B, NH, 1, T]
        std::shared_ptr<TensorType> att_decode_tensor_;    // [B, NH, 1, T]
        std::shared_ptr<TensorType> v_out_decode_tensor_;  // [B, NH, 1, HS]

        // ====================================================================
        // Optimized path — transient scratch tensors (self-allocated for A/B period)
        // TEMP: Migrate to shared LlamaTransformer workspace after validation.
        // ====================================================================

        std::shared_ptr<TensorType> q_permute_tensor_optimized_;      // [B, NH, chunk, HS]
        std::shared_ptr<TensorType> preatt_tensor_optimized_;         // [B, NH, chunk, T]
        std::shared_ptr<TensorType> att_tensor_optimized_;            // [B, NH, chunk, T]
        std::shared_ptr<TensorType> v_out_tensor_optimized_;          // [B, NH, chunk, HS]

        // ====================================================================
        // Backward state tensors (training only, legacy path)
        // ====================================================================

        std::shared_ptr<TensorType> dq_tensor_;
        std::shared_ptr<TensorType> dK_tensor_;
        std::shared_ptr<TensorType> dV_tensor_;
        std::shared_ptr<TensorType> dK_exp_tensor_;
        std::shared_ptr<TensorType> dV_exp_tensor_;
        std::shared_ptr<TensorType> dpreatt_tensor_;
        std::shared_ptr<TensorType> datt_tensor_;
        std::shared_ptr<TensorType> dVout_tensor_;

        // ====================================================================
        // Raw device pointers — legacy path
        // ====================================================================

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
        // Raw device pointers — optimized path
        // TEMP: Remove legacy pointer block above once the optimized path is validated.
        // ====================================================================

        NativeType* k_opt_{ nullptr };
        NativeType* v_opt_{ nullptr };
        NativeType* q_permute_opt_{ nullptr };
        NativeType* preatt_opt_{ nullptr };
        NativeType* att_opt_{ nullptr };
        NativeType* v_out_opt_{ nullptr };
        NativeType* preatt_decode_opt_{ nullptr };
        NativeType* att_decode_opt_{ nullptr };
        NativeType* v_out_decode_opt_{ nullptr };

        std::size_t state_memory_size_{ 0 };

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

        static NativeType* raw( const std::shared_ptr<TensorType>& t )
        {
            return static_cast<NativeType*>(t->rawData());
        }

        // ====================================================================
        // Legacy path — initializeState
        // ====================================================================

        void initializeState( const BuildContext& build_context )
        {
            // REVIEW: GQA gradient tensors (dQ, dK, dV, dK_exp, dV_exp, dpreatt, datt, dVout)
            // are owned internally by this op rather than at the component level, deviating
            // from the standard Mila pattern. This was a pragmatic choice driven by the
            // complexity of the gradient tensor set not fitting a clean setGradients() signature.
            // Consequently, gradient memory is reported here as state rather than being
            // tracked separately. Revisit when GQA gradient ownership is migrated to the
            // component level.

            auto device = context_->getDeviceId();

            const shape_t q_shape = { B_, NH_,  T_, HS_ };
            const shape_t kv_shape = { B_, NKV_, T_, HS_ };
            const shape_t kv_exp = { B_, NH_,  T_, HS_ };

            const shape_t dec_att = { B_, NH_,  1,  T_ };
            const shape_t dec_vout = { B_, NH_,  1,  HS_ };

            auto make = [&]( const shape_t& shape, const std::string& name )
                {
                    auto tensor = std::make_shared<TensorType>( device, shape, name );
                    state_memory_size_ += tensor->getStorageSize();

                    return tensor;
                };

            // ================================================================
            // Always allocated — KV cache and decode path buffers
            // ================================================================

            // REVIEW: k_exp_tensor_ and v_exp_tensor_ are allocated at full T_ (max_seq_len) because
            // kvcache_expand_kv materializes the full KV history on every prefill chunk and decode step.
            // Eliminating these allocations would require a fused GQA kernel that reads directly from
            // k_tensor_/v_tensor_ and performs the GS head expansion implicitly, avoiding the expanded
            // materialization entirely. This is a significant architectural change best deferred to a
            // dedicated optimization pass.

            q_tensor_ = make( q_shape, "gqa.q" );
            q_ = raw( q_tensor_ );

            k_tensor_ = make( kv_shape, "gqa.k" );
            k_ = raw( k_tensor_ );

            v_tensor_ = make( kv_shape, "gqa.v" );
            v_ = raw( v_tensor_ );

            k_exp_tensor_ = make( kv_exp, "gqa.k_exp" );
            k_exp_ = raw( k_exp_tensor_ );

            v_exp_tensor_ = make( kv_exp, "gqa.v_exp" );
            v_exp_ = raw( v_exp_tensor_ );

            preatt_decode_tensor_ = make( dec_att, "gqa.preatt_decode" );
            preatt_decode_ = raw( preatt_decode_tensor_ );

            att_decode_tensor_ = make( dec_att, "gqa.att_decode" );
            att_decode_ = raw( att_decode_tensor_ );

            v_out_decode_tensor_ = make( dec_vout, "gqa.v_out_decode" );
            v_out_decode_ = raw( v_out_decode_tensor_ );

            // ================================================================
            // Inference only — prefill attention buffers
            // Sized to kPrefillChunkSize x context_length. Partial-chunk calls
            // write tight-packed output into the head of this allocation using
            // plans with strideC = chunk_len * T.
            // ================================================================

            if ( build_context.isInferenceMode() )
            {
                const shape_t prefill_att_shape = { B_, NH_, kPrefillChunkSize, T_ };
                const shape_t prefill_vout_shape = { B_, NH_, kPrefillChunkSize, HS_ };

                preatt_tensor_ = make( prefill_att_shape, "gqa.preatt_prefill" );
                preatt_ = raw( preatt_tensor_ );

                att_tensor_ = make( prefill_att_shape, "gqa.att_prefill" );
                att_ = raw( att_tensor_ );

                v_out_tensor_ = make( prefill_vout_shape, "gqa.v_out_prefill" );
                v_out_ = raw( v_out_tensor_ );
            }

            // ================================================================
            // Training only — full attention matrices and gradient buffers
            // ================================================================

            if ( build_context.isTrainingMode() )
            {
                const shape_t att_shape = { B_, NH_, T_, T_ };
                const shape_t vout_shape = { B_, NH_, T_, HS_ };

                preatt_tensor_ = make( att_shape, "gqa.preatt" );
                preatt_ = raw( preatt_tensor_ );

                att_tensor_ = make( att_shape, "gqa.att" );
                att_ = raw( att_tensor_ );

                v_out_tensor_ = make( vout_shape, "gqa.v_out" );
                v_out_ = raw( v_out_tensor_ );

                dq_tensor_ = make( q_shape, "gqa.dq" );
                dq_ = raw( dq_tensor_ );

                dK_tensor_ = make( kv_shape, "gqa.dK" );
                dK_ = raw( dK_tensor_ );

                dV_tensor_ = make( kv_shape, "gqa.dV" );
                dV_ = raw( dV_tensor_ );

                dK_exp_tensor_ = make( kv_exp, "gqa.dK_exp" );
                dK_exp_ = raw( dK_exp_tensor_ );

                dV_exp_tensor_ = make( kv_exp, "gqa.dV_exp" );
                dV_exp_ = raw( dV_exp_tensor_ );

                dpreatt_tensor_ = make( att_shape, "gqa.dpreatt" );
                dpreatt_ = raw( dpreatt_tensor_ );

                datt_tensor_ = make( att_shape, "gqa.datt" );
                datt_ = raw( datt_tensor_ );

                dVout_tensor_ = make( vout_shape, "gqa.dVout" );
                dVout_ = raw( dVout_tensor_ );
            }

            // DEBUG:
            std::cout << "CudaGQAOp::build: built with batch_size = " << B_
                << ", num_heads = " << NH_ << ", head_size = " << HS_ << ", max_seq_length = " << T_ << ", kPrefillChunkSize = " << kPrefillChunkSize << "\n";
            std::cout << "CudaGroupedQueryAttentionOp state memory size: "
                << (state_memory_size_ / (1024.0 * 1024.0)) << " MiB\n";
            // END DEBUG
        }

        // ====================================================================
        // Optimized path — initializeState_optimized
        // Allocates only the KV cache and transient scratch for the NKV-layout path.
        // TEMP: Remove initializeState() and rename this once the optimized path is validated.
        // ====================================================================

        void initializeState_optimized( const BuildContext& build_context )
        {
            auto device = context_->getDeviceId();

            const shape_t kv_shape = { B_, NKV_, T_, HS_ };
            const shape_t dec_att = { B_, NH_,  1,  T_ };
            const shape_t dec_vout = { B_, NH_,  1,  HS_ };

            auto make = [&]( const shape_t& shape, const std::string& name )
                {
                    auto tensor = std::make_shared<TensorType>( device, shape, name );
                    state_memory_size_ += tensor->getStorageSize();

                    return tensor;
                };

            // KV cache — compact NKV layout, retained across calls
            k_tensor_ = make( kv_shape, "gqa.k" );
            k_opt_ = raw( k_tensor_ );

            v_tensor_ = make( kv_shape, "gqa.v" );
            v_opt_ = raw( v_tensor_ );

            // Decode scratch — minor per-layer allocation, migrated to shared workspace in Phase 3
            //preatt_decode_tensor_ = make( dec_att, "gqa.preatt_decode" );
            //preatt_decode_opt_ = raw( preatt_decode_tensor_ );

            //att_decode_tensor_ = make( dec_att, "gqa.att_decode" );
            //att_decode_opt_ = raw( att_decode_tensor_ );

            //v_out_decode_tensor_ = make( dec_vout, "gqa.v_out_decode" );
            //v_out_decode_opt_ = raw( v_out_decode_tensor_ );

            //if ( build_context.isInferenceMode() )
            //{
            //    // Transient Q permute and attention scratch — self-allocated for the A/B period.
            //    // Migrated to the shared LlamaTransformer workspace in Phase 2/3.
            //    const shape_t q_perm_shape = { B_, NH_, kPrefillChunkSize, HS_ };
            //    const shape_t prefill_att_shape = { B_, NH_, kPrefillChunkSize, T_ };
            //    const shape_t prefill_vout_shape = { B_, NH_, kPrefillChunkSize, HS_ };

            //    q_permute_tensor_optimized_ = make( q_perm_shape, "gqa.q_perm_opt" );
            //    q_permute_opt_ = raw( q_permute_tensor_optimized_ );

            //    preatt_tensor_optimized_ = make( prefill_att_shape, "gqa.preatt_opt" );
            //    preatt_opt_ = raw( preatt_tensor_optimized_ );

            //    att_tensor_optimized_ = make( prefill_att_shape, "gqa.att_opt" );
            //    att_opt_ = raw( att_tensor_optimized_ );

            //    v_out_tensor_optimized_ = make( prefill_vout_shape, "gqa.v_out_opt" );
            //    v_out_opt_ = raw( v_out_tensor_optimized_ );
            //}

            // DEBUG:
            //std::cout << "CudaGQAOp::build [optimized]: batch_size=" << B_
            //    << " num_heads=" << NH_ << " num_kv_heads=" << NKV_
            //    << " head_size=" << HS_ << " max_seq_length=" << T_
            //    << " kPrefillChunkSize=" << kPrefillChunkSize << "\n";
            //std::cout << "CudaGroupedQueryAttentionOp [optimized] state memory size: "
            //    << (state_memory_size_ / (1024.0 * 1024.0)) << " MiB\n";
            // END DEBUG
        }

        // ====================================================================
        // Legacy path — buildCublasLtPlans
        // ====================================================================

        void buildCublasLtPlans()
        {
            const cudaDataType_t cuda_dt = getCudaDataType();
            cublasComputeType_t compute_type;
            cudaDataType_t scale_type;
            getComputeTypes( compute_type, scale_type );

            qk_score_plan_ = Detail::build_qk_score_plan<NativeType>(
                cublaslt_handle_, B_, NH_, T_, HS_,
                cuda_dt, compute_type, scale_type );

            att_value_plan_ = Detail::build_att_value_plan<NativeType>(
                cublaslt_handle_, B_, NH_, T_, HS_,
                cuda_dt, compute_type, scale_type );

            qk_prefill_plan_ = Detail::build_qk_prefill_plan<NativeType>(
                cublaslt_handle_, B_, NH_,
                static_cast<int>(kPrefillChunkSize), T_, HS_, prefill_chunk_size_,
                cuda_dt, compute_type, scale_type );

            att_value_prefill_plan_ = Detail::build_att_value_prefill_plan<NativeType>(
                cublaslt_handle_, B_, NH_,
                prefill_chunk_size_, T_, HS_, prefill_chunk_size_,
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

        // ====================================================================
        // Optimized path — buildCublasLtPlans_optimized
        // Builds plans against the compact [B, NKV, T, HS] layout. GS Q heads
        // are folded into M so batch_count = B * NKV throughout. No expansion
        // buffers are required. See GqaMemory.md Phase 1.
        // TEMP: Remove buildCublasLtPlans() and rename this once validated.
        // ====================================================================

        void buildCublasLtPlans_optimized()
        {
            const cudaDataType_t cuda_dt = getCudaDataType();
            cublasComputeType_t compute_type;
            cudaDataType_t scale_type;
            getComputeTypes( compute_type, scale_type );

            qk_prefill_plan_optimized_ = Detail::build_qk_prefill_plan_optimized<NativeType>(
                cublaslt_handle_,
                B_, NKV_, GS_,
                prefill_chunk_size_, T_, HS_,
                cuda_dt, compute_type, scale_type );

            att_value_prefill_plan_optimized_ = Detail::build_att_value_prefill_plan_optimized<NativeType>(
                cublaslt_handle_,
                B_, NKV_, GS_,
                prefill_chunk_size_, T_, HS_,
                cuda_dt, compute_type, scale_type );

            qk_decode_plan_optimized_ = Detail::build_qk_decode_plan_optimized<NativeType>(
                cublaslt_handle_,
                B_, NKV_, GS_,
                T_, HS_,
                cuda_dt, compute_type, scale_type );

            att_value_decode_plan_optimized_ = Detail::build_att_value_decode_plan_optimized<NativeType>(
                cublaslt_handle_,
                B_, NKV_, GS_,
                T_, HS_,
                cuda_dt, compute_type, scale_type );
        }

        // ====================================================================
        // Legacy path — prefillImpl / decodeImpl
        // Exact original bodies, renamed only to allow gate dispatch.
        // TEMP: Remove once the optimized path is validated.
        // ====================================================================

        void prefillImpl(
            const ITensor& q, const ITensor& k, const ITensor& v,
            ITensor& output,
            int position_offset )
        {
            const int chunk_len = static_cast<int>(q.shape()[ 1 ]);

            if ( position_offset < 0 || position_offset + chunk_len > active_max_seq_len_ )
            {
                throw std::invalid_argument( std::format(
                    "CudaGroupedQueryAttentionOp::prefill: position_offset {} + chunk_len {} exceeds max_seq_len {}",
                    position_offset, chunk_len, active_max_seq_len_ ) );
            }

            const NativeType* Xq = static_cast<const NativeType*>(q.rawData());
            const NativeType* Xk = static_cast<const NativeType*>(k.rawData());
            const NativeType* Xv = static_cast<const NativeType*>(v.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());
            cudaStream_t stream = context_->getStream();

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const float scale = 1.0f / sqrtf( static_cast<float>(HS_) );

            Detail::cuda_gqa_kernels<NativeType>::kvcache_write_q(
                q_, Xq, B_, chunk_len, NH_, HS_, position_offset, T_, stream );

            Detail::cuda_gqa_kernels<NativeType>::kvcache_write_kv(
                k_, v_, Xk, Xv, B_, chunk_len, NKV_, HS_, position_offset, T_, stream );

            const int total_kv_len = position_offset + chunk_len;

            Detail::cuda_gqa_kernels<NativeType>::kvcache_expand_kv(
                k_exp_, v_exp_, k_, v_,
                B_, total_kv_len, T_, NH_, NKV_, HS_, 0, stream );

            const bool is_full_chunk = (chunk_len == static_cast<int>(kPrefillChunkSize));
            const auto& qk_plan = is_full_chunk ? qk_prefill_plan_ : getOrBuildPartialQKPlan( chunk_len );
            const auto& av_plan = is_full_chunk ? att_value_prefill_plan_ : getOrBuildPartialAVPlan( chunk_len );

            const NativeType* q_chunk = q_ + static_cast<int64_t>(position_offset) * HS_;

            execute_plan<NativeType>(
                cublaslt_handle_, qk_plan,
                &scale, q_chunk, k_exp_, &beta, preatt_,
                nullptr, stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            Detail::cuda_gqa_kernels<NativeType>::prefill_softmax(
                att_, preatt_,
                B_, NH_, T_, kPrefillChunkSize, chunk_len, position_offset, stream );

            execute_plan<NativeType>(
                cublaslt_handle_, av_plan,
                &alpha, att_, v_exp_, &beta, v_out_,
                nullptr, stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            const int padded_T = is_full_chunk ? static_cast<int>(kPrefillChunkSize) : chunk_len;

            Detail::cuda_gqa_kernels<NativeType>::prefill_unpermute_output_padded(
                v_out_, Y,
                B_, chunk_len, padded_T, NH_, HS_, stream );

            cached_seq_len_ = position_offset + chunk_len;
        }

        void decodeImpl(
            const ITensor& q, const ITensor& k, const ITensor& v,
            ITensor& output,
            int position )
        {
            if ( position < 0 || position >= active_max_seq_len_ )
                throw std::invalid_argument(
                    "CudaGroupedQueryAttentionOp::decode position out of range" );

            const int actual_len = position + 1;

            const NativeType* Xq = static_cast<const NativeType*>( q.rawData() );
            const NativeType* Xk = static_cast<const NativeType*>( k.rawData() );
            const NativeType* Xv = static_cast<const NativeType*>( v.rawData() );
            NativeType* Y = static_cast<NativeType*>( output.rawData() );
            cudaStream_t stream = context_->getStream();

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const float scale = 1.0f / sqrtf( static_cast<float>(HS_) );

            Detail::cuda_gqa_kernels<NativeType>::kvcache_write_q(
                q_, Xq, B_, 1, NH_, HS_, position, T_, stream );

            Detail::cuda_gqa_kernels<NativeType>::kvcache_write_kv(
                k_, v_, Xk, Xv, B_, 1, NKV_, HS_, position, T_, stream );

            Detail::cuda_gqa_kernels<NativeType>::kvcache_expand_kv(
                k_exp_, v_exp_, k_, v_,
                B_, actual_len, T_, NH_, NKV_, HS_, 0, stream );

            const NativeType* q_decode = q_ + static_cast<int64_t>(position) * HS_;

            execute_plan<NativeType>(
                cublaslt_handle_, qk_decode_plan_,
                &scale, q_decode, k_exp_, &beta, preatt_decode_,
                nullptr, stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            Detail::cuda_gqa_kernels<NativeType>::softmax_decode_forward(
                att_decode_, 1.0f, preatt_decode_,
                B_, NH_, T_, actual_len, stream );

            execute_plan<NativeType>(
                cublaslt_handle_, att_value_decode_plan_,
                &alpha, att_decode_, v_exp_, &beta, v_out_decode_,
                nullptr, stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            Detail::cuda_gqa_kernels<NativeType>::unpermute_output(
                v_out_decode_, Y, B_, 1, NH_, HS_, stream );

            if ( actual_len > cached_seq_len_ )
                cached_seq_len_ = actual_len;
        }

        // ====================================================================
        // Optimized path — prefill_optimized / decode_optimized
        // NKV-layout plans, no q_tensor_, no expansion buffers.
        // See GqaMemory.md Phase 1 and Phase 2.
        // TEMP: Bodies are stubs pending Phase 1 plan builder implementation.
        //       Replace stub bodies with full NKV-layout implementation in Phase 1.
        // ====================================================================

        void prefill_optimized(
            const ITensor& q, const ITensor& k, const ITensor& v,
            ITensor& output,
            int position_offset )
        {
            const int chunk_len = static_cast<int>(q.shape()[ 1 ]);

            if ( position_offset < 0 || position_offset + chunk_len > active_max_seq_len_ )
            {
                throw std::invalid_argument( std::format(
                    "CudaGroupedQueryAttentionOp::prefill_optimized: position_offset {} + chunk_len {} exceeds max_seq_len {}",
                    position_offset, chunk_len, active_max_seq_len_ ) );
            }

            const NativeType* Xq = static_cast<const NativeType*>(q.rawData());
            const NativeType* Xk = static_cast<const NativeType*>(k.rawData());
            const NativeType* Xv = static_cast<const NativeType*>(v.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());
            cudaStream_t stream = context_->getStream();

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const float scale = 1.0f / sqrtf( static_cast<float>(HS_) );

            // Write K/V into compact [B, NKV, T, HS] cache at position_offset
            Detail::cuda_gqa_kernels<NativeType>::kvcache_write_kv(
                k_opt_, v_opt_, Xk, Xv, B_, chunk_len, NKV_, HS_, position_offset, T_, stream );

            // Permute Q from [B, chunk, NH*HS] into compact [B, NH, chunk, HS] scratch.
            // strideA = chunk*HS between heads — matches the NKV-layout plan geometry.
            Detail::cuda_gqa_kernels<NativeType>::permute_q_compact(
                q_permute_opt_, Xq, B_, chunk_len, NH_, HS_, stream );

            const bool is_full_chunk = (chunk_len == static_cast<int>(kPrefillChunkSize));
            const auto& qk_plan = is_full_chunk
                ? qk_prefill_plan_optimized_
                : getOrBuildPartialQKPlan_optimized( chunk_len );
            const auto& av_plan = is_full_chunk
                ? att_value_prefill_plan_optimized_
                : getOrBuildPartialAVPlan_optimized( chunk_len );

            execute_plan<NativeType>(
                cublaslt_handle_, qk_plan,
                &scale, q_permute_opt_, k_opt_, &beta, preatt_opt_,
                nullptr, stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            Detail::cuda_gqa_kernels<NativeType>::prefill_softmax(
                att_opt_, preatt_opt_,
                B_, NH_, T_, kPrefillChunkSize, chunk_len, position_offset, stream );

            execute_plan<NativeType>(
                cublaslt_handle_, av_plan,
                &alpha, att_opt_, v_opt_, &beta, v_out_opt_,
                nullptr, stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            // POSSIBLE BUG: Was - const int padded_T = static_cast<int>(kPrefillChunkSize);
            const int padded_T = is_full_chunk ? static_cast<int>(kPrefillChunkSize) : chunk_len;

            Detail::cuda_gqa_kernels<NativeType>::prefill_unpermute_output_padded(
                v_out_opt_, Y,
                B_, chunk_len, padded_T, NH_, HS_, stream );

            cached_seq_len_ = position_offset + chunk_len;
        }

        void decode_optimized(
            const ITensor& q, const ITensor& k, const ITensor& v,
            ITensor& output,
            int position )
        {
            if ( position < 0 || position >= active_max_seq_len_ )
                throw std::invalid_argument(
                    "CudaGroupedQueryAttentionOp::decode_optimized position out of range" );

            const int actual_len = position + 1;

            const NativeType* Xq = static_cast<const NativeType*>( q.rawData() );
            const NativeType* Xk = static_cast<const NativeType*>( k.rawData() );
            const NativeType* Xv = static_cast<const NativeType*>( v.rawData() );
            NativeType* Y = static_cast<NativeType*>( output.rawData() );
            cudaStream_t stream = context_->getStream();

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const float scale = 1.0f / sqrtf( static_cast<float>(HS_) );

            // Write K/V into compact [B, NKV, T, HS] cache at position
            Detail::cuda_gqa_kernels<NativeType>::kvcache_write_kv(
                k_opt_, v_opt_, Xk, Xv, B_, 1, NKV_, HS_, position, T_, stream );

            // Permute single Q token from [B, 1, NH*HS] into compact [B, NH, 1, HS] scratch.
            // strideA = 1*HS = HS between heads — matches the NKV-layout decode plan geometry.
            Detail::cuda_gqa_kernels<NativeType>::permute_q_compact(
                q_permute_opt_, Xq, B_, 1, NH_, HS_, stream );

            execute_plan<NativeType>(
                cublaslt_handle_, qk_decode_plan_optimized_,
                &scale, q_permute_opt_, k_opt_, &beta, preatt_decode_opt_,
                nullptr, stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            Detail::cuda_gqa_kernels<NativeType>::softmax_decode_forward(
                att_decode_opt_, 1.0f, preatt_decode_opt_,
                B_, NH_, T_, actual_len, stream );

            execute_plan<NativeType>(
                cublaslt_handle_, att_value_decode_plan_optimized_,
                &alpha, att_decode_opt_, v_opt_, &beta, v_out_decode_opt_,
                nullptr, stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            Detail::cuda_gqa_kernels<NativeType>::unpermute_output(
                v_out_decode_opt_, Y, B_, 1, NH_, HS_, stream );

            if ( actual_len > cached_seq_len_ )
                cached_seq_len_ = actual_len;
        }
        // ====================================================================
        // Legacy partial-chunk plan cache helpers
        // ====================================================================

        const Detail::CublasLtMatMulPlan<NativeType>& getOrBuildPartialQKPlan( int chunk_len )
        {
            auto it = qk_partial_prefill_plan_cache_.find( chunk_len );

            if ( it == qk_partial_prefill_plan_cache_.end() )
            {
                const cudaDataType_t cuda_dt = getCudaDataType();
                cublasComputeType_t compute_type;
                cudaDataType_t scale_type;
                getComputeTypes( compute_type, scale_type );

                it = qk_partial_prefill_plan_cache_.emplace(
                    chunk_len,
                    Detail::build_qk_prefill_plan<NativeType>(
                        cublaslt_handle_, B_, NH_, chunk_len, T_, HS_, prefill_chunk_size_,
                        cuda_dt, compute_type, scale_type ) ).first;
            }

            return it->second;
        }

        const Detail::CublasLtMatMulPlan<NativeType>& getOrBuildPartialAVPlan( int chunk_len )
        {
            auto it = att_value_partial_prefill_plan_cache_.find( chunk_len );

            if ( it == att_value_partial_prefill_plan_cache_.end() )
            {
                const cudaDataType_t cuda_dt = getCudaDataType();
                cublasComputeType_t compute_type;
                cudaDataType_t scale_type;
                getComputeTypes( compute_type, scale_type );

                it = att_value_partial_prefill_plan_cache_.emplace(
                    chunk_len,
                    Detail::build_att_value_prefill_plan<NativeType>(
                        cublaslt_handle_, B_, NH_, chunk_len, T_, HS_, prefill_chunk_size_,
                        cuda_dt, compute_type, scale_type ) ).first;
            }

            return it->second;
        }

        // ====================================================================
        // Optimized partial-chunk plan cache helpers
        // TEMP: Remove legacy helpers above and rename these once validated.
        // ====================================================================

        const Detail::CublasLtMatMulPlan<NativeType>& getOrBuildPartialQKPlan_optimized( int chunk_len )
        {
            auto it = qk_partial_prefill_plan_cache_optimized_.find( chunk_len );

            if ( it == qk_partial_prefill_plan_cache_optimized_.end() )
            {
                const cudaDataType_t cuda_dt = getCudaDataType();
                cublasComputeType_t compute_type;
                cudaDataType_t scale_type;
                getComputeTypes( compute_type, scale_type );

                it = qk_partial_prefill_plan_cache_optimized_.emplace(
                    chunk_len,
                    Detail::build_qk_prefill_plan_optimized<NativeType>(
                        cublaslt_handle_,
                        B_, NKV_, GS_,
                        chunk_len, T_, HS_,
                        cuda_dt, compute_type, scale_type ) ).first;
            }

            return it->second;
        }

        const Detail::CublasLtMatMulPlan<NativeType>& getOrBuildPartialAVPlan_optimized( int chunk_len )
        {
            auto it = att_value_partial_prefill_plan_cache_optimized_.find( chunk_len );

            if ( it == att_value_partial_prefill_plan_cache_optimized_.end() )
            {
                const cudaDataType_t cuda_dt = getCudaDataType();
                cublasComputeType_t compute_type;
                cudaDataType_t scale_type;
                getComputeTypes( compute_type, scale_type );

                it = att_value_partial_prefill_plan_cache_optimized_.emplace(
                    chunk_len,
                    Detail::build_att_value_prefill_plan_optimized<NativeType>(
                        cublaslt_handle_,
                        B_, NKV_, GS_,
                        chunk_len, T_, HS_,
                        cuda_dt, compute_type, scale_type ) ).first;
            }

            return it->second;
        }

        // ====================================================================
        // Precision helpers — shared by both paths
        // ====================================================================

        cudaDataType_t getCudaDataType() const
        {
            if constexpr ( std::is_same_v<NativeType, float> )
                return CUDA_R_32F;
            else if constexpr ( std::is_same_v<NativeType, nv_bfloat16> )
                return CUDA_R_16BF;
            else if constexpr ( std::is_same_v<NativeType, half> )
                return CUDA_R_16F;
        }

        void getComputeTypes( cublasComputeType_t& compute_type, cudaDataType_t& scale_type ) const
        {
            scale_type = CUDA_R_32F;

            if constexpr ( std::is_same_v<NativeType, half> )
            {
                compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
            }
            else if constexpr ( std::is_same_v<NativeType, nv_bfloat16> )
            {
                compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
            }
            else
            {
                compute_type = CUBLAS_COMPUTE_32F;
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

            //OperationRegistry::instance()
            //    .registerUnaryOperation<DeviceType::Cuda,
            //    TensorDataType::FP32,
            //    TensorDataType::FP32>(
            //        opName,
            //        []( IExecutionContext* ctx,
            //            const ComponentConfig& cfg )
            //        -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP32>>
            //        {
            //            return std::make_shared<CudaGroupedQueryAttentionOp<TensorDataType::FP32>>(
            //                ctx,
            //                static_cast<const GroupedQueryAttentionConfig&>(cfg) );
            //        } );

            //OperationRegistry::instance()
            //    .registerUnaryOperation<DeviceType::Cuda, TensorDataType::BF16, TensorDataType::BF16>(
            //        opName,
            //        []( IExecutionContext* ctx,
            //            const ComponentConfig& cfg )
            //        -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::BF16>>
            //        {
            //            return std::make_shared<CudaGroupedQueryAttentionOp<TensorDataType::BF16>>(
            //                ctx,
            //                static_cast<const GroupedQueryAttentionConfig&>(cfg) );
            //        } );
        }
    };
}