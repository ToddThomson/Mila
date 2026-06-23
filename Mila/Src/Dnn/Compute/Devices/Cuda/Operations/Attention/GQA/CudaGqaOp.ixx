/**
 * @file CudaGqaOp.ixx
 * @brief CUDA Grouped-Query Attention (GQA) operation using cuBLASLt.
 */

module;
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <format>
#include <stdexcept>
#include <cstdint>
#include <type_traits>
#include <sstream>
#include <unordered_map>
#include "Kernels/CudaGqa.cuh"

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
import Compute.Device;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.OperationType;
import Dnn.Component;
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

    /**
     * @brief CUDA Grouped-Query Attention operation (inference only).
     *
     * GQA generalises MHA by allowing num_kv_heads < num_heads. Every group of
     * (num_heads / num_kv_heads) Q heads shares a single K/V head, reducing
     * KV cache memory and bandwidth proportionally to the group size.
     *
     * cuBLASLt plans are built against the compact [B, NKV, T, HS] layout with
     * grouped head strides (GS Q heads folded into M, batch_count = B * NKV), so
     * K and V are never materialized into an expanded [B, NH, T, HS] buffer. The
     * KV cache is the only retained allocation; all transient scratch is supplied
     * via setState() from the shared LlamaTransformer workspace. See GqaMemory.md
     * Phase 1 and Phase 2. A training backward() is a future aspiration and is not
     * implemented (Llama 3.x is inference only).
     *
     * Prefill pass (with KV cache):
     *  1. kvcache_write_kv          -> append K/V into compact [B,NKV,T,HS] cache
     *  2. permute_q_compact         -> Q [B,NH,chunk,HS] scratch
     *  3. qk plan                   -> preatt [B,NH,chunk,T]
     *  4. prefill_softmax           -> att [B,NH,chunk,T]
     *  5. att-value plan            -> v_out [B,NH,chunk,HS]
     *  6. prefill_unpermute_output  -> Y [B,chunk,C]
     *
     * Decode pass (one token):
     *  1. kvcache_write_kv          -> append single K/V into the cache
     *  2. permute_q_compact         -> single Q token [B,NH,1,HS] scratch
     *  3. qk decode plan            -> preatt_decode [B,NH,1,T]
     *  4. softmax_decode            -> att_decode
     *  5. att-value decode plan     -> v_out_decode [B,NH,1,HS]
     *  6. unpermute_output          -> Y [B,1,C]
     *
     * @tparam TPrecision Tensor element type and cuBLASLt data/compute type.
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaGqaOp : public Operation<DeviceType::Cuda, TPrecision>, public IKvInference
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ConfigType = GqaConfig;

        CudaGqaOp( IExecutionContext* context, const GqaConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaGqaOp" ) )
            , config_( config )
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
         * Must be called before prefill() or decode().
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

            prefill_optimized( q, k, v, output, position_offset );
        }

        void decode(
            const ITensor& q, const ITensor& k, const ITensor& v,
            ITensor& output,
            int position ) override
        {
            ensureKVCacheEnabled();

            decode_optimized( q, k, v, output, position );
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
            window_ = static_cast<int>(config_.getWindow());
            attention_scale_ = config_.getAttentionScale();

            // Tuned prefill chunk size, threaded down from LlamaTransformer via BuildContext.
            // Training-mode contexts carry no prefill size; fall back to the full sequence
            // length so the (vestigial) prefill plans are still built with a valid row count.
            const int64_t prefill_size = context.getPrefillSize();
            prefill_chunk_size_ = static_cast<int>( prefill_size > 0 ? prefill_size : T_ );

            active_max_seq_len_ = T_;
            cached_seq_len_ = 0;
            kv_cache_enabled_ = false;

            initializeState_optimized( context );

            cublaslt_handle_ = context_->getCublasLtHandle();

            if ( !cublaslt_handle_ )
                throw std::runtime_error(
                    "CudaGroupedQueryAttentionOp requires cuBLASLt. "
                    "Ensure CUDA 10.1 or newer." );

            buildCublasLtPlans_optimized();

            Operation<DeviceType::Cuda, TPrecision>::build( context );
        }

        // ====================================================================
        // FUTURE: GQA training path (not implemented — Llama 3.x is inference only)
        //
        // These are the standalone full-sequence (non-KV-cache) training entry
        // points. They are throwing stubs today so the GroupedQueryAttention
        // component contract compiles; drive GQA via prefill()/decode() for
        // inference. The bodies are intentionally left unimplemented rather than
        // ported, because the retired legacy versions ran on the EXPANDED
        // [B,NH,T,HS] layout that this op no longer allocates.
        //
        // A real implementation is a deliberate future design task. An open
        // question (see GqaMemory.md): keep the expanded layout for training (its
        // expand_kv <-> reduce_kv_grad pairing makes the gradient clean) or derive
        // the backward on the compact NKV layout used for inference. The retired
        // expanded-layout sketch — preserved here as the structural starting point,
        // never numerically validated — was:
        //
        //   forward  (full sequence, caches preatt/att [B,NH,T,T], v_out [B,NH,T,HS]):
        //     1. permute_qkv      -> Q[B,NH,T,HS], K/V[B,NKV,T,HS]
        //     2. expand_kv        -> k_exp/v_exp [B,NH,T,HS]
        //     3. qk score gemm    -> preatt [B,NH,T,T]
        //     4. softmax_forward  -> att
        //     5. att-value gemm   -> v_out [B,NH,T,HS]
        //     6. unpermute_output -> Y [B,T,NH*HS]
        //
        //   backward (needs dq/dK/dV, dK_exp/dV_exp, dpreatt/datt/dVout buffers):
        //     1. unpermute_backward -> dVout
        //     2. dV_exp = att^T @ dVout
        //     3. dAtt  = dVout @ v_exp^T
        //     4. softmax_backward   -> dPreatt
        //     5. dQ    = dPreatt @ k_exp
        //     6. dK_exp = dPreatt^T @ Q
        //     7. reduce_kv_grad     -> dK/dV [B,NKV,T,HS] (sum over group)
        //     8. permute_backward   -> dX [B,T,(NH+2*NKV)*HS]
        // ====================================================================

        void forward( const ITensor&, ITensor& ) const
        {
            throw std::runtime_error(
                "CudaGqaOp::forward (standalone training path) is not implemented; "
                "GQA is inference only - drive via prefill()/decode()." );
        }

        void backward( const ITensor&, const ITensor&, ITensor& ) const
        {
            throw std::runtime_error(
                "CudaGqaOp::backward is not implemented; GQA is inference only. "
                "See the FUTURE note above for the retired expanded-layout sketch." );
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

        // -------- Static model topology (set in build(), never change) --------

        int B_{ 0 };   ///< Batch size
        int T_{ 0 };   ///< Max sequence length
        int C_{ 0 };   ///< Model dim  = NH * HS
        int NH_{ 0 };  ///< Number of Q heads
        int NKV_{ 0 }; ///< Number of KV heads
        int HS_{ 0 };  ///< Head dim   = C / NH
        int GS_{ 0 };  ///< Group size = NH / NKV
        int window_{ 0 }; ///< Sliding-window size (0 = global/unbounded causal)
        float attention_scale_{ 0.0f }; ///< QK softmax scale (config-derived; 1/sqrt(HS) for Llama, 1.0 for Gemma)

        int prefill_chunk_size_{ 0 };
        int active_max_seq_len_{ 0 };
        int cached_seq_len_{ 0 };
        bool kv_cache_enabled_{ false };

        cublasLtHandle_t cublaslt_handle_{ nullptr };

        // ====================================================================
        // cuBLASLt plans (compact NKV layout, grouped head strides)
        // ====================================================================

        Detail::CublasLtMatMulPlan<NativeType> qk_prefill_plan_optimized_;
        Detail::CublasLtMatMulPlan<NativeType> att_value_prefill_plan_optimized_;
        Detail::CublasLtMatMulPlan<NativeType> qk_decode_plan_optimized_;
        Detail::CublasLtMatMulPlan<NativeType> att_value_decode_plan_optimized_;

        std::unordered_map<int, Detail::CublasLtMatMulPlan<NativeType>> qk_partial_prefill_plan_cache_optimized_;
        std::unordered_map<int, Detail::CublasLtMatMulPlan<NativeType>> att_value_partial_prefill_plan_cache_optimized_;

        // ====================================================================
        // State tensors — KV cache (compact NKV layout, retained across calls)
        // ====================================================================

        std::shared_ptr<TensorType> k_tensor_;     // [B, NKV, T, HS]
        std::shared_ptr<TensorType> v_tensor_;     // [B, NKV, T, HS]

        // ====================================================================
        // Raw device pointers — KV cache plus shared transient scratch (via setState)
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
        // initializeState_optimized — allocates the compact NKV-layout KV cache.
        // Transient scratch is wired externally via setState() from the shared
        // LlamaTransformer workspace.
        // ====================================================================

        void initializeState_optimized( const BuildContext& )
        {
            auto device = context_->getDeviceId();

            const shape_t kv_shape = { B_, NKV_, T_, HS_ };

            auto make = [&]( const shape_t& shape, const std::string& name )
                {
                    auto tensor = std::make_shared<TensorType>( device, shape, name );
                    state_memory_size_ += tensor->getStorageSize();

                    return tensor;
                };

            // KV cache — compact NKV layout, retained across calls. All transient
            // scratch (Q permute, prefill/decode attention buffers) is wired via
            // setState() from the shared LlamaTransformer workspace.
            k_tensor_ = make( kv_shape, "gqa.k" );
            k_opt_ = raw( k_tensor_ );

            v_tensor_ = make( kv_shape, "gqa.v" );
            v_opt_ = raw( v_tensor_ );
        }

        // ====================================================================
        // buildCublasLtPlans_optimized — plans against the compact [B, NKV, T, HS]
        // layout. GS Q heads are folded into M so batch_count = B * NKV throughout.
        // No expansion buffers required. See GqaMemory.md Phase 1.
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
        // prefill_optimized / decode_optimized — NKV-layout plans, no expansion
        // buffers. See GqaMemory.md Phase 1 and Phase 2.
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
            const float scale = attention_scale_;   // config-derived: 1/sqrt(HS) (Llama) or explicit (Gemma 1.0)

            // Write K/V into compact [B, NKV, T, HS] cache at position_offset
            Detail::cuda_gqa_kernels<NativeType>::kvcache_write_kv(
                k_opt_, v_opt_, Xk, Xv, B_, chunk_len, NKV_, HS_, position_offset, T_, stream );

            // Permute Q from [B, chunk, NH*HS] into compact [B, NH, chunk, HS] scratch.
            // strideA = chunk*HS between heads — matches the NKV-layout plan geometry.
            Detail::cuda_gqa_kernels<NativeType>::permute_q_compact(
                q_permute_opt_, Xq, B_, chunk_len, NH_, HS_, stream );

            const bool is_full_chunk = (chunk_len == prefill_chunk_size_);
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
                B_, NH_, T_, prefill_chunk_size_, chunk_len, position_offset, window_, stream );

            execute_plan<NativeType>(
                cublaslt_handle_, av_plan,
                &alpha, att_opt_, v_opt_, &beta, v_out_opt_,
                nullptr, stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );

            const int padded_T = is_full_chunk ? prefill_chunk_size_ : chunk_len;

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
            const float scale = attention_scale_;   // config-derived: 1/sqrt(HS) (Llama) or explicit (Gemma 1.0)

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
                B_, NH_, T_, actual_len, window_, stream );

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
        // Partial-chunk plan cache helpers (compact NKV layout)
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

}
