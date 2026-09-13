/**
 * @file CudaRopeOp.ixx
 * @brief CUDA implementation of the Rope (rotary positional embedding) operation.
 *
 * Applies RoPE to projected Q and K tensors in preparation for GQA attention.
 * Supports full-sequence forward/backward, chunked prefill with position offset,
 * and single-token decode via IPositionalPairedOp.
 */

module;
#include <cuda_fp16.h>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <format>
#include <sstream>
#include <iostream>

export module Compute.CudaRopeOp;
import :Dispatch;
import :Cache;

import Dnn.Component;
import Dnn.Components.RopeConfig;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.OperationBase;
import Compute.IPositionalPairedOp;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.OperationType;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;

import Logging.Logger;
import Cuda.Debug;

namespace Mila::Dnn::Compute::Cuda::Rope
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute::Cuda; // For DEBUG:

    // ========================================================================
    // CudaRopeOp
    // ========================================================================

    /**
     * @brief CUDA implementation of the Rope (rotary positional embedding) operation.
     *
     * Takes the projected Q and K tensors produced by linear layers and applies
     * position-dependent rotations so that attention scores encode relative position
     * implicitly through the inner product.
     *
     * Design:
     * - No learned parameters. The cos/sin cache holds one row per position the build
     *   context's sequence length covers, and is shared across all ops with identical
     *   parameters via RopeCacheRegistry. A row depends only on its position, so a shorter
     *   table holds exactly the leading rows of a longer one. build_cache() is called
     *   exactly once per unique key.
     * - Two-phase initialization: build() acquires the shared cache and validates
     *   shapes; forward(), backward(), prefill(), and decode() are pure hot-path
     *   dispatch.
     * - GQA-aware: Q and K may have different head counts (n_heads vs n_kv_heads).
     * - Backward is exact: RoPE is an orthogonal rotation, so the gradient is the
     *   inverse rotation (negate sin terms). No extra buffers needed.
     *
     * Input/output shapes:
     *   Q:  [B, T, n_heads,    head_dim]
     *   K:  [B, T, n_kv_heads, head_dim]
     *   Q', K' -- same shapes as inputs.
     *
     * Decode shapes (T=1, explicit position):
     *   Q:  [B, 1, n_heads,    head_dim]
     *   K:  [B, 1, n_kv_heads, head_dim]
     *
     * @tparam TPrecision Precision of Q/K tensors (FP32 or FP16).
     */
    export template<TensorDataType TComputePrecision>
        requires PrecisionSupportedOnDevice<TComputePrecision, DeviceType::Cuda>
    class CudaRopeOp : public Operation<DeviceType::Cuda, TComputePrecision>, public IPositionalPairedOp
    {
    public:

        using MR = CudaDeviceMemoryResource;
        using TensorType = Tensor<TComputePrecision, MR>;
        using ComputeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TComputePrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ConfigType = RopeConfig;
        using CacheKey = RopeCacheRegistry::CacheKey;

        CudaRopeOp( IExecutionContext* context, const RopeConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaRopeOp" ) ), config_( config )
        {
            config_.validate();
        }

        ~CudaRopeOp()
        {
            releaseCache();
        }

        CudaRopeOp( const CudaRopeOp& ) = delete;
        CudaRopeOp& operator=( const CudaRopeOp& ) = delete;

        CudaRopeOp( CudaRopeOp&& other ) noexcept
            : context_( other.context_ )
            , config_( std::move( other.config_ ) )
            , cos_cache_( other.cos_cache_ )
            , owns_cache_( other.owns_cache_ )
            , sin_cache_( other.sin_cache_ )
            , cache_key_( other.cache_key_ )
            , batch_size_( other.batch_size_ )
            , seq_length_( other.seq_length_ )
        {
            this->is_built_ = other.is_built_;
            other.cos_cache_ = nullptr;
            other.sin_cache_ = nullptr;
            other.owns_cache_ = false;
            other.is_built_ = false;
        }

        CudaRopeOp& operator=( CudaRopeOp&& other ) noexcept
        {
            if ( this != &other )
            {
                releaseCache();
                context_ = other.context_;
                config_ = std::move( other.config_ );
                owns_cache_ = other.owns_cache_;
                cos_cache_ = other.cos_cache_;
                sin_cache_ = other.sin_cache_;
                cache_key_ = other.cache_key_;
                batch_size_ = other.batch_size_;
                seq_length_ = other.seq_length_;
                this->is_built_ = other.is_built_;

                other.cos_cache_ = nullptr;
                other.sin_cache_ = nullptr;
                other.owns_cache_ = false;
                other.is_built_ = false;
            }

            return *this;
        }

        /**
         * @brief Prepare the operation for a concrete input shape (cold path).
         *
         * The sequence length T of the build context is the number of positions this op
         * can ever rotate: the tables hold T rows, and prefill and decode refuse any
         * position at or past T. A caller that decodes must therefore build at the full
         * context length, not at a prefill chunk.
         *
         * @param build_context  Build context carrying the Q/K input shape [B, T, ...].
         * @throws std::invalid_argument if T exceeds the trained maximum sequence length.
         */
        void build( const BuildContext& build_context ) override
        {
            const auto& shape = build_context.inputShape();
            const dim_t table_rows = shape[ 1 ];

            if ( table_rows > config_.getMaxSequenceLength() )
                throw std::invalid_argument( std::format(
                    "CudaRopeOp::build: sequence length {} exceeds the trained maximum {}",
                    table_rows, config_.getMaxSequenceLength() ) );

            batch_size_ = static_cast<int>(shape[ 0 ]);
            seq_length_ = static_cast<int>(table_rows);

            const CacheKey cache_key = makeCacheKey( table_rows );

            if ( this->is_built_ && cache_key == cache_key_ )
                return;

            releaseCache();

            // NOTE: Cache data type is always float32 regardless of input precision to
            // preserve accuracy of the trigonometric computations.

            cache_key_ = cache_key;

            auto [cos_ptr, sin_ptr, is_new] =
                RopeCacheRegistry::instance().acquire( cache_key_, tableBytes( table_rows ) );

            owns_cache_ = is_new;

            cos_cache_ = static_cast<float*>(cos_ptr);
            sin_cache_ = static_cast<float*>(sin_ptr);

            if ( is_new )
            {
                Detail::cuda_rope_impl<ComputeType>::build_cache(
                    cos_cache_, sin_cache_,
                    static_cast<int>(table_rows),
                    static_cast<int>(config_.getHeadDim()),
                    config_.getBase(),
                    static_cast<int>(config_.getRotaryDim()),
                    rotaryLayoutCode(),
                    context_->getStream() );

                // Ensure cache is ready before any op can use it.
                context_->synchronize(); 
            }

            this->is_built_ = true;
        }

        /**
         * @brief Full-sequence forward pass.
         *
         * Applies RoPE to Q and K across the full sequence with position_offset = 0.
         * Used for training forward passes.
         */
        void forward(
            const ITensor& Q_in, const ITensor& K_in,
            ITensor& Q_out, ITensor& K_out ) const
        {
            ensureBuilt();

            const auto& q_shape = Q_in.shape();
            int B = static_cast<int>(q_shape[ 0 ]);
            int T = static_cast<int>(q_shape[ 1 ]);

            validateRuntimeShape( B, T );

            dispatchForward( Q_in, K_in, Q_out, K_out, B, T, 0 );
        }

        // ====================================================================
        // Backward (training)
        // ====================================================================

        /**
         * @brief Backward pass (hot path).
         *
         * RoPE is an orthogonal rotation (R^T R = I), so the Jacobian is R^T.
         * The backward pass is therefore the inverse rotation: rotate the upstream
         * gradients by -theta (negate sin terms). No new parameters are accumulated.
         */
        void backward(
            const ITensor& dQ_out, const ITensor& dK_out,
            ITensor& dQ_in, ITensor& dK_in ) const
        {
            ensureBuilt();

            const auto& q_shape = dQ_out.shape();
            int B = static_cast<int>(q_shape[ 0 ]);
            int T = static_cast<int>(q_shape[ 1 ]);

            validateRuntimeShape( B, T );

            Detail::cuda_rope_impl<ComputeType>::backward(
                static_cast<ComputeType*>(dQ_in.rawData()),
                static_cast<ComputeType*>(dK_in.rawData()),
                static_cast<const ComputeType*>(dQ_out.rawData()),
                static_cast<const ComputeType*>(dK_out.rawData()),
                cos_cache_, sin_cache_,
                B, T,
                static_cast<int>(config_.getNumHeads()),
                static_cast<int>(config_.getNumKVHeads()),
                static_cast<int>(config_.getHeadDim()),
                static_cast<int>(config_.getRotaryDim()),
                rotaryLayoutCode(),
                context_->getStream() );
        }

        // ====================================================================
        // Positional inference (IPositionalPairedOp)
        // ====================================================================

        /**
         * @brief Chunked prefill with explicit position offset.
         *
         * Applies RoPE to Q and K using absolute positions
         * [position_offset .. position_offset + T - 1] for the cos/sin cache lookup.
         *
         * @param Q_in            Input Q  [B, T, n_heads,    head_dim].
         * @param K_in            Input K  [B, T, n_kv_heads, head_dim].
         * @param Q_out           Output Q [B, T, n_heads,    head_dim].
         * @param K_out           Output K [B, T, n_kv_heads, head_dim].
         * @param position_offset Absolute position of the first token in this chunk.
         */
        void prefill(
            const ITensor& Q_in, const ITensor& K_in,
            ITensor& Q_out, ITensor& K_out,
            dim_t position_offset ) override
        {
            ensureBuilt();

            const auto& q_shape = Q_in.shape();
            int B = static_cast<int>(q_shape[ 0 ]);
            int T = static_cast<int>(q_shape[ 1 ]);

            if ( position_offset < 0 || position_offset + T > seq_length_ )
                throw std::invalid_argument( std::format(
                    "CudaRopeOp::prefill: position_offset {} + T {} exceeds the {} positions this op was built for",
                    position_offset, T, seq_length_ ) );

            requireInPlaceForPrefixLayout( Q_in.rawData(), Q_out.rawData(), "prefill" );
            requireInPlaceForPrefixLayout( K_in.rawData(), K_out.rawData(), "prefill" );

            dispatchForward( Q_in, K_in, Q_out, K_out, B, T, narrowToKernelIndex( position_offset ) );
        }

        /**
         * @brief Single-token decode with explicit position.
         *
         * Reads only the cache row at `position`. Used for KV-cache autoregressive
         * generation where T=1.
         *
         * @param Q_in   Input Q  [B, 1, n_heads,    head_dim].
         * @param K_in   Input K  [B, 1, n_kv_heads, head_dim].
         * @param Q_out  Output Q [B, 1, n_heads,    head_dim].
         * @param K_out  Output K [B, 1, n_kv_heads, head_dim].
         * @param position Zero-based absolute sequence position.
         */
        void decode(
            const ITensor& Q_in, const ITensor& K_in,
            ITensor& Q_out, ITensor& K_out,
            dim_t position ) override
        {
            ensureBuilt();

            if ( position < 0 || position >= seq_length_ )
                throw std::invalid_argument( std::format(
                    "CudaRopeOp::decode: position {} out of range [0, {})",
                    position, seq_length_ ) );

            int B = static_cast<int>( Q_in.shape()[ 0 ] );

            Detail::cuda_rope_impl<ComputeType>::decode(
                static_cast<ComputeType*>( Q_out.rawData() ),
                static_cast<ComputeType*>( K_out.rawData() ),
                static_cast<const ComputeType*>( Q_in.rawData() ),
                static_cast<const ComputeType*>( K_in.rawData() ),
                cos_cache_, sin_cache_,
                B, narrowToKernelIndex( position ),
                static_cast<int>(config_.getNumHeads()),
                static_cast<int>(config_.getNumKVHeads()),
                static_cast<int>(config_.getHeadDim()),
                static_cast<int>(config_.getRotaryDim()),
                rotaryLayoutCode(),
                context_->getStream() );
            
            // DEBUG: context_->synchronize();
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        OperationType getOperationType() const override
        {
            return OperationType::RopeOp;
        }

        std::string getName() const override
        {
            return "Cuda::RopeOp";
        }

        /**
         * @brief Cos/sin cache bytes needed for this configuration.
         *
         * CAUTION -- this is NOT per-instance cost. The caches live in the process-wide
         * RopeCacheRegistry keyed on (theta, built sequence length, head_dim), so across a
         * 48-layer model only the first op to acquire a given key allocates; the rest alias
         * it and report zero from getStateMemorySize(). This returns what making the cache
         * exist costs, once, for the sequence length the context carries.
         *
         * The consequence is that a caller summing this over every layer overcounts by
         * (layers - 1) caches. Deduplication belongs to the transformer, which knows the
         * distinct key set from its config -- the same shape as the tied-weight correction.
         * Registry state cannot be consulted here instead: before any build, nothing is
         * cached, so every layer would answer "I own it".
         */
        std::size_t getRequiredStateMemorySize( const BuildContext& build_context ) const override
        {
            return tableBytes( build_context.inputShape()[ 1 ] ) * 2; // cos and sin caches
        }

        std::size_t getStateMemorySize() const override
        {
            if ( !owns_cache_ )
                return 0;

            return tableBytes( seq_length_ ) * 2; // cos and sin caches
        }

    private:
        
        RopeConfig config_;
        CudaExecutionContext* context_;

        bool owns_cache_{ false };

        float* cos_cache_{ nullptr };
        float* sin_cache_{ nullptr };

        CacheKey cache_key_{};
        int batch_size_{ 0 };
        int seq_length_{ 0 };

        void dispatchForward(
            const ITensor& Q_in, const ITensor& K_in,
            ITensor& Q_out, ITensor& K_out,
            int B, int T, int position_offset ) const
        {
            Detail::cuda_rope_impl<ComputeType>::forward(
                static_cast<ComputeType*>(Q_out.rawData()),
                static_cast<ComputeType*>(K_out.rawData()),
                static_cast<const ComputeType*>(Q_in.rawData()),
                static_cast<const ComputeType*>(K_in.rawData()),
                cos_cache_, sin_cache_,
                B, T,
                static_cast<int>(config_.getNumHeads()),
                static_cast<int>(config_.getNumKVHeads()),
                static_cast<int>(config_.getHeadDim()),
                static_cast<int>(config_.getRotaryDim()),
                rotaryLayoutCode(),
                position_offset,
                context_->getStream() );

            // DEBUG: context_->synchronize();
        }

        /**
         * @brief The layout as the kernels take it: 0 = WholeHead, 1 = RotaryPrefix.
         *
         * RotaryPrefix writes only the leading rotary_dim channels, so the pass-through tail
         * is whatever the OUTPUT buffer already held. Every consumer today rotates in place
         * (QwenAttentionBlock takes a view over the q_norm output), which makes that the
         * input's own values. An out-of-place call would leave the tail unwritten -- checked
         * below rather than left to surface as garbage.
         */
        int rotaryLayoutCode() const noexcept
        {
            return config_.getRotaryLayout() == RotaryLayout::RotaryPrefix ? 1 : 0;
        }

        void requireInPlaceForPrefixLayout(
            const void* in_ptr, const void* out_ptr, const char* caller ) const
        {
            if ( rotaryLayoutCode() == 1 && in_ptr != out_ptr )
            {
                throw std::runtime_error( std::format(
                    "CudaRopeOp::{}: RotaryLayout::RotaryPrefix rotates only the leading "
                    "{} of {} channels and does not copy the remainder, so it requires an "
                    "in-place call; got distinct input and output buffers.",
                    caller, config_.getRotaryDim(), config_.getHeadDim() ) );
            }
        }

        /// Bytes of ONE of the cos or sin tables at the given row count.
        std::size_t tableBytes( dim_t table_rows ) const noexcept
        {
            return static_cast<std::size_t>( table_rows * (config_.getHeadDim() / 2) ) * sizeof( float );
        }

        CacheKey makeCacheKey( dim_t table_rows ) const noexcept
        {
            // Precision is FP32 regardless of TPrecision: the cache is always
            // float. This allows BF16 and FP32 ops with identical configs to
            // share one registry entry.
            return {
                context_->getDeviceId().index,
                table_rows,
                config_.getHeadDim(),
                config_.getRotaryDim(),
                rotaryLayoutCode(),
                config_.getBase(),
                TensorDataType::FP32
            };
        }

        void releaseCache() noexcept
        {
            if ( this->is_built_ )
            {
                RopeCacheRegistry::instance().release( cache_key_ );
                cos_cache_ = nullptr;
                sin_cache_ = nullptr;
                
                this->is_built_ = false;
            }
        }

        void ensureBuilt() const
        {
            if ( !this->is_built_ )
                throw std::runtime_error( "CudaRopeOp: build() must be called before forward/backward/prefill/decode." );
        }

        void validateRuntimeShape( int B, int T ) const
        {
            if ( B > batch_size_ || T > seq_length_ )
                throw std::runtime_error( std::format(
                    "CudaRopeOp: runtime shape [{}, {}] exceeds built max [{}, {}]",
                    B, T, batch_size_, seq_length_ ) );
        }
    };

}
