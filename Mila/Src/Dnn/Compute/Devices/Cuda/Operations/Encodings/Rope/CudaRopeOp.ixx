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

export module Compute.CudaRopeOp;
import :Dispatch;
import :Cache;

import Dnn.Component;
import Dnn.Components.Rope;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Compute.Precision;
import Compute.PairedOperation;
import Compute.IPositionalPairedOp;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.OperationType;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.OperationRegistrarHelpers;

namespace Mila::Dnn::Compute::Cuda::Rope
{
    using namespace Mila::Dnn;

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
     * - No learned parameters. The cos/sin cache is computed from fixed frequencies
     *   on the first build() call and shared across all ops with identical parameters
     *   via RopeCacheRegistry. Subsequent ops with the same config reuse the existing
     *   device allocation; build_cache() is called exactly once per unique config.
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
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaRopeOp : public PairedOperation<DeviceType::Cuda, TPrecision>, public IPositionalPairedOp
    {
    public:

        using MR = CudaDeviceMemoryResource;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
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
            , sin_cache_( other.sin_cache_ )
            , cache_key_( other.cache_key_ )
            , batch_size_( other.batch_size_ )
            , seq_length_( other.seq_length_ )
        {
            this->is_built_ = other.is_built_;
            other.cos_cache_ = nullptr;
            other.sin_cache_ = nullptr;
            other.is_built_ = false;
        }

        CudaRopeOp& operator=( CudaRopeOp&& other ) noexcept
        {
            if ( this != &other )
            {
                releaseCache();
                context_ = other.context_;
                config_ = std::move( other.config_ );
                cos_cache_ = other.cos_cache_;
                sin_cache_ = other.sin_cache_;
                cache_key_ = other.cache_key_;
                batch_size_ = other.batch_size_;
                seq_length_ = other.seq_length_;
                this->is_built_ = other.is_built_;

                other.cos_cache_ = nullptr;
                other.sin_cache_ = nullptr;
                other.is_built_ = false;
            }

            return *this;
        }

        /**
         * @brief Prepare the operation for a concrete input shape (cold path).
         *
         * On the first call, acquires a shared cos/sin cache from RopeCacheRegistry
         * and fills it if this is the first op with this configuration. Subsequent
         * calls on the same instance update the runtime shape limits only; the
         * shared cache is not re-acquired.
         *
         * @param build_context  Build context carrying the Q/K input shape [B, T, ...].
         */
        void build( const BuildContext& build_context ) override
        {
            const auto& shape = build_context.inputShape();
            batch_size_ = static_cast<int>(shape[ 0 ]);
            seq_length_ = static_cast<int>(shape[ 1 ]);

            if ( this->is_built_ )
                return;

            const std::size_t cache_bytes =
                config_.getMaxSequenceLength() * (config_.getHeadDim() / 2) * sizeof( NativeType );

            cache_key_ = makeCacheKey();

            auto [cos_ptr, sin_ptr, is_new] =
                RopeCacheRegistry::instance().acquire( cache_key_, cache_bytes );

            cos_cache_ = static_cast<NativeType*>(cos_ptr);
            sin_cache_ = static_cast<NativeType*>(sin_ptr);
            this->is_built_ = true;

            if ( is_new )
            {
                Detail::cuda_rope_impl<NativeType>::build_cache(
                    cos_cache_, sin_cache_,
                    static_cast<int>(config_.getMaxSequenceLength()),
                    static_cast<int>(config_.getHeadDim()),
                    config_.getBase(),
                    context_->getStream() );

                context_->synchronize(); // Ensure cache is ready before any op can use it.
            }
        }

        /**
         * @brief Full-sequence forward pass.
         *
         * Applies RoPE to Q and K across the full sequence with position_offset = 0.
         * Used for training forward passes.
         */
        void forward(
            const ITensor& Q_in, const ITensor& K_in,
            ITensor& Q_out, ITensor& K_out ) const override
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
            ITensor& dQ_in, ITensor& dK_in ) const override
        {
            ensureBuilt();

            const auto& q_shape = dQ_out.shape();
            int B = static_cast<int>(q_shape[ 0 ]);
            int T = static_cast<int>(q_shape[ 1 ]);

            validateRuntimeShape( B, T );

            Detail::cuda_rope_impl<NativeType>::backward(
                static_cast<NativeType*>(dQ_in.rawData()),
                static_cast<NativeType*>(dK_in.rawData()),
                static_cast<const NativeType*>(dQ_out.rawData()),
                static_cast<const NativeType*>(dK_out.rawData()),
                cos_cache_, sin_cache_,
                B, T,
                static_cast<int>(config_.getNumHeads()),
                static_cast<int>(config_.getNumKVHeads()),
                static_cast<int>(config_.getHeadDim()),
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
            int position_offset ) override
        {
            ensureBuilt();

            const auto& q_shape = Q_in.shape();
            int B = static_cast<int>(q_shape[ 0 ]);
            int T = static_cast<int>(q_shape[ 1 ]);

            if ( position_offset < 0 ||
                static_cast<size_t>( position_offset + T ) > config_.getMaxSequenceLength() )
                throw std::invalid_argument( std::format(
                    "CudaRopeOp::prefill: position_offset {} + T {} exceeds max_seq_len {}",
                    position_offset, T, config_.getMaxSequenceLength() ) );

            dispatchForward( Q_in, K_in, Q_out, K_out, B, T, position_offset );
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
            int position ) override
        {
            ensureBuilt();

            if ( position < 0 || static_cast<size_t>( position ) >= config_.getMaxSequenceLength() )
                throw std::invalid_argument( std::format(
                    "CudaRopeOp::decode: position {} out of range [0, {})",
                    position, config_.getMaxSequenceLength() ) );

            int B = static_cast<int>( Q_in.shape()[ 0 ] );

            Detail::cuda_rope_impl<NativeType>::decode(
                static_cast<NativeType*>( Q_out.rawData() ),
                static_cast<NativeType*>( K_out.rawData() ),
                static_cast<const NativeType*>( Q_in.rawData() ),
                static_cast<const NativeType*>( K_in.rawData() ),
                cos_cache_, sin_cache_,
                B, position,
                static_cast<int>(config_.getNumHeads()),
                static_cast<int>(config_.getNumKVHeads()),
                static_cast<int>(config_.getHeadDim()),
                context_->getStream() );
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

    private:
        RopeConfig config_;
        CudaExecutionContext* context_;

        NativeType* cos_cache_{ nullptr };
        NativeType* sin_cache_{ nullptr };

        CacheKey cache_key_{};
        int batch_size_{ 0 };
        int seq_length_{ 0 };

        void dispatchForward(
            const ITensor& Q_in, const ITensor& K_in,
            ITensor& Q_out, ITensor& K_out,
            int B, int T, int position_offset ) const
        {
            Detail::cuda_rope_impl<NativeType>::forward(
                static_cast<NativeType*>(Q_out.rawData()),
                static_cast<NativeType*>(K_out.rawData()),
                static_cast<const NativeType*>(Q_in.rawData()),
                static_cast<const NativeType*>(K_in.rawData()),
                cos_cache_, sin_cache_,
                B, T,
                static_cast<int>(config_.getNumHeads()),
                static_cast<int>(config_.getNumKVHeads()),
                static_cast<int>(config_.getHeadDim()),
                position_offset,
                context_->getStream() );

            context_->synchronize();
        }

        CacheKey makeCacheKey() const noexcept
        {
            return {
                context_->getDeviceId().index,
                config_.getMaxSequenceLength(),
                config_.getHeadDim(),
                config_.getBase(),
                TPrecision
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

    // ========================================================================
    // Registrar
    // ========================================================================

    export class CudaRopeOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "RopeOp";

            registerPairedOpType<DeviceType::Cuda, CudaRopeOp<TensorDataType::FP32>, TensorDataType::FP32>( opName );

            //registerPairedOpType<DeviceType::Cuda, CudaRopeOp<TensorDataType::FP16>, TensorDataType::FP16>( opName );
        }
    };
}