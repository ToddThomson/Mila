/**
 * @file CudaRopeOp.ixx
 * @brief CUDA implementation of the Rope (rotary positional embedding) operation.
 *
 * Applies RoPE to projected Q and K tensors in preparation for GQA attention.
 * Supports full-sequence forward/backward passes and a position-aware single-token
 * decode pass via IPositionalDecode.
 *
 * Data flow:
 *   wte lookup → [B,T,C] → linear projections
 *       → Q [B,T,n_heads,head_dim], K [B,T,n_kv_heads,head_dim]
 *       → CudaRopeOp
 *       → rotated Q, rotated K
 *       → CudaGqaOp
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

import Dnn.Components.Rope;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Compute.Precision;
import Compute.PairedOperation;
import Compute.IPositionalDecode;
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
     *   shapes; forward(), backward(), and decode() are pure hot-path dispatch.
     * - GQA-aware: Q and K may have different head counts (n_heads vs n_kv_heads).
     * - Implements IPositionalDecode for KV-cache autoregressive generation.
     * - Backward is exact: RoPE is an orthogonal rotation, so the gradient is the
     *   inverse rotation (negate sin terms). No extra buffers needed.
     *
     * Input/output shapes:
     *   Q:  [B, T, n_heads,    head_dim]
     *   K:  [B, T, n_kv_heads, head_dim]
     *   Q', K' — same shapes as inputs.
     *
     * Decode shapes (T=1, explicit position):
     *   Q:  [B, 1, n_heads,    head_dim]
     *   K:  [B, 1, n_kv_heads, head_dim]
     *
     * @tparam TPrecision Precision of Q/K tensors (FP32 or FP16).
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaRopeOp : public PairedOperation<DeviceType::Cuda, TPrecision>, public IPositionalDecode
    {
    public:

        using MR                  = CudaDeviceMemoryResource;
        using PairedOperationBase = PairedOperation<DeviceType::Cuda, TPrecision>;
        using TensorType          = Tensor<TPrecision, MR>;
        using NativeType          = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ConfigType          = RopeConfig;
        using CacheKey            = RopeCacheRegistry::CacheKey;

        CudaRopeOp( IExecutionContext* context, const RopeConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaRopeOp" ) ), config_( config )
        {
            config_.validate();
        }

        ~CudaRopeOp()
        {
            releaseCache();
        }

        // Disable copy: cos_cache_ / sin_cache_ are non-owning views into the shared
        // registry — copying would alias without incrementing the reference count.
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
            , built_( other.built_ )
        {
            other.cos_cache_ = nullptr;
            other.sin_cache_ = nullptr;
            other.built_     = false;
        }

        CudaRopeOp& operator=( CudaRopeOp&& other ) noexcept
        {
            if ( this != &other )
            {
                releaseCache();
                context_    = other.context_;
                config_     = std::move( other.config_ );
                cos_cache_  = other.cos_cache_;
                sin_cache_  = other.sin_cache_;
                cache_key_  = other.cache_key_;
                batch_size_ = other.batch_size_;
                seq_length_ = other.seq_length_;
                built_      = other.built_;

                other.cos_cache_ = nullptr;
                other.sin_cache_ = nullptr;
                other.built_     = false;
            }

            return *this;
        }

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /**
         * @brief Prepare the operation for a concrete input shape (cold path).
         *
         * On the first call, acquires a shared cos/sin cache from RopeCacheRegistry
         * and fills it if this is the first op with this configuration. Subsequent
         * calls on the same instance update the runtime shape limits only; the
         * shared cache is not re-acquired.
         *
         * @param input_shape  Shape of the Q input tensor [B, T, n_heads, head_dim].
         *
         * @throws std::invalid_argument if input_shape is not rank-4 or incompatible
         *                               with the configuration.
         * @throws CudaError if device memory allocation fails on first acquisition.
         */
        void build( const shape_t& input_shape )
        {
            validateInputShape( input_shape );

            batch_size_ = static_cast<int>(input_shape[ 0 ]);
            seq_length_ = static_cast<int>(input_shape[ 1 ]);

            if ( built_ )
                return;

            const std::size_t cache_bytes =
                config_.getMaxSequenceLength() * (config_.getHeadDim() / 2) * sizeof( NativeType );

            cache_key_ = makeCacheKey();

            auto [cos_ptr, sin_ptr, is_new] =
                RopeCacheRegistry::instance().acquire( cache_key_, cache_bytes );

            cos_cache_ = static_cast<NativeType*>( cos_ptr );
            sin_cache_ = static_cast<NativeType*>( sin_ptr );
            built_     = true;

            if ( is_new )
            {
                Detail::cuda_rope_impl<NativeType>::build_cache(
                    cos_cache_, sin_cache_,
                    static_cast<int>(config_.getMaxSequenceLength()),
                    static_cast<int>(config_.getHeadDim()),
                    config_.getBase(),
                    context_->getStream() );
            }
        }

        // ====================================================================
        // Forward
        // ====================================================================

        /**
         * @brief Full-sequence forward pass (hot path).
         *
         * Applies RoPE to Q and K across the full sequence. Both Q and K are
         * rotated in a single dispatch.
         *
         * @param Q_in   Input Q  [B, T, n_heads,    head_dim].
         * @param K_in   Input K  [B, T, n_kv_heads, head_dim].
         * @param Q_out  Output Q [B, T, n_heads,    head_dim].
         * @param K_out  Output K [B, T, n_kv_heads, head_dim].
         *
         * @throws std::runtime_error if build() has not been called or if B/T
         *                            exceed the built maximum.
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

            Detail::cuda_rope_impl<NativeType>::forward(
                static_cast<NativeType*>(Q_out.rawData()),
                static_cast<NativeType*>(K_out.rawData()),
                static_cast<const NativeType*>(Q_in.rawData()),
                static_cast<const NativeType*>(K_in.rawData()),
                cos_cache_, sin_cache_,
                B, T,
                static_cast<int>(config_.getNumHeads()),
                static_cast<int>(config_.getNumKvHeads()),
                static_cast<int>(config_.getHeadDim()),
                context_->getStream() );
        }

        // ====================================================================
        // Backward
        // ====================================================================

        /**
         * @brief Backward pass (hot path).
         *
         * RoPE is an orthogonal rotation (R^T R = I), so the Jacobian is R^T.
         * The backward pass is therefore the inverse rotation: rotate the upstream
         * gradients by -θ (negate sin terms). No new parameters are accumulated.
         *
         * @param dQ_out  Upstream gradient for Q  [B, T, n_heads,    head_dim].
         * @param dK_out  Upstream gradient for K  [B, T, n_kv_heads, head_dim].
         * @param dQ_in   Output gradient w.r.t. Q input  [B, T, n_heads,    head_dim].
         * @param dK_in   Output gradient w.r.t. K input  [B, T, n_kv_heads, head_dim].
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
                static_cast<int>(config_.getNumKvHeads()),
                static_cast<int>(config_.getHeadDim()),
                context_->getStream() );
        }

        // ====================================================================
        // Decode (IPositionalDecode)
        // ====================================================================

        /**
         * @brief IPositionalDecode compliance overload — use the typed decode overload instead.
         *
         * @see decode( Q_in, K_in, Q_out, K_out, position )
         */
        void decode( const ITensor& input, ITensor& output, int position ) const override
        {
            throw std::logic_error(
                "CudaRopeOp::decode(ITensor) — use the typed decode(Q_in, K_in, Q_out, K_out, position) overload." );
        }

        /**
         * @brief Typed single-token decode with explicit Q/K tensors.
         *
         * Reads only the cache row at `position`. Used for KV-cache autoregressive
         * generation where T=1.
         *
         * @param Q_in    Input Q  [B, 1, n_heads,    head_dim].
         * @param K_in    Input K  [B, 1, n_kv_heads, head_dim].
         * @param Q_out   Output Q [B, 1, n_heads,    head_dim].
         * @param K_out   Output K [B, 1, n_kv_heads, head_dim].
         * @param position Absolute sequence position for the cache row.
         *
         * @throws std::invalid_argument if position is out of range.
         * @throws std::runtime_error if build() has not been called.
         */
        void decode(
            const ITensor& Q_in, const ITensor& K_in,
            ITensor& Q_out, ITensor& K_out,
            int position ) const
        {
            ensureBuilt();

            if ( position < 0 || static_cast<size_t>(position) >= config_.getMaxSequenceLength() )
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
                static_cast<int>(config_.getNumKvHeads()),
                static_cast<int>(config_.getHeadDim()),
                context_->getStream() );
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        OperationType getOperationType() const
        {
            return OperationType::RopeOp;
        }

        std::string getName() const
        {
            return "Cuda::RopeOp";
        }

    private:
        RopeConfig            config_;
        CudaExecutionContext* context_;

        // Non-owning views into the shared RopeCacheRegistry entry.
        NativeType* cos_cache_{ nullptr };
        NativeType* sin_cache_{ nullptr };

        CacheKey cache_key_{};
        int      batch_size_{ 0 };
        int      seq_length_{ 0 };
        bool     built_{ false };

        // ====================================================================
        // Internal helpers
        // ====================================================================

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
            if ( built_ )
            {
                RopeCacheRegistry::instance().release( cache_key_ );
                cos_cache_ = nullptr;
                sin_cache_ = nullptr;
                built_     = false;
            }
        }

        void ensureBuilt() const
        {
            if ( !built_ )
                throw std::runtime_error( "CudaRopeOp: build() must be called before forward/backward/decode." );
        }

        void validateInputShape( const shape_t& shape ) const
        {
            if ( shape.size() != 4 )
                throw std::invalid_argument(
                    "CudaRopeOp: Q input must be rank-4 [B, T, n_heads, head_dim]" );

            if ( shape[ 1 ] > config_.getMaxSequenceLength() )
                throw std::invalid_argument( std::format(
                    "CudaRopeOp: sequence length {} exceeds max_seq_len {}",
                    shape[ 1 ], config_.getMaxSequenceLength() ) );

            if ( shape[ 2 ] != config_.getNumHeads() )
                throw std::invalid_argument( std::format(
                    "CudaRopeOp: n_heads mismatch: tensor has {}, config expects {}",
                    shape[ 2 ], config_.getNumHeads() ) );

            if ( shape[ 3 ] != config_.getHeadDim() )
                throw std::invalid_argument( std::format(
                    "CudaRopeOp: head_dim mismatch: tensor has {}, config expects {}",
                    shape[ 3 ], config_.getHeadDim() ) );
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