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

import Dnn.Component;
import Dnn.Components.RopeConfig;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.OperationBase;
import Compute.DeviceAllocation;
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
     * - No learned parameters and no state. Each cos and sin is calculated where it is used,
     *   once per (token, pair) and shared across the token's Q and K heads (Rope.Rotation.cuh),
     *   in one launch for Q and K together.
     * - build() records the shape: the built sequence length bounds every position prefill and
     *   decode may rotate.
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
     * @tparam TComputePrecision Precision of Q/K tensors (FP32 or BF16).
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

        CudaRopeOp( IExecutionContext* context, const RopeConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaRopeOp" ) ), config_( config )
        {
            config_.validate();

            const RopeFrequencyScaling& scaling = config_.getFrequencyScaling();

            angles_ = Detail::angle_parameters(
                static_cast<int>( config_.getHeadDim() ), static_cast<int>( config_.getRotaryDim() ),
                rotaryLayoutCode(), config_.getBase(),
                scaling.factor, scaling.low_frequency_factor, scaling.high_frequency_factor,
                narrowToKernelIndex( scaling.original_context_length ) );
        }

        /**
         * @brief Prepare the operation for a concrete input shape (cold path).
         *
         * The sequence length T of the build context is the number of positions this op
         * can ever rotate: prefill and decode refuse any position at or past T. A caller
         * that decodes must therefore build at the full context length, not at a prefill
         * chunk.
         *
         * @param build_context  Build context carrying the Q/K input shape [B, T, ...].
         * @throws std::invalid_argument if T exceeds the trained maximum sequence length.
         */
        void build( const BuildContext& build_context ) override
        {
            const auto& shape = build_context.inputShape();
            const dim_t sequence_length = shape[ 1 ];

            if ( sequence_length > config_.getMaxSequenceLength() )
                throw std::invalid_argument( std::format(
                    "CudaRopeOp::build: sequence length {} exceeds the trained maximum {}",
                    sequence_length, config_.getMaxSequenceLength() ) );

            batch_size_ = static_cast<int>(shape[ 0 ]);
            seq_length_ = static_cast<int>(sequence_length);

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
                angles_,
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
         * Applies RoPE to Q and K at absolute positions
         * [position_offset .. position_offset + T - 1].
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
         * Rotates at the context's decode position, which the kernel reads on the device;
         * `position` is the same value, checked here on the host (DecodeGraph.md section 4.1).
         * Used for KV-cache autoregressive generation where T=1.
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
                angles_,
                B, context_->getDecodePosition(),
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

    private:

        CudaExecutionContext* context_;
        RopeConfig config_;

        Detail::AngleParameters angles_{};
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
                angles_,
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
