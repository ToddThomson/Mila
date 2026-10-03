/**
 * @file CudaTokenEmbeddingOp.ixx
 * @brief CUDA implementation of the TokenEmbedding operation.
 *
 * Pure vocabulary lookup: output[b,t,:] = wte[X[b,t],:].
 * No positional information. Positional encoding is handled downstream
 * by a dedicated encoding component (RoPE, ALiBi, or Learned).
 *
 * TTableQuantization = PerChannelFp8<> stores the table as FP8_E4M3 with one
 * float32 absmax scale per vocabulary row, PerGroupInt6<32> as six-bit codes with
 * one IEEE half scale per 32 elements of a row; both dequantize inline during the
 * gather (D4 Design B). quantize() and setTableScales() are only callable on
 * quantized instantiations; the quantized path is inference-only.
 *
 * @tparam TInput              Data type of token index input (INT32).
 * @tparam TPrecision          Precision of embedding output (FP32 or BF16).
 * @tparam TTableQuantization  Table quantization policy (NoWeightQuant, PerChannelFp8<> or PerGroupInt6<32>).
 */

module;
#include <cuda_fp16.h>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <format>
#include <sstream>

export module Compute.CudaTokenEmbeddingOp;
import :Dispatch;
import :Quantize;

import Dnn.Components.TokenEmbeddingConfig;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Quantization.Weight.Policies;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.OperationType;
import Dnn.Component;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Serialization.Tensor;

// DEBUG:
import Cuda.Debug;

namespace Mila::Dnn::Compute::Cuda::TokenEmbedding
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Serialization;

    export template<TensorDataType TInput, TensorDataType TPrecision = TInput,
        WeightQuantPolicy TTableQuantization = NoWeightQuant>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaTokenEmbeddingOp : public Operation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using OperationBaseType = Operation<DeviceType::Cuda, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ConfigType = TokenEmbeddingConfig;

        static constexpr bool kIsQuantized = TTableQuantization::kIsQuantized;
        static constexpr bool kIsInt6Table = HasInt6Codes<TTableQuantization>;

        static_assert( !kIsQuantized || TTableQuantization::kPerChannel || kIsInt6Table,
            "CudaTokenEmbeddingOp: a quantized table is FP8 per vocabulary row or INT6 per 32 elements of a row" );

        static_assert( !kIsQuantized || TPrecision == TensorDataType::BF16,
            "CudaTokenEmbeddingOp: the quantized table gather-dequant paths are BF16-only" );

        static constexpr TensorDataType kTableDtype = kIsQuantized
            ? TTableQuantization::kStorageDtype : TPrecision;

        using TableNativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<kTableDtype>::device_type;

        // FP32 per row for FP8, IEEE half per group for INT6. Unused when unquantized.
        using TableScaleNativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TTableQuantization::kScaleDtype>::device_type;

        /// Bytes of one stored table row of embedding_dim elements.
        static constexpr int64_t storedRowElements( int64_t embedding_dim ) noexcept
        {
            if constexpr ( kIsInt6Table )
                return embedding_dim * TTableQuantization::kStorageBitsPerElement / 8;
            else
                return embedding_dim;
        }

        CudaTokenEmbeddingOp( IExecutionContext* context, const TokenEmbeddingConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaTokenEmbeddingOp" ) ),
            config_( config )
        {
            config_.validate();
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

        /**
         * @brief Bind the wte parameter tensor (module retains ownership).
         *
         * @param wte Token embedding table -- CUDA tensor of shape [vocab_size, C], or [vocab_size, 3 * C / 4]
         *            bytes for an INT6 table.
         *
         * @throws std::invalid_argument on null, non-CUDA, or shape-mismatched tensor.
         */
        void setParameters( ITensor* wte, ITensor* ) override
        {
            if ( !wte )
                throw std::invalid_argument( "CudaTokenEmbeddingOp::setParameters - wte is required" );

            if ( wte->getDeviceType() != DeviceType::Cuda )
                throw std::invalid_argument( "CudaTokenEmbeddingOp::setParameters - wte must be a CUDA tensor" );

            const auto& shape = wte->shape();

            if ( shape.size() != 2 )
                throw std::invalid_argument( "CudaTokenEmbeddingOp::setParameters - wte must be a 2D tensor [vocab_size, C]" );

            if ( shape[ 0 ] != config_.getVocabSize() )
                throw std::invalid_argument( std::format(
                    "CudaTokenEmbeddingOp::setParameters - wte vocab_size {} does not match config {}",
                    shape[ 0 ], config_.getVocabSize() ) );

            if ( shape[ 1 ] != storedRowElements( config_.getEmbeddingDim() ) )
                throw std::invalid_argument( std::format(
                    "CudaTokenEmbeddingOp::setParameters - wte row extent {} does not match {} for embedding_dim {}",
                    shape[ 1 ], storedRowElements( config_.getEmbeddingDim() ), config_.getEmbeddingDim() ) );

            if constexpr ( kIsInt6Table )
            {
                // The gather's 4-byte low-nibble loads and 16-byte stores, and the head's matvec, need it.
                if ( config_.getEmbeddingDim() % 64 != 0 )
                    throw std::invalid_argument( std::format(
                        "CudaTokenEmbeddingOp::setParameters - an INT6 table needs embedding_dim a multiple of 64, got {}",
                        config_.getEmbeddingDim() ) );
            }

            wte_ = static_cast<TableNativeType*>(wte->rawData());
            vocab_size_ = static_cast<int>(shape[ 0 ]);
            embedding_dim_ = static_cast<int>(config_.getEmbeddingDim());
        }

        /**
         * @brief Bind the table scale tensor (module retains ownership).
         *
         * Must be bound before build(). quantize() fills the allocation at load time.
         *
         * @param scales Device tensor: FP32 [vocab_size] for an FP8 table, FP16 [vocab_size, C / 32] for INT6.
         */
        void setTableScales( ITensor* scales ) requires kIsQuantized
        {
            if ( !scales )
                throw std::invalid_argument( "CudaTokenEmbeddingOp::setTableScales - scales tensor is required" );

            if ( scales->getDeviceType() != DeviceType::Cuda )
                throw std::invalid_argument( "CudaTokenEmbeddingOp::setTableScales - scales must be a CUDA tensor" );

            table_scales_ = static_cast<const TableScaleNativeType*>(scales->rawData());
        }

        /**
         * @brief Quantize a BF16 host table blob into the policy's storage format.
         *
         * Runs once at model load time. Delegates to the Detail:: entries pre-compiled by
         * NVCC in the :Quantize partition, which chunk the table over rows through the
         * load's staging buffer, under its limit; the full BF16 table is about 2 GB on the
         * 12B build, so staging it whole would set the load's peak. All device work is
         * issued on the execution context stream; the caller synchronizes after loading
         * (the BF16 source blob is uploaded asynchronously and never retained on device).
         *
         * @param blob           Host BF16 table blob from the model archive.
         * @param table_out      Device table tensor at the policy's storage extent.
         * @param scales_out     Device scale tensor at the policy's scale dtype and extent.
         * @param expected_shape Logical table shape [vocab_size, embedding_dim], for validation.
         */
        void quantize(
            const ITensorBlob& blob,
            ITensor& table_out,
            ITensor& scales_out,
            const shape_t& expected_shape ) requires kIsQuantized
        {
            const int64_t vocab_size = static_cast<int64_t>( expected_shape[ 0 ] );
            const int64_t embedding_dim = static_cast<int64_t>( expected_shape[ 1 ] );

            const size_t src_bytes = static_cast<size_t>( vocab_size * embedding_dim ) * sizeof( uint16_t );
            const size_t staging_bytes = std::min( src_bytes, context_->getLoadStagingLimitBytes() );

            void* staging = context_->getLoadStagingBuffer( staging_bytes );

            if constexpr ( kIsInt6Table )
            {
                Detail::quantize_table_int6_per_group( blob, table_out, scales_out, expected_shape,
                    TTableQuantization::kQuantizationGroupSize, staging, staging_bytes, context_->getStream() );
            }
            else
            {
                Detail::quantize_table_fp8_per_row( blob, table_out, scales_out, expected_shape,
                    staging, staging_bytes, context_->getStream() );
            }
        }

        /**
         * @brief Bind the wte gradient tensor for training (module retains ownership).
         *
         * @param wte_grad Gradient buffer for wte -- CUDA tensor of shape [vocab_size, C].
         *
         * @throws std::invalid_argument on null or non-CUDA tensor.
         */
        void setGradients( ITensor* wte_grad, ITensor* ) override
        {
            if ( !wte_grad )
                throw std::invalid_argument( "CudaTokenEmbeddingOp::setGradients - wte_grad is required" );

            if ( wte_grad->getDeviceType() != DeviceType::Cuda )
                throw std::invalid_argument( "CudaTokenEmbeddingOp::setGradients - wte_grad must be a CUDA tensor" );

            wte_grad_ = static_cast<NativeType*>(wte_grad->rawData());
        }

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /**
         * @brief Prepare the operation for a concrete input shape (cold path).
         *
         * @param input_shape Token index input shape [B, T].
         *
         * @throws std::runtime_error    if wte is not bound.
         * @throws std::invalid_argument if input shape is invalid.
         */
        void build( const BuildContext& config ) override
        {
            if ( !wte_ )
            {
                throw std::runtime_error( "CudaTokenEmbeddingOp::build requires wte bound via setParameters() before build()." );
            }

            if constexpr ( kIsQuantized )
            {
                if ( !table_scales_ )
                    throw std::runtime_error( "CudaTokenEmbeddingOp::build requires table scales bound via setTableScales() before build()." );
            }

            const auto& input_shape = config.inputShape();

            validateInputShape( input_shape );

            batch_size_ = static_cast<int>(input_shape[ 0 ]);
            seq_length_ = static_cast<int>(input_shape[ 1 ]);

            OperationBaseType::build( config );
        }

        // ====================================================================
        // Forward
        // ====================================================================

        /**
         * @brief Full-sequence forward pass (hot path).
         *
         * For each (b, t): output[b,t,:] = wte[X[b,t],:].
         *
         * @param input  Token indices [B, T] (INT32).
         * @param output Pre-allocated embeddings [B, T, C].
         */
        void forward( const ITensor& input, ITensor& output ) const
        {
            const auto& shape = input.shape();
            int B = static_cast<int>(shape[ 0 ]);
            int T = static_cast<int>(shape[ 1 ]);

            validateRuntimeShape( B, T );

            const int32_t* X = static_cast<const int32_t*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            if constexpr ( kIsInt6Table )
            {
                Detail::cuda_token_embedding_int6_impl::forward(
                    Y, X, wte_, table_scales_, B, T, embedding_dim_, context_->getStream() );
            }
            else if constexpr ( kIsQuantized )
            {
                Detail::cuda_token_embedding_fp8_impl::forward(
                    Y, X, wte_, table_scales_, B, T, embedding_dim_, context_->getStream() );
            }
            else
            {
                Detail::cuda_token_embedding_impl<NativeType>::forward(
                    Y, X, wte_, B, T, embedding_dim_, context_->getStream() );
            }

            // DEBUG: synchronize and print output stats
            // context_->synchronize();
            // print_stats( "emb.output", Y, output.shape(), 8, context_->getStream() );
        }

        // ====================================================================
        // Backward
        // ====================================================================

        /**
         * @brief Backward pass accumulating gradients into wte (hot path).
         *
         * Token indices are non-differentiable; input_grad is unused.
         *
         * @param input       Token indices used in forward [B, T] (INT32).
         * @param output_grad Upstream embedding gradient [B, T, C].
         * @param input_grad  Unused (non-differentiable input).
         */
        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& /*input_grad*/ ) const
        {
            if constexpr ( kIsQuantized )
            {
                throw std::logic_error(
                    "CudaTokenEmbeddingOp::backward - the quantized table path is inference-only" );
            }
            else
            {
                const auto& shape = input.shape();
                int B = static_cast<int>(shape[ 0 ]);
                int T = static_cast<int>(shape[ 1 ]);

                validateRuntimeShape( B, T );

                const int32_t* X = static_cast<const int32_t*>(input.rawData());
                const NativeType* dY = static_cast<const NativeType*>(output_grad.rawData());

                Detail::cuda_token_embedding_impl<NativeType>::backward(
                    wte_grad_, dY, X, B, T, embedding_dim_, context_->getStream() );
            }
        }

        // ====================================================================
        // Decode
        // ====================================================================

        /**
         * @brief Single-token decode pass (hot path).
         *
         * Computes output[b,:] = wte[X[b,0],:] for each batch element.
         * No position argument -- positional encoding is handled downstream.
         *
         * @param input  Single-token indices [B, 1] (INT32).
         * @param output Pre-allocated output buffer [B, C].
         */
        void decode( const ITensor& input, ITensor& output ) const
        {
            int B = static_cast<int>(input.shape()[ 0 ]);

            const int32_t* X = static_cast<const int32_t*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            if constexpr ( kIsInt6Table )
            {
                Detail::cuda_token_embedding_int6_impl::decode(
                    Y, X, wte_, table_scales_, B, embedding_dim_, context_->getStream() );
            }
            else if constexpr ( kIsQuantized )
            {
                Detail::cuda_token_embedding_fp8_impl::decode(
                    Y, X, wte_, table_scales_, B, embedding_dim_, context_->getStream() );
            }
            else
            {
                Detail::cuda_token_embedding_impl<NativeType>::decode(
                    Y, X, wte_, B, embedding_dim_, context_->getStream() );
            }
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        OperationType getOperationType() const override
        {
            return OperationType::TokenEmbeddingOp;
        }

        std::string getName() const override
        {
            return "Cuda::Embeddings::TokenEmbeddingOp";
        }

    private:
        TokenEmbeddingConfig  config_;
        CudaExecutionContext* context_;

        TableNativeType* wte_{ nullptr };
        NativeType* wte_grad_{ nullptr };

        // Dequantization scales at the policy's dtype. Non-null only when kIsQuantized.
        const TableScaleNativeType* table_scales_{ nullptr };

        int vocab_size_{ 0 };
        int embedding_dim_{ 0 };
        int batch_size_{ 0 };
        int seq_length_{ 0 };

        void validateInputShape( const shape_t& shape ) const
        {
            if ( shape.size() != 2 )
                throw std::invalid_argument( "CudaTokenEmbeddingOp: input must be rank-2 [B, T]" );
        }

        void validateRuntimeShape( int B, int T ) const
        {
            if ( B > batch_size_ || T > seq_length_ )
                throw std::runtime_error( std::format(
                    "CudaTokenEmbeddingOp: runtime shape [{}, {}] exceeds built max [{}, {}]",
                    B, T, batch_size_, seq_length_ ) );
        }
    };

}
