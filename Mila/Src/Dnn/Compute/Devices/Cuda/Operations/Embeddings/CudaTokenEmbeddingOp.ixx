/**
 * @file CudaTokenEmbeddingOp.ixx
 * @brief CUDA implementation of the TokenEmbedding operation.
 *
 * Pure vocabulary lookup: output[b,t,:] = wte[X[b,t],:].
 * No positional information. Positional encoding is handled downstream
 * by a dedicated encoding component (RoPE, ALiBi, or Learned).
 *
 * @tparam TInput     Data type of token index input (INT32).
 * @tparam TPrecision Precision of embedding output (FP32 or FP16).
 */

module;
#include <cuda_fp16.h>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <format>

export module Compute.CudaTokenEmbeddingOp;
import :Dispatch;

import Dnn.Components.TokenEmbedding;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Compute.Precision;
import Compute.UnaryOperation;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.OperationType;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.OperationRegistrarHelpers;

namespace Mila::Dnn::Compute::Cuda::TokenEmbedding
{
    using namespace Mila::Dnn;

    export template<TensorDataType TInput, TensorDataType TPrecision = TInput>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaTokenEmbeddingOp
        : public UnaryOperation<DeviceType::Cuda, TInput, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TInput, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ConfigType = TokenEmbeddingConfig;

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
         * @param wte Token embedding table — CUDA tensor of shape [vocab_size, C].
         *
         * @throws std::invalid_argument on null, non-CUDA, or shape-mismatched tensor.
         */
        void setParameters( ITensor* wte )
        {
            if ( !wte )
                throw std::invalid_argument( "CudaTokenEmbeddingOp::setParameters - wte is required" );

            if ( wte->getDeviceType() != DeviceType::Cuda )
                throw std::invalid_argument( "CudaTokenEmbeddingOp::setParameters - wte must be a CUDA tensor" );

            const auto& shape = wte->shape();

            if ( shape.size() != 2 )
                throw std::invalid_argument( "CudaTokenEmbeddingOp::setParameters - wte must be a 2D tensor [vocab_size, C]" );

            if ( static_cast<int>(shape[ 0 ]) != config_.vocab_size )
                throw std::invalid_argument( std::format(
                    "CudaTokenEmbeddingOp::setParameters - wte vocab_size {} does not match config {}",
                    shape[ 0 ], config_.vocab_size ) );

            if ( static_cast<int>(shape[ 1 ]) != config_.embedding_dim )
                throw std::invalid_argument( std::format(
                    "CudaTokenEmbeddingOp::setParameters - wte embedding_dim {} does not match config {}",
                    shape[ 1 ], config_.embedding_dim ) );

            wte_ = static_cast<NativeType*>(wte->rawData());
            vocab_size_ = static_cast<int>(shape[ 0 ]);
            embedding_dim_ = static_cast<int>(shape[ 1 ]);
        }

        /**
         * @brief Bind the wte gradient tensor for training (module retains ownership).
         *
         * @param wte_grad Gradient buffer for wte — CUDA tensor of shape [vocab_size, C].
         *
         * @throws std::invalid_argument on null or non-CUDA tensor.
         */
        void setGradients( ITensor* wte_grad )
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
        void build( const shape_t& input_shape ) override
        {
            if ( !wte_ )
                throw std::runtime_error( "CudaTokenEmbeddingOp::build requires wte bound via setParameters() before build()." );

            validateInputShape( input_shape );

            batch_size_ = static_cast<int>(input_shape[ 0 ]);
            seq_length_ = static_cast<int>(input_shape[ 1 ]);

            UnaryOperationBase::build( input_shape );
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
        void forward( const ITensor& input, ITensor& output ) const override
        {
            const auto& shape = input.shape();
            int B = static_cast<int>(shape[ 0 ]);
            int T = static_cast<int>(shape[ 1 ]);

            validateRuntimeShape( B, T );

            const int32_t* X = static_cast<const int32_t*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            Detail::cuda_token_embedding_impl<NativeType>::forward(
                Y, X, wte_, B, T, embedding_dim_, context_->getStream() );
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
            ITensor& input_grad ) const override
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

        // ====================================================================
        // Decode
        // ====================================================================

        /**
         * @brief Single-token decode pass (hot path).
         *
         * Computes output[b,:] = wte[X[b,0],:] for each batch element.
         * No position argument — positional encoding is handled downstream.
         *
         * @param input  Single-token indices [B, 1] (INT32).
         * @param output Pre-allocated output buffer [B, C].
         */
        void decode( const ITensor& input, ITensor& output ) const
        {
            int B = static_cast<int>(input.shape()[ 0 ]);

            const int32_t* X = static_cast<const int32_t*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            Detail::cuda_token_embedding_impl<NativeType>::decode(
                Y, X, wte_, B, embedding_dim_, context_->getStream() );
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

        NativeType* wte_{ nullptr };
        NativeType* wte_grad_{ nullptr };

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

    export class CudaTokenEmbeddingOpRegistrar
    {
    public:
        static void registerOperations()
        {
            registerUnaryOpType<DeviceType::Cuda,
                CudaTokenEmbeddingOp<TensorDataType::INT32, TensorDataType::FP32>,
                TensorDataType::INT32, TensorDataType::FP32>( "TokenEmbeddingOp" );

            registerUnaryOpType<DeviceType::Cuda,
                CudaTokenEmbeddingOp<TensorDataType::INT32, TensorDataType::FP16>,
                TensorDataType::INT32, TensorDataType::FP16>( "TokenEmbeddingOp" );
        }
    };
}