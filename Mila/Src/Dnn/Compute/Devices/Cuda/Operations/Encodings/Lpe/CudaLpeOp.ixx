/**
 * @file CudaLpeOp.ixx
 * @brief CUDA implementation of the Lpe (token + positional embedding) operation.
 *
 * Supports full-sequence forward/backward passes and a position-aware single-token
 * decode pass via IPositionalDecode.
 */

module;
#include <cuda_fp16.h>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <format>

export module Compute.CudaLpeOp;
import :Dispatch;

import Dnn.Components.Lpe;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Compute.Precision;
import Compute.IPositionalDecode;
import Compute.UnaryOperation;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.OperationType;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.OperationRegistrarHelpers;

namespace Mila::Dnn::Compute::Cuda::Lpe
{
    using namespace Mila::Dnn;

    /**
     * @brief CUDA implementation of the Lpe (token + positional embedding) operation.
     *
     * Combines a token embedding lookup (wte) with a positional embedding lookup (wpe)
     * on CUDA devices, supporting FP32 and FP16 precision.
     *
     * Design:
     * - Two-phase initialization: build() performs all setup once; forward(), backward(),
     *   and decode() are pure hot-path dispatch with no per-call overhead.
     * - Parameters (wte, wpe) are bound via setParameters() before build() and owned
     *   by the calling Lpe component.
     * - Token indices (INT32) are non-differentiable; no input gradient is produced.
     * - Implements IPositionalDecode so the owning Lpe component can call decode()
     *   with the correct absolute sequence position during KV-cache autoregressive
     *   generation, avoiding the wpe[0] bug that forward() with T=1 would produce.
     *
     * @tparam TInput     Data type of token index input (typically INT32).
     * @tparam TPrecision Precision of embedding output (FP32 or FP16).
     */
    export template<TensorDataType TInput, TensorDataType TPrecision = TInput>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaLpeOp : public UnaryOperation<DeviceType::Cuda, TInput, TPrecision>,
                      public IPositionalDecode
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TInput, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ConfigType = LpeConfig;

        CudaLpeOp( IExecutionContext* context, const LpeConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaLpeOp" ) ), config_( config )
        {
            config_.validate();
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

        /**
         * @brief Bind wte and wpe parameter tensors (module retains ownership).
         *
         * Caches native device pointers and validates tensor shapes against the
         * configuration. Must be called before build().
         *
         * @param wte Token embedding table — CUDA tensor of shape [vocab_size, C].
         * @param wpe Positional embedding table — CUDA tensor of shape [max_seq_len, C].
         *
         * @throws std::invalid_argument on null, non-CUDA, or shape-mismatched tensors.
         */
        void setParameters( ITensor* wte, ITensor* wpe ) override
        {
            if ( !wte || !wpe )
                throw std::invalid_argument( "CudaLpeOp::setParameters - both wte and wpe are required" );

            if ( wte->getDeviceType() != DeviceType::Cuda || wpe->getDeviceType() != DeviceType::Cuda )
                throw std::invalid_argument( "CudaLpeOp::setParameters - parameters must be CUDA tensors" );

            const auto& wte_shape = wte->shape();
            const auto& wpe_shape = wpe->shape();

            if ( wte_shape.size() != 2 || wpe_shape.size() != 2 )
                throw std::invalid_argument( "CudaLpeOp::setParameters - wte and wpe must be 2D tensors" );

            if ( wte_shape[ 1 ] != wpe_shape[ 1 ] )
                throw std::invalid_argument( "CudaLpeOp::setParameters - wte and wpe embedding dimensions must match" );

            wte_ = static_cast<NativeType*>( wte->rawData() );
            wpe_ = static_cast<NativeType*>( wpe->rawData() );

            wte_vocab_size_    = static_cast<int>( wte_shape[ 0 ] );
            wte_embedding_dim_ = static_cast<int>( wte_shape[ 1 ] );
            wpe_max_seq_len_   = static_cast<int>( wpe_shape[ 0 ] );
            wpe_embedding_dim_ = static_cast<int>( wpe_shape[ 1 ] );
        }

        /**
         * @brief Bind wte and wpe gradient tensors for training (module retains ownership).
         *
         * @param wte_grad Gradient buffer for wte — CUDA tensor of shape [vocab_size, C].
         * @param wpe_grad Gradient buffer for wpe — CUDA tensor of shape [max_seq_len, C].
         *
         * @throws std::invalid_argument on null or non-CUDA tensors.
         */
        void setGradients( ITensor* wte_grad, ITensor* wpe_grad ) override
        {
            if ( !wte_grad || !wpe_grad )
                throw std::invalid_argument( "CudaLpeOp::setGradients - both wte_grad and wpe_grad are required" );

            if ( wte_grad->getDeviceType() != DeviceType::Cuda || wpe_grad->getDeviceType() != DeviceType::Cuda )
                throw std::invalid_argument( "CudaLpeOp::setGradients - gradients must be CUDA tensors" );

            wte_grad_ = static_cast<NativeType*>( wte_grad->rawData() );
            wpe_grad_ = static_cast<NativeType*>( wpe_grad->rawData() );
        }

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /**
         * @brief Prepare the operation for a concrete input shape (cold path).
         *
         * Validates parameters, caches B, T, and C for hot-path dispatch, and
         * verifies that the sequence length fits within the positional embedding table.
         * Must be called after setParameters() and before forward(), backward(), or decode().
         *
         * @param input_shape Token index input shape [B, T].
         *
         * @throws std::runtime_error  if parameters are not bound.
         * @throws std::invalid_argument if input shape is invalid or sequence length
         *                               exceeds the positional embedding capacity.
         */
        void build( const shape_t& input_shape ) override
        {
            if ( !wte_ || !wpe_ )
                throw std::runtime_error( "CudaLpeOp::build requires parameters bound via setParameters() before build()." );

            validateInputShape( input_shape );

            batch_size_    = static_cast<int>( input_shape[ 0 ] );
            seq_length_    = static_cast<int>( input_shape[ 1 ] );
            embedding_dim_ = wte_embedding_dim_;

            if ( seq_length_ > wpe_max_seq_len_ )
                throw std::invalid_argument( "CudaLpeOp::build - sequence length exceeds positional embedding capacity" );

            if ( embedding_dim_ != config_.getEmbeddingDim() )
                throw std::invalid_argument( "CudaLpeOp::build - parameter embedding dimension does not match configuration" );

            UnaryOperationBase::build( input_shape );
        }

        // ====================================================================
        // Forward
        // ====================================================================

        /**
         * @brief Full-sequence forward pass (hot path).
         *
         * For each (b, t): output[b,t,:] = wte[X[b,t],:] + wpe[t,:].
         *
         * @param input  Token indices [B, T] (INT32).
         * @param output Pre-allocated embeddings [B, T, C].
         *
         * @throws std::runtime_error if the input shape exceeds the built maximum.
         */
        void forward( const ITensor& input, ITensor& output ) const override
        {
            const auto& input_shape = input.shape();
            int B = static_cast<int>( input_shape[ 0 ] );
            int T = static_cast<int>( input_shape[ 1 ] );

            if ( B > batch_size_ || T > seq_length_ )
                throw std::runtime_error( std::format(
                    "CudaLpeOp: input shape [{}, {}] exceeds built max [{}, {}]",
                    B, T, batch_size_, seq_length_ ) );

            const int32_t* X = static_cast<const int32_t*>( input.rawData() );
            NativeType* Y    = static_cast<NativeType*>( output.rawData() );

            Detail::cuda_lpe_impl<NativeType>::forward(
                Y, X, wte_, wpe_, B, T, embedding_dim_, context_->getStream() );
        }

        // ====================================================================
        // Backward
        // ====================================================================

        /**
         * @brief Backward pass accumulating gradients into wte and wpe (hot path).
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
            const auto& input_shape = input.shape();
            int B = static_cast<int>( input_shape[ 0 ] );
            int T = static_cast<int>( input_shape[ 1 ] );

            if ( B > batch_size_ || T > seq_length_ )
                throw std::runtime_error( std::format(
                    "CudaLpeOp: input shape [{}, {}] exceeds built max [{}, {}]",
                    B, T, batch_size_, seq_length_ ) );

            const int32_t* X  = static_cast<const int32_t*>( input.rawData() );
            const NativeType* dY = static_cast<const NativeType*>( output_grad.rawData() );

            Detail::cuda_lpe_impl<NativeType>::backward(
                wte_grad_, wpe_grad_, X, dY, B, T, embedding_dim_, context_->getStream() );
        }

        // ====================================================================
        // Decode (IPositionalDecode)
        // ====================================================================

        /**
         * @brief Single-token decode with an explicit sequence position (hot path).
         *
         * Computes output[b,:] = wte[X[b,0],:] + wpe[position,:] for each batch
         * element. The dispatch implementation shifts the wpe pointer to row
         * `position` and calls the forward kernel with T=1, so no dedicated decode
         * kernel is required.
         *
         * Precondition: build() must have been called. position must be in
         * [0, max_seq_len).
         *
         * @param input    Single-token indices [B, 1] (INT32).
         * @param output   Pre-allocated output buffer [B, 1, C] (only first B*C
         *                 elements are written).
         * @param position Zero-based absolute sequence position for the wpe lookup.
         *
         * @throws std::invalid_argument if position is out of range.
         */
        void decode( const ITensor& input, ITensor& output, int position ) const override
        {
            if ( position < 0 || position >= wpe_max_seq_len_ )
                throw std::invalid_argument( std::format(
                    "CudaLpeOp::decode: position {} out of range [0, {})",
                    position, wpe_max_seq_len_ ) );

            int B = static_cast<int>( input.shape()[ 0 ] );

            const int32_t* X = static_cast<const int32_t*>( input.rawData() );
            NativeType* Y    = static_cast<NativeType*>( output.rawData() );

            Detail::cuda_lpe_impl<NativeType>::decode(
                Y, X, wte_, wpe_, B, position, embedding_dim_, context_->getStream() );
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        OperationType getOperationType() const override
        {
            return OperationType::LpeOp;
        }

        std::string getName() const override
        {
            return "Cuda::LpeOp";
        }

    private:
        LpeConfig config_;
        CudaExecutionContext* context_;

        NativeType* wte_{ nullptr };
        NativeType* wpe_{ nullptr };
        NativeType* wte_grad_{ nullptr };
        NativeType* wpe_grad_{ nullptr };

        int wte_vocab_size_{ 0 };
        int wte_embedding_dim_{ 0 };
        int wpe_max_seq_len_{ 0 };
        int wpe_embedding_dim_{ 0 };

        int batch_size_{ 0 };
        int seq_length_{ 0 };
        int embedding_dim_{ 0 };

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.size() != 2 )
                throw std::invalid_argument( "CudaLpeOp: input must have rank 2 (batch_size, sequence_length)" );

            if ( input_shape[ 1 ] > config_.getMaxSequenceLength() )
                throw std::invalid_argument( "CudaLpeOp: sequence length exceeds configured maximum" );
        }
    };

    export class CudaLpeOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "LpeOp";

            registerUnaryOpType<DeviceType::Cuda,
                CudaLpeOp<TensorDataType::INT32, TensorDataType::FP32>,
                TensorDataType::INT32, TensorDataType::FP32>( opName );

            registerUnaryOpType<DeviceType::Cuda,
                CudaLpeOp<TensorDataType::INT32, TensorDataType::FP16>,
                TensorDataType::INT32, TensorDataType::FP16>( opName );
        }
    };
}