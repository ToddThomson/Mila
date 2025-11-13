/**
 * @file CudaEncoderOp.ixx
 * @brief CUDA implementation of Encoder operation for token and positional embeddings (TensorDataType-based).
 *
 * Implements forward and backward passes for combining token embeddings (wte)
 * and positional embeddings (wpe) on CUDA devices.
 */

module;
#include <cuda_fp16.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>
#include "Kernels/CudaOps.h"
#include <type_traits>

export module Compute.CudaEncoderOp;

import Dnn.Modules.Encoder;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorHostTypeMap;
import Dnn.ConfigurationBase;
import Compute.Precision;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.CudaExecutionContext;
import Compute.CudaDeviceResources;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute
{
    namespace Detail
    {
        /**
         * @brief CUDA kernel dispatcher for Encoder operations.
         *
         * Specialized for float (FP32) and half (FP16) native CUDA types.
         */
        template <typename TNative>
            requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
        struct cuda_encoder_impl;

        /**
         * @brief Single-precision (float) specialization for CUDA encoder operations.
         */
        template <>
        struct cuda_encoder_impl<float>
        {
            cuda_encoder_impl() = default;

            static inline void forward(
                float* Y, const int32_t* X,
                const float* wte, const float* wpe,
                int B, int T, int C,
                cudaStream_t stream )
            {
                cuda_encoder_forward_fp32( Y, X, wte, wpe, B, T, C, stream );
            }

            static inline void backward(
                float* dwte, float* dwpe,
                const int32_t* X, const float* dY,
                int B, int T, int C,
                cudaStream_t stream )
            {
                // FIXME: cuda_encoder_backward_fp32( dwte, dwpe, X, dY, B, T, C, stream );
            }
        };

        /**
         * @brief Half-precision (half) specialization for CUDA encoder operations.
         */
        template <>
        struct cuda_encoder_impl<half>
        {
            cuda_encoder_impl() = default;

            static inline void forward(
                half* Y, const int32_t* X,
                const half* wte, const half* wpe,
                int B, int T, int C,
                cudaStream_t stream )
            {
                cuda_encoder_forward_fp16( Y, X, wte, wpe, B, T, C, stream );
            }

            static inline void backward(
                half* dwte, half* dwpe,
                const int32_t* X, const half* dY,
                int B, int T, int C,
                cudaStream_t stream )
            {
                // FIXME: cuda_encoder_backward_fp16( dwte, dwpe, X, dY, B, T, C, stream );
            }
        };
    }

    using namespace Mila::Dnn;

    /**
     * @brief CUDA implementation of Encoder operation using abstract TensorDataType API.
     *
     * Template parameter TPrecision selects the abstract tensor precision (e.g. FP32, FP16).
     * NativeType is the corresponding CUDA device representation for that precision.
     *
     * Design philosophy:
     * - Two-phase initialization: build() does all setup, forward()/backward() are pure dispatch
     * - Module owns wte/wpe parameters and binds them via setParameters()
     * - All dimension computation and validation happens once in build()
     * - Forward/backward are hot-path methods with minimal overhead
     * - Token indices (INT32) are non-differentiable, no input gradient computed
     */
    export template<TensorDataType TInput, TensorDataType TPrecision = TInput>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaEncoderOp : public UnaryOperation<DeviceType::Cuda, TInput, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        //using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TInput, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        //using Parameters = std::vector<std::shared_ptr<TensorType>>;
        //using OutputState = std::vector<std::shared_ptr<TensorType>>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaEncoderOp( std::shared_ptr<CudaExecutionContext> context, const EncoderConfig& config )
            : config_( config ), context_( context ), impl_()
        {
            if (!context_)
            {
                throw std::runtime_error( "CudaEncoderOp requires a CUDA execution context" );
            }

            config_.validate();
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

        /**
         * @brief Set parameter tensor references (module remains owner).
         *
         * The operation caches native device pointers for hot-path access.
         * Both wte (token embeddings) and wpe (positional embeddings) are required.
         *
         * Note: build() requires parameters to be bound before it is called.
         */
        void setParameters( ITensor* wte, ITensor* wpe ) override
        {
            if (!wte || !wpe)
            {
                throw std::invalid_argument( "CudaEncoderOp::setParameters - both wte and wpe parameters are required" );
            }

            if (wte->getDeviceType() != DeviceType::Cuda || wpe->getDeviceType() != DeviceType::Cuda)
            {
                throw std::invalid_argument( "CudaEncoderOp::setParameters - parameters must be CUDA tensors" );
            }

            wte_ = static_cast<NativeType*>(wte->rawData());
            wpe_ = static_cast<NativeType*>(wpe->rawData());

            // Store shapes for validation
            const auto& wte_shape = wte->shape();
            const auto& wpe_shape = wpe->shape();

            if (wte_shape.size() != 2 || wpe_shape.size() != 2)
            {
                throw std::invalid_argument( "CudaEncoderOp::setParameters - wte and wpe must be 2D tensors" );
            }

            wte_vocab_size_ = static_cast<int>(wte_shape[0]);
            wte_embedding_dim_ = static_cast<int>(wte_shape[1]);

            wpe_max_seq_len_ = static_cast<int>(wpe_shape[0]);
            wpe_embedding_dim_ = static_cast<int>(wpe_shape[1]);

            if (wte_embedding_dim_ != wpe_embedding_dim_)
            {
                throw std::invalid_argument( "CudaEncoderOp::setParameters - wte and wpe must have same embedding dimension" );
            }
        }

        /**
         * @brief Set parameter gradient tensor references for training.
         *
         * The operation caches native device gradient pointers for hot-path write access
         * during backward(). Both wte_grad and wpe_grad are required.
         */
        void setParameterGradients( ITensor* wte_grad, ITensor* wpe_grad ) override
        {
            if (!wte_grad || !wpe_grad)
            {
                throw std::invalid_argument( "CudaEncoderOp::setParameterGradients - both gradients are required" );
            }

            if (wte_grad->getDeviceType() != DeviceType::Cuda || wpe_grad->getDeviceType() != DeviceType::Cuda)
            {
                throw std::invalid_argument( "CudaEncoderOp::setParameterGradients - gradients must be CUDA tensors" );
            }

            wte_grad_ = static_cast<NativeType*>(wte_grad->rawData());
            wpe_grad_ = static_cast<NativeType*>(wpe_grad->rawData());
        }

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /**
         * @brief Build the operation for a concrete input shape.
         *
         * This is the COLD PATH where all setup, validation, and computation happens ONCE.
         * After build() completes, forward() and backward() become pure dispatch methods.
         *
         * Responsibilities:
         *  1. Validate parameters are bound via setParameters()
         *  2. Validate input shape (must be [B, T] for token indices)
         *  3. Compute and cache kernel dispatch dimensions [B, T, C]
         *  4. Validate sequence length against configured maximum
         *
         * After build(), the operation is ready for zero-overhead forward/backward dispatch.
         */
        void build( const shape_t& input_shape ) override
        {
            if (wte_ == nullptr || wpe_ == nullptr)
            {
                throw std::runtime_error( "CudaEncoderOp::build requires parameters bound via setParameters() before build()." );
            }

            validateInputShape( input_shape );

            cached_batch_size_ = static_cast<int>(input_shape[0]);
            cached_seq_length_ = static_cast<int>(input_shape[1]);
            cached_embedding_dim_ = wte_embedding_dim_;

            // Validate sequence length against configured maximum
            if (cached_seq_length_ > wpe_max_seq_len_)
            {
                throw std::invalid_argument(
                    "CudaEncoderOp::build - sequence length exceeds positional embedding capacity" );
            }

            // Validate embedding dimensions match configuration
            if (cached_embedding_dim_ != config_.getChannels())
            {
                throw std::invalid_argument(
                    "CudaEncoderOp::build - parameter embedding dimension doesn't match configuration" );
            }

            //UnaryOperationBase::build( input_shape );
        }

        // ====================================================================
        // Forward pass
        // ====================================================================

        /**
         * @brief Forward pass - HOT PATH, pure dispatch to CUDA kernel.
         *
         * All setup, validation, and dimension computation was done in build().
         * This method extracts raw pointers and dispatches directly to the kernel
         * using pre-computed cached dimensions.
         *
         * For each position (b, t) in the batch:
         *   output[b, t, :] = wte[input[b, t], :] + wpe[t, :]
         *
         * Zero redundant work - maximum performance.
         */
        void forward( const ITensor& input, ITensor& output ) const override
        {
            // Input is INT32 token indices, output is NativeType embeddings
            const int32_t* X = static_cast<const int32_t*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            cudaStream_t stream = context_->getStream();

            Detail::cuda_encoder_impl<NativeType>::forward(
                Y, X,
                wte_, wpe_,
                cached_batch_size_,
                cached_seq_length_,
                cached_embedding_dim_,
                stream
            );
        }

        // ====================================================================
        // Backward pass
        // ====================================================================

        /**
         * @brief Backward pass - HOT PATH, pure dispatch to CUDA kernel.
         *
         * Similar to forward(), this method does minimal work and dispatches
         * directly to the backward kernel using cached dimensions from build().
         *
         * Accumulates gradients into wte and wpe embedding tables.
         * Token indices are discrete (non-differentiable), so no input gradient.
         */
        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            const int32_t* X = static_cast<const int32_t*>(input.rawData());
            const NativeType* dY = static_cast<const NativeType*>(output_grad.rawData());

            NativeType* dwte = wte_grad_;
            NativeType* dwpe = wpe_grad_;

            cudaStream_t stream = context_->getStream();

            Detail::cuda_encoder_impl<NativeType>::backward(
                dwte, dwpe,
                X, dY,
                cached_batch_size_,
                cached_seq_length_,
                cached_embedding_dim_,
                stream
            );

            // input_grad is unused (token indices are non-differentiable)
        }

        OperationType getOperationType() const override
        {
            return OperationType::EncoderOp;
        }

        std::string getName() const override
        {
            return "Cuda::EncoderOp";
        }

        const EncoderConfig& getConfig() const
        {
            return config_;
        }

    private:
        EncoderConfig config_;
        std::shared_ptr<CudaExecutionContext> context_;
        Detail::cuda_encoder_impl<NativeType> impl_;

        // Cached native device parameter pointers (module owns underlying tensors)
        NativeType* wte_{ nullptr };  // Token embeddings (V, C)
        NativeType* wpe_{ nullptr };  // Position embeddings (maxT, C)

        // Cached native device parameter gradient pointers (module owns underlying tensors)
        NativeType* wte_grad_{ nullptr };
        NativeType* wpe_grad_{ nullptr };

        // Parameter dimensions for validation
        int wte_vocab_size_{ 0 };
        int wte_embedding_dim_{ 0 };
        int wpe_max_seq_len_{ 0 };
        int wpe_embedding_dim_{ 0 };

        // Cached dimension values computed once in build() for hot-path dispatch
        int cached_batch_size_{ 0 };
        int cached_seq_length_{ 0 };
        int cached_embedding_dim_{ 0 };

        void validateInputShape( const shape_t& input_shape ) const
        {
            if (input_shape.size() != 2)
            {
                throw std::invalid_argument(
                    "CudaEncoderOp: input must have rank 2 (batch_size, sequence_length)" );
            }

            if (input_shape[1] > config_.getMaxSequenceLength())
            {
                throw std::invalid_argument(
                    "CudaEncoderOp: sequence length exceeds configured maximum" );
            }
        }
    };

    export class CudaEncoderOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "EncoderOp";

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::INT32, TensorDataType::FP32>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::INT32, TensorDataType::FP32>>
                {
                    const auto& encoder_config = dynamic_cast<const EncoderConfig&>(config);
                    return std::make_shared<CudaEncoderOp<TensorDataType::INT32, TensorDataType::FP32>>( context, encoder_config );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::INT32, TensorDataType::FP16>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::INT32, TensorDataType::FP16>>
                {
                    const auto& encoder_config = dynamic_cast<const EncoderConfig&>(config);
                    return std::make_shared<CudaEncoderOp<TensorDataType::INT32, TensorDataType::FP16>>( context, encoder_config );
                }
            );
        }

        static inline bool isRegistered = []()
            {
                registerOperations();
                return true;
            }();
    };
}