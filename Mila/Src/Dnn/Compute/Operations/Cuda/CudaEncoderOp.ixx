/**
 * @file CudaEncoderOp.ixx
 * @brief Implementation of the CUDA-based Encoder operation for transformer models.
 */

module;
#include <cuda_fp16.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include "Kernels/CudaOps.h"

export module Compute.CudaEncoderOp;

import Dnn.Modules.Encoder;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.ConfigurationBase;
import Compute.Precision;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.MemoryResource;
import Compute.CudaMemoryResource;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    namespace Detail
    {
        /**
         * @brief Primary template for precision-specific CUDA encoder implementations.
         *
         * @tparam TOutput The floating-point precision type (float or half)
         */
        template <typename TOutput>
        struct cuda_encoder_impl;

        /**
         * @brief Single-precision (float) specialization for CUDA encoder operations.
         */
        template <>
        struct cuda_encoder_impl<float> {
            static inline void forward( float* Y, const int* X, const float* wte, const float* wpe, int B, int T, int C, cudaStream_t stream ) {
                cuda_encoder_forward_fp32( Y, X, wte, wpe, B, T, C, stream );
            }
        };

        /**
         * @brief Half-precision (half) specialization for CUDA encoder operations.
         */
        template <>
        struct cuda_encoder_impl<half> {
            static inline void forward( half* Y, const int* X, const half* wte, const half* wpe, int B, int T, int C, cudaStream_t stream ) {
                cuda_encoder_forward_fp16( Y, X, wte, wpe, B, T, C, stream );
            }
        };
    }

    /**
     * @brief CUDA implementation of the Encoder operation for transformer models.
     *
     * This class provides a CUDA-based implementation of the Encoder operation, which performs
     * token embedding lookups and positional embedding additions. It transforms discrete
     * token IDs into continuous vector representations by combining:
     * 1. Token embeddings from a learned vocabulary table (wte)
     * 2. Positional embeddings that encode sequence position information (wpe)
     *
     * The implementation is optimized for NVIDIA GPUs using CUDA for high-performance computation,
     * supporting both integer and half-precision floating-point operations.
     *
     * @tparam int The data type of the input tensor elements (typically uint16_t or int for token IDs).
     * @tparam TDataType The data type used for computation and output (typically half or float).
     */
    export template<typename TInput, typename TOutput = TInput>
		requires ValidFloatTensorTypes<TInput,TOutput>
    class CudaEncoderOp : public UnaryOperation<DeviceType::Cuda, int, TOutput> {
        public:
            using MR = typename CudaDevice::MR;
			using OperationBase = UnaryOperation<DeviceType::Cuda, int, TOutput>;

            /**
             * @brief Constructs a new CUDA Encoder operation with the default device context.
             *
             * Initializes the operation with a CUDA device context (defaults to CUDA:0).
             */
            CudaEncoderOp( const EncoderConfig& config )
                : OperationBase( OperationType::EncoderOp ), config_( config ) {}

            /**
             * @brief Constructs a new CUDA Encoder operation with a specific device context.
             *
             * @param context The device context to use for this operation.
             * @throws std::runtime_error If the context is not for a CUDA device.
             */
            CudaEncoderOp( std::shared_ptr<DeviceContext> context, const EncoderConfig& config )
                : OperationBase( OperationType::EncoderOp, context ), config_( config ) {
            }

            /**
             * @brief Performs the forward pass of the Encoder operation on CUDA.
             *
             * Transforms input token IDs into continuous embeddings by:
             * 1. Looking up token embeddings from the embedding table (wte)
             * 2. Adding positional embeddings (wpe) based on token position
             *
             * The computation is performed on the GPU using CUDA kernels for optimal performance.
             *
             * @param input Input tensor of shape [B, TDataType] containing token IDs, where B is batch size and TDataType is sequence length.
             * @param parameters Vector of parameter tensors [wte, wpe] where wte is of shape [V, C] (vocabulary size � embedding dimension)
             *                   and wpe is of shape [maxT, C] (maximum sequence length � embedding dimension).
             * @param properties Additional attributes for the operation.
             * @param output Output tensor of shape [B, TDataType, C] containing the resulting embeddings.
             * @param output_state Cache for intermediate results (not used in this operation).
             */
            void forward(
                const Tensor<int, MR>& input,
                const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& parameters,
                const OperationAttributes& properties,
                Tensor<TOutput, MR>& output,
                std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_state ) const override {

				// TODO: Argument validation. 

                auto X = input.raw_data();
                auto Y = output.raw_data();

                auto wte = parameters[ 0 ];
                auto wpe = parameters[ 1 ];

                int B = input.shape()[ 0 ];
                int T = input.shape()[ 1 ];
                int C = wte->shape()[ 1 ];

                cudaStream_t stream = this->getDeviceContext()->getStream();

                Detail::cuda_encoder_impl<TOutput>::forward( Y, X, wte->raw_data(), wpe->raw_data(), B, T, C, stream );
            }

            /**
             * @brief Performs the backward pass of the Encoder operation.
             *
             * Computes gradients with respect to the embedding tables (token and position).
             *
             * @param input Input tensor from the forward pass.
             * @param output Output tensor from the forward pass.
             * @param output_gradient Gradient of the loss with respect to the output.
             * @param parameters Parameters tensor from forward pass.
             * @param parameter_gradients Gradients for parameters (embedding tables).
             * @param input_gradient Gradient of the loss with respect to the input (typically not used for discrete inputs).
             * @param properties Additional attributes for the operation.
             * @param output_state Cache tensors from forward pass.
             */
            void backward(
                const Tensor<int, MR>& input,
                const Tensor<TOutput, MR>& output,
                const Tensor<TOutput, MR>& output_gradient,
                const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& parameters,
                std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& parameter_gradients,
                Tensor<int, MR>& input_gradient,
                const OperationAttributes& properties,
                const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_state ) const {

                if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cuda ) ) {
                    throw std::runtime_error( "CudaEncoderOp::backward can only be executed on CUDA device" );
                }

                // Implementation for backward pass for Encoder operation
                // Typically this would update gradients for the embedding tables (wte and wpe)

                // Get CUDA stream from device context
                cudaStream_t stream = this->getDeviceContext()->getStream();

                // FIXME: Implement backward pass using appropriate CUDA kernels
                // cuda_encoder_backward(...);
            }

            /**
             * @brief Gets the name of this operation.
             *
             * @return std::string The name of the operation ("Cuda::EncoderOp").
             */
            std::string getName() const override {
                return "Cuda::EncoderOp";
            }
            
        private:
            EncoderConfig config_; ///< Configuration for the encoder operation.
    };

    /**
     * @brief Class responsible for registering the CudaEncoderOp operation.
     *
     * The CudaEncoderOpRegistrar class registers the CudaEncoderOp operation with the OperationRegistry.
     * It associates the operation name "Cuda::EncoderOp" with a factory function that creates
     * instances of CudaEncoderOp with appropriate template parameters.
     */
    export class CudaEncoderOpRegistrar {
    public:
        /**
         * @brief Registers the CudaEncoderOp operation with the OperationRegistry.
         *
         * This function registers the CudaEncoderOp operation for the CUDA device type
         * with the OperationRegistry. It associates the operation name "Cuda::EncoderOp"
         * with a factory function that creates instances of CudaEncoderOp.
         */
        static void registerOperations() {
            const std::string opName = "Cuda::EncoderOp";

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, int, float>(
                opName,
                []( std::shared_ptr<DeviceContext> context, const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, int, float>> {
                    const auto& encoderConfig = static_cast<const EncoderConfig&>( config );
                    return context ? std::make_shared<CudaEncoderOp<float>>( context, encoderConfig )
                        : std::make_shared<CudaEncoderOp<float>>( encoderConfig );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, int, half>(
                opName,
                []( std::shared_ptr<DeviceContext> context, const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, int, half>> {
                    const auto& encoderConfig = static_cast<const EncoderConfig&>( config );
                    return context ? std::make_shared<CudaEncoderOp<half>>( context, encoderConfig )
                        : std::make_shared<CudaEncoderOp<half>>( encoderConfig );
                }
            );
        }

        /**
         * @brief Self-registration mechanism that registers the operation during startup.
         *
         * This static member ensures the operation is registered when the program starts
         * without requiring explicit registration calls.
         */
        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}
