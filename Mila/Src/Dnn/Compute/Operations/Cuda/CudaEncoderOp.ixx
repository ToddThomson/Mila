/**
 * @file CudaEncoderOp.ixx
 * @brief Implementation of the CUDA-based Encoder operation for transformer models.
 */

module;
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include "Kernels/Cuda.Ops.h"
#include <cuda_fp16.h>

export module Compute.CudaEncoderOp;

import Dnn.Tensor;
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

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
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
     * @tparam TInput The data type of the input tensor elements (typically uint16_t or int for token IDs).
     * @tparam TPrecision The data type used for computation and output (typically half or float).
     */
    export
        template<typename TInput = uint16_t, typename TPrecision = half>
        requires (std::is_same_v<TInput, int> || std::is_same_v<TInput, uint16_t>) &&
    (std::is_same_v<TPrecision, half> || std::is_same_v<TPrecision, float>)
        class CudaEncoderOp : public UnaryOperation<TInput, TPrecision, DeviceType::Cuda> {
        public:
            using MR = typename CudaDevice::MR;

            /**
             * @brief Constructs a new CUDA Encoder operation with the default device context.
             *
             * Initializes the operation with a CUDA device context (defaults to CUDA:0).
             */
            CudaEncoderOp() : UnaryOperation<TInput, TPrecision, DeviceType::Cuda>( OperationType::EncoderOp ) {}

            /**
             * @brief Constructs a new CUDA Encoder operation with a specific device context.
             *
             * @param context The device context to use for this operation.
             * @throws std::runtime_error If the context is not for a CUDA device.
             */
            CudaEncoderOp( std::shared_ptr<DeviceContext> context )
                : UnaryOperation<TInput, TPrecision, DeviceType::Cuda>( OperationType::EncoderOp, context ) {
                if ( !context->isDeviceType( DeviceType::Cuda ) ) {
                    throw std::runtime_error( "CudaEncoderOp requires a CUDA device context" );
                }
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
             * @param input Input tensor of shape [B, T] containing token IDs, where B is batch size and T is sequence length.
             * @param parameters Vector of parameter tensors [wte, wpe] where wte is of shape [V, C] (vocabulary size � embedding dimension)
             *                   and wpe is of shape [maxT, C] (maximum sequence length � embedding dimension).
             * @param properties Additional attributes for the operation.
             * @param output Output tensor of shape [B, T, C] containing the resulting embeddings.
             * @param output_cache Cache for intermediate results (not used in this operation).
             */
            void forward(
                const Tensor<TInput, MR>& input,
                const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
                const OperationAttributes& properties,
                Tensor<TPrecision, MR>& output,
                std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_cache ) const override {

                // Verify we're operating on CUDA memory
                if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cuda ) ) {
                    throw std::runtime_error( "CudaEncoderOp::forward can only be executed on CUDA memory" );
                }

                auto X = input.data();
                auto Y = output.data();

                auto wte = parameters[ 0 ];
                auto wpe = parameters[ 1 ];

                int B = input.shape()[ 0 ];
                int T = input.shape()[ 1 ];
                int C = wte->shape()[ 1 ];

                // Get CUDA stream from device context
                cudaStream_t stream = this->getDeviceContext()->getStream();

                // FIXME: Pass the stream to the kernel
                // cuda_encoder_forward(Y, X, wte->data(), wpe->data(), B, T, C, stream);
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
             * @param output_cache Cache tensors from forward pass.
             */
            void backward(
                const Tensor<TInput, MR>& input,
                const Tensor<TPrecision, MR>& output,
                const Tensor<TPrecision, MR>& output_gradient,
                const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
                std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameter_gradients,
                Tensor<TInput, MR>& input_gradient,
                const OperationAttributes& properties,
                const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_cache ) const {

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

            // Register float precision version with device context awareness
            OperationRegistry::instance().registerOperation<uint16_t, float, DeviceType::Cuda>(
                opName,
                "float",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<uint16_t, float, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaEncoderOp<uint16_t, float>>( context )
                        : std::make_shared<CudaEncoderOp<uint16_t, float>>();
                }
            );

            // Register half precision version with device context awareness
            OperationRegistry::instance().registerOperation<uint16_t, half, DeviceType::Cuda>(
                opName,
                "half",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<uint16_t, half, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaEncoderOp<uint16_t, half>>( context )
                        : std::make_shared<CudaEncoderOp<uint16_t, half>>();
                }
            );

            // Register int input version with device context awareness
            OperationRegistry::instance().registerOperation<int, float, DeviceType::Cuda>(
                opName,
                "int_float",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<int, float, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaEncoderOp<int, float>>( context )
                        : std::make_shared<CudaEncoderOp<int, float>>();
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
