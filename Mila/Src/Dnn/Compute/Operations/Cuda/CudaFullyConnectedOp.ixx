/**
 * @file CudaFullyConnectedOp.ixx
 * @brief Implementation of the CUDA-based Fully Connected operation for neural networks.
 */

module;
#include <cublasLt.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <exception>
#include "Kernels/Cuda.Ops.h"

export module Compute.CudaMatMulOp;

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
import Compute.CublasLtMatMulBias;
import Utils.Logger;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CUDA implementation of the Fully Connected operation for neural networks.
     *
     * This class provides a CUDA-based implementation of the Fully Connected operation,
     * which performs a matrix multiplication between the input and a weight matrix,
     * optionally adds a bias, and produces an output. The operation is accelerated
     * on NVIDIA GPUs using CUDA for high-performance computation.
     *
     * The implementation delegates the actual matrix multiplication to optimized CUDA kernels
     * that may leverage libraries like cuBLAS or custom implementations for maximum performance.
     *
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TDataType The data type of the output tensor elements (defaults to the input type).
     */
    export template<typename TInput, typename TPrecision = TInput>
    class CudaFullyConnectedOp : public UnaryOperation<TInput, TPrecision, DeviceType::Cuda> {
    public:
        using MR = typename CudaDevice::MR;
        /**
         * @brief Constructs a new CUDA Fully Connected operation with the default device context.
         *
         * Initializes the operation with a CUDA device context (defaults to CUDA:0).
         */
        CudaFullyConnectedOp() : UnaryOperation<TInput, TPrecision, DeviceType::Cuda>( OperationType::FullyConnectedOp ) {}

        /**
         * @brief Constructs a new CUDA Fully Connected operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CUDA device.
         */
        CudaFullyConnectedOp( std::shared_ptr<DeviceContext> context )
            : UnaryOperation<TInput, TPrecision, DeviceType::Cuda>( OperationType::FullyConnectedOp, context ) {
        }

        /**
         * @brief Performs the forward pass of the Fully Connected operation on CUDA.
         *
         * Computes the matrix multiplication between input and weights, adds bias if provided,
         * and stores the result in the output tensor. The computation is performed on the GPU
         * using CUDA kernels for optimal performance.
         *
         * @param input Input tensor of shape [B, TDataType, C] where B is batch size, TDataType is sequence length, and C is input feature dimension.
         * @param parameters Vector of parameter tensors [weight, bias] where weight is of shape [OC, C] and bias is of shape [OC].
         * @param properties Additional attributes for the operation.
         * @param output Output tensor of shape [B, TDataType, OC] where OC is output feature dimension.
         * @param output_cache Cache for intermediate results (not used in this operation).
         */
        void forward(
            const Tensor<TInput, MR>& input,
            const std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameters,
            const OperationAttributes& properties,
            Tensor<TPrecision, MR>& output,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_cache ) const override {

            auto X = input.raw_data();
            auto Y = output.raw_data();

            auto weight = parameters[ 0 ]->raw_data();
            float* bias = nullptr;

            if ( parameters.size() == 2 ) {
                bias = parameters[ 1 ]->raw_data();
            }

            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int C = input.shape()[ 2 ];
            int OC = output.shape()[ 2 ];

            cudaStream_t stream = this->getDeviceContext()->getStream();

            // Try to use cuBLASLt first if available
            try {
                cublasLtHandle_t cublasLtHandle = this->getDeviceContext()->getCublasLtHandle();
                if ( cublasLtHandle ) {
                    cublaslt_matmul_forward<TPrecision>( Y, X, weight, bias, B, T, C, OC, stream, cublasLtHandle );
                    
                    return;
                }
            }
            catch ( const std::exception& e ) {
                Utils::Logger::warning( std::string( "cuBLASLt failed, falling back to custom CUDA kernel: " ) + e.what() );
            }

            // Fallback to custom cuda kernel...
            cuda_matmul_forward( Y, X, weight, bias, B, T, C, OC, stream );
        }

        /**
         * @brief Performs the backward pass of the Fully Connected operation.
         *
         * Computes gradients with respect to inputs, weights, and biases.
         *
         * @param input Input tensor from the forward pass.
         * @param output Output tensor from the forward pass.
         * @param output_gradient Gradient of the loss with respect to the output.
         * @param parameters Parameters tensor from forward pass [weight, bias].
         * @param parameter_gradients Gradients for parameters [d_weight, d_bias].
         * @param input_gradient Gradient of the loss with respect to the input.
         * @param properties Additional attributes for the operation.
         * @param output_cache Cache tensors from forward pass.
         */
        void backward(
            const Tensor<TInput, MR>& input,
            const Tensor<TPrecision, MR>& output,
            const Tensor<TPrecision, MR>& output_gradient,
            const std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameter_gradients,
            Tensor<TInput, MR>& input_gradient,
            const OperationAttributes& properties,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_cache ) const {

            // Verify we're operating on CUDA memory
            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cuda ) ) {
                throw std::runtime_error( "CudaFullyConnectedOp::backward can only be executed on CUDA memory" );
            }

            // Extract dimensions
            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int C = input.shape()[ 2 ];
            int OC = output.shape()[ 2 ];

            // Extract tensors
            auto X = input.data();
            auto dY = output_gradient.data();
            auto dX = input_gradient.data();
            auto W = parameters[ 0 ]->data();
            auto dW = parameter_gradients[ 0 ]->data();
            auto dB = parameter_gradients.size() > 1 ? parameter_gradients[ 1 ]->data() : nullptr;

            // Get CUDA stream from device context
            cudaStream_t stream = this->getDeviceContext()->getStream();

            // Call CUDA backward kernels with stream
            // FIXME: Implement these CUDA kernels
            // cuda_matmul_backward_input(dX, dY, W, B, TDataType, C, OC, stream);
            // cuda_matmul_backward_weight(dW, X, dY, B, TDataType, C, OC, stream);
            // if (dB != nullptr) {
            //     cuda_matmul_backward_bias(dB, dY, B, TDataType, OC, stream);
            // }
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cuda::FullyConnectedOp").
         */
        std::string getName() const override {
            return "Cuda::FullyConnectedOp";
        }
    };

    /**
     * @brief Class responsible for registering the CudaFullyConnectedOp operation.
     *
     * The CudaFullyConnectedOpRegistrar class registers the CudaFullyConnectedOp operation with the OperationRegistry.
     * It associates the operation name "Cuda::FullyConnectedOp" with a factory function that creates instances of CudaFullyConnectedOp.
     */
    export class CudaFullyConnectedOpRegistrar {
    public:
        /**
         * @brief Registers the CudaFullyConnectedOp operation with the OperationRegistry.
         *
         * This function registers the CudaFullyConnectedOp operation for the CUDA device type
         * with the OperationRegistry. It associates the operation name "Cuda::FullyConnectedOp"
         * with a factory function that creates instances of CudaFullyConnectedOp.
         */
        static void registerOperations() {
            const std::string opName = "Cuda::FullyConnectedOp";

            // Updated to use device context-aware registration
            OperationRegistry::instance().registerOperation<float, float, DeviceType::Cuda>(
                opName,
                "Default",  // Default empty variant for backward compatibility
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<float, float, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaFullyConnectedOp<float, float>>( context )
                        : std::make_shared<CudaFullyConnectedOp<float, float>>();
                }
            );

            // Add additional precision variants if needed, for example:
            /*
            OperationRegistry::instance().registerOperation<float, half, DeviceType::Cuda>(
                opName,
                "half_precision",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<float, half, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaFullyConnectedOp<float, half>>( context )
                        : std::make_shared<CudaFullyConnectedOp<float, half>>();
                }
            );
            */
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
