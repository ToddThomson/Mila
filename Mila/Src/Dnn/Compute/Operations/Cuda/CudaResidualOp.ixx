/**
 * @file CudaResidualOp.ixx
 * @brief Implementation of the CUDA-based residual operation for neural networks.
 */

module;
#include <vector>
#include <iostream>
#include "Kernels/Cuda.Ops.h"

export module Compute.CudaResidualOp;

import Dnn.Tensor;
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaMemoryResource;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;
    /**
     * @brief CUDA implementation of the residual operation for neural networks.
     *
     * This class provides a CUDA-based implementation of the residual operation,
     * which performs element-wise addition of two input tensors.
     * It is commonly used in residual connections in neural network architectures
     * such as ResNet and Transformers to help with gradient flow and mitigate the
     * vanishing gradient problem. The implementation is optimized for NVIDIA GPUs
     * using CUDA for high-performance computation.
     *
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TDataType The data type of the output tensor elements (defaults to the input type).
     */
    export
        template<typename TInput, typename TPrecision = TInput>
    class CudaResidualOp : public BinaryOperation<TInput, TPrecision, DeviceType::Cuda> {
    public:
        using MR = typename CudaDevice::MR;

        /**
         * @brief Constructs a new CUDA Residual operation with the default device context.
         *
         * Initializes the operation with a CUDA device context (defaults to CUDA:0).
         */
        CudaResidualOp() : BinaryOperation<TInput, TPrecision, DeviceType::Cuda>( OperationType::ResidualOp ) {}

        /**
         * @brief Constructs a new CUDA Residual operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CUDA device.
         */
        CudaResidualOp( std::shared_ptr<DeviceContext> context )
            : BinaryOperation<TInput, TPrecision, DeviceType::Cuda>( OperationType::ResidualOp, context ) {
            // Ensure the device is CUDA-compatible
            if ( !context->isDeviceType( DeviceType::Cuda ) ) {
                throw std::runtime_error( "CudaResidualOp requires a CUDA device context" );
            }
        }

        /**
         * @brief Performs the forward pass of the residual operation on CUDA.
         *
         * Adds two input tensors element-wise and stores the result in the output tensor.
         * The computation is performed on the GPU using CUDA kernels for optimal performance.
         *
         * @param input1 The first input tensor to be added.
         * @param input2 The second input tensor to be added.
         * @param parameters Additional parameters (not used in this operation).
         * @param properties Additional attributes for the operation.
         * @param output The output tensor where the results will be stored.
         * @param output_cache Cache for intermediate results (not used in this operation).
         */
        void forward(
            const Tensor<TInput, MR>& input1,
            const Tensor<TInput, MR>& input2,
            const std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameters,
            const OperationAttributes& properties,
            Tensor<TPrecision, MR>& output,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_cache ) const override {

            // Verify we're operating on CUDA memory
            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cuda ) ) {
                throw std::runtime_error( "CudaResidualOp::forward can only be executed on CUDA memory" );
            }

            auto X1 = input1.data();
            auto X2 = input2.data();
            auto Y = output.data();
            int N = input1.size();

            cudaStream_t stream = this->getDeviceContext()->getStream();
            cuda_residual_forward( Y, X1, X2, N, stream );
        }

        /**
         * @brief Performs the backward pass of the residual operation.
         *
         * Computes gradients with respect to both inputs by propagating the output
         * gradient to each input.
         *
         * @param input1 First input tensor from the forward pass.
         * @param input2 Second input tensor from the forward pass.
         * @param output Output tensor from the forward pass.
         * @param output_gradient Gradient of the loss with respect to the output.
         * @param parameters Parameters tensor from forward pass (not used in this operation).
         * @param parameter_gradients Gradients for parameters (not used in this operation).
         * @param input1_gradient Gradient of the loss with respect to input1.
         * @param input2_gradient Gradient of the loss with respect to input2.
         * @param properties Additional attributes for the operation.
         * @param output_cache Cache tensors from forward pass (not used in this operation).
         */
        void backward(
            const Tensor<TInput, MR>& input1,
            const Tensor<TInput, MR>& input2,
            const Tensor<TPrecision, MR>& output,
            const Tensor<TPrecision, MR>& output_gradient,
            const std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameter_gradients,
            Tensor<TInput, MR>& input1_gradient,
            Tensor<TInput, MR>& input2_gradient,
            const OperationAttributes& properties,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_cache ) const {

            // Verify we're operating on CUDA memory
            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cuda ) ) {
                throw std::runtime_error( "CudaResidualOp::backward can only be executed on CUDA device" );
            }

            // Extract tensors
            const TPrecision* dY = output_gradient.data();
            TInput* dX1 = input1_gradient.data();
            TInput* dX2 = input2_gradient.data();
            int N = input1.size();

            cudaStream_t stream = this->getDeviceContext()->getStream();

            // For residual connection, the gradient just flows through to both inputs
            // FIXME: cuda_residual_backward( dX1, dX2, dY, N, stream );
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cuda::ResidualOp").
         */
        std::string getName() const override {
            return "Cuda::ResidualOp";
        }
    };

    /**
     * @brief Class responsible for registering the CudaResidualOp operation.
     *
     * The CudaResidualOpRegistrar class registers the CudaResidualOp operation with the OperationRegistry.
     * It associates the operation name "Cuda::ResidualOp" with a factory function that creates
     * instances of CudaResidualOp.
     */
    export class CudaResidualOpRegistrar {
    public:
        /**
         * @brief Registers the CudaResidualOp operation with the OperationRegistry.
         *
         * This function registers the CudaResidualOp operation for the CUDA device type
         * with the OperationRegistry. It associates the operation name "Cuda::ResidualOp"
         * with a factory function that creates instances of CudaResidualOp.
         */
        static void registerOperations() {
            const std::string opName = "Cuda::ResidualOp";

            // Updated to use device context-aware registration
            OperationRegistry::instance().registerOperation<float, float, DeviceType::Cuda>(
                opName,
                "Default",  // Default empty variant for backward compatibility
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<float, float, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaResidualOp<float, float>>( context )
                        : std::make_shared<CudaResidualOp<float, float>>();
                }
            );

            // Add additional precision variants if needed, for example:
            /*
            OperationRegistry::instance().registerOperation<float, half, DeviceType::Cuda>(
                opName,
                "half_precision",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<float, half, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaResidualOp<float, half>>( context )
                        : std::make_shared<CudaResidualOp<float, half>>();
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

