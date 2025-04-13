/**
 * @file CudaGeluOp.ixx
 * @brief Implementation of the CUDA-based GELU activation function for neural networks.
 */

module;
#include <vector>
#include <memory>
#include <iostream>
#include <cuda_fp16.h>
#include "Kernels/Cuda.Ops.h"

export module Compute.CudaGeluOp;

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

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief CUDA implementation of the GELU activation function for neural networks.
     *
     * This class provides a CUDA-based implementation of the Gaussian Error Linear Unit (GELU)
     * activation function, which is commonly used in transformer architectures. GELU is a smooth
     * approximation of the ReLU function that applies a non-linear transformation to its input.
     *
     * The implementation leverages CUDA for GPU acceleration, providing efficient computation
     * for large neural network models.
     *
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TPrecision The data type used for computation and output (defaults to the input type).
     */
    export
        template<typename TInput, typename TPrecision = TInput>
    class CudaGeluOp : public UnaryOperation<TInput, TPrecision, DeviceType::Cuda> {
    public:
        using MR = typename CudaDevice::MR;

        /**
         * @brief Constructs a new CUDA GELU operation with the default device context.
         *
         * Initializes the operation with a CUDA device context (defaults to CUDA:0).
         */
        CudaGeluOp() : UnaryOperation<TInput, TPrecision, DeviceType::Cuda>( OperationType::GeluOp ) {}

        /**
         * @brief Constructs a new CUDA GELU operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CUDA device.
         */
        CudaGeluOp( std::shared_ptr<DeviceContext> context )
            : UnaryOperation<TInput, TPrecision, DeviceType::Cuda>( OperationType::GeluOp, context ) {
            if ( !context->isDeviceType( DeviceType::Cuda ) ) {
                throw std::runtime_error( "CudaGeluOp requires a CUDA device context" );
            }
        }

        /**
         * @brief Performs the forward pass of the GELU activation function on CUDA.
         *
         * Computes the GELU transformation of the input elements:
         * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/?) * (x + 0.044715 * x^3)))
         *
         * @param input Input tensor containing the values to transform.
         * @param parameters Additional parameters (not used in this operation).
         * @param properties Additional attributes for the operation.
         * @param output Output tensor to store the transformed values.
         * @param output_cache Cache for intermediate results (not used in this operation).
         */
        void forward(
            const Tensor<TInput, MR>& input,
            const std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameters,
            const OperationAttributes& properties,
            Tensor<TPrecision, MR>& output,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_cache ) const override {

            // Verify we're operating on CUDA memory
            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cuda ) ) {
                throw std::runtime_error( "CudaGeluOp::forward can only be executed on CUDA memory" );
            }

            auto X = input.data();
            auto Y = output.data();
            int N = input.size();

            // Get CUDA stream from device context
            cudaStream_t stream = this->getDeviceContext()->getStream();

            // Call CUDA kernel with stream
            cuda_gelu_forward( Y, X, N, stream );
        }

        /**
         * @brief Performs the backward pass of the GELU activation function.
         *
         * Computes gradients with respect to inputs for the GELU function.
         *
         * @param input Input tensor from the forward pass.
         * @param output Output tensor from the forward pass.
         * @param output_gradient Gradient of the loss with respect to the output.
         * @param parameters Parameters tensor from forward pass (not used).
         * @param parameter_gradients Gradients for parameters (not used).
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
                throw std::runtime_error( "CudaGeluOp::backward can only be executed on CUDA memory" );
            }

            // Get tensor data pointers
            const TInput* X = input.data();
            const TPrecision* dY = output_gradient.data();
            TInput* dX = input_gradient.data();
            int N = input.size();

            // Get CUDA stream from device context
            cudaStream_t stream = this->getDeviceContext()->getStream();

            // Call CUDA kernel with stream
            cuda_gelu_backward( dX, X, dY, N, stream );
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cuda::GeluOp").
         */
        std::string getName() const override {
            return "Cuda::GeluOp";
        }
    };

    /**
     * @brief Class responsible for registering the CudaGeluOp operation.
     *
     * The CudaGeluOpRegistrar class registers the CudaGeluOp operation with the OperationRegistry.
     * It associates the operation name "Cuda::GeluOp" with a factory function that creates instances of CudaGeluOp.
     */
    export class CudaGeluOpRegistrar {
    public:
        /**
        * @brief Registers the CudaGeluOp operation with the OperationRegistry.
        *
        * This function registers the CudaGeluOp operation for the CUDA device type
        * with the OperationRegistry. It associates the operation name "Cuda::GeluOp"
        * with a factory function that creates instances of CudaGeluOp.
        */
        static void registerOperations() {
            const std::string opName = "Cuda::GeluOp";

            // Updated to use device context-aware registration
            OperationRegistry::instance().registerOperation<float, float, DeviceType::Cuda>(
                opName,
                "Default",  // Default empty variant for backward compatibility
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<float, float, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaGeluOp<float, float>>( context )
                        : std::make_shared<CudaGeluOp<float, float>>();
                }
            );

            /* FIXME: FP16 Precision is not supported yet
            OperationRegistry::instance().registerOperation<float, half, DeviceType::Cuda>(
                opName,
                "half_precision",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<float, half, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaGeluOp<float, half>>( context )
                        : std::make_shared<CudaGeluOp<float, half>>();
                }
            );*/
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
